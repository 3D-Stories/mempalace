"""Saves of logs past the 500 MB full-read cap (local patch, step 6).

Measured on the live hub, 2026-10-11: a save of a Claude Code log over the cap had
nothing to mine (the scan drops the file), yet it still read the labels of every
drawer in the palace and ran the unthrottled quick_check, holding the palace write
lock for 29-44 minutes while every other write waited.

The contract under test:
(a) a save with no files to mine does no palace-wide work at all;
(b) a log past the cap that has a bookmark keeps resuming from it, and files exactly
    what a full read of the same bytes would file;
(c) a log past the cap is never read whole, and never loses its bookmark.
"""

import json
import os

import pytest

from mempalace import convo_miner
from mempalace.convo_miner import _bookmark_path, _bookmark_valid, mine_convos

from test_convo_bookmarks import WING, Log, _full_mine, _snapshot


@pytest.fixture
def palace(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_CONVO_BOOKMARKS", raising=False)
    return str(tmp_path / "palace")


@pytest.fixture
def log(tmp_path):
    (tmp_path / "logs").mkdir()
    lg = Log(tmp_path / "logs" / "0a1b2c3d-session.jsonl")
    lg.grow(0, 12)
    return lg


@pytest.fixture
def watch(monkeypatch):
    """Record every palace-wide step a save can take, and every whole-file read."""
    calls = []
    real_mined, real_hashes = convo_miner.prefetch_mined_set, convo_miner.prefetch_content_hashes
    real_norm = convo_miner.normalize_conversations

    def spy(name, real):
        def wrapper(*a, **k):
            calls.append(name)
            return real(*a, **k)

        return wrapper

    monkeypatch.setattr(convo_miner, "prefetch_mined_set", spy("prefetch_mined", real_mined))
    monkeypatch.setattr(convo_miner, "prefetch_content_hashes", spy("prefetch_hashes", real_hashes))
    monkeypatch.setattr(convo_miner, "normalize_conversations", spy("full_read", real_norm))
    monkeypatch.setattr(
        convo_miner, "_validate_palace_fts5_after_mine", lambda p: calls.append("quick_check")
    )
    monkeypatch.setattr(
        convo_miner, "_compute_hallways_for_wing_safe", lambda *a, **k: calls.append("hallways")
    )
    return calls


def _cap_at(monkeypatch, size):
    """Put both 500 MB caps (the scan's and the bookmark writer's) at ``size`` bytes."""
    monkeypatch.setattr(convo_miner, "MAX_FILE_SIZE", size)
    monkeypatch.setattr(convo_miner, "_BOOKMARK_MAX_FILE", size)


def _reference(tmp_path, log_path, name):
    """A full read of the log's current bytes into a fresh palace, with no cap."""
    mp = pytest.MonkeyPatch()
    try:
        _cap_at(mp, 500 * 1024 * 1024)
        target = str(tmp_path / f"palace_{name}")
        _full_mine(log_path, target)
        return _snapshot(target)
    finally:
        mp.undo()


# ── (a) a save with no files does no palace-wide work ──


def test_empty_directory_save_does_no_palace_wide_work(log, palace, tmp_path, watch):
    mine_convos(log.path, palace, wing=WING)
    before = _snapshot(palace)
    watch.clear()
    empty = tmp_path / "empty"
    empty.mkdir()
    mine_convos(str(empty), palace, wing=WING)
    assert watch == [], f"a save with no files still did palace-wide work: {watch}"
    assert _snapshot(palace) == before


def test_oversize_log_without_bookmark_does_no_palace_wide_work(
    log, palace, tmp_path, watch, monkeypatch, capsys
):
    other = Log(tmp_path / "logs" / "9f8e7d6c-other.jsonl")
    other.grow(0, 12)
    mine_convos(other.path, palace, wing=WING)  # the palace is not empty
    before = _snapshot(palace)
    watch.clear()
    _cap_at(monkeypatch, os.path.getsize(log.path) - 1)  # `log` has no bookmark
    capsys.readouterr()
    mine_convos(log.path, palace, wing=WING)
    assert watch == [], f"an oversize log with no bookmark still did palace-wide work: {watch}"
    assert _snapshot(palace) == before
    assert "no usable bookmark" in capsys.readouterr().err


# ── (b) a bookmarked log past the cap keeps resuming ──


def test_bookmarked_log_past_the_cap_resumes_and_matches_a_full_read(
    log, palace, tmp_path, monkeypatch, capsys
):
    mine_convos(log.path, palace, wing=WING)  # full read, writes the bookmark
    assert _bookmark_path(palace, log.path, "exchange", WING).exists()
    _cap_at(monkeypatch, os.path.getsize(log.path))
    for round_ in range(2):  # round 2 resumes from a bookmark written past the cap
        log.grow(100 + round_ * 10, 4)
        assert os.path.getsize(log.path) > convo_miner.MAX_FILE_SIZE
        capsys.readouterr()
        mine_convos(log.path, palace, wing=WING)
        out = capsys.readouterr().out
        assert "bookmark: read" in out, f"round {round_}: the save did not resume:\n{out}"
    assert _snapshot(palace) == _reference(tmp_path, log.path, "ref")


def test_bookmarked_log_past_the_cap_does_no_palace_wide_reads(log, palace, watch, monkeypatch):
    mine_convos(log.path, palace, wing=WING)
    _cap_at(monkeypatch, os.path.getsize(log.path))
    log.grow(100, 4)
    watch.clear()
    mine_convos(log.path, palace, wing=WING)
    assert "prefetch_mined" not in watch and "prefetch_hashes" not in watch
    assert "full_read" not in watch


def test_bookmark_recorded_past_the_cap_is_valid(log, palace):
    mine_convos(log.path, palace, wing=WING)
    with open(_bookmark_path(palace, log.path, "exchange", WING), encoding="utf-8") as fh:
        bm = json.load(fh)
    assert _bookmark_valid(bm)
    bm["size"] = 700 * 1024 * 1024
    bm["resume_offset"] = 600 * 1024 * 1024
    assert _bookmark_valid(bm), "a bookmark recorded past 500 MB would never load again"


def test_new_part_over_the_tail_cap_declines(log, palace, monkeypatch, capsys):
    mine_convos(log.path, palace, wing=WING)
    monkeypatch.setattr(convo_miner, "_BOOKMARK_MAX_TAIL", 10)
    log.grow(100, 4)
    capsys.readouterr()
    mine_convos(log.path, palace, wing=WING)
    out = capsys.readouterr().out
    assert "the new part passed" in out
    assert "bookmark: read" not in out


# ── (c) a log past the cap is never read whole and keeps its bookmark ──


def test_oversize_log_whose_bookmark_declines_is_not_read_and_keeps_it(
    log, palace, tmp_path, watch, monkeypatch, capsys
):
    mine_convos(log.path, palace, wing=WING)
    bm_path = _bookmark_path(palace, log.path, "exchange", WING)
    before = _snapshot(palace)
    _cap_at(monkeypatch, os.path.getsize(log.path))
    monkeypatch.setattr(convo_miner, "_BOOKMARK_MAX_TAIL", 10)  # the quick path declines
    log.grow(100, 4)
    watch.clear()
    capsys.readouterr()
    mine_convos(log.path, palace, wing=WING)
    assert watch == [], f"a declined oversize save did palace-wide work: {watch}"
    assert _snapshot(palace) == before
    assert bm_path.exists(), "a declined oversize save dropped the log's only bookmark"
    assert "no usable bookmark" in capsys.readouterr().err


def test_dry_run_of_an_oversize_log_does_nothing(log, palace, tmp_path, watch, monkeypatch):
    mine_convos(log.path, palace, wing=WING)
    before = _snapshot(palace)
    _cap_at(monkeypatch, os.path.getsize(log.path))
    log.grow(100, 4)
    watch.clear()
    mine_convos(log.path, palace, wing=WING, dry_run=True)
    assert watch == []
    assert _snapshot(palace) == before


# ── Review F1 (GPT-6.1 Sol, 2026-10-11): opening the palace is palace-wide work too ──


@pytest.fixture
def no_open(monkeypatch):
    """Fail the test if a save opens the palace collection at all."""

    def refuse(*a, **k):
        raise AssertionError("the save opened the palace collection")

    monkeypatch.setattr(convo_miner, "_open_convo_collection", refuse)


def test_empty_directory_save_does_not_open_the_palace(palace, tmp_path, no_open):
    empty = tmp_path / "empty"
    empty.mkdir()
    mine_convos(str(empty), palace, wing=WING)


def test_oversize_log_without_bookmark_does_not_open_the_palace(log, palace, monkeypatch, no_open):
    _cap_at(monkeypatch, os.path.getsize(log.path) - 1)
    mine_convos(log.path, palace, wing=WING)


def test_oversize_dry_run_does_not_open_the_palace(log, palace, monkeypatch):
    mine_convos(log.path, palace, wing=WING)
    _cap_at(monkeypatch, os.path.getsize(log.path))
    log.grow(100, 4)
    monkeypatch.setattr(
        convo_miner,
        "_open_convo_collection",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened the palace")),
    )
    mine_convos(log.path, palace, wing=WING, dry_run=True)


def test_directory_holding_only_an_oversize_log_does_no_work(
    log, palace, tmp_path, watch, monkeypatch, capsys
):
    mine_convos(log.path, palace, wing=WING)
    before = _snapshot(palace)
    bm_path = _bookmark_path(palace, log.path, "exchange", WING)
    _cap_at(monkeypatch, os.path.getsize(log.path))
    log.grow(100, 4)
    monkeypatch.setattr(
        convo_miner,
        "_open_convo_collection",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened the palace")),
    )
    watch.clear()
    mine_convos(os.path.dirname(log.path), palace, wing=WING)
    assert watch == []
    assert _snapshot(palace) == before
    assert bm_path.exists()


def test_oversize_log_with_bookmarks_switched_off_is_skipped_and_keeps_its_bookmark(
    log, palace, watch, monkeypatch, capsys
):
    mine_convos(log.path, palace, wing=WING)
    before = _snapshot(palace)
    bm_path = _bookmark_path(palace, log.path, "exchange", WING)
    _cap_at(monkeypatch, os.path.getsize(log.path))
    log.grow(100, 4)
    monkeypatch.setenv("MEMPALACE_CONVO_BOOKMARKS", "0")
    monkeypatch.setattr(
        convo_miner,
        "_open_convo_collection",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened the palace")),
    )
    watch.clear()
    capsys.readouterr()
    mine_convos(log.path, palace, wing=WING)
    assert watch == []
    assert _snapshot(palace) == before
    assert bm_path.exists()
    assert "no usable bookmark" in capsys.readouterr().err
