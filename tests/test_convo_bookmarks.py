"""Bookmark re-mine of a growing Claude Code log (local patch).

The contract under test: a re-mine that resumes from a bookmark leaves the palace
holding exactly what a full re-read of the same log would hold — same drawer ids,
same text, same labels — and any doubt sends the mine down the full path.
"""

import json
import os

import chromadb
import pytest

from mempalace import convo_miner
from mempalace.convo_miner import _bookmark_path, mine_convos

WING = "bm_test"
# Labels that legitimately differ between a bookmark save and a full read:
# the time of filing and of the write, and the per-file completion labels the
# bookmark replaces. ``authored_at`` is compared separately (see _assert_same).
VOLATILE = {
    "filed_at",
    "last_modified",
    "source_mtime",
    "chunk_total",
    "content_hash",
    "authored_at",
}


def _ts(n: int) -> str:
    return f"2026-10-03T{10 + n // 3600:02d}:{(n // 60) % 60:02d}:{n % 60:02d}.000Z"


class Log:
    """A synthetic Claude Code JSONL session that only ever grows."""

    def __init__(self, path):
        self.path = str(path)
        self.n = 0
        open(self.path, "w").close()

    def _write(self, entry):
        self.n += 1
        entry.setdefault("timestamp", _ts(self.n))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def turn(self, i: int, tool: bool = True):
        self._write(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": f"Question {i}: how should the cache layer handle key {i} "
                    + "under heavy load and eviction pressure? " * 3,
                },
            }
        )
        content = [{"type": "text", "text": f"Answer {i}. " + "Use a bounded LRU with TTL. " * 8}]
        if tool:
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{i:04d}",
                    "name": "Read" if i % 3 == 1 else "Bash",
                    "input": {"command": f"ls /srv/{i}"},
                }
            )
        self._write({"type": "assistant", "message": {"role": "assistant", "content": content}})
        if tool:
            self.tool_result(i)
            self._write(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"Done with {i}."}],
                    },
                }
            )

    def tool_result(self, i: int):
        self._write(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"toolu_{i:04d}",
                            "content": f"file-{i}.txt",
                        }
                    ],
                },
            }
        )

    def grow(self, start: int, count: int, tool: bool = True):
        for i in range(start, start + count):
            self.turn(i, tool=tool)


def _snapshot(palace, keep=()):
    col = chromadb.PersistentClient(path=palace).get_collection("mempalace_drawers")
    got = col.get(include=["documents", "metadatas"])
    out = {}
    for drawer_id, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        out[drawer_id] = (
            doc,
            {k: v for k, v in (meta or {}).items() if k not in VOLATILE or k in keep},
        )
    return out


def _full_mine(log_path, palace):
    os.environ["MEMPALACE_CONVO_BOOKMARKS"] = "0"
    try:
        mine_convos(log_path, palace, wing=WING)
    finally:
        os.environ.pop("MEMPALACE_CONVO_BOOKMARKS", None)


def _mine(log_path, palace):
    """Mine with bookmarks on, and replay the same mine into a shadow palace with
    bookmarks off (the unpatched full path), so the two histories stay in step."""
    mine_convos(log_path, palace, wing=WING)
    _full_mine(log_path, palace + "_replay")


def _without_authored_at(snap):
    return {
        k: (d, {a: b for a, b in m.items() if a != "authored_at"}) for k, (d, m) in snap.items()
    }


def _assert_same(palace, tmp_path, log_path):
    """1. Exactly what the unpatched full path holds after the same saves, labels
    included (``authored_at`` too: the full path keeps it on unchanged drawers).
    2. Same drawer ids, text and labels as a full read into a FRESH palace, except
    ``authored_at``, which a fresh palace stamps with the log's newest time."""
    got = _snapshot(palace, keep=("authored_at",))
    assert got == _snapshot(palace + "_replay", keep=("authored_at",))
    assert _without_authored_at(got) == _without_authored_at(
        _full_reference(tmp_path, log_path, keep=("authored_at",))
    )


def _full_reference(tmp_path, log_path, name="ref", keep=()):
    """What a full read of the log's current bytes files, in a fresh palace."""
    palace = str(tmp_path / f"palace_{name}")
    _full_mine(log_path, palace)
    return _snapshot(palace, keep=keep)


@pytest.fixture
def log(tmp_path):
    (tmp_path / "logs").mkdir()
    lg = Log(tmp_path / "logs" / "0a1b2c3d-session.jsonl")
    lg.grow(0, 12)
    return lg


@pytest.fixture
def palace(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_CONVO_BOOKMARKS", raising=False)
    return str(tmp_path / "palace")


def _bm(palace, log):
    return _bookmark_path(palace, log.path, "exchange", WING)


def test_first_full_mine_writes_a_bookmark(log, palace):
    _mine(log.path, palace)
    assert _bm(palace, log).exists(), "a full mine of a long log must leave a bookmark"


def test_grown_log_bookmark_save_matches_a_full_read(log, palace, tmp_path, capsys):
    _mine(log.path, palace)
    for round_ in range(3):
        log.grow(100 + round_ * 10, 4)
        capsys.readouterr()
        _mine(log.path, palace)
        out = capsys.readouterr().out
        assert "bookmark: read" in out, f"round {round_}: the save did not use the bookmark:\n{out}"
    _assert_same(palace, tmp_path, log.path)


def test_old_tool_result_appended_later_matches_a_full_read(log, palace, tmp_path, capsys):
    """Claude Code re-appends tool_result lines for calls made long before; the
    tool name (from the old tool_use) decides their formatting. Call 1 is a Read,
    whose result the full path omits; under an unknown name it would be kept."""
    _mine(log.path, palace)
    log.grow(200, 2)
    log.tool_result(1)
    log.grow(202, 2)
    capsys.readouterr()
    _mine(log.path, palace)
    assert "bookmark: read" in capsys.readouterr().out
    _assert_same(palace, tmp_path, log.path)


def test_unchanged_log_is_a_no_op(log, palace, capsys):
    _mine(log.path, palace)
    log.grow(300, 2)
    _mine(log.path, palace)
    before = _snapshot(palace)
    capsys.readouterr()
    _mine(log.path, palace)
    assert "unchanged (bookmark)" in capsys.readouterr().out
    assert _snapshot(palace) == before


def test_rewritten_prefix_declines_to_the_full_path(log, palace, tmp_path, capsys):
    _mine(log.path, palace)
    data = open(log.path, encoding="utf-8").read().replace("Question 0:", "Question Z:", 1)
    with open(log.path, "w", encoding="utf-8") as fh:
        fh.write(data)
    log.grow(400, 3)
    capsys.readouterr()
    _mine(log.path, palace)
    out = capsys.readouterr().out
    assert "bookmark not used" in out and "bookmark: read" not in out
    _assert_same(palace, tmp_path, log.path)


def test_truncated_log_declines_to_the_full_path(log, palace, tmp_path, capsys):
    _mine(log.path, palace)
    lines = open(log.path, encoding="utf-8").read().splitlines(keepends=True)
    with open(log.path, "w", encoding="utf-8") as fh:
        fh.writelines(lines[: len(lines) * 2 // 3])
    capsys.readouterr()
    _mine(log.path, palace)
    assert "the log shrank" in capsys.readouterr().out
    _assert_same(palace, tmp_path, log.path)


def test_missing_anchor_drawer_declines_to_the_full_path(log, palace, tmp_path, capsys):
    _mine(log.path, palace)
    bm = json.loads(_bm(palace, log).read_text())
    from mempalace.ids import make_convo_drawer_id

    anchor = make_convo_drawer_id(WING, bm["room"], log.path, "exchange", bm["base"] - 1)
    for p in (palace, palace + "_replay"):
        chromadb.PersistentClient(path=p).get_collection("mempalace_drawers").delete(ids=[anchor])
    log.grow(500, 2)
    capsys.readouterr()
    _mine(log.path, palace)
    assert "the palace does not hold exactly the bookmarked drawers" in capsys.readouterr().out
    _assert_same(palace, tmp_path, log.path)


def test_kill_switch_writes_no_bookmark(log, palace, monkeypatch):
    monkeypatch.setenv("MEMPALACE_CONVO_BOOKMARKS", "0")
    mine_convos(log.path, palace, wing=WING)
    assert not _bm(palace, log).exists()


def test_directory_sweep_skips_a_bookmarked_log_and_full_read_drops_it(log, palace, capsys):
    other = Log(os.path.join(os.path.dirname(log.path), "9f8e7d6c-other.jsonl"))
    other.grow(900, 6)
    _mine(other.path, palace)
    _mine(log.path, palace)
    log.grow(600, 2)
    _mine(log.path, palace)  # bookmark save: old drawers keep old mtimes
    before = _snapshot(palace)
    capsys.readouterr()
    _mine(os.path.dirname(log.path), palace)
    assert "Files skipped (already filed): 2" in capsys.readouterr().out
    assert _snapshot(palace) == before
    log.grow(700, 1)
    _mine(os.path.dirname(log.path), palace)  # sweep re-reads it in full
    assert not _bm(palace, log).exists(), "a full read must drop the log's stale bookmark"


def test_post_mine_steps_are_throttled_on_bookmark_saves(log, palace, monkeypatch):
    calls = []
    monkeypatch.setattr(
        convo_miner, "_validate_palace_fts5_after_mine", lambda p: calls.append("qc")
    )
    monkeypatch.setattr(
        convo_miner, "_compute_hallways_for_wing_safe", lambda *a, **k: calls.append("hall")
    )
    mine_convos(log.path, palace, wing=WING)  # full path: both run, both stamped
    assert calls == ["hall", "qc"]
    log.grow(800, 2)
    mine_convos(log.path, palace, wing=WING)
    assert calls == ["hall", "qc"], (
        "a bookmark save inside the throttle window re-ran post-mine steps"
    )
    monkeypatch.setenv("MEMPALACE_FASTPATH_POSTMINE_SECS", "0")
    log.grow(810, 2)
    mine_convos(log.path, palace, wing=WING)
    assert calls == ["hall", "qc", "hall", "qc"]


# ── Review findings (GPT-6.1 Sol, 2026-10-03): each test below failed before its repair ──


def _write_raw(log, data: bytes):
    with open(log.path, "ab") as fh:
        fh.write(data)


def test_g1_rewrite_far_from_any_window_declines(tmp_path, palace, capsys):
    (tmp_path / "logs").mkdir()
    big = Log(tmp_path / "logs" / "1b2c3d4e-big.jsonl")
    big.grow(0, 260)  # well past three 64 KiB windows
    _mine(big.path, palace)
    data = open(big.path, encoding="utf-8").read()
    assert len(data) > 4 * 64 * 1024
    with open(big.path, "w", encoding="utf-8") as fh:
        fh.write(data.replace("Question 120:", "Question X20:", 1))
    big.grow(1000, 2)
    capsys.readouterr()
    _mine(big.path, palace)
    assert "bookmark: read" not in capsys.readouterr().out
    _assert_same(palace, tmp_path, big.path)


def test_g2_bare_cr_records_decline(log, palace, tmp_path, capsys):
    _mine(log.path, palace)
    user = {
        "type": "user",
        "timestamp": _ts(900),
        "message": {
            "role": "user",
            "content": "Question CR: what about a carriage return separated record here?",
        },
    }
    asst = {
        "type": "assistant",
        "timestamp": _ts(901),
        "message": {"role": "assistant", "content": [{"type": "text", "text": "Answer CR. " * 20}]},
    }
    _write_raw(log, (json.dumps(user) + "\r" + json.dumps(asst) + "\n").encode())
    capsys.readouterr()
    _mine(log.path, palace)
    assert "bookmark: read" not in capsys.readouterr().out
    _assert_same(palace, tmp_path, log.path)


def test_g2_bom_log_gets_no_bookmark(tmp_path, palace):
    (tmp_path / "logs").mkdir()
    lg = Log(tmp_path / "logs" / "2c3d4e5f-bom.jsonl")
    lg.grow(0, 12)
    body = open(lg.path, "rb").read()
    first = json.loads(body.split(b"\n", 1)[0])
    first["timestamp"] = "2099-01-01T00:00:00.000Z"
    with open(lg.path, "wb") as fh:
        fh.write(b"\xef\xbb\xbf" + json.dumps(first).encode() + b"\n" + body.split(b"\n", 1)[1])
    _mine(lg.path, palace)
    assert not _bm(palace, lg).exists()
    lg.grow(100, 2)
    _mine(lg.path, palace)
    _assert_same(palace, tmp_path, lg.path)


def test_g2_invalid_utf8_in_a_timestamp_declines(log, palace, tmp_path, capsys):
    _mine(log.path, palace)
    rec = json.dumps(
        {
            "type": "user",
            "timestamp": "2099-12-31T23:59:59.000ZQQ",
            "message": {
                "role": "user",
                "content": "Question bad bytes: is this timestamp read the same way by both paths?",
            },
        }
    )
    _write_raw(log, rec.replace("QQ", "\\u00e9").encode().replace(b"\\u00e9", b"\xff") + b"\n")
    log.grow(1100, 1)
    capsys.readouterr()
    _mine(log.path, palace)
    assert "bookmark: read" not in capsys.readouterr().out
    _assert_same(palace, tmp_path, log.path)


def _drop_drawer(palace, log, index):
    bm = json.loads(_bm(palace, log).read_text())
    from mempalace.ids import make_convo_drawer_id

    drawer = make_convo_drawer_id(WING, bm["room"], log.path, "exchange", index)
    for p in (palace, palace + "_replay"):
        chromadb.PersistentClient(path=p).get_collection("mempalace_drawers").delete(ids=[drawer])


def test_g3_missing_prefix_drawer_declines_on_a_grown_save(log, palace, tmp_path, capsys):
    _mine(log.path, palace)
    _drop_drawer(palace, log, 1)  # not the anchor
    log.grow(1200, 2)
    capsys.readouterr()
    _mine(log.path, palace)
    assert "bookmark: read" not in capsys.readouterr().out
    _assert_same(palace, tmp_path, log.path)


def test_g3_missing_anchor_is_not_hidden_by_an_unchanged_save(log, palace, tmp_path, capsys):
    _mine(log.path, palace)
    log.grow(1300, 2)
    _mine(log.path, palace)
    bm = json.loads(_bm(palace, log).read_text())
    _drop_drawer(palace, log, bm["base"] - 1)
    capsys.readouterr()
    _mine(log.path, palace)
    assert "unchanged (bookmark)" not in capsys.readouterr().out
    _assert_same(palace, tmp_path, log.path)


def test_g3_interrupted_save_leaves_no_orphans(log, palace, tmp_path, monkeypatch):
    _mine(log.path, palace)
    long_asst = {
        "type": "assistant",
        "timestamp": _ts(1400),
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Long unterminated answer. " * 400}],
        },
    }
    _write_raw(log, json.dumps(long_asst).encode())  # no newline: a record still being written
    real_save = convo_miner._save_bookmark

    def crash(path, bm):
        raise KeyboardInterrupt("simulated crash after the drawer writes")

    monkeypatch.setattr(convo_miner, "_save_bookmark", crash)
    with pytest.raises(KeyboardInterrupt):
        mine_convos(log.path, palace, wing=WING)
    # The crash came after every drawer write, so the replay saves at this moment too.
    _full_mine(log.path, palace + "_replay")
    monkeypatch.setattr(convo_miner, "_save_bookmark", real_save)
    _write_raw(
        log,
        json.dumps(
            {
                "type": "user",
                "timestamp": _ts(1401),
                "message": {"role": "user", "content": "glued"},
            }
        ).encode()
        + b"\n",
    )
    log.grow(1402, 2)
    _mine(log.path, palace)
    _assert_same(palace, tmp_path, log.path)


def test_g4_stale_conversation_hash_never_hides_another_file(log, palace, tmp_path):
    early = open(log.path, "rb").read()
    _mine(log.path, palace)
    log.grow(1500, 2)
    _mine(log.path, palace)  # bookmark save: the log's conversation hash is now stale
    copy = os.path.join(os.path.dirname(log.path), "3d4e5f6a-copy.jsonl")
    with open(copy, "wb") as fh:
        fh.write(early)  # a different file holding the log's EARLY content
    _mine(copy, palace)
    col = chromadb.PersistentClient(path=palace).get_collection("mempalace_drawers")
    got = col.get(where={"source_file": copy}, include=["metadatas"])
    convos = [m for m in got["metadatas"] if (m or {}).get("ingest_mode") == "convos"]
    assert convos, (
        "the copy was skipped as a duplicate of a stale hash (only a registry marker was filed)"
    )


def test_g5_spellcheck_on_means_no_bookmark(log, palace, monkeypatch):
    import mempalace.spellcheck as sc

    class FakeSpeller:
        def __call__(self, word):
            return word

    monkeypatch.setattr(sc, "_get_speller", lambda: FakeSpeller())
    mine_convos(log.path, palace, wing=WING)
    assert not _bm(palace, log).exists()


def test_g6_structurally_invalid_bookmark_falls_back(log, palace, tmp_path, capsys):
    _mine(log.path, palace)
    log.grow(1600, 2)
    _mine(log.path, palace)  # bookmark save: old drawers keep old mtimes, so no mtime skip
    _bm(palace, log).write_text(json.dumps({"v": convo_miner._BOOKMARK_VERSION}))
    _mine(log.path, palace)  # must fall back, not crash
    _bm(palace, log).write_text(json.dumps({"v": convo_miner._BOOKMARK_VERSION}))
    log.grow(1610, 1)
    _mine(os.path.dirname(log.path), palace)  # nor may a directory mine
    _assert_same(palace, tmp_path, log.path)


# ── Repair verification (second round): each test below failed before its correction ──


def test_g2_unfinished_last_line_with_bad_bytes_declines(log, palace, tmp_path, capsys):
    """Both readers strip a line before parsing it; a leading vertical tab hides it from json."""
    _mine(log.path, palace)
    log.grow(1700, 1)
    rec = json.dumps(
        {
            "type": "user",
            "timestamp": "2099-01-01T00:00:00ZQQ",
            "message": {"role": "user", "content": "unfinished"},
        }
    )
    _write_raw(
        log, b"\x0b" + rec.replace("QQ", "").encode().replace(b'Z"', b'Z\xff"', 1)
    )  # no newline
    capsys.readouterr()
    _mine(log.path, palace)
    assert "bookmark: read" not in capsys.readouterr().out
    _assert_same(palace, tmp_path, log.path)


def test_g6_bookmark_missing_prefix_ts_falls_back(log, palace, tmp_path, capsys):
    _mine(log.path, palace)
    bm = json.loads(_bm(palace, log).read_text())
    del bm["prefix_ts"]
    _bm(palace, log).write_text(json.dumps(bm))
    log.grow(1800, 2)
    capsys.readouterr()
    _mine(log.path, palace)
    assert "bookmark: read" not in capsys.readouterr().out
    _assert_same(palace, tmp_path, log.path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id_recipe", True),
        ("chunk_size", -1),
        ("chunk_size", 0),
        ("min_chunk_size", -1),
        ("normalize_version", 0),
        ("size", 0),
        ("mtime", float("nan")),
        ("prefix_sha", "z" * 64),
        ("prev_hash", "short"),
        ("room", ""),
        ("tool_map", {"1": 2, 3: "x"}),
        ("total", 10**12),
        ("size", 600 * 1024 * 1024),
    ],
)
def test_g6_out_of_range_fields_are_rejected(log, palace, field, value):
    _mine(log.path, palace)
    bm = json.loads(_bm(palace, log).read_text())
    assert convo_miner._bookmark_valid(bm)
    bm[field] = value
    assert not convo_miner._bookmark_valid(bm), f"{field}={value!r} was accepted"


def test_g6_a_bookmark_the_validator_cannot_even_judge_is_no_bookmark(
    log, palace, tmp_path, capsys
):
    """Second verifier: mtime = 10**400 made math.isfinite raise OverflowError at load."""
    _mine(log.path, palace)
    bm = json.loads(_bm(palace, log).read_text())
    bm["mtime"] = 10**400
    _bm(palace, log).write_text(json.dumps(bm))
    log.grow(1900, 2)
    capsys.readouterr()
    _mine(log.path, palace)  # must fall back to the full read, not raise
    assert "bookmark: read" not in capsys.readouterr().out
    _assert_same(palace, tmp_path, log.path)
