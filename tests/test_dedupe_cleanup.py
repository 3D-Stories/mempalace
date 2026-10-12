"""One-time duplicates cleanup (3D-Stories fork, sysop#297).

A palace saved before the save-time skip holds boilerplate and same-chat
repeats. The cleanup removes exactly the rows a save with the skip would not
have stored, keeps every chat's first copy and every copy another chat shares,
and fixes labels and bookmarks so nothing downstream notices.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import chromadb
import pytest

from mempalace import dedupe_cleanup as dc
from mempalace.convo_miner import mine_convos
from mempalace.palace import MineAlreadyRunning, prefetch_mined_set
from test_convo_bookmarks import WING, Log, _bm
from test_convo_save_dedupe import CAVEAT, STOP, STOP_ANSWER, say


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMPALACE_SAVE_DEDUPE", raising=False)
    monkeypatch.delenv("MEMPALACE_CONVO_BOOKMARKS", raising=False)
    monkeypatch.setenv("MEMPALACE_CONFIG_DIR", str(tmp_path / "cfg"))


def _chat(path):
    """Pieces 0-2 unique, 3 boilerplate, 4 a stop-hook message, 5-6 unique,
    7 the same stop-hook message again, 8-13 unique."""
    lg = Log(path)
    lg.grow(0, 3)
    say(lg, CAVEAT, "ok")
    say(lg, STOP, STOP_ANSWER)
    lg.grow(3, 2)
    say(lg, STOP, STOP_ANSWER)
    lg.grow(5, 6)
    return lg


@pytest.fixture
def logs(tmp_path):
    (tmp_path / "logs").mkdir()
    return tmp_path / "logs"


def _legacy_mine(path, palace, monkeypatch):
    """A save as the hub made it before the skip existed: every piece stored."""
    monkeypatch.setenv("MEMPALACE_SAVE_DEDUPE", "0")
    try:
        mine_convos(str(path), palace, wing=WING)
    finally:
        monkeypatch.delenv("MEMPALACE_SAVE_DEDUPE")


def _col(palace):
    return chromadb.PersistentClient(path=palace).get_collection("mempalace_drawers")


def _rows(palace):
    got = _col(palace).get(include=["documents", "metadatas"])
    return {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}


def _by_chat(palace):
    out: dict = {}
    for _id, (doc, meta) in _rows(palace).items():
        if meta.get("ingest_mode") == "convos":
            out.setdefault(meta["source_file"], {})[meta["chunk_index"]] = doc
    return out


@pytest.fixture
def legacy(logs, tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    lg = _chat(logs / "0a1b2c3d-session.jsonl")
    _legacy_mine(lg.path, palace, monkeypatch)
    return palace, lg


def test_a_dry_run_writes_nothing(legacy):
    palace, lg = legacy
    before = _rows(palace)
    bm_before = _bm(palace, lg).read_bytes() if _bm(palace, lg).exists() else None

    plan = dc.plan_cleanup(palace)

    assert plan.counts["boilerplate"] == 1
    assert plan.counts["repeat"] == 1
    assert _rows(palace) == before
    assert (_bm(palace, lg).read_bytes() if _bm(palace, lg).exists() else None) == bm_before


def test_apply_leaves_what_a_save_with_the_skip_would_store(legacy, tmp_path):
    palace, lg = legacy
    reference = str(tmp_path / "reference")
    mine_convos(lg.path, reference, wing=WING)

    dc.apply_cleanup(palace, dc.plan_cleanup(palace))

    assert _by_chat(palace) == _by_chat(reference)
    assert set(_by_chat(palace)[lg.path]) == set(range(14)) - {3, 7}


def test_no_unique_text_is_lost(legacy):
    palace, lg = legacy
    before = set(_by_chat(palace)[lg.path].values())

    dc.apply_cleanup(palace, dc.plan_cleanup(palace))

    after = set(_by_chat(palace)[lg.path].values())
    lost = before - after
    assert all(dc.is_boilerplate(text) for text in lost)
    assert len(lost) == 1


def test_cross_chat_copies_survive(logs, tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    a = _chat(logs / "aaaa0000-session.jsonl")
    b = _chat(logs / "bbbb0000-session.jsonl")
    b.grow(50, 1)  # piece 14: otherwise b is the same conversation and is not stored again
    _legacy_mine(a.path, palace, monkeypatch)
    _legacy_mine(b.path, palace, monkeypatch)

    dc.apply_cleanup(palace, dc.plan_cleanup(palace))

    chats = _by_chat(palace)
    assert set(chats[a.path]) == set(range(14)) - {3, 7}
    assert set(chats[b.path]) == set(range(15)) - {3, 7}
    assert set(chats[a.path].values()) <= set(chats[b.path].values())


def test_already_saved_answers_do_not_change(legacy):
    palace, lg = legacy
    before = prefetch_mined_set(_col(palace))

    dc.apply_cleanup(palace, dc.plan_cleanup(palace))

    assert prefetch_mined_set(_col(palace)) == before
    assert lg.path in before


def test_a_bookmarked_chat_still_uses_its_bookmark(legacy, tmp_path, capsys):
    palace, lg = legacy
    assert _bm(palace, lg).exists()

    dc.apply_cleanup(palace, dc.plan_cleanup(palace))

    bm = json.loads(_bm(palace, lg).read_text())
    assert bm["skipped"] == [3, 7]
    lg.grow(100, 2)
    say(lg, STOP, STOP_ANSWER)
    lg.grow(102, 2)
    capsys.readouterr()
    mine_convos(lg.path, palace, wing=WING)
    assert "bookmark: read" in capsys.readouterr().out

    reference = str(tmp_path / "reference")
    mine_convos(lg.path, reference, wing=WING)
    assert _by_chat(palace) == _by_chat(reference)


def test_a_removed_anchor_moves_to_the_last_stored_piece(logs, tmp_path, monkeypatch, capsys):
    palace = str(tmp_path / "palace")
    lg = _chat(logs / "0a1b2c3d-session.jsonl")
    say(lg, STOP, STOP_ANSWER)  # piece 14 repeats piece 4
    lg.grow(20, 1)  # piece 15, the bookmark base
    _legacy_mine(lg.path, palace, monkeypatch)
    bm = json.loads(_bm(palace, lg).read_text())
    assert bm["base"] == 15

    dc.apply_cleanup(palace, dc.plan_cleanup(palace))

    bm = json.loads(_bm(palace, lg).read_text())
    assert 14 in bm["skipped"]
    assert bm["prev_hash"] == _sha(_by_chat(palace)[lg.path][13])
    lg.grow(200, 1)
    capsys.readouterr()
    mine_convos(lg.path, palace, wing=WING)
    assert "bookmark: read" in capsys.readouterr().out


def test_the_conversation_hash_moves_to_the_first_remaining_piece(logs, tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    lg = Log(logs / "0a1b2c3d-session.jsonl")
    say(lg, CAVEAT, "ok")  # piece 0 is boilerplate
    lg.grow(0, 5)
    _legacy_mine(lg.path, palace, monkeypatch)
    hashed = [m for _d, m in _rows(palace).values() if m.get("content_hash")]
    assert [m["chunk_index"] for m in hashed] == [0]

    dc.apply_cleanup(palace, dc.plan_cleanup(palace))

    hashed_after = [m for _d, m in _rows(palace).values() if m.get("content_hash")]
    assert [m["chunk_index"] for m in hashed_after] == [1]
    assert hashed_after[0]["content_hash"] == hashed[0]["content_hash"]


def test_a_drawer_a_closet_names_is_kept(legacy):
    palace, lg = legacy
    plan = dc.plan_cleanup(palace)
    repeat_id = next(i for i, why in plan.remove.items() if why == "repeat")
    closets = chromadb.PersistentClient(path=palace).get_or_create_collection("mempalace_closets")
    closets.add(ids=["c1"], documents=[f"topic|ent|→{repeat_id}"], metadatas=[{"wing": WING}])

    plan = dc.plan_cleanup(palace)
    result = dc.apply_cleanup(palace, plan)

    assert repeat_id in _rows(palace)
    assert result["referenced_kept"] == 1


def test_a_drawer_a_triple_names_is_kept(legacy, monkeypatch):
    palace, lg = legacy
    plan = dc.plan_cleanup(palace)
    repeat_id = next(i for i, why in plan.remove.items() if why == "repeat")
    import sqlite3

    kg = sqlite3.connect(os.path.join(palace, "knowledge_graph.sqlite3"))
    kg.execute("create table if not exists triples (id text, source_drawer_id text)")
    kg.execute("insert into triples values ('t1', ?)", (repeat_id,))
    kg.commit()
    kg.close()

    result = dc.apply_cleanup(palace, dc.plan_cleanup(palace))

    assert repeat_id in _rows(palace)
    assert result["referenced_kept"] == 1


def test_a_row_whose_text_changed_since_the_plan_is_kept(legacy):
    palace, lg = legacy
    plan = dc.plan_cleanup(palace)
    repeat_id = next(i for i, why in plan.remove.items() if why == "repeat")
    _col(palace).update(ids=[repeat_id], documents=["now a different text"])

    result = dc.apply_cleanup(palace, plan)

    assert repeat_id in _rows(palace)
    assert result["mismatch_kept"] == 1


def test_apply_refuses_while_a_save_holds_the_palace(legacy):
    palace, _ = legacy
    plan = dc.plan_cleanup(palace)
    before = _rows(palace)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from mempalace.palace import mine_palace_lock\n"
            "with mine_palace_lock(sys.argv[1]):\n"
            "    print('held', flush=True)\n"
            "    sys.stdin.readline()\n",
            palace,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(MineAlreadyRunning):
            dc.apply_cleanup(palace, plan)
    finally:
        holder.communicate("go\n", timeout=30)
    assert _rows(palace) == before


def test_candidates_list_texts_held_by_many_chats(logs, tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    shared = "Shared standing instruction that every session repeats verbatim."
    for n in range(3):
        lg = Log(logs / f"{n:04d}0000-session.jsonl")
        say(lg, shared, "ack")
        lg.grow(n * 10, 2)
        _legacy_mine(lg.path, palace, monkeypatch)
    out = tmp_path / "review.md"

    n = dc.write_candidates(palace, out, min_chats=3)

    text = out.read_text()
    assert n == 1
    line = next(ln for ln in text.splitlines() if ln.startswith("- [ ] "))
    sha = line[6:70]
    assert "| 3 chats |" in line
    assert sha == _sha(next(d for d, _m in _rows(palace).values() if d.startswith("> " + shared)))


def test_only_ticked_lines_reach_the_list(tmp_path):
    good, other = "a" * 64, "b" * 64
    review = tmp_path / "review.md"
    review.write_text(
        f"# review\n- [x] {good} | 30 chats | 40 rows | text\n- [ ] {other} | 26 chats | 30 rows | t\n"
    )

    n = dc.write_list_from_review(review)

    data = json.loads((tmp_path / "cfg" / "boilerplate.json").read_text())
    assert n == 1
    assert data == {"version": 1, "sha256": [good]}


def test_a_malformed_ticked_line_writes_nothing(tmp_path):
    review = tmp_path / "review.md"
    review.write_text("- [x] not-a-hash | 30 chats | 40 rows | text\n")

    with pytest.raises(ValueError):
        dc.write_list_from_review(review)

    assert not (tmp_path / "cfg" / "boilerplate.json").exists()


def test_an_existing_list_is_backed_up_first(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "boilerplate.json").write_text(json.dumps({"version": 1, "sha256": ["c" * 64]}))
    review = tmp_path / "review.md"
    review.write_text(f"- [x] {'d' * 64} | 30 chats | 40 rows | text\n")

    dc.write_list_from_review(review)

    backups = list(cfg.glob("boilerplate.json.bak-*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text())["sha256"] == ["c" * 64]


def _cli(*argv):
    from unittest.mock import patch

    from mempalace.cli import main

    with patch("sys.argv", ["mempalace", *argv]):
        try:
            main()
        except SystemExit as exc:
            return exc.code
    return 0


def test_the_command_is_a_dry_run_by_default(legacy, capsys):
    palace, _ = legacy
    before = _rows(palace)

    code = _cli("--palace", palace, "dedupe-cleanup")

    out = capsys.readouterr().out
    assert code in (0, None)
    assert "dry run" in out
    assert "boilerplate 1" in out and "repeat 1" in out
    assert _rows(palace) == before


def test_apply_needs_yes(legacy, capsys):
    palace, _ = legacy
    before = _rows(palace)

    code = _cli("--palace", palace, "dedupe-cleanup", "--apply")

    assert code == 2
    assert _rows(palace) == before


def test_apply_with_yes_removes_the_planned_rows(legacy, capsys):
    palace, lg = legacy

    code = _cli("--palace", palace, "dedupe-cleanup", "--apply", "--yes")

    assert code in (0, None)
    assert set(_by_chat(palace)[lg.path]) == set(range(14)) - {3, 7}
    assert "removed 2" in capsys.readouterr().out


def test_the_command_writes_candidates_and_the_list(legacy, tmp_path, capsys):
    palace, _ = legacy
    review = tmp_path / "review.md"

    assert _cli(
        "--palace", palace, "dedupe-cleanup", "--candidates", str(review), "--min-chats", "1"
    ) in (0, None)
    lines = [ln for ln in review.read_text().splitlines() if ln.startswith("- [ ] ")]
    assert lines
    review.write_text(lines[0].replace("- [ ] ", "- [x] ", 1) + "\n")

    assert _cli("dedupe-cleanup", "--write-list", str(review)) in (0, None)
    data = json.loads((tmp_path / "cfg" / "boilerplate.json").read_text())
    assert data["sha256"] == [lines[0][6:70]]


def test_a_file_of_several_conversations_is_left_alone(tmp_path, monkeypatch):
    from test_convo_miner import _privacy_export_bundle

    shared = [
        "Shared question, long enough to stand as a piece of its own in either chat?",
        "Shared answer, also long enough to stand as a piece of its own in either chat.",
    ]
    a = shared + ["Chat A asks its own question here?", "Chat A gets its own answer here."]
    b = shared + ["Chat B asks its own question here?", "Chat B gets its own answer here."]
    (tmp_path / "export").mkdir()
    export = tmp_path / "export" / "export.json"
    export.write_text(json.dumps(_privacy_export_bundle([a, b])))
    palace = str(tmp_path / "palace")
    _legacy_mine(export, palace, monkeypatch)

    plan = dc.plan_cleanup(palace)

    assert plan.remove == {}
    assert plan.counts["chats_left_alone"] == 1
