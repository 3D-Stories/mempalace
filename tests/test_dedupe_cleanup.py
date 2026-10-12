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

from pathlib import Path

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
    assert repeat_id not in plan.remove  # the dry run counts it as kept, not removed
    assert repeat_id in plan.referenced_kept
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

    plan = dc.plan_cleanup(palace)
    assert repeat_id not in plan.remove
    assert repeat_id in plan.referenced_kept
    result = dc.apply_cleanup(palace, plan)

    assert repeat_id in _rows(palace)
    assert result["referenced_kept"] == 1


def _refused_as_outdated(palace, plan, tmp_path):
    before = _rows(palace)
    backup = tmp_path / "backup"

    with pytest.raises(dc.PlanOutdated):
        dc.apply_cleanup(palace, plan, backup_path=str(backup))

    assert _rows(palace) == before
    assert not backup.exists()
    assert not (Path(palace) / dc.MARKER_NAME).exists()


def test_a_row_whose_text_changed_since_the_plan_refuses_the_run(legacy, tmp_path):
    palace, lg = legacy
    plan = dc.plan_cleanup(palace)
    repeat_id = next(i for i, why in plan.remove.items() if why == "repeat")
    _col(palace).update(ids=[repeat_id], documents=["now a different text"])

    _refused_as_outdated(palace, plan, tmp_path)


def test_boilerplate_that_changed_but_is_still_boilerplate_refuses_the_run(legacy, tmp_path):
    palace, lg = legacy
    plan = dc.plan_cleanup(palace)
    bp_id = next(i for i, why in plan.remove.items() if why == "boilerplate")
    doc = _rows(palace)[bp_id][0]
    _col(palace).update(ids=[bp_id], documents=[doc + " changed after the plan"])
    assert dc.is_boilerplate(doc + " changed after the plan", plan.listed)

    _refused_as_outdated(palace, plan, tmp_path)


def test_a_row_added_since_the_plan_refuses_the_run(tmp_path):
    palace = str(tmp_path / "palace")
    a, b, c, d = (f"unique text {x} " * 8 for x in "abcd")
    _seed(palace, [(0, a, 5, 0), (1, b, 5, 0), (2, a, 5, 0)])
    plan = dc.plan_cleanup(palace)
    assert plan.remove
    _seed(palace, [(3, c, 5, 0), (4, d, 5, 0)])

    _refused_as_outdated(palace, plan, tmp_path)


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


# --- G2 (review finding 2): label sets as step 8 counts them


def _seed(palace, rows, source="/chats/mixed.jsonl", mtime=1000.0):
    """rows: [(chunk_index, text, chunk_total, chunk_skipped or None)] for one chat."""
    from mempalace.convo_miner import NORMALIZE_VERSION
    from mempalace.palace import get_collection

    metas, ids, docs = [], [], []
    for index, text, total, skipped in rows:
        meta = {
            "source_file": source,
            "wing": WING,
            "room": "general",
            "chunk_index": index,
            "ingest_mode": "convos",
            "extract_mode": "exchange",
            "normalize_version": NORMALIZE_VERSION,
            "chunk_total": total,
            "chunk_hash": _sha(text),
            "source_mtime": mtime,
        }
        if skipped is not None:
            meta["chunk_skipped"] = skipped
        ids.append(f"{os.path.basename(source)}-{index}")
        docs.append(text)
        metas.append(meta)
    get_collection(palace).upsert(ids=ids, documents=docs, metadatas=metas)


def _labels(palace, source="/chats/mixed.jsonl"):
    return {
        m["chunk_index"]: m.get("chunk_skipped")
        for _d, m in _rows(palace).values()
        if m.get("source_file") == source
    }


def test_only_the_label_set_that_lost_rows_is_relabelled(tmp_path):
    palace = str(tmp_path / "palace")
    a, b, c, d, e = (f"unique text {x} " * 8 for x in "abcde")
    # One save's set (total 5): pieces 0-2. A later save's set (total 7): 3, 5, 6.
    _seed(
        palace, [(0, a, 5, 0), (1, b, 5, 0), (2, c, 5, 0), (3, d, 7, 0), (5, a, 7, 0), (6, e, 7, 0)]
    )

    dc.apply_cleanup(palace, dc.plan_cleanup(palace))

    labels = _labels(palace)
    assert 5 not in labels  # the repeat of piece 0 is gone
    assert [labels[i] for i in (0, 1, 2)] == [0, 0, 0]  # the other set is untouched
    assert [labels[i] for i in (3, 6)] == [1, 1]


def test_a_chat_with_a_malformed_skip_count_is_left_alone(tmp_path):
    palace = str(tmp_path / "palace")
    a, b, c = (f"unique text {x} " * 8 for x in "abc")
    _seed(palace, [(0, a, 5, 0), (1, b, 5, 0), (2, c, 5, 0), (5, a, 7, "bad"), (6, a, 7, "bad")])
    before = _rows(palace)

    plan = dc.plan_cleanup(palace)
    dc.apply_cleanup(palace, plan)

    assert not plan.remove
    assert plan.counts["chats_left_alone"] == 1
    assert _rows(palace) == before


@pytest.mark.parametrize("skipped", [0, None], ids=["labelled", "pre-step-8"])
def test_already_saved_answers_do_not_change_for_mixed_label_sets(tmp_path, skipped):
    from mempalace.palace import file_already_mined

    palace = str(tmp_path / "palace")
    source = tmp_path / "mixed.jsonl"
    source.write_text("x")
    os.utime(source, (1000.0, 1000.0))
    a, b, c, d = (f"unique text {x} " * 8 for x in "abcd")
    # Two incomplete sets sharing one mtime: the first (total 5) lacks 3 and 4; the
    # later one (total 9) holds two repeats of the first set's text.
    _seed(palace, [(0, a, 5, skipped), (1, b, 5, skipped), (2, c, 5, skipped)], source=str(source))
    _seed(palace, [(5, a, 9, skipped), (6, d, 9, skipped), (7, b, 9, skipped)], source=str(source))

    def answers():
        col = _col(palace)
        return (
            prefetch_mined_set(col).get(str(source), "absent"),
            file_already_mined(col, str(source), check_mtime=True),
        )

    before = answers()
    assert before == ("absent", False)
    dc.apply_cleanup(palace, dc.plan_cleanup(palace))
    assert answers() == before


# --- G3 (review findings 3 and 4): a bookmarked chat keeps a usable bookmark


def _boilerplate_chat(path, repeats, unique_tail):
    lg = Log(path)
    for _ in range(repeats):
        say(lg, CAVEAT, "ok")
    if unique_tail:
        lg.grow(0, 1)
    return lg


def test_a_bookmarked_chat_that_would_lose_every_piece_before_its_bookmark_is_left_alone(
    logs, tmp_path, monkeypatch
):
    palace = str(tmp_path / "palace")
    lg = _boilerplate_chat(logs / "0a1b2c3d-session.jsonl", 40, unique_tail=True)
    _legacy_mine(lg.path, palace, monkeypatch)
    bm_before = json.loads(_bm(palace, lg).read_text())
    assert all(i < bm_before["base"] for i in range(40))  # every boilerplate piece is before base
    before = _rows(palace)

    plan = dc.plan_cleanup(palace)
    result = dc.apply_cleanup(palace, plan)

    assert not plan.remove and plan.counts["chats_left_alone"] == 1
    assert result.get("bookmarks_dropped", 0) == 0
    assert _rows(palace) == before
    assert json.loads(_bm(palace, lg).read_text()) == bm_before


def test_a_bookmarked_chat_where_keep_one_fires_is_left_alone(logs, tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    lg = _boilerplate_chat(logs / "0a1b2c3d-session.jsonl", 40, unique_tail=False)
    _legacy_mine(lg.path, palace, monkeypatch)
    assert _bm(palace, lg).exists()
    bm_before = _bm(palace, lg).read_text()
    before = _rows(palace)

    plan = dc.plan_cleanup(palace)
    dc.apply_cleanup(palace, plan)

    assert not plan.remove and plan.counts["chats_left_alone"] == 1
    assert _rows(palace) == before
    assert _bm(palace, lg).read_text() == bm_before


def test_the_command_names_why_chats_were_left_alone(tmp_path, capsys):
    palace = str(tmp_path / "palace")
    a, b, c = (f"unique text {x} " * 8 for x in "abc")
    _seed(palace, [(0, a, 5, 0), (1, b, 5, 0), (2, c, 5, 0), (5, a, 7, "bad"), (6, a, 7, "bad")])

    _cli("--palace", palace, "dedupe-cleanup")

    assert "left alone: label sets 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "rows",
    [
        # More rows than the total: removing 2 would push the count to the total.
        [(0, "A", 2, 0), (1, "A", 2, 0), (2, "A", 2, 0), (3, "B", 2, 0), (4, "C", 2, 0)],
        # A total that is not a number.
        [(0, "A", "5", 0), (1, "B", "5", 0), (2, "A", "5", 0)],
    ],
    ids=["count-reaches-total", "text-total"],
)
def test_a_label_set_that_cannot_take_the_count_leaves_the_chat_alone(tmp_path, rows):
    palace = str(tmp_path / "palace")
    _seed(palace, [(i, f"text {t} " * 10, total, skip) for i, t, total, skip in rows])
    before = _rows(palace)

    plan = dc.plan_cleanup(palace)
    dc.apply_cleanup(palace, plan)

    assert not plan.remove and plan.left_alone == {"label sets": 1}
    assert _rows(palace) == before


@pytest.mark.parametrize("later_total", [2, None], ids=["complete-set", "no-total"])
def test_a_set_that_would_lose_every_row_and_change_the_answer_leaves_the_chat_alone(
    tmp_path, later_total
):
    from mempalace.palace import file_already_mined

    palace = str(tmp_path / "palace")
    source = tmp_path / "chat.jsonl"
    source.write_text("x")
    os.utime(source, (2000.0, 2000.0))
    a, b, c = (f"unique text {x} " * 8 for x in "abc")
    _seed(palace, [(0, a, 3, 0), (1, b, 3, 0), (2, c, 3, 0)], source=str(source), mtime=1000.0)
    later = [(3, a, later_total, None), (4, b, later_total, None)]
    _seed(palace, later, source=str(source), mtime=2000.0)
    assert file_already_mined(_col(palace), str(source), check_mtime=True)

    plan = dc.plan_cleanup(palace)
    dc.apply_cleanup(palace, plan)

    assert not plan.remove and plan.left_alone == {"label sets": 1}
    assert file_already_mined(_col(palace), str(source), check_mtime=True)


def test_a_bookmarked_chat_missing_its_anchor_row_is_left_alone(legacy):
    palace, lg = legacy
    bm = json.loads(_bm(palace, lg).read_text())
    anchor = bm["base"] - 1
    gone = next(i for i, (_d, m) in _rows(palace).items() if m.get("chunk_index") == anchor)
    _col(palace).delete(ids=[gone])

    plan = dc.plan_cleanup(palace)
    dc.apply_cleanup(palace, plan)

    assert not plan.remove and plan.left_alone == {"bookmark": 1}
    assert json.loads(_bm(palace, lg).read_text()) == bm


# --- verification round 1 (sysop#297): the second correction ---------------------


def _left_alone_whole(palace, reason):
    before = _rows(palace)
    plan = dc.plan_cleanup(palace)
    dc.apply_cleanup(palace, plan)
    assert not plan.remove and plan.left_alone == {reason: 1}
    assert _rows(palace) == before


def test_relabelling_never_merges_two_label_sets(tmp_path):
    """0-3 count 4 of 5 (skip 0), 4 counts 1 of 4 (total 5, skip 1): both incomplete. Removing
    repeat 3 would relabel 0-2 to skip 1 and merge them with 4 into a complete set."""
    palace = str(tmp_path / "palace")
    a, b, c, d = (f"unique text {x} " * 8 for x in "abcd")
    _seed(palace, [(0, a, 5, 0), (1, b, 5, 0), (2, c, 5, 0), (3, a, 5, 0), (4, d, 5, 1)])

    _left_alone_whole(palace, "label sets")


@pytest.mark.parametrize("skip", [0.0, 1.0])
def test_a_float_skip_count_leaves_the_chat_alone(tmp_path, skip):
    palace = str(tmp_path / "palace")
    a, b = (f"unique text {x} " * 8 for x in "ab")
    _seed(palace, [(0, a, 3, skip), (1, b, 3, skip), (2, a, 3, skip)])

    _left_alone_whole(palace, "label sets")


def test_a_text_total_anywhere_in_the_chat_leaves_it_alone(tmp_path):
    palace = str(tmp_path / "palace")
    a, b, z = (f"unique text {x} " * 8 for x in "abz")
    _seed(palace, [(0, a, 3, 0), (1, b, 3, 0), (2, a, 3, 0)])
    _seed(palace, [(5, z, "5", 0)], mtime=2000.0)  # untouched by the cleanup

    _left_alone_whole(palace, "label sets")


def test_a_new_anchor_without_a_hash_leaves_the_chat_alone(legacy):
    palace, lg = legacy
    bm = json.loads(_bm(palace, lg).read_text())
    planned = dc.plan_cleanup(palace).remove
    col = _col(palace)
    kept = [
        d
        for d, (_doc, m) in _rows(palace).items()
        if m.get("chunk_index", bm["base"]) < bm["base"] and d not in planned
    ]
    col.update(ids=kept, metadatas=[{"chunk_hash": ""}] * len(kept))

    _left_alone_whole(palace, "bookmark")
    assert json.loads(_bm(palace, lg).read_text()) == bm


# --- recovery (owner, 2026-10-12): the backup taken before any change is the undo


def _stopped(legacy, monkeypatch, tmp_path, step="_apply_metadata"):
    palace, lg = legacy
    before = _rows(palace)
    bm_before = _bm(palace, lg).read_text()
    backup = str(tmp_path / "backup")

    def _crash(*_a, **_k):
        raise RuntimeError("killed")

    with monkeypatch.context() as mp:
        mp.setattr(dc, step, _crash)
        with pytest.raises(RuntimeError, match="killed"):
            dc.apply_cleanup(palace, dc.plan_cleanup(palace), backup_path=backup)
    return palace, lg, backup, before, bm_before


@pytest.mark.parametrize("step", ["_apply_deletes", "_apply_metadata", "_apply_bookmarks"])
def test_a_run_stopped_part_way_leaves_its_marker_and_a_full_backup(
    legacy, monkeypatch, tmp_path, step
):
    palace, lg, backup, before, bm_before = _stopped(legacy, monkeypatch, tmp_path, step)

    marker = json.loads((Path(palace) / dc.MARKER_NAME).read_text())
    assert marker["palace"] == os.path.join(backup, "palace")
    assert _rows(marker["palace"]) == before  # the palace as it was before any change
    assert (Path(marker["bookmarks"]) / _bm(palace, lg).name).read_text() == bm_before
    assert not (Path(marker["palace"]) / dc.MARKER_NAME).exists()


def test_a_stopped_run_blocks_saves_cleanups_and_the_command(legacy, monkeypatch, tmp_path, capsys):
    palace, lg, backup, _before, _bm_before = _stopped(legacy, monkeypatch, tmp_path)
    lg.grow(11, 2)
    rows = _rows(palace)

    with pytest.raises(dc.CleanupInterrupted, match="restore"):
        mine_convos(lg.path, palace, wing=WING)
    with pytest.raises(dc.CleanupInterrupted, match="restore"):
        dc.apply_cleanup(palace, dc.plan_cleanup(palace), backup_path=str(tmp_path / "b2"))
    capsys.readouterr()
    dry = _cli("--palace", palace, "dedupe-cleanup")
    applied = _cli("--palace", palace, "dedupe-cleanup", "--apply", "--yes")

    out = capsys.readouterr().out
    assert dry == 4 and applied == 4
    assert os.path.join(backup, "palace") in out
    assert _rows(palace) == rows
    assert not (tmp_path / "b2").exists()


def test_an_existing_backup_path_is_refused_before_anything_changes(legacy, tmp_path):
    palace, _lg = legacy
    before = _rows(palace)
    taken = tmp_path / "taken"
    taken.mkdir()

    with pytest.raises(FileExistsError):
        dc.apply_cleanup(palace, dc.plan_cleanup(palace), backup_path=str(taken))

    assert _rows(palace) == before
    assert not (Path(palace) / dc.MARKER_NAME).exists()


def test_a_finished_run_removes_its_marker_and_keeps_its_backup(legacy, tmp_path):
    palace, _lg = legacy
    backup = str(tmp_path / "backup")

    result = dc.apply_cleanup(palace, dc.plan_cleanup(palace), backup_path=backup)

    assert result["backup"] == backup
    assert os.path.isdir(os.path.join(backup, "palace"))
    assert not (Path(palace) / dc.MARKER_NAME).exists()
    again = dc.apply_cleanup(palace, dc.plan_cleanup(palace), backup_path=str(tmp_path / "b2"))
    assert "backup" not in again  # nothing left to change, so nothing is copied
    assert not (tmp_path / "b2").exists()


def test_a_chat_save_works_after_a_finished_run(legacy, tmp_path):
    palace, lg = legacy
    dc.apply_cleanup(palace, dc.plan_cleanup(palace), backup_path=str(tmp_path / "backup"))
    lg.grow(11, 2)
    before = len(_rows(palace))

    mine_convos(lg.path, palace, wing=WING)

    assert len(_rows(palace)) > before


# --- step 8's real scope (D3) and a valid replacement bookmark (D6)


def test_a_source_held_under_another_wing_is_left_alone(tmp_path):
    from mempalace.palace import get_collection

    palace = str(tmp_path / "palace")
    a, b = (f"unique text {x} " * 8 for x in "ab")
    _seed(palace, [(0, a, 3, 0), (1, b, 3, 0), (2, a, 3, 0)])
    meta = dict(_rows(palace)["mixed.jsonl-0"][1], wing="elsewhere")
    get_collection(palace).upsert(ids=["elsewhere-0"], documents=["other wing"], metadatas=[meta])
    before = _rows(palace)

    plan = dc.plan_cleanup(palace)
    dc.apply_cleanup(palace, plan)

    assert not plan.remove and plan.left_alone == {"shared source": 2}  # both wings' chats
    assert _rows(palace) == before


def test_a_chat_with_an_older_version_row_is_left_alone(tmp_path):
    palace = str(tmp_path / "palace")
    a, b = (f"unique text {x} " * 8 for x in "ab")
    _seed(palace, [(0, a, 3, 0), (1, b, 3, 0), (2, a, 3, 0)])
    _col(palace).update(ids=["mixed.jsonl-1"], metadatas=[{"normalize_version": 1}])

    _left_alone_whole(palace, "older version")


def test_a_chat_with_two_complete_file_times_is_left_alone(tmp_path):
    palace = str(tmp_path / "palace")
    a, b, c = (f"unique text {x} " * 8 for x in "abc")
    _seed(palace, [(0, a, 3, 0), (1, b, 3, 0), (2, a, 3, 0)])
    _seed(palace, [(5, c, 1, 0)], mtime=2000.0)

    _left_alone_whole(palace, "label sets")


def test_a_new_anchor_with_an_unusable_hash_leaves_the_chat_alone(legacy):
    palace, lg = legacy
    bm = json.loads(_bm(palace, lg).read_text())
    planned = dc.plan_cleanup(palace).remove
    kept = [
        d
        for d, (_doc, m) in _rows(palace).items()
        if m.get("chunk_index", bm["base"]) < bm["base"] and d not in planned
    ]
    _col(palace).update(ids=kept, metadatas=[{"chunk_hash": "not-hex"}] * len(kept))

    _left_alone_whole(palace, "bookmark")
    assert json.loads(_bm(palace, lg).read_text()) == bm


def test_the_backup_is_taken_before_the_palace_database_is_opened(legacy, tmp_path, monkeypatch):
    import mempalace.palace as palace_mod

    palace, _lg = legacy
    backup = tmp_path / "backup"
    seen = []
    real = palace_mod.get_collection

    def _watch(*a, **k):
        seen.append((backup / "palace" / "chroma.sqlite3").exists())
        return real(*a, **k)

    monkeypatch.setattr(palace_mod, "get_collection", _watch)
    dc.apply_cleanup(palace, dc.plan_cleanup(palace), backup_path=str(backup))

    assert seen and all(seen)


def test_an_existing_backup_path_is_refused_before_the_database_is_opened(
    legacy, tmp_path, monkeypatch
):
    import mempalace.palace as palace_mod

    palace, _lg = legacy
    taken = tmp_path / "taken"
    taken.mkdir()

    def _opened(*_a, **_k):
        raise AssertionError("the palace database was opened")

    monkeypatch.setattr(palace_mod, "get_collection", _opened)
    with pytest.raises(FileExistsError):
        dc.apply_cleanup(palace, dc.plan_cleanup(palace), backup_path=str(taken))


@pytest.mark.parametrize(
    "argv",
    [["--write-list", "REVIEW"], ["--candidates", "OUT"], ["--apply"]],
    ids=["write-list", "candidates", "unconfirmed-apply"],
)
def test_a_stopped_run_blocks_every_mode_of_the_command(
    legacy, monkeypatch, tmp_path, capsys, argv
):
    palace, _lg, backup, _before, _bm_before = _stopped(legacy, monkeypatch, tmp_path)
    review = tmp_path / "review.md"
    review.write_text("- [x] " + "a" * 64 + " | 1 chats | 1 rows | x\n")
    argv = [{"REVIEW": str(review), "OUT": str(tmp_path / "out.md")}.get(a, a) for a in argv]
    capsys.readouterr()

    code = _cli("--palace", palace, "dedupe-cleanup", *argv)

    out = capsys.readouterr().out
    assert code == 4
    assert os.path.join(backup, "palace") in out
    assert not (tmp_path / "out.md").exists()
    assert not (tmp_path / "cfg" / "boilerplate.json").exists()


@pytest.mark.parametrize("mtime", ["1000.0", None], ids=["text-time", "no-time"])
def test_a_row_whose_file_time_is_not_a_number_leaves_the_chat_alone(tmp_path, mtime):
    palace = str(tmp_path / "palace")
    a, b = (f"unique text {x} " * 8 for x in "ab")
    _seed(palace, [(0, a, 3, 0), (1, b, 3, 0), (2, a, 3, 0)])
    col = _col(palace)
    got = col.get(ids=["mixed.jsonl-1"], include=["metadatas", "embeddings"])
    meta = dict(got["metadatas"][0])
    if mtime is None:
        meta.pop("source_mtime")
    else:
        meta["source_mtime"] = mtime
    col.delete(ids=["mixed.jsonl-1"])  # an upsert merges labels, so it cannot drop one
    col.add(ids=["mixed.jsonl-1"], documents=[b], metadatas=[meta], embeddings=got["embeddings"])

    _left_alone_whole(palace, "label sets")


def test_a_chat_with_a_row_without_text_is_left_alone(tmp_path):
    palace = str(tmp_path / "palace")
    a = "unique text a " * 8
    _seed(palace, [(0, a, 3, 0), (2, a, 3, 0)])
    col = _col(palace)
    got = col.get(ids=["mixed.jsonl-0"], include=["metadatas", "embeddings"])
    meta = dict(got["metadatas"][0], chunk_index=1)
    col.add(ids=["mixed.jsonl-1"], metadatas=[meta], embeddings=got["embeddings"])  # no text

    _left_alone_whole(palace, "missing text")


# --- verification round 4 should-fix findings (prepared for the owner, unverified)


def test_a_file_time_a_float_cannot_hold_leaves_the_chat_alone(tmp_path):
    """2**53 and 2**53 + 1 are two raw times but one float time: file_already_mined
    and prefetch_mined_set would group the rows differently."""
    palace = str(tmp_path / "palace")
    a, b, c = (f"unique text {x} " * 8 for x in "abc")
    _seed(palace, [(0, a, 4, 0), (1, a, 4, 0)], mtime=2**53)
    _seed(palace, [(2, b, 4, 0), (3, c, 4, 0)], mtime=2**53 + 1)

    _left_alone_whole(palace, "label sets")


@pytest.mark.parametrize("inside", ["palace", "bookmarks"])
def test_a_backup_inside_what_it_copies_is_refused(legacy, tmp_path, inside):
    from mempalace.convo_miner import _bookmark_dir

    palace, _lg = legacy
    before = _rows(palace)
    root = Path(palace) if inside == "palace" else _bookmark_dir(palace)
    target = root / "nested-backup"

    with pytest.raises(ValueError, match="inside"):
        dc.apply_cleanup(palace, dc.plan_cleanup(palace), backup_path=str(target))

    assert not target.exists()
    assert _rows(palace) == before
    assert not (Path(palace) / dc.MARKER_NAME).exists()
