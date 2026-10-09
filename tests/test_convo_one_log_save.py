"""One-log saves read only that log's rows (local patch, step 3).

The Stop hook saves one Claude Code log at a time. A save that cannot use a
bookmark (for example a one-prompt worker log, which has no user turn to
bookmark) used to read every row of the palace twice and rebuild the wing's
hallways: 73 of 81.5 s on a 1.9 MB log against an 84k-row palace (2026-10-03).
The contract under test: a one-log save decides exactly what the whole-palace
reads would decide, reads only that log's rows, and runs the hallway rebuild
and the SQLite quick_check at most once per throttle window. A directory sweep
is unchanged.
"""

import json
import os
import shutil

import chromadb
import pytest

from mempalace import convo_miner
from mempalace.backends.chroma import ChromaCollection
from mempalace.convo_miner import mine_convos

WING = "one_log_test"


class Log:
    """A synthetic Claude Code JSONL session that only ever grows."""

    def __init__(self, path):
        self.path = str(path)
        self.n = 0
        open(self.path, "w").close()

    def _write(self, entry):
        self.n += 1
        entry.setdefault(
            "timestamp", f"2026-10-03T12:{(self.n // 60) % 60:02d}:{self.n % 60:02d}.000Z"
        )
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def user(self, i: int):
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

    def assistant(self, i: int):
        self._write(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"Step {i}. " + "Use a bounded LRU with TTL. " * 8}
                    ],
                },
            }
        )

    def grow(self, start: int, count: int):
        """Full exchanges: one user turn and one reply each."""
        for i in range(start, start + count):
            self.user(i)
            self.assistant(i)


@pytest.fixture
def logs(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    a = Log(d / "0a1b2c3d-session.jsonl")
    a.grow(0, 12)
    b = Log(d / "4e5f6a7b-session.jsonl")
    b.grow(100, 12)
    return a, b


@pytest.fixture
def palace(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_CONVO_BOOKMARKS", raising=False)
    monkeypatch.delenv("MEMPALACE_FASTPATH_POSTMINE_SECS", raising=False)
    return str(tmp_path / "palace")


@pytest.fixture
def whole_palace_reads(monkeypatch):
    """Record every collection.get() that names neither ids nor a where filter."""
    seen = []
    real_get = ChromaCollection.get

    def get(self, **kwargs):
        if kwargs.get("ids") is None and kwargs.get("where") is None:
            seen.append(kwargs.get("offset"))
        return real_get(self, **kwargs)

    monkeypatch.setattr(ChromaCollection, "get", get)
    return seen


@pytest.fixture
def post_mine_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        convo_miner, "_validate_palace_fts5_after_mine", lambda p: calls.append("qc")
    )
    monkeypatch.setattr(
        convo_miner, "_compute_hallways_for_wing_safe", lambda *a, **k: calls.append("hall")
    )
    return calls


def _col(palace):
    return chromadb.PersistentClient(path=palace).get_collection("mempalace_drawers")


def _convo_rows(palace, source_file):
    got = _col(palace).get(where={"source_file": source_file}, include=["metadatas"])
    return [m for m in got["metadatas"] if (m or {}).get("ingest_mode") == "convos"]


def _snapshot(palace):
    got = _col(palace).get(include=["documents", "metadatas"])
    volatile = {"filed_at", "last_modified"}
    return {
        i: (d, {k: v for k, v in (m or {}).items() if k not in volatile})
        for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
    }


def test_one_log_full_save_reads_no_whole_palace(logs, palace, whole_palace_reads, monkeypatch):
    a, b = logs
    mine_convos(a.path, palace, wing=WING)
    monkeypatch.setenv("MEMPALACE_CONVO_BOOKMARKS", "0")  # force the full path
    whole_palace_reads.clear()
    mine_convos(b.path, palace, wing=WING)
    assert _convo_rows(palace, b.path), "the second log was not stored"
    assert whole_palace_reads == [], (
        f"a one-log save read the whole palace ({len(whole_palace_reads)} unfiltered pages)"
    )


def test_one_prompt_log_save_reads_no_whole_palace(logs, palace, whole_palace_reads):
    """The real shape: a worker log with one user turn gets no bookmark, so every
    save of it takes the full path."""
    a, _ = logs
    mine_convos(a.path, palace, wing=WING)
    worker = Log(os.path.join(os.path.dirname(a.path), "8c9d0e1f-worker.jsonl"))
    worker.user(500)
    for i in range(30):
        worker.assistant(500 + i)
    mine_convos(worker.path, palace, wing=WING)
    for i in range(30, 40):
        worker.assistant(500 + i)
    whole_palace_reads.clear()
    mine_convos(worker.path, palace, wing=WING)
    assert not os.path.exists(convo_miner._bookmark_path(palace, worker.path, "exchange", WING)), (
        "test premise: a one-prompt log has no bookmark"
    )
    assert whole_palace_reads == [], (
        f"a one-prompt log's save read the whole palace ({len(whole_palace_reads)} pages)"
    )


def test_one_log_full_saves_throttle_post_mine_steps(logs, palace, post_mine_calls, monkeypatch):
    a, _ = logs
    monkeypatch.setenv("MEMPALACE_CONVO_BOOKMARKS", "0")
    mine_convos(a.path, palace, wing=WING)  # no stamps yet: both run
    assert post_mine_calls == ["hall", "qc"]
    a.grow(800, 2)
    mine_convos(a.path, palace, wing=WING)
    assert post_mine_calls == ["hall", "qc"], (
        "a one-log full save inside the throttle window re-ran the post-mine steps"
    )
    monkeypatch.setenv("MEMPALACE_FASTPATH_POSTMINE_SECS", "0")  # the kill switch
    a.grow(810, 2)
    mine_convos(a.path, palace, wing=WING)
    assert post_mine_calls == ["hall", "qc", "hall", "qc"]


def test_directory_sweep_still_runs_post_mine_steps_every_time(
    logs, palace, post_mine_calls, monkeypatch
):
    a, _ = logs
    monkeypatch.setenv("MEMPALACE_CONVO_BOOKMARKS", "0")
    folder = os.path.dirname(a.path)
    mine_convos(folder, palace, wing=WING)
    a.grow(900, 2)
    mine_convos(folder, palace, wing=WING)
    assert post_mine_calls == ["hall", "qc", "hall", "qc"]


def test_one_log_save_still_skips_a_duplicate_conversation(logs, palace, monkeypatch):
    a, _ = logs
    monkeypatch.setenv("MEMPALACE_CONVO_BOOKMARKS", "0")
    mine_convos(a.path, palace, wing=WING)
    copy = os.path.join(os.path.dirname(a.path), "9a8b7c6d-copy.jsonl")
    shutil.copyfile(a.path, copy)
    mine_convos(copy, palace, wing=WING)
    assert _convo_rows(palace, copy) == [], "the same conversation at a new path was filed again"
    reg = _col(palace).get(where={"source_file": copy}, include=["metadatas"])
    assert [m.get("ingest_mode") for m in reg["metadatas"]] == ["registry"]


def test_one_log_save_skips_an_unchanged_log_and_redoes_a_partial_one(
    logs, palace, monkeypatch, capsys
):
    a, _ = logs
    monkeypatch.setenv("MEMPALACE_CONVO_BOOKMARKS", "0")
    mine_convos(a.path, palace, wing=WING)
    full = _snapshot(palace)
    capsys.readouterr()
    mine_convos(a.path, palace, wing=WING)
    assert "Files skipped (already filed): 1" in capsys.readouterr().out
    col = _col(palace)
    rows = col.get(where={"source_file": a.path}, include=["metadatas"])
    victim = next(i for i, m in zip(rows["ids"], rows["metadatas"]) if m.get("chunk_index") == 3)
    col.delete(ids=[victim])  # a save that crashed between batches
    mine_convos(a.path, palace, wing=WING)
    assert _snapshot(palace) == full, "the partial log was not restored to its full set"


def test_known_limit_one_log_save_files_again_what_a_multi_hash_row_holds(
    logs, palace, tmp_path, monkeypatch
):
    """Known limit, by design: a one-log save looks hashes up by equality, so it
    cannot see a hash inside another file's comma-joined list (a privacy-export
    bundle). The conversation is then filed again: duplication, never loss. The
    live palace held 0 such rows on 2026-10-03."""
    a, _ = logs
    monkeypatch.setenv("MEMPALACE_CONVO_BOOKMARKS", "0")
    probe = str(tmp_path / "probe_palace")
    mine_convos(a.path, probe, wing=WING)
    h = next(m["content_hash"] for m in _convo_rows(probe, a.path) if m.get("content_hash"))
    bundle = os.path.join(str(tmp_path), "export-bundle.json")
    mine_convos(logs[1].path, palace, wing=WING)  # create the palace
    col = _col(palace)
    seed = col.get(where={"source_file": logs[1].path}, include=["metadatas"], limit=1)
    meta = dict(seed["metadatas"][0], source_file=bundle, content_hash="0" * 64 + "," + h)
    col.upsert(ids=["bundle-row-0"], documents=["bundle text"], metadatas=[meta])
    mine_convos(a.path, palace, wing=WING)
    assert _convo_rows(palace, a.path), "the conversation was not filed"


# ── Review findings (GPT-6.1 Sol, 2026-10-03, step 3): each test below failed before its repair ──


def _bundle(path, conversations):
    """A Claude.ai privacy export: one JSON array, one object per conversation."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "chat_messages": [
                        {"sender": "human" if i % 2 == 0 else "assistant", "text": t}
                        for i, t in enumerate(turns)
                    ]
                }
                for turns in conversations
            ],
            fh,
        )


def _convo(tag, n=5, tail=""):
    turns = []
    for i in range(n):
        turns.append(f"{tag} question {i}: how do I size the {tag} cache for key {i}? " * 3)
        turns.append(f"{tag} answer {i}. " + "Use a bounded LRU with a TTL per key. " * 8)
    turns[-1] += tail
    return turns


def _texts(palace):
    return _col(palace).get(include=["documents"])["documents"]


def test_g1_bundle_keeps_a_conversation_an_incomplete_copy_also_holds(tmp_path, palace):
    """Finding 1: a bundle's own comma-joined hash row must still win over a later,
    incomplete single-conversation copy, or the bundle's re-save deletes the text."""
    c1 = _convo("alpha", tail=" UNIQUE-ALPHA-TAIL")
    c2, c3 = _convo("beta"), _convo("gamma")
    a, b = str(tmp_path / "bundle-a.json"), str(tmp_path / "copy-b.json")
    _bundle(a, [c1, c2])
    mine_convos(a, palace, wing=WING)
    _bundle(b, [c1])
    mine_convos(b, palace, wing=WING)  # filed again (known limit): duplication only
    col = _col(palace)
    got = col.get(where={"source_file": b}, include=["documents"])
    tail_ids = [i for i, d in zip(got["ids"], got["documents"]) if "UNIQUE-ALPHA-TAIL" in d]
    assert tail_ids, "test premise: the copy holds the tail"
    col.delete(ids=tail_ids)  # the copy's save crashed before its last drawers
    _bundle(a, [c1, c2, c3])  # the bundle is re-exported with one more conversation
    os.utime(a, (os.path.getmtime(a) + 5, os.path.getmtime(a) + 5))
    mine_convos(a, palace, wing=WING)
    assert any("UNIQUE-ALPHA-TAIL" in d for d in _texts(palace)), (
        "the bundle's re-save dropped a conversation that only an incomplete copy held"
    )


def test_g1_a_bundle_save_sends_no_multi_value_hash_filter(tmp_path, palace, monkeypatch):
    """Finding 2: one $in value per hash overflows SQLite's variable limit on a big
    bundle (32,766 here), and the swallowed error returns no hashes at all."""
    seen = []
    real_get = ChromaCollection.get

    def get(self, **kwargs):
        ch = (kwargs.get("where") or {}).get("content_hash")
        if isinstance(ch, dict) and len(ch.get("$in", [])) > 1:
            seen.append(len(ch["$in"]))
        return real_get(self, **kwargs)

    a = str(tmp_path / "bundle.json")
    _bundle(a, [_convo("one")])
    mine_convos(a, palace, wing=WING)
    monkeypatch.setattr(ChromaCollection, "get", get)
    _bundle(a, [_convo("one"), _convo("two"), _convo("three")])
    os.utime(a, (os.path.getmtime(a) + 5, os.path.getmtime(a) + 5))
    mine_convos(a, palace, wing=WING)
    assert seen == [], f"a bundle's save filtered on {seen} hashes in one query"


def test_g2_deferred_hallways_run_when_due_even_on_an_unchanged_save(
    logs, palace, post_mine_calls, monkeypatch
):
    """Finding 3: a rebuild skipped by the throttle must not wait for the next save
    that files drawers; the first save after the window runs it."""
    a, _ = logs
    monkeypatch.setenv("MEMPALACE_CONVO_BOOKMARKS", "0")
    mine_convos(a.path, palace, wing=WING)  # rebuild at time 0
    a.grow(700, 2)
    mine_convos(a.path, palace, wing=WING)  # files drawers inside the window: deferred
    assert post_mine_calls == ["hall", "qc"]
    stamp = convo_miner._bookmark_dir(palace) / f"{convo_miner._hallway_stamp(WING)}.stamp"
    old = stamp.stat().st_mtime - 4000
    os.utime(stamp, (old, old))  # the window has passed
    mine_convos(a.path, palace, wing=WING)  # unchanged: files nothing
    assert post_mine_calls.count("hall") == 2, (
        f"a deferred hallway rebuild stayed overdue: {post_mine_calls}"
    )
    os.utime(stamp, (old, old))
    mine_convos(a.path, palace, wing=WING)  # nothing deferred any more
    assert post_mine_calls.count("hall") == 2, "an unchanged save rebuilt with no work pending"


def test_g2_deferred_hallways_run_on_an_unchanged_bookmark_save(
    log_with_bookmark, palace, post_mine_calls
):
    log = log_with_bookmark
    log.grow(720, 2)
    mine_convos(log.path, palace, wing=WING)  # bookmark save inside the window: deferred
    assert post_mine_calls.count("hall") == 1
    stamp = convo_miner._bookmark_dir(palace) / f"{convo_miner._hallway_stamp(WING)}.stamp"
    old = stamp.stat().st_mtime - 4000
    os.utime(stamp, (old, old))
    mine_convos(log.path, palace, wing=WING)  # unchanged (bookmark)
    assert post_mine_calls.count("hall") == 2, (
        f"a deferred rebuild stayed overdue on an unchanged bookmark save: {post_mine_calls}"
    )


@pytest.fixture
def log_with_bookmark(logs, palace, post_mine_calls):
    a, _ = logs
    mine_convos(a.path, palace, wing=WING)  # full read: rebuild, stamps, bookmark
    assert os.path.exists(convo_miner._bookmark_path(palace, a.path, "exchange", WING))
    assert post_mine_calls == ["hall", "qc"]
    return a


# ── Repair verification (fresh GPT-6.1 Sol, 2026-10-04): each test below failed first ──


def _drop(palace, source_file, marker):
    col = _col(palace)
    got = col.get(where={"source_file": source_file}, include=["documents"])
    ids = [i for i, d in zip(got["ids"], got["documents"]) if marker in d]
    assert ids, f"test premise: {source_file} holds {marker}"
    col.delete(ids=ids)


def test_g1_bundle_shrunk_to_one_conversation_still_sees_its_own_row(tmp_path, palace):
    """A bundle re-exported with ONE conversation still carries its old comma-joined
    hash; it must win over a later incomplete copy, as in the whole-palace lookup,
    or the bundle is registered as a duplicate and never re-filed."""
    c1 = _convo("alpha", tail=" UNIQUE-ALPHA-TAIL")
    a, b = str(tmp_path / "bundle-a.json"), str(tmp_path / "copy-b.json")
    _bundle(a, [c1, _convo("beta")])
    mine_convos(a, palace, wing=WING)
    _bundle(b, [c1])
    mine_convos(b, palace, wing=WING)
    _drop(palace, a, "UNIQUE-ALPHA-TAIL")  # both saves crashed before their last drawers
    _drop(palace, b, "UNIQUE-ALPHA-TAIL")
    _bundle(a, [c1])
    os.utime(a, (os.path.getmtime(a) + 5, os.path.getmtime(a) + 5))
    mine_convos(a, palace, wing=WING)
    assert any("UNIQUE-ALPHA-TAIL" in d for d in _texts(palace)), (
        "the shrunk bundle was judged a duplicate of an incomplete copy and not re-filed"
    )


def test_g2_a_swallowed_hallway_fetch_error_keeps_the_rebuild_pending(logs, palace, monkeypatch):
    """compute_hallways_for_wing catches its own fetch error and returns []; that must
    not count as a finished rebuild."""
    a, _ = logs
    monkeypatch.setenv("MEMPALACE_CONVO_BOOKMARKS", "0")
    mine_convos(a.path, palace, wing=WING)
    a.grow(730, 2)
    mine_convos(a.path, palace, wing=WING)  # deferred: pending
    pending = (
        convo_miner._bookmark_dir(palace) / f"{convo_miner._hallway_stamp(WING)}-pending.stamp"
    )
    assert pending.exists(), "test premise: the rebuild is pending"
    stamp = convo_miner._bookmark_dir(palace) / f"{convo_miner._hallway_stamp(WING)}.stamp"
    real_get = ChromaCollection.get

    def get(self, **kwargs):
        if kwargs.get("where") == {"wing": WING}:  # the hallway rebuild's fetch
            raise RuntimeError("collection fetch failed")
        return real_get(self, **kwargs)

    monkeypatch.setattr(ChromaCollection, "get", get)
    old = stamp.stat().st_mtime - 4000
    os.utime(stamp, (old, old))
    mine_convos(a.path, palace, wing=WING)  # due: the rebuild runs and its fetch fails
    assert pending.exists(), "a swallowed fetch error cleared the pending rebuild"
