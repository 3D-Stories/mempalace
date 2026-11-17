"""The Stop hook reads only what was appended, and its hub wait fits its limit.

Claude Code kills the Stop hook after the ``timeout`` in
``.claude-plugin/hooks/hooks.json`` (30 s). Two things pushed every Stop of a
large session past it:

* the hook parsed the whole transcript on every Stop, and a second time on a
  save Stop. Transcripts reach gigabytes;
* it then waited up to 60 s for the hub's diary write.

A killed save Stop never wrote its ``_last_save`` marker, so the next Stop
tried the same save again, and the hub filed one more copy of the note each
time.

``_count_human_messages`` and ``_extract_recent_messages`` are the reference
throughout: the incremental scan must return exactly what they return for the
same bytes.
"""

import builtins
import io
import json
import os
import random
import signal
import socket
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mempalace.hooks_cli as hooks_cli

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_hook_state(monkeypatch, tmp_path):
    """An existing, per-test palace root, and no real mine is ever spawned."""
    root = tmp_path / ".mempalace"
    state_dir = root / "hook_state"
    state_dir.mkdir(parents=True)
    monkeypatch.setattr(hooks_cli, "PALACE_ROOT", root)
    monkeypatch.setattr(hooks_cli, "STATE_DIR", state_dir)
    monkeypatch.setattr(hooks_cli, "_MINE_PID_DIR", state_dir / "mine_pids")
    monkeypatch.setattr(hooks_cli, "_state_dir_initialized", False)
    monkeypatch.setattr(hooks_cli, "_ingest_transcript", MagicMock(return_value=True))
    monkeypatch.setattr(hooks_cli, "_maybe_auto_ingest", MagicMock())
    monkeypatch.setenv("MEMPALACE_HUB_FORWARD", "0")
    return state_dir


# ── Transcript builders ──────────────────────────────────────────────────


def _line(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _user(content, **extra) -> dict:
    entry = {"type": "user", "message": {"role": "user", "content": content}}
    entry.update(extra)
    return entry


def _codex(message, kind="user_message") -> dict:
    return {"type": "event_msg", "payload": {"type": kind, "message": message}}


def _assistant(text) -> dict:
    return {"type": "assistant", "message": {"role": "assistant", "content": text}}


def _every_row_shape() -> list:
    """One line per transcript shape the two reference functions tell apart."""
    rows = [
        # Claude Code rows: string content, raw and escaped multibyte text
        _line(_user("why does the résumé parser drop 日本語 names? 🎉")),
        json.dumps(_user("escaped résumé 日本語 🎉 text")).encode("ascii"),
        _line(_user("x" * 150 + "é" * 150)),  # cut at 200 characters, not bytes
        _line(_user("big " + "payload " * 1500)),  # longer than one read buffer
        # list content
        _line(
            _user(
                [{"type": "text", "text": "list content"}, {"type": "text", "text": "two blocks"}]
            )
        ),
        _line(_user(["a bare string", 7, None, {"type": "text", "text": "mixed list"}])),
        # tool_result lists: counted as an exchange, never a recent message
        _line(_user([{"type": "tool_result", "tool_use_id": "toolu_1", "content": "42 passed"}])),
        _line(
            _user(
                [
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "a"},
                    {"type": "tool_result", "tool_use_id": "toolu_3", "content": "b"},
                ]
            )
        ),
        _line(_assistant("sure, on it")),
        # Codex event_msg rows
        _line(_codex("codex asks about the diff")),
        _line(_codex("codex answers", kind="agent_message")),
        _line(_codex("<command-message>status</command-message>")),
        _line(_codex("   ")),
        _line(_codex(5)),
        _line({"type": "event_msg", "payload": "not a dict"}),
        _line({"type": "event_msg", "message": "a string", "payload": {"type": "user_message"}}),
        # <command-message> rows: the count skips them wherever the tag sits,
        # the recent rule only when the message opens with it
        _line(_user("<command-message>mempalace:status</command-message>")),
        _line(_user("run it <command-message>inline</command-message> please")),
        _line(_user([{"type": "text", "text": "<command-message>commit</command-message>"}])),
        # harness boilerplate
        _line(_user("  \n <system-reminder>indented</system-reminder>")),
        _line(_user("the <task-notification> tag quoted mid-sentence")),
        # empty and odd content
        _line(_user("")),
        _line(_user("   \n\t ")),
        _line(_user(None)),
        _line(_user(5)),
        _line({"type": "user", "message": {"role": "user"}}),
        _line({"event_message": {"role": "user", "content": "seen by the recent rule only"}}),
        _line({"type": "summary", "summary": "no message key"}),
        # valid JSON that is not an object
        b"[1, 2, 3]",
        b'"just a string"',
        b"5",
        b"null",
        # invalid JSON lines
        b'{"message": {"role": "user", "content": "cut off',
        b"not json at all",
        b'{"message": {"role": "user", "content": "twice"}} {"again": 1}',
        # blank lines
        b"",
        b"   ",
        b"\t",
        # bytes that are not UTF-8, and padding around a valid row
        b'{"message": {"role": "user", "content": "bad \xff\xfe bytes \xc3"}}',
        b'  {"message": {"role": "user", "content": "padded with spaces"}}  ',
    ]
    rows += [
        _line(_user(marker + " injected")) for marker in hooks_cli._HARNESS_BOILERPLATE_MARKERS
    ]
    rows += [_line(_user(f"question {i} about topic {i % 7}")) for i in range(50)]
    return rows


def _transcript_bytes(rng, ends_with_newline: bool) -> bytes:
    rows = _every_row_shape()
    rng.shuffle(rows)
    # The last line is one the full parse counts even with no newline after it.
    rows.append(_line(_user("the last line of the file")))
    blob = b"".join(row + (b"\r\n" if rng.random() < 0.2 else b"\n") for row in rows)
    return blob if ends_with_newline else blob.rstrip(b"\r\n")


def _reference(path) -> tuple:
    return (
        hooks_cli._count_human_messages(str(path)),
        hooks_cli._extract_recent_messages(str(path)),
    )


def _scan(path):
    """Run the incremental scan; fail if it quietly used the full-parse fallback."""
    fell_back = AssertionError("the scan fell back to a full parse")
    with patch.object(hooks_cli, "_count_human_messages", side_effect=fell_back):
        with patch.object(hooks_cli, "_extract_recent_messages", side_effect=fell_back):
            return hooks_cli._scan_transcript(str(path))


def _state(path) -> dict:
    return json.loads(hooks_cli._scan_state_path(Path(path).resolve()).read_text(encoding="utf-8"))


def _assert_equals_full_parse(path, where=""):
    scan = _scan(path)
    assert scan.complete, where
    count, recent = _reference(path)
    assert scan.count == count, f"count differs {where}"
    assert scan.recent == recent, f"recent messages differ {where}"
    return scan


# ── Counting the bytes really read from the transcript ───────────────────


class _CountingRaw(io.FileIO):
    """A raw file that adds up every byte the OS hands back."""

    def __init__(self, path, counter):
        super().__init__(path, "rb")
        self._counter = counter

    def readinto(self, buffer):
        n = super().readinto(buffer)
        self._counter["bytes"] += n or 0
        return n

    def read(self, size=-1):
        data = super().read(size)
        self._counter["bytes"] += len(data or b"")
        return data

    def readall(self):
        data = super().readall()
        self._counter["bytes"] += len(data or b"")
        return data


@contextmanager
def _count_transcript_reads(path):
    """Count bytes read from ``path`` through ``open()``, in text or binary mode."""
    counter = {"bytes": 0, "opens": 0}
    real_open = builtins.open
    target = os.path.realpath(str(path))

    def counting_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, **kw):
        is_target = (
            isinstance(file, (str, os.PathLike)) and os.path.realpath(os.fspath(file)) == target
        )
        if not is_target or "r" not in mode or "+" in mode:
            return real_open(file, mode, buffering, encoding, errors, newline, **kw)
        counter["opens"] += 1
        buffered = io.BufferedReader(_CountingRaw(target, counter))
        if "b" in mode:
            return buffered
        return io.TextIOWrapper(buffered, encoding=encoding, errors=errors, newline=newline)

    with patch("builtins.open", counting_open):
        yield counter


# A buffered reader pulls a whole buffer for the fingerprint and may read one
# more past the end, so "only the new bytes" allows a few buffers of slack.
_READ_SLACK = 4 * io.DEFAULT_BUFFER_SIZE + 8192


def _run_stop(transcript, session_id="sess") -> dict:
    outputs = []
    with patch.object(hooks_cli, "_output", side_effect=outputs.append):
        hooks_cli.hook_stop(
            {
                "session_id": session_id,
                "stop_hook_active": False,
                "transcript_path": str(transcript),
            },
            "claude-code",
        )
    assert len(outputs) == 1, outputs
    return outputs[0]


def _hook_log() -> str:
    log = hooks_cli.STATE_DIR / "hook.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _big_transcript(path, user_rows: int, filler_rows: int = 1000) -> int:
    """A transcript that is mostly assistant output, with ``user_rows`` exchanges."""
    filler = _line(_assistant("x" * 2000)) + b"\n"
    with open(path, "wb") as f:
        f.write(_line(_user("opening question", cwd="/home/u/proj")) + b"\n")
        for i in range(1, user_rows):
            f.write(filler * (filler_rows // user_rows))
            f.write(_line(_user(f"question {i} about the parser")) + b"\n")
        f.write(filler * (filler_rows // user_rows))
    return path.stat().st_size


# ── 1. Equality with the full parse, after every append ──────────────────


@pytest.mark.parametrize("ends_with_newline", [True, False])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_incremental_numbers_equal_the_full_parse_after_every_append(
    tmp_path, seed, ends_with_newline
):
    rng = random.Random(seed)
    blob = _transcript_bytes(rng, ends_with_newline)
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"")

    written = 0
    mid_line_steps = 0
    scan = None
    while written < len(blob):
        step = rng.choice(
            [1, 2, 7]
            + [rng.randint(1, 60), rng.randint(1, 120), rng.randint(1, 120)]
            + [rng.randint(60, 400), rng.randint(400, 1500)]
        )
        with open(path, "ab") as f:
            f.write(blob[written : written + step])
        written = min(len(blob), written + step)

        scan = _assert_equals_full_parse(path, where=f"at byte {written} (seed {seed})")
        if rng.random() < 0.1:  # a Stop with nothing appended since the last one
            again = _assert_equals_full_parse(path, where=f"again at byte {written}")
            assert (again.count, again.recent) == (scan.count, scan.recent)
        # A line with no newline yet is counted in the answer but not consumed.
        assert _state(path)["offset"] == blob.rfind(b"\n", 0, written) + 1
        if blob[written - 1 : written] != b"\n":
            mid_line_steps += 1

    assert mid_line_steps > 10, "the steps must end in the middle of lines too"
    assert scan.count > 60
    assert len(scan.recent) == hooks_cli._RECENT_MSG_COUNT
    if not ends_with_newline:
        assert scan.recent[-1] == "the last line of the file"
        assert _state(path)["offset"] < len(blob)


def test_scan_of_a_missing_or_empty_transcript_is_zero(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    scan = _scan(empty)
    assert (scan.count, scan.recent, scan.complete) == (0, [], True)

    scan = _scan(tmp_path / "missing.jsonl")
    assert (scan.count, scan.recent, scan.complete) == (0, [], True)


def test_scan_rejects_the_paths_the_full_parse_rejects(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_bytes(_line(_user("not a transcript")) + b"\n")
    scan = _scan(notes)
    assert (scan.count, scan.recent, scan.complete) == (0, [], True)
    assert hooks_cli._count_human_messages(str(notes)) == 0

    scan = hooks_cli._scan_transcript("../../etc/passwd")
    assert (scan.count, scan.recent, scan.complete) == (0, [], True)
    assert "WARNING: transcript_path rejected by validator" in _hook_log()
    assert not list(hooks_cli.STATE_DIR.glob("scan_*")), "a rejected path keeps no state"


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(signal, "SIGALRM"),
    reason="requires POSIX FIFOs and SIGALRM",
)
def test_scan_never_opens_a_fifo(tmp_path):
    """Opening a FIFO for reading blocks until a writer shows up: forever, in a hook."""
    fifo = tmp_path / "session.jsonl"
    os.mkfifo(fifo)

    class Blocked(BaseException):
        pass

    def fire(signum, frame):
        raise Blocked("the scan blocked on a FIFO")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, 5.0)
    try:
        scan = _scan(fifo)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert (scan.count, scan.recent, scan.complete) == (0, [], True)


# ── 2. Only the new bytes are read ───────────────────────────────────────


def test_second_stop_reads_only_the_appended_bytes(tmp_path):
    path = tmp_path / "session.jsonl"
    size = _big_transcript(path, user_rows=1, filler_rows=2000)
    assert size > 4_000_000

    with _count_transcript_reads(path) as first:
        assert _run_stop(path) == {}
    assert first["bytes"] >= size

    appended = _line(_user("a second question")) + b"\n" + (_line(_assistant("ok")) + b"\n") * 3
    with open(path, "ab") as f:
        f.write(appended)

    with _count_transcript_reads(path) as second:
        assert _run_stop(path) == {}

    assert second["bytes"] >= len(appended)
    assert second["bytes"] <= len(appended) + _READ_SLACK, (
        f"the second Stop read {second['bytes']} bytes of a {size + len(appended)} byte "
        f"transcript to pick up {len(appended)} new ones"
    )
    assert "Session sess: 2 exchanges, 2 since last save" in _hook_log()


# ── 3. Restart when the file is not the one that was scanned ─────────────


def _two_different_transcripts(size_hint=40):
    first = b"".join(_line(_user(f"first file, question {i}")) + b"\n" for i in range(size_hint))
    other = b"".join(_line(_user(f"other file, question {i}")) + b"\n" for i in range(size_hint))
    return first, other


@pytest.mark.parametrize("rows", [40, 400])
def test_scan_restarts_when_the_transcript_is_truncated(tmp_path, rows):
    """Shorter than the saved offset, whether or not the fingerprinted head survives."""
    path = tmp_path / "session.jsonl"
    first, _ = _two_different_transcripts(rows)
    path.write_bytes(first)
    _assert_equals_full_parse(path)

    shorter = first[: len(first) // 2 + 17]  # ends mid-line
    assert (len(shorter) > hooks_cli._SCAN_FINGERPRINT_BYTES) == (rows == 400)
    path.write_bytes(shorter)
    scan = _assert_equals_full_parse(path, where="after truncation")
    assert 0 < scan.count < rows
    assert _state(path)["offset"] == shorter.rfind(b"\n") + 1


@pytest.mark.parametrize("extra_rows", [0, 25])
def test_scan_restarts_when_the_transcript_is_replaced(tmp_path, extra_rows):
    """Same size or larger: only the fingerprint can tell it is another file."""
    path = tmp_path / "session.jsonl"
    first, other = _two_different_transcripts()
    path.write_bytes(first)
    _assert_equals_full_parse(path)

    replacement = other + b"".join(
        _line(_user(f"other file, extra {i}")) + b"\n" for i in range(extra_rows)
    )
    assert len(replacement) == len(first) + (len(replacement) - len(other))
    assert len(other) == len(first), "the same size: only the fingerprint can tell"
    path.write_bytes(replacement)
    scan = _assert_equals_full_parse(path, where="after replacement")
    assert scan.count == 40 + extra_rows
    assert all(message.startswith("other file") for message in scan.recent)


def test_scan_restarts_when_the_saved_offset_is_not_a_line_start(tmp_path):
    """Same opening bytes, different later lines: the offset lands inside a line."""
    path = tmp_path / "session.jsonl"
    head = b"".join(_line(_user(f"shared opening {i}")) + b"\n" for i in range(120))
    assert len(head) > hooks_cli._SCAN_FINGERPRINT_BYTES
    path.write_bytes(head + _line(_user("short")) + b"\n")
    _assert_equals_full_parse(path)

    path.write_bytes(head + _line(_user("a much longer and different closing line")) + b"\n")
    scan = _assert_equals_full_parse(path, where="after a same-head replacement")
    assert scan.recent[-1] == "a much longer and different closing line"


# ── 4. A corrupt, missing or unreadable state file ───────────────────────


@pytest.mark.parametrize(
    "damage",
    [
        b"",
        b"\x00\xff not json",
        b"[1, 2, 3]",
        b'{"offset": 10}',
        b'{"v": 1, "offset": "ten", "count": 1, "recent": [], "fp": "", "fp_len": 0}',
        b'{"v": 1, "offset": -5, "count": 1, "recent": [], "fp": "", "fp_len": 0}',
        b'{"v": 1, "offset": 10, "count": 999, "recent": [7], "fp": "", "fp_len": 0}',
        b'{"v": 1, "offset": 10, "count": 999, "recent": [], "fp": "", "fp_len": 999999999}',
        b'{"v": 999, "offset": 10, "count": 999, "recent": [], "fp": "", "fp_len": 0}',
        None,  # the state file is deleted
    ],
)
def test_scan_survives_a_corrupt_or_missing_state_file(tmp_path, damage):
    path = tmp_path / "session.jsonl"
    first, other = _two_different_transcripts()
    path.write_bytes(first)
    _assert_equals_full_parse(path)

    state_file = hooks_cli._scan_state_path(path.resolve())
    if damage is None:
        state_file.unlink()
    else:
        state_file.write_bytes(damage)
    with open(path, "ab") as f:
        f.write(other)

    scan = _assert_equals_full_parse(path, where="with a damaged state file")
    assert scan.count == 80
    # The scan repaired its own state, so the next Stop is incremental again.
    assert _state(path)["offset"] == len(first) + len(other)
    assert _state(path)["count"] == 80


def test_scan_falls_back_to_the_full_parse_when_the_state_is_unreadable(tmp_path):
    path = tmp_path / "session.jsonl"
    first, _ = _two_different_transcripts()
    path.write_bytes(first)
    # A directory where the state file belongs: it can be neither read nor replaced.
    hooks_cli._scan_state_path(path.resolve()).mkdir()

    scan = hooks_cli._scan_transcript(str(path))

    assert scan.complete
    assert (scan.count, scan.recent) == _reference(path)
    assert "falling back to a full parse" in _hook_log()


def test_scan_falls_back_to_the_full_parse_on_any_incremental_error(tmp_path):
    path = tmp_path / "session.jsonl"
    first, _ = _two_different_transcripts()
    path.write_bytes(first)

    with patch.object(hooks_cli, "_scan_transcript_incremental", side_effect=OSError("disk")):
        scan = hooks_cli._scan_transcript(str(path))

    assert scan.complete
    assert (scan.count, scan.recent) == _reference(path)


# ── 5. The time box ──────────────────────────────────────────────────────


def test_time_boxed_scan_saves_progress_and_converges(tmp_path, monkeypatch):
    path = tmp_path / "session.jsonl"
    blob = _transcript_bytes(random.Random(7), ends_with_newline=False)
    path.write_bytes(blob)
    lines = blob.count(b"\n")
    monkeypatch.setattr(hooks_cli, "_SCAN_TIME_BOX_S", 0.0)

    scan = _scan(path)
    assert not scan.complete, "a box of zero seconds cannot finish this file"
    offsets = []
    calls = 1
    while not scan.complete:
        offsets.append(_state(path)["offset"])  # an unfinished scan saved its progress
        assert calls <= lines + 2, "repeated Stops never finished the scan"
        scan = _scan(path)
        calls += 1

    assert len(offsets) > 3
    assert offsets[0] > 0
    assert offsets == sorted(set(offsets)), "every unfinished Stop must move forward"
    assert (scan.count, scan.recent) == _reference(path)


def test_stop_with_an_unfinished_scan_saves_nothing_and_keeps_the_marker(tmp_path, monkeypatch):
    path = tmp_path / "session.jsonl"
    rows = [_line(_user(f"question {i} about the parser")) + b"\n" for i in range(40)]
    path.write_bytes(b"".join(rows))
    assert len(rows) >= hooks_cli.SAVE_INTERVAL
    monkeypatch.setattr(hooks_cli, "_SCAN_TIME_BOX_S", 0.0)
    marker = hooks_cli.STATE_DIR / "sess_last_save"
    save = MagicMock(return_value={"count": 30, "themes": []})
    monkeypatch.setattr(hooks_cli, "_save_diary_direct", save)

    assert _run_stop(path) == {}
    save.assert_not_called()
    hooks_cli._ingest_transcript.assert_not_called()
    assert not marker.exists()
    assert _hook_log().count("transcript scan unfinished") == 1
    assert "TRIGGERING SAVE" not in _hook_log()

    stops = 1
    while not save.called and stops < len(rows) + 2:
        output = _run_stop(path)
        stops += 1
        if not save.called:
            assert output == {}
            assert not marker.exists()

    assert save.call_count == 1, f"no save after {stops} Stops"
    assert stops > 2, "the box of zero seconds should have needed several Stops"
    count, recent = _reference(path)
    assert save.call_args.kwargs["recent_messages"] == recent
    assert marker.read_text(encoding="utf-8") == str(count) == "40"
    assert f"TRIGGERING SAVE at exchange {count}" in _hook_log()


# ── 6. A save Stop parses the transcript once, and files the same text ───


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 2, 3, 4, 5)


def test_save_stop_reads_the_transcript_once_and_files_the_same_entry(tmp_path, monkeypatch):
    path = tmp_path / "session.jsonl"
    size = _big_transcript(path, user_rows=40)
    assert size > 2_000_000
    monkeypatch.setattr(hooks_cli, "datetime", _FixedDatetime)
    # A slow CI box must still finish the scan in one Stop.
    monkeypatch.setattr(hooks_cli, "_SCAN_TIME_BOX_S", 3600.0, raising=False)
    filed = []

    def fake_hub(agent_name, entry, topic, wing, **kwargs):
        filed.append((agent_name, entry, topic, wing))
        return True

    monkeypatch.setattr(hooks_cli, "_forward_diary_to_hub", fake_hub)

    # What the full parse files for this transcript: the entry text before the change.
    before = hooks_cli._save_diary_direct(
        str(path), "sess", wing="wing_proj", toast=False, agent_name="claude"
    )
    assert before["count"] == 30

    with _count_transcript_reads(path) as counter:
        output = _run_stop(path)

    assert output["systemMessage"].startswith("✦ 30 memories woven into the palace")
    assert len(filed) == 2
    assert filed[1] == filed[0], "the Stop hook must file the same entry as the full parse"
    assert "|msgs:30|recent:" in filed[1][1]
    assert counter["bytes"] <= size + _READ_SLACK, (
        f"a save Stop read {counter['bytes']} bytes of a {size} byte transcript: "
        "more than one pass over it"
    )
    assert (hooks_cli.STATE_DIR / "sess_last_save").read_text(encoding="utf-8") == "40"


def test_save_diary_direct_reads_the_transcript_end_itself_when_not_handed_messages(
    tmp_path, monkeypatch
):
    """The precompact and session-end hooks call it with no messages (sysop #373)."""
    path = tmp_path / "session.jsonl"
    path.write_bytes(_line(_user("from the transcript")) + b"\n")
    filed = []
    monkeypatch.setattr(
        hooks_cli,
        "_forward_diary_to_hub",
        lambda agent_name, entry, topic, wing, **kwargs: filed.append((entry, kwargs)) or True,
    )

    with patch.object(
        hooks_cli,
        "_extract_recent_messages_tail",
        wraps=getattr(hooks_cli, "_extract_recent_messages_tail", None),
    ) as parse:
        hooks_cli._save_diary_direct(str(path), "sess", agent_name="claude")
        assert parse.call_count == 1
        hooks_cli._save_diary_direct(
            str(path), "sess", agent_name="claude", recent_messages=["handed over"]
        )
        assert parse.call_count == 1, "handed messages must not trigger a second parse"

    assert filed[0][0].endswith("|msgs:1|recent:from the transcript")
    assert filed[1][0].endswith("|msgs:1|recent:handed over")
    # No hook deadline was given: the hub wait keeps its default.
    assert filed[0][1].get("timeout") is None


# ── 7. The hub wait fits inside the hook ─────────────────────────────────


def _stop_hook_limit_s() -> float:
    config = json.loads((REPO_ROOT / ".claude-plugin" / "hooks" / "hooks.json").read_text())
    return float(config["hooks"]["Stop"][0]["hooks"][0]["timeout"])


class _Clock:
    """Stands in for the ``time`` module inside hooks_cli: a clock the test moves."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class _FakeHub:
    """``urlopen`` for a hub that answers /healthz and needs ``needs_s`` to file a diary.

    Nothing sleeps: a wait moves the fake clock when one is given, and a
    timeout shorter than the hub needs raises what a real socket raises.
    """

    def __init__(self, needs_s, clock=None, health_s=0.0):
        self.needs_s = needs_s
        self.clock = clock
        self.health_s = health_s
        self.timeouts = {}

    def _wait(self, seconds):
        if self.clock is not None:
            self.clock.now += seconds

    def __call__(self, request, timeout=None):
        url = request.full_url
        kind = "health" if url.endswith("/healthz") else "write"
        self.timeouts[kind] = timeout
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.status = 200
        if kind == "health":
            self._wait(self.health_s)
            return response
        if timeout is not None and timeout < self.needs_s:
            self._wait(timeout)
            raise socket.timeout("timed out")
        self._wait(self.needs_s)
        result = {"success": True, "entry_id": "diary_1"}
        payload = {"result": {"content": [{"text": json.dumps(result)}]}}
        response.read.return_value = json.dumps(payload).encode("utf-8")
        return response


@pytest.fixture
def live_hub(monkeypatch):
    """A registered hub, so ``_forward_diary_to_hub`` goes to the (fake) network."""
    from mempalace import server_registry

    monkeypatch.delenv("MEMPALACE_HUB_FORWARD", raising=False)
    monkeypatch.setattr(server_registry, "read_live_serverinfo", lambda palace: {"pid": -1})
    monkeypatch.setattr(server_registry, "client_base_url", lambda info: "http://hub.invalid:1")
    monkeypatch.setattr(server_registry, "load_server_token", lambda palace: "")


def _save_transcript(tmp_path) -> Path:
    path = tmp_path / "session.jsonl"
    path.write_bytes(
        b"".join(_line(_user(f"question {i} about the parser")) + b"\n" for i in range(20))
    )
    return path


def test_stop_hook_hub_wait_is_shorter_than_the_hook_limit(tmp_path, live_hub):
    """A wait longer than the hook's own limit can never end: the hook is killed first."""
    path = _save_transcript(tmp_path)
    hub = _FakeHub(needs_s=12.0)

    with patch("urllib.request.urlopen", hub):
        output = _run_stop(path)

    assert "systemMessage" in output, "a hub that answers in 12 s must still be waited for"
    assert hub.timeouts["write"] < _stop_hook_limit_s()
    assert hub.timeouts["write"] >= 12.0


def test_slow_hub_times_out_inside_the_budget_and_the_marker_still_advances(
    tmp_path, monkeypatch, live_hub
):
    path = _save_transcript(tmp_path)
    clock = _Clock()
    monkeypatch.setattr(hooks_cli, "time", clock)
    started = clock.now
    real_scan = hooks_cli._scan_transcript

    def slow_scan(transcript_path):
        clock.now += 10.0  # a first scan that used its whole time box
        return real_scan(transcript_path)

    monkeypatch.setattr(hooks_cli, "_scan_transcript", slow_scan)
    hub = _FakeHub(needs_s=600.0, clock=clock, health_s=2.0)  # slower than any budget
    forward_calls = []
    real_forward = hooks_cli._forward_diary_to_hub

    def spy_forward(*args, **kwargs):
        forward_calls.append((clock.now, kwargs.get("timeout")))
        status = real_forward(*args, **kwargs)
        forward_calls.append(status)
        return status

    monkeypatch.setattr(hooks_cli, "_forward_diary_to_hub", spy_forward)

    with patch("urllib.request.urlopen", hub):
        output = _run_stop(path)

    budget = hooks_cli._STOP_HOOK_BUDGET_S
    reserve = hooks_cli._STOP_HOOK_RESERVE_S
    assert budget < _stop_hook_limit_s()
    (called_at, timeout), status = forward_calls
    remaining = budget - reserve - (called_at - started)
    assert timeout == pytest.approx(remaining)
    assert timeout == pytest.approx(budget - reserve - 10.0)
    # The health probe's two seconds come out of the same allowance.
    assert hub.timeouts["write"] == pytest.approx(remaining - 2.0)
    assert status is None, "the hub may have filed it: never a second local copy"
    assert "did not complete the diary checkpoint" in _hook_log()

    # The hook got to its marker logic by itself, inside its budget.
    assert clock.now - started <= budget - reserve + 1e-6
    hooks_cli._ingest_transcript.assert_called_once_with(str(path))
    assert (hooks_cli.STATE_DIR / "sess_last_save").read_text(encoding="utf-8") == "20"
    assert output == {}


def test_slow_hub_and_no_ingest_leaves_the_marker_for_a_retry(tmp_path, monkeypatch, live_hub):
    """The #2403 rule is unchanged: nothing landed and nothing dispatched, so retry."""
    path = _save_transcript(tmp_path)
    hooks_cli._ingest_transcript.return_value = False

    with patch("urllib.request.urlopen", _FakeHub(needs_s=600.0)):
        output = _run_stop(path)

    assert output == {}
    assert not (hooks_cli.STATE_DIR / "sess_last_save").exists()


def test_hub_wait_never_drops_below_the_floor(tmp_path, monkeypatch, live_hub):
    """A hook that already used its whole budget still gives the hub a moment."""
    path = _save_transcript(tmp_path)
    clock = _Clock()
    monkeypatch.setattr(hooks_cli, "time", clock)
    real_scan = hooks_cli._scan_transcript

    def very_slow_scan(transcript_path):
        clock.now += 28.0
        return real_scan(transcript_path)

    monkeypatch.setattr(hooks_cli, "_scan_transcript", very_slow_scan)
    hub = _FakeHub(needs_s=600.0, clock=clock)

    with patch("urllib.request.urlopen", hub):
        _run_stop(path)

    floor = hooks_cli._HUB_DIARY_MIN_TIMEOUT_S
    assert 0 < floor <= 2.0
    assert hub.timeouts["write"] == pytest.approx(floor)
    assert hub.timeouts["health"] <= floor


def test_hub_forward_keeps_its_default_wait_for_other_callers(live_hub):
    """Session-end runs in the background and precompact has 90 s: no Stop deadline."""
    hub = _FakeHub(needs_s=12.0)

    with patch("urllib.request.urlopen", hub):
        status = hooks_cli._forward_diary_to_hub("claude", "CHECKPOINT:x", "checkpoint", "w")

    assert status is True
    assert hub.timeouts == {"health": 2.0, "write": 60.0}


# ── 8. SessionEnd and PreCompact read only the end of the transcript (sysop #373) ──
#
# Both hooks call ``_save_diary_direct`` with no messages. It used to parse the
# whole transcript for the last 30 user messages; on a 3 GB transcript that one
# parse took about as long as the 10 s SessionEnd limit. It now reads backwards
# from the end of the file and stops once it has them.


def _fake_hub(monkeypatch):
    filed = []
    monkeypatch.setattr(
        hooks_cli,
        "_forward_diary_to_hub",
        lambda agent_name, entry, topic, wing, **kwargs: filed.append(entry) or True,
    )
    return filed


def _recent_part(entry: str) -> str:
    return entry.split("|recent:", 1)[1]


def _expected_recent(path, monkeypatch) -> str:
    """The recent part of the entry filed from the full parse's messages."""
    filed = _fake_hub(monkeypatch)
    hooks_cli._save_diary_direct(
        str(path),
        "sess",
        agent_name="claude",
        recent_messages=hooks_cli._extract_recent_messages(str(path)),
    )
    return _recent_part(filed[0])


def _run_session_end(path):
    with patch.object(hooks_cli, "MempalaceConfig") as cfg:
        cfg.return_value.hooks_auto_save = True
        cfg.return_value.hook_desktop_toast = False
        with patch.object(hooks_cli, "_output"):
            hooks_cli.hook_session_end(
                {"session_id": "sess", "transcript_path": str(path)}, "claude-code"
            )


def _run_precompact(path):
    with patch.object(hooks_cli, "MempalaceConfig") as cfg:
        cfg.return_value.hooks_auto_save = True
        cfg.return_value.hooks_mine_transcript = False  # the branch that files a diary entry
        cfg.return_value.hook_desktop_toast = False
        with patch.object(hooks_cli, "_mine_sync"), patch.object(hooks_cli, "_output"):
            hooks_cli.hook_precompact(
                {"session_id": "sess", "transcript_path": str(path)}, "claude-code"
            )


@pytest.mark.parametrize("run_hook", [_run_session_end, _run_precompact])
def test_session_end_and_precompact_never_parse_the_whole_transcript(
    tmp_path, monkeypatch, run_hook
):
    path = tmp_path / "session.jsonl"
    _big_transcript(path, user_rows=40)
    expected = _expected_recent(path, monkeypatch)
    filed = _fake_hub(monkeypatch)

    whole = AssertionError("the whole transcript was parsed")
    with patch.object(hooks_cli, "_extract_recent_messages", side_effect=whole) as parse:
        run_hook(path)
    assert parse.call_count == 0
    assert len(filed) == 1, "the hook must still file its diary entry"
    assert "|msgs:30|" in filed[0]
    assert _recent_part(filed[0]) == expected


@pytest.mark.parametrize("chunk", [1, 7, 64, 333, 4096, 1 << 20])
@pytest.mark.parametrize("ends_with_newline", [True, False])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_tail_read_equals_the_full_parse(tmp_path, monkeypatch, seed, ends_with_newline, chunk):
    monkeypatch.setattr(hooks_cli, "_RECENT_TAIL_CHUNK", chunk, raising=False)
    rng = random.Random(seed)
    path = tmp_path / "session.jsonl"
    path.write_bytes(_transcript_bytes(rng, ends_with_newline))
    for count in (1, 5, hooks_cli._RECENT_MSG_COUNT, 10_000):
        assert hooks_cli._extract_recent_messages_tail(
            str(path), count
        ) == hooks_cli._extract_recent_messages(str(path), count), f"count {count}"


def test_tail_read_of_a_missing_empty_or_odd_path_is_empty(tmp_path):
    assert hooks_cli._extract_recent_messages_tail(str(tmp_path / "absent.jsonl")) == []
    (tmp_path / "empty.jsonl").write_bytes(b"")
    assert hooks_cli._extract_recent_messages_tail(str(tmp_path / "empty.jsonl")) == []
    assert hooks_cli._extract_recent_messages_tail(str(tmp_path)) == []
    fifo = tmp_path / "pipe.jsonl"
    os.mkfifo(fifo)
    assert hooks_cli._extract_recent_messages_tail(str(fifo)) == []  # never blocks on open


def test_tail_read_reads_only_the_end_of_a_big_transcript(tmp_path):
    path = tmp_path / "session.jsonl"
    size = _big_transcript(path, user_rows=400, filler_rows=4000)
    with _count_transcript_reads(path) as counter:
        recent = hooks_cli._extract_recent_messages_tail(str(path))
    assert recent == hooks_cli._extract_recent_messages(str(path))
    assert counter["bytes"] < size // 4, f"read {counter['bytes']} of {size} bytes"


def test_tail_read_stops_at_its_byte_cap(tmp_path, monkeypatch):
    path = tmp_path / "session.jsonl"
    old = _line(_user("old question")) + b"\n"
    filler = _line(_assistant("x" * 2000)) + b"\n"
    path.write_bytes(old + filler * 50 + _line(_user("new question")) + b"\n")
    monkeypatch.setattr(hooks_cli, "_RECENT_TAIL_MAX_BYTES", 20_000, raising=False)
    assert hooks_cli._extract_recent_messages_tail(str(path)) == ["new question"]
