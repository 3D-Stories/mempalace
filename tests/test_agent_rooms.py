"""Agent rooms (RFC 006): room semantics over the logstream, and the per-reader
read position they rest on."""

import os

import pytest

from mempalace import agent_rooms as ar
from mempalace.logstream import Logstream


@pytest.fixture
def ls(tmp_dir):
    stream = Logstream(os.path.join(tmp_dir, "logstream.sqlite3"), replica_id="replica-a")
    yield stream
    stream.close()


def _open(ls, agenda="Should BM25 move into the Rust engine?"):
    return ar.open_room(
        ls, project="MemPalace", from_agent="operator", name="Search Brainstorm", agenda=agenda
    )["room"]["room_id"]


def _bodies(result):
    return [e["body"] for e in result["events"]]


class TestNames:
    def test_room_slug_is_kebab_case(self):
        assert ar.room_slug("Search Brainstorm!") == "search-brainstorm"
        assert ar.room_slug("  --x--  ") == "x"
        assert ar.room_slug("!!!") == "room"

    def test_room_slug_rejects_empty(self):
        with pytest.raises(ValueError):
            ar.room_slug("  ")

    def test_room_slug_is_a_valid_palace_room(self):
        from mempalace.config import sanitize_name

        for name in ("Search Brainstorm", "x", "a" * 80, "über ideas"):
            sanitize_name(ar.room_slug(name), "room")


class TestOpen:
    def test_open_returns_room_and_handoff(self, ls):
        result = ar.open_room(
            ls, project="MemPalace", from_agent="operator", name="Search Brainstorm", agenda="Q?"
        )
        room = result["room"]
        assert room["room_id"].startswith("room_search-brainstorm_")
        assert room["name"] == "search-brainstorm"
        assert room["project"] == room["wing"] == "mempalace"
        assert room["stream"] == "project/mempalace"
        assert room["agenda"] == "Q?"
        assert room["closed"] is False
        assert room["room_id"] in result["handoff"]
        assert "mempalace_room_read" in result["handoff"]

    def test_each_opening_is_a_new_room(self, ls):
        assert _open(ls) != _open(ls)

    def test_room_events_are_broadcast_on_the_rooms_channel(self, ls):
        room_id = _open(ls)
        ar.say_in_room(ls, room_id=room_id, from_agent="a", body="one")
        events = ls.list_events(correlation_id=room_id, order="asc")
        assert {e["room"] for e in events} == {ar.ROOMS_LOGSTREAM_ROOM}
        assert {e["to_agent"] for e in events} == {"*"}
        assert {e["status"] for e in events} == {None}

    def test_unknown_room_is_rejected(self, ls):
        with pytest.raises(ValueError, match="not found"):
            ar.read_room(ls, room_id="room_nope_0000", agent="a")
        with pytest.raises(ValueError, match="not a room id"):
            ar.read_room(ls, room_id="task_x", agent="a")


class TestRead:
    def test_first_read_returns_everything_then_only_what_is_new(self, ls):
        room_id = _open(ls)
        ar.say_in_room(ls, room_id=room_id, from_agent="a", body="from a")
        assert _bodies(ar.read_room(ls, room_id=room_id, agent="a")) == [
            "Should BM25 move into the Rust engine?",
            "from a",
        ]
        assert _bodies(ar.read_room(ls, room_id=room_id, agent="a")) == []
        ar.say_in_room(ls, room_id=room_id, from_agent="b", body="from b")
        assert _bodies(ar.read_room(ls, room_id=room_id, agent="a")) == ["from b"]

    def test_later_reads_leave_out_your_own_messages(self, ls):
        room_id = _open(ls)
        ar.read_room(ls, room_id=room_id, agent="a")
        ar.say_in_room(ls, room_id=room_id, from_agent="a", body="mine")
        ar.say_in_room(ls, room_id=room_id, from_agent="b", body="theirs")
        assert _bodies(ar.read_room(ls, room_id=room_id, agent="a")) == ["theirs"]

    def test_message_landing_between_read_and_say_is_not_skipped(self, ls):
        """The hazard of a client-carried cursor: jump it to your own write and
        whatever landed in between is gone. The hub never moves a position
        past an event it did not return."""
        room_id = _open(ls)
        ar.read_room(ls, room_id=room_id, agent="a")
        ar.say_in_room(ls, room_id=room_id, from_agent="b", body="in between")
        said = ar.say_in_room(ls, room_id=room_id, from_agent="a", body="my turn")
        assert said["unread"] == 1
        assert _bodies(ar.read_room(ls, room_id=room_id, agent="a")) == ["in between"]

    def test_full_page_reports_more_and_the_next_read_continues(self, ls):
        room_id = _open(ls, agenda="")
        for i in range(5):
            ar.say_in_room(ls, room_id=room_id, from_agent="b", body=f"m{i}")
        first = ar.read_room(ls, room_id=room_id, agent="a", limit=3)
        assert first["more"] is True
        second = ar.read_room(ls, room_id=room_id, agent="a", limit=3)
        assert second["more"] is False
        assert _bodies(first) + _bodies(second) == ["", "m0", "m1", "m2", "m3", "m4"]

    def test_positions_are_per_agent_and_per_room(self, ls):
        one, two = _open(ls), _open(ls)
        ar.read_room(ls, room_id=one, agent="a")
        assert len(ar.read_room(ls, room_id=one, agent="b")["events"]) == 1
        assert len(ar.read_room(ls, room_id=two, agent="a")["events"]) == 1

    def test_position_survives_a_reopened_logstream(self, ls, tmp_dir):
        room_id = _open(ls)
        ar.read_room(ls, room_id=room_id, agent="a")
        ar.say_in_room(ls, room_id=room_id, from_agent="b", body="later")
        ls.close()
        again = Logstream(os.path.join(tmp_dir, "logstream.sqlite3"), replica_id="replica-a")
        try:
            assert _bodies(ar.read_room(again, room_id=room_id, agent="a")) == ["later"]
        finally:
            again.close()

    def test_replicated_event_arriving_late_is_still_delivered(self, ls):
        """Positions are local rowids, and a remote event gets a fresh local
        rowid on arrival — so an event authored earlier on another replica
        but applied here after a read is still ahead of the reader's position."""
        room_id = _open(ls)
        ar.read_room(ls, room_id=room_id, agent="a")
        ls.apply_remote_event(
            {
                "id": "evt_20260101T000000_remote000001",
                "type": "room.message",
                "stream": "project/mempalace",
                "room": ar.ROOMS_LOGSTREAM_ROOM,
                "from_agent": "b",
                "to_agent": "*",
                "correlation_id": room_id,
                "body": "authored long ago on another replica",
                "created_at": "2026-01-01T00:00:00Z",
                "origin_replica": "replica-b",
                "origin_seq": 1,
                "hlc": "0000000000001-000000-replica-b",
            }
        )
        assert _bodies(ar.read_room(ls, room_id=room_id, agent="a")) == [
            "authored long ago on another replica"
        ]

    def test_positions_are_not_replicated(self, ls):
        """A position is a local rowid; it must never travel as an op."""
        room_id = _open(ls)
        ar.read_room(ls, room_id=room_id, agent="a")
        ops = ls.list_ops("replica-a")
        assert all(op["type"].startswith("room.") for op in ops)
        assert len(ops) == 1

    def test_read_correlation_rejects_bad_limit(self, ls):
        room_id = _open(ls)
        for bad in (0, -1, True, "5"):
            with pytest.raises(ValueError):
                ls.read_correlation(room_id, "a", limit=bad)


class TestSay:
    def test_say_can_address_one_participant(self, ls):
        room_id = _open(ls)
        said = ar.say_in_room(
            ls, room_id=room_id, from_agent="a", body="b, check this", to_agent="b"
        )
        assert said["event"]["to_agent"] == "b"

    def test_empty_message_is_rejected(self, ls):
        room_id = _open(ls)
        with pytest.raises(ValueError, match="must not be empty"):
            ar.say_in_room(ls, room_id=room_id, from_agent="a", body="  ")

    def test_closed_room_refuses_messages(self, ls):
        room_id = _open(ls)
        ar.close_room(ls, room_id=room_id, from_agent="operator")
        with pytest.raises(ValueError, match="closed"):
            ar.say_in_room(ls, room_id=room_id, from_agent="a", body="too late")


class TestClose:
    def test_close_appends_once_and_returns_the_transcript(self, ls):
        room_id = _open(ls)
        ar.say_in_room(ls, room_id=room_id, from_agent="a", body="point")
        first = ar.close_room(ls, room_id=room_id, from_agent="operator", outcome="Decided.")
        assert first["already_closed"] is False
        assert first["room"]["closed"] is True
        assert [e["type"] for e in first["transcript"]] == [
            "room.open",
            "room.message",
            "room.close",
        ]

        again = ar.close_room(ls, room_id=room_id, from_agent="operator", outcome="ignored")
        assert again["already_closed"] is True
        assert again["event"]["id"] == first["event"]["id"]
        assert [e["id"] for e in again["transcript"]] == [e["id"] for e in first["transcript"]]
        closes = ls.list_events(correlation_id=room_id, type=ar.ROOM_CLOSE)
        assert len(closes) == 1

    def test_transcript_stops_at_the_close(self, ls):
        room_id = _open(ls)
        close = ar.close_room(ls, room_id=room_id, from_agent="operator")
        # An event appended under the room's correlation after the close (by
        # a raw event_append) is not part of what was closed.
        ls.append_event(
            type="room.message",
            stream="project/mempalace",
            room=ar.ROOMS_LOGSTREAM_ROOM,
            from_agent="late",
            correlation_id=room_id,
            body="after",
        )
        transcript = ar.room_transcript(ls, room_id, close["transcript"][-1]["seq"])
        assert "after" not in [e["body"] for e in transcript]

    def test_transcript_pages_past_one_list_page(self, ls, monkeypatch):
        monkeypatch.setattr(ar, "_TRANSCRIPT_PAGE", 2)
        room_id = _open(ls)
        for i in range(4):
            ar.say_in_room(ls, room_id=room_id, from_agent="a", body=f"m{i}")
        closed = ar.close_room(ls, room_id=room_id, from_agent="operator", outcome="done")
        assert len(closed["transcript"]) == 6


class TestTranscriptDrawers:
    def test_one_drawer_per_body_bearing_turn_with_a_locator(self, ls):
        room_id = _open(ls, agenda="")
        ar.say_in_room(ls, room_id=room_id, from_agent="a", body="same words")
        ar.say_in_room(ls, room_id=room_id, from_agent="b", body="same words", to_agent="a")
        closed = ar.close_room(ls, room_id=room_id, from_agent="operator", outcome="Outcome.")
        drawers = ar.transcript_drawers(closed["transcript"])
        # Empty agenda is skipped; identical bodies stay two distinct drawers.
        assert len(drawers) == 3
        assert len(set(drawers)) == 3
        first, second, last = drawers
        assert first.startswith("[room.message evt_") and " from=a at=" in first
        assert " from=b to=a at=" in second
        assert first.endswith("\nsame words") and second.endswith("\nsame words")
        assert last.startswith("[room.close ") and last.endswith("\nOutcome.")

    def test_body_is_kept_verbatim(self, ls):
        body = "  line one\n\n\tline two  \n"
        room_id = _open(ls)
        ar.say_in_room(ls, room_id=room_id, from_agent="a", body=body)
        closed = ar.close_room(ls, room_id=room_id, from_agent="operator")
        drawers = ar.transcript_drawers(closed["transcript"])
        assert any(d.split("\n", 1)[1] == body for d in drawers)
