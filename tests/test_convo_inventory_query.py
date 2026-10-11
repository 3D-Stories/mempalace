"""The bookmark inventory check asks the palace one selective question (local patch).

``_inventory_matches`` decides whether a bookmark save may resume: the palace must
hold exactly the drawers the bookmark expects for one log. Asked as a four-way
``$and`` over ``source_file``, ``wing``, ``ingest_mode`` and ``extract_mode``, the
three near-universal fields made chromadb build huge candidate sets: 7.9 s on a
481k-drawer palace, against 0.19 s for ``source_file`` alone with the same 577 ids
(measured 2026-10-11). That check runs inside the hub's exclusive write lock on
every bookmark save, so the slow form held every other MCP call behind it.

The contract under test: the query filters on ``source_file`` only, and the other
three fields are still enforced, in Python, with the same answer as before.
"""

import pytest

from mempalace import convo_miner
from mempalace.convo_miner import _inventory_matches, make_convo_drawer_id

SRC = "/logs/session.jsonl"
WING = "sessions"
ROOM = "general"
MODE = "exchange"


def _bm(total):
    return {"room": ROOM, "total": total}


def _row(i, **overrides):
    meta = {
        "source_file": SRC,
        "wing": WING,
        "ingest_mode": "convos",
        "extract_mode": MODE,
    }
    meta.update(overrides)
    drawer_id = make_convo_drawer_id(meta["wing"], ROOM, SRC, meta["extract_mode"], i)
    return drawer_id, meta


def _matches(meta, where):
    if "$and" in where:
        return all(_matches(meta, clause) for clause in where["$and"])
    return all(meta.get(k) == v for k, v in where.items())


class FakeCollection:
    """Answers ``get(where=...)`` with string equality for the filters used here, and records
    each call. Malformed backend answers are exercised separately, through ``RawCollection``."""

    def __init__(self, rows, error=None):
        self.rows = rows
        self.error = error
        self.calls = []

    def get(self, *, where=None, include=None, **_):
        self.calls.append({"where": where, "include": include})
        if self.error is not None:
            raise self.error
        hits = [(i, m) for i, m in self.rows if m is not None and _matches(m, where)]
        out = {"ids": [i for i, _ in hits]}
        if include and "metadatas" in include:
            out["metadatas"] = [m for _, m in hits]
        return out


def _check(col, total=3):
    return _inventory_matches(col, _bm(total), SRC, WING, MODE)


def test_inventory_query_filters_on_source_file_only():
    col = FakeCollection([_row(i) for i in range(3)])
    assert _check(col) is True
    assert len(col.calls) == 1
    assert col.calls[0]["where"] == {"source_file": SRC}
    assert col.calls[0]["include"] == ["metadatas"]


@pytest.mark.parametrize(
    "other",
    [{"wing": "elsewhere"}, {"ingest_mode": "projects"}, {"extract_mode": "general"}],
)
def test_rows_from_another_wing_or_mode_do_not_count(other):
    rows = [_row(i) for i in range(3)] + [_row(7, **other)]
    assert _check(FakeCollection(rows)) is True


def test_a_missing_drawer_is_not_a_match():
    assert _check(FakeCollection([_row(0), _row(2)])) is False


def test_an_extra_matching_drawer_is_not_a_match():
    assert _check(FakeCollection([_row(i) for i in range(4)])) is False


def test_a_query_error_is_not_a_match():
    assert _check(FakeCollection([], error=RuntimeError("palace closed"))) is False


class RawCollection:
    """Returns one fixed ``get`` answer, so malformed backend results can be fed in."""

    def __init__(self, ids, metadatas):
        self.answer = {"ids": ids, "metadatas": metadatas}

    def get(self, **_):
        return self.answer


def _expected():
    ids, metas = zip(*[_row(i) for i in range(3)])
    return list(ids), list(metas)


def test_more_ids_than_labels_is_not_a_match():
    ids, metas = _expected()
    assert _check(RawCollection(ids + [_row(3)[0]], metas)) is False


def test_padded_empty_labels_are_not_a_match():
    # The chroma adapter pads a short metadata list with {} (backends/chroma.py).
    ids, metas = _expected()
    assert _check(RawCollection(ids + [_row(3)[0]], metas + [{}])) is False


def test_missing_labels_are_not_a_match():
    ids, metas = _expected()
    assert _check(RawCollection(ids + [_row(3)[0]], metas + [None])) is False


def test_unreadable_labels_are_not_a_match_and_do_not_raise():
    ids, metas = _expected()
    assert _check(RawCollection(ids + [_row(3)[0]], metas + ["bad"])) is False


def test_labels_naming_another_log_are_not_a_match():
    ids, metas = _expected()
    stray_id, stray = _row(3)
    stray = dict(stray, source_file="/logs/other.jsonl")
    assert _check(RawCollection(ids + [stray_id], metas + [stray])) is False


def test_a_labelled_row_with_no_wing_is_ignored_as_before():
    # Equality semantics: a row missing ``wing`` never matched the old $and either.
    ids, metas = _expected()
    assert _check(RawCollection(ids + ["other-id"], metas + [{"source_file": SRC}])) is True


def test_an_unreadable_answer_is_not_a_match_and_does_not_raise():
    col = RawCollection([], [])
    col.answer = None
    assert _check(col) is False


def test_a_large_bookmark_asks_the_four_way_question(monkeypatch):
    monkeypatch.setattr(convo_miner, "_INVENTORY_LABEL_FETCH_MAX", 2)
    col = FakeCollection([_row(i) for i in range(3)])
    assert _check(col) is True
    assert col.calls[0]["where"] == {
        "$and": [
            {"source_file": SRC},
            {"wing": WING},
            {"ingest_mode": "convos"},
            {"extract_mode": MODE},
        ]
    }
    assert col.calls[0]["include"] == []


def test_a_bookmark_at_the_limit_still_fetches_labels(monkeypatch):
    monkeypatch.setattr(convo_miner, "_INVENTORY_LABEL_FETCH_MAX", 3)
    col = FakeCollection([_row(i) for i in range(3)])
    assert _check(col) is True
    assert col.calls[0]["where"] == {"source_file": SRC}


def test_a_large_bookmark_with_an_extra_drawer_is_not_a_match(monkeypatch):
    monkeypatch.setattr(convo_miner, "_INVENTORY_LABEL_FETCH_MAX", 2)
    assert _check(FakeCollection([_row(i) for i in range(4)])) is False


def test_a_large_bookmark_query_error_is_not_a_match(monkeypatch):
    monkeypatch.setattr(convo_miner, "_INVENTORY_LABEL_FETCH_MAX", 2)
    assert _check(FakeCollection([], error=RuntimeError("palace closed"))) is False


def test_the_label_fetch_limit_matches_the_measurement():
    # Crossover measured near 16k drawers on 2026-10-11; 10k keeps a margin.
    assert convo_miner._INVENTORY_LABEL_FETCH_MAX == 10_000
