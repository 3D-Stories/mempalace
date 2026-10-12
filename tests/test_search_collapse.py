"""Search collapse (3D-Stories fork): each text appears once in search results.

The best-ranked hit of an identical text stays and gains ``also_in`` (the
other chats holding that text); later copies drop out, so the top-N slice
backfills from the rest of the candidate pool. ``MEMPALACE_SEARCH_COLLAPSE=0``
restores upstream behaviour.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

from mempalace.backends import LexicalResult
from mempalace.layers import Layer3
from mempalace.palace import get_collection
from mempalace.searcher import (
    _candidate_pool_limits,
    _collapse_identical_hits,
    _print_search_results_bm25_only,
    _search_collapse_enabled,
    search,
    search_memories,
)

REPEATED = "quasar token rotation runbook: rotate the quasar signing key, then restart quasar"
DISTINCT = [
    "quasar token rotation postmortem: the relay cached the old key for an hour",
    "quasar token rotation checklist for the staging cluster and its replicas",
    "quasar token rotation cadence: every ninety days, with a seven day overlap",
]


def _hit(drawer_id, text, source):
    return {
        "drawer_id": drawer_id,
        "text": text,
        "source_file": source.rsplit("/", 1)[-1],
        "source_path": source,
        "matched_via": "drawer",
    }


@pytest.fixture(autouse=True)
def _collapse_on(monkeypatch):
    monkeypatch.delenv("MEMPALACE_SEARCH_COLLAPSE", raising=False)


def test_collapse_keeps_first_ranked_copy_and_notes_other_chats():
    hits = [
        _hit("a", "same text", "/chats/a.jsonl"),
        _hit("u", "unique text", "/chats/u.jsonl"),
        _hit("b", "same text", "/chats/b.jsonl"),
        _hit("c", "same text", "/chats/c.jsonl"),
    ]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["a", "u"]
    assert result[0]["also_in"] == {
        "count": 2,
        "sources": ["/chats/b.jsonl", "/chats/c.jsonl"],
    }
    assert "also_in" not in result[1]


def test_collapse_drops_same_chat_repeat_without_a_note():
    hits = [
        _hit("a0", "same text", "/chats/a.jsonl"),
        _hit("a1", "same text", "/chats/a.jsonl"),
        _hit("u", "unique text", "/chats/u.jsonl"),
    ]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["a0", "u"]
    assert "also_in" not in result[0]


def test_collapse_counts_every_other_chat_but_lists_three_once_each():
    hits = [
        _hit("a", "same text", "/chats/a.jsonl"),
        _hit("b0", "same text", "/chats/b.jsonl"),
        _hit("b1", "same text", "/chats/b.jsonl"),
        _hit("c", "same text", "/chats/c.jsonl"),
        _hit("a1", "same text", "/chats/a.jsonl"),
        _hit("d", "same text", "/chats/d.jsonl"),
        _hit("e", "same text", "/chats/e.jsonl"),
    ]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["a"]
    assert result[0]["also_in"] == {
        "count": 4,
        "sources": ["/chats/b.jsonl", "/chats/c.jsonl", "/chats/d.jsonl"],
    }


def test_collapse_passes_non_string_and_blank_text_through():
    hits = [
        {"drawer_id": "n0", "text": None, "source_path": "/chats/a.jsonl"},
        {"drawer_id": "n1", "text": None, "source_path": "/chats/b.jsonl"},
        {"drawer_id": "m", "source_path": "/chats/c.jsonl"},
        _hit("w0", "   ", "/chats/d.jsonl"),
        _hit("w1", "   ", "/chats/e.jsonl"),
        _hit("e0", "", "/chats/f.jsonl"),
        _hit("e1", "", "/chats/g.jsonl"),
    ]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["n0", "n1", "m", "w0", "w1", "e0", "e1"]
    assert all("also_in" not in h for h in result)


def test_collapse_does_not_mutate_dropped_copies_or_reorder():
    hits = [
        _hit("u1", "first unique", "/chats/u1.jsonl"),
        _hit("a", "same text", "/chats/a.jsonl"),
        _hit("u2", "second unique", "/chats/u2.jsonl"),
        _hit("b", "same text", "/chats/b.jsonl"),
        _hit("c", "same text", "/chats/c.jsonl"),
    ]
    dropped_before = copy.deepcopy(hits[3:])
    kept = hits[:3]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["u1", "a", "u2"]
    assert all(r is k for r, k in zip(result, kept))
    assert hits[3:] == dropped_before


def test_unknown_sources_are_never_counted_as_chats():
    hits = [
        _hit("a", "same text", "/chats/a.jsonl"),
        {"drawer_id": "q", "text": "same text", "source_path": "", "source_file": "?"},
        {"drawer_id": "n", "text": "same text", "source_file": "?"},
        _hit("u", "other text", "/chats/u.jsonl"),
        {"drawer_id": "w", "text": "other text", "source_path": "", "source_file": "?"},
    ]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["a", "u"]
    assert all("also_in" not in h for h in result)


def test_an_unknown_winner_counts_no_other_chats():
    hits = [
        {"drawer_id": "q", "text": "same text", "source_path": "", "source_file": "?"},
        _hit("a", "same text", "/chats/a.jsonl"),
        _hit("b", "same text", "/chats/b.jsonl"),
    ]

    result = _collapse_identical_hits(hits)

    assert [h["drawer_id"] for h in result] == ["q"]
    assert "also_in" not in result[0]


def test_nested_metadata_source_is_the_identity():
    hits = [
        {"text": "same text", "metadata": {"source_file": "/chats/a.jsonl"}},
        {"text": "same text", "metadata": {"source_file": "/chats/b.jsonl"}},
    ]

    result = _collapse_identical_hits(hits)

    assert len(result) == 1
    assert result[0]["also_in"] == {"count": 1, "sources": ["/chats/b.jsonl"]}


@pytest.mark.parametrize("value", ["0", "false", "FALSE", " no ", "off", "Off"])
def test_kill_switch_spellings_disable_collapse(monkeypatch, value):
    monkeypatch.setenv("MEMPALACE_SEARCH_COLLAPSE", value)
    hits = [
        _hit("a", "same text", "/chats/a.jsonl"),
        _hit("b", "same text", "/chats/b.jsonl"),
    ]

    assert _search_collapse_enabled() is False
    result = _collapse_identical_hits(hits)
    assert [h["drawer_id"] for h in result] == ["a", "b"]
    assert all("also_in" not in h for h in result)


@pytest.mark.parametrize("value", ["1", "", "true", "yes", "on", "anything"])
def test_other_switch_values_keep_collapse_on(monkeypatch, value):
    monkeypatch.setenv("MEMPALACE_SEARCH_COLLAPSE", value)

    assert _search_collapse_enabled() is True


def test_switch_unset_keeps_collapse_on():
    assert _search_collapse_enabled() is True


def _seed(palace_path):
    sources = [f"/chats/copy-{i}.jsonl" for i in range(4)]
    ids = [f"copy_{i}" for i in range(4)] + [f"distinct_{i}" for i in range(3)]
    docs = [REPEATED] * 4 + DISTINCT
    metas = [
        {"wing": "sessions", "room": "technical", "source_file": s, "chunk_index": 0}
        for s in sources
    ] + [
        {
            "wing": "sessions",
            "room": "technical",
            "source_file": f"/chats/distinct-{i}.jsonl",
            "chunk_index": 0,
        }
        for i in range(3)
    ]
    get_collection(palace_path).upsert(ids=ids, documents=docs, metadatas=metas)
    return sources


def _assert_collapsed(result, sources):
    assert "error" not in result, result
    hits = result["results"]
    texts = [h["text"] for h in hits]
    assert len(hits) == 3
    assert len(set(texts)) == 3
    kept = [h for h in hits if h["text"] == REPEATED]
    assert len(kept) == 1
    note = kept[0]["also_in"]
    assert note["count"] == 3
    assert len(note["sources"]) == 3
    assert set(note["sources"]) == set(sources) - {kept[0]["source_path"]}


def test_vector_search_returns_each_text_once_and_backfills(palace_path):
    sources = _seed(palace_path)

    result = search_memories("quasar token rotation", palace_path, n_results=3)

    _assert_collapsed(result, sources)


def test_vector_disabled_fallback_returns_each_text_once(palace_path):
    sources = _seed(palace_path)

    result = search_memories(
        "quasar",
        palace_path,
        n_results=3,
        vector_disabled=True,
        collection_name="mempalace_drawers",
    )

    _assert_collapsed(result, sources)


@pytest.mark.parametrize("vector_disabled", [False, True])
def test_kill_switch_restores_repeats_end_to_end(palace_path, monkeypatch, vector_disabled):
    _seed(palace_path)
    monkeypatch.setenv("MEMPALACE_SEARCH_COLLAPSE", "0")

    result = search_memories(
        "quasar",
        palace_path,
        n_results=7,
        vector_disabled=vector_disabled,
        collection_name="mempalace_drawers",
    )

    assert "error" not in result, result
    texts = [h["text"] for h in result["results"]]
    assert texts.count(REPEATED) == 4
    assert all("also_in" not in h for h in result["results"])


# --- G1: union keeps its whole vector fetch so collapse can backfill ---------

EXTRA = "quasar token rotation dashboard: who owns the alert and the runbook"
_UNION_SOURCES = [f"/chats/{c}.jsonl" for c in "abcdef"]


def _union_collection():
    col = MagicMock()
    col.distance_metric = "cosine"
    col.query.return_value = {
        "ids": [[f"id{i}" for i in range(6)]],
        "documents": [[REPEATED, REPEATED, *DISTINCT, EXTRA]],
        "metadatas": [
            [
                {
                    "wing": "sessions",
                    "room": "technical",
                    "source_file": s,
                    "chunk_index": 0,
                    "filed_at": "2026-10-01T00:00:00",
                }
                for s in _UNION_SOURCES
            ]
        ],
        "distances": [[0.05, 0.06, 0.1, 0.2, 0.3, 0.4]],
    }
    col.lexical_search.return_value = LexicalResult(hits=[])
    return col


@pytest.mark.parametrize("since", [None, "2026-09-01"])
def test_union_backfills_from_its_whole_vector_fetch(since):
    col = _union_collection()
    with (
        patch("mempalace.searcher.get_collection", return_value=col),
        patch("mempalace.searcher.get_closets_collection", side_effect=RuntimeError("no closets")),
    ):
        result = search_memories(
            "quasar token rotation",
            "/fake/palace",
            n_results=2,
            candidate_strategy="union",
            since=since,
        )

    assert "error" not in result, result
    texts = [h["text"] for h in result["results"]]
    assert len(texts) == 2
    assert len(set(texts)) == 2
    assert REPEATED in texts


def test_union_pool_keeps_the_whole_fetch_only_while_collapse_is_on(monkeypatch):
    assert _candidate_pool_limits("union", 5) == (15, 15)
    monkeypatch.setenv("MEMPALACE_SEARCH_COLLAPSE", "0")
    assert _candidate_pool_limits("union", 5) == (15, 5)


# --- G2: the command-line search, its keyword fallback, and Layer3 -----------


def _cli_collection(docs, sources):
    col = MagicMock()
    col.distance_metric = "cosine"
    col.metadata = {"hnsw:space": "cosine"}
    col.query.return_value = {
        "ids": [[f"id{i}" for i in range(len(docs))]],
        "documents": [docs],
        "metadatas": [
            [
                {
                    "wing": "sessions",
                    "room": "technical",
                    "source_file": s,
                    "filed_at": "2026-10-01T00:00:00",
                }
                for s in sources
            ]
        ],
        "distances": [[0.05 * (i + 1) for i in range(len(docs))]],
    }
    return col


def _cli_search(col, **kwargs):
    with (
        patch("mempalace.searcher.resolve_backend_name", return_value="sqlite_exact"),
        patch("mempalace.searcher._open_collection_or_explain", return_value=col),
    ):
        search("quasar token rotation", "/unused", **kwargs)


def test_cli_search_prints_each_text_once_with_a_note(capsys):
    col = _cli_collection([REPEATED, REPEATED], ["/chats/a.jsonl", "/chats/b.jsonl"])

    _cli_search(col, n_results=2)

    out = capsys.readouterr().out
    assert out.count(REPEATED) == 1
    assert "Also in: 1 other chat (b.jsonl)" in out


def test_cli_date_window_search_backfills_from_its_wider_fetch(capsys):
    col = _cli_collection(
        [REPEATED, REPEATED, *DISTINCT],
        ["/chats/a.jsonl", "/chats/b.jsonl", "/chats/c.jsonl", "/chats/d.jsonl", "/chats/e.jsonl"],
    )

    _cli_search(col, n_results=2, since="2026-09-01")

    out = capsys.readouterr().out
    assert out.count(REPEATED) == 1
    assert out.count("  [") == 2
    assert DISTINCT[0] in out


def test_cli_search_kill_switch_prints_copies(capsys, monkeypatch):
    monkeypatch.setenv("MEMPALACE_SEARCH_COLLAPSE", "0")
    col = _cli_collection([REPEATED, REPEATED], ["/chats/a.jsonl", "/chats/b.jsonl"])

    _cli_search(col, n_results=2)

    out = capsys.readouterr().out
    assert out.count(REPEATED) == 2
    assert "Also in" not in out


def test_cli_keyword_fallback_prints_the_note(palace_path, capsys):
    _seed(palace_path)

    _print_search_results_bm25_only("quasar", palace_path, None, None, 3)

    out = capsys.readouterr().out
    assert out.count(REPEATED) == 1
    assert "Also in: 3 other chats (" in out


def test_layer3_search_raw_returns_each_text_once(palace_path):
    _seed(palace_path)

    hits = Layer3(palace_path).search_raw("quasar token rotation", n_results=7)

    texts = [h["text"] for h in hits]
    assert len(texts) == 4
    assert len(set(texts)) == 4
    kept = next(h for h in hits if h["text"] == REPEATED)
    assert kept["also_in"]["count"] == 3


def test_layer3_search_prints_each_text_once(palace_path):
    _seed(palace_path)

    out = Layer3(palace_path).search("quasar token rotation", n_results=7)

    assert out.count(REPEATED) == 1
    assert "Also in: 3 other chats (" in out


def test_layer3_kill_switch_returns_copies(palace_path, monkeypatch):
    _seed(palace_path)
    monkeypatch.setenv("MEMPALACE_SEARCH_COLLAPSE", "0")

    hits = Layer3(palace_path).search_raw("quasar token rotation", n_results=7)

    assert [h["text"] for h in hits].count(REPEATED) == 4


def _hub_args():
    from types import SimpleNamespace

    return SimpleNamespace(query="quasar", wing=None, room=None, since=None, before=None)


def test_hub_structured_results_print_the_note(capsys):
    from mempalace import cli

    hit = {
        "wing": "w",
        "room": "r",
        "source_file": "a.jsonl",
        "similarity": 0.9,
        "bm25_score": 0.1,
        "text": REPEATED,
        "also_in": {"count": 1, "sources": ["/chats/b.jsonl"]},
    }

    assert cli._print_hub_search_result(_hub_args(), {"results": [hit]})

    out = capsys.readouterr().out
    assert out.count(REPEATED) == 1
    assert "Also in: 1 other chat (b.jsonl)" in out


def test_hub_structured_results_without_also_in_print_no_note(capsys):
    from mempalace import cli

    hit = {"wing": "w", "room": "r", "source_file": "a.jsonl", "similarity": 0.9, "text": "x"}

    assert cli._print_hub_search_result(_hub_args(), {"results": [hit]})

    assert "Also in" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "also_in", [{"count": 1}, {"sources": ["/a"]}, "two", {"count": 1, "sources": [3]}]
)
def test_hub_structured_results_with_a_malformed_note_still_print(capsys, also_in):
    from mempalace import cli

    hit = {"wing": "w", "room": "r", "source_file": "a.jsonl", "text": REPEATED, "also_in": also_in}

    assert cli._print_hub_search_result(_hub_args(), {"results": [hit]})

    out = capsys.readouterr().out
    assert out.count(REPEATED) == 1
    assert "Also in" not in out


@pytest.mark.parametrize(
    "note, expected",
    [
        (
            {"count": 5, "sources": ["/chats/a.jsonl", "/chats/b.jsonl", "/chats/c.jsonl"]},
            "Also in: 5 other chats (a.jsonl, b.jsonl, c.jsonl, ...)",
        ),
        (
            {"count": 2, "sources": ["/chats/a.jsonl", "/chats/b.jsonl"]},
            "Also in: 2 other chats (a.jsonl, b.jsonl)",
        ),
    ],
    ids=["more-than-listed", "all-listed"],
)
def test_the_note_marks_chats_it_does_not_list(note, expected):
    from mempalace.also_in import also_in_note

    assert also_in_note({"also_in": note}) == expected
