"""Save-time skip rule (3D-Stories fork, sysop#297): which pieces of a chat are not stored."""

from __future__ import annotations

import hashlib
import json

import pytest

from mempalace.save_dedupe import (
    BOILERPLATE_MARKERS,
    is_boilerplate,
    load_boilerplate_hashes,
    plan_skips,
    save_dedupe_enabled,
    text_sha,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _dedupe_on(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMPALACE_SAVE_DEDUPE", raising=False)
    monkeypatch.setenv("MEMPALACE_CONFIG_DIR", str(tmp_path / "cfg"))


def test_text_sha_is_sha256_of_utf8():
    assert text_sha("é piece") == _sha("é piece")


@pytest.mark.parametrize("marker", BOILERPLATE_MARKERS)
@pytest.mark.parametrize("prefix", ["", "> ", "  > ", "\n"])
def test_markers_at_the_start_are_boilerplate(marker, prefix):
    assert is_boilerplate(prefix + marker + " rest of the block", frozenset())


def test_a_marker_inside_the_text_is_not_boilerplate():
    assert not is_boilerplate("> please explain <local-command-caveat> to me", frozenset())


def test_the_quote_prefix_is_stripped_only_once():
    assert not is_boilerplate("> > <command-message>x</command-message>", frozenset())


def test_a_listed_fingerprint_is_boilerplate():
    text = "> Carry authorized work through to the requested result."
    assert is_boilerplate(text, frozenset({_sha(text)}))
    assert not is_boilerplate(text, frozenset())


def test_plan_skips_boilerplate_and_later_repeats_keeping_the_first():
    pieces = [
        "> <local-command-caveat>caveat</local-command-caveat>",
        "> real question one",
        "> real question two",
        "> real question one",
        "> <command-message>reload</command-message>",
        "> real question two",
        "> real question three",
    ]

    skip, forced = plan_skips(pieces, listed=frozenset())

    assert skip == {0, 3, 4, 5}
    assert forced is False


def test_plan_skips_against_prior_fingerprints():
    pieces = ["> stored earlier", "> new text", "> new text"]

    skip, forced = plan_skips(pieces, prior=frozenset({_sha("> stored earlier")}))

    assert skip == {0, 2}
    assert forced is False


def test_a_chat_that_is_all_boilerplate_keeps_its_first_piece():
    pieces = [
        "> <local-command-caveat>a</local-command-caveat>",
        "> <command-message>b</command-message>",
    ]

    skip, forced = plan_skips(pieces)

    assert skip == {1}
    assert forced is True


def test_all_repeats_of_prior_text_are_skipped_without_forcing():
    pieces = ["> stored earlier", "> stored earlier"]

    skip, forced = plan_skips(pieces, prior=frozenset({_sha("> stored earlier")}), keep_one=False)

    assert skip == {0, 1}
    assert forced is False


def test_empty_piece_list():
    assert plan_skips([]) == (set(), False)


@pytest.mark.parametrize("value", ["0", "false", "OFF", " no "])
def test_switch_off_skips_nothing(monkeypatch, value):
    monkeypatch.setenv("MEMPALACE_SAVE_DEDUPE", value)
    pieces = ["> <command-message>x</command-message>", "> a", "> a"]

    assert save_dedupe_enabled() is False
    assert plan_skips(pieces) == (set(), False)


@pytest.mark.parametrize("value", ["1", "", "yes", "anything"])
def test_other_switch_values_keep_it_on(monkeypatch, value):
    monkeypatch.setenv("MEMPALACE_SAVE_DEDUPE", value)
    assert save_dedupe_enabled() is True


def _write_list(tmp_path, payload) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (cfg / "boilerplate.json").write_text(text, encoding="utf-8")


def test_list_missing_is_empty():
    assert load_boilerplate_hashes() == frozenset()


def test_list_valid_is_loaded(tmp_path):
    a, b = _sha("a"), _sha("b")
    _write_list(tmp_path, {"version": 1, "sha256": [a, b]})

    assert load_boilerplate_hashes() == frozenset({a, b})


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        [],
        {"sha256": [_sha("a")]},
        {"version": 2, "sha256": [_sha("a")]},
        {"version": 1},
        {"version": 1, "sha256": "abc"},
        {"version": 1, "sha256": [_sha("a"), "not-hex"]},
        {"version": 1, "sha256": [_sha("a").upper()]},
        {"version": 1, "sha256": [_sha("a")[:63]]},
        {"version": 1, "sha256": [123]},
        {"version": True, "sha256": [_sha("a")]},
    ],
)
def test_any_doubt_about_the_list_means_an_empty_list(tmp_path, payload):
    _write_list(tmp_path, payload)

    assert load_boilerplate_hashes() == frozenset()


def test_plan_skips_reads_the_list_by_default(tmp_path):
    text = "> Carry authorized work through to the requested result."
    _write_list(tmp_path, {"version": 1, "sha256": [_sha(text)]})

    skip, _ = plan_skips(["> real question", text])

    assert skip == {1}


def test_keep_one_off_lets_every_new_piece_be_skipped():
    pieces = ["> <command-message>a</command-message>", "> <command-message>b</command-message>"]

    assert plan_skips(pieces, keep_one=False) == ({0, 1}, False)


def test_keep_one_fires_even_with_prior_fingerprints():
    pieces = ["> stored earlier", "> <command-message>a</command-message>"]

    skip, forced = plan_skips(pieces, prior=frozenset({_sha("> stored earlier")}))

    assert skip == {1}
    assert forced is True
