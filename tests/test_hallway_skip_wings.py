"""Step 4 local fix: ``hallways.skip_wings`` keeps named wings out of the hallway graph.

The hub rebuilt the hallways of the raw-transcript wing ``sessions`` (467k drawers) about every
30 minutes, inside its write lock: a 7-minute stall that rewrote an 835 MB hallways.json whose
1.38 million ``sessions`` pairs nothing ever read. A wing listed in ``hallways.skip_wings``
(config.json) or ``MEMPALACE_HALLWAY_SKIP_WINGS`` (comma-separated, overrides the file) is never
walked, and its old records are dropped from the hallway file once. Unlisted wings behave exactly
as before; the default list is empty.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

with patch.dict("sys.modules", {"chromadb": MagicMock()}):
    from mempalace import hallways as hallways_mod


def _config(tmp_path, skip_wings=None):
    """A real MempalaceConfig over a temporary config.json and palace."""
    from mempalace.config import MempalaceConfig

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    data = {}
    if skip_wings is not None:
        data["hallways"] = {"skip_wings": skip_wings}
    (cfg_dir / "config.json").write_text(json.dumps(data), encoding="utf-8")
    return MempalaceConfig(config_dir=cfg_dir, palace_path=tmp_path / "palace")


def _fake_collection(drawers):
    """Paginated fake collection, same shape as tests/test_hallways.py uses."""
    col = MagicMock()
    col.count.return_value = len(drawers)

    def _get(limit=None, offset=0, include=None, where=None, ids=None, **kwargs):
        page = drawers[offset : offset + limit] if limit is not None else drawers
        return {"ids": [f"d{i}" for i in range(offset, offset + len(page))], "metadatas": page}

    col.get.side_effect = _get
    return col


PAIRED = [
    {"wing": "sessions", "room": "technical", "entities": "Aya;Lumi"},
    {"wing": "sessions", "room": "technical", "entities": "Aya;Lumi"},
]


def _record(wing, a, b):
    return {"id": f"{wing}:{a}:{b}", "wing": wing, "entity_a": a, "entity_b": b, "count": 2}


def _write_hallways(cfg, records):
    hallways_mod._save_hallways(records, config=cfg)
    return cfg.hallway_file


# --- the setting -------------------------------------------------------------------------------


def test_skip_wings_defaults_to_empty(tmp_path, monkeypatch):
    # Fails if the property is missing, or if it defaults to skipping anything.
    monkeypatch.delenv("MEMPALACE_HALLWAY_SKIP_WINGS", raising=False)
    assert _config(tmp_path).hallway_skip_wings == []


def test_skip_wings_read_from_config_file(tmp_path, monkeypatch):
    # Fails if the property does not read hallways.skip_wings from config.json.
    monkeypatch.delenv("MEMPALACE_HALLWAY_SKIP_WINGS", raising=False)
    assert _config(tmp_path, ["sessions", "wing_x"]).hallway_skip_wings == ["sessions", "wing_x"]


@pytest.mark.parametrize(
    "env, expected",
    [("sessions", ["sessions"]), (" sessions , wing_x ,", ["sessions", "wing_x"]), ("", [])],
)
def test_env_overrides_config_file(tmp_path, monkeypatch, env, expected):
    # Fails if the environment variable is ignored, or does not override the file
    # (an empty value must switch skipping off even when the file lists wings).
    monkeypatch.setenv("MEMPALACE_HALLWAY_SKIP_WINGS", env)
    assert _config(tmp_path, ["from_file"]).hallway_skip_wings == expected


@pytest.mark.parametrize("bad", ["sessions", {"sessions": True}, 7, [1, None, "sessions", ""]])
def test_malformed_values_skip_nothing(tmp_path, monkeypatch, bad):
    # Fails if a value that is not a list made only of strings skips any wing: skipping drops
    # records, so a malformed setting must fall back to computing (a bare string must not be
    # read character by character, and a mixed list is not partly accepted).
    monkeypatch.delenv("MEMPALACE_HALLWAY_SKIP_WINGS", raising=False)
    assert _config(tmp_path, bad).hallway_skip_wings == []


def test_string_list_is_trimmed(tmp_path, monkeypatch):
    # Guard: fails if surrounding spaces or empty strings in a valid list change what is skipped.
    monkeypatch.delenv("MEMPALACE_HALLWAY_SKIP_WINGS", raising=False)
    assert _config(tmp_path, [" sessions ", "", "wing_x"]).hallway_skip_wings == [
        "sessions",
        "wing_x",
    ]


# --- compute_hallways_for_wing -------------------------------------------------------------------


def test_skipped_wing_is_never_walked(tmp_path, monkeypatch):
    # Fails if compute_hallways_for_wing fetches a skipped wing's drawers or makes hallways for it.
    monkeypatch.delenv("MEMPALACE_HALLWAY_SKIP_WINGS", raising=False)
    cfg = _config(tmp_path, ["sessions"])
    col = _fake_collection(PAIRED)
    assert hallways_mod.compute_hallways_for_wing("sessions", col=col, config=cfg) == []
    col.get.assert_not_called()
    assert hallways_mod.list_hallways(wing="sessions", config=cfg) == []


def test_skipping_drops_the_wings_old_records_and_keeps_the_rest(tmp_path, monkeypatch):
    # Fails if a skipped wing's stale records stay in the file, or other wings' records are lost.
    monkeypatch.delenv("MEMPALACE_HALLWAY_SKIP_WINGS", raising=False)
    cfg = _config(tmp_path, ["sessions"])
    kept = [_record("wing_a", "Aya", "Lumi"), _record("session_notes", "Bo", "Cy")]
    _write_hallways(
        cfg, [_record("sessions", "!!", "--path"), *kept, _record("sessions", "x", "y")]
    )
    hallways_mod.compute_hallways_for_wing("sessions", col=_fake_collection(PAIRED), config=cfg)
    assert hallways_mod.list_hallways(config=cfg) == kept


def test_skipped_wing_with_no_records_leaves_the_file_untouched(tmp_path, monkeypatch):
    # Fails if every skipped save rewrites the hallway file even when there is nothing to drop.
    monkeypatch.delenv("MEMPALACE_HALLWAY_SKIP_WINGS", raising=False)
    cfg = _config(tmp_path, ["sessions"])
    path = _write_hallways(cfg, [_record("wing_a", "Aya", "Lumi")])
    before = (Path(path).read_bytes(), os.stat(path).st_mtime_ns)
    with patch.object(hallways_mod, "_save_hallways", wraps=hallways_mod._save_hallways) as save:
        hallways_mod.compute_hallways_for_wing("sessions", col=_fake_collection(PAIRED), config=cfg)
    save.assert_not_called()
    assert (Path(path).read_bytes(), os.stat(path).st_mtime_ns) == before


def test_unlisted_wing_is_computed_as_before(tmp_path, monkeypatch):
    # Guard: fails if skipping leaks onto wings the setting does not name.
    monkeypatch.delenv("MEMPALACE_HALLWAY_SKIP_WINGS", raising=False)
    cfg = _config(tmp_path, ["sessions"])
    drawers = [dict(d, wing="wing_a") for d in PAIRED]
    created = hallways_mod.compute_hallways_for_wing(
        "wing_a", col=_fake_collection(drawers), config=cfg
    )
    assert [(h["wing"], h["entity_a"], h["entity_b"]) for h in created] == [
        ("wing_a", "Aya", "Lumi")
    ]


def test_env_skip_applies_without_a_config_object(tmp_path, monkeypatch):
    # Fails if the skip only works when a caller passes config= (the project miner may not).
    monkeypatch.setenv("MEMPALACE_HALLWAY_SKIP_WINGS", "sessions")
    hallway_file = tmp_path / "hallways.json"
    monkeypatch.setattr(hallways_mod, "_get_hallway_file", lambda *a, **kw: str(hallway_file))
    monkeypatch.setattr(hallways_mod, "_legacy_hallway_file", lambda: str(tmp_path / "legacy.json"))
    col = _fake_collection(PAIRED)
    assert hallways_mod.compute_hallways_for_wing("sessions", col=col) == []
    col.get.assert_not_called()


# --- the chat miner's wrapper ---------------------------------------------------------------------


def test_chat_miner_counts_a_skipped_wing_as_done(tmp_path, monkeypatch):
    # Fails if the convo miner's wrapper reports a skipped wing as a failed rebuild, which would
    # keep the rebuild pending and retry it on every save.
    monkeypatch.delenv("MEMPALACE_HALLWAY_SKIP_WINGS", raising=False)
    from mempalace import convo_miner

    cfg = _config(tmp_path, ["sessions"])
    col = _fake_collection(PAIRED)
    assert convo_miner._compute_hallways_for_wing_safe("sessions", col, 5, config=cfg) is True
    col.get.assert_not_called()
