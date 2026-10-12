"""One-time cleanup of boilerplate and same-chat repeats (3D-Stories fork, sysop#297).

A palace saved before the save-time skip (step 8) holds rows a save with the
skip would not store. The cleanup removes exactly those rows and nothing else:

- a chat is ``(source_file, wing, extract_mode)``, and only ``convos`` rows count;
- a chat whose stamp is several conversation hashes joined by commas came from
  a file of several conversations, which the save leaves undeduplicated, so it
  is left alone;
- per chat, in piece order, ``plan_skips`` decides with the same rule a full read
  uses: boilerplate, or a repeat of an earlier piece's text, keeping the first
  copy, and keeping the first piece when every piece would go;
- a row a closet pointer or a knowledge-graph triple names is never removed;
- before each delete the row is read again, and a text that no longer matches
  the plan (no longer boilerplate, or no longer equal to its kept first copy)
  keeps the row.

It then fixes what depends on those rows: each (chat, ``source_mtime``) group's
``chunk_skipped`` grows by the rows removed from it, so no chat's "already
saved?" answer changes; a removed conversation hash moves to the chat's first
remaining row; and each affected bookmark gains the removed pieces in
``skipped`` and re-anchors on the last stored piece before its base.

Planning reads the palace's SQLite file directly and writes nothing. Applying
holds the palace's write lock for the whole run and refuses at once when a save
holds it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .save_dedupe import is_boilerplate as _is_boilerplate
from .save_dedupe import load_boilerplate_hashes, plan_skips, text_sha

DRAWERS = "mempalace_drawers"
CLOSETS = "mempalace_closets"
_KEYS = (
    "chroma:document",
    "source_file",
    "wing",
    "room",
    "extract_mode",
    "ingest_mode",
    "chunk_index",
    "chunk_total",
    "chunk_skipped",
    "source_mtime",
    "content_hash",
)
_BATCH = 500
_PREVIEW = 120
_TICKED = re.compile(r"^- \[[xX]\] ")
_TICKED_SHA = re.compile(r"^- \[[xX]\] ([0-9a-f]{64}) \|")


def is_boilerplate(text: str, listed: Optional[frozenset] = None) -> bool:
    """The save-time rule, with the reviewed list on disk by default."""
    return _is_boilerplate(text, load_boilerplate_hashes() if listed is None else listed)


def _read_rows(palace_path: str, collection: str) -> dict:
    """``{drawer_id: {key: value}}`` for one collection, straight from SQLite, read-only."""
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    marks = ",".join("?" for _ in _KEYS)
    query = f"""
        SELECT e.embedding_id, m.key, m.string_value, m.int_value, m.float_value, m.bool_value
          FROM embeddings e
          JOIN segments s ON e.segment_id = s.id
          JOIN collections c ON s.collection = c.id
          JOIN embedding_metadata m ON m.id = e.id
         WHERE c.name = ? AND m.key IN ({marks})
    """
    rows: dict = defaultdict(dict)
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        for drawer_id, key, sval, ival, fval, bval in db.execute(query, (collection, *_KEYS)):
            if sval is not None:
                value = sval
            elif ival is not None:
                value = ival
            elif fval is not None:
                value = fval
            elif bval is not None:
                value = bool(bval)
            else:
                value = None
            rows[drawer_id][key] = value
    return dict(rows)


def _referenced_ids(palace_path: str) -> set:
    """Drawer ids a closet pointer or a knowledge-graph triple names."""
    from .searcher import _extract_drawer_ids_from_closet

    found: set = set()
    try:
        closets = _read_rows(palace_path, CLOSETS)
    except sqlite3.Error:
        closets = {}
    for meta in closets.values():
        doc = meta.get("chroma:document")
        if isinstance(doc, str):
            found.update(_extract_drawer_ids_from_closet(doc))
    kg_path = os.path.join(palace_path, "knowledge_graph.sqlite3")
    if os.path.exists(kg_path):
        try:
            with closing(sqlite3.connect(f"file:{kg_path}?mode=ro", uri=True)) as kg:
                for (drawer_id,) in kg.execute(
                    "SELECT source_drawer_id FROM triples WHERE source_drawer_id IS NOT NULL"
                ):
                    found.add(str(drawer_id))
        except sqlite3.Error as exc:
            if "no such" not in str(exc):
                raise
    return found


def _chat_key(meta: dict):
    return (meta.get("source_file"), meta.get("wing"), meta.get("extract_mode"))


def _is_index(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass
class CleanupPlan:
    remove: dict = field(default_factory=dict)  # drawer_id -> "boilerplate" | "repeat"
    first_copy: dict = field(default_factory=dict)  # repeat drawer_id -> kept drawer_id
    chats: dict = field(default_factory=dict)  # chat key -> [(chunk_index, drawer_id)]
    listed: frozenset = frozenset()
    referenced_kept: set = field(default_factory=set)
    skipped_chats: int = 0  # chats left alone: unusable numbers, or several conversations
    rows_seen: int = 0
    seconds: float = 0.0

    @property
    def counts(self) -> dict:
        reasons = Counter(self.remove.values())
        return {
            "rows": self.rows_seen,
            "boilerplate": reasons["boilerplate"],
            "repeat": reasons["repeat"],
            "referenced_kept": len(self.referenced_kept),
            "chats_affected": len(self.per_chat),
            "chats_left_alone": self.skipped_chats,
        }

    @property
    def per_chat(self) -> Counter:
        owner = {drawer_id: key for key, pieces in self.chats.items() for _i, drawer_id in pieces}
        return Counter(owner[d][0] for d in self.remove)


def plan_cleanup(
    palace_path: str, *, listed: Optional[frozenset] = None, collection: str = DRAWERS
) -> CleanupPlan:
    """Decide what the cleanup removes. Reads the palace, writes nothing."""
    started = time.monotonic()
    listed = load_boilerplate_hashes() if listed is None else listed
    rows = _read_rows(palace_path, collection)
    referenced = _referenced_ids(palace_path)
    plan = CleanupPlan(listed=listed)
    grouped: dict = defaultdict(list)
    for drawer_id, meta in rows.items():
        if meta.get("ingest_mode") != "convos" or not isinstance(meta.get("chroma:document"), str):
            continue
        plan.rows_seen += 1
        grouped[_chat_key(meta)].append((meta.get("chunk_index"), drawer_id))
    for key, pieces in grouped.items():
        indexes = [i for i, _d in pieces]
        if not all(_is_index(i) for i in indexes) or len(set(indexes)) != len(indexes):
            plan.skipped_chats += 1  # cannot order it, so it cannot be judged
            continue
        if any("," in str(rows[d].get("content_hash") or "") for _i, d in pieces):
            # A file of several conversations is saved as one joined text whose
            # stamp is their hashes joined by commas; the save leaves such a file
            # undeduplicated (step 8), so the cleanup leaves it alone too.
            plan.skipped_chats += 1
            continue
        pieces.sort()
        plan.chats[key] = pieces
        texts = [rows[d]["chroma:document"] for _i, d in pieces]
        skip, _ = plan_skips(texts, listed=listed, keep_one=True)
        first_by_sha: dict = {}
        for pos, (_i, drawer_id) in enumerate(pieces):
            sha = text_sha(texts[pos])
            if pos not in skip:
                first_by_sha.setdefault(sha, drawer_id)
                continue
            if drawer_id in referenced:
                plan.referenced_kept.add(drawer_id)
                first_by_sha.setdefault(sha, drawer_id)
                continue
            if _is_boilerplate(texts[pos], listed):
                plan.remove[drawer_id] = "boilerplate"
            else:
                plan.remove[drawer_id] = "repeat"
                plan.first_copy[drawer_id] = first_by_sha[sha]
    plan.seconds = time.monotonic() - started
    return plan


def _get(col, ids: list, include: list) -> dict:
    out: dict = {}
    for start in range(0, len(ids), _BATCH):
        got = col.get(ids=ids[start : start + _BATCH], include=include)
        docs = got.get("documents") or [None] * len(got["ids"])
        metas = got.get("metadatas") or [None] * len(got["ids"])
        for drawer_id, doc, meta in zip(got["ids"], docs, metas):
            out[drawer_id] = (doc, meta or {})
    return out


def _mtime_key(meta: dict):
    value = meta.get("source_mtime")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return ("unreadable", repr(value))


def _skip_count(meta: dict) -> int:
    value = meta.get("chunk_skipped", 0)
    return value if _is_index(value) and value >= 0 else 0


def apply_cleanup(palace_path: str, plan: CleanupPlan, *, collection: str = DRAWERS) -> dict:
    """Remove the planned rows and fix labels and bookmarks, under the palace lock."""
    from .convo_miner import (
        _anchor_index,
        _bookmark_path,
        _bookmark_valid,
        _drop_bookmark,
        _load_bookmark,
        _save_bookmark,
    )
    from .palace import get_collection
    from .palace import mine_palace_lock

    started = time.monotonic()
    result = Counter()
    with mine_palace_lock(palace_path):
        col = get_collection(palace_path, collection, create=False)
        referenced = _referenced_ids(palace_path)
        planned = list(plan.remove)
        firsts = sorted(set(plan.first_copy.values()))
        now = _get(col, planned + firsts, ["documents", "metadatas"])

        removed_by_chat: dict = defaultdict(list)  # chat key -> [(index, meta)]
        to_delete: list = []
        for drawer_id in planned:
            if drawer_id in referenced:
                result["referenced_kept"] += 1
                continue
            doc, meta = now.get(drawer_id, (None, None))
            if not isinstance(doc, str):
                result["missing"] += 1
                continue
            if plan.remove[drawer_id] == "boilerplate":
                ok = _is_boilerplate(doc, plan.listed)
            else:
                first_doc, first_meta = now.get(plan.first_copy[drawer_id], (None, {}))
                ok = (
                    isinstance(first_doc, str)
                    and first_doc == doc
                    and _chat_key(first_meta) == _chat_key(meta)
                )
            if not ok:
                result["mismatch_kept"] += 1
                continue
            removed_by_chat[_chat_key(meta)].append((meta.get("chunk_index"), meta))
            to_delete.append(drawer_id)
            result[f"removed_{plan.remove[drawer_id]}"] += 1
        result["referenced_kept"] += len(plan.referenced_kept)

        for start in range(0, len(to_delete), _BATCH):
            col.delete(ids=to_delete[start : start + _BATCH])
        result["removed"] = len(to_delete)

        relabel: dict = {}
        removed_ids = set(to_delete)
        for key, removed in removed_by_chat.items():
            remaining_ids = [d for _i, d in plan.chats[key] if d not in removed_ids]
            remaining = _get(col, remaining_ids, ["metadatas"])
            by_group = Counter(_mtime_key(meta) for _i, meta in removed)
            for drawer_id, (_doc, meta) in remaining.items():
                gone = by_group.get(_mtime_key(meta), 0)
                if gone:
                    new = dict(meta)
                    new["chunk_skipped"] = _skip_count(meta) + gone
                    relabel[drawer_id] = new
            lost_hash = next(
                (m.get("content_hash") for _i, m in removed if m.get("content_hash")), None
            )
            ordered = sorted(
                (meta.get("chunk_index"), drawer_id) for drawer_id, (_d, meta) in remaining.items()
            )
            if lost_hash and ordered:
                first_id = ordered[0][1]
                first_meta = relabel.get(first_id) or dict(remaining[first_id][1])
                if not first_meta.get("content_hash"):
                    first_meta["content_hash"] = lost_hash
                    relabel[first_id] = first_meta
                    result["content_hash_moved"] += 1

            source_file, wing, extract_mode = key
            path = _bookmark_path(palace_path, source_file, extract_mode, wing)
            bm = _load_bookmark(path)
            if bm is None:
                continue
            gone_indexes = {i for i, _m in removed if i < bm["total"]}
            skipped = sorted(set(bm.get("skipped", [])) | gone_indexes)
            stored = {meta.get("chunk_index"): meta for _d, (_doc, meta) in remaining.items()}
            anchor = _anchor_index(bm["base"], frozenset(skipped))
            anchor_meta = stored.get(anchor) if anchor is not None else None
            new_bm = dict(bm)
            new_bm["skipped"] = skipped
            if anchor_meta is not None and anchor_meta.get("chunk_hash"):
                new_bm["prev_hash"] = anchor_meta["chunk_hash"]
            if anchor_meta is None or not _bookmark_valid(new_bm):
                _drop_bookmark(path)
                result["bookmarks_dropped"] += 1
                continue
            _save_bookmark(path, new_bm)
            result["bookmarks_updated"] += 1

        ids = list(relabel)
        for start in range(0, len(ids), _BATCH):
            batch = ids[start : start + _BATCH]
            col.update(ids=batch, metadatas=[relabel[d] for d in batch])
        result["relabelled"] = len(ids)
    result["seconds"] = round(time.monotonic() - started, 3)
    return dict(result)


def write_candidates(
    palace_path: str, out_path, *, min_chats: int = 25, collection: str = DRAWERS
) -> int:
    """Write texts held by at least ``min_chats`` chats, for the owner's review."""
    rows = _read_rows(palace_path, collection)
    chats: dict = defaultdict(set)
    count: Counter = Counter()
    sample: dict = {}
    for meta in rows.values():
        doc = meta.get("chroma:document")
        if meta.get("ingest_mode") != "convos" or not isinstance(doc, str):
            continue
        sha = text_sha(doc)
        chats[sha].add(meta.get("source_file"))
        count[sha] += 1
        sample.setdefault(sha, doc)
    picked = [
        sha
        for sha, holders in chats.items()
        if len(holders) >= min_chats and not _is_boilerplate(sample[sha], frozenset())
    ]
    picked.sort(key=lambda sha: (-count[sha], sha))
    lines = [
        "# Boilerplate candidates",
        "",
        f"Texts held by at least {min_chats} chats. Tick `[x]` on each line that is",
        "boilerplate, then run `mempalace dedupe-cleanup --write-list <this file>`.",
        "",
    ]
    for sha in picked:
        preview = sample[sha][:_PREVIEW].replace("\\", "\\\\").replace("\n", "\\n")
        lines.append(f"- [ ] {sha} | {len(chats[sha])} chats | {count[sha]} rows | {preview}")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(picked)


def write_list_from_review(review_path) -> int:
    """Write ``<config dir>/boilerplate.json`` from the ticked lines of a review file."""
    from .config import _default_config_dir

    picked = set()
    for number, line in enumerate(Path(review_path).read_text(encoding="utf-8").splitlines(), 1):
        if not _TICKED.match(line):
            continue
        found = _TICKED_SHA.match(line)
        if not found:
            raise ValueError(f"line {number} is ticked but holds no 64-character sha256")
        picked.add(found.group(1))
    cfg = _default_config_dir()
    cfg.mkdir(parents=True, exist_ok=True)
    target = cfg / "boilerplate.json"
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        shutil.copy2(target, cfg / f"boilerplate.json.bak-{stamp}")
    tmp = target.with_name(f"boilerplate.json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"version": 1, "sha256": sorted(picked)}) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return len(picked)
