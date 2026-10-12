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
- applying plans again under the palace lock and refuses the whole run unless
  that plan is identical, the sha256 of every text it removes included.

It then fixes what depends on those rows: each label set step 8 counts by,
``(source_mtime, chunk_total, chunk_skipped)`` with the stored values as they
are, has its ``chunk_skipped`` grow by
the rows removed from it, so its stored plus skipped count stays the same; a
removed conversation hash moves to the chat's first remaining row; and each
affected bookmark gains the removed pieces in ``skipped`` and re-anchors on the
last stored piece before its base, taking that piece's ``chunk_hash``.

A chat is left alone, whole, when that cannot be done safely: its source file
is also held under another wing or mode (step 8 answers per file, across
wings), any row is of an older ``normalize_version`` (step 8 ignores those),
any row has no text (step 8 counts it; the skip rule cannot judge it), any
row's file time is not a number, any row's count or total is not a plain
integer in range, more than one file
time already has a complete set, the complete file times would change (two
label sets merging into one, for instance), or the new bookmark would have a
missing anchor, fail ``_bookmark_valid``, or rely on keep-one. A
chat over the 500 MB cap has no full read, so its bookmark is its only way to
save again.

Planning reads the palace's SQLite file directly and writes nothing. Applying
holds the palace's write lock for the whole run and refuses at once when a save
holds it. The plan holds every final value; when there is anything to change,
applying copies the palace folder and its bookmark folder (kept outside it) to
a backup before it opens the palace for writing, writes ``MARKER_NAME`` in the
palace naming that backup, writes the planned values, and removes the marker. A run that stops part-way leaves the marker:
chat saves and every mode of the cleanup command refuse until the backup is
restored, which is
the only recovery. Nothing finishes a stopped run in place. Other writers (the
MCP drawer tools, which need the hub) are not blocked by the marker, and a
write by them before the restore would be lost by it: run the cleanup with the
hub and the sweep stopped, and keep them stopped until it finishes or the
backup is restored.
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
    "chunk_hash",
    "normalize_version",
)
_BATCH = 500
MARKER_NAME = "dedupe-cleanup-running.json"


class CleanupInterrupted(RuntimeError):
    """A previous apply stopped part-way; its palace backup must be restored first."""


class PlanOutdated(RuntimeError):
    """The palace changed between the plan and apply; plan again."""


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
    digests: dict = field(default_factory=dict)  # removed drawer_id -> sha256 of its text
    updates: dict = field(default_factory=dict)  # drawer_id -> labels it takes
    bookmarks: dict = field(default_factory=dict)  # bookmark path -> its new content
    chats: dict = field(default_factory=dict)  # chat key -> [(chunk_index, drawer_id)]
    listed: frozenset = frozenset()
    referenced_kept: set = field(default_factory=set)
    skipped_chats: int = 0  # chats left alone, whole; reasons in left_alone
    left_alone: Counter = field(default_factory=Counter)
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
            "left_alone_reasons": dict(self.left_alone),
        }

    @property
    def per_chat(self) -> Counter:
        owner = {drawer_id: key for key, pieces in self.chats.items() for _i, drawer_id in pieces}
        return Counter(owner[d][0] for d in self.remove)


def _leave_alone(plan: "CleanupPlan", reason: str) -> None:
    plan.skipped_chats += 1
    plan.left_alone[reason] += 1


def _group(meta: dict) -> tuple:
    """The label set step 8's "already saved?" checks count a row in."""
    from .palace import _skip_key

    total = meta.get("chunk_total")
    return (meta.get("source_mtime"), total, _skip_key(total, meta.get("chunk_skipped")))


def _complete(group: tuple, count: int) -> bool:
    from .palace import _stored_target

    _mtime, total, skip = group
    return total is None or count >= _stored_target(total, skip)


def _labels_well_formed(metas: dict) -> bool:
    """Every row's file time is a number, and ``chunk_total`` and ``chunk_skipped`` are
    absent or a plain int in range."""
    for meta in metas.values():
        mtime = meta.get("source_mtime")
        if isinstance(mtime, bool) or not isinstance(mtime, (int, float)):
            return False  # step 8 groups by the raw stored time; only numbers match it
        if float(mtime) != mtime:
            return False  # file_already_mined keeps it raw, prefetch_mined_set makes it a float
        total, skip = meta.get("chunk_total"), meta.get("chunk_skipped")
        if total is not None and not (_is_index(total) and total >= 1):
            return False
        if skip is not None and not (
            _is_index(skip) and skip >= 0 and (total is None or skip < total)
        ):
            return False
    return True


def _answers(groups: Counter) -> set:
    """The file times step 8 calls saved: those with a complete label set."""
    return {group[0] for group, count in groups.items() if _complete(group, count)}


def _relabel(metas: dict, removed: set) -> Optional[dict]:
    """New ``chunk_skipped`` per remaining row, or None when the chat must be left alone.

    Each label set that loses k rows takes ``chunk_skipped + k``, so its stored
    plus skipped count stays as it was. The whole chat is refused when any row's
    count or total is not a plain integer in range, when more than one file time
    already has a complete set, when a count would reach its total, or when the
    complete file times would change (two label sets merging, or the only
    complete set losing every row).
    """
    if not _labels_well_formed(metas):
        return None
    before = Counter(_group(m) for m in metas.values())
    if len(_answers(before)) > 1:
        return None  # which complete file time step 8 reports first is not modelled
    gone = Counter(_group(metas[d]) for d in removed)
    new_skip: dict = {}
    for group, k in gone.items():
        mtime, total, skip = group
        left = before[group] - k
        if total is None:
            continue
        new = skip + k
        if not 0 <= new < total:
            return None
        if left == 0:
            continue  # a complete set losing every row already failed new < total
        new_skip[group] = new
    after_of = {
        group: (group[0], group[1], new_skip[group]) if group in new_skip else group
        for group in before
    }
    after = Counter(after_of[_group(m)] for d, m in metas.items() if d not in removed)
    if _answers(after) != _answers(before):  # also catches two label sets merging
        return None
    return {
        d: new_skip[_group(m)]
        for d, m in metas.items()
        if d not in removed and _group(m) in new_skip
    }


def _bookmark_after(palace_path: str, key: tuple, indexes: dict, removed: set, hashes: dict):
    """(path, new bookmark, anchor piece) for a chat's bookmark after ``removed`` go;
    (None, None, None) with no bookmark; False when the chat must be left alone.

    ``indexes`` maps every stored drawer id of the chat to its piece number and
    ``hashes`` to its ``chunk_hash``. The new anchor must be stored, its hash
    becomes ``prev_hash`` (the next bookmark save checks exactly that), and the
    whole new bookmark must pass ``_bookmark_valid``, which needs that hash to be
    64 hex characters.
    """
    from .convo_miner import _anchor_index, _bookmark_path, _bookmark_valid, _load_bookmark

    source_file, wing, extract_mode = key
    if not (source_file and wing and extract_mode):
        return None, None, None
    path = _bookmark_path(palace_path, source_file, extract_mode, wing)
    bm = _load_bookmark(path)
    if bm is None:
        return None, None, None
    gone = {indexes[d] for d in removed}
    skipped = sorted(set(bm.get("skipped", [])) | {i for i in gone if i < bm["total"]})
    anchor = _anchor_index(bm["base"], frozenset(skipped))
    stored = {i: d for d, i in indexes.items() if d not in removed}
    if anchor is None or anchor not in stored:
        return False  # a chat over 500 MB would have no way left to save
    new_bm = dict(bm)
    new_bm["skipped"] = skipped
    new_bm["prev_hash"] = hashes.get(stored[anchor])
    if not _bookmark_valid(new_bm):
        return False  # a bookmark the next save would treat as absent
    return path, new_bm, anchor


def plan_cleanup(
    palace_path: str, *, listed: Optional[frozenset] = None, collection: str = DRAWERS
) -> CleanupPlan:
    """Decide what the cleanup removes. Reads the palace, writes nothing."""
    started = time.monotonic()
    listed = load_boilerplate_hashes() if listed is None else listed
    rows = _read_rows(palace_path, collection)
    referenced = _referenced_ids(palace_path)
    plan = CleanupPlan(listed=listed)
    from .palace import NORMALIZE_VERSION

    grouped: dict = defaultdict(list)
    holders: dict = defaultdict(set)  # source_file -> every (wing, mode, ingest) holding it
    for drawer_id, meta in rows.items():
        holders[meta.get("source_file")].add(
            (meta.get("wing"), meta.get("extract_mode"), meta.get("ingest_mode"))
        )
        if meta.get("ingest_mode") != "convos":
            continue
        plan.rows_seen += 1
        grouped[_chat_key(meta)].append((meta.get("chunk_index"), drawer_id))
    for key, pieces in grouped.items():
        if len(holders[key[0]]) > 1:
            # Step 8 answers "already saved?" per source file across every wing;
            # the cleanup judges one chat at a time, so a shared file is left alone.
            _leave_alone(plan, "shared source")
            continue
        if any(rows[d].get("normalize_version") != NORMALIZE_VERSION for _i, d in pieces):
            # Step 8 ignores rows of an older version; the cleanup counts them.
            _leave_alone(plan, "older version")
            continue
        if any(not isinstance(rows[d].get("chroma:document"), str) for _i, d in pieces):
            # Step 8 counts a row with no text; the skip rule cannot judge it.
            _leave_alone(plan, "missing text")
            continue
        indexes = [i for i, _d in pieces]
        if not all(_is_index(i) for i in indexes) or len(set(indexes)) != len(indexes):
            _leave_alone(plan, "unusable piece numbers")  # cannot order it, so cannot judge it
            continue
        if any("," in str(rows[d].get("content_hash") or "") for _i, d in pieces):
            # A file of several conversations is saved as one joined text whose
            # stamp is their hashes joined by commas; the save leaves such a file
            # undeduplicated (step 8), so the cleanup leaves it alone too.
            _leave_alone(plan, "several conversations")
            continue
        pieces.sort()
        texts = [rows[d]["chroma:document"] for _i, d in pieces]
        skip, kept_one = plan_skips(texts, listed=listed, keep_one=True)
        remove: dict = {}
        kept_referenced: set = set()
        for pos, (_i, drawer_id) in enumerate(pieces):
            if pos not in skip:
                continue
            if drawer_id in referenced:
                kept_referenced.add(drawer_id)
                continue
            remove[drawer_id] = "boilerplate" if _is_boilerplate(texts[pos], listed) else "repeat"
        if not remove:
            plan.referenced_kept |= kept_referenced
            continue
        metas = {d: rows[d] for _i, d in pieces}
        relabel = _relabel(metas, set(remove))
        if relabel is None:
            _leave_alone(plan, "label sets")
            continue
        bookmark = _bookmark_after(
            palace_path,
            key,
            {d: i for i, d in pieces},
            set(remove),
            {d: rows[d].get("chunk_hash") for _i, d in pieces},
        )
        if bookmark is False or (bookmark[0] is not None and kept_one):
            _leave_alone(plan, "bookmark")
            continue
        updates = {d: {"chunk_skipped": n} for d, n in relabel.items()}
        lost_hash = next(
            (
                metas[d]["content_hash"]
                for _i, d in pieces
                if d in remove and metas[d].get("content_hash")
            ),
            None,
        )
        remaining = [d for _i, d in pieces if d not in remove]
        if lost_hash and remaining and not metas[remaining[0]].get("content_hash"):
            updates.setdefault(remaining[0], {})["content_hash"] = lost_hash
        plan.chats[key] = pieces
        plan.remove.update(remove)
        plan.digests.update({d: text_sha(rows[d]["chroma:document"]) for d in remove})
        plan.updates.update(updates)
        if bookmark[0] is not None:
            plan.bookmarks[str(bookmark[0])] = bookmark[1]
        plan.referenced_kept |= kept_referenced
    plan.seconds = time.monotonic() - started
    return plan


def _marker(palace_path: str) -> Path:
    return Path(palace_path) / MARKER_NAME


def _write_marker(path: Path, actions: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(actions, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _apply_deletes(col, ids: list) -> None:
    for start in range(0, len(ids), _BATCH):
        col.delete(ids=ids[start : start + _BATCH])


def _apply_metadata(col, updates: list) -> None:
    """Set the given labels; chromadb merges them into each row's labels."""
    for start in range(0, len(updates), _BATCH):
        batch = updates[start : start + _BATCH]
        col.update(ids=[d for d, _u in batch], metadatas=[u for _d, u in batch])


def _apply_bookmarks(bookmarks: list) -> None:
    from .convo_miner import _save_bookmark

    for path, bm in bookmarks:
        _save_bookmark(Path(path), bm)


def _essentials(plan: CleanupPlan) -> tuple:
    """Everything apply would write, so two plans can be compared."""
    return (plan.remove, plan.digests, plan.updates, plan.bookmarks)


def default_backup_path(palace_path: str) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    return f"{os.path.abspath(palace_path)}.pre-dedupe-{stamp}"


def _backup(palace_path: str, backup_path: str) -> dict:
    """Copy the palace folder and its bookmark folder (kept outside it) to ``backup_path``."""
    from .convo_miner import _bookmark_dir

    target = Path(backup_path).resolve()
    bookmarks = _bookmark_dir(palace_path)
    for source in (Path(palace_path).resolve(), bookmarks.resolve()):
        if target == source or source in target.parents:
            raise ValueError(f"the backup {target} is inside {source}, which it copies")
    target.mkdir(parents=True)  # refuses an existing path
    shutil.copytree(palace_path, target / "palace")
    if bookmarks.exists():
        shutil.copytree(bookmarks, target / "convo_bookmarks")
    return {
        "palace": str(target / "palace"),
        "palace_to": os.path.abspath(palace_path),
        "bookmarks": str(target / "convo_bookmarks") if bookmarks.exists() else None,
        "bookmarks_to": str(bookmarks),
    }


def restore_hint(palace_path: str) -> str:
    """What to do about a cleanup that stopped part-way, read from its marker."""
    marker = _marker(palace_path)
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        lines = [f"restore {info['palace']} in place of {info['palace_to']}"]
        if info.get("bookmarks"):
            lines.append(f"and {info['bookmarks']} in place of {info['bookmarks_to']}")
        else:
            lines.append(f"and remove {info['bookmarks_to']}")
        return " ".join(lines) + ", then run again"
    except (OSError, ValueError, KeyError, TypeError):
        return f"restore the palace backup taken before it, then remove {marker}"


def apply_cleanup(
    palace_path: str,
    plan: CleanupPlan,
    *,
    backup_path: Optional[str] = None,
    collection: str = DRAWERS,
) -> dict:
    """Back up, then remove the planned rows and fix labels and bookmarks, under the palace lock.

    Under the lock it plans again from the palace as it is and refuses with
    ``PlanOutdated`` unless that plan is identical to ``plan``, text fingerprints
    included. It then copies the palace folder and its bookmark folder to
    ``backup_path`` (by default beside the palace) before this process opens the
    palace for writing (planning reads its SQLite file read-only), writes ``MARKER_NAME`` naming that copy, writes the planned
    values, and removes the marker. A plan with nothing to change does nothing
    and takes no copy. Refuses with ``CleanupInterrupted`` while a marker exists.
    """
    from .palace import get_collection, mine_palace_lock

    started = time.monotonic()
    marker = _marker(palace_path)
    result = Counter()
    with mine_palace_lock(palace_path):
        if marker.exists():
            raise CleanupInterrupted(
                f"an earlier cleanup stopped part-way: {restore_hint(palace_path)}"
            )
        if plan.remove:
            fresh = plan_cleanup(palace_path, listed=plan.listed, collection=collection)
            if _essentials(fresh) != _essentials(plan):
                raise PlanOutdated("the palace changed since the plan; plan again")
            backup_path = backup_path or default_backup_path(palace_path)
            info = _backup(palace_path, backup_path)
            _write_marker(marker, {**info, "started": datetime.now().isoformat()})
            col = get_collection(palace_path, collection, create=False)
            _apply_deletes(col, sorted(plan.remove))
            _apply_metadata(col, [[d, u] for d, u in sorted(plan.updates.items())])
            _apply_bookmarks([[p, bm] for p, bm in sorted(plan.bookmarks.items())])
            marker.unlink()
            result["backup"] = backup_path
        reasons = Counter(plan.remove.values())
        result["removed"] = len(plan.remove)
        result["removed_boilerplate"] = reasons["boilerplate"]
        result["removed_repeat"] = reasons["repeat"]
        result["relabelled"] = sum("chunk_skipped" in u for u in plan.updates.values())
        result["content_hash_moved"] = sum("content_hash" in u for u in plan.updates.values())
        result["bookmarks_updated"] = len(plan.bookmarks)
        result["referenced_kept"] = len(plan.referenced_kept)
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
