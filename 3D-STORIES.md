# 3D-Stories fork of MemPalace

This fork carries the local fixes that the 3D-Stories MemPalace hub runs, as plain commits on a
release branch. Upstream is [MemPalace/mempalace](https://github.com/MemPalace/mempalace).

## Branches and tags

- `develop`: upstream's default branch, kept as forked. Do not commit here.
- `3d-stories/v<upstream version>`: upstream's release tag plus the local commits below. Changes
  land by pull request into this branch.
- `v<upstream version>-3ds.<n>`: the tag the hub installs. A new local commit means a new `<n>`.

## The local commits on `3d-stories/v3.10.0`

1. **Backport of upstream PR #2408** by @PostProtoroman, unchanged, with its own test update: a
   re-save of a grown chat log re-embeds only what changed.
2. **Bookmark saves and `---` lines:** a save of a growing Claude Code log reads only the new bytes,
   and a line starting with `---` no longer drops the text after it. The two palace-wide post-save
   steps run at most every `MEMPALACE_FASTPATH_POSTMINE_SECS` (default 1800).
3. **One-log saves:** a save of one file reads only that file's rows for its duplicate checks.
4. **`hallways.skip_wings`:** a listed wing is never walked for hallways, and its old records are
   dropped once. Default: none.

Each commit carries its own tests under `tests/`. Measured numbers and the reasons for each step
are in the 3D-Stories `claude-skills` repository, `tools/mempalace-hub/README.md`.

## Install on the hub

```bash
pipx install --force 'git+https://github.com/3D-Stories/mempalace@v3.10.0-3ds.1'
```

Pin the dependencies to the versions already installed, so a reinstall cannot move ChromaDB under
the palace. Restart `mempalace-hub` afterwards, only when no mine is running.

## A new upstream release

1. Fetch upstream tags and create `3d-stories/v<new>` at the new tag.
2. Rebase the local commits onto it, dropping any that upstream now carries (check #2408 first).
3. Run the whole suite (`uv run pytest tests/ --ignore=tests/benchmarks -q`) and compare with the
   same run on the bare upstream tag.
4. Open a pull request into the new branch, then tag `v<new>-3ds.1`.

## Sending fixes upstream

Steps 2 to 4 are this fork's own work. Each can go upstream as a pull request from a branch cut off
upstream `develop`. Step 1 is already upstream as PR #2408.
