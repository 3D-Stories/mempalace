# Loaded into mempalace.cli via exec (see __init__.py). Not a standalone module.
if __name__ != "mempalace.cli":
    raise ImportError(f"{__name__} is an implementation fragment; import mempalace.cli")


def cmd_dedupe_cleanup(args):
    """One-time removal of boilerplate and same-chat repeats (3D-Stories fork, sysop#297).

    A dry run by default. ``--candidates`` writes texts many chats hold for review,
    ``--write-list`` turns the ticked lines into the boilerplate list, and
    ``--apply --yes`` performs the plan under the palace lock.
    """
    from .. import dedupe_cleanup as dc
    from ..palace import MineAlreadyRunning

    if args.write_list:
        count = dc.write_list_from_review(args.write_list)
        print(f"  ✓ boilerplate list written: {count} texts")
        return

    palace_path = os.path.abspath(
        os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    )
    if args.candidates:
        count = dc.write_candidates(palace_path, args.candidates, min_chats=args.min_chats)
        print(f"  ✓ {count} candidate texts written to {args.candidates}")
        return
    if args.apply and not args.yes:
        print("  ✗ --apply removes rows; stop the hub and the sweep, back up the palace,")
        print("    then run again with --apply --yes")
        raise SystemExit(2)

    plan = dc.plan_cleanup(palace_path)
    counts = plan.counts
    mode = "apply" if args.apply else "dry run"
    print(f"  dedupe-cleanup ({mode}) on {palace_path}")
    print(
        f"  rows {counts['rows']}  boilerplate {counts['boilerplate']}  repeat {counts['repeat']}"
        f"  referenced kept {counts['referenced_kept']}"
    )
    print(
        f"  chats affected {counts['chats_affected']}  left alone {counts['chats_left_alone']}"
        f"  planned in {plan.seconds:.1f}s"
    )
    for source, n in plan.per_chat.most_common(args.top):
        print(f"    {n:>7}  {source}")
    if not args.apply:
        print("  dry run: nothing written")
        return
    try:
        result = dc.apply_cleanup(palace_path, plan)
    except MineAlreadyRunning as exc:
        print(f"  ✗ {exc}")
        raise SystemExit(3) from exc
    print(
        f"  removed {result.get('removed', 0)}"
        f" (boilerplate {result.get('removed_boilerplate', 0)},"
        f" repeat {result.get('removed_repeat', 0)})"
        f"  kept on mismatch {result.get('mismatch_kept', 0)}"
        f"  referenced kept {result.get('referenced_kept', 0)}"
    )
    print(
        f"  relabelled {result.get('relabelled', 0)}"
        f"  conversation hashes moved {result.get('content_hash_moved', 0)}"
        f"  bookmarks updated {result.get('bookmarks_updated', 0)}"
        f"  dropped {result.get('bookmarks_dropped', 0)}"
        f"  in {result.get('seconds', 0)}s"
    )
