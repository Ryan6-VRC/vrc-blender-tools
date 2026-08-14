"""Headless CLI: rename scene objects as a set.

Run:
  blender <in.blend> --background --factory-startup --python cli/rename_objects.py -- \
      --in <in.blend> --out <out.blend> --rename OLD=NEW [--rename OLD=NEW ...] \
      [--report <report.json>]

  # Preview: the plan and its refusals. No mutation; --out must be omitted.
  blender <in.blend> --background --factory-startup --python cli/rename_objects.py -- \
      --in <in.blend> --whatif --rename OLD=NEW [...] [--report <report.json>]

Renames as a SET, so ``own-base``'s canonical ``Face=Body Body=Body_Base`` swap works
where a pair-by-pair rename would silently land ``Face`` on ``Body.001``. The report
carries ``source_map`` ({ourName: sourceName}) — the map the Unity material-copy step
needs, so it stops being maintained by hand.

Object names only; the mesh datablock name is left alone. Both decisions, and the
scan's deliberately scene-wide reach, are justified on
avatarprep.core.rename_objects.

Exit codes: 0 = renamed (--out saved) · 1 = REFUSED (--out NOT saved) · 2 = ERROR
(bad --rename grammar, unopenable --in, write failure, --out with --whatif).
"""
import os
import sys
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import (enable_avatarprep, write_report, open_blend, run_cli,
                         add_force_load_repair)


def _parse_renames(items):
    """Parse repeated ``OLD=NEW`` into a dict, refusing the two grammar errors that
    ``_common.kv`` would silently absorb.

    ``kv`` is right for its other callers and stays as it is; here a dropped ``=``
    would become ``{'Body': ''}`` and rename the object to ``Object``, and a repeated
    OLD would collapse last-wins with one pair vanishing unreported. Both are the
    caller's mistake and both must fail in grammar, not in geometry."""
    pairs = {}
    for item in items:
        if "=" not in item:
            print("AVATARPREP: ERROR --rename %r is not OLD=NEW (missing '=')" % item)
            sys.exit(2)
        old, _, new = item.partition("=")
        old, new = old.strip(), new.strip()
        if not old or not new:
            print("AVATARPREP: ERROR --rename %r has an empty side" % item)
            sys.exit(2)
        if old in pairs:
            print("AVATARPREP: ERROR --rename names %r twice (%r then %r); one source, "
                  "one target" % (old, pairs[old], new))
            sys.exit(2)
        pairs[old] = new
    if not pairs:
        print("AVATARPREP: ERROR no --rename pairs given")
        sys.exit(2)
    return pairs


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="rename_objects")
    p.add_argument("--in", dest="in_path", required=True,
                   help=".blend holding the objects to rename")
    p.add_argument("--out", dest="out_path", default=None,   # not required under --whatif
                   help="Where to save the renamed .blend; omit under --whatif")
    p.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                   help="Rename object OLD to NEW. Repeatable, and resolved as a SET, "
                        "so Face=Body Body=Body_Base is a legal swap")
    p.add_argument("--whatif", dest="whatif", action="store_true",
                   help="Report the plan and its refusals; no mutation, no --out")
    p.add_argument("--report", dest="report", default=None,
                   help="Write the full result dict here as JSON, on REFUSED too")
    add_force_load_repair(p)
    args = p.parse_args(argv)
    if not args.whatif and not args.out_path:
        p.error("--out is required unless --whatif is given")
    if args.whatif and args.out_path:
        p.error("--out is meaningless under --whatif (preview mutates nothing)")
    return args


def main():
    args = _parse_args()
    import bpy
    pairs = _parse_renames(args.rename)
    open_blend(args.in_path, writes=not args.whatif,
               force_load_repair=args.force_load_repair)
    enable_avatarprep()
    from avatarprep.core.rename_objects import rename_objects, RenameRefused

    try:
        result = rename_objects(pairs, whatif=args.whatif)
    except RenameRefused as refused:
        print("AVATARPREP: rename REFUSED —", refused)
        for o in refused.offenders:
            print("AVATARPREP: OFFENDER", o)
        print("AVATARPREP: nothing was renamed; --out NOT written.")
        if args.report:
            write_report(args.report, {"refused": str(refused),
                                       "offenders": refused.offenders})
        sys.exit(1)

    verb = "would rename" if args.whatif else "renamed"
    print("AVATARPREP: %s %d object(s)" % (verb, len(result["renamed"])))
    for old in sorted(result["renamed"]):
        print("AVATARPREP:   %s -> %s" % (old, result["renamed"][old]))
    for name in result["cycles"]:
        print("AVATARPREP: cycle broken via a temp name at %r" % name)
    for w in result["warnings"]:
        print("AVATARPREP: WARNING", w)

    if args.whatif:
        if args.report:
            write_report(args.report, result)
        return

    # Save the deliverable BEFORE the diagnostic report, so a report-write failure
    # cannot discard a successful rename (prune_bones' ordering, same reason).
    out_path = os.path.abspath(args.out_path)
    try:
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=out_path)
    except Exception as e:
        print("AVATARPREP: ERROR failed to save out:", e)
        sys.exit(2)
    print("AVATARPREP: saved ->", out_path)

    if args.report:
        write_report(args.report, result)


if __name__ == "__main__":
    run_cli(main, "rename_objects")
