"""Headless CLI: prune zero-weight bones orphaned by dropped meshes.

Run:
  blender <in.blend> --background --factory-startup --python cli/prune_bones.py -- \
      --in <in.blend> --out <out.blend> [--armature <name>] [--report <report.json>]

  # Read-only preview: the removal plan, grouped into chains. No mutation, no --out.
  blender <in.blend> --background --factory-startup --python cli/prune_bones.py -- \
      --in <in.blend> --whatif [--armature <name>] [--report <report.json>]

Prune has no PASS/FAIL — it always succeeds; over-pruning surfaces downstream, not
here (see docs/blender.md), which is what --whatif is for: the removal list drives a
keep/cut call, and reading it must not cost you the armature. Exit 0 on success;
2 = ERROR (bad name, write failure).
"""
import os
import sys
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, resolve_arm, write_report


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="prune_bones")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", default=None)   # not required under --whatif
    p.add_argument("--armature", dest="armature", default=None)
    p.add_argument("--whatif", dest="whatif", action="store_true",
                   help="Report the removal plan as rooted chains; no mutation, no --out")
    p.add_argument("--report", dest="report", default=None)
    args = p.parse_args(argv)
    if not args.whatif and not args.out_path:
        p.error("--out is required unless --whatif is given")
    return args


def _prepare_path(path, kind):
    try:
        abspath = os.path.abspath(path)
        d = os.path.dirname(abspath)
        if d:
            os.makedirs(d, exist_ok=True)
        return abspath
    except Exception as e:
        print("AVATARPREP: ERROR failed to prepare %s path:" % kind, e)
        sys.exit(2)


def main():
    args = _parse_args()
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    enable_avatarprep()
    from avatarprep.core import scene_utils
    from avatarprep.core.prune_bones import prune_zero_weight_bones

    if args.armature:
        armature = resolve_arm(args.armature, "armature")
    else:
        armature = scene_utils.find_armature()
        if armature is None:
            print("AVATARPREP: ERROR no armature found")
            sys.exit(2)

    result = prune_zero_weight_bones(armature, what_if=args.whatif)

    if args.whatif:
        print("AVATARPREP: whatif — would prune (kept %d, deleted %d) in %d chain(s)"
              % (result["kept"], result["deleted"], len(result["chains"])))
        for ch in result["chains"]:
            print("AVATARPREP: chain %s (%d bone(s)) under %s%s"
                  % (ch["root"], len(ch["bones"]), ch["parent"] or "<root>",
                     " [parent WEIGHTED — would-be physbone tip]" if ch["parent_weighted"] else ""))
            for name in ch["bones"]:
                print("AVATARPREP:   would prune", name)
        for tip in result["kept_tips"]:
            print("AVATARPREP: kept tip %s (physbone tail of weighted %s)"
                  % (tip["bone"], tip["parent"]))
        # Tripwire: measured empty across the vendor library, so anything here means
        # this asset breaks the assumption the keep rules are designed around.
        for obj in result["bone_parented_objects"]:
            print("AVATARPREP: WARNING bone-parented %s %r rides bone %r%s"
                  % (obj["type"], obj["object"], obj["bone"],
                     " — THAT BONE WOULD BE PRUNED" if obj["bone_pruned"] else ""))
        if args.report:
            write_report(args.report, result)
        return

    print("AVATARPREP: pruned (kept %d, deleted %d)"
          % (result["kept"], result["deleted"]))
    for name in result["deleted_bones"]:
        print("AVATARPREP: pruned bone", name)

    # Save the deliverable (--out) BEFORE the diagnostic report, so a
    # report-write failure can't discard a successful prune's output.
    out_path = _prepare_path(args.out_path, "out")
    try:
        bpy.ops.wm.save_as_mainfile(filepath=out_path)
    except Exception as e:
        print("AVATARPREP: ERROR failed to save out:", e)
        sys.exit(2)
    print("AVATARPREP: saved ->", out_path)

    if args.report:
        write_report(args.report, result)


if __name__ == "__main__":
    main()
