"""Headless CLI: prune zero-weight bones orphaned by dropped meshes.

Run:
  blender <in.blend> --background --factory-startup --python cli/prune_bones.py -- \
      --in <in.blend> --out <out.blend> [--armature <name>] [--report <report.json>]

Prune has no PASS/FAIL — it always succeeds; over-pruning surfaces downstream, not
here (see docs/blender.md). Exit 0 on success; 2 = ERROR (bad name, write failure).
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
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--armature", dest="armature", default=None)
    p.add_argument("--report", dest="report", default=None)
    return p.parse_args(argv)


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

    result = prune_zero_weight_bones(armature)
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
