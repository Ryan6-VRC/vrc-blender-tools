"""Headless CLI: apply the armature's current pose as its new rest pose.

Run with::

    blender <in.blend> --background --factory-startup \
        --python cli/apply_pose.py -- --in <in.blend> --out <out.blend>

Loads ``--in``, calls the core function and saves to ``--out``.

Exit codes: 0 = baked (--out saved) · 1 = --out NOT saved: REFUSED (a bound mesh the
depsgraph will not evaluate would bake undeformed) · 2 = ERROR (unopenable --in, no
armature, write failure).
"""

import os
import sys
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli, add_force_load_repair


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(prog="apply_pose")
    p.add_argument("--in", dest="in_path", required=True,
                   help=".blend whose armature pose becomes its rest pose")
    p.add_argument("--out", dest="out_path", required=True,
                   help="Where to save the result")
    add_force_load_repair(p)
    return p.parse_args(argv)


def main():
    args = _parse_args()
    import bpy

    open_blend(args.in_path, writes=True, force_load_repair=args.force_load_repair)
    enable_avatarprep()
    from avatarprep.core import scene_utils, rest_pose

    armature = scene_utils.find_armature()
    if armature is None:
        print("AVATARPREP: ERROR no armature found")
        sys.exit(2)

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')

    try:
        result = rest_pose.apply_pose(armature)
    except rest_pose.RestPoseRefused as refused:
        print("AVATARPREP: apply_pose REFUSED —", refused)
        for o in refused.offenders:
            print("AVATARPREP: OFFENDER", o)
        print("AVATARPREP: nothing was baked; --out NOT written.")
        sys.exit(1)
    print("AVATARPREP: rest pose applied on %s (meshes=%d)"
          % (result["armature"], len(result["meshes_processed"])))
    for m in result["meshes_processed"]:
        print("AVATARPREP: mesh %s rebased (%s, shape_keys=%d)"
              % (m["name"], m["method"], m["shape_key_count"]))

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    run_cli(main, "apply_pose")
