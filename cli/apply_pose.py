"""Headless CLI: apply the armature's current pose as its new rest pose.

Run with::

    blender <in.blend> --background --factory-startup \
        --python cli/apply_pose.py -- --in <in.blend> --out <out.blend> [--scale-test]

The script loads ``--in``, enables the AvatarPrep extension from the bundled
source package (deterministic, no dependence on user prefs), calls the pure core
function, and saves the result to ``--out``. ``--scale-test`` first scales the
whole armature pose by 1.2x so the change is clearly non-identity (used by the
verification harness).
"""

import os
import sys
import argparse


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(prog="apply_pose")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--scale-test", action="store_true",
                   help="Scale the armature pose by 1.2x before applying (test)")
    return p.parse_args(argv)


def _enable_avatarprep():
    """Make the source ``avatarprep`` package importable and register it."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import avatarprep
    try:
        avatarprep.register()
    except Exception:
        # Already registered in this session; fine.
        pass
    return avatarprep


def main():
    args = _parse_args()
    import bpy

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))

    _enable_avatarprep()
    from avatarprep.core import scene_utils, rest_pose

    armature = scene_utils.find_armature()
    if armature is None:
        print("AVATARPREP: ERROR no armature found")
        sys.exit(1)

    # Put the armature into pose mode (required by the workflow).
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')

    if args.scale_test:
        for pbone in armature.pose.bones:
            pbone.scale = (1.2, 1.2, 1.2)
        bpy.context.view_layer.update()
        print("AVATARPREP: applied 1.2x pose scale to %d bones"
              % len(armature.pose.bones))

    result = rest_pose.apply_pose(armature)
    print("AVATARPREP: apply_pose result =", result)

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    main()
