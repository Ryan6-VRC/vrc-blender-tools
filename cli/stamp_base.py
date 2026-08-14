"""Headless CLI: stamp ``avatarprep_base`` (avatar body lineage) on an armature.

Run:
  blender <in.blend> --background --factory-startup --python cli/stamp_base.py -- \
      --in <in.blend> --out <out.blend> --armature <armatureObjectName> --base <label>

NOTE the deliberate divergence from compat/merge's ``--base``: here ``--armature``
names the armature OBJECT (as prune_bones does) and ``--base`` is the lineage LABEL
STRING to stamp (e.g. ``shinano``), NOT an armature object. Base identity is an
agent assertion, never guessed — this door is the only writer of avatarprep_base.

Exit codes: 0 = stamped (--out saved) · 1 = REFUSED (--out NOT saved; the source
loaded only with repairs — --force-load-repair accepts them deliberately) ·
2 = ERROR (bad armature name, unopenable --in, write failure).
"""
import os
import sys
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, resolve_arm, open_blend, run_cli, add_force_load_repair


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="stamp_base")
    p.add_argument("--in", dest="in_path", required=True,
                   help=".blend holding the armature to stamp")
    p.add_argument("--out", dest="out_path", required=True,
                   help="Where to save the stamped .blend")
    p.add_argument("--armature", dest="armature", required=True,
                   help="name of the armature OBJECT to stamp")
    p.add_argument("--base", dest="base", required=True,
                   help="avatar lineage LABEL string to stamp (e.g. 'shinano'); "
                        "this is a label, NOT an armature object")
    add_force_load_repair(p)
    return p.parse_args(argv)


def main():
    args = _parse_args()
    import bpy
    open_blend(args.in_path, writes=True, force_load_repair=args.force_load_repair)
    enable_avatarprep()
    from avatarprep.core import scene_utils

    arm = resolve_arm(args.armature, "armature")
    scene_utils.write_stamp(arm, scene_utils.STAMP_BASE, args.base)
    print("AVATARPREP: stamped base %s on %s" % (args.base, arm.name))

    try:
        out_path = os.path.abspath(args.out_path)
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=out_path)
    except Exception as e:
        print("AVATARPREP: ERROR failed to save out:", e)
        sys.exit(2)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    run_cli(main, "stamp_base")
