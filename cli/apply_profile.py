"""Headless CLI: apply one proportion edge to a blend.

Run:
  blender <in.blend> --background --factory-startup --python cli/apply_profile.py -- \
      --in <in.blend> --out <out.blend> --edge <edge.json> [--skip-shapekeys] \
      [--bone-override OLD=NEW ...] [--shapekey-override NAME=VALUE ...] [--report <report.json>]
"""
import os
import sys
import json
import argparse


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="apply_profile")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--edge", dest="edge", required=True)
    p.add_argument("--armature", dest="armature", default=None,
                   help="Armature object to target; required when the scene has more than one")
    p.add_argument("--skip-shapekeys", action="store_true")
    p.add_argument("--bone-override", action="append", default=[])
    p.add_argument("--shapekey-override", action="append", default=[])
    p.add_argument("--report", dest="report", default=None)
    return p.parse_args(argv)


def _enable_avatarprep():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import avatarprep
    try:
        avatarprep.register()
    except Exception:
        pass


def _kv(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def _resolve_armature(name):
    """Resolve the target armature at the CLI boundary — fail loud, never guess.

    A named ``--armature`` must exist and be an armature; with no name the scene
    must hold exactly one armature (>1 aborts naming them, so an owned .blend that
    appended a base-body reference can't silently target the wrong rig). The core
    ``find_armature`` stays permissive for its operator callers by design."""
    import bpy
    arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if name:
        obj = bpy.context.scene.objects.get(name)
        if obj is None or obj.type != 'ARMATURE':
            print("AVATARPREP: ERROR --armature %r is not an armature in this scene" % name)
            sys.exit(1)
        return obj
    if not arms:
        print("AVATARPREP: ERROR no armature found")
        sys.exit(1)
    if len(arms) > 1:
        print("AVATARPREP: ERROR multiple armatures (%s); pass --armature <name>"
              % ", ".join(sorted(a.name for a in arms)))
        sys.exit(1)
    return arms[0]


def main():
    args = _parse_args()
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    _enable_avatarprep()
    from avatarprep.core import proportions

    armature = _resolve_armature(args.armature)

    bone_overrides = _kv(args.bone_override)
    sk_raw = _kv(args.shapekey_override)
    shapekey_overrides = {k: (None if v.lower() == "null" else float(v))
                          for k, v in sk_raw.items()}
    try:
        report = proportions.apply_profile(
            armature, None, args.edge, bone_overrides=bone_overrides,
            shapekey_overrides=shapekey_overrides,
            skip_shapekeys=args.skip_shapekeys)
    except proportions.EdgeError as e:
        print("AVATARPREP: ERROR", e)
        sys.exit(1)
    print("AVATARPREP: applied %s -> %s (%d scale ops, %d shapekeys, %d bakes, %d warnings)"
          % (report["source"], report["target"], report["scales_applied"],
             len(report["shapekeys"]), len(report["bakes"]), len(report["warnings"])))
    for w in report["warnings"]:
        print("AVATARPREP: WARNING", w)

    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("AVATARPREP: report ->", report_path)

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    main()
