"""Headless CLI: validate one proportion edge against the live scene (read-only gate).

No mutation, no ``--out``. Loads the scene armature + its bound meshes, structurally
loads the edge, and checks it against the rig: missing bones/shapekeys surface as named
offenders (exit 1), softer issues as warnings. Mirrors apply_profile's args so an edge
that validates clean here applies there unchanged.

Run:
  blender <in.blend> --background --factory-startup --python cli/validate_profile.py -- \
      --in <in.blend> --edge <edge.json> [--skip-shapekeys] \
      [--bone-override OLD=NEW ...] [--shapekey-override NAME=VALUE ...] [--report <report.json>]
"""
import os
import sys
import json
import argparse


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="validate_profile")
    p.add_argument("--in", dest="in_path", required=True)
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
    from avatarprep.core import scene_utils, proportions

    armature = _resolve_armature(args.armature)

    bone_overrides = _kv(args.bone_override)
    sk_raw = _kv(args.shapekey_override)
    shapekey_overrides = {k: (None if v.lower() == "null" else float(v))
                          for k, v in sk_raw.items()}
    try:
        edge = proportions.load_edge(os.path.abspath(args.edge))
    except proportions.EdgeError as e:
        print("AVATARPREP: ERROR", e)
        sys.exit(1)

    meshes = scene_utils.get_bound_meshes(armature)
    report = proportions.validate_profile(
        armature, meshes, edge, bone_overrides=bone_overrides,
        shapekey_overrides=shapekey_overrides, skip_shapekeys=args.skip_shapekeys)

    offenders = report["offenders"]
    warnings = report["warnings"]
    verdict = "FAIL" if offenders else "PASS"
    print("AVATARPREP: validate %s %s -> %s (%d offenders, %d warnings)"
          % (verdict, edge["source"], edge["target"], len(offenders), len(warnings)))
    for o in offenders:
        print("AVATARPREP: OFFENDER", o)
    for w in warnings:
        print("AVATARPREP: WARNING", w)

    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("AVATARPREP: report ->", report_path)

    sys.exit(1 if offenders else 0)


if __name__ == "__main__":
    main()
