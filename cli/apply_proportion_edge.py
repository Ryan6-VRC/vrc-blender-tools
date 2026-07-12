"""Headless CLI: apply one proportion edge to a blend (or validate with --whatif).

Run:
  blender <in.blend> --background --factory-startup --python cli/apply_proportion_edge.py -- \
      --in <in.blend> --out <out.blend> --edge <edge.json> [--skip-shapekeys] \
      [--bone-override OLD=NEW ...] [--shapekey-override NAME=VALUE ...] [--report <report.json>]

  # Read-only gate: validate the edge against the scene, no mutation, no --out.
  blender <in.blend> --background --factory-startup --python cli/apply_proportion_edge.py -- \
      --in <in.blend> --edge <edge.json> --whatif [--skip-shapekeys] \
      [--bone-override OLD=NEW ...] [--shapekey-override NAME=VALUE ...] [--report <report.json>]
"""
import os
import sys
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, kv, write_report


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="apply_proportion_edge")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", default=None)   # not required under --whatif
    p.add_argument("--edge", dest="edge", required=True)
    p.add_argument("--armature", dest="armature", default=None,
                   help="Armature object to target; required when the scene has more than one")
    p.add_argument("--whatif", dest="whatif", action="store_true",
                   help="Validate the edge against the scene and report offenders; no mutation, no --out")
    p.add_argument("--skip-shapekeys", action="store_true")
    p.add_argument("--bone-override", action="append", default=[])
    p.add_argument("--shapekey-override", action="append", default=[])
    p.add_argument("--report", dest="report", default=None)
    args = p.parse_args(argv)
    if not args.whatif and not args.out_path:
        p.error("--out is required unless --whatif is given")
    return args


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
    enable_avatarprep()
    from avatarprep.core import scene_utils, proportions

    armature = _resolve_armature(args.armature)

    bone_overrides = kv(args.bone_override)
    sk_raw = kv(args.shapekey_override)
    shapekey_overrides = {k: (None if v.lower() == "null" else float(v))
                          for k, v in sk_raw.items()}

    if args.whatif:
        # Read-only gate (folded from the former standalone validate CLI): load the edge,
        # check it against the rig, report offenders/warnings, exit 1 on offenders. No save.
        try:
            edge = proportions.load_edge(os.path.abspath(args.edge))
        except proportions.EdgeError as e:
            print("AVATARPREP: ERROR", e)
            sys.exit(1)

        meshes = scene_utils.get_bound_meshes(armature)
        report = proportions.validate_proportion_edge(
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
            write_report(args.report, report)

        sys.exit(1 if offenders else 0)

    try:
        report = proportions.apply_proportion_edge(
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
        write_report(args.report, report)

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    main()
