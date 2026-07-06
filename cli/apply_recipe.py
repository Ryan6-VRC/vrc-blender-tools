"""Headless CLI: replay a proportioning recipe (ordered edges + per-edge overrides).

Run:
  blender <in.blend> --background --factory-startup --python cli/apply_recipe.py -- \
      --in <in.blend> --out <out.blend> --recipe <recipe.json> [--armature <name>] [--report <report.json>]
"""
import os
import sys
import json
import argparse


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="apply_recipe")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--recipe", dest="recipe", required=True)
    p.add_argument("--armature", dest="armature", default=None,
                   help="Armature object to target; required when the scene has more than one")
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
    with open(args.recipe, "r", encoding="utf-8") as fh:
        recipe = json.load(fh)
    recipe_dir = os.path.dirname(os.path.abspath(args.recipe))

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    _enable_avatarprep()
    from avatarprep.core import proportions

    armature = _resolve_armature(args.armature)

    sk_over = recipe.get("shapekey_overrides", {})
    bn_over = recipe.get("bone_overrides", {})

    reports = []
    for edge_ref in recipe["path"]:
        edge_path = edge_ref if os.path.isabs(edge_ref) else os.path.join(recipe_dir, edge_ref)
        try:
            report = proportions.apply_profile(
                armature, None, edge_path,
                bone_overrides=bn_over.get(edge_ref, {}),
                shapekey_overrides=sk_over.get(edge_ref, {}))
        except proportions.EdgeError as e:
            print("AVATARPREP: ERROR at edge %r:" % edge_ref, e)
            sys.exit(1)
        reports.append({"edge": edge_ref, "report": report})
        print("AVATARPREP: edge %r -> %s (%d scale ops, %d shapekeys, %d bakes, %d warnings)"
              % (edge_ref, report["state"], report["scales_applied"],
                 len(report["shapekeys"]), len(report["bakes"]), len(report["warnings"])))
        for w in report["warnings"]:
            print("AVATARPREP: WARNING [%s]" % edge_ref, w)

    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, indent=2)
        print("AVATARPREP: report ->", report_path)

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    main()
