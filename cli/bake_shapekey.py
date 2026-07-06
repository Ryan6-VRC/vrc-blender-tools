"""Headless CLI: bake one shape key into Basis (Approach 2).

Run:
  blender <in.blend> --background --factory-startup --python cli/bake_shapekey.py -- \
      --in <in.blend> --out <out.blend> --mesh <MeshName> --key <KeyName> \
      [--value 1.0] [--protect-group neck]
"""
import os
import sys
import argparse


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="bake_shapekey")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--mesh", dest="mesh", required=True)
    p.add_argument("--key", dest="key", required=True)
    p.add_argument("--value", dest="value", type=float, default=1.0)
    p.add_argument("--protect-group", dest="protect_group", default="neck")
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


def main():
    args = _parse_args()
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    _enable_avatarprep()
    from avatarprep.core import shapekey_bake

    mesh = bpy.data.objects.get(args.mesh)
    if mesh is None or mesh.type != 'MESH':
        print("AVATARPREP: ERROR mesh %r not found or not a mesh" % args.mesh)
        sys.exit(1)
    try:
        report = shapekey_bake.bake_shapekey_to_basis(
            mesh, args.key, args.value, protect_group=args.protect_group)
    except shapekey_bake.BakeError as e:
        print("AVATARPREP: ERROR", e)
        sys.exit(1)
    print("AVATARPREP: bake_shapekey report =", report)

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    main()
