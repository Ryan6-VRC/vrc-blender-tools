"""Headless CLI: import an FBX into a fresh scene and save a .blend.

Run with::

    blender --background --factory-startup \
        --python cli/import_fbx.py -- --fbx <in.fbx> --out <out.blend> [--global-scale N]

Wraps avatarprep.core.import_fbx (Blender's current importer — never the legacy one,
which reorients bones ~90deg). Prints the observe_import sanity snapshot.
"""
import os
import sys
import argparse


def _parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(prog="import_fbx")
    p.add_argument("--fbx", dest="fbx_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--global-scale", dest="global_scale", type=float, default=None)
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

    _enable_avatarprep()
    from avatarprep.core import import_fbx as import_mod

    fbx_path = os.path.abspath(args.fbx_path)
    if not os.path.exists(fbx_path):
        print("AVATARPREP: ERROR --fbx not found:", fbx_path)
        sys.exit(1)

    settings = {}
    if args.global_scale is not None:
        settings["global_scale"] = args.global_scale

    snap = import_mod.import_fbx(fbx_path, **settings)
    print("AVATARPREP: imported %s (armatures=%d meshes=%d bones=%d shapekeys=%d height_m=%s)"
          % (os.path.basename(fbx_path), snap["armatures"], snap["meshes"],
             snap["bones"], snap["shapekeys"], snap["height_m"]))
    for m in snap["unparented_meshes"]:
        print("AVATARPREP: WARNING unparented mesh", m)

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    main()
