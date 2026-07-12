"""Headless CLI: export the scene to a Unity/VRChat-correct FBX.

Run with::

    blender <in.blend> --background --factory-startup \
        --python cli/export_unity_fbx.py -- --in <in.blend> --out <out.fbx> [--no-embed]

Loads ``--in``, enables AvatarPrep from the bundled source package, and writes
``--out`` using the CATS export recipe.
"""

import os
import sys
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(prog="export_unity_fbx")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--armature", dest="armature", default=None,
                   help="Scope the export to this armature + its bound meshes (owned "
                        "re-export: selection-only, strips paths, no texture embed)")
    p.add_argument("--no-embed", dest="no_embed", action="store_true",
                   help="Do not embed textures (whole-scene export only)")
    return p.parse_args(argv)


def main():
    args = _parse_args()
    import bpy

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))

    enable_avatarprep()
    from avatarprep.core import fbx_export

    armature = None
    if args.armature:
        arm = bpy.context.scene.objects.get(args.armature)
        if arm is None or arm.type != 'ARMATURE':
            print("AVATARPREP: ERROR --armature %r is not an armature in this scene"
                  % args.armature)
            sys.exit(1)
        armature = arm
    else:
        # Whole-scene export is valid, but an owned .blend that appended a
        # disposable base-body reference would silently bake that second rig into
        # the FBX. Warn loud (not fatal — multi-object scenes are legitimate).
        arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
        if len(arms) > 1:
            print("AVATARPREP: WARNING %d armatures in scene (%s); exporting WHOLE "
                  "SCENE — pass --armature <name> to scope to one rig"
                  % (len(arms), ", ".join(sorted(a.name for a in arms))))

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # A scoped export forces its own strip/no-embed recipe; --no-embed only
    # applies to the whole-scene path.
    fbx_export.export_unity_fbx(out_path, armature_obj=armature,
                                embed_textures=not args.no_embed)

    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print("AVATARPREP: exported FBX ->", out_path, "(%d bytes)" % size)


if __name__ == "__main__":
    main()
