"""Headless CLI: import an FBX into a fresh scene and save a .blend.

Run with::

    blender --background --factory-startup \
        --python cli/import_fbx.py -- --fbx <in.fbx> --out <out.blend> [--global-scale N] \
        [--fit-reference <base.blend>]

Wraps avatarprep.core.import_fbx (Blender's current importer — never the legacy one,
which reorients bones ~90deg). Prints the observe_import sanity snapshot.

``--fit-reference`` links the base body the import will be fitted to, as objects by a
``//``-relative path (``scene_utils.link_fit_reference``): its rig and the meshes it carries,
in a local collection ``<stem>_ref``. The doors that pick an armature unnamed skip it, so the vendor
rig keeps its ``Armature`` name through the edge and the rename, and
``apply_proportion_edge --absent-bones stub`` reads its bones.

Exit 0 on a saved import; 1 on a missing --fbx or a fit reference that cannot be linked
(nothing saved); 2 on bad args or a crash.
"""
import os
import sys
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, run_cli


def _parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(prog="import_fbx")
    p.add_argument("--fbx", dest="fbx_path", required=True,
                   help="FBX to import into a fresh scene")
    p.add_argument("--out", dest="out_path", required=True,
                   help="Where to save the resulting .blend")
    p.add_argument("--global-scale", dest="global_scale", type=float, default=None,
                   help="Override the importer's global scale; omit to keep Blender's "
                        "default, which is what a Unity-bound round trip wants")
    p.add_argument("--fit-reference", dest="fit_reference", default=None, metavar="BLEND",
                   help="Link this base body's rig and its meshes as objects by a //-relative "
                        "path, for the fit doors to read; never exported or edited")
    return p.parse_args(argv)


def main():
    args = _parse_args()
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    enable_avatarprep()
    from avatarprep.core import import_fbx as import_mod

    fbx_path = os.path.abspath(args.fbx_path)
    if not os.path.exists(fbx_path):
        print("AVATARPREP: ERROR --fbx not found:", fbx_path)
        sys.exit(1)
    if args.fit_reference:
        drives = {os.path.splitdrive(os.path.abspath(x))[0].lower() for x in (args.fit_reference, args.out_path)}
        if len(drives) > 1:
            print("AVATARPREP: ERROR --fit-reference %s and --out %s are on different drives, so no // "
                  "path can link them; keep the base blend and the output on one drive"
                  % (os.path.abspath(args.fit_reference), os.path.abspath(args.out_path)))
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
    # height_m counts these at their UNDEFORMED shape: the depsgraph will not evaluate
    # them, and no measure in this repo escapes that. Staying silent would let the height
    # read as a clean measurement.
    for m in snap["unevaluated_meshes"]:
        print("AVATARPREP: WARNING mesh not evaluated, measured undeformed:", m)

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ref = None
    if args.fit_reference:
        from avatarprep.core import scene_utils
        try:
            ref = scene_utils.link_fit_reference(args.fit_reference)
        except scene_utils.FitReferenceError as e:
            print("AVATARPREP: ERROR", e)
            print("AVATARPREP: nothing was saved; --out NOT written.")
            sys.exit(1)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    if ref is not None:
        stored = ref["library"].filepath
        try:
            relative = scene_utils.relative_library_path(ref["library"])
        except scene_utils.FitReferenceError as e:
            print("AVATARPREP: ERROR", e)
            print("AVATARPREP: %s is saved with the absolute library path %s, which resolves only on "
                  "this machine; re-import beside the base blend" % (out_path, stored))
            sys.exit(1)
        if relative != stored:
            bpy.ops.wm.save_as_mainfile(filepath=out_path)
        print("AVATARPREP: fit reference linked: %s with %s from %s in collection %s"
              % (ref["armature"], ", ".join(ref["meshes"]) or "no mesh", relative, ref["collection"]))
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    run_cli(main, "import_fbx")
