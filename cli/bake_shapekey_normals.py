"""Headless CLI: make a mesh shade as a shape key at weight 1, at every weight of the key.

Run:
  blender <in.blend> --background --factory-startup --python cli/bake_shapekey_normals.py -- \
      --in <in.blend> (--out <out.blend> | --whatif) --mesh <MeshName> --key <KeyName> \
      [--fade-deg 0.1,1.0] [--head-mesh-names Body] [--force-load-repair]

Adds to every corner's custom normal (its computed normal when the mesh has none) the
change the key at weight 1 makes to the computed corner normal, so the surface shades as
the key at full weight whatever weight Unity plays it at. The use: a nipple dome a key
raises under a skin-tight top must read domed at a weight of 20 too, and Unity imports
blendshape normals as None, so without this the keyed shape shades with the flat Basis
normals. The change is faded in by its own angle, from nothing at the first
``--fade-deg`` to whole at the second (a smoothstep), so float noise off the key's
footprint writes nothing. Computed normals are those of a copy with its custom normals
cleared, so sharp edges still split them. No vertex moves; the key must be relative to
the Basis.

A saving run records ``{key: recipe}`` in the mesh object's ``avatarprep_normals``, the
canonical command line printed as ``recipe:``; a key already recorded refuses, since a
second pass would add the change twice: re-derive from the blend before that bake.
``--whatif`` runs the same and saves nothing. ``--head-mesh-names`` is the same
head-refusal list as ``bake_shapekey`` (its docstring has the reasoning): the head's
normals are never rewritten.

Exit 0 on success; 1 on an ``AVATARPREP: ERROR`` refusal (nothing saved); 2 on bad args
or a crash.
"""
import os
import sys
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli, add_force_load_repair


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="bake_shapekey_normals")
    p.add_argument("--in", dest="in_path", required=True,
                   help=".blend holding the mesh")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", dest="out_path", default=None,
                      help="Where to save the .blend with the baked normals")
    mode.add_argument("--whatif", action="store_true",
                      help="Bake in memory, print the counts and save nothing")
    p.add_argument("--mesh", dest="mesh", required=True,
                   help="Mesh object whose custom normals take the key's shading")
    p.add_argument("--key", dest="key", required=True,
                   help="Shape key whose shading at weight 1 is baked")
    p.add_argument("--fade-deg", dest="fade_deg", default="0.1,1.0", metavar="LO,HI",
                   help="Degrees of normal change over which a corner's change fades in: none "
                        "at LO, whole at HI (default 0.1,1.0)")
    p.add_argument("--head-mesh-names", dest="head_mesh_names", default="Body",
                   help="Comma-separated meshes to REFUSE (default 'Body'), matched as "
                        "bake_shapekey matches them; pass '' to disable")
    add_force_load_repair(p)
    a = p.parse_args(argv)
    try:
        a.fade = tuple(float(x) for x in a.fade_deg.split(","))
    except ValueError:
        a.fade = ()
    if len(a.fade) != 2:
        p.error("--fade-deg wants two angles in degrees, LO,HI (e.g. 0.1,1.0), got %r" % a.fade_deg)
    return a


def main():
    args = _parse_args()
    import bpy
    open_blend(args.in_path, writes=not args.whatif, force_load_repair=args.force_load_repair)
    enable_avatarprep()
    from avatarprep.core import shapekey_bake

    mesh = bpy.data.objects.get(args.mesh)
    if mesh is None or mesh.type != 'MESH':
        print("AVATARPREP: ERROR mesh %r not found or not a mesh" % args.mesh)
        sys.exit(1)
    head_names = tuple(n.strip() for n in args.head_mesh_names.split(",") if n.strip())
    try:
        report = shapekey_bake.bake_shapekey_normals(
            mesh, args.key, fade=args.fade, head_mesh_names=head_names)
    except shapekey_bake.BakeError as e:
        print("AVATARPREP: ERROR", e)
        sys.exit(1)
    print("AVATARPREP: recipe: %s" % report["recipe"])
    print("AVATARPREP: baked %s normals on %s: %d corners take its change whole, %d faded, "
          "max turn %.1f deg, no vertex moved"
          % (report["key"], report["mesh"], report["whole"], report["faded"],
             report["max_turn_deg"]))

    if args.whatif:
        print("AVATARPREP: whatif; nothing saved")
        return
    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    run_cli(main, "bake_shapekey_normals")
