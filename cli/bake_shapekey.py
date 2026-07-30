"""Headless CLI: bake one shape key into Basis (Approach 2).

Run:
  blender <in.blend> --background --factory-startup --python cli/bake_shapekey.py -- \
      --in <in.blend> --out <out.blend> --mesh <MeshName> --key <KeyName> \
      [--value 1.0] [--protect-group neck] [--head-mesh-names Body]

``--head-mesh-names`` is the comma-separated head-refusal list (default ``Body`` —
a NAME CONVENTION standing in for a geometric check; suffix/case-insensitive).
Baking a mesh on the list is refused because the bake recomputes its normals and
profiles never morph the head. Passing your own list (or '' to disable) is an
explicit assertion about where the head lives on THIS rig — e.g. a combined-mesh
rig whose ``Body`` includes the head keeps the refusal; a torso-only ``Body``
with the face on ``Face`` wants ``--head-mesh-names Face``.
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
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="bake_shapekey")
    p.add_argument("--in", dest="in_path", required=True,
                   help=".blend holding the mesh to bake")
    p.add_argument("--out", dest="out_path", required=True,
                   help="Where to save the baked .blend")
    p.add_argument("--mesh", dest="mesh", required=True,
                   help="Mesh object whose shape key is baked into its Basis")
    p.add_argument("--key", dest="key", required=True,
                   help="Shape key to bake")
    p.add_argument("--value", dest="value", type=float, default=1.0,
                   help="Fraction of the key to bake in (default 1.0 = fully)")
    p.add_argument("--protect-group", dest="protect_group", default="neck",
                   help="Vertex group held at its Basis position through the bake "
                        "(default 'neck', which keeps the seam to the head still)")
    p.add_argument("--head-mesh-names", dest="head_mesh_names", default="Body",
                   help="Comma-separated meshes to REFUSE to bake, matched "
                        "case-insensitively by suffix (default 'Body'). The bake "
                        "recomputes normals and profiles never morph the head, so this "
                        "asserts where the head lives on THIS rig; pass '' to disable")
    return p.parse_args(argv)


def main():
    args = _parse_args()
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    enable_avatarprep()
    from avatarprep.core import shapekey_bake

    mesh = bpy.data.objects.get(args.mesh)
    if mesh is None or mesh.type != 'MESH':
        print("AVATARPREP: ERROR mesh %r not found or not a mesh" % args.mesh)
        sys.exit(1)
    head_names = tuple(n.strip() for n in args.head_mesh_names.split(",") if n.strip())
    try:
        report = shapekey_bake.bake_shapekey_to_basis(
            mesh, args.key, args.value, protect_group=args.protect_group,
            head_mesh_names=head_names)
    except shapekey_bake.BakeError as e:
        print("AVATARPREP: ERROR", e)
        if "head mesh" in str(e):
            print("AVATARPREP: NOTE the head list is --head-mesh-names (default "
                  "'Body', a name convention); overriding it asserts where the "
                  "head lives on this rig — see the CLI docstring")
        sys.exit(1)
    print("AVATARPREP: baked %s=%g into Basis on %s (cumulative=%g, protected_loops=%d)"
          % (report["key"], report["value"], report["mesh"],
             report["baked_cumulative"], report["protected_loops"]))

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    main()
