"""Headless CLI: copy the operator's hand edit from a staged scene into the garment blend.

Run:
  blender --background --factory-startup --python cli/harvest_handedit.py -- \\
      --in <garment.blend> --scene <scene.blend> --meshes M[,M] [--whatif]

The counterpart of ``stage_handedit``: reads ``--scene`` as saved on disk (so run it
only after the operator has saved and said the edit is done), then writes each named
mesh's edit into the same-named mesh of ``--in`` and saves ``--in`` over itself. The
garment blend keeps its own rig, parenting, links and stamps.

**What travels**, copied by vertex index: the mesh's vertex positions and Basis, every
shape key's positions as they are (Blender already carried a Basis edit onto each key
relative to it, so adding the Basis delta again would double-apply it), every vertex's
weights normalised to sum 1 over its groups, and the raw ``custom_normal`` values.
**What does not travel**: UVs, colour and every other attribute, shape-key values (the
stage set the row's), sharp edges and faces, seams, materials, vertex-group names and
order. A group the edit weights must already exist on the garment mesh.

**Gates, before anything is written**: identical vertex count, edge and loop topology
and polygon sizes; identical shape-key names in order, each with the same
``relative_key``; ``matrix_world`` within 1e-5 per element; ``custom_normal`` present on
both sides or neither, with the same domain and type; identical ``sharp_edge`` and
``sharp_face``, which shape the custom-normal spaces the raw values are relative to;
local, single-user mesh data on the garment side. A failed gate names the mesh and the
assertion and writes nothing. ``--whatif`` runs every gate, reports what the copy would
rewrite (vertices moved on Basis, per key, weights rewritten, normals changed), and
writes nothing.

The save is compressed, and writes no ``.blend1``: Blender's backup rotation is off for
it, so an existing ``.blend1`` beside ``--in`` is left as it was. A garment blend
Blender repaired as it read it refuses the write unless ``--force-load-repair``.

Exit 0 = harvested (or a clean ``--whatif``) · 1 = REFUSED (a gate failed; nothing
written) · 2 = ERROR (bad args, unopenable input, a crash). Core:
``avatarprep/core/handedit.py``.
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

TOOL = "harvest_handedit"


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: %s ? => FAIL: bad args: %s" % (TOOL, message))
        sys.exit(2)


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = _Parser(prog=TOOL)
    p.add_argument("--in", dest="in_path", required=True,
                   help="the garment .blend the edit is copied into; saved over itself")
    p.add_argument("--scene", required=True, help="the edit scene stage_handedit built, as saved")
    p.add_argument("--meshes", required=True, help="comma-separated garment mesh object(s)")
    p.add_argument("--whatif", action="store_true",
                   help="run every gate and report what the copy would rewrite; write nothing")
    add_force_load_repair(p)
    a = p.parse_args(argv)
    a.mesh_names = [n.strip() for n in a.meshes.split(",") if n.strip()]
    if not a.mesh_names:
        p.error("--meshes names no mesh")
    if os.path.abspath(a.in_path) == os.path.abspath(a.scene):
        p.error("--in and --scene are the same file")
    return a


def main():
    args = _parse_args()
    import bpy
    label = ",".join(args.mesh_names)
    in_path = os.path.abspath(args.in_path)

    open_blend(os.path.abspath(args.scene), writes=False)
    enable_avatarprep()
    from avatarprep.core import handedit as core
    try:
        snaps = core.read_edit(args.mesh_names)
    except core.HandEditRefused as refused:
        print("AVATARPREP: %s %s => FAIL: %s" % (TOOL, label, refused))
        sys.exit(1)

    open_blend(in_path, writes=not args.whatif, force_load_repair=args.force_load_repair)
    offenders, changes = core.check_harvest(snaps)
    if offenders:
        print("AVATARPREP: %s %s => FAIL: %d gate(s) failed; the index copy would be invalid"
              % (TOOL, label, len(offenders)))
        for o in offenders:
            print("AVATARPREP: OFFENDER", o)
        print("AVATARPREP: nothing was written; --in NOT saved.")
        sys.exit(1)
    for n in args.mesh_names:
        c = changes[n]
        keys = ", ".join("%s %d" % kv for kv in sorted(c["keys"].items())) or "none"
        print("AVATARPREP: %s %s gates PASS; Basis moved %d vertices (max %.6f m); keys moved: "
              "%s; weights rewritten on %d vertices; custom normals %s"
              % (TOOL, n, c["moved"], c["max_move"], keys, c["reweighted"],
                 "changed" if c["normals_changed"] else "unchanged"))
    if args.whatif:
        print("AVATARPREP: %s %s => OK (whatif; nothing written)" % (TOOL, label))
        return

    core.apply_harvest(snaps)
    prefs = bpy.context.preferences.filepaths
    versions = prefs.save_version
    prefs.save_version = 0  # no .blend1 beside the garment blend, and none rotated away
    try:
        bpy.ops.wm.save_as_mainfile(filepath=in_path, compress=True)
    except Exception as e:
        print("AVATARPREP: ERROR failed to save --in:", e)
        sys.exit(2)
    finally:
        prefs.save_version = versions
    print("AVATARPREP: %s %s => OK | saved=%s" % (TOOL, label, in_path))


if __name__ == "__main__":
    run_cli(main, TOOL)
