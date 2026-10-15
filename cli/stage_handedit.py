"""Headless CLI: stage a garment on a posable body for the operator's hand edit.

Run:
  blender --background --factory-startup --python cli/stage_handedit.py -- \\
      --body <body.blend> --garment <garment.blend> --meshes M[,M] --out <scene.blend> \\
      [--body-mesh NAME] [--poses <corpus.blend> [--bone-override OLD=NEW ...] \\
      [--absent-bones refuse|drop]] [--set MESH:KEY=VALUE ...]

Builds a fresh scene holding the body armature, one body mesh and the named garment
meshes, bound so the operator can pose the body and edit the garment on the posed
shape, and saves it compressed to ``--out``. ``harvest_handedit`` copies the edit back
into the garment blend afterwards. Reads its inputs and never writes them.

- **The body**: ``--body-mesh`` names the body mesh and its Armature modifier names
  the rig; without it the body blend must hold one local armature and one mesh
  deforming with it, or the door refuses naming the candidates.
- **The garment**: each ``--meshes`` entry is a local mesh in the garment blend with
  one Armature modifier, all on one rig. It is re-parented to the body rig keeping its
  world transform, and its Armature modifier is re-pointed at the body rig with
  ``show_in_editmode`` and ``show_on_cage`` on, so an edit-mode move lands on the posed
  shape. A garment rig of its own keeps its bones, not its object: every bone a garment
  vertex group weights that the body rig lacks is copied onto the body rig at its rest,
  with its garment-only ancestors. A weighted group naming a bone on neither rig refuses
  by name. The garment rig's bone constraints are not carried, and a modifier whose
  target the scene does not carry (a Mirror modifier's mirror object, say) keeps a
  dangling target; each is named on a WARNING line.
- **The row**: ``--set MESH:KEY=VALUE`` (repeatable) sets a shape-key value on the body
  mesh or a garment mesh, in Blender's 0..1 (a Unity weight of 20 is 0.2); pass every
  key the row's prefab sets, since a garment edited off its row is fixed against the
  wrong body. Shape-key actions on those meshes are cleared first, because a vendor
  FBX's action resets key values on every frame change. Each garment mesh's active key
  is parked on Basis, since an edit-mode move lands on the active key.
- **Poses**: ``--poses`` reads the corpus's one animated armature, only its own action
  slot, and only ``rotation_quaternion`` channels with all four components (anything
  else refuses by bone). Each timeline marker is one pose, frame 0 is the rest pose,
  and each rotation is rebuilt on the body's bone carrying its world-space rotation
  across both rests. ``--bone-override OLD=NEW`` maps a corpus bone onto the body's
  spelling; a corpus bone the body lacks refuses unless ``--absent-bones drop`` leaves
  it out, named on a ``DROPPED`` line, and one the body carries under another spelling
  (case, ``_``, ``.``) refuses even then. The body reproduces the corpus's world pose
  where the mapped bones' parent chains match; a mapped bone below a dropped one misses
  the dropped bone's rotation. Without ``--poses`` the scene opens at rest.

Bone locations are locked on every bone with a parent, and the pivot is individual
origins, so a multi-bone rotation does not pull bones out of their sockets. The door
refuses an existing ``--out``: that path is where the operator's unharvested edit lives.
A body or garment Blender repaired as it read it only warns here, since the scene is
scratch; the harvest refuses to write a repaired garment blend.

Exit 0 = staged (``--out`` saved) · 1 = REFUSED (named offenders; nothing written) ·
2 = ERROR (bad args, unopenable input, a crash). Core: ``avatarprep/core/handedit.py``.
"""
import os
import sys
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli, kv

TOOL = "stage_handedit"


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: %s ? => FAIL: bad args: %s" % (TOOL, message))
        sys.exit(2)


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = _Parser(prog=TOOL)
    p.add_argument("--body", required=True, help=".blend holding the body armature and mesh")
    p.add_argument("--garment", required=True, help=".blend holding the garment meshes")
    p.add_argument("--meshes", required=True, help="comma-separated garment mesh object(s)")
    p.add_argument("--out", dest="out_path", required=True,
                   help="where to save the edit scene; must not exist yet")
    p.add_argument("--body-mesh", dest="body_mesh", default=None,
                   help="the body mesh; its Armature modifier names the body rig. Needed when "
                        "the body blend holds more than one armature or bound mesh")
    p.add_argument("--poses", default=None,
                   help=".blend pose corpus: one armature whose action keys one pose per "
                        "timeline marker, in rotation_quaternion channels")
    p.add_argument("--bone-override", action="append", default=[], metavar="OLD=NEW",
                   help="map a pose-corpus bone onto the body's spelling. Repeatable")
    p.add_argument("--absent-bones", choices=("refuse", "drop"), default=None,
                   help="what a corpus bone the body lacks does: 'refuse' (default) or 'drop', "
                        "which leaves its rotation out and names it on a DROPPED line. A bone "
                        "the body carries under another spelling (case, '_', '.') still "
                        "refuses: map it with --bone-override. A mapped bone below a dropped "
                        "one keeps its own rotation but not the dropped bone's, so its world "
                        "pose differs from the corpus's")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="MESH:KEY=VALUE",
                   help="set a shape-key value (0..1) on the body mesh or a garment mesh. "
                        "Repeatable")
    a = p.parse_args(argv)
    a.mesh_names = [n.strip() for n in a.meshes.split(",") if n.strip()]
    if not a.mesh_names:
        p.error("--meshes names no mesh")
    if not a.poses and (a.bone_override or a.absent_bones):
        p.error("--bone-override and --absent-bones map pose-corpus bones and need --poses")
    a.absent_bones = a.absent_bones or "refuse"
    a.set_rows = []
    for item in a.sets:
        left, eq, value = item.rpartition("=")
        mesh, colon, key = left.partition(":")
        if not eq or not colon or not mesh or not key:
            p.error("--set wants MESH:KEY=VALUE, got %r" % item)
        try:
            a.set_rows.append((mesh, key, float(value)))
        except ValueError:
            p.error("--set %s: %r is not a number" % (left, value))
    return a


def _refused(label, refused):
    print("AVATARPREP: %s %s => FAIL: %s" % (TOOL, label, refused))
    for o in refused.offenders:
        print("AVATARPREP: OFFENDER", o)
    print("AVATARPREP: nothing was written; --out NOT saved.")
    sys.exit(1)


def main():
    args = _parse_args()
    import bpy
    label = ",".join(args.mesh_names)
    out_path = os.path.abspath(args.out_path)
    if os.path.exists(out_path):
        print("AVATARPREP: %s %s => FAIL: --out %s already exists; it may hold an edit not yet "
              "harvested — harvest or delete it, or name another path" % (TOOL, label, out_path))
        sys.exit(1)
    body_path = os.path.abspath(args.body)
    garment_path = os.path.abspath(args.garment)

    open_blend(body_path, writes=False)
    enable_avatarprep()
    from avatarprep.core import handedit as core
    try:
        body = core.read_body(args.body_mesh)
        print("AVATARPREP: %s body %s / %s (%s)" % (TOOL, body["armature"], body["mesh"],
                                                    "--body-mesh" if args.body_mesh else "discovered"))
        open_blend(garment_path, writes=False)
        garment = core.read_garment(args.mesh_names, body)
        core.check_sets(args.set_rows, body, garment)
        poses = None
        if args.poses:
            open_blend(os.path.abspath(args.poses), writes=False)
            poses = core.read_poses(body, kv(args.bone_override), args.absent_bones)
            print("AVATARPREP: %s poses rig %s slot %s: %d bone(s), %d pose(s)"
                  % (TOOL, poses["rig"], poses["slot"], len(poses["mapped"]), len(poses["markers"])))
            for old, new in sorted(poses["mapped"].items()):
                if old != new:
                    print("AVATARPREP: %s poses %s -> %s" % (TOOL, old, new))
            for b in poses["dropped"]:
                print("AVATARPREP: DROPPED pose bone %s: no bone on %s" % (b, body["armature"]))
        bpy.ops.wm.read_factory_settings(use_empty=True)
        result = core.build_scene(body_path, garment_path, body, garment, poses, args.set_rows)
    except core.HandEditRefused as refused:
        _refused(label, refused)
        return

    if result["copied_bones"]:
        print("AVATARPREP: %s copied %d garment bone(s) from %s onto %s: %s"
              % (TOOL, len(result["copied_bones"]), result["garment_rig"], result["armature"],
                 ", ".join(result["copied_bones"])))
    for mesh, action in result["cleared"]:
        print("AVATARPREP: %s cleared shape-key action %s on %s" % (TOOL, action, mesh))
    for mesh, key, value in args.set_rows:
        print("AVATARPREP: %s set %s:%s=%g" % (TOOL, mesh, key, value))
    for w in result["warnings"]:
        print("AVATARPREP: WARNING", w)
    if poses:
        for f, n in poses["markers"]:
            print("AVATARPREP: %s marker %d %s" % (TOOL, f, n))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=True)
    except Exception as e:
        print("AVATARPREP: ERROR failed to save --out:", e)
        sys.exit(2)
    print("AVATARPREP: %s %s rig=%s copied=%d poses=%d => OK | saved=%s"
          % (TOOL, label, result["armature"], len(result["copied_bones"]),
             len(poses["markers"]) if poses else 0, out_path))


if __name__ == "__main__":
    run_cli(main, TOOL)
