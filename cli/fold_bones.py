"""Headless CLI: fold a doomed bone chain's weight into surviving neighbours.

Run (hand-authored map):
  blender --background --factory-startup --python cli/fold_bones.py -- \\
      --in <in.blend> --targets Hair,Hair_Extra --bones "Ribbon*" --map blend.json \\
      [--neighbours Bun,Head] [--report <json>] [--whatif | --out <out.blend> | --in-place]

Run (auto-suggest the map, then stop):
  blender --background --factory-startup --python cli/fold_bones.py -- \\
      --in <in.blend> --targets Hair --bones "Ribbon*" --map auto --map-out blend.json \\
      [--neighbours Bun,Head] [--report <json>]

``--targets`` names the mesh(es) the fold rewrites; the target armature is whichever
single ARMATURE-modifier object they share (there is no ``--armature`` — a fold with
targets bound to more than one armature is an ERROR, not a guess). ``--bones`` is a
comma-separated list of case-insensitive globs or exact names over that armature's
bones — the doomed set; a pattern matching nothing REFUSES (this is also what a rerun
on an already-folded output hits, since the doomed names are gone).

``--map auto --map-out PATH`` computes a suggested ``{doomed: {survivor: fraction}}``
table (nearest-two-candidate-bones inverse-distance split from each doomed bone's
weighted centroid), writes it to ``--map-out``, prints it with its ambiguity flag, and
ALWAYS exits 1 — auto only ever writes the table, it never runs the fold. Edit the file,
then pass the same path as ``--map`` to run it for real.

Exit 0 on a real run's ``=> OK``; 1 on ``=> FAIL:`` (a gate refused, or the deliberate
auto stop); 2 on unresolvable input (bad args, missing mesh/armature, a crash).
Gates, the fold math, and ``auto_map``'s algorithm: ``avatarprep/core/fold_bones.py``.
"""
import json
import os
import sys
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli, add_force_load_repair, write_report

TOOL = "fold_bones"


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: %s ? => FAIL: bad args: %s" % (TOOL, message))
        sys.exit(2)


def _csv(value):
    return [n.strip() for n in value.split(",") if n.strip()]


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = _Parser(prog="fold_bones")
    p.add_argument("--in", dest="in_path", required=True, help=".blend holding the targets")
    p.add_argument("--targets", required=True,
                   help="comma-separated mesh object(s) to fold; their shared ARMATURE-"
                        "modifier object is the target armature")
    p.add_argument("--bones", dest="bones", required=True,
                   help="comma-separated case-insensitive glob(s)/name(s) over the target "
                        "armature's bones; the doomed set")
    p.add_argument("--map", dest="map_arg", required=True,
                   help="path to a {doomed: {survivor: fraction}} JSON, or the literal 'auto'")
    p.add_argument("--map-out", dest="map_out", default=None,
                   help="write the auto-suggested table here (required with --map auto, "
                        "meaningless otherwise)")
    p.add_argument("--neighbours", dest="neighbours", default=None,
                   help="comma-separated candidate survivor bones for --map auto; default is "
                        "the armature's surviving deform bones not under a doomed bone")
    p.add_argument("--report", dest="report", default=None, help="JSON report path")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--whatif", action="store_true",
                      help="compute the fold in memory and report it; write nothing")
    mode.add_argument("--out", dest="out_path", default=None, help="save the folded result here")
    mode.add_argument("--in-place", dest="in_place", action="store_true",
                      help="save over --in")
    add_force_load_repair(p)
    a = p.parse_args(argv)

    a.target_names = _csv(a.targets)
    a.bone_patterns = _csv(a.bones)
    a.neighbour_names = _csv(a.neighbours) if a.neighbours else None

    if a.map_arg == "auto":
        if not a.map_out:
            p.error("--map auto requires --map-out PATH")
        if a.whatif or a.out_path or a.in_place:
            p.error("--map auto only ever writes the suggestion table and refuses to run; "
                    "--whatif/--out/--in-place do not apply")
    else:
        if a.map_out:
            p.error("--map-out is only meaningful with --map auto")
        if not (a.whatif or a.out_path or a.in_place):
            p.error("one of --whatif, --out, --in-place is required for a real run")
    return a


def _fail(reason, offenders=()):
    print("AVATARPREP: %s => FAIL: %s" % (TOOL, reason))
    for o in offenders:
        print("AVATARPREP: OFFENDER", o)
    sys.exit(1)


def _fail_refused(refused):
    _fail(str(refused), refused.offenders)


def _resolve_mesh(name):
    import bpy
    ob = bpy.context.scene.objects.get(name)
    if ob is None or ob.type != 'MESH':
        print("AVATARPREP: ERROR --targets %r is not a mesh object in this scene" % name)
        sys.exit(2)
    return ob


def _resolve_target_armature(meshes):
    arms = {}
    for m in meshes:
        found = None
        for mod in m.modifiers:
            if mod.type == 'ARMATURE' and mod.object is not None:
                found = mod.object
                break
        if found is None:
            print("AVATARPREP: ERROR target mesh %r has no ARMATURE modifier" % m.name)
            sys.exit(2)
        arms[found.name] = found
    if len(arms) != 1:
        print("AVATARPREP: ERROR --targets bind to %d distinct armature(s) (%s); fold_bones "
              "needs exactly one" % (len(arms), ", ".join(sorted(arms))))
        sys.exit(2)
    return next(iter(arms.values()))


def main():
    args = _parse_args()
    import bpy
    writes = args.map_arg != "auto" and not args.whatif
    open_blend(args.in_path, writes=writes, force_load_repair=args.force_load_repair)
    enable_avatarprep()
    from avatarprep.core import scene_utils
    from avatarprep.core import fold_bones as core

    targets = [_resolve_mesh(n) for n in args.target_names]
    armature = _resolve_target_armature(targets)

    try:
        doomed = core.resolve_doomed(armature, args.bone_patterns)
    except core.NoBonesMatched as e:
        print("AVATARPREP: ERROR", e)
        sys.exit(2)
    doomed_set = set(doomed)

    if args.map_arg == "auto":
        try:
            table, info = core.auto_map(armature, targets, doomed_set,
                                        neighbours=args.neighbour_names)
        except core.FoldRefused as refused:
            _fail_refused(refused)
            return
        except ValueError as e:
            print("AVATARPREP: ERROR", e)
            sys.exit(2)

        write_report(args.map_out, table)
        for bone in sorted(doomed_set):
            i = info.get(bone, {})
            if i.get("weightless"):
                print("AVATARPREP: %s auto %s weightless — no row written" % (TOOL, bone))
                continue
            flag = " AMBIGUOUS (ratio %.3f, within 20%% of 1)" % i["ratio"] if i["ambiguous"] else ""
            print("AVATARPREP: %s auto %s -> %s ratio=%s%s"
                  % (TOOL, bone, table.get(bone), i.get("ratio"), flag))
        if args.report:
            write_report(args.report, {"table": table, "info": info})
        _fail("auto writes the table; run it with --map <file> after editing")
        return

    try:
        with open(args.map_arg, "r", encoding="utf-8") as fh:
            bone_map = json.load(fh)
    except Exception as e:
        print("AVATARPREP: ERROR failed to read --map %r: %s" % (args.map_arg, e))
        sys.exit(2)
    if not isinstance(bone_map, dict):
        print("AVATARPREP: ERROR --map %r is not a JSON object" % args.map_arg)
        sys.exit(2)

    try:
        result = core.fold_bones(armature, targets, doomed_set, bone_map, whatif=args.whatif)
    except core.FoldRefused as refused:
        _fail_refused(refused)
        return
    except ValueError as e:
        print("AVATARPREP: ERROR", e)
        sys.exit(2)

    trailer = ""
    if args.report:
        write_report(args.report, result)
        trailer += " | report=%s" % os.path.abspath(args.report)

    if args.whatif:
        print("AVATARPREP: %s bones=%d touched=%d capped=%d => OK (whatif)%s"
              % (TOOL, len(doomed_set), result["touched"], result["capped"], trailer))
        return

    cmdline = "fold_bones " + " ".join(sys.argv[sys.argv.index("--") + 1:])
    for m in targets:
        scene_utils.write_stamp(m, scene_utils.STAMP_FOLDED, cmdline)

    out_path = os.path.abspath(args.in_path) if args.in_place else os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=out_path)
    except Exception as e:
        print("AVATARPREP: ERROR failed to save out:", e)
        sys.exit(2)
    trailer += " | saved=%s" % out_path

    print("AVATARPREP: %s bones=%d touched=%d capped=%d => OK%s"
          % (TOOL, len(result["bones_removed"]), result["touched"], result["capped"], trailer))


if __name__ == "__main__":
    run_cli(main, "fold_bones")
