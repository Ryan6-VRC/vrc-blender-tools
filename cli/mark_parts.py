"""Headless CLI: author a Delete or Hide carrier key that moves a selected part of a garment rigidly out of sight.

Run:
  blender --background --factory-startup --python cli/mark_parts.py -- \
      --in <costume.blend> --targets MESH[,MESH] --key NAME --region KIND:NAME... \
      [--islands any|half|all] [--to BONE|x,y,z] [--threshold 0.01] \
      [--reference-bone Hips] [--forward BONE] [--replace] \
      (--whatif | --out <out.blend>) [--force-load-repair]

``--region`` is the fit family's grammar, read on each target's Basis: ``group:NAME`` (weight
over 0.5, summed over the groups a glob matches), ``bone:NAME`` (largest influence on a bone
the glob matches or one under it, over the deform bones of the mesh's armature),
``slot:MATERIAL`` (every polygon touching the vertex in a slot whose material the glob
matches), ``zone:front|back|left|right|midline:M|above:BONE|below:BONE`` (the armature's body
frame from ``--reference-bone``'s head; ``midline:M`` is within M metres either side of the
lateral midline). Terms joined by ``+`` intersect; repeated ``--region`` flags add together.
Names are case-sensitive globs.

``--islands`` takes each edge-linked island whole when ``any``, at least ``half`` (the
default) or ``all`` of its vertices are in the regions, and nothing outside a taken island
moves. Each taken island moves rigidly, its centroid to the head of the deform bone carrying
its largest summed weight, or to ``--to`` (a bone's head, or a world point ``x,y,z``) for every
island. An island moving ``--threshold`` metres or less refuses: Modular Avatar's ShapeChanger
keeps a vertex its shape moves by its threshold or less (0.01, ``VertexFilterByShape``'s
default; pass the component's when it differs).

``--whatif`` prints every island taken, the vertex count and the smallest and largest move,
and writes nothing. A saving run adds the key at value 0, appended last in each target's key
order (``--replace`` rewrites a same-named key where it stands and refuses one another key is
relative to), stamps ``avatarprep_authored`` ``{key: recipe}`` on each target with the
canonical command line printed as ``recipe:`` (defaults omitted, no paths or mode flags),
leaves Object Mode with the active key on Basis, and saves to ``--out`` (which may be ``--in``).

Refuses (exit 1, nothing saved): zero vertices selected, a linked or override target, a
target sharing its mesh, a posed armature, an island under the threshold, an existing key
without ``--replace``. Exit 2 on bad args, a mesh not in the file, or a crash. Core:
``avatarprep/core/parts.py``; the key writer is ``coverage.add_key``, shared with
``mark_coverage``.
"""
import os
import sys
import argparse
import shlex

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli, add_force_load_repair

TOOL = "mark_parts"
SHOWN = 12


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: %s ? => FAIL: bad args: %s" % (TOOL, message))
        sys.exit(2)


def _parse_to(text, error):
    """A bone name, or a world point when ``text`` is three comma-separated numbers."""
    parts = text.split(",")
    if len(parts) == 3:
        try:
            return tuple(float(x) for x in parts)
        except ValueError:
            pass
    if "," in text:
        error("--to %r is neither a bone name nor a world point x,y,z" % text)
    return text


def _parse_args(argv):
    p = _Parser(prog=TOOL)
    p.add_argument("--in", dest="in_path", required=True, help="costume .blend holding the targets")
    p.add_argument("--targets", required=True, help="comma-separated garment meshes to author the key on")
    p.add_argument("--key", required=True, help="the shape key to author, one per run")
    p.add_argument("--region", dest="regions", action="append", default=[], metavar="KIND:NAME",
                   help="group:NAME, bone:NAME, slot:MATERIAL or zone:front|back|left|right|midline:M|"
                        "above:BONE|below:BONE; names are globs; + intersects, repeats add (repeatable)")
    p.add_argument("--islands", choices=("any", "half", "all"), default="half",
                   help="take an edge-linked island whole when any, at least half, or all of its "
                        "vertices are in the regions")
    p.add_argument("--to", dest="to", default=None, metavar="BONE|x,y,z",
                   help="move every island's centroid to this bone's head or world point, instead of "
                        "each island's dominant bone head")
    p.add_argument("--threshold", type=float, default=0.01,
                   help="the Delete threshold in metres; an island moving this far or less refuses")
    p.add_argument("--reference-bone", dest="reference_bone", default="Hips",
                   help="bone whose head is the zone: frame's origin")
    p.add_argument("--forward", dest="forward_bone", default=None, metavar="BONE",
                   help="read the zone: frame's forward from BONE's tail minus head instead of the feet")
    p.add_argument("--replace", action="store_true", help="rewrite a same-named key where it stands")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--whatif", action="store_true", help="select and plan; save nothing")
    mode.add_argument("--out", dest="out_path", default=None, help="save the marked costume here (may be --in)")
    add_force_load_repair(p)
    a = p.parse_args(argv)
    a.target_list = [n.strip() for n in a.targets.split(",") if n.strip()]
    if not a.target_list:
        p.error("--targets named nothing")
    if not a.regions:
        p.error("name the part with --region")
    enable_avatarprep()
    from avatarprep.core import fit
    for r in a.regions:
        try:
            fit.parse_region(r)
        except ValueError as e:
            p.error(str(e))
    if not a.threshold > 0:
        p.error("--threshold must be positive")
    a.to_value = _parse_to(a.to, p.error) if a.to is not None else None
    return a


def recipe(a) -> str:
    out = [TOOL, "--targets", ",".join(a.target_list), "--key", a.key]
    for r in a.regions:
        out += ["--region", r]
    if a.islands != "half":
        out += ["--islands", a.islands]
    if a.to is not None:
        out += ["--to", a.to]
    if a.threshold != 0.01:
        out += ["--threshold", str(a.threshold)]
    if a.reference_bone != "Hips":
        out += ["--reference-bone", a.reference_bone]
    if a.forward_bone:
        out += ["--forward", a.forward_bone]
    return shlex.join(out)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = _parse_args(argv)
    import bpy
    open_blend(a.in_path, writes=not a.whatif, force_load_repair=a.force_load_repair)
    from avatarprep.core import parts
    parts.object_mode()
    label = "%s %s" % (a.key, ",".join(a.target_list))
    line = recipe(a)
    print("AVATARPREP: recipe: %s" % line)

    def refuse(msg):
        print("AVATARPREP: %s %s => FAIL: %s" % (TOOL, label, msg))
        sys.exit(1)

    targets = []
    for n in a.target_list:
        ob = bpy.context.scene.objects.get(n)
        if ob is None or ob.type != 'MESH':
            print("AVATARPREP: %s %s => FAIL: --targets %r is not a mesh in this scene" % (TOOL, label, n))
            sys.exit(2)
        targets.append(ob)

    plans = []
    try:
        for ob in targets:
            parts.check_target(ob)
            sel = parts.select(ob, a.regions, islands=a.islands, reference_bone=a.reference_bone,
                               forward_bone=a.forward_bone)
            plans.append((ob, sel, parts.plan(ob, sel, to=a.to_value, threshold=a.threshold)))
    except parts.PartsError as e:
        refuse(str(e))

    for ob, sel, pl in plans:
        for r in pl["islands"][:SHOWN]:
            print("AVATARPREP: %s island at vertex %d: %d vertices, dominant %s, to %s, moves %.4f m"
                  % (ob.name, r["island"], r["vertices"], r["dominant"] or "-", r["target"], r["delta_m"]))
        if len(pl["islands"]) > SHOWN:
            print("AVATARPREP: %s ... %d more islands taken" % (ob.name, len(pl["islands"]) - SHOWN))
        left = len(sel["islands"]) - len(sel["taken"])
        print("AVATARPREP: %s islands taken %d (%s), %d touched by the regions but left; vertices %d of %d "
              "in the regions, %d moved; move %.4f..%.4f m"
              % (ob.name, len(sel["taken"]), a.islands, left, int(sel["qualifying"].sum()),
                 len(ob.data.vertices), pl["vertices"], pl["min_delta"], pl["max_delta"]))

    saved = None
    if not a.whatif:
        try:
            for ob, _, pl in plans:
                parts.write(ob, a.key, pl, recipe=line, replace=a.replace)
        except parts.PartsError as e:
            refuse(str(e))
        parts.object_mode()
        saved = os.path.abspath(a.out_path)
        os.makedirs(os.path.dirname(saved) or ".", exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=saved)
    trailer = " | saved=%s" % saved if saved else " (whatif; nothing saved)"
    print("AVATARPREP: %s %s targets=%d islands=%d vertices=%d move=%.4f..%.4fm => OK%s"
          % (TOOL, label, len(plans), sum(len(p["islands"]) for _, _, p in plans),
             sum(p["vertices"] for _, _, p in plans), min(p["min_delta"] for _, _, p in plans),
             max(p["max_delta"] for _, _, p in plans), trailer))


if __name__ == "__main__":
    run_cli(main, TOOL)
