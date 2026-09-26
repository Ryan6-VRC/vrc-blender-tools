"""Headless CLI: push the part of a garment the body pulls through under motion outward, by a fraction of a millimetre, into its rest shape.

Run:
  blender --background --factory-startup --python cli/push_garment.py -- \
      --in <costume.blend> --targets A,B --amount M (--sweep BONE[:AXIS][:MIN..MAX[:STEPS]] | --region KIND:NAME)... \
      [--source Body_Base] [--shape K=V | MESH:K=V]... [--cut-shape K]... [--cut-threshold-m 0.01] \
      [--near 0.0005] [--falloff 0.025] [--rim-hold 0] [--allow-static] [--reference-bone Hips] \
      [--forward BONE] [--report JSON] (--whatif | --out <out.blend> | --in-place) [--force-load-repair]

Seeds come from the same sweep ``report_fit`` runs (``--sweep`` bones, or the set derived
from the ``--region`` vertices); ``--region`` also limits the seeds to its vertices, several
regions adding together. A step's seeds are the garment vertices inside the posed body or
within ``--near`` metres of it, and the corners of triangles a body vertex within 5 mm comes
through; physbone and cut vertices never seed. Dynamic seeds are any step's seeds minus the
rest step's: contact already there at rest is the seat's or a shape's to fix, so a garment
with none refuses unless ``--allow-static``, which pushes the rest contact instead.

Every vertex within ``--falloff`` metres of a seed (rest, straight line) moves outward along
its rest normal, turned to face away from the nearest body triangle, by
``amount * smoothstep(1 - d / falloff)``; ``--rim-hold R`` multiplies that by
``smoothstep(e / R)``, e the distance to the nearest boundary vertex, so open edges stay on
the skin. The same object-space delta goes onto the mesh, Basis and every shape key.

A saving run stamps each target object with ``avatarprep_pushed``, the canonical command
line printed as ``recipe:`` (defaults omitted, no paths or mode flags); a stamped mesh
refuses, since a rerun would push twice: re-derive from the blend before the push.
``--whatif`` plans everything and saves nothing.

Exit 0 on ``=> OK``; 1 on ``=> FAIL:`` (ran and refused, nothing saved); 2 on unresolvable
input (bad args, a missing mesh) or a crash.
"""
import os
import sys
import argparse
import shlex

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli, add_force_load_repair, write_report
from cli import _fit

TOOL = "push_garment"
DEFAULTS = {"near": 0.0005, "falloff": 0.025, "rim_hold": 0.0}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: %s ? => FAIL: bad args: %s" % (TOOL, message))
        sys.exit(2)


def _parse_args(argv):
    p = _Parser(prog=TOOL)
    p.add_argument("--in", dest="in_path", required=True, help="costume .blend holding the targets and the body")
    p.add_argument("--targets", required=True, help="comma-separated garment meshes to push")
    p.add_argument("--amount", type=float, required=True, help="metres the seeds move outward")
    _fit.add_measure_args(p, sweep_help="sweep this body bone for seeds (repeatable). " + _fit.SWEEP_FORM)
    p.add_argument("--near", type=float, default=DEFAULTS["near"], help="metres outside the body that still seeds")
    p.add_argument("--falloff", type=float, default=DEFAULTS["falloff"], help="metres around a seed the push fades over")
    p.add_argument("--rim-hold", dest="rim_hold", type=float, default=DEFAULTS["rim_hold"],
                   help="metres from a boundary vertex over which the push fades in (0 holds nothing)")
    p.add_argument("--allow-static", dest="allow_static", action="store_true",
                   help="push the rest contact when the sweep adds none")
    p.add_argument("--report", dest="report_path", default=None, help="JSON report path")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--whatif", action="store_true", help="plan everything; save nothing")
    mode.add_argument("--out", dest="out_path", default=None, help="save the pushed costume here")
    mode.add_argument("--in-place", dest="in_place", action="store_true", help="save over --in")
    add_force_load_repair(p)
    a = p.parse_args(argv)
    enable_avatarprep()
    _fit.finish_measure_args(a, p.error)
    if not a.sweeps and not a.regions:
        p.error("name the motion with --sweep or the spot with --region; a push over every bone the garment "
                "touches would move contact nobody asked about")
    if not (a.amount > 0 and a.falloff > 0 and a.near >= 0 and a.rim_hold >= 0):
        p.error("--amount and --falloff must be positive, --near and --rim-hold not negative")
    if a.amount > a.falloff:
        p.error("--amount %g m is more than --falloff %g m; both are metres (0.0005 is half a millimetre)"
                % (a.amount, a.falloff))
    if a.out_path and os.path.abspath(a.out_path) == os.path.abspath(a.in_path):
        p.error("--out is --in itself; pass --in-place to save over it")
    return a


def recipe(a) -> str:
    out = [TOOL, "--targets", ",".join(a.target_list), "--amount", str(a.amount)] + _fit.recipe_measure(a)
    for flag, key in (("--near", "near"), ("--falloff", "falloff"), ("--rim-hold", "rim_hold")):
        if getattr(a, key) != DEFAULTS[key]:
            out += [flag, str(getattr(a, key))]
    if a.allow_static:
        out += ["--allow-static"]
    return shlex.join(out)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = _parse_args(argv)
    import bpy
    repair = open_blend(a.in_path, writes=not a.whatif, force_load_repair=a.force_load_repair)
    from avatarprep.core import fit, scene_utils
    source, targets = _fit.resolve_meshes(TOOL, a)
    label = "%s->%s" % (source.name, ",".join(t.name for t in targets))
    line = recipe(a)
    print("AVATARPREP: recipe: %s" % line)
    out = {"tool": TOOL, "recipe": line, "load_repair": repair, "measured_in": os.path.abspath(a.in_path)}

    def refuse(msg):
        print("AVATARPREP: %s %s => FAIL: %s" % (TOOL, label, msg))
        if a.report_path:
            out["error"] = msg
            write_report(a.report_path, out)
        sys.exit(1)

    for t in targets:
        stamp = scene_utils.read_stamp(t, scene_utils.STAMP_PUSHED)
        if stamp is not None:
            refuse("%s was already pushed by `%s`; a second push would move it twice. Re-derive from the blend "
                   "before that push, or leave it" % (t.name, stamp))
        if scene_utils.is_linked(t) or not scene_utils.is_editable(t):
            refuse("%s is linked or override data and cannot be written; push the local garment this "
                   "costume owns (own-mergeable makes one)" % t.name)
        if t.data.users > 1:
            refuse("%s shares its mesh with %d other user(s); make it single-user first" % (t.name, t.data.users - 1))

    try:
        data = fit.load(source, targets, shapes=a.shape_list, cut_shapes=a.cut_shapes,
                        cut_threshold=a.cut_threshold, reference_bone=a.reference_bone,
                        forward_bone=a.forward_bone, groups=fit.region_groups(a.regions))
        regions = fit.resolve_regions(data, a.regions)
        focus = None
        if regions:
            focus = []
            for i in range(len(targets)):
                m = None
                for masks in regions.values():
                    m = masks[i].copy() if m is None else (m | masks[i])
                focus.append(m)
        the_plan = fit.plan(data, a.sweep_list, focus)
        _fit.print_context(data, the_plan)
        plans = [fit.plan_push(data, i, the_plan, amount=a.amount, near=a.near, falloff=a.falloff,
                               rim_hold=a.rim_hold, allow_static=a.allow_static,
                               focus=None if focus is None else focus[i])
                 for i in range(len(targets))]
    except fit.FitError as e:
        refuse(str(e))

    out["sweep"] = the_plan["bones"]
    out["targets"] = []
    for t, pl in zip(targets, plans):
        busiest = sorted((s for s in pl["per_step"] if s["dynamic"]), key=lambda s: -s["dynamic"])[:3]
        print("AVATARPREP: %s seeds: rest %d, any step %d, dynamic %d; pushing %d %s seeds: moved %d (full %d), "
              "max %.3f mm, rim vertices moved %d, normals turned outward %d"
              % (t.name, pl["rest_seeds"], pl["posed_seeds"], pl["dynamic_seeds"], pl["seeds"], pl["kind"],
                 pl["moved"], pl["full"], pl["max_mm"], pl["rim_moved"], pl["turned"]))
        if busiest:
            print("AVATARPREP: %s most dynamic seeds at: %s" % (
                t.name, ", ".join("%s (%d)" % (s["step"], s["dynamic"]) for s in busiest)))
        out["targets"].append({"mesh": t.name, **{k: v for k, v in pl.items() if k not in ("delta", "seeded")}})

    saved = None
    if not a.whatif:
        try:
            for i, (t, pl) in enumerate(zip(targets, plans)):
                out["targets"][i]["shape_keys"] = fit.apply_push(t, pl["delta"])
                scene_utils.write_stamp(t, scene_utils.STAMP_PUSHED, line)
        except fit.FitError as e:
            refuse(str(e))
        saved = os.path.abspath(a.in_path) if a.in_place else os.path.abspath(a.out_path)
        os.makedirs(os.path.dirname(saved) or ".", exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=saved)
    out["saved"] = saved
    if a.report_path:
        write_report(a.report_path, out)
    trailer = " | report=%s" % os.path.abspath(a.report_path) if a.report_path else ""
    trailer += " | saved=%s" % saved if saved else " (whatif; nothing saved)"
    print("AVATARPREP: %s %s meshes=%d moved=%d max=%.3fmm => OK%s"
          % (TOOL, label, len(targets), sum(p["moved"] for p in plans), max(p["max_mm"] for p in plans), trailer))


if __name__ == "__main__":
    run_cli(main, TOOL)
