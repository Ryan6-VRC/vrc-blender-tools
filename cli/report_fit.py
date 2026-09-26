"""Headless CLI: report how skinned garments fit their body through a sweep of bone turns; a Report, never a verdict.

Run:
  blender --background --factory-startup --python cli/report_fit.py -- \
      --in <costume.blend> --targets A,B [--source Body_Base] [--shape K=V | MESH:K=V]... \
      [--cut-shape K]... [--cut-threshold-m 0.01] [--sweep BONE[:AXIS][:MIN..MAX[:STEPS]]]... \
      [--region group:NAME | bone:NAME | zone:NAME]... [--reference-bone Hips] [--forward BONE] \
      [--label L] [--render DIR] [--report JSON] [--whatif]

Reads ``--in`` and never saves it. Prints the body frame and each garment's extents in it
(the numbers ``transfer_weights --exclude-blend`` takes), the classes, the swept bones with
the joint-table row each took, the legend, rest context, and per garment and region the
worst step of every metric. ``--region`` names the spot: without ``--sweep`` the sweep set
is derived from the regions' vertices, and each region gets its own rows beside ``all``.
``--whatif`` prints the step count and a time estimate from one timed step, then stops.
``--render DIR`` writes a close-up sheet per garment (per region when given) at its worst
stretch, new-penetration and body-through steps.

``=> OK`` means measured. Exit 0 on OK; 1 on ``=> FAIL:`` (a refusal such as a bone that
moves no skin under the garment or an empty region, or a failed render); 2 on unresolvable
input (bad args, a missing mesh) or a crash. Metric meanings: the printed legend
(``avatarprep/core/fit.py`` ``LEGEND``).
"""
import os
import sys
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli, write_report
from cli import _fit

TOOL = "report_fit"
RENDER_METRICS = ("stretch", "new_pen", "body_through")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: %s ? => FAIL: bad args: %s" % (TOOL, message))
        sys.exit(2)


def add_args(p):
    """``report_fit``'s flags after ``--in``; ``compare_fit`` takes them verbatim."""
    p.add_argument("--targets", required=True, help="comma-separated garment meshes to measure")
    _fit.add_measure_args(p)
    p.add_argument("--render", dest="render_dir", default=None, help="write close-up sheets into this directory")
    p.add_argument("--report", dest="report_path", default=None, help="JSON report path (every step's numbers)")
    p.add_argument("--whatif", action="store_true", help="print the step count and a time estimate; measure nothing else")


def _parse_args(argv):
    p = _Parser(prog=TOOL)
    p.add_argument("--in", dest="in_path", required=True, help="costume .blend holding the garments and the body; never saved")
    p.add_argument("--label", default=None, help="names this run in the result line and render files (default: the blend's stem)")
    add_args(p)
    a = p.parse_args(argv)
    enable_avatarprep()
    _fit.finish_measure_args(a, p.error)
    return a


def plan_for(a, fit, data, bones=None):
    """``(regions, plan, region sets)`` for loaded ``data``; refusals raise ``FitError``. Without
    ``--sweep`` the sweep set comes from the regions' vertices, or every vertex."""
    regions = fit.resolve_regions(data, a.regions)
    focus = None
    if regions and a.sweep_list is None:
        focus = [_union([m[i] for m in regions.values()]) for i in range(len(data["garments"]))]
    the_plan = fit.plan(data, a.sweep_list, focus, bones=bones)
    return regions, the_plan, fit.region_sets(data, regions)


def _union(masks):
    out = masks[0].copy()
    for m in masks[1:]:
        out |= m
    return out


def print_worst(fit, gname, label, w, rest):
    r = rest["regions"][label]
    print("AVATARPREP: %s [%s] rest: skin-tight deeper than 1 mm %d, edge within 10 mm deeper than 1 mm %d"
          % (gname, label, r["pen"], r["edge_pen"]))
    for metric, _ in fit.WORST_KEYS:
        print("AVATARPREP: %s [%s] %s" % (gname, label, _fit.fmt_metric(metric, w[metric])))


def render_steps(fit, result, regions, garment):
    """``[(region, metric, step name)]`` a garment's sheets are drawn at: its worst stretch step
    and, where any, its worst new-penetration and body-through steps, per region (``all`` when
    none was named), each step once."""
    out = []
    for rl in (list(regions) or ["all"]):
        done = set()
        for metric in RENDER_METRICS:
            v = result["worst"][garment][rl][metric]
            if v is None or v["step"] in done or (metric != "stretch" and not v["count"]):
                continue
            done.add(v["step"])
            out.append((rl, metric, v["step"]))
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = _parse_args(argv)
    repair = open_blend(a.in_path, writes=False)
    from avatarprep.core import fit
    source, targets = _fit.resolve_meshes(TOOL, a)
    label = a.label or os.path.splitext(os.path.basename(a.in_path))[0]
    out = {"tool": TOOL, "label": label, "measured_in": os.path.abspath(a.in_path), "load_repair": repair,
           "legend": list(fit.LEGEND)}
    try:
        data = fit.load(source, targets, shapes=a.shape_list, cut_shapes=a.cut_shapes,
                        cut_threshold=a.cut_threshold, reference_bone=a.reference_bone,
                        forward_bone=a.forward_bone, groups=fit.region_groups(a.regions))
        regions, the_plan, sets = plan_for(a, fit, data)
    except fit.FitError as e:
        print("AVATARPREP: %s %s => FAIL: %s" % (TOOL, label, e))
        if a.report_path:
            out["error"] = str(e)
            write_report(a.report_path, out)
        sys.exit(1)
    _fit.print_context(data, the_plan)
    out.update({"frame": _fit.frame_json(data["frame"]), "sweep": the_plan["bones"],
                "garments": {g["name"]: {"extents": fit.extents(data, g),
                                         "classes": {k: int(v.sum()) for k, v in g["cls"].items()}}
                             for g in data["garments"]}})
    if a.whatif:
        secs = fit.estimate(data, the_plan, sets)
        print("AVATARPREP: %s %s steps=%d estimate=%.0fs => OK (whatif; one step timed, nothing else measured)"
              % (TOOL, label, len(the_plan["steps"]), secs))
        if a.report_path:
            out["estimate_seconds"] = round(secs, 1)
            write_report(a.report_path, out)
        return

    result = fit.sweep(data, the_plan, sets)
    _fit.print_legend()
    for gi, g in enumerate(data["garments"]):
        for rl in sets[gi]:
            print_worst(fit, g["name"], rl, result["worst"][g["name"]][rl], result["rest"][g["name"]])
        sx = result["worst"][g["name"]]["all"]["self_x"]
        print("AVATARPREP: %s body self-x: rest %d, worst %d @ %s"
              % (g["name"], result["rest"][g["name"]]["self_x"], sx["count"] if sx else 0,
                 sx["step"] if sx else "-"))
    out.update({k: result[k] for k in ("rest", "steps", "worst", "seconds")})

    renders, failed = [], False
    if a.render_dir:
        by_name = {s["name"]: s for s in the_plan["steps"]}
        for gi, g in enumerate(data["garments"]):
            for rl, metric, step in render_steps(fit, result, regions, g["name"]):
                line = fit.render_step(data, gi, by_name[step], os.path.abspath(a.render_dir),
                                       "%s_%s_%s" % (label, g["name"], metric), focus=sets[gi][rl])
                ok = "=> OK" in line
                failed |= not ok
                renders.append({"garment": g["name"], "region": rl, "metric": metric, "step": step,
                                "png": fit.png_of(line), "line": line})
                print("AVATARPREP: render %s [%s] %s @ %s: %s"
                      % (g["name"], rl, metric, step, fit.png_of(line) if ok else line))
    out["renders"] = renders
    if a.report_path:
        write_report(a.report_path, out)
    trailer = " | report=%s" % os.path.abspath(a.report_path) if a.report_path else ""
    if a.render_dir:
        trailer += " | renders=%d in %s" % (len(renders), os.path.abspath(a.render_dir))
    if failed:
        print("AVATARPREP: %s %s steps=%d => FAIL: a render failed (its line is above); every number above "
              "was measured%s" % (TOOL, label, len(the_plan["steps"]), trailer))
        sys.exit(1)
    print("AVATARPREP: %s %s steps=%d seconds=%.1f => OK%s"
          % (TOOL, label, len(the_plan["steps"]), result["seconds"], trailer))


if __name__ == "__main__":
    run_cli(main, TOOL)
