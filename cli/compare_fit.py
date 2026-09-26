"""Headless CLI: compare how skinned garments fit across labelled inputs, per metric and region; a Compare, never a verdict.

Run:
  blender --background --factory-startup --python cli/compare_fit.py -- \
      --in A.blend --in B.blend [--label A --label B] <report_fit's flags after --in>
  blender --background --factory-startup --python cli/compare_fit.py -- \
      --in A.blend --simulate-transfer [--label A] <report_fit's flags after --in>

Every input is measured exactly as ``report_fit`` measures it, over the same steps: named
``--sweep`` bones, or else the union of every input's derived sweep set. The first input is
the baseline; each line gives every input's worst step per metric and region and its
difference from the baseline. ``--simulate-transfer`` measures one input twice: as it is,
and with each garment carrying the body's weights interpolated at its nearest body points
(``transfer_weights``' write rule, garment bones kept; ``avatarprep/core/fit.py``
``simulated``). How the result is read is the weightpaint skill's. ``--render DIR`` writes
one sheet per compared step (each input's worst stretch, new-penetration and body-through
steps), the inputs side by side in ``--in`` order, at most ``MAX_RENDER_STEPS`` (4) sheets
per garment and region; a line names the steps past the cap, which get no sheet. Inputs are
read and never saved.

``=> OK`` means measured. Exit 0 on OK; 1 on ``=> FAIL:`` (a refusal in any input, inputs
that sweep different steps, a failed render); 2 on unresolvable input or a crash. Metric
meanings: the printed legend (``avatarprep/core/fit.py`` ``LEGEND``).
"""
import os
import sys
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli, write_report
from cli import _fit
from cli import report_fit

TOOL = "compare_fit"
MAX_RENDER_STEPS = 4


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: %s ? => FAIL: bad args: %s" % (TOOL, message))
        sys.exit(2)


def _parse_args(argv):
    p = _Parser(prog=TOOL)
    p.add_argument("--in", dest="in_paths", action="append", required=True,
                   help="a costume .blend to measure (repeatable; the first is the baseline); never saved")
    p.add_argument("--label", dest="labels", action="append", default=[],
                   help="names the matching --in (repeatable, in --in order; default: each blend's stem)")
    p.add_argument("--simulate-transfer", dest="simulate", action="store_true",
                   help="with one --in: compare it against itself carrying the body's interpolated weights")
    report_fit.add_args(p)
    a = p.parse_args(argv)
    if a.simulate and len(a.in_paths) != 1:
        p.error("--simulate-transfer compares one --in against its simulated transfer; got %d" % len(a.in_paths))
    if not a.simulate and len(a.in_paths) < 2:
        p.error("compare_fit wants two or more --in, or one with --simulate-transfer")
    if a.labels and len(a.labels) != len(a.in_paths):
        p.error("%d --label for %d --in; give one per --in or none" % (len(a.labels), len(a.in_paths)))
    a.label_list = a.labels or [os.path.splitext(os.path.basename(x))[0] for x in a.in_paths]
    if len(set(a.label_list)) != len(a.label_list):
        p.error("labels must differ (%s); pass --label" % ", ".join(a.label_list))
    enable_avatarprep()
    _fit.finish_measure_args(a, p.error)
    return a


def _fail(label, msg, a, out):
    print("AVATARPREP: %s %s => FAIL: %s" % (TOOL, label, msg))
    if a.report_path:
        out["error"] = msg
        write_report(a.report_path, out)
    sys.exit(1)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = _parse_args(argv)
    from avatarprep.core import fit
    run_label = " vs ".join(a.label_list + (["simulated"] if a.simulate else []))
    out = {"tool": TOOL, "legend": list(fit.LEGEND), "inputs": []}

    inputs = []  # (label, path, data)
    for label, path in zip(a.label_list, a.in_paths):
        repair = open_blend(path, writes=False)
        source, targets = _fit.resolve_meshes(TOOL, a)
        try:
            data = fit.load(source, targets, shapes=a.shape_list, cut_shapes=a.cut_shapes,
                            cut_threshold=a.cut_threshold, reference_bone=a.reference_bone,
                            forward_bone=a.forward_bone, groups=fit.region_groups(a.regions))
            if a.simulate:
                sim = fit.simulated(data)
        except fit.FitError as e:
            _fail(label, str(e), a, out)
        for g in data["garments"]:
            g.pop("object", None)  # the next --in replaces the file; hold arrays only
        inputs.append((label, path, data, repair))
        if a.simulate:
            for g in sim["garments"]:
                g.pop("object", None)
            inputs.append((label + ":simulated", path, sim, repair))

    first, bones = [], None
    for label, path, data, repair in inputs:
        try:
            first.append(report_fit.plan_for(a, fit, data))
        except fit.FitError as e:
            _fail(label, str(e), a, out)
    if a.sweep_list is None:
        bones = []
        for _, the_plan, _ in first:
            bones += [b["bone"] for b in the_plan["bones"] if b["bone"] not in bones]
    planned = []
    for (label, path, data, repair), got in zip(inputs, first):
        if bones is not None and [b["bone"] for b in got[1]["bones"]] != bones:
            try:
                got = report_fit.plan_for(a, fit, data, bones=bones)
            except fit.FitError as e:
                _fail(label, "%s (another input's derived sweep set holds it)" % e, a, out)
        planned.append((label, path, data, repair) + tuple(got))
    names = [[s["name"] for s in p[5]["steps"]] for p in planned]
    if any(n != names[0] for n in names[1:]):
        _fail(run_label, "the inputs sweep different steps (their rigs differ), so their numbers do not "
                         "compare step for step; name the bones with --sweep", a, out)

    for label, path, data, repair, regions, the_plan, sets in planned:
        _fit.print_context(data, the_plan, prefix="%s: " % label)
    if a.whatif:
        secs = sum(fit.estimate(p[2], p[5], p[6]) for p in planned)
        print("AVATARPREP: %s %s inputs=%d steps=%d estimate=%.0fs => OK (whatif; one step timed per input, "
              "nothing else measured)" % (TOOL, run_label, len(planned), len(names[0]), secs))
        if a.report_path:
            out["estimate_seconds"] = round(secs, 1)
            write_report(a.report_path, out)
        return

    results = []
    for label, path, data, repair, regions, the_plan, sets in planned:
        r = fit.sweep(data, the_plan, sets)
        results.append(r)
        out["inputs"].append({"label": label, "measured_in": os.path.abspath(path), "load_repair": repair,
                              "frame": _fit.frame_json(data["frame"]), "sweep": the_plan["bones"],
                              **{k: r[k] for k in ("rest", "steps", "worst", "seconds")}})
    _fit.print_legend()

    labels = [p[0] for p in planned]
    base = planned[0]
    deltas = {}
    for gi, g in enumerate(base[2]["garments"]):
        deltas[g["name"]] = {}
        for rl in base[6][gi]:
            d = deltas[g["name"]][rl] = {}
            rests = [r["rest"][g["name"]]["regions"][rl] for r in results]
            for key, words in (("pen", "rest skin-tight deeper than 1 mm"), ("edge_pen", "rest edge deeper than 1 mm")):
                vals = [x[key] for x in rests]
                d["rest_" + key] = {"values": dict(zip(labels, vals)),
                                    "delta": {lb: v - vals[0] for lb, v in zip(labels[1:], vals[1:])}}
                print("AVATARPREP: %s [%s] %s: %s" % (g["name"], rl, words, _line(labels, vals, None)))
            for metric, _ in fit.WORST_KEYS:
                ws = [r["worst"][g["name"]][rl][metric] for r in results]
                vals = [_fit.metric_value(metric, w) for w in ws]
                d[metric] = {"values": dict(zip(labels, vals)), "steps": dict(zip(labels, [w and w["step"] for w in ws])),
                             "delta": {lb: (None if v is None or vals[0] is None else round(v - vals[0], 3))
                                       for lb, v in zip(labels[1:], vals[1:])}}
                print("AVATARPREP: %s [%s] %s: %s" % (g["name"], rl, metric.replace("_", "-"), _line(labels, vals, ws)))
    out["deltas"] = deltas

    renders, failed = [], False
    if a.render_dir:
        from avatarprep.core import render_mesh
        rdir = os.path.abspath(a.render_dir)
        for gi, g in enumerate(base[2]["garments"]):
            for rl in (list(base[4]) or ["all"]):
                steps = []
                for r in results:
                    for rl2, metric, step in report_fit.render_steps(fit, r, {rl: None} if rl != "all" else {},
                                                                      g["name"]):
                        if step not in steps:
                            steps.append(step)
                if len(steps) > MAX_RENDER_STEPS:
                    print("AVATARPREP: render %s [%s] skipped %d step(s) past the %d-sheet cap: %s"
                          % (g["name"], rl, len(steps) - MAX_RENDER_STEPS, MAX_RENDER_STEPS,
                             ", ".join(steps[MAX_RENDER_STEPS:])))
                for step in steps[:MAX_RENDER_STEPS]:
                    pngs = []
                    for label, path, data, repair, regions, the_plan, sets in planned:
                        st = {s["name"]: s for s in the_plan["steps"]}[step]
                        line = fit.render_step(data, gi, st, rdir, "cmp_%s" % label, focus=sets[gi][rl])
                        if "=> OK" not in line:
                            print("AVATARPREP: render %s [%s] %s @ %s: %s" % (g["name"], rl, label, step, line))
                            failed = True
                            break
                        pngs.append(fit.png_of(line))
                    if len(pngs) != len(planned):
                        continue
                    target = os.path.join(rdir, "fitcompare_%s_%s_%s.png" % (
                        render_mesh._sanitize("_".join(labels)), render_mesh._sanitize(g["name"]),
                        render_mesh._sanitize("%s_%s" % (rl, step))))
                    render_mesh.stitch(pngs, target)
                    for p in pngs:
                        os.remove(p)
                    renders.append({"garment": g["name"], "region": rl, "step": step, "columns": labels, "png": target})
                    print("AVATARPREP: render %s [%s] @ %s, columns %s: %s" % (g["name"], rl, step, ", ".join(labels), target))
    out["renders"] = renders
    if a.report_path:
        write_report(a.report_path, out)
    trailer = " | report=%s" % os.path.abspath(a.report_path) if a.report_path else ""
    if a.render_dir:
        trailer += " | renders=%d in %s" % (len(renders), os.path.abspath(a.render_dir))
    if failed:
        print("AVATARPREP: %s %s => FAIL: a render failed (its line is above); every number above was measured%s"
              % (TOOL, run_label, trailer))
        sys.exit(1)
    print("AVATARPREP: %s %s inputs=%d steps=%d seconds=%.1f => OK%s"
          % (TOOL, run_label, len(planned), len(names[0]), sum(r["seconds"] for r in results), trailer))


def _line(labels, vals, worsts):
    parts = []
    for i, (lb, v) in enumerate(zip(labels, vals)):
        txt = "-" if v is None else ("%g" % v)
        if worsts is not None and worsts[i] is not None and v:
            txt += " @ %s" % worsts[i]["step"]
        if i and v is not None and vals[0] is not None:
            txt += " (%+g)" % round(v - vals[0], 3)
        parts.append("%s %s" % (lb, txt))
    return " | ".join(parts)


if __name__ == "__main__":
    run_cli(main, TOOL)
