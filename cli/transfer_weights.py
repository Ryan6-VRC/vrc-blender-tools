"""Headless CLI: transfer the body's skin weights onto garments, every garment bone's weight kept exactly.

Run:
  blender --background --factory-startup --python cli/transfer_weights.py -- \
      --in <costume.blend> --targets A,B [--source Body_Base] [--shape K=V | MESH:K=V]... \
      [--mask G] [--max-distance 0.05] [--normal-angle 30] [--no-flip] [--smooth N[,F]] \
      [--source-exclude GLOB[,GLOB] --source-exclude-max W] [--exclude-blend AXIS,CENTER,WIDTH] \
      [--exclude-blend-lateral CENTER,WIDTH] [--exclude-blend-smooth N[,ALPHA[,RADIUS]]] \
      [--allow-no-loops] [--allow-unseated] [--reference-bone Hips] [--forward BONE] \
      [--viz <review.blend>] [--report <json>] (--whatif | --out <out.blend> | --in-place) \
      [--force-load-repair]

Runs on the costume blend, where the body is normally a library link and is never written.
``--whatif`` runs the whole transfer in memory, asserts included, and saves nothing. A
saving run stamps each target object with ``avatarprep_weights``, the canonical
command line printed as ``recipe:`` (flags in the order above, defaults omitted, no paths
or mode flags), which reproduces the same weights on a rerun. ``--viz`` saves a copy whose
targets carry an ``avatarprep_matched`` colour layer (white matched, magenta inpainted) for
``render_mesh --shading vertexcolor``; the saved deliverable never carries it.

Axes are words: ``forward`` (the body's facing, read from the feet), ``back``, ``lateral``
(the body's left, the ``.L`` side), ``up``, measured from ``--reference-bone``'s head.

Exit 0 on ``=> OK``; 1 on ``=> FAIL:`` (ran and refused, nothing saved); 2 on unresolvable
input (bad args, missing mesh, missing dependencies) or a crash. Steps, defaults and every
refusal: ``avatarprep/core/weight_transfer.py``.
"""
import os
import sys
import argparse
import shlex

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import (enable_avatarprep, open_blend, run_cli, add_force_load_repair,
                         write_report, ensure_deps)

TOOL = "transfer_weights"
DEFAULTS = {"source": "Body_Base", "max_distance": 0.05, "normal_angle": 30.0,
            "reference_bone": "Hips"}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: %s ? => FAIL: bad args: %s" % (TOOL, message))
        sys.exit(2)


def _floats(text, n_min, n_max, flag, p, form):
    parts = text.split(",")
    if not n_min <= len(parts) <= n_max:
        p.error("%s wants %s, got %r" % (flag, form, text))
    try:
        return [float(x) for x in parts]
    except ValueError:
        p.error("%s wants %s, got %r" % (flag, form, text))


def _parse_args(argv):
    p = _Parser(prog=TOOL)
    p.add_argument("--in", dest="in_path", required=True, help="costume .blend holding the targets and the body")
    p.add_argument("--targets", required=True, help="comma-separated garment meshes to reweight")
    p.add_argument("--source", default=DEFAULTS["source"], help="body mesh the weights come from (linked is fine; never written)")
    p.add_argument("--shape", dest="shapes", action="append", default=[], metavar="K=V|MESH:K=V",
                   help="a shape value the row wears, set for the transfer only; bare K lands on every "
                        "listed mesh carrying it (repeatable)")
    p.add_argument("--mask", default=None, help="write only vertices weighted over 0.5 in this group")
    p.add_argument("--max-distance", dest="max_distance", type=float, default=DEFAULTS["max_distance"],
                   help="metres; a farther target vertex is inpainted, not matched")
    p.add_argument("--normal-angle", dest="normal_angle", type=float, default=DEFAULTS["normal_angle"],
                   help="degrees between source and target normals for a match")
    p.add_argument("--no-flip", dest="flip", action="store_false",
                   help="refuse a match whose normal faces the source head-on (a band where two limbs touch)")
    p.add_argument("--smooth", default=None, metavar="N[,F]",
                   help="N smoothing passes at factor F (default 0.2) near inpainted vertices")
    p.add_argument("--source-exclude", dest="source_exclude", default=None, metavar="GLOB[,GLOB]",
                   help="source deform bones (with descendants) whose skin no match may draw from")
    p.add_argument("--source-exclude-max", dest="source_exclude_max", type=float, default=None, metavar="W",
                   help="required with --source-exclude: a source vertex over this share on those bones is dropped")
    p.add_argument("--exclude-blend", dest="exclude_blend", default=None, metavar="AXIS,CENTER,WIDTH",
                   help="mix narrowed (past CENTER+WIDTH/2 along AXIS) and whole transfers")
    p.add_argument("--exclude-blend-lateral", dest="exclude_blend_lateral", default=None, metavar="CENTER,WIDTH",
                   help="fade the narrowed share to 0 away from the midline")
    p.add_argument("--exclude-blend-smooth", dest="exclude_blend_smooth", default=None,
                   metavar="N[,ALPHA[,RADIUS]]",
                   help="N Laplacian steps (alpha 0.5, radius 0.04 m or inf) near leg-hole loops")
    p.add_argument("--allow-no-loops", dest="allow_no_loops", action="store_true",
                   help="blend without the smoothing when no leg-hole loop is found")
    p.add_argument("--allow-unseated", dest="allow_unseated", action="store_true",
                   help="transfer onto a target whose baked shape state differs from the body's")
    p.add_argument("--reference-bone", dest="reference_bone", default=DEFAULTS["reference_bone"],
                   help="source bone whose head is the blend's origin")
    p.add_argument("--forward", dest="forward_bone", default=None, metavar="BONE",
                   help="read forward from BONE's tail minus head instead of the feet")
    p.add_argument("--viz", default=None, help="save a review copy with the matched/inpainted colour layer here")
    p.add_argument("--report", dest="report_path", default=None, help="JSON report path")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--whatif", action="store_true", help="run everything in memory; save nothing")
    mode.add_argument("--out", dest="out_path", default=None, help="save the reweighted costume here")
    mode.add_argument("--in-place", dest="in_place", action="store_true", help="save over --in")
    add_force_load_repair(p)
    a = p.parse_args(argv)

    a.target_list = [n.strip() for n in a.targets.split(",") if n.strip()]
    if not a.target_list:
        p.error("--targets named nothing")
    a.shape_list = []
    for item in a.shapes:
        left, sep, value = item.rpartition("=")
        if not sep or not left:
            p.error("--shape wants K=V or MESH:K=V, got %r" % item)
        mesh, colon, key = left.partition(":")
        try:
            a.shape_list.append((mesh if colon else None, key if colon else left, float(value)))
        except ValueError:
            p.error("--shape %s: %r is not a number" % (left, value))
    if a.smooth is not None:
        parts = a.smooth.split(",")
        try:
            a.smooth_t = (int(parts[0]), float(parts[1]) if len(parts) > 1 else 0.2)
        except ValueError:
            p.error("--smooth wants N[,F], got %r" % a.smooth)
        if len(parts) > 2:
            p.error("--smooth wants N[,F], got %r" % a.smooth)
    else:
        a.smooth_t = None
    a.exclude = [g for g in (a.source_exclude or "").split(",") if g]
    if bool(a.exclude) != (a.source_exclude_max is not None):
        p.error("--source-exclude and --source-exclude-max go together; the share is per garment, so "
                "there is no default")
    a.blend = None
    if a.exclude_blend is not None:
        axis, _, rest = a.exclude_blend.partition(",")
        if axis.strip("+-").upper() in ("X", "Y", "Z"):
            p.error("--exclude-blend axis %r is a world axis, which reads differently on a rig facing -Y "
                    "than +Y; name forward, back, lateral or up (forward is the body's facing, from the feet)"
                    % axis)
        c, w = _floats(rest, 2, 2, "--exclude-blend", p, "AXIS,CENTER,WIDTH (e.g. forward,-0.06,0.03)")
        a.blend = (axis, c, w)
    a.lateral = None
    if a.exclude_blend_lateral is not None:
        a.lateral = tuple(_floats(a.exclude_blend_lateral, 2, 2, "--exclude-blend-lateral", p, "CENTER,WIDTH"))
    a.blend_smooth = None
    if a.exclude_blend_smooth is not None:
        f = a.exclude_blend_smooth.split(",")
        if len(f) > 3:
            p.error("--exclude-blend-smooth wants N[,ALPHA[,RADIUS]], got %r" % a.exclude_blend_smooth)
        try:
            a.blend_smooth = (int(f[0]), float(f[1]) if len(f) > 1 and f[1] else 0.5,
                              float(f[2]) if len(f) > 2 and f[2] else 0.04)
        except ValueError:
            p.error("--exclude-blend-smooth wants N[,ALPHA[,RADIUS]], got %r" % a.exclude_blend_smooth)
    if a.out_path and os.path.abspath(a.out_path) == os.path.abspath(a.in_path):
        p.error("--out is --in itself; pass --in-place to save over it")
    return a


def recipe(a) -> str:
    """The canonical command line: weight-determining flags in the door's order, defaults
    omitted, no paths or mode flags."""
    out = [TOOL, "--targets", ",".join(a.target_list)]
    if a.source != DEFAULTS["source"]:
        out += ["--source", a.source]
    for mesh, key, value in a.shape_list:
        out += ["--shape", "%s%s=%s" % (mesh + ":" if mesh else "", key, value)]
    if a.mask:
        out += ["--mask", a.mask]
    if a.max_distance != DEFAULTS["max_distance"]:
        out += ["--max-distance", str(a.max_distance)]
    if a.normal_angle != DEFAULTS["normal_angle"]:
        out += ["--normal-angle", str(a.normal_angle)]
    if not a.flip:
        out += ["--no-flip"]
    if a.smooth_t:
        out += ["--smooth", "%d,%s" % a.smooth_t]
    if a.exclude:
        out += ["--source-exclude", ",".join(a.exclude), "--source-exclude-max", str(a.source_exclude_max)]
    if a.blend:
        out += ["--exclude-blend", "%s,%s,%s" % a.blend]
    if a.lateral:
        out += ["--exclude-blend-lateral", "%s,%s" % a.lateral]
    if a.blend_smooth:
        out += ["--exclude-blend-smooth", "%d,%s,%s" % a.blend_smooth]
    if a.allow_no_loops:
        out += ["--allow-no-loops"]
    if a.allow_unseated:
        out += ["--allow-unseated"]
    if a.reference_bone != DEFAULTS["reference_bone"]:
        out += ["--reference-bone", a.reference_bone]
    if a.forward_bone:
        out += ["--forward", a.forward_bone]
    return shlex.join(out)


def _resolve_mesh(name, flag):
    import bpy
    ob = bpy.data.objects.get(name)
    if ob is None or ob.type != 'MESH':
        print("AVATARPREP: %s ? => FAIL: %s %r is not a mesh in this file" % (TOOL, flag, name))
        sys.exit(2)
    return ob


def _fmt_row(r):
    band = "-" if r["matched_pct_boundary_band"] is None else "%.1f%%" % r["matched_pct_boundary_band"]
    inner = "-" if r["matched_pct_interior"] is None else "%.1f%%" % r["matched_pct_interior"]
    parts = "; ".join("%d verts at %s mm" % (p["verts"], ",".join("%.0f" % c for c in p["centre_mm"]))
                      for p in r["unmatched_parts"]) or "none"
    return ("%s verts=%d matched=%d (boundary band %s, interior %s, via flip %d) inpainted=%d "
            "written=%d touched=%d kept-garment=%d kept-full=%d capped=%d contact max|dw|=%.3f "
            "mean=%.3f unmatched-parts=%s"
            % (r["mesh"], r["verts"], r["matched"], band, inner, r["matched_flipped"], r["inpainted"],
               r["written"], r["touched"], r["kept_garment"], r["kept_full"], r["capped"],
               r["contact_max_dw"], r["contact_mean_dw"], parts))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = _parse_args(argv)
    ensure_deps(TOOL)
    import bpy
    repair = open_blend(args.in_path, writes=not args.whatif or bool(args.viz),
                        force_load_repair=args.force_load_repair)
    enable_avatarprep()
    from avatarprep.core import scene_utils
    from avatarprep.core import weight_transfer as WT

    source = _resolve_mesh(args.source, "--source")
    targets = [_resolve_mesh(n, "--targets") for n in args.target_list]
    label = "%s->%s" % (source.name, ",".join(t.name for t in targets))
    line = recipe(args)
    print("AVATARPREP: recipe: %s" % line)
    out = {"recipe": line, "load_repair": repair, "measured_in": os.path.abspath(args.in_path)}

    try:
        rep = WT.transfer_weights(
            source, targets, shapes=args.shape_list, mask=args.mask, max_distance=args.max_distance,
            normal_angle=args.normal_angle, flip=args.flip, smooth=args.smooth_t, exclude=args.exclude,
            exclude_max=args.source_exclude_max, blend=args.blend, blend_lateral=args.lateral,
            blend_smooth=args.blend_smooth, allow_no_loops=args.allow_no_loops,
            allow_unseated=args.allow_unseated, reference_bone=args.reference_bone,
            forward_bone=args.forward_bone)
    except WT.WeightTransferError as e:
        print("AVATARPREP: %s %s => FAIL: %s" % (TOOL, label, e))
        if args.report_path:
            out["error"] = str(e)
            write_report(args.report_path, out)
        sys.exit(1)

    masks = {r["mesh"]: r.pop("matched_mask") for r in rep["targets"]}
    out.update(rep)
    if rep["excluded"]:
        x = rep["excluded"]
        print("AVATARPREP: source-exclude %s share > %g: dropped %d vertices, %d of %d triangles"
              % (",".join(x["bones"]), x["max"], x["verts"], x["tris"], x["of_tris"]))
    if rep["frame"]:
        f = rep["frame"]
        print("AVATARPREP: frame from %s head; forward from %s: %s"
              % (f["reference_bone"], f["from"], " ".join("%s=%s" % (k, f[k]) for k in WT.AXES if k in f)))
    for r in rep["targets"]:
        print("AVATARPREP: %s" % _fmt_row(r))
        if r.get("blend"):
            b = r["blend"]
            print("AVATARPREP:   blend narrowed=%d whole=%d ramp=%d smoothed=%d near %d leg-hole loops"
                  % (b["narrowed"], b["whole"], b["ramp"], b["smoothed"], b["loops"]))

    saved = None
    if not args.whatif:
        for t in targets:
            scene_utils.write_stamp(t, scene_utils.STAMP_WEIGHTS, line)
    viz = None
    if args.viz:
        viz = os.path.abspath(args.viz)
        os.makedirs(os.path.dirname(viz) or ".", exist_ok=True)
        for t in targets:
            WT.paint_matched(t, masks[t.name])
        bpy.ops.wm.save_as_mainfile(filepath=viz, copy=True)
        for t in targets:
            WT.clear_matched(t)
    if not args.whatif:
        saved = os.path.abspath(args.in_path) if args.in_place else os.path.abspath(args.out_path)
        os.makedirs(os.path.dirname(saved) or ".", exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=saved)
    out["saved"] = saved
    out["viz"] = viz
    if args.report_path:
        write_report(args.report_path, out)

    trailer = ""
    if args.report_path:
        trailer += " | report=%s" % os.path.abspath(args.report_path)
    if viz:
        trailer += " | viz=%s" % viz
    trailer += " | saved=%s" % saved if saved else " (whatif; nothing saved)"
    print("AVATARPREP: %s %s meshes=%d written=%d capped=%d => OK%s"
          % (TOOL, label, len(targets), sum(r["written"] for r in rep["targets"]),
             sum(r["capped"] for r in rep["targets"]), trailer))


if __name__ == "__main__":
    run_cli(main, TOOL)
