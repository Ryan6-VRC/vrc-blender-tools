"""Headless CLI: mark the body triangles a garment set hides, as a Delete carrier shape key.

Run:
  blender --background --factory-startup --python cli/mark_coverage.py -- \
      --in <costume.blend> --body Body_Base --garments Dress,Socks --shape-name Cover_X \
      [--distance-mm 30] [--weight-tol 0.35] [--angle-deg 80] [--hem-margin-mm 5] \
      [--cut-threshold-m 0.01] [--delta-m 0.02] [--cut-shape NAME]... \
      [--report <json>] [--render <dir>] [--whatif | --out <base-copy.blend> | --in-place] \
      [--replace] [--force-load-repair]

Measures on the costume blend, where the body is normally a library link. ``--whatif``
measures, reports and renders and writes no blend. Without it the door opens the base
blend the body's mesh links from, checks the body there against what it measured (vertex
count and Basis hash), writes the carrier shape key on it, and saves to ``--out`` (or over
the base with ``--in-place``). A local body needs no second file and is written in
``--in`` itself, saved to ``--out``.

Exit 0 on ``=> OK``; 1 on ``=> FAIL:`` (ran, refused: nothing covered, changed body, key
exists); 2 on unresolvable input (missing body/garment, bad args) or a crash. Criterion
and carrier rule: ``avatarprep/core/coverage.py``.
"""
import os
import sys
import json
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli, add_force_load_repair, write_report

TOOL = "markcoverage"


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: %s ? => FAIL: bad args: %s" % (TOOL, message))
        sys.exit(2)


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = _Parser(prog="mark_coverage")
    p.add_argument("--in", dest="in_path", required=True, help="costume .blend to measure in")
    p.add_argument("--body", required=True, help="body mesh object (a library link is fine)")
    p.add_argument("--garments", required=True,
                   help="comma-separated garment mesh objects that count as cover; explicit, "
                        "no default — a garment behind a Clothing toggle never covers")
    p.add_argument("--shape-name", dest="shape_name", required=True,
                   help="carrier shape key name, one per run")
    p.add_argument("--distance-mm", dest="distance_mm", type=float, default=30.0,
                   help="how far above the skin a garment may sit and still cover")
    p.add_argument("--weight-tol", dest="weight_tol", type=float, default=0.35,
                   help="max total-variation distance between body and garment weights (0..1)")
    p.add_argument("--angle-deg", dest="angle_deg", type=float, default=80.0,
                   help="max angle between the body normal and the direction to the garment")
    p.add_argument("--hem-margin-mm", dest="hem_margin_mm", type=float, default=5.0,
                   help="decline within this distance of a garment boundary edge")
    p.add_argument("--cut-threshold-m", dest="cut_threshold", type=float, default=0.01,
                   help="the consumer's Delete threshold: what --cut-shape already removes, "
                        "and what --delta-m must clear")
    p.add_argument("--delta-m", dest="delta", type=float, default=0.02,
                   help="carrier displacement along the inverted normal")
    p.add_argument("--cut-shape", dest="cut_shapes", action="append", default=[],
                   help="a shape the row's existing Deletes already remove (repeatable)")
    p.add_argument("--report", dest="report", default=None, help="JSON report path")
    p.add_argument("--render", dest="render_dir", default=None,
                   help="directory for the review contact sheets")
    p.add_argument("--whatif", action="store_true", help="measure only; write no blend")
    p.add_argument("--out", dest="out_path", default=None,
                   help="where to save the body's blend with the carrier key added")
    p.add_argument("--in-place", dest="in_place", action="store_true",
                   help="save over the body's own blend instead of --out")
    p.add_argument("--replace", action="store_true", help="overwrite a same-named shape key")
    add_force_load_repair(p)
    a = p.parse_args(argv)
    if a.delta <= a.cut_threshold:
        p.error("--delta-m %g must exceed --cut-threshold-m %g or the carrier never triggers"
                % (a.delta, a.cut_threshold))
    if not a.whatif and not a.out_path and not a.in_place:
        p.error("pass --whatif, --out, or --in-place")
    if a.out_path and a.in_place:
        p.error("--out and --in-place are exclusive")
    return a


def _fail(label, reason):
    print("AVATARPREP: %s %s => FAIL: %s" % (TOOL, label or "?", reason))
    sys.exit(1)


def _resolve_mesh(name, role):
    import bpy
    ob = bpy.context.scene.objects.get(name)
    if ob is None or ob.type != 'MESH':
        print("AVATARPREP: ERROR --%s %r is not a mesh object in this scene" % (role, name))
        sys.exit(2)
    return ob


def _render_sheets(body, result, garments, label, out_dir):
    """Three sheets through render_mesh: the marked body alone; the marked body beside the
    garments (solid, so silhouettes read); the body with the carrier triangles removed."""
    from avatarprep.core import coverage
    from avatarprep.core.render_mesh import render
    names = [g.name for g in garments]
    pngs = []
    marked = coverage.marked_copy(body, result, name="__cover_marked", garment_order=names)
    removed = None
    try:
        line = render(label=label + "_marked", only=[marked.name], angles=["front", "back", "left", "right"],
                      shading="vertexcolor", out_dir=out_dir)
        if "=> FAIL:" in line:
            raise RuntimeError(line)
        pngs.append(line.split("png=")[-1].strip())
        line = render(label=label + "_with_garments", only=[marked.name] + names,
                      angles=["front", "back"], shading="solid", out_dir=out_dir)
        if "=> FAIL:" in line:
            raise RuntimeError(line)
        pngs.append(line.split("png=")[-1].strip())
        removed = coverage.marked_copy(body, result, name="__cover_removed", garment_order=names,
                                       remove_carrier=True)
        line = render(label=label + "_removed", only=[removed.name], angles=["front", "back", "left", "right"],
                      shading="vertexcolor", out_dir=out_dir)
        if "=> FAIL:" in line:
            raise RuntimeError(line)
        pngs.append(line.split("png=")[-1].strip())
    finally:
        coverage.remove_marked_copy(marked)
        if removed is not None:
            coverage.remove_marked_copy(removed)
    return pngs


def main():
    args = _parse_args()
    import bpy
    open_blend(args.in_path, writes=not args.whatif, force_load_repair=args.force_load_repair)
    enable_avatarprep()
    from avatarprep.core import coverage

    body = _resolve_mesh(args.body, "body")
    garments = [_resolve_mesh(n.strip(), "garments") for n in args.garments.split(",") if n.strip()]
    if not garments:
        print("AVATARPREP: ERROR --garments named nothing")
        sys.exit(2)
    label = args.shape_name

    distance = args.distance_mm / 1000.0
    hem = args.hem_margin_mm / 1000.0
    try:
        result = coverage.measure(body, garments, distance=distance, weight_tol=args.weight_tol,
                                  angle_deg=args.angle_deg, hem_margin=hem,
                                  cut_threshold=args.cut_threshold, cut_shapes=args.cut_shapes)
    except coverage.CoverageError as e:
        print("AVATARPREP: ERROR", e)
        sys.exit(2)

    pngs = []
    if args.render_dir:
        try:
            pngs = _render_sheets(body, result, garments, label, os.path.abspath(args.render_dir))
        except Exception as e:
            _fail(label, "render failed: %s" % e)

    declined = coverage.declined_summary(result)
    if args.report:
        rep = {k: v for k, v in result.items() if k not in ("covered", "claimed_by", "cut", "near")}
        rep["shape_name"] = args.shape_name
        rep["delta_m"] = args.delta
        rep["declined"] = declined
        rep["renders"] = pngs
        rep["measured_in"] = os.path.abspath(args.in_path)
        write_report(args.report, rep)

    if result["realised_triangles"] == 0:
        _fail(label, "no triangle is covered by %s at distance=%gmm weight_tol=%g "
                     "(declined %s)" % (",".join(g.name for g in garments), args.distance_mm,
                                        args.weight_tol,
                                        " ".join("%s:%d" % kv for kv in declined.items())))

    saved = None
    if not args.whatif:
        lib = result["body_library"]
        if lib:
            base_path = bpy.path.abspath(lib)
            open_blend(base_path, writes=True, force_load_repair=args.force_load_repair)
        else:
            base_path = os.path.abspath(args.in_path)
        me = bpy.data.meshes.get(result["body_mesh"])
        if me is None:
            _fail(label, "mesh %r not found in %s" % (result["body_mesh"], base_path))
        try:
            coverage.write_carrier(me, result, shape_name=args.shape_name, delta=args.delta,
                                   replace=args.replace)
        except coverage.CoverageError as e:
            _fail(label, str(e))
        saved = base_path if args.in_place else os.path.abspath(args.out_path)
        os.makedirs(os.path.dirname(saved) or ".", exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=saved)

    trailer = ""
    if args.report:
        trailer += " | report=%s" % os.path.abspath(args.report)
    if pngs:
        trailer += " | png=%s" % ",".join(pngs)
    if saved:
        trailer += " | saved=%s" % saved
    print("AVATARPREP: %s %s body=%s garments=%d cut=%d covered=%d realised=%d residue=%d declined=%s => OK%s"
          % (TOOL, label, result["body"], len(garments), result["already_cut_triangles"],
             result["covered_triangles"], result["realised_triangles"], result["residue_triangles"],
             ",".join("%s:%d" % kv for kv in declined.items()) or "-", trailer))


if __name__ == "__main__":
    run_cli(main, "mark_coverage")
