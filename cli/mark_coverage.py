"""Headless CLI: mark the body triangles a garment set hides, as a Delete carrier shape key.

Run:
  blender --background --factory-startup --python cli/mark_coverage.py -- \
      --in <costume.blend> --body Body_Base --garments Dress,Socks --shape-name Cover_X \
      [--shape NAME=VALUE]... [--cut-shape NAME]... \
      [--cone-deg 75] [--share 0.5] [--reach-m 1.0] [--kin 2] [--fold-bones GLOB]... \
      [--cut-threshold-m 0.01] [--delta-m 0.02] \
      [--report <json>] [--render <dir>] [--out-marked <blend>] \
      [--whatif | --out <base-copy.blend> | --in-place] [--replace] [--force-load-repair]

Measures on the costume blend, where the body is normally a library link. A coverage
measurement is of a CONFIGURATION: the shapes that configuration sets on the body and the
garments (a waist slimmer, a heel lift, a collar removed) go on with ``--shape`` first, or
the skin is measured where it does not sit in game. ``--whatif`` measures and reports,
renders the three sheets when ``--render`` names a directory, and writes no body blend.
Without it the door opens the base blend the body's mesh links from, checks the body there
against what it measured (vertex count and Basis hash), writes the carrier shape key on
it, and saves to ``--out`` (or over the base with ``--in-place``). A local body needs no
second file and is written in ``--in`` itself, saved to ``--out``. ``--out-marked`` saves
a copy of the costume scene with the marked and carrier-removed bodies beside the
garments, in the measured configuration, for inspecting hems at the cut.

Exit 0 on ``=> OK``; 1 on ``=> FAIL:`` (ran, refused: nothing covered, changed body, key
exists); 2 on unresolvable input (missing body/garment/shape, bad args) or a crash.
Criterion and carrier rule: ``avatarprep/core/coverage.py``.
"""
import os
import sys
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


def _parse_shapes(items, p):
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name:
            p.error("--shape wants NAME=VALUE, got %r" % item)
        try:
            out[name] = float(value)
        except ValueError:
            p.error("--shape %s: %r is not a number" % (name, value))
    return out


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
    p.add_argument("--shape", dest="shapes", action="append", default=[],
                   help="NAME=VALUE the configuration sets on the body or a garment "
                        "(repeatable; 0..1; refused when no listed mesh has the key)")
    p.add_argument("--cut-shape", dest="cut_shapes", action="append", default=[],
                   help="a shape the row's existing Deletes already remove (repeatable)")
    p.add_argument("--cone-deg", dest="cone_deg", type=float, default=75.0,
                   help="half-angle of the ray cone around the skin normal; 90 is the hemisphere")
    p.add_argument("--share", type=float, default=0.5,
                   help="min share of a hit face's skin weight on bones the skin point rides "
                        "(kin and fold applied) for the face to block the ray")
    p.add_argument("--reach-m", dest="reach", type=float, default=1.0,
                   help="how far a ray travels before it counts as escaped")
    p.add_argument("--kin", type=int, default=2,
                   help="parent-or-child steps of the armature that count as the same bone")
    p.add_argument("--fold-bones", dest="fold", action="append", default=[],
                   help="glob of bones that read as their nearest unfolded ancestor "
                        "(repeatable; a physbone chain known to be stiff)")
    p.add_argument("--cut-threshold-m", dest="cut_threshold", type=float, default=0.01,
                   help="the consumer's Delete threshold: what --cut-shape already removes, "
                        "and what --delta-m must clear")
    p.add_argument("--delta-m", dest="delta", type=float, default=0.02,
                   help="carrier displacement along the inverted normal")
    p.add_argument("--report", dest="report", default=None, help="JSON report path")
    p.add_argument("--render", dest="render_dir", default=None,
                   help="directory for the review contact sheets")
    p.add_argument("--out-marked", dest="out_marked", default=None,
                   help="save a copy of the costume scene with the marked and carrier-removed "
                        "bodies beside the garments, for inspection")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--whatif", action="store_true", help="measure only; write no body blend")
    mode.add_argument("--out", dest="out_path", default=None,
                      help="where to save the body's blend with the carrier key added")
    mode.add_argument("--in-place", dest="in_place", action="store_true",
                      help="save over the body's own blend instead of --out")
    p.add_argument("--replace", action="store_true", help="overwrite a same-named shape key")
    add_force_load_repair(p)
    a = p.parse_args(argv)
    a.shape_values = _parse_shapes(a.shapes, p)
    if a.delta <= a.cut_threshold:
        p.error("--delta-m %g must exceed --cut-threshold-m %g or the carrier never triggers"
                % (a.delta, a.cut_threshold))
    if not (0.0 < a.cone_deg <= 90.0):
        p.error("--cone-deg %g must be in (0, 90]" % a.cone_deg)
    if not (0.0 < a.share <= 1.0):
        p.error("--share %g must be in (0, 1]; at 0 every face blocks and the whole body goes" % a.share)
    if a.kin < 0:
        p.error("--kin must be >= 0")
    if a.out_path and os.path.abspath(a.out_path) == os.path.abspath(a.in_path):
        p.error("--out is the costume blend itself; the body's blend is what gets saved there "
                "(pass --in-place to write over the body's own file)")
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


def _render_sheets(body, result, garments, label, out_dir, shapes):
    """Three sheets through render_mesh: the marked body alone; the marked body beside the
    garments (solid, so silhouettes read); the body with the carrier polygons removed. The
    garments render in the measured configuration."""
    from avatarprep.core import coverage
    from avatarprep.core.render_mesh import render
    names = [g.name for g in garments]
    pngs = []
    with coverage.shaped([body] + list(garments), shapes):
        return _render_sheets_shaped(body, result, names, label, out_dir, shapes, coverage, render, pngs)


def _render_sheets_shaped(body, result, names, label, out_dir, shapes, coverage, render, pngs):
    marked = coverage.marked_copy(body, result, name="__cover_marked", garment_order=names, shapes=shapes)
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
                                       remove_carrier=True, shapes=shapes)
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
    open_blend(args.in_path, writes=not args.whatif or bool(args.out_marked),
               force_load_repair=args.force_load_repair)
    enable_avatarprep()
    from avatarprep.core import coverage

    body = _resolve_mesh(args.body, "body")
    garments = [_resolve_mesh(n.strip(), "garments") for n in args.garments.split(",") if n.strip()]
    if not garments:
        print("AVATARPREP: ERROR --garments named nothing")
        sys.exit(2)
    label = args.shape_name

    try:
        result = coverage.measure(body, garments, cone_deg=args.cone_deg, share=args.share, reach=args.reach,
                                  kin=args.kin, fold=args.fold, shapes=args.shape_values,
                                  cut_threshold=args.cut_threshold, cut_shapes=args.cut_shapes)
    except coverage.CoverageError as e:
        print("AVATARPREP: ERROR", e)
        sys.exit(2)

    pngs = []
    if args.render_dir:
        try:
            pngs = _render_sheets(body, result, garments, label, os.path.abspath(args.render_dir), args.shape_values)
        except Exception as e:
            _fail(label, "render failed: %s" % e)
    marked_path = None
    if args.out_marked:
        marked_path = os.path.abspath(args.out_marked)
        os.makedirs(os.path.dirname(marked_path) or ".", exist_ok=True)
        try:
            coverage.save_marked(marked_path, body, result, garments, label=label, shapes=args.shape_values)
        except Exception as e:
            _fail(label, "marked save failed: %s" % e)

    declined = result["declined"]
    declined_txt = ",".join("%s:%d" % kv for kv in declined.items()) or "-"

    def report(saved):
        if not args.report:
            return
        rep = {k: v for k, v in result.items() if k not in ("covered", "claimed_by", "cut", "near")}
        rep["shape_name"] = args.shape_name
        rep["delta_m"] = args.delta
        rep["renders"] = pngs
        rep["marked_blend"] = marked_path
        rep["measured_in"] = os.path.abspath(args.in_path)
        rep["saved"] = saved
        write_report(args.report, rep)

    if result["realised_triangles"] == 0:
        report(None)
        if result["covered_triangles"]:
            _fail(label, "%d triangles are covered but none has a vertex with every surrounding polygon "
                         "covered, so the carrier is empty (residue %d): the cover is too thin to carry"
                  % (result["covered_triangles"], result["residue_triangles"]))
        _fail(label, "no triangle is covered by %s at cone=%g share=%g kin=%d (declined %s)"
              % (",".join(g.name for g in garments), args.cone_deg, args.share, args.kin, declined_txt))

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
    report(saved)

    trailer = ""
    if args.report:
        trailer += " | report=%s" % os.path.abspath(args.report)
    if pngs:
        trailer += " | png=%s" % ",".join(pngs)
    if marked_path:
        trailer += " | marked=%s" % marked_path
    if saved:
        trailer += " | saved=%s" % saved
    print("AVATARPREP: %s %s body=%s garments=%d cut=%d covered=%d realised=%d residue=%d restvisible=%d "
          "declined=%s => OK%s"
          % (TOOL, label, result["body"], len(garments), result["already_cut_triangles"],
             result["covered_triangles"], result["realised_triangles"], result["residue_triangles"],
             result["rest_visible_triangles"], declined_txt, trailer))


if __name__ == "__main__":
    run_cli(main, "mark_coverage")
