"""Flags and printed lines shared by the fit doors (``report_fit``, ``compare_fit``, ``push_garment``).

Only importable after a door's structural ``sys.path`` insert. The flag names are
``transfer_weights``'s where the concept is the same; syntax errors exit 2 through the
door's parser, and what the scene must hold (a bone, a group) is the core's to refuse.
"""
import sys

WORLD = ("x", "y", "z")
SWEEP_FORM = ("AXIS is forward, back, lateral, up, own, a or b; angles in degrees, STEPS per direction "
              "(default 3). No AXIS sweeps every axis of the bone's joint-table row, over MIN..MAX when "
              "given, else the row's range. An axis in the row keeps its flexion sign (positive = "
              "flexion); an axis outside it needs MIN..MAX and turns by the right-hand rule about the "
              "axis (back is forward reversed)")


def add_measure_args(p, *, sweep_help=None):
    """The measurement flags every fit door takes, after ``--in`` and ``--targets``."""
    p.add_argument("--source", default="Body_Base", help="body mesh measured against (linked is fine; never written)")
    p.add_argument("--shape", dest="shapes", action="append", default=[], metavar="K=V|MESH:K=V",
                   help="a shape value the row wears, set for the measurement only; bare K lands on every "
                        "listed mesh carrying it (repeatable)")
    p.add_argument("--cut-shape", dest="cut_shapes", action="append", default=[], metavar="K",
                   help="a shape the row's Deletes remove: vertices it moves are left out, on every listed "
                        "mesh carrying it (repeatable)")
    p.add_argument("--cut-threshold-m", dest="cut_threshold", type=float, default=0.01,
                   help="the Delete threshold: how far a --cut-shape must move a vertex to remove it "
                        "(Modular Avatar's ShapeChanger default; read the prefab's when it differs)")
    p.add_argument("--sweep", dest="sweeps", action="append", default=[], metavar="BONE[:AXIS][:MIN..MAX[:STEPS]]",
                   help=sweep_help or "sweep this body bone instead of the derived set (repeatable). " + SWEEP_FORM)
    p.add_argument("--region", dest="regions", action="append", default=[], metavar="KIND:NAME",
                   help="group:NAME (weight over 0.5), bone:NAME (largest influence at or under it), or "
                        "zone:front|back|left|right|above:BONE|below:BONE; terms joined by + intersect "
                        "(repeatable)")
    p.add_argument("--reference-bone", dest="reference_bone", default="Hips",
                   help="body bone whose head is the frame's origin")
    p.add_argument("--forward", dest="forward_bone", default=None, metavar="BONE",
                   help="read forward from BONE's tail minus head instead of the feet")


def parse_sweep(text, error):
    """``(bone, axis|None, (lo, hi)|None, steps|None)`` from ``BONE[:AXIS][:MIN..MAX[:STEPS]]``,
    read from the right so a bone name may itself hold a colon."""
    toks = text.split(":")
    axis = rng = steps = None
    if len(toks) >= 3 and toks[-1].isdigit() and ".." in toks[-2]:
        steps = int(toks.pop())
        if steps < 1:
            error("--sweep %s: STEPS must be at least 1" % text)
    if len(toks) >= 2 and ".." in toks[-1]:
        lo, _, hi = toks.pop().partition("..")
        try:
            rng = (float(lo), float(hi))
        except ValueError:
            error("--sweep %s: the range wants MIN..MAX in degrees, e.g. -30..120" % text)
        if not rng[0] <= 0.0 <= rng[1] or rng[0] == rng[1]:
            error("--sweep %s: MIN..MAX must run through 0 (rest), e.g. -30..120 or 0..90" % text)
    if len(toks) >= 2 and toks[-1].lower().strip("+-") in WORLD:
        error("--sweep %s: %s is a world axis, which reads differently on a rig facing -Y than +Y; name "
              "forward, back, lateral, up, own, a or b" % (text, toks[-1]))
    from avatarprep.core.fit import AXIS_WORDS
    if len(toks) >= 2 and toks[-1].lower() in AXIS_WORDS:
        axis = toks.pop().lower()
    bone = ":".join(toks)
    if not bone:
        error("--sweep %r names no bone" % text)
    return bone, axis, rng, steps


def finish_measure_args(a, error):
    """Validate and normalise the shared flags on a parsed namespace."""
    from avatarprep.core import fit
    a.target_list = [n.strip() for n in a.targets.split(",") if n.strip()]
    if not a.target_list:
        error("--targets named nothing")
    from cli._common import parse_shapes
    a.shape_list = parse_shapes(a.shapes, error)
    a.sweep_list = [parse_sweep(s, error) for s in a.sweeps] or None
    for r in a.regions:
        try:
            fit.parse_region(r)
        except ValueError as e:
            error(str(e))
    if not a.cut_threshold > 0:
        error("--cut-threshold-m must be positive")


def recipe_measure(a, defaults_source="Body_Base"):
    """The shared flags in canonical order, defaults omitted."""
    out = []
    for s in a.sweeps:
        out += ["--sweep", s]
    for r in a.regions:
        out += ["--region", r]
    if a.source != defaults_source:
        out += ["--source", a.source]
    for mesh, key, value in a.shape_list:
        out += ["--shape", "%s%s=%s" % (mesh + ":" if mesh else "", key, value)]
    for c in a.cut_shapes:
        out += ["--cut-shape", c]
    if a.cut_threshold != 0.01:
        out += ["--cut-threshold-m", str(a.cut_threshold)]
    if a.reference_bone != "Hips":
        out += ["--reference-bone", a.reference_bone]
    if a.forward_bone:
        out += ["--forward", a.forward_bone]
    return out


def resolve_meshes(tool, a):
    import bpy
    out = {}
    for flag, names in (("--source", [a.source]), ("--targets", a.target_list)):
        for n in names:
            ob = bpy.data.objects.get(n)
            if ob is None or ob.type != 'MESH':
                print("AVATARPREP: %s ? => FAIL: %s %r is not a mesh in this file" % (tool, flag, n))
                sys.exit(2)
            out[n] = ob
    return out[a.source], [out[n] for n in a.target_list]


# --- printed lines ------------------------------------------------------------------------------

def _vec(v):
    return "(%s)" % ",".join("%.3g" % (float(x) + 0.0) for x in v)


def frame_json(frame):
    return {k: ([round(float(x), 6) + 0.0 for x in v] if hasattr(v, "__len__") and not isinstance(v, str) else v)
            for k, v in frame.items()}


def print_context(data, the_plan, prefix=""):
    """Frame, extents, classes and the sweep: the lines every fit door prints first."""
    from avatarprep.core import fit
    f = data["frame"]
    print("AVATARPREP: %sframe from %s head; forward from %s: forward=%s lateral=%s up=%s"
          % (prefix, f["reference_bone"], f["from"], _vec(f["forward"]), _vec(f["lateral"]), _vec(f["up"])))
    for g in data["garments"]:
        ex = fit.extents(data, g)
        print("AVATARPREP: %s%s extents in the frame (m from %s head): %s"
              % (prefix, g["name"], f["reference_bone"],
                 " ".join("%s %+.3f..%+.3f" % (k, lo, hi) for k, (lo, hi) in ex.items())))
        c = g["cls"]
        print("AVATARPREP: %s%s classes: physbone %d, edge %d, skin-tight %d, loose %d, cut %d"
              % (prefix, g["name"], c["physbone"].sum(), c["edge"].sum(), c["skin-tight"].sum(),
                 c["loose"].sum(), g["cut"].sum()))
    for b in the_plan["bones"]:
        axes = ", ".join("%s %+g..%+g x%d" % (x["axis"], x["lo"], x["hi"], x["steps"]) for x in b["axes"])
        why = "named" if b["explicit"] else "largest share: garment %.2f, body near it %.2f" % (
            b["garment_share"], b["body_share"])
        print("AVATARPREP: %ssweep %s row=%s side=%s (%s): %s" % (prefix, b["bone"], b["row"], b["side"], why, axes))
    print("AVATARPREP: %ssweep: %d steps over %d bones" % (prefix, len(the_plan["steps"]), len(the_plan["bones"])))


def print_legend():
    from avatarprep.core import fit
    for line in fit.LEGEND:
        print("AVATARPREP: legend: %s" % line)


def fmt_metric(metric, v):
    """One metric's worst-step value as printed; ``v`` is ``fit.worst``'s entry."""
    if v is None:
        return "%s: no vertex in its population" % metric.replace("_", "-")
    if metric in ("new_pen", "edge_poke", "body_through"):
        if not v["count"]:
            return "%s 0 at every step" % metric.replace("_", "-")
        return "%s %d (max %.2f mm) @ %s, body self-x there %d" % (
            metric.replace("_", "-"), v["count"], v["max_mm"], v["step"], v["self_x"])
    if metric in ("stretch", "excess"):
        return "%s max %.3f, p95 %.3f over %d edges, %d over 1.3 @ %s" % (
            metric, v["max"], v["p95"], v["n"], v["over"], v["step"])
    return "slide p95 %.2f mm over %d contact vertices, max %.2f mm @ %s" % (v["p95"], v["n"], v["max"], v["step"])


def metric_value(metric, v):
    """The number a metric's worst step is ranked by, for deltas."""
    if v is None:
        return None
    return v[{"new_pen": "count", "edge_poke": "count", "body_through": "count",
              "stretch": "max", "excess": "max", "slide": "p95"}[metric]]
