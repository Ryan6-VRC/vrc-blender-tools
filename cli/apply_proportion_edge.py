"""Headless CLI: apply one proportion edge to a blend (or validate with --whatif).

Run:
  blender <in.blend> --background --factory-startup --python cli/apply_proportion_edge.py -- \
      --in <in.blend> --out <out.blend> --edge <edge.json> [--skip-shapekeys] \
      [--bone-override OLD=NEW ...] [--shapekey-override NAME=VALUE ...] [--report <report.json>]

  # Preview: validate the edge, then report the geometry it would produce. Writes
  # nothing; --out must be omitted (passing it errors — a preview never writes a
  # deliverable).
  blender <in.blend> --background --factory-startup --python cli/apply_proportion_edge.py -- \
      --in <in.blend> --edge <edge.json> --whatif [--skip-shapekeys] \
      [--bone-override OLD=NEW ...] [--shapekey-override NAME=VALUE ...] [--report <report.json>]

--whatif mutates nothing ON DISK. Once the validate gate is clean it trial-applies the
real engine in memory and measures the result at each stage boundary, then discards it
— so the reported min-z / crown / height and the achieved per-bone lengths are measured,
not predicted. That is what a height-touching edge previously needed a full
author-apply-measure-reauthor pass to learn. Costs 1.6-4.5 s on a full avatar.

Aggregate extremes answer the HEIGHT question and nothing wider: a morph that reshapes
mid-body without reaching the feet or the crown legitimately moves min-z and max-z by
zero, so a 0.000 delta is not evidence that a shapekeys block did nothing. Per-mesh
bounds in --report are the next place to look.

Exit codes: 0 = would apply (with numbers) · 1 = offenders, would not apply ·
2 = ERROR (bad edge path, unopenable --in, --out combined with --whatif).
"""
import os
import sys
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import (enable_avatarprep, kv, write_report, open_blend, run_cli,
                         add_force_load_repair)


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="apply_proportion_edge")
    p.add_argument("--in", dest="in_path", required=True,
                   help=".blend to apply the edge to")
    p.add_argument("--out", dest="out_path", default=None,   # not required under --whatif
                   help="Where to save the edited .blend; omit under --whatif")
    p.add_argument("--edge", dest="edge", required=True, metavar="EDGE.JSON",
                   help="Path to the proportion-edge JSON describing the bone and "
                        "shape-key changes")
    p.add_argument("--armature", dest="armature", default=None,
                   help="Armature object to target; required when the scene has more than one")
    p.add_argument("--whatif", dest="whatif", action="store_true",
                   help="Validate the edge, then report the geometry it would produce "
                        "(measured by an in-memory trial that is never saved); no --out")
    p.add_argument("--skip-shapekeys", action="store_true",
                   help="Apply only the edge's bone changes, leaving shape keys untouched")
    p.add_argument("--bone-override", action="append", default=[], metavar="OLD=NEW",
                   help="Retarget one of the edge's bone names onto this rig's spelling. "
                        "Repeatable")
    p.add_argument("--shapekey-override", action="append", default=[], metavar="NAME=VALUE",
                   help="Set an edge shape-key's value. NAME=null DROPS that key from the "
                        "edge, and a NAME the edge does not carry is ADDED to it — an added "
                        "key must exist on some bound mesh or the run refuses, so a typo'd "
                        "name surfaces as 'shapekey not found on any mesh'. Repeatable")
    p.add_argument("--report", dest="report", default=None,
                   help="Write the full result dict here as JSON")
    add_force_load_repair(p)
    args = p.parse_args(argv)
    if not args.whatif and not args.out_path:
        p.error("--out is required unless --whatif is given")
    if args.whatif and args.out_path:
        p.error("--out is meaningless under --whatif (preview mutates nothing)")
    return args


def _resolve_armature(name):
    """Resolve the target armature at the CLI boundary — fail loud, never guess.

    A named ``--armature`` must exist and be an armature; with no name the scene
    must hold exactly one armature (>1 aborts naming them, so an owned .blend that
    appended a base-body reference can't silently target the wrong rig). The core
    ``find_armature`` stays permissive for its operator callers by design."""
    import bpy
    arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if name:
        obj = bpy.context.scene.objects.get(name)
        if obj is None or obj.type != 'ARMATURE':
            print("AVATARPREP: ERROR --armature %r is not an armature in this scene" % name)
            sys.exit(1)
        return obj
    if not arms:
        print("AVATARPREP: ERROR no armature found")
        sys.exit(1)
    if len(arms) > 1:
        print("AVATARPREP: ERROR multiple armatures (%s); pass --armature <name>"
              % ", ".join(sorted(a.name for a in arms)))
        sys.exit(1)
    return arms[0]


def _geometry_report(stages, edge, bone_overrides, repair):
    """Assemble the staged geometry block from the trial's measurements.

    ``repair`` rides along deliberately: these numbers were measured on whatever the
    load produced, and a number measured on a repaired load must not be copyable into
    a tracked edge JSON without that fact attached."""
    from avatarprep.core import measure
    pre = stages[0][1]
    out = {"loaded_with_repairs": repair, "stages": [], "bones": {},
           "scale_ops": []}
    prev = pre
    # ``m["unevaluated"]`` is deliberately not carried into the entry: the gate above
    # exits 1 before any stage is measured, so on every reachable path it is empty. Carry
    # it the moment that gate softens — a report is the one artifact that outlives the
    # stdout warning, and a measured number in it with no caveat beside it is the defect
    # measure.measure_geometry's contract names.
    for name, m in stages:
        entry = {"stage": name, "aggregate": m["aggregate"], "per_mesh": m["per_mesh"]}
        if name != "pre":
            entry["delta_from_previous"] = measure.aggregate_delta(prev, m)
        out["stages"].append(entry)
        prev = m
    final = stages[-1][1]
    out["delta_total"] = measure.aggregate_delta(pre, final)
    # Every bone's head/tail on both ends, so any span an author cares about is
    # derivable from the report without applying (shoulder-to-wrist is
    # |UpperArm.head - Hand.head|); measure.py owns why it isn't pre-chosen here.
    out["bones"] = {"pre": pre["bones"], "post": final["bones"]}
    named = set()
    for i, op in enumerate(edge["scales"]):
        names = [bone_overrides.get(b, b) for b in op["bones"]]
        named.update(names)
        out["scale_ops"].append({
            "index": i, "value": op["value"], "space": op["space"],
            "pivot": op["pivot"], "bones": names,
            "lengths": measure.bone_length_deltas(pre, final, names)})
    out["collateral_lengths"] = measure.collateral_lengths(pre, final, named)
    return out


def _print_geometry(geometry, repair):
    """Printed tier: aggregate per stage plus achieved per-bone lengths. Per-mesh
    bounds and every bone's head/tail stay in --report; a 22-mesh avatar would bury
    the two numbers that decide a height edge."""
    for entry in geometry["stages"]:
        agg = entry["aggregate"]
        if agg is None:
            print("AVATARPREP: whatif stage %-10s no mesh geometry to measure"
                  % entry["stage"])
            continue
        d = entry.get("delta_from_previous")
        suffix = ""
        if d:
            suffix = "  (d_min_z %+.6f d_max_z %+.6f d_height %+.6f)" % (
                d["d_min_z"], d["d_max_z"], d["d_height"])
        print("AVATARPREP: whatif stage %-10s min_z %.6f max_z %.6f height %.6f%s"
              % (entry["stage"], agg["min"][2], agg["max"][2], agg["height"], suffix))
    for op in geometry["scale_ops"]:
        for row in op["lengths"]:
            pct = "n/a" if row["pct"] is None else "%+.2f%%" % row["pct"]
            print("AVATARPREP: whatif scales[%d] bone %s length %.6f -> %.6f (%s achieved)"
                  % (op["index"], row["bone"], row["before"], row["after"], pct))
    collateral = geometry["collateral_lengths"]
    if collateral:
        print("AVATARPREP: whatif %d unnamed bone(s) also changed length — the scale "
              "spread past the bones the edge names:" % len(collateral))
        for row in collateral[:12]:
            pct = "n/a" if row["pct"] is None else "%+.2f%%" % row["pct"]
            print("AVATARPREP: whatif   %s %.6f -> %.6f (%s)"
                  % (row["bone"], row["before"], row["after"], pct))
        if len(collateral) > 12:
            print("AVATARPREP: whatif   ... %d more; --report has all of them"
                  % (len(collateral) - 12))
    else:
        # Stated positively: silence here would be indistinguishable from not looking.
        print("AVATARPREP: whatif no bone outside the edge's scale ops changed length")
    total = geometry["delta_total"]
    if total:
        print("AVATARPREP: whatif total d_min_z %+.6f d_max_z %+.6f d_height %+.6f"
              % (total["d_min_z"], total["d_max_z"], total["d_height"]))
    if repair:
        # Re-stated AFTER the numbers: a reader who scrolled to the measurements must
        # not miss that they describe a state Blender altered as it read the file.
        print("AVATARPREP: WARNING the numbers above were measured on a repaired load: %s"
              % repair)


def main():
    args = _parse_args()
    import bpy
    repair = open_blend(args.in_path, writes=not args.whatif,
                        force_load_repair=args.force_load_repair)
    enable_avatarprep()
    from avatarprep.core import scene_utils, proportions, measure

    armature = _resolve_armature(args.armature)

    bone_overrides = kv(args.bone_override)
    sk_raw = kv(args.shapekey_override)
    shapekey_overrides = {k: (None if v.lower() == "null" else float(v))
                          for k, v in sk_raw.items()}

    if args.whatif:
        # Read-only gate (folded from the former standalone validate CLI): load the edge,
        # check it against the rig, report offenders/warnings, exit 1 on offenders. No save.
        try:
            edge = proportions.load_edge(os.path.abspath(args.edge))
        except proportions.EdgeError as e:
            print("AVATARPREP: ERROR", e)
            sys.exit(1)

        meshes = scene_utils.get_bound_meshes(armature)
        report = proportions.validate_proportion_edge(
            armature, meshes, edge, bone_overrides=bone_overrides,
            shapekey_overrides=shapekey_overrides, skip_shapekeys=args.skip_shapekeys)

        offenders = report["offenders"]
        warnings = report["warnings"]
        verdict = "FAIL" if offenders else "PASS"
        print("AVATARPREP: validate %s %s -> %s (%d offenders, %d warnings)"
              % (verdict, edge["source"], edge["target"], len(offenders), len(warnings)))
        for o in offenders:
            print("AVATARPREP: OFFENDER", o)
        for w in warnings:
            print("AVATARPREP: WARNING", w)

        if offenders:
            if args.report:
                write_report(args.report, report)
            sys.exit(1)

        # Gate clean, so measure what the edge would actually do. The trial applies the
        # real engine in memory and is never saved (--out is refused at parse time), so
        # every number below is measured rather than predicted -- validated bit-exact
        # against a real apply on both shipped longlimb edges and on a shapekey-bearing
        # fixture. Cost is 1.6-4.5 s on a full avatar.
        stages = []
        proportions.apply_proportion_edge(
            armature, meshes, edge, bone_overrides=bone_overrides,
            shapekey_overrides=shapekey_overrides, skip_shapekeys=args.skip_shapekeys,
            stage_hook=lambda name: stages.append(
                (name, measure.measure_geometry(armature, meshes))))

        geometry = _geometry_report(stages, edge, bone_overrides, repair)
        report["geometry"] = geometry
        _print_geometry(geometry, repair)

        # A backstop, not the protection: validate_proportion_edge makes an unevaluated
        # mesh an OFFENDER, so the exit above fires on the same predicate over the same
        # mesh list and nothing here is reachable today (measured: a rig with one hidden
        # bound mesh exits 1 at the gate, with no geometry block in the report at all).
        # Kept because the reachable version of this door is one softened offender away,
        # and then a --whatif readout would claim to be measured geometry while carrying
        # an unmeasured mesh -- which is the thing the numbers above cannot show.
        for name in sorted({n for _, m in stages for n in m["unevaluated"]}):
            print("AVATARPREP: WARNING mesh not evaluated, measured undeformed:", name)

        if args.report:
            write_report(args.report, report)

        sys.exit(0)

    try:
        report = proportions.apply_proportion_edge(
            armature, None, args.edge, bone_overrides=bone_overrides,
            shapekey_overrides=shapekey_overrides,
            skip_shapekeys=args.skip_shapekeys)
    except proportions.EdgeError as e:
        print("AVATARPREP: ERROR", e)
        sys.exit(1)
    print("AVATARPREP: applied %s -> %s (%d scale ops, %d shapekeys, %d bakes, %d warnings)"
          % (report["source"], report["target"], report["scales_applied"],
             len(report["shapekeys"]), len(report["bakes"]), len(report["warnings"])))
    for w in report["warnings"]:
        print("AVATARPREP: WARNING", w)

    if args.report:
        write_report(args.report, report)

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: saved ->", out_path)


if __name__ == "__main__":
    run_cli(main, "apply_proportion_edge")
