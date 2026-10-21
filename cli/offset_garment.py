"""Headless CLI: offset a skin-tight garment off the body by an even layer, so it gains a uniform cloth thickness whatever its topology.

Run:
  blender --background --factory-startup --python cli/offset_garment.py -- \
      --in <costume.blend> --targets A,B --amount M [--source Body_Base] [--shape K=V | MESH:K=V]... \
      [--tight 0.003] [--loose 0.006] [--report JSON] (--whatif | --out <out.blend> | --in-place) \
      [--force-load-repair]

Each target vertex, at its Basis, finds its nearest point on ``--source`` and moves outward
along the body's normal there by ``--amount`` metres times a fade: whole while its distance
to the skin is within ``--tight``, falling by a smoothstep to nothing at ``--loose``, and
nothing beyond. The skin-tight tube thickens evenly while frills, ruffles and straps standing
off the skin stay where they are. The body is read at the row's state, its shape keys at their
stored values with each ``--shape`` overriding one (a Unity weight of 100 is 1): only the
source's state places the skin, so a ``--shape`` the source lacks refuses. Its normal is the
area-weighted vertex normal interpolated across the nearest triangle, so the direction turns
smoothly over the skin rather than faceting on a coarse limb. ``--loose`` minus ``--tight``
must exceed 1.5 times ``--amount``, or layers between them can cross. This is a rest-shape
change for the whole garment; ``push_garment`` is the door for one spot the body comes
through under motion.

Asserted before anything saves: no vertex beyond ``--loose`` moves, none moves farther than
``--amount``, and the vertex count and every key's offset from Basis hold, since the same
object-space delta goes onto the mesh, Basis and every shape key. A saving run stamps each
target object with ``avatarprep_offset``, the canonical command line printed as ``recipe:``
(defaults omitted, no paths or mode flags); a stamped mesh refuses, since a rerun would offset
twice: re-derive from the blend before the offset. ``--whatif`` plans everything and saves
nothing.

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
from cli._common import enable_avatarprep, open_blend, run_cli, add_force_load_repair, write_report, parse_shapes
from cli import _fit

TOOL = "offset_garment"
DEFAULTS = {"source": "Body_Base", "tight": 0.003, "loose": 0.006}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: %s ? => FAIL: bad args: %s" % (TOOL, message))
        sys.exit(2)


def _parse_args(argv):
    p = _Parser(prog=TOOL)
    p.add_argument("--in", dest="in_path", required=True, help="costume .blend holding the targets and the body")
    p.add_argument("--targets", required=True, help="comma-separated garment meshes to offset")
    p.add_argument("--amount", type=float, required=True,
                   help="metres the skin-tight vertices move outward, above 0 and at most 0.005")
    p.add_argument("--source", default=DEFAULTS["source"],
                   help="body mesh the garment is offset off (linked is fine; never written)")
    p.add_argument("--shape", dest="shapes", action="append", default=[], metavar="K=V|MESH:K=V",
                   help="a shape value the row wears on the body, set for the offset only (repeatable); only "
                        "the --source's state is read, so a key it lacks refuses")
    p.add_argument("--tight", type=float, default=DEFAULTS["tight"],
                   help="metres from the skin within which a vertex moves the whole --amount")
    p.add_argument("--loose", type=float, default=DEFAULTS["loose"],
                   help="metres from the skin beyond which a vertex holds; between, the move fades by a smoothstep")
    p.add_argument("--report", dest="report_path", default=None, help="JSON report path")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--whatif", action="store_true", help="plan everything; save nothing")
    mode.add_argument("--out", dest="out_path", default=None, help="save the offset costume here")
    mode.add_argument("--in-place", dest="in_place", action="store_true", help="save over --in")
    add_force_load_repair(p)
    a = p.parse_args(argv)
    enable_avatarprep()
    a.target_list = [n.strip() for n in a.targets.split(",") if n.strip()]
    if not a.target_list:
        p.error("--targets named nothing")
    a.shape_list = parse_shapes(a.shapes, p.error)
    if a.out_path and os.path.abspath(a.out_path) == os.path.abspath(a.in_path):
        p.error("--out is --in itself; pass --in-place to save over it")
    return a


def recipe(a) -> str:
    out = [TOOL, "--targets", ",".join(a.target_list), "--amount", str(a.amount)]
    if a.source != DEFAULTS["source"]:
        out += ["--source", a.source]
    for mesh, key, value in a.shape_list:
        out += ["--shape", "%s%s=%s" % (mesh + ":" if mesh else "", key, value)]
    for flag, key in (("--tight", "tight"), ("--loose", "loose")):
        if getattr(a, key) != DEFAULTS[key]:
            out += [flag, str(getattr(a, key))]
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
        stamp = scene_utils.read_stamp(t, scene_utils.STAMP_OFFSET)
        if stamp is not None:
            refuse("%s was already offset by `%s`; a second offset would move it twice. Re-derive from the blend "
                   "before that offset, or leave it" % (t.name, stamp))
        if scene_utils.is_linked(t) or not scene_utils.is_editable(t):
            refuse("%s is linked or override data and cannot be written; offset the local garment this "
                   "costume owns (own-mergeable makes one)" % t.name)
        if t.data.users > 1:
            refuse("%s shares its mesh with %d other user(s); make it single-user first" % (t.name, t.data.users - 1))

    try:
        plans = [fit.plan_offset(source, t, amount=a.amount, shapes=a.shape_list, tight=a.tight, loose=a.loose)
                 for t in targets]
    except fit.FitError as e:
        refuse(str(e))

    out["targets"] = []
    for t, pl in zip(targets, plans):
        print("AVATARPREP: %s verts %d: moved %d (full %d, partial %d), held %d; max %.3f mm along %s's normals"
              % (t.name, pl["verts"], pl["moved"], pl["full"], pl["partial"], pl["held"], pl["max_mm"], source.name))
        if pl["tight"]:
            print("AVATARPREP: %s skin distance of the %d vertices within --tight: median %.2f -> %.2f mm, "
                  "min %.2f -> %.2f mm" % (t.name, pl["tight"], pl["tight_median_mm"][0], pl["tight_median_mm"][1],
                                           pl["tight_min_mm"][0], pl["tight_min_mm"][1]))
        print("AVATARPREP: %s moved vertices inside %s at rest: before %d, after %d"
              % (t.name, source.name, pl["inside_before"], pl["inside_after"]))
        out["targets"].append({k: v for k, v in pl.items() if k != "delta"})

    saved = None
    if not a.whatif:
        try:
            for i, (t, pl) in enumerate(zip(targets, plans)):
                out["targets"][i]["shape_keys"] = fit.apply_push(t, pl["delta"])
                scene_utils.write_stamp(t, scene_utils.STAMP_OFFSET, line)
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
