"""Headless CLI: seat a keyless garment on the body it now wears, and optionally give it the body's keys.

Run:
  blender --background --factory-startup --python cli/transfer_shapekeys.py -- \
      --in <in.blend> --source Body_Base --targets Top,base \
      [--authored Breasts_big=0.5 ...] [--keys Breasts_flat,Breasts_big] \
      [--no-seat] [--smooth N] [--scan Breasts_big] [--whatif] [--out <out.blend>] [--report <json>]

The garment was cut against some configuration of the vendor body; the venue body rests
at another (its ``avatarprep_baked`` map plus live key values). ``--authored`` names the
configuration the garment was cut against, per key (absent keys read as 0, the vendor's
neutral body); the seat moves the garment from there to the body's state. ``--keys`` names
which of the body's keys land on the garment as relative keys — an optional call, since not
every venue wants a garment carrying body morphs; a seat alone is a complete fix. A key
that is added carries the body's state as its live value, with the authored offset folded
into Basis and recorded as ``-authored``; a key that is not folds the whole seat into Basis.

``--scan KEY`` reports the garment's gap against the body at each candidate value of KEY,
so an unknown authored value can be read off it: the value with a near-zero minimum and no
penetration. ``--whatif`` measures everything and writes nothing. The source is never
written. Behaviour and the mechanism choice: ``avatarprep/core/shapekey_transfer.py``.
"""
import os
import sys
import argparse
import json

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli, add_force_load_repair, write_report


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("AVATARPREP: transfer_shapekeys ? => FAIL: bad args: %s" % message)
        sys.exit(2)


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = _Parser(prog="transfer_shapekeys")
    p.add_argument("--in", dest="in_path", required=True, help=".blend holding the garment and its linked body")
    p.add_argument("--out", dest="out_path", default=None,
                   help="Where to save the seated .blend (required unless --whatif)")
    p.add_argument("--source", required=True, help="Body mesh object carrying the keys (linked is fine; never written)")
    p.add_argument("--targets", required=True, help="Comma-separated garment mesh objects to seat")
    p.add_argument("--keys", default="",
                   help="Comma-separated body keys to add to each garment as relative keys (optional)")
    p.add_argument("--authored", action="append", default=[], metavar="KEY=VALUE",
                   help="Body configuration the garment was cut against; repeatable. A key not "
                        "named reads as 0")
    p.add_argument("--no-seat", dest="seat", action="store_false",
                   help="Add keys only; leave the garment where it sits")
    p.add_argument("--scan", action="append", default=[], metavar="KEY",
                   help="Report the garment's gap at each candidate value of KEY (0, .25, .5, .75, 1)")
    p.add_argument("--footprint-mm", dest="footprint_mm", type=float, default=1.0,
                   help="A garment vertex moves only where its nearest body point moves more than "
                        "this under the key (default 1.0)")
    p.add_argument("--smooth", type=int, default=0, metavar="N",
                   help="Laplacian-smooth each transferred displacement N times over the garment's "
                        "edges, inside the footprint: softens a band that buckles where the seat "
                        "crosses a fold, at a small cost in gap fidelity (default 0)")
    p.add_argument("--whatif", action="store_true", help="Measure and report; write nothing")
    p.add_argument("--report", dest="report_path", default=None, help="Write the JSON report here")
    add_force_load_repair(p)
    return p.parse_args(argv)


def _kv(items):
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            print("AVATARPREP: transfer_shapekeys ? => FAIL: bad --authored %r (want KEY=VALUE)" % it)
            sys.exit(2)
        try:
            out[k] = float(v)
        except ValueError:
            print("AVATARPREP: transfer_shapekeys ? => FAIL: bad --authored value %r" % it)
            sys.exit(2)
    return out


def _fmt(st):
    if not st or st.get("n", 0) == 0:
        return "n=0"
    return "n=%d min=%.2fmm p5=%.2fmm median=%.2fmm penetrating=%d" % (
        st["n"], st["min_mm"], st["p5_mm"], st["median_mm"], st["penetrating"])


def main():
    args = _parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    authored = _kv(args.authored)
    will_write = bool(keys or authored) and not args.whatif
    if args.whatif and args.out_path:
        print("AVATARPREP: transfer_shapekeys ? => FAIL: --out is meaningless under --whatif (preview mutates nothing)")
        sys.exit(2)
    if not args.seat and not keys:
        print("AVATARPREP: transfer_shapekeys ? => FAIL: --no-seat needs --keys, or the run does nothing")
        sys.exit(2)
    if will_write and not args.out_path:
        print("AVATARPREP: transfer_shapekeys ? => FAIL: --out is required to seat or add keys; --whatif previews, --scan alone needs neither")
        sys.exit(2)
    if not (keys or authored or args.scan):
        print("AVATARPREP: transfer_shapekeys ? => FAIL: nothing to do: pass --keys, --authored or --scan")
        sys.exit(2)
    import bpy
    repair = open_blend(args.in_path, writes=will_write, force_load_repair=args.force_load_repair)
    enable_avatarprep()
    from avatarprep.core import shapekey_transfer as T

    source = bpy.data.objects.get(args.source)
    if source is None or source.type != 'MESH':
        print("AVATARPREP: transfer_shapekeys %s => FAIL: --source %r is not a mesh in this file"
              % (args.source, args.source))
        sys.exit(1)
    targets = []
    for name in (n.strip() for n in args.targets.split(",") if n.strip()):
        t = bpy.data.objects.get(name)
        if t is None or t.type != 'MESH':
            print("AVATARPREP: transfer_shapekeys %s => FAIL: --targets %r is not a mesh in this file"
                  % (args.source, name))
            sys.exit(1)
        targets.append(t)
    label = "%s->%s" % (args.source, ",".join(t.name for t in targets))
    out = {"load_repair": repair, "scans": [], "transfer": None}

    try:
        for key in args.scan:
            for t in targets:
                sc = T.scan_authored(source, t, key, authored=authored, neutral=keys)
                out["scans"].append(sc)
                if not sc["scan"]:
                    print("AVATARPREP: scan %s on %s: %d vertices in the key's core — not coupled to it"
                          % (key, t.name, sc["core_verts"]))
                    continue
                print("AVATARPREP: scan %s on %s (%d core vertices)" % (key, t.name, sc["core_verts"]))
                for val, st in sc["scan"].items():
                    print("AVATARPREP:   %s=%s  %s" % (key, val, _fmt(st)))
        if keys or authored:
            rep = T.transfer_shapekeys(source, targets, keys, authored, seat=args.seat,
                                       footprint=args.footprint_mm / 1000.0, smooth=args.smooth,
                                       whatif=args.whatif)
            out["transfer"] = rep
            print("AVATARPREP: authored %s; body state %s; seat %s; keys added %s"
                  % (json.dumps(rep["authored"]), json.dumps(rep["state"]),
                     json.dumps(rep["seat"]) or "none", ",".join(keys) or "none"))
            for row in rep["targets"]:
                print("AVATARPREP: %s footprint=%d/%d" % (row["target"], row["footprint_verts"], row["verts"]))
                print("AVATARPREP:   before %s" % _fmt(row["before"]))
                print("AVATARPREP:   after  %s" % _fmt(row["after"]))
                fid = row["fidelity_to_authored_mm"]
                print("AVATARPREP:   fidelity to authored gap p95=%smm max=%smm; leak=%d; max move=%.1fmm"
                      % (fid.get("p95", "-"), fid.get("max", "-"), row["leak_outside_footprint"],
                         row["max_move_mm"]))
                if row["live_values"]:
                    print("AVATARPREP:   live %s" % json.dumps(row["live_values"]))
                if row["baked_written"]:
                    print("AVATARPREP:   avatarprep_baked %s" % json.dumps(row["baked_written"]))
    except T.TransferError as e:
        print("AVATARPREP: transfer_shapekeys %s => FAIL: %s" % (label, e))
        if args.report_path:
            out["error"] = str(e)
            write_report(args.report_path, out)
        sys.exit(1)

    if args.report_path:
        write_report(args.report_path, out)
    if out["transfer"] is None:
        print("AVATARPREP: transfer_shapekeys %s => OK (scan only; nothing written)" % label)
        return
    if args.whatif:
        print("AVATARPREP: transfer_shapekeys %s => OK (whatif; nothing written)" % label)
        return
    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("AVATARPREP: transfer_shapekeys %s => OK | saved -> %s" % (label, out_path))


if __name__ == "__main__":
    run_cli(main, "transfer_shapekeys")
