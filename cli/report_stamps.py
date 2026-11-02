"""Headless CLI: read-only report of a .blend's avatarprep provenance stamps.

Run:
  blender <in.blend> --background --factory-startup --python cli/report_stamps.py -- \
      --in <in.blend> [--shapekeys [SUBSTR]]

The read/query counterpart of stamp_base: every avatarprep_ stamp
(``scene_utils.report_stamps``) in one call. Never saves. Exit 0 always, except exit 2
on a bad --in or open failure.

``--shapekeys`` is the enumeration door for "which key is the blink", in place of ad-hoc
bpy. Each mesh line (and its JSON entry) carries ``library`` / ``data_library`` like the
stamp lines, because a head that is a library override of a sibling base carries no baked
stamp, so this line is the only one naming its library.
"""
import os
import sys
import json
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="report_stamps")
    p.add_argument("--in", dest="in_path", required=True,
                   help=".blend to open read-only and report on; never saved")
    p.add_argument("--shapekeys", dest="shapekeys", nargs="?", const="", metavar="SUBSTR",
                   default=None,
                   help="Also list shape-key names per mesh (basis excluded), optionally "
                        "narrowed by a case-insensitive substring")
    return p.parse_args(argv)


def main():
    args = _parse_args()
    import bpy
    open_blend(args.in_path, writes=False)
    enable_avatarprep()
    from avatarprep.core import scene_utils

    report = scene_utils.report_stamps(bpy.context.scene)

    def _linked(e):
        # data-linked is a local object over read-only linked data.
        if e.get("library"):
            return " linked=%s" % e["library"]
        if e.get("data_library"):
            return " data-linked=%s" % e["data_library"]
        return ""

    def _print_mesh(owner, m):
        if m.get("corrupt") is not None:
            print("AVATARPREP: WARNING mesh %s baked=CORRUPT %s (%s)%s"
                  % (m["name"], m["corrupt"], owner, _linked(m)))
        elif "baked" in m:
            print("AVATARPREP: mesh %s baked=%s (%s)%s"
                  % (m["name"], m["baked"], owner, _linked(m)))
        if "weights" in m:
            print("AVATARPREP: mesh %s weights=%s (%s)%s"
                  % (m["name"], m["weights"], owner, _linked(m)))
        if "folded" in m:
            print("AVATARPREP: mesh %s folded=%r (%s)%s"
                  % (m["name"], m["folded"], owner, _linked(m)))
        if "pushed" in m:
            print("AVATARPREP: mesh %s pushed=%s (%s)%s"
                  % (m["name"], m["pushed"], owner, _linked(m)))
        if "offset" in m:
            print("AVATARPREP: mesh %s offset=%s (%s)%s"
                  % (m["name"], m["offset"], owner, _linked(m)))
        if "authored" in m:
            if isinstance(m["authored"], dict):
                for key, line in sorted(m["authored"].items()):
                    print("AVATARPREP: mesh %s authored %s=%s (%s)%s"
                          % (m["name"], key, line, owner, _linked(m)))
            else:
                print("AVATARPREP: WARNING mesh %s authored=CORRUPT %s (%s)%s"
                      % (m["name"], m["authored"], owner, _linked(m)))
        if "normals" in m:
            if isinstance(m["normals"], dict):
                for key, line in sorted(m["normals"].items()):
                    print("AVATARPREP: mesh %s normals %s=%s (%s)%s"
                          % (m["name"], key, line, owner, _linked(m)))
            else:
                print("AVATARPREP: WARNING mesh %s normals=CORRUPT %s (%s)%s"
                      % (m["name"], m["normals"], owner, _linked(m)))

    for a in report["armatures"]:
        base = a["base"] if a["base"] is not None else "unknown"
        print("AVATARPREP: armature %s base=%s state=%r (%s)%s"
              % (a["name"], base, a["state"], a["state_kind"], _linked(a)))
        for m in a["meshes"]:
            _print_mesh("armature %s" % a["name"], m)
    for m in report["unbound"]:
        _print_mesh("unbound", m)

    if args.shapekeys is not None:
        needle = args.shapekeys.casefold()
        listing = {}
        for ob in bpy.context.scene.objects:
            if ob.type != 'MESH':
                continue
            sk = ob.data.shape_keys
            all_keys = [k.name for k in sk.key_blocks[1:]] if sk else []
            names = [n for n in all_keys if needle in n.casefold()]
            entry = {"names": names,
                     "library": scene_utils.library_path(ob),
                     "data_library": scene_utils.library_path(ob.data)}
            listing[ob.name] = entry
            print("AVATARPREP: mesh %s shapekeys[%d/%d]: %s%s"
                  % (ob.name, len(names), len(all_keys), ", ".join(names), _linked(entry)))
        report["shapekeys"] = listing

    print("AVATARPREP: report_stamps =", json.dumps(report))
    sys.exit(0)


if __name__ == "__main__":
    run_cli(main, "report_stamps")
