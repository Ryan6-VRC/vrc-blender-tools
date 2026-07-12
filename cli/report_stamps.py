"""Headless CLI: read-only report of a .blend's avatarprep provenance stamps.

Run:
  blender <in.blend> --background --factory-startup --python cli/report_stamps.py -- \
      --in <in.blend>

The read/query counterpart of stamp_base — inspects a file's avatarprep_base /
avatarprep_state (per armature) and avatarprep_baked (per baked mesh) in one call.
Opens the blend read-only; never saves. A report never "fails" — exit 0 always
(a bad --in / open failure is the only ERROR exit 2).
"""
import os
import sys
import json
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="report_stamps")
    p.add_argument("--in", dest="in_path", required=True)
    return p.parse_args(argv)


def main():
    args = _parse_args()
    import bpy
    try:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    except Exception as e:
        print("AVATARPREP: ERROR failed to open --in:", e)
        sys.exit(2)
    enable_avatarprep()
    from avatarprep.core import scene_utils

    report = scene_utils.report_stamps(bpy.context.scene)

    def _print_mesh(owner, m):
        # A corrupt (non-map) stamp is the one genuine fault → WARNING (greppable,
        # matching the Slice-E CLI family); a clean baked map prints a plain line.
        if m.get("corrupt") is not None:
            print("AVATARPREP: WARNING mesh %s baked=CORRUPT %s (%s)"
                  % (m["name"], m["corrupt"], owner))
        else:
            print("AVATARPREP: mesh %s baked=%s (%s)" % (m["name"], m["baked"], owner))

    for a in report["armatures"]:
        base = a["base"] if a["base"] is not None else "unknown"
        print("AVATARPREP: armature %s base=%s state=%r (%s)"
              % (a["name"], base, a["state"], a["state_kind"]))
        for m in a["meshes"]:
            _print_mesh("armature %s" % a["name"], m)
    for m in report["unbound"]:
        _print_mesh("unbound", m)

    print("AVATARPREP: report_stamps =", json.dumps(report))
    sys.exit(0)


if __name__ == "__main__":
    main()
