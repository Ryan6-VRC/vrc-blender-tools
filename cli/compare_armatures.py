"""Headless CLI: read-only seam-compatibility diff of two armatures (no mutation).

Run:
  blender <in.blend> --background --factory-startup --python cli/compare_armatures.py -- \
      --in <in.blend> --base <armatureObjectName> --merge <armatureObjectName> \
      [--tol <float>] [--report <report.json>]

Exit codes: 0 = clean (compat PASS) · 1 = incompatible (compat FAIL) ·
2 = ERROR (bad armature name, report write failure, argparse usage).
"""
import os
import sys
import argparse

# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, resolve_arm, write_report


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="compare_armatures")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--base", dest="base", required=True)
    p.add_argument("--merge", dest="merge", required=True)
    p.add_argument("--tol", dest="tol", type=float, default=1e-4)
    p.add_argument("--report", dest="report", default=None)
    return p.parse_args(argv)


def main():
    args = _parse_args()
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    enable_avatarprep()
    from avatarprep.core.merge_armatures import compare_armatures, report_offenders

    base = resolve_arm(args.base, "base")
    merge = resolve_arm(args.merge, "merge")

    report = compare_armatures(base, merge, tol=args.tol)
    verdict = "PASS" if report["clean"] else "FAIL"
    counts = ("matched=%d only_in_base=%d only_in_merge=%d renames=%d "
              "parent_mismatch=%d position_mismatch=%d stamp_mismatch=%d warnings=%d"
              % (len(report["matched"]), len(report["only_in_base"]),
                 len(report["only_in_merge"]), len(report["suspected_renames"]),
                 len(report["parent_mismatches"]), len(report["position_mismatches"]),
                 len(report["stamp_mismatches"]), len(report["warnings"])))
    print("AVATARPREP: compat %s %s vs %s (%s)"
          % (verdict, args.base, args.merge, counts))
    for line in report_offenders(report):
        print("AVATARPREP: OFFENDER", line)
    for line in report["warnings"]:
        print("AVATARPREP: WARNING", line)

    if args.report:
        write_report(args.report, report)

    sys.exit(0 if report["clean"] else 1)


if __name__ == "__main__":
    main()
