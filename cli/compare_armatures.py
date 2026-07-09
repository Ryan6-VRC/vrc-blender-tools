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
import json
import argparse


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="compare_armatures")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--base", dest="base", required=True)
    p.add_argument("--merge", dest="merge", required=True)
    p.add_argument("--tol", dest="tol", type=float, default=1e-4)
    p.add_argument("--report", dest="report", default=None)
    return p.parse_args(argv)


def _enable_avatarprep():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import avatarprep
    try:
        avatarprep.register()
    except Exception:
        pass


def _resolve_arm(name, arg):
    import bpy
    obj = bpy.context.scene.objects.get(name)
    if obj is None or obj.type != 'ARMATURE':
        print("AVATARPREP: ERROR --%s %r is not an armature in this scene" % (arg, name))
        sys.exit(2)
    return obj


def main():
    args = _parse_args()
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    _enable_avatarprep()
    from avatarprep.core.merge_armatures import compare_armatures, report_offenders

    base = _resolve_arm(args.base, "base")
    merge = _resolve_arm(args.merge, "merge")

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
        try:
            report_path = os.path.abspath(args.report)
            d = os.path.dirname(report_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
        except Exception as e:
            print("AVATARPREP: ERROR failed to write report:", e)
            sys.exit(2)
        print("AVATARPREP: report ->", report_path)

    sys.exit(0 if report["clean"] else 1)


if __name__ == "__main__":
    main()
