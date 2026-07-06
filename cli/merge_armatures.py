"""Headless CLI: union-merge two armatures by bone name, behind the compat gate.

Run:
  blender <in.blend> --background --factory-startup --python cli/merge_armatures.py -- \
      --in <in.blend> --out <out.blend> --base <armatureObjectName> --merge <armatureObjectName> \
      [--rename OLD=NEW ...] [--force] [--skip-apply-transforms] [--whatif] [--report <report.json>]

Exit codes: 0 = merged (verdict PASS, --out saved) · 1 = verdict FAIL (--out NOT
saved) · 2 = ERROR (bad armature name, mid-merge exception, write failure).

--whatif previews: the gates run for real, nothing mutates, nothing is saved (--out
optional and ignored); exit 0 = would merge, 1 = would FAIL.

On FAIL the safety net is not saving --out; --report (if given) is still written
with the FULL result dict (carries postcheck) so a postcheck FAIL can be triaged.
"""
import os
import sys
import json
import argparse


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="merge_armatures")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", default=None)
    p.add_argument("--base", dest="base", required=True)
    p.add_argument("--merge", dest="merge", required=True)
    p.add_argument("--rename", action="append", default=[])
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-stamps", dest="force_stamps", action="store_true")
    p.add_argument("--whatif", action="store_true")
    p.add_argument("--skip-apply-transforms", dest="skip_apply_transforms",
                   action="store_true")
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


def _kv(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def _resolve_arm(name, arg):
    import bpy
    obj = bpy.context.scene.objects.get(name)
    if obj is None or obj.type != 'ARMATURE':
        print("AVATARPREP: ERROR --%s %r is not an armature in this scene" % (arg, name))
        sys.exit(2)
    return obj


def _write_report(path, data):
    try:
        report_path = os.path.abspath(path)
        d = os.path.dirname(report_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception as e:
        print("AVATARPREP: ERROR failed to write report:", e)
        sys.exit(2)
    print("AVATARPREP: report ->", report_path)


def main():
    args = _parse_args()
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    _enable_avatarprep()
    from avatarprep.core.merge_armatures import merge_armatures

    if not args.whatif and not args.out_path:
        print("AVATARPREP: ERROR --out is required unless --whatif")
        sys.exit(2)

    base = _resolve_arm(args.base, "base")
    merge = _resolve_arm(args.merge, "merge")
    rename_map = _kv(args.rename)

    try:
        result = merge_armatures(
            base, merge, rename_map=rename_map, force=args.force,
            force_stamps=args.force_stamps,
            apply_transforms=not args.skip_apply_transforms, whatif=args.whatif)
    except Exception as e:
        print("AVATARPREP: ERROR merge raised:", e)
        sys.exit(2)

    verdict = result.get("verdict")
    if verdict == "PASS":
        label = "whatif PASS (would merge)" if args.whatif else "PASS"
        print("AVATARPREP: merge %s %s <- %s (unified %d, added %d)"
              % (label, args.base, args.merge, result.get("bones_unified", 0),
                 result.get("bones_added", 0)))
        # Fail loud on advisory warnings and on each overridden category.
        for line in (result.get("report") or {}).get("warnings", []):
            print("AVATARPREP: WARNING", line)
        for line in result.get("forced_structural") or []:
            print("AVATARPREP: FORCED STRUCTURAL", line)
        for line in result.get("forced_stamp") or []:
            print("AVATARPREP: FORCED STAMP", line)
        if args.whatif:
            if args.report:
                _write_report(args.report, result)
            sys.exit(0)
        # Save the deliverable (--out) BEFORE the diagnostic report, so a
        # report-write failure can't discard a successful merge's output.
        try:
            out_path = os.path.abspath(args.out_path)
            d = os.path.dirname(out_path)
            if d:
                os.makedirs(d, exist_ok=True)
            bpy.ops.wm.save_as_mainfile(filepath=out_path)
        except Exception as e:
            print("AVATARPREP: ERROR failed to save out:", e)
            sys.exit(2)
        print("AVATARPREP: saved ->", out_path)
        if args.report:
            _write_report(args.report, result)
        sys.exit(0)

    # FAIL: do NOT save --out (rollback covers only pre-mutation FAILs).
    print("AVATARPREP: merge %sFAIL %s <- %s (reason %s)"
          % ("whatif " if args.whatif else "", args.base, args.merge,
             result.get("reason", "postcheck")))
    # Core populates offenders on every FAIL (pre-mutation and postcheck alike).
    for line in (result.get("offenders") or []):
        print("AVATARPREP: OFFENDER", line)
    # A co-occurring missing-stamp warning must show on FAIL too (report is None
    # for the early same-armature/preflight/rename_map FAILs — guarded).
    for line in (result.get("report") or {}).get("warnings", []):
        print("AVATARPREP: WARNING", line)
    if args.report:
        _write_report(args.report, result)  # FULL dict — carries postcheck
    sys.exit(1)


if __name__ == "__main__":
    main()
