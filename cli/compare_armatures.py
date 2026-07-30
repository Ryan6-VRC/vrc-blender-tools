"""Headless CLI: read-only seam-compatibility diff of two armatures (no mutation).

Run:
  blender <in.blend> --background --factory-startup --python cli/compare_armatures.py -- \
      --in <in.blend> --base <armatureObjectName> --merge <armatureObjectName> \
      [--merge-in <other.blend|other.fbx>] [--tol <float>] [--noise-tol <float>] \
      [--report <report.json>]

``--merge-in`` compares across two files: ``--merge`` names the armature in THAT
file (appended from a ``.blend`` / imported from a ``.fbx`` into the open scene —
in memory only, nothing is saved), so two separately-imported rigs both named
``Armature`` compare without hand-rolled append scripts. A collision auto-suffix
(``Armature.001``) is resolved and reported; it does not affect the verdict.

Positional thresholds: ``--tol`` (default 1e-4) to ``--noise-tol`` (default 1e-3)
is the named-warning noise tier; above ``--noise-tol`` is an offender. See
``compare_armatures``'s docstring for the calibration.

Exit codes: 0 = clean (compat PASS) · 1 = incompatible (compat FAIL) ·
2 = ERROR (bad armature name, bad --merge-in, report write failure, argparse usage).
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
    p.add_argument("--merge-in", dest="merge_in", default=None)
    p.add_argument("--tol", dest="tol", type=float, default=1e-4)
    p.add_argument("--noise-tol", dest="noise_tol", type=float, default=1e-3)
    p.add_argument("--report", dest="report", default=None)
    return p.parse_args(argv)


def _is_numeric_suffix_of(name, base):
    """True for Blender's collision suffixes only (``Armature.001`` of
    ``Armature``) — never for a genuinely different rig (``Armature.NoirLace``)."""
    return name.startswith(base + ".") and name[len(base) + 1:].isdigit()


def _load_merge_side(path, merge_name):
    """Bring ``merge_name`` from a second file into the open scene (in memory
    only; this CLI never saves). Returns the armature OBJECT — its scene name
    may be auto-suffixed on collision, which is why the object, not the name,
    is the handle. In-grammar ERROR + exit 2 on any resolution failure,
    including an unreadable file (an uncaught raise would exit 0 = compat PASS)."""
    import bpy
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".blend":
        try:
            with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
                if merge_name not in data_from.objects:
                    print("AVATARPREP: ERROR --merge %r not found in --merge-in %s"
                          % (merge_name, path))
                    sys.exit(2)
                data_to.objects = [merge_name]
        except Exception as e:  # SystemExit above passes through untouched
            print("AVATARPREP: ERROR --merge-in unreadable: %s" % e)
            sys.exit(2)
        obj = data_to.objects[0]
        if obj is None or obj.type != 'ARMATURE':
            print("AVATARPREP: ERROR --merge %r in --merge-in is not an armature"
                  % merge_name)
            sys.exit(2)
        bpy.context.scene.collection.objects.link(obj)
    elif ext == ".fbx":
        from avatarprep.core import import_fbx, scene_utils
        before = set(bpy.data.objects)
        try:
            import_fbx.import_fbx(path)
        except Exception as e:
            print("AVATARPREP: ERROR --merge-in unreadable: %s" % e)
            sys.exit(2)
        new_arms = [o for o in bpy.data.objects
                    if o not in before and o.type == 'ARMATURE']
        # Match --merge among the NEW armatures only, accepting Blender's
        # numeric collision suffix (the base scene may own the plain name).
        # No exactly-one shortcut: a typo must never silently pick "the one".
        named = [o for o in new_arms
                 if o.name == merge_name or _is_numeric_suffix_of(o.name, merge_name)]
        if len(named) != 1:
            print("AVATARPREP: ERROR --merge %r matched %d of the imported "
                  "armatures in --merge-in %s (imported: %s)"
                  % (merge_name, len(named), path, [o.name for o in new_arms]))
            sys.exit(2)
        obj = named[0]
        # A compare door must not invent provenance it then gates on:
        # import_fbx stamps state=unproportioned, which is a GATING dimension —
        # left in place, the verdict would depend on the merge side's container
        # format (.fbx FAILs where the same rig via .blend warns-and-passes).
        for key in (scene_utils.STAMP_BASE, scene_utils.STAMP_STATE):
            if key in obj:
                del obj[key]
    else:
        print("AVATARPREP: ERROR --merge-in must be a .blend or .fbx, got %s" % path)
        sys.exit(2)
    bpy.context.view_layer.update()
    if obj.name != merge_name:
        print("AVATARPREP: merge-in resolved %r -> %r (collision auto-suffix)"
              % (merge_name, obj.name))
    return obj


def main():
    args = _parse_args()
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    enable_avatarprep()
    from avatarprep.core.merge_armatures import compare_armatures, report_offenders

    base = resolve_arm(args.base, "base")
    if args.merge_in:
        merge = _load_merge_side(args.merge_in, args.merge)
    else:
        merge = resolve_arm(args.merge, "merge")

    try:
        report = compare_armatures(base, merge, tol=args.tol, noise_tol=args.noise_tol)
    except ValueError as e:  # bad --tol/--noise-tol pair
        print("AVATARPREP: ERROR", e)
        sys.exit(2)
    verdict = "PASS" if report["clean"] else "FAIL"
    counts = ("matched=%d only_in_base=%d only_in_merge=%d renames=%d "
              "parent_mismatch=%d position_mismatch=%d position_noise=%d "
              "stamp_mismatch=%d warnings=%d"
              % (len(report["matched"]), len(report["only_in_base"]),
                 len(report["only_in_merge"]), len(report["suspected_renames"]),
                 len(report["parent_mismatches"]), len(report["position_mismatches"]),
                 len(report["position_noise"]),
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
