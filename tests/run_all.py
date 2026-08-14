"""The repo gate: run every suite headless and enforce exit code AND token.

A suite passes only when its Blender process exits 0 AND its ``<TOKEN> OK``
line printed — either signal alone can lie (``_harness.py`` owns the why).

Runs under plain Python, not Blender:

    python tests/run_all.py [--blender <path-to-blender.exe>]

The Blender binary comes from ``--blender`` or the ``BLENDER`` env var; there
is no default, because a hardcoded machine path in a public repo rots.

The suite list is explicit — no discovery magic decides what runs. But a
``tests/test_*.py`` that is neither listed nor in EXCLUDE fails the gate by
name: an unlisted suite would otherwise silently never run, which is the same
crash-reads-as-pass hole one level up.
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# suite file -> the token its verdict line carries
SUITES = {
    "test_cli_seam.py": "CLI_SEAM_TEST",
    "test_fbx_export.py": "FBXEXPORT_TEST",
    "test_fbx_orientation.py": "FBXORIENT_TEST",
    "test_import_fbx.py": "IMPORT_TEST",
    "test_merge_armatures.py": "MERGE_TEST",
    "test_open_blend.py": "OPENBLEND_TEST",
    "test_profiles_library.py": "PROFILES_TEST",
    "test_proportions.py": "PROP_TEST",
    "test_prune_bones.py": "PRUNE_TEST",
    "test_rename_objects.py": "RENAME_TEST",
    "test_render_mesh.py": "RENDERMESH_TEST",
    "test_report_stamps.py": "REPORT_TEST",
    "test_resolve_armature.py": "RESOLVE_TEST",
    "test_rest_pose.py": "RESTPOSE_TEST",
    "test_shapekey_bake.py": "BAKE_TEST",
}

# test_*.py deliberately not run by the gate (none today; name and justify any
# addition here, or it fails below).
EXCLUDE = set()

# Wide enough for test_cli_seam, which spawns its own Blender subprocesses.
TIMEOUT_SECS = 1800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=os.environ.get("BLENDER"))
    ap.add_argument("--only", action="append", metavar="SUITE",
                    help="run just the named suite file(s); the run then "
                         "reports PARTIAL, never 'RUN_ALL OK' — a partial run "
                         "is not the gate")
    args = ap.parse_args()
    if not args.blender or not os.path.isfile(args.blender):
        print("RUN_ALL FAIL: no Blender binary — pass --blender or set BLENDER "
              "(got %r)" % args.blender)
        sys.exit(2)

    present = {os.path.basename(p) for p in glob.glob(os.path.join(HERE, "test_*.py"))}
    unlisted = sorted(present - set(SUITES) - EXCLUDE)
    missing = sorted(set(SUITES) - present)
    if unlisted or missing:
        if unlisted:
            print("RUN_ALL FAIL: suite(s) on disk but not in SUITES/EXCLUDE — "
                  "they would silently never run: %s" % ", ".join(unlisted))
        if missing:
            print("RUN_ALL FAIL: SUITES names file(s) that do not exist: %s"
                  % ", ".join(missing))
        sys.exit(2)

    selected = args.only or sorted(SUITES)
    bad = [s for s in selected if s not in SUITES]
    if bad:
        print("RUN_ALL FAIL: --only names unknown suite(s): %s" % ", ".join(bad))
        sys.exit(2)

    failures = []
    for suite in selected:
        token = SUITES[suite]
        cmd = [args.blender, "--background", "--factory-startup",
               "--python", os.path.join(HERE, suite)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=TIMEOUT_SECS)
            out = (proc.stdout or "") + (proc.stderr or "")
            code = proc.returncode
        except subprocess.TimeoutExpired:
            out, code = "", None
        # Whole-line anchor, not a substring: Blender exits 0 on an unhandled
        # script exception, so the token is the only trustworthy signal — and a
        # SyntaxError beside the OK print echoes that source line into the
        # traceback, where a substring match would read the crash as a pass.
        token_ok = any(ln.strip().startswith("%s OK" % token)
                       for ln in out.splitlines())
        ok = code == 0 and token_ok
        print("RUN_ALL %-28s %s (exit=%s, token=%s)"
              % (suite, "PASS" if ok else "FAIL", code,
                 "seen" if token_ok else "MISSING"))
        if not ok:
            failures.append(suite)
            tail = [ln for ln in out.splitlines() if ln.strip()][-15:]
            for ln in tail:
                print("    | " + ln)

    print("RUN_ALL %d/%d suites passed" % (len(selected) - len(failures), len(selected)))
    if failures:
        print("RUN_ALL FAIL:", ", ".join(failures))
        sys.exit(1)
    if args.only:
        # Never the gate's verdict line — anything grepping for the gate must
        # not mistake a hand-picked subset for a full pass.
        print("RUN_ALL PARTIAL %d/%d — not the gate"
              % (len(selected), len(SUITES)))
    else:
        print("RUN_ALL OK")


if __name__ == "__main__":
    main()
