"""Headless test for the shared CLI open path.

The POLICY is tested purely (classify_open's docstring says why).

The end-to-end tier covers only the failures a test can synthesize -- a missing file, a
non-.blend, a directory. The 'repaired' branch has no fixture: the one file known to
trigger a reported repair is a vendor-licensed avatar base body that cannot ship in a
public repo. To re-measure, open any .blend and compare the raise against
``bpy.data.filepath``.
"""
import os
import subprocess
import sys
import tempfile

import bpy

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_classify_open():
    from avatarprep.core.scene_utils import classify_open
    kind, detail = classify_open(None, True, [])
    check(kind == "clean" and detail == "", "clean: landed, no raise, no missing libs")

    kind, detail = classify_open("Error: boom\n", True, [])
    check(kind == "repaired" and "boom" in detail, "a raise on a landed load is a repair")
    check("\n" not in detail, "detail should be stripped of newlines for one-line grammar")

    # The class that never raises (scene_utils' File-open classification).
    kind, detail = classify_open(None, True, ["lib.blend"])
    check(kind == "repaired" and "lib.blend" in detail,
          "a missing linked library is a repair even with no raise")
    check("library:" in detail, "one missing lib should read singular, got %r" % detail)
    kind, detail = classify_open(None, True, ["a.blend", "b.blend"])
    check("libraries:" in detail, "two missing libs should read plural, got %r" % detail)

    kind, detail = classify_open("Error: boom", True, ["lib.blend"])
    check(kind == "repaired" and "boom" in detail and "lib.blend" in detail,
          "both repair channels should be reported together")

    # Not landing outranks everything: the previous main is still live.
    kind, detail = classify_open("Error: no such file", False, [])
    check(kind == "failed" and "no such file" in detail, "not landed is a failure")
    kind, _ = classify_open(None, False, ["lib.blend"])
    check(kind == "failed", "not landed outranks a missing library")


def test_open_policy():
    from avatarprep.core.scene_utils import open_policy
    expected = {
        # (kind, writes, force): verdict
        ("clean", False, False): "proceed", ("clean", False, True): "proceed",
        ("clean", True, False): "proceed", ("clean", True, True): "proceed",
        ("repaired", False, False): "warn", ("repaired", False, True): "warn",
        ("repaired", True, False): "refuse", ("repaired", True, True): "forced",
        ("failed", False, False): "error", ("failed", False, True): "error",
        ("failed", True, False): "error", ("failed", True, True): "error",
    }
    for (kind, writes, force), want in sorted(expected.items()):
        got = open_policy(kind, writes=writes, force_load_repair=force)
        check(got == want, "open_policy(%s, writes=%s, force=%s) = %s, want %s"
              % (kind, writes, force, got, want))
    # The asymmetry as its own assertion, so a regression names it.
    check(open_policy("repaired", writes=False, force_load_repair=False) == "warn"
          and open_policy("repaired", writes=True, force_load_repair=False) == "refuse",
          "a repair must block the write while leaving reads and previews open")


def _run_cli(args):
    """Drive a CLI in its own Blender process; returns (exit_code, output)."""
    blender = bpy.app.binary_path
    proc = subprocess.run([blender, "--background", "--factory-startup", "--python"]
                          + args, capture_output=True, text=True, timeout=300)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def test_end_to_end_failures():
    """The three failures a public repo can synthesize. All must be ERROR + exit 2."""
    root = _repo_root()
    door = os.path.join(root, "cli", "report_stamps.py")
    tmp = tempfile.mkdtemp(prefix="avatarprep_open_")

    missing = os.path.join(tmp, "no_such_file.blend")
    rc, out = _run_cli([door, "--", "--in", missing])
    check(rc == 2, "missing --in should exit 2, got %s" % rc)
    check("AVATARPREP: ERROR cannot open" in out,
          "missing --in should emit the in-grammar ERROR, got %r" % out[-400:])
    check("Traceback" not in out, "missing --in must not leak a traceback")

    not_a_blend = os.path.join(tmp, "notablend.txt")
    with open(not_a_blend, "w", encoding="utf-8") as fh:
        fh.write("this is not a blend file\n")
    rc, out = _run_cli([door, "--", "--in", not_a_blend])
    check(rc == 2, "non-.blend --in should exit 2, got %s" % rc)
    check("AVATARPREP: ERROR cannot open" in out, "non-.blend should emit ERROR")

    rc, out = _run_cli([door, "--", "--in", tmp])
    check(rc == 2, "directory --in should exit 2, got %s" % rc)
    check("AVATARPREP: ERROR cannot open" in out, "directory should emit ERROR")


def test_missing_after_a_successful_load():
    """The regression that a cold test cannot see.

    A cold process never reaches the samefile guard (open_blend's comment), because
    ``bpy.data.filepath`` is '' and short-circuits first; the previous main has to be live.
    This also pins that a failed open does NOT proceed against it."""
    root = _repo_root()
    tmp = tempfile.mkdtemp(prefix="avatarprep_open2_")
    good = os.path.join(tmp, "good.blend")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.save_as_mainfile(filepath=good)

    script = os.path.join(tmp, "seeded_open.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(
            "import sys, os\n"
            "sys.path.insert(0, %r)\n"
            "from cli._common import open_blend\n"
            "open_blend(%r, writes=False)\n"
            "print('SEEDED: first open landed')\n"
            "open_blend(%r, writes=False)\n"
            "print('SEEDED: REACHED CODE AFTER A FAILED OPEN')\n"
            % (root, good, os.path.join(tmp, "gone.blend")))
    rc, out = _run_cli([script])
    check("SEEDED: first open landed" in out, "the seeding open should have succeeded")
    check(rc == 2, "a missing file after a good load should exit 2, got %s" % rc)
    check("Traceback" not in out,
          "the samefile guard is missing: %r" % out[-500:])
    check("REACHED CODE AFTER A FAILED OPEN" not in out,
          "a failed open must exit, not fall through to the still-live previous main")


def test_crash_wrapper():
    """An unhandled exception inside a CLI must exit 2."""
    root = _repo_root()
    tmp = tempfile.mkdtemp(prefix="avatarprep_open3_")
    script = os.path.join(tmp, "boom.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "from cli._common import run_cli\n"
            "def main():\n"
            "    raise ValueError('synthetic')\n"
            "run_cli(main, 'synthetic_tool')\n" % root)
    rc, out = _run_cli([script])
    check(rc == 2, "an unhandled CLI exception should exit 2, got %s" % rc)
    check("crashed before reaching a verdict" in out,
          "the crash should be named in grammar, got %r" % out[-300:])
    check("ValueError" in out, "the traceback should be kept for triage")

    # SystemExit passes through untouched: a printed verdict is not a crash.
    script2 = os.path.join(tmp, "verdict.py")
    with open(script2, "w", encoding="utf-8") as fh:
        fh.write(
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "from cli._common import run_cli\n"
            "def main():\n"
            "    print('AVATARPREP: thing REFUSED - on purpose')\n"
            "    sys.exit(1)\n"
            "run_cli(main, 'synthetic_tool')\n" % root)
    rc, out = _run_cli([script2])
    check(rc == 1, "a deliberate sys.exit(1) must survive run_cli, got %s" % rc)
    check("crashed" not in out, "a printed verdict must not be reported as a crash")


def main():
    sys.path.insert(0, _repo_root())
    test_classify_open()
    test_open_policy()
    test_end_to_end_failures()
    test_missing_after_a_successful_load()
    test_crash_wrapper()
    if FAILURES:
        for f in FAILURES:
            print("OPENBLEND_TEST FAIL:", f)
        sys.exit(1)
    print("OPENBLEND_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "OPENBLEND_TEST")
