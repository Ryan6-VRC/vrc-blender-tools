"""Fill the gitignored dependency folders the weight-transfer core imports from.

Run under any Python 3.11+ (no bpy):

    python tools/provision_deps.py [--blender <path-to-blender.exe>]

The Blender binary comes from ``--blender`` or the ``BLENDER`` env var, as in
``tests/run_all.py``. Blender's own bundled interpreter runs pip, so the wheels
match the ABI the doors run under. Two folders are filled:

* ``avatarprep/wheels/`` - the wheel files ``blender_manifest.toml`` lists, for an
  extension build (``blender --command extension build``).
* ``deps/`` - those wheels installed with ``--no-deps``, which ``cli/_common.py``'s
  ``ensure_deps()`` appends to ``sys.path`` for the headless doors.

Both are built in temporary folders and swapped in together, only when pip succeeded
and the downloaded file names equal the manifest's ``wheels`` list, so a failed or
offline run leaves the previous folders working (``swap_in`` owns how the swap keeps
that promise when a folder is held open). ``--no-deps`` keeps numpy out: Blender
bundles its own. The pins below and the manifest's ``wheels`` list move together.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib

PINS = ["scipy==1.18.1", "robust-laplacian==1.0.0"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(REPO, "deps")
WHEELS = os.path.join(REPO, "avatarprep", "wheels")
MANIFEST = os.path.join(REPO, "avatarprep", "blender_manifest.toml")
ASIDE = ".provision-old"


def bundled_python(blender):
    """Blender's bundled interpreter: ``<blender dir>/<version>/python/bin/python(.exe)``."""
    root = os.path.dirname(os.path.abspath(blender))
    hits = sorted(glob.glob(os.path.join(root, "*", "python", "bin", "python.exe"))
                  + glob.glob(os.path.join(root, "*", "python", "bin", "python3*")))
    return hits[0] if hits else None


def swap_in(pairs):
    """Move each staged ``(new, dest)`` folder into place, all or none. Every existing ``dest``
    is renamed to ``dest + ASIDE`` before any ``new`` moves, so a folder a running Blender or
    a scanner holds open fails the swap with nothing replaced; a failure after that moves the
    placed ``new`` folders back and renames every original home. ``new`` sits on ``dest``'s
    volume, so each move is a rename. Returns the set-aside folders it could not delete once
    all are in place. Raises ``OSError`` with every ``dest`` as it was, or ``RuntimeError``
    naming what the rollback could not restore."""
    aside, placed = [], []
    try:
        for _, dest in pairs:
            if os.path.lexists(dest):
                os.rename(dest, dest + ASIDE)
                aside.append(dest)
        for new, dest in pairs:
            os.rename(new, dest)
            placed.append((new, dest))
    except OSError as e:
        stuck = []
        for new, dest in reversed(placed):
            try:
                os.rename(dest, new)
            except OSError:
                stuck.append("%s (the new folder is still there)" % dest)
        for dest in reversed(aside):
            try:
                os.rename(dest + ASIDE, dest)
            except OSError:
                stuck.append("%s (the previous folder is at %s)" % (dest, dest + ASIDE))
        if stuck:
            raise RuntimeError("the swap failed (%s) and could not be rolled back: %s"
                               % (e, "; ".join(stuck)))
        raise
    left = []
    for dest in aside:
        shutil.rmtree(dest + ASIDE, ignore_errors=True)
        if os.path.lexists(dest + ASIDE):
            left.append(dest + ASIDE)
    return left


def fail(msg, code=1):
    print("PROVISION FAIL: %s; deps/ and avatarprep/wheels/ were left as they were" % msg)
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser(prog="provision_deps")
    ap.add_argument("--blender", default=os.environ.get("BLENDER"))
    args = ap.parse_args()
    if not args.blender or not os.path.isfile(args.blender):
        fail("no Blender binary; pass --blender or set BLENDER (got %r)" % args.blender, 2)
    py = bundled_python(args.blender)
    if py is None:
        fail("no bundled python under %s/<version>/python/bin" % os.path.dirname(os.path.abspath(args.blender)), 2)
    with open(MANIFEST, "rb") as fh:
        listed = sorted(os.path.basename(w) for w in tomllib.load(fh).get("wheels", []))
    if not listed:
        fail("%s lists no wheels" % MANIFEST, 2)

    with tempfile.TemporaryDirectory(dir=REPO, prefix=".provision-") as tmp:
        new_wheels, new_deps = os.path.join(tmp, "wheels"), os.path.join(tmp, "deps")
        steps = [
            [py, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--dest", new_wheels] + PINS,
            [py, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", "--no-index",
             "--find-links", new_wheels, "--target", new_deps] + PINS,
        ]
        for cmd in steps:
            print("PROVISION:", " ".join(cmd))
            if subprocess.run(cmd).returncode != 0:
                fail("pip exited non-zero")
        got = sorted(os.path.basename(p) for p in glob.glob(os.path.join(new_wheels, "*.whl")))
        if got != listed:
            fail("downloaded %s but blender_manifest.toml lists %s; move PINS and the manifest's "
                 "wheels together" % (got, listed))
        try:
            left = swap_in([(new_wheels, WHEELS), (new_deps, DEPS)])
        except OSError as e:
            fail("could not swap the new folders in (%s); close any Blender using deps/, or delete a "
                 "leftover *%s folder, and rerun" % (e, ASIDE))
        except RuntimeError as e:
            print("PROVISION FAIL: %s; move those folders back by hand" % e)
            sys.exit(1)
    for path in left:
        print("PROVISION WARN: the previous folder %s could not be deleted; delete it by hand" % path)
    print("PROVISION: wheels %s" % ", ".join(got))
    print("PROVISION OK: %s and %s filled for %s" % (DEPS, WHEELS, py))


if __name__ == "__main__":
    main()
