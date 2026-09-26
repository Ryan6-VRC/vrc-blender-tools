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

Both are built in temporary folders and swapped in only when pip succeeded and the
downloaded file names equal the manifest's ``wheels`` list, so a failed or offline
run leaves the previous folders working. ``--no-deps`` keeps numpy out: Blender
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


def bundled_python(blender):
    """Blender's bundled interpreter: ``<blender dir>/<version>/python/bin/python(.exe)``."""
    root = os.path.dirname(os.path.abspath(blender))
    hits = sorted(glob.glob(os.path.join(root, "*", "python", "bin", "python.exe"))
                  + glob.glob(os.path.join(root, "*", "python", "bin", "python3*")))
    return hits[0] if hits else None


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
        for new, dest in ((new_wheels, WHEELS), (new_deps, DEPS)):
            shutil.rmtree(dest, ignore_errors=True)
            shutil.move(new, dest)
    print("PROVISION: wheels %s" % ", ".join(got))
    print("PROVISION OK: %s and %s filled for %s" % (DEPS, WHEELS, py))


if __name__ == "__main__":
    main()
