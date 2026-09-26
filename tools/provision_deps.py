"""Fill the gitignored dependency folders the weight-transfer core imports from.

Run under any Python (no bpy):

    python tools/provision_deps.py [--blender <path-to-blender.exe>]

The Blender binary comes from ``--blender`` or the ``BLENDER`` env var, as in
``tests/run_all.py``. Blender's own bundled interpreter runs pip, so the wheels
match the ABI the doors run under. Two folders are filled:

* ``avatarprep/wheels/`` - the wheel files ``blender_manifest.toml`` lists, for an
  extension build (``blender --command extension build``).
* ``deps/`` - those wheels installed with ``--no-deps``, which ``cli/_common.py``'s
  ``ensure_deps()`` appends to ``sys.path`` for the headless doors.

``--no-deps`` keeps numpy out: Blender bundles its own, and a second copy on the
path would shadow nothing but could mislead. The pins below and the manifest's
``wheels`` list move together.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

PINS = ["scipy==1.18.1", "robust-laplacian==1.0.0"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(REPO, "deps")
WHEELS = os.path.join(REPO, "avatarprep", "wheels")


def bundled_python(blender):
    """Blender's bundled interpreter: ``<blender dir>/<version>/python/bin/python(.exe)``."""
    root = os.path.dirname(os.path.abspath(blender))
    hits = sorted(glob.glob(os.path.join(root, "*", "python", "bin", "python.exe"))
                  + glob.glob(os.path.join(root, "*", "python", "bin", "python3*")))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(prog="provision_deps")
    ap.add_argument("--blender", default=os.environ.get("BLENDER"))
    args = ap.parse_args()
    if not args.blender or not os.path.isfile(args.blender):
        print("PROVISION FAIL: no Blender binary; pass --blender or set BLENDER (got %r)" % args.blender)
        sys.exit(2)
    py = bundled_python(args.blender)
    if py is None:
        print("PROVISION FAIL: no bundled python under %s/<version>/python/bin"
              % os.path.dirname(os.path.abspath(args.blender)))
        sys.exit(2)
    for d in (DEPS, WHEELS):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    steps = [
        [py, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--dest", WHEELS] + PINS,
        [py, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", "--no-index",
         "--find-links", WHEELS, "--target", DEPS] + PINS,
    ]
    for cmd in steps:
        print("PROVISION:", " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            print("PROVISION FAIL: pip exited non-zero; deps/ is incomplete")
            sys.exit(1)
    wheels = sorted(os.path.basename(p) for p in glob.glob(os.path.join(WHEELS, "*.whl")))
    print("PROVISION: wheels %s" % ", ".join(wheels))
    print("PROVISION OK: %s and %s filled for %s" % (DEPS, WHEELS, py))


if __name__ == "__main__":
    main()
