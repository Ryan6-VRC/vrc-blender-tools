"""Headless CLI: multi-angle Workbench contact-sheet render of a .blend's render-visible meshes.

Run:
  blender --background --factory-startup --python cli/mesh_grab.py -- \
      --in <file.blend> [--label <name>] [--only a,b,c] \
      [--angles front,back] [--shading solid|vertexcolor] [--resolution 1024]

Opens ``--in`` read-only via wm.open_mainfile (never saved), calls
``avatarprep.core.mesh_grab.grab()``, prints the one-line summary, and exits:
  0 — grab() returned an ``=> OK`` line.
  1 — grab() returned an ``=> FAIL`` line (a ran-but-failed refusal the agent acts on).
  2 — setup/infra: --in failed to open, or grab() raised unexpectedly.

``--only`` and ``--angles`` split on comma into lists before the call (so an object name
containing a comma cannot be targeted — rename it, or use the default whole-scene render).
A bare traceback must never escape the result grammar: exit-2 paths emit an in-grammar
``AVATARPREP: meshgrab ? => FAIL: ...`` line.
"""
import os
import sys
import argparse


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="mesh_grab")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--label", dest="label", default=None)
    p.add_argument("--only", dest="only", default=None)
    p.add_argument("--angles", dest="angles", default=None)
    p.add_argument("--shading", dest="shading", default="solid")
    p.add_argument("--resolution", dest="resolution", type=int, default=1024)
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


def _split(value):
    """Comma-split a cli arg into a stripped, non-empty list; None/empty -> None."""
    if not value:
        return None
    items = [s.strip() for s in value.split(",") if s.strip()]
    return items or None


def main():
    args = _parse_args()
    import bpy

    try:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.in_path))
    except Exception as e:
        print("AVATARPREP: meshgrab ? => FAIL: could not open --in: %s" % e)
        sys.exit(2)

    _enable_avatarprep()
    from avatarprep.core.mesh_grab import grab

    try:
        line = grab(
            label=args.label,
            only=_split(args.only),
            angles=_split(args.angles),
            shading=args.shading,
            resolution=args.resolution,
        )
    except Exception as e:
        print("AVATARPREP: meshgrab ? => FAIL: %s" % e)
        sys.exit(2)

    print(line)
    sys.exit(0 if "=> OK" in line else 1)


if __name__ == "__main__":
    main()
