"""Headless CLI: multi-angle Workbench contact-sheet render of a .blend's render-visible meshes.

Run:
  blender --background --factory-startup --python cli/render_mesh.py -- \
      --in <file.blend> [--label <name>] [--only a,b,c] \
      [--angles front,back] [--shading solid|vertexcolor] [--resolution 1024]

Opens ``--in`` read-only via wm.open_mainfile (never saved), calls
``avatarprep.core.render_mesh.render()``, prints the one-line summary, and exits:
  0 — render() returned an ``=> OK`` line.
  1 — render() returned an ``=> FAIL`` line (a ran-but-failed refusal the agent acts on).
  2 — setup/infra: --in failed to open, or render() raised unexpectedly.

``--only`` and ``--angles`` split on comma into lists before the call (so an object name
containing a comma cannot be targeted — rename it, or use the default whole-scene render).
A bare traceback must never escape the result grammar: exit-2 paths emit an in-grammar
``AVATARPREP: rendermesh ? => FAIL: ...`` line.
"""
import os
import sys
import argparse


class _Parser(argparse.ArgumentParser):
    """Argparse errors (missing --in, non-int --resolution) exit IN-GRAMMAR, not bare usage —
    a traceback/usage dump must never escape the AVATARPREP: result grammar."""
    def error(self, message):
        print("AVATARPREP: rendermesh ? => FAIL: bad args: %s" % message)
        sys.exit(2)


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = _Parser(prog="render_mesh")
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
        print("AVATARPREP: rendermesh ? => FAIL: could not open --in: %s" % e)
        sys.exit(2)

    try:
        _enable_avatarprep()
        from avatarprep.core.render_mesh import render
    except Exception as e:
        print("AVATARPREP: rendermesh ? => FAIL: could not load avatarprep: %s" % e)
        sys.exit(2)

    try:
        line = render(
            label=args.label,
            only=_split(args.only),
            angles=_split(args.angles),
            shading=args.shading,
            resolution=args.resolution,
        )
    except Exception as e:
        print("AVATARPREP: rendermesh ? => FAIL: %s" % e)
        sys.exit(2)

    print(line)
    # classify on the FAIL marker, not the OK substring: a refusal can echo a raw arg (e.g. an angle
    # literally "=> OK") into its reason, but sanitized labels/notes can never contain "=> FAIL:".
    sys.exit(1 if "=> FAIL:" in line else 0)


if __name__ == "__main__":
    main()
