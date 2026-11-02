"""Headless CLI: multi-angle Workbench contact-sheet render of a .blend's render-visible meshes.

Run:
  blender --background --factory-startup --python cli/render_mesh.py -- \
      --in <file.blend> [--label <name>] [--only a,b,c] \
      [--angles front,back] [--shading solid|vertexcolor] [--resolution 1024] \
      [--out <dir>]

Exit 0 when ``avatarprep.core.render_mesh.render()`` returns an ``=> OK`` line; 1 on an
``=> FAIL`` line (a ran-but-failed refusal the agent acts on); 2 on bad args, an ``--in``
that fails to open, or render() raising. Every exit-2 path prints an in-grammar line:
``AVATARPREP: rendermesh ? => FAIL: ...``, or ``AVATARPREP: ERROR ...`` from ``open_blend``
(a bad ``--in``) or ``run_cli`` (a crash).
"""
import os
import sys
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep, open_blend, run_cli


class _Parser(argparse.ArgumentParser):
    """Argparse errors (missing --in, non-int --resolution) exit IN-GRAMMAR, not bare usage."""
    def error(self, message):
        print("AVATARPREP: rendermesh ? => FAIL: bad args: %s" % message)
        sys.exit(2)


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = _Parser(prog="render_mesh")
    p.add_argument("--in", dest="in_path", required=True,
                   help=".blend to open read-only and render; never saved")
    p.add_argument("--label", dest="label", default=None,
                   help="Name for the sheet, carried into the summary line and filename")
    p.add_argument("--only", dest="only", default=None,
                   help="Comma-separated object names to render instead of the whole scene "
                        "(so a name containing a comma cannot be targeted — rename it)")
    p.add_argument("--angles", dest="angles", default=None,
                   help="Comma-separated camera angles, e.g. front,back; omit for the "
                        "default set")
    p.add_argument("--shading", dest="shading", default="solid",
                   help="Workbench shading: solid or vertexcolor (default solid)")
    p.add_argument("--resolution", dest="resolution", type=int, default=1024,
                   help="Per-tile pixel resolution (default 1024)")
    p.add_argument("--out", dest="out_dir", default=None,
                   help="Directory to write the sheet into. Omit for the self-pruning temp "
                        "home; an explicit directory is yours to manage and is never pruned")
    return p.parse_args(argv)


def _split(value):
    """Comma-split a cli arg into a stripped, non-empty list; None/empty -> None."""
    if not value:
        return None
    items = [s.strip() for s in value.split(",") if s.strip()]
    return items or None


def main():
    args = _parse_args()
    import bpy

    open_blend(args.in_path, writes=False)

    try:
        enable_avatarprep()
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
            out_dir=os.path.abspath(args.out_dir) if args.out_dir else None,
        )
    except Exception as e:
        print("AVATARPREP: rendermesh ? => FAIL: %s" % e)
        sys.exit(2)

    print(line)
    # classify on the FAIL marker, not the OK substring: a refusal can echo a raw arg (e.g. an angle
    # literally "=> OK") into its reason, but sanitized labels/notes can never contain "=> FAIL:".
    sys.exit(1 if "=> FAIL:" in line else 0)


if __name__ == "__main__":
    run_cli(main, "render_mesh")
