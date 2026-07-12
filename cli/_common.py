"""Shared plumbing for the headless CLIs.

Only importable AFTER a CLI's structural ``sys.path`` insert (its first lines) —
a fresh ``--background --python`` process has no repo path, so the 2-line insert
stays per-file and everything that follows it lives here.
"""
import os
import sys
import json


def enable_avatarprep():
    """Import the bundled ``avatarprep`` source package and register it."""
    import avatarprep
    try:
        avatarprep.register()
    except Exception:
        # Already registered in this session; fine.
        pass
    return avatarprep


def resolve_arm(name, arg):
    """Resolve ``--<arg> <name>`` to an armature object; in-grammar ERROR + exit 2."""
    import bpy
    obj = bpy.context.scene.objects.get(name)
    if obj is None or obj.type != 'ARMATURE':
        print("AVATARPREP: ERROR --%s %r is not an armature in this scene" % (arg, name))
        sys.exit(2)
    return obj


def write_report(path, data):
    """Write a ``--report`` JSON (makedirs first); in-grammar ERROR + exit 2 on failure."""
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


def kv(items):
    """Parse repeated ``KEY=VALUE`` args into a dict."""
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k] = v
    return out
