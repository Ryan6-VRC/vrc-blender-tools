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


def open_blend(path, *, writes, force_load_repair=False):
    """Open ``path`` as this CLI's subject and apply the load-repair policy.

    The one opener for every CLI here. A bare ``bpy.ops.wm.open_mainfile`` dies with a
    traceback on a file Blender loaded fine, and Blender exits 0 on an unhandled
    ``--python`` exception — so that traceback reads to a caller as a clean run.

    ``writes`` says whether THIS INVOCATION will save a deliverable: a door that can
    save passes ``writes=not args.whatif``. A repaired load blocks only the write,
    never a preview or a read. Returns the repair detail (str) or ``None``, so the
    caller can stamp it into its ``--report``.

    The verdict logic — and why a repair is refused rather than warned — lives in
    ``scene_utils.classify_open`` / ``open_policy``; this is the grammar + exit shim."""
    import bpy
    from avatarprep.core import scene_utils

    requested = os.path.abspath(path)
    raised = None
    try:
        bpy.ops.wm.open_mainfile(filepath=requested)
    except RuntimeError as e:
        raised = str(e)

    # Did the load land on the file we asked for? ``os.path.samefile`` stats BOTH
    # paths and raises on a missing one — the likeliest bad ``--in`` — so unguarded it
    # would crash instead of classifying, and exit 0 doing it.
    try:
        landed = bool(bpy.data.filepath) and os.path.samefile(bpy.data.filepath, requested)
    except OSError:
        landed = False

    # Blender substitutes placeholders for an unresolvable linked library and only
    # WARNS, so this class never reaches ``raised`` and must be read from state.
    missing_libs = [lib.filepath for lib in bpy.data.libraries
                    if lib.filepath and not os.path.exists(bpy.path.abspath(lib.filepath))]

    kind, detail = scene_utils.classify_open(raised, landed, missing_libs)
    verdict = scene_utils.open_policy(kind, writes=writes,
                                      force_load_repair=force_load_repair)

    if verdict == "error":
        print("AVATARPREP: ERROR cannot open %s: %s" % (requested, detail or "load failed"))
        sys.exit(2)
    if verdict == "refuse":
        print("AVATARPREP: open REFUSED — %s loaded with repairs; saving would persist "
              "Blender's alteration into the deliverable" % requested)
        print("AVATARPREP: OFFENDER", detail)
        print("AVATARPREP: REMEDY read the source with report_stamps --shapekeys to see "
              "what the repair dropped, then re-save a repaired source — or pass "
              "--force-load-repair to accept the repaired state deliberately")
        print("AVATARPREP: nothing was mutated; --out NOT written.")
        sys.exit(1)
    if verdict == "forced":
        print("AVATARPREP: FORCED LOAD REPAIR", detail)
    elif verdict == "warn":
        print("AVATARPREP: WARNING %s loaded with repairs — Blender altered data as it "
              "read the file, so every number below is measured on the repaired state: %s"
              % (requested, detail))
    return detail or None


def add_force_load_repair(p):
    """Add ``--force-load-repair`` to a door that saves. One definition, so the flag's
    help text cannot drift between doors."""
    p.add_argument("--force-load-repair", dest="force_load_repair", action="store_true",
                   help="Proceed even though Blender repaired the source as it read it, "
                        "persisting that repair into the output. The refusal names what "
                        "was repaired — read it first")


def run_cli(main, tool):
    """Crash-safe entry for a CLI's ``main()``.

    Blender exits 0 on an unhandled ``--python`` exception, so an uncaught raise reads
    as ``0 = did the thing`` with no ``--out`` written — the same false-clean shape the
    load-repair refusal exists to close. ``SystemExit`` passes through: a CLI that
    already printed its verdict and exited is not a crash. ``tests/_harness.py`` is
    the suite-side half of the same contract."""
    import traceback
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("AVATARPREP: ERROR %s crashed before reaching a verdict" % tool)
        sys.exit(2)


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
