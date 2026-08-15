"""Import an FBX into Blender and report a sanity snapshot.

Uses Blender's current FBX importer ``bpy.ops.wm.fbx_import`` — the operator the
File > Import menu labels "FBX" in Blender 5.x. The deprecated legacy Python
importer (``bpy.ops.import_scene.fbx``, menu "FBX (legacy)") is intentionally NOT
used: its ``automatic_bone_orientation`` reorients some bones (hips, upper arms)
~90 deg from the source FBX, which silently corrupts bone-local pose operations
downstream (proportioning, rest-pose work). The new importer keeps the source
orientation and has no such option, matching the hand-authored reference rigs.

Supports both headless (``blender --background``) runs and windowed/MCP-driven
sessions where a VIEW_3D area is available.
"""

from typing import Any, Dict

import bpy

from . import scene_utils, measure


def _read_unit_scale_factor(path: str):
    """The source file's ``GlobalSettings.UnitScaleFactor`` (cm per file unit:
    ``100.0`` = meter-unit file, ``1.0`` = cm-unit file), or ``None`` if
    unreadable. Read from the FILE because the importer normalizes both classes
    into identical scene state — nothing scene-side records which the vendor
    shipped. Diagnostic only: never fails an import."""
    try:
        from io_scene_fbx import parse_fbx
        root, _ = parse_fbx.parse(path)
        for gs in (e for e in root.elems if e.id == b"GlobalSettings"):
            for p70 in (c for c in gs.elems if c.id == b"Properties70"):
                for p in p70.elems:
                    if p.props[0] == b"UnitScaleFactor":
                        return float(p.props[4])
    except Exception:
        pass
    return None


def import_fbx(path: str, **settings) -> Dict[str, Any]:
    """Import ``path`` as FBX and return an :func:`observe_import` snapshot,
    plus ``unit_scale_factor`` — the source file's unit class (see
    :func:`_read_unit_scale_factor`; ``export_unity_fbx``'s docstring says what
    to make of it).

    Any keyword in ``settings`` is forwarded to ``bpy.ops.wm.fbx_import`` (e.g.
    ``global_scale``, ``use_custom_normals``, ``ignore_leaf_bones``).

    Works both headless (``--background``) and in a running Blender with a
    VIEW_3D area present (e.g. when driven over MCP).
    """
    kwargs = dict(filepath=path)
    kwargs.update(settings)

    # Find a window whose screen has a VIEW_3D area (scan ALL windows, not just
    # the first — a VIEW_3D may live in a second window).
    wm = bpy.context.window_manager
    win, area = None, None
    for w in (wm.windows if wm else []):
        a = next((a for a in w.screen.areas if a.type == 'VIEW_3D'), None)
        if a:
            win, area = w, a
            break

    # The windowed/MCP branch is the non-clean case (the session may already hold
    # objects), so capture what exists before importing and diff afterwards.
    before = set(bpy.data.objects)

    def _finish(new_objects):
        """Stamp, then observe. Runs under the SAME context the import ran under.

        ``observe_import`` resolves ``bpy.context.view_layer`` (to force an update and
        get a depsgraph), and under ``temp_override(window=…)`` the scene and view layer
        follow that window. Since the branch above deliberately scans EVERY window for a
        VIEW_3D, a two-window session can import into one window's view layer while the
        caller's context points at another — where the new objects are absent, and
        ``evaluated_get`` would hand back unevaluated data with no error. Observing
        inside the override keeps both halves on one view layer. (The old ``bound_box``
        read was view-layer-independent, so this exposure arrived with the fix.)"""
        # Stamp every newly-imported armature with the reserved ``unproportioned`` origin
        # state (a fresh import is, by definition, unproportioned — the as-shipped shape).
        # Base lineage is NOT touched here — base is a deliberate agent assertion made
        # only through the stamp_base door, never guessed at import. A fresh import reads
        # base=absent (honest/unknown).
        for arm in (o for o in new_objects if o.type == 'ARMATURE'):
            scene_utils.write_stamp(arm, scene_utils.STAMP_STATE, "unproportioned")
        return observe_import(new_objects)

    if win and area:                                  # windowed (MCP) path
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        ctx = {"window": win, "area": area}
        if region:
            ctx["region"] = region
        scene_utils.op_override(bpy.ops.wm.fbx_import, ctx, execution_context='EXEC_DEFAULT', **kwargs)
        with bpy.context.temp_override(**ctx):
            snap = _finish([o for o in bpy.data.objects if o not in before])
    else:                                             # no VIEW_3D context (headless OR windowed without a VIEW_3D area)
        bpy.ops.wm.fbx_import(**kwargs)
        snap = _finish([o for o in bpy.data.objects if o not in before])

    snap["unit_scale_factor"] = _read_unit_scale_factor(path)
    return snap


def observe_import(objects=None) -> Dict[str, Any]:
    """Return a sanity snapshot of the imported objects.

    Args:
        objects: Iterable of objects to report on. If ``None``, falls back to
            every object in the .blend (``bpy.data.objects``) — correct only for
            a clean session (a fresh headless import). Callers that import into a
            session already holding objects must pass the newly-created set.

    Keys:
      * ``armatures``         — number of ARMATURE objects
      * ``meshes``            — number of MESH objects
      * ``bones``             — total bone count across all armatures (0 if none)
      * ``bones_per_armature``— list of per-armature bone counts
      * ``shapekeys``         — total shape-key count across all meshes (basis excluded)
      * ``height_m``          — world-space height in metres of the meshes' EVALUATED
                                geometry (0 when none has any). Rounded to 4 dp, which
                                is now the dominant error on accessory-scale objects: a
                                fixed +-0.05 mm reads ~0.04% on a 0.1 m prop against
                                ~0.004% on a body. Fine for a gut-check, not a tolerance.
      * ``unparented_meshes`` — names of MESH objects with no parent
      * ``unevaluated_meshes``— names of meshes the depsgraph will not evaluate, whose
                                measurement is therefore silently their UNDEFORMED
                                shape. The one blind spot no measure here escapes, so it
                                is named rather than left looking like a clean read
                                (``rest_pose.unevaluated_meshes`` owns the predicate).
    """
    objs = list(bpy.data.objects) if objects is None else list(objects)
    arms = [o for o in objs if o.type == 'ARMATURE']
    meshes = [o for o in objs if o.type == 'MESH']
    bones_per_armature = [len(a.data.bones) for a in arms]

    bounds = measure._world_bounds(meshes)
    height_m = (round(bounds["max"][2] - bounds["min"][2], 4)
                if bounds["min"] is not None else 0)

    total_sk = sum(
        (len(m.data.shape_keys.key_blocks) - 1) if m.data.shape_keys else 0
        for m in meshes
    )

    return {
        "armatures": len(arms),
        "meshes": len(meshes),
        "bones": sum(bones_per_armature),
        "bones_per_armature": bones_per_armature,
        "shapekeys": total_sk,
        "height_m": height_m,
        "unparented_meshes": [m.name for m in meshes if m.parent is None],
        "unevaluated_meshes": bounds["unevaluated"],
    }
