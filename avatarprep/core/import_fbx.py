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
from mathutils import Vector

from . import scene_utils


def import_fbx(path: str, **settings) -> Dict[str, Any]:
    """Import ``path`` as FBX and return an :func:`observe_import` snapshot.

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

    if win and area:                                  # windowed (MCP) path
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        ctx = {"window": win, "area": area}
        if region:
            ctx["region"] = region
        scene_utils.op_override(bpy.ops.wm.fbx_import, ctx, execution_context='EXEC_DEFAULT', **kwargs)
    else:                                             # no VIEW_3D context (headless OR windowed without a VIEW_3D area)
        bpy.ops.wm.fbx_import(**kwargs)

    new_objects = [o for o in bpy.data.objects if o not in before]

    # Stamp every newly-imported armature with the reserved ``unproportioned`` origin
    # state (a fresh import is, by definition, unproportioned — the as-shipped shape).
    # Base lineage is NOT touched here — base is a deliberate agent assertion made only
    # through the stamp_base door, never guessed at import. A fresh import reads
    # base=absent (honest/unknown).
    for arm in (o for o in new_objects if o.type == 'ARMATURE'):
        scene_utils.write_stamp(arm, scene_utils.STAMP_STATE, "unproportioned")

    return observe_import(new_objects)


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
      * ``height_m``          — world-space bounding-box height in metres (0 if no meshes)
      * ``unparented_meshes`` — names of MESH objects with no parent
    """
    objs = list(bpy.data.objects) if objects is None else list(objects)
    arms = [o for o in objs if o.type == 'ARMATURE']
    meshes = [o for o in objs if o.type == 'MESH']
    bones_per_armature = [len(a.data.bones) for a in arms]

    zmin, zmax = 1e9, -1e9
    for m in meshes:
        for c in m.bound_box:
            wz = (m.matrix_world @ Vector(c)).z
            zmin = min(zmin, wz)
            zmax = max(zmax, wz)

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
        "height_m": round(zmax - zmin, 4) if meshes else 0,
        "unparented_meshes": [m.name for m in meshes if m.parent is None],
    }
