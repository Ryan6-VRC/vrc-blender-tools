"""Export the scene to a Unity/VRChat-correct FBX.

Thin wrapper over ``bpy.ops.export_scene.fbx`` with the parameter set Unity /
VRChat expect for avatar import (each value is documented inline below).
"""

from typing import Optional

import bpy


def export_unity_fbx(filepath: str,
                     armature_obj: Optional[bpy.types.Object] = None,
                     object_types=None,
                     use_mesh_modifiers: bool = False,
                     add_leaf_bones: bool = False,
                     bake_anim: bool = False,
                     apply_scale_options: str = 'FBX_SCALE_ALL',
                     path_mode: str = 'COPY',
                     embed_textures: bool = True,
                     use_selection: bool = False,
                     keep_object_rotation: bool = False,
                     **extra) -> str:
    """Export ``filepath`` as an FBX using the CATS / Unity recipe.

    The defaults are the VRChat-correct settings:
      * ``object_types={'EMPTY', 'ARMATURE', 'MESH', 'OTHER'}``
      * ``use_mesh_modifiers=False``
      * ``add_leaf_bones=False``
      * ``bake_anim=False``
      * ``apply_scale_options='FBX_SCALE_ALL'``
      * ``embed_textures=True``
      * ``path_mode='COPY'`` (required for embedding to work)

    **Orientation:** an armature *object* rotation is treated as importer
    residue, not content — ``wm.fbx_import`` represents a source FBX's axis
    convention as an object rotation (e.g. -180° Z), and the exporter re-derives
    its own conversion, so carrying it through double-counts: the file gains an
    extra 180° and Unity shows the avatar backwards. A Blender re-import cannot
    see this (the importer symmetrically undoes it); parsing the file can, and
    ``tests/test_fbx_orientation.py`` pins it against the Felis fixture. Each
    exported armature's object rotation is therefore cleared UNAPPLIED for the
    export (bone/mesh data untouched) and restored after. A deliberately rotated
    armature is the rare exception: pass ``keep_object_rotation=True``.

    **Scale:** ``FBX_SCALE_ALL`` is this repo's canonical export layout — a
    ``UnitScaleFactor=100`` (meter-unit) file with no compensating node scales,
    identical to what meter-unit vendors ship. Vendors also ship cm-unit
    (``UnitScaleFactor=1``) files — the import snapshot's ``unit_scale_factor``
    names the source's class — but owned exports do NOT mimic the source: Unity
    normalizes file units at import, and world-space parity there is the owning
    skill's gate. (``FBX_SCALE_NONE`` instead writes a cm-unit file with 100x
    root node scales; measured, and not what any probed vendor ships.)

    ``armature_obj`` scopes the export to one rig: it selects that armature plus
    its bound meshes and exports selection-only. Because a scoped export is by
    construction an *owned* re-export (the owned meshes reuse the vendor materials
    by GUID in Unity, so nothing needs embedding) it also forces
    ``path_mode='STRIP'`` and ``embed_textures=False`` — otherwise Blender would
    re-embed textures by the vendor author's unresolvable absolute paths, emitting
    warnings and junk sub-assets. With ``armature_obj=None`` the whole scene is
    exported (``use_selection=False``) on the VRChat embed recipe, matching CATS.

    Returns the filepath written.
    """
    if object_types is None:
        object_types = {'EMPTY', 'ARMATURE', 'MESH', 'OTHER'}

    from . import scene_utils

    # ``select_all`` (and the FBX exporter) poll for OBJECT mode; a caller that left
    # the scene in POSE/EDIT — apply_proportion_edge exits in POSE on its object-only
    # edge path — otherwise crashes ``select_all.poll() failed, context is incorrect``.
    # Force OBJECT here so apply-then-export in one script is safe for every caller.
    active = bpy.context.view_layer.objects.active
    if active is not None and active.mode != 'OBJECT':
        scene_utils.op_override(bpy.ops.object.mode_set,
                                {'active_object': active, 'object': active},
                                mode='OBJECT')

    if armature_obj is not None:
        bpy.ops.object.select_all(action='DESELECT')
        armature_obj.select_set(True)
        for m in scene_utils.get_bound_meshes(armature_obj):
            m.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        use_selection = True
        path_mode = 'STRIP'
        embed_textures = False

    kwargs = dict(
        filepath=filepath,
        object_types=object_types,
        use_mesh_modifiers=use_mesh_modifiers,
        add_leaf_bones=add_leaf_bones,
        bake_anim=bake_anim,
        apply_scale_options=apply_scale_options,
        path_mode=path_mode,
        embed_textures=embed_textures,
        use_selection=use_selection,
    )
    kwargs.update(extra)

    # Clear importer-residue object rotation on every exported armature (see
    # docstring), restore after. Child meshes ride along (their local transforms
    # are relative to the armature), so nothing moves relative to the rig.
    if keep_object_rotation:
        cleared = []
    elif armature_obj is not None:
        cleared = [armature_obj]
    else:
        cleared = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    saved = [(o, o.rotation_euler[:], o.rotation_quaternion[:],
              o.rotation_axis_angle[:]) for o in cleared]
    try:
        for o in cleared:
            o.rotation_euler = (0.0, 0.0, 0.0)
            o.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            o.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        bpy.ops.export_scene.fbx('EXEC_DEFAULT', **kwargs)
    finally:
        for o, eul, quat, aa in saved:
            o.rotation_euler = eul
            o.rotation_quaternion = quat
            o.rotation_axis_angle = aa
    return filepath
