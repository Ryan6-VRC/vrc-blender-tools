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

    if armature_obj is not None:
        from . import scene_utils
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

    bpy.ops.export_scene.fbx('EXEC_DEFAULT', **kwargs)
    return filepath
