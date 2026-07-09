"""Shape-key-safe "apply current pose as rest pose" bake.

After :func:`apply_pose`, the armature's CURRENT pose becomes its REST
pose and every bound mesh's geometry is baked so it looks identical to the posed
result -- *without* losing shape-key information. A naive "add an armature
modifier and apply it" corrupts every non-active shape key, because applying a
modifier only re-evaluates the active key. We avoid that by evaluating each shape
key through the armature deformation independently and writing the resulting
deformed coordinates straight back into the key blocks.

Pure ``bpy``, headless-safe: no ``Operator`` subclasses, no UI/panel code, every
operator is invoked through :func:`scene_utils.op_override` with an explicit
context, and the caller's active object / selection / mode are saved and restored.
"""
from typing import List, Optional

import bpy
import numpy as np

from . import scene_utils


def _eval_coords(mesh_obj: bpy.types.Object, n: int) -> np.ndarray:
    """Flat ``3*n`` array of the mesh's CURRENT evaluated (armature-deformed)
    vertex coordinates, in the object's local space."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    arr = np.empty(n * 3, dtype=np.float64)
    eval_obj.data.vertices.foreach_get("co", arr)
    return arr


def _capture(mesh_obj: bpy.types.Object, armature_obj: bpy.types.Object):
    """Capture the deformed geometry for ``mesh_obj`` at the current pose.

    Returns ``(mesh_obj, method, shape_key_count, captured)`` where ``captured``
    is either a single flat coord array (no shape keys) or a list of one array
    per shape key (basis first), each holding that key's fully deformed positions.

    Only ``armature_obj``'s deformation is captured: during the snapshot we enable
    the mesh's ARMATURE modifier(s) that target ``armature_obj`` and disable every
    other modifier (a foreign-armature modifier or a stack modifier would
    otherwise contaminate the bake; a disabled target modifier would otherwise be
    captured undeformed). A mesh bound only by parenting has no such modifier, so
    nothing is enabled and its geometry is captured (and written back) unchanged.
    """
    me = mesh_obj.data
    n = len(me.vertices)

    # Isolate this armature's deformation. Snapshot every modifier's viewport flag
    # so we can restore it, then enable only the target-armature modifier(s).
    saved_show = [(mod, mod.show_viewport) for mod in mesh_obj.modifiers]
    for mod in mesh_obj.modifiers:
        mod.show_viewport = (mod.type == 'ARMATURE' and mod.object == armature_obj)

    try:
        sk = me.shape_keys
        if sk and sk.key_blocks:
            kbs = sk.key_blocks
            prev_show = mesh_obj.show_only_shape_key
            prev_idx = mesh_obj.active_shape_key_index
            prev_mute = [kb.mute for kb in kbs]
            prev_vg = [kb.vertex_group for kb in kbs]

            # Pin one key at a time at full value: unmuted AND with its vertex-group
            # mask cleared, so the captured .co is the key's UNMASKED deformed shape.
            # Leaving the mask on would let Blender re-apply it at runtime over the
            # already-masked baked coords -- scaling a fractional offset by w^2.
            for kb in kbs:
                kb.mute = False
                kb.vertex_group = ""
            mesh_obj.show_only_shape_key = True

            captured = []
            for i in range(len(kbs)):
                mesh_obj.active_shape_key_index = i
                bpy.context.view_layer.update()
                captured.append(_eval_coords(mesh_obj, n))

            mesh_obj.show_only_shape_key = prev_show
            mesh_obj.active_shape_key_index = prev_idx
            for kb, muted, vg in zip(kbs, prev_mute, prev_vg):
                kb.mute = muted
                kb.vertex_group = vg

            return (mesh_obj, "shapekeys", len(kbs), captured)

        return (mesh_obj, "geometry", 0, _eval_coords(mesh_obj, n))
    finally:
        for mod, show in saved_show:
            mod.show_viewport = show


def _apply_pose(armature_obj: bpy.types.Object) -> None:
    """Set the armature's current pose as its rest pose (headless).

    Caller-visible state (active object / selection / mode) is restored by the
    :class:`scene_utils.SavedSelection` guard in :func:`apply_pose`.
    """
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    ctx = {'active_object': armature_obj, 'object': armature_obj}
    if armature_obj.mode != 'POSE':
        scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='POSE')
    # selected=False (default) applies ALL bones regardless of selection state.
    scene_utils.op_override(bpy.ops.pose.armature_apply, ctx)
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')


def _write(mesh_obj: bpy.types.Object, method: str, captured) -> None:
    """Write the captured deformed geometry back. The rest pose is now identity,
    so the armature modifier leaves this geometry untouched -- the mesh now looks
    exactly like the posed result it was captured from."""
    me = mesh_obj.data
    if method == "shapekeys":
        kbs = me.shape_keys.key_blocks
        for i, kb in enumerate(kbs):
            kb.data.foreach_set("co", captured[i])
        # Keep the base mesh vertices consistent with the (deformed) basis key.
        me.vertices.foreach_set("co", captured[0])
    else:
        me.vertices.foreach_set("co", captured)
    me.update()


def apply_pose(armature_obj: bpy.types.Object,
               mesh_objs: Optional[List[bpy.types.Object]] = None) -> dict:
    """Bake the armature's current pose into the rest pose, shape-key-safely.

    For each bound mesh the deformed geometry is captured at the current pose
    (per shape key when present), the pose is applied as the new rest, and the
    captured geometry is written back so the visible result is unchanged while
    every shape key keeps producing its correct deformed offset.
    """
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        raise ValueError("apply_pose requires a valid ARMATURE object, got %r"
                         % (None if armature_obj is None else armature_obj.type))

    if mesh_objs is None:
        mesh_objs = scene_utils.get_bound_meshes(armature_obj)

    saved = scene_utils.SavedSelection()
    try:
        # Phase 1: capture deformed geometry for every mesh while still posed.
        plans = [_capture(mesh_obj, armature_obj) for mesh_obj in mesh_objs]

        # Phase 2: make the current pose the rest pose (bones reshape here).
        _apply_pose(armature_obj)

        # Phase 3: write the captured geometry back under the new identity rest.
        meshes_processed = []
        for mesh_obj, method, sk_count, captured in plans:
            _write(mesh_obj, method, captured)
            meshes_processed.append({"name": mesh_obj.name,
                                     "method": method,
                                     "shape_key_count": sk_count})

        return {"armature": armature_obj.name, "meshes_processed": meshes_processed}
    finally:
        saved.restore()
