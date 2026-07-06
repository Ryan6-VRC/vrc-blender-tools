"""Synthetic headless test for ``prune_zero_weight_bones``.

Run::

    blender --background --factory-startup --python tests/test_prune_bones.py

Prints ``PRUNE_TEST OK`` and exits 0 on success; prints
``PRUNE_TEST FAIL: <reason>`` and exits 1 on any failed assertion or exception.
"""

import os
import sys

import bpy
from mathutils import Vector


def _enable_avatarprep():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _clear_scene():
    """Remove all default objects left by --factory-startup."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _build_armature():
    """Create the test armature, covering every keep/delete case.

    Bone layout::

        Spine  (weighted) ─── Chest      (weighted)
               └──────────── Upper      (weighted) ─── Upper_end  (zero-weight leaf)
        Skirt  (zero)     ─── Skirt_end  (zero-weight leaf)
        Hook   (zero)     ← Empty object parented here via BONE parent type
        Scalp  (zero)     ─── Hair1 (zero) ─── Hair2 (zero) ─── Hair3 (zero)
                              (fully zero-weight chain — deleted whole)
    """
    arm_data = bpy.data.armatures.new("TestArmatureData")
    arm_obj = bpy.data.objects.new("TestArmature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    # Enter Edit Mode to add bones (use op_override for headless safety).
    from avatarprep.core import scene_utils
    ctx = {'active_object': arm_obj, 'object': arm_obj}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    ebs = arm_obj.data.edit_bones

    spine = ebs.new("Spine")
    spine.head = Vector((0.0, 0.0, 0.0))
    spine.tail = Vector((0.0, 0.0, 0.3))

    chest = ebs.new("Chest")
    chest.head = Vector((0.0, 0.0, 0.3))
    chest.tail = Vector((0.0, 0.0, 0.6))
    chest.parent = spine

    upper = ebs.new("Upper")
    upper.head = Vector((0.3, 0.0, 0.3))
    upper.tail = Vector((0.3, 0.0, 0.6))
    upper.parent = spine

    upper_end = ebs.new("Upper_end")
    upper_end.head = Vector((0.3, 0.0, 0.6))
    upper_end.tail = Vector((0.3, 0.0, 0.7))
    upper_end.parent = upper

    skirt = ebs.new("Skirt")
    skirt.head = Vector((0.0, 0.0, -0.1))
    skirt.tail = Vector((0.0, 0.0, -0.4))

    skirt_end = ebs.new("Skirt_end")
    skirt_end.head = Vector((0.0, 0.0, -0.4))
    skirt_end.tail = Vector((0.0, 0.0, -0.7))
    skirt_end.parent = skirt

    hook = ebs.new("Hook")
    hook.head = Vector((0.0, 0.5, 0.3))
    hook.tail = Vector((0.0, 0.5, 0.5))

    # Fully zero-weight multi-bone chain off an unweighted root. None of these
    # carry weight, so the whole chain must be deleted (only depth-1 zero-weight
    # leaves of a weighted bone are preserved).
    prev = None
    for i, name in enumerate(["Scalp", "Hair1", "Hair2", "Hair3"]):
        b = ebs.new(name)
        b.head = Vector((-0.3, 0.0, 0.6 + 0.1 * i))
        b.tail = Vector((-0.3, 0.0, 0.7 + 0.1 * i))
        if prev is not None:
            b.parent = prev
        prev = b

    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')
    return arm_obj


def _build_mesh(arm_obj):
    """Create a quad mesh skinned to Spine, Chest, and Upper only."""
    verts = [
        (-0.05, -0.05, 0.3),
        (0.05, -0.05, 0.3),
        (0.05,  0.05, 0.3),
        (-0.05,  0.05, 0.3),
    ]
    mesh_data = bpy.data.meshes.new("BodyMeshData")
    mesh_data.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh_data.update()
    mesh_obj = bpy.data.objects.new("BodyMesh", mesh_data)
    bpy.context.collection.objects.link(mesh_obj)

    vertex_indices = list(range(len(verts)))
    for bone_name, weight in [("Spine", 0.3), ("Chest", 0.5), ("Upper", 0.2)]:
        vg = mesh_obj.vertex_groups.new(name=bone_name)
        vg.add(vertex_indices, weight, 'REPLACE')

    mod = mesh_obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm_obj
    return mesh_obj


def _attach_empty(arm_obj):
    """Parent an Empty to Hook via BONE parent type (attachment point)."""
    empty = bpy.data.objects.new("HookAttachment", None)
    bpy.context.collection.objects.link(empty)
    empty.location = Vector((0.0, 0.5, 0.4))
    empty.parent = arm_obj
    empty.parent_type = 'BONE'
    empty.parent_bone = 'Hook'
    return empty


def main():
    _clear_scene()
    _enable_avatarprep()

    from avatarprep.core.prune_bones import prune_zero_weight_bones

    arm_obj = _build_armature()
    _build_mesh(arm_obj)
    _attach_empty(arm_obj)

    try:
        result = prune_zero_weight_bones(arm_obj)
    except Exception as e:
        print("PRUNE_TEST FAIL: exception:", e)
        sys.exit(1)
    bones_remaining = {b.name for b in arm_obj.data.bones}

    failures = []

    def expect_present(name):
        if name not in bones_remaining:
            failures.append("expected %r to be KEPT but it was deleted" % name)

    def expect_absent(name):
        if name in bones_remaining:
            failures.append("expected %r to be DELETED but it was kept" % name)

    expect_present("Spine")       # (a) weighted
    expect_present("Chest")       # (a) weighted
    expect_present("Upper")       # (a) weighted
    expect_present("Upper_end")   # (b) zero-weight leaf, parent weighted
    expect_present("Hook")        # (c) attachment parent
    expect_absent("Skirt")        # zero-weight, no weighted descendants, no attachment
    expect_absent("Skirt_end")    # zero-weight leaf, parent not weighted
    # Fully zero-weight chain off an unweighted root: deleted whole.
    expect_absent("Scalp")
    expect_absent("Hair1")
    expect_absent("Hair2")
    expect_absent("Hair3")

    # Guard the return dict so a miscounted result is caught too.
    if result.get("kept") != 5:
        failures.append("expected result['kept'] == 5, got %r" % result.get("kept"))
    if result.get("deleted") != 6:
        failures.append("expected result['deleted'] == 6, got %r" % result.get("deleted"))

    expected_deleted = {"Skirt", "Skirt_end", "Scalp", "Hair1", "Hair2", "Hair3"}
    deleted_bones = result.get("deleted_bones")
    if not isinstance(deleted_bones, list):
        failures.append("expected result['deleted_bones'] to be a list, got %r" % type(deleted_bones))
    else:
        if set(deleted_bones) != expected_deleted:
            failures.append("expected deleted_bones == %r, got %r"
                            % (sorted(expected_deleted), sorted(deleted_bones)))
        if result.get("deleted") != len(deleted_bones):
            failures.append("expected deleted == len(deleted_bones) (%d), got %r"
                            % (len(deleted_bones), result.get("deleted")))

    if failures:
        for f in failures:
            print("PRUNE_TEST FAIL:", f)
        sys.exit(1)
    else:
        print("PRUNE_TEST OK")
        print("  result:", result)
        print("  bones remaining:", sorted(bones_remaining))


if __name__ == "__main__":
    main()
