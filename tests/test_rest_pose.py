"""Synthetic, asset-free oracle for the shape-key-safe rest-pose bake.

Run:
  blender --background --factory-startup --python tests/test_rest_pose.py

Prints RESTPOSE_TEST OK and exits 0 on success; RESTPOSE_TEST FAIL: <reason>
and exits 1 otherwise. Builds its own posed armature + three meshes (no / basis-
only / multi shape keys) -- no external avatar -- applies the current pose as the
new rest pose, and asserts the invariants verify.py checks on real avatars.
"""
import os
import sys

import bpy
from mathutils import Vector

SCALE = 1.2
TOL = 0.03
FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def _enable_avatarprep():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _make_arm():
    arm_data = bpy.data.armatures.new("ArmData")
    arm = bpy.data.objects.new("Armature", arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    from avatarprep.core import scene_utils
    ctx = {'active_object': arm, 'object': arm}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    b = arm.data.edit_bones.new("Root")
    b.head = Vector((0, 0, 0)); b.tail = Vector((0, 0, 0.2))
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')
    return arm


def _make_mesh(arm, name, n_keys):
    md = bpy.data.meshes.new(name + "Data")
    md.from_pydata([(-0.05, -0.05, 0.0), (0.05, -0.05, 0.0),
                    (0.05, 0.05, 0.2), (-0.05, 0.05, 0.2)], [], [(0, 1, 2, 3)])
    md.update()
    ob = bpy.data.objects.new(name, md)
    bpy.context.collection.objects.link(ob)
    vg = ob.vertex_groups.new(name="Root")
    vg.add([0, 1, 2, 3], 1.0, 'REPLACE')
    if n_keys >= 1:
        ob.shape_key_add(name="Basis")
    if n_keys >= 2:
        big = ob.shape_key_add(name="Big")
        big.data[2].co = big.data[2].co + Vector((0.0, 0.0, 0.1))
        big.data[3].co = big.data[3].co + Vector((0.0, 0.0, 0.1))
    mod = ob.modifiers.new("Armature", 'ARMATURE'); mod.object = arm
    ob.parent = arm
    return ob


def _max_nonbasis_offset(mesh):
    kb = mesh.data.shape_keys.key_blocks
    basis = kb[0].data
    best = 0.0
    for k in kb[1:]:
        for i, pt in enumerate(k.data):
            best = max(best, (pt.co - basis[i].co).length)
    return best


def _coords(mesh):
    return [v.co.copy() for v in mesh.data.vertices]


def _make_masked_mesh(arm, name):
    """Mesh with a Basis + 'Big' key, where 'Big' is masked by a FRACTIONAL
    (0.5) vertex group -- the case that exposes the w^2 double-masking bug."""
    ob = _make_mesh(arm, name, 2)
    vg = ob.vertex_groups.new(name="Mask")
    vg.add([2, 3], 0.5, 'REPLACE')
    ob.data.shape_keys.key_blocks["Big"].vertex_group = "Mask"
    return ob


def _make_disabled_mod_mesh(arm, name):
    """No-shape-key mesh whose ARMATURE modifier is disabled in the viewport --
    a naive capture reads it undeformed and bakes rest geometry back."""
    ob = _make_mesh(arm, name, 0)
    for mod in ob.modifiers:
        if mod.type == 'ARMATURE':
            mod.show_viewport = False
    return ob


def _eval_key(mesh, key_name):
    """Evaluated (deformed, masked) vertex coords with only ``key_name`` shown.
    This is the runtime-visible result of that shape key."""
    kbs = mesh.data.shape_keys.key_blocks
    prev_show = mesh.show_only_shape_key
    prev_idx = mesh.active_shape_key_index
    mesh.show_only_shape_key = True
    mesh.active_shape_key_index = kbs.find(key_name)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cos = [v.co.copy() for v in mesh.evaluated_get(depsgraph).data.vertices]
    mesh.show_only_shape_key = prev_show
    mesh.active_shape_key_index = prev_idx
    return cos


def main():
    _enable_avatarprep()
    _clear_scene()
    from avatarprep.core import rest_pose

    arm = _make_arm()
    mesh_none = _make_mesh(arm, "BodyNone", 0)
    mesh_basis = _make_mesh(arm, "BodyBasis", 1)
    mesh_multi = _make_mesh(arm, "BodyMulti", 2)
    mesh_masked = _make_masked_mesh(arm, "BodyMasked")     # Finding 1 (vertex-group mask)
    mesh_disabled = _make_disabled_mod_mesh(arm, "BodyDisabledMod")  # Finding 2 (modifier off)

    before_len = arm.data.bones["Root"].length
    before_offset = _max_nonbasis_offset(mesh_multi)
    check(before_offset > 0.0, "fixture sanity: multi-key mesh must start with nonzero offset")
    before_none = _coords(mesh_none)
    before_basis_co = _coords(mesh_basis)
    before_disabled = _coords(mesh_disabled)

    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    arm.pose.bones["Root"].scale = (SCALE, SCALE, SCALE)
    bpy.context.view_layer.update()

    # Finding 1: the masked key's runtime-visible deformed result must be
    # IDENTICAL before and after the bake (double-masking would scale it by w^2).
    masked_before = _eval_key(mesh_masked, "Big")

    all_meshes = [mesh_none, mesh_basis, mesh_multi, mesh_masked, mesh_disabled]
    exc = None
    try:
        rest_pose.apply_pose(arm, all_meshes)
    except Exception as e:  # noqa: BLE001
        exc = e
        import traceback
        traceback.print_exc()
    check(exc is None, "apply_pose raised: %s" % exc)

    after = arm.data.bones.get("Root")
    ratio = (after.length / before_len) if after else None
    check(ratio is not None and abs(ratio - SCALE) <= TOL,
          "rest length ratio %r not within %.2f of %.2f" % (ratio, TOL, SCALE))

    # multi-key: shape keys preserved
    kb = mesh_multi.data.shape_keys.key_blocks
    check(len(kb) == 2, "multi mesh shape key count changed -> %d" % len(kb))
    check(kb[0].name == "Basis", "multi mesh basis renamed -> %r" % kb[0].name)
    check(_max_nonbasis_offset(mesh_multi) > 0.0, "multi mesh non-basis key lost deformation")

    # basis-only: single key survives, geometry baked (changed)
    kbb = mesh_basis.data.shape_keys.key_blocks
    check(len(kbb) == 1 and kbb[0].name == "Basis", "basis-only mesh basis key not preserved")
    changed_basis = any((a - b).length > 1e-5 for a, b in zip(_coords(mesh_basis), before_basis_co))
    check(changed_basis, "basis-only mesh geometry was not baked by the pose")

    # no-shape-key: geometry baked (changed)
    changed_none = any((a - b).length > 1e-5 for a, b in zip(_coords(mesh_none), before_none))
    check(changed_none, "no-shape-key mesh geometry was not baked by the pose")

    # Finding 1: masked key's visible result unchanged across the bake (no w^2).
    masked_after = _eval_key(mesh_masked, "Big")
    max_drift = max((a - b).length for a, b in zip(masked_after, masked_before))
    check(max_drift < 1e-4,
          "masked shape key drifted by %.6f across the bake (double-masking?)" % max_drift)

    # Finding 2: a mesh whose armature modifier was disabled is still baked.
    changed_disabled = any((a - b).length > 1e-5
                           for a, b in zip(_coords(mesh_disabled), before_disabled))
    check(changed_disabled,
          "mesh with a disabled armature modifier was not baked (captured undeformed)")

    # Finding 3: caller's active object and the armature's POSE mode are restored.
    check(bpy.context.view_layer.objects.active is arm, "active object not restored after bake")
    check(arm.mode == 'POSE', "armature mode not restored to POSE after bake (got %r)" % arm.mode)

    if FAILURES:
        print("RESTPOSE_TEST FAIL:", "; ".join(FAILURES))
        sys.exit(1)
    print("RESTPOSE_TEST OK")
    sys.exit(0)


main()
