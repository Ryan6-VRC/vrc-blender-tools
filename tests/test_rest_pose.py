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


def _add_repo_root_to_path():
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



def _posed_rig_with_mesh():
    """One bone scaled 2x in Y, one bound quad sitting at the bone tip. Deformation is
    a clean z 1.0 -> 2.0, so an undeformed bake is unmistakable."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ad = bpy.data.armatures.new("A")
    arm = bpy.data.objects.new("Armature", ad)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    b = ad.edit_bones.new("Root")
    b.head = Vector((0, 0, 0))
    b.tail = Vector((0, 0, 1))
    bpy.ops.object.mode_set(mode='OBJECT')
    me = bpy.data.meshes.new("BodyData")
    me.from_pydata([(0, 0, 1.0), (0.1, 0, 1.0), (0, 0.1, 1.0)], [], [(0, 1, 2)])
    me.update()
    ob = bpy.data.objects.new("Body", me)
    bpy.context.collection.objects.link(ob)
    vg = ob.vertex_groups.new(name="Root")
    vg.add([0, 1, 2], 1.0, 'REPLACE')
    mod = ob.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    ob.parent = arm
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    arm.pose.bones["Root"].scale = (1.0, 2.0, 1.0)
    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm, ob


def _to_subcollection(ob):
    col = bpy.data.collections.new("Sub")
    bpy.context.scene.collection.children.link(col)
    bpy.context.scene.collection.objects.unlink(ob)
    col.objects.link(ob)
    return col, bpy.context.view_layer.layer_collection.children[col.name]


def test_refuses_unevaluated_meshes():
    """A mesh the depsgraph will not evaluate bakes UNDEFORMED and reports itself
    processed. Refuse instead -- and refuse on exactly the states that break.

    The three that bake correctly are asserted too, because the tempting predicates get
    them wrong: visible_get() reads False for hide_set and for a LAYER collection's
    hide_viewport, both of which bake fine, so it would refuse working files.
    """
    from avatarprep.core import rest_pose

    def hide_viewport(a, o):
        o.hide_viewport = True

    def eye(a, o):
        o.hide_set(True)

    def coll_exclude(a, o):
        _to_subcollection(o)[1].exclude = True

    def coll_hide(a, o):
        _to_subcollection(o)[0].hide_viewport = True

    def layer_coll_hide(a, o):
        _to_subcollection(o)[1].hide_viewport = True

    cases = [
        ("visible", lambda a, o: None, False),
        ("obj.hide_viewport", hide_viewport, True),
        ("obj.hide_set", eye, False),
        ("collection excluded", coll_exclude, True),
        ("collection.hide_viewport", coll_hide, True),
        ("layer_coll.hide_viewport", layer_coll_hide, False),
    ]
    for label, setup, should_refuse in cases:
        arm, ob = _posed_rig_with_mesh()
        setup(arm, ob)
        bpy.context.view_layer.update()
        before = [round(v.co.z, 4) for v in ob.data.vertices]
        try:
            rest_pose.apply_pose(arm, [ob])
            refused = False
        except rest_pose.RestPoseRefused as e:
            refused = True
            check(ob.name in " | ".join(e.offenders),
                  "%s: the refusal should name the mesh, got %s" % (label, e.offenders))
            check([round(v.co.z, 4) for v in ob.data.vertices] == before,
                  "%s: a refusal must not have mutated the mesh" % label)
        check(refused == should_refuse,
              "%s: refused=%s, expected %s" % (label, refused, should_refuse))
        if not refused:
            baked = round(ob.data.vertices[0].co.z, 3)
            check(abs(baked - 2.0) < 1e-2,
                  "%s: baked z should be 2.0 (deformed), got %s -- if this is 1.0 the "
                  "predicate let an unevaluated mesh through" % (label, baked))


def test_edge_refuses_before_mutating():
    """The edge path must decline at validate, not at the bake: apply_proportion_edge
    calls apply_pose partway through, where a refusal would land on half-transformed
    geometry with the state stamp at its crash sentinel."""
    from avatarprep.core import proportions, scene_utils
    arm, ob = _posed_rig_with_mesh()
    bpy.ops.object.mode_set(mode='POSE')
    arm.pose.bones["Root"].scale = (1.0, 1.0, 1.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    arm["avatarprep_base"] = "a"
    arm["avatarprep_state"] = "s0"
    ob.hide_viewport = True
    bpy.context.view_layer.update()
    edge = {"source": "s0", "target": "s1", "source_base": "a",
            "scales": [{"bones": ["Root"], "value": [1.0, 1.3, 1.0]}]}
    report = proportions.validate_proportion_edge(arm, [ob], proportions.load_edge(edge))
    check(any("not evaluated" in o for o in report["offenders"]),
          "validate should offend on an unevaluated mesh, got %s" % report["offenders"])
    try:
        proportions.apply_proportion_edge(arm, [ob], edge)
        FAILURES.append("apply_proportion_edge should have aborted on the offender")
    except proportions.EdgeError:
        pass
    check(scene_utils.read_stamp(arm, scene_utils.STAMP_STATE) == "s0",
          "the state stamp must be untouched -- not left at the crash sentinel")


def main():
    _add_repo_root_to_path()
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

    test_refuses_unevaluated_meshes()
    test_edge_refuses_before_mutating()
    if FAILURES:
        print("RESTPOSE_TEST FAIL:", "; ".join(FAILURES))
        sys.exit(1)
    print("RESTPOSE_TEST OK")
    sys.exit(0)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "RESTPOSE_TEST")
