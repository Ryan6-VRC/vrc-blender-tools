"""Synthetic headless test for avatarprep.core.shapekey_bake.

Run: blender --background --factory-startup --python tests/test_shapekey_bake.py
Prints BAKE_TEST OK / BAKE_TEST FAIL: <reason>.
"""
import os
import sys

import bpy
from mathutils import Vector

FAILURES = []

def check(cond, msg):
    if not cond:
        FAILURES.append(msg)

def expect_raises(fn, substr, label):
    try:
        fn()
    except Exception as e:
        if substr.lower() not in str(e).lower():
            FAILURES.append("%s: raised but %r lacked %r" % (label, str(e), substr))
        return
    FAILURES.append("%s: expected exception mentioning %r" % (label, substr))

def _enable():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

def _clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)

def _grid_mesh(name="Body_Base"):
    verts = [(0,0,0),(0,1,0),(0.5,0,0),(0.5,1,0),(1,0,0),(1,1,0)]
    faces = [(0,1,3,2),(2,3,5,4)]
    md = bpy.data.meshes.new(name + "Data")
    md.from_pydata(verts, [], faces); md.update()
    ob = bpy.data.objects.new(name, md)
    bpy.context.collection.objects.link(ob)
    return ob

def _add_key(ob, name, fn):
    if ob.data.shape_keys is None:
        ob.shape_key_add(name="Basis")
    kb = ob.shape_key_add(name=name)
    for i, p in enumerate(kb.data):
        p.co = fn(i, Vector(ob.data.vertices[i].co))
    return kb

def test_bake_moves_basis_and_retains_morph():
    from avatarprep.core import shapekey_bake as B
    ob = _grid_mesh()
    _add_key(ob, "Chest", lambda i, co: co + Vector((0,0,1)) if co.x > 0.5 else co)
    _add_key(ob, "Hip", lambda i, co: co + Vector((0,0,1)) if co.x < 0.5 else co)
    basis = ob.data.shape_keys.key_blocks["Basis"]
    hip = ob.data.shape_keys.key_blocks["Hip"]
    # The real guarantee: editing Basis drags relative keys uniformly, so the
    # sibling's relative delta (key.co - basis.co), and thus its effect, is
    # preserved at EVERY vertex -- including those the bake moves.
    rel_before = [Vector(hip.data[i].co) - Vector(basis.data[i].co)
                  for i in range(len(ob.data.vertices))]
    rep = B.bake_shapekey_to_basis(ob, "Chest", 1.0)
    check(rep["had_custom_normals"] is False, "no custom normals -> flag False")
    check(abs(basis.data[4].co.z - 1.0) < 1e-5, "right-col basis should rise to z=1")
    check(abs(basis.data[0].co.z - 0.0) < 1e-5, "left-col basis should stay z=0")
    check("Chest" in ob.data.shape_keys.key_blocks, "Chest morph must be retained")
    rel_after = [Vector(hip.data[i].co) - Vector(basis.data[i].co)
                 for i in range(len(ob.data.vertices))]
    drift = max((rel_after[i] - rel_before[i]).length for i in range(len(rel_before)))
    check(drift < 1e-5, "sibling relative-delta must be preserved at all verts (drift=%g)" % drift)

def test_protect_group_case_insensitive():
    from avatarprep.core import shapekey_bake as B
    ob = _grid_mesh()
    _add_key(ob, "Chest", lambda i, co: co + Vector((0,0,1)) if co.x > 0.5 else co)
    me = ob.data
    vg = ob.vertex_groups.new(name="Neck")  # capitalized; default arg is "neck"
    vg.add([0,1], 1.0, 'REPLACE')
    me.normals_split_custom_set([(0.0,0.0,1.0)] * len(me.loops))
    rep = B.bake_shapekey_to_basis(ob, "Chest", 1.0)  # default protect_group="neck"
    check(rep["protected_loops"] > 0, "case-insensitive protect_group should match 'Neck'")

def test_refuses_head_mesh():
    from avatarprep.core import shapekey_bake as B
    ob = _grid_mesh(name="Body")
    _add_key(ob, "Chest", lambda i, co: co)
    expect_raises(lambda: B.bake_shapekey_to_basis(ob, "Chest", 1.0), "head", "head mesh refusal")

def test_missing_key():
    from avatarprep.core import shapekey_bake as B
    ob = _grid_mesh()
    ob.shape_key_add(name="Basis")
    expect_raises(lambda: B.bake_shapekey_to_basis(ob, "Ghost", 1.0), "not found", "missing key")

def test_protected_normals_retained():
    from avatarprep.core import shapekey_bake as B
    ob = _grid_mesh()
    _add_key(ob, "Chest", lambda i, co: co + Vector((0,0,1)) if co.x > 0.5 else co)
    me = ob.data
    vg = ob.vertex_groups.new(name="neck")
    vg.add([0,1], 1.0, 'REPLACE')
    me.normals_split_custom_set([(0.0,0.0,1.0)] * len(me.loops))
    rep = B.bake_shapekey_to_basis(ob, "Chest", 1.0, protect_group="neck")
    check(rep["had_custom_normals"], "mesh should report custom normals")
    check(rep["protected_loops"] > 0, "neck loops should be protected")
    now = [tuple(cn.vector) for cn in me.corner_normals]
    prot = [l.index for l in me.loops if l.vertex_index in (0, 1)]
    check(all(abs(now[li][2] - 1.0) < 1e-4 for li in prot),
          "protected loops must keep authored +Z normal")
    nonprot = [l.index for l in me.loops if l.vertex_index in (4, 5)]
    check(any(abs(now[li][2] - 1.0) > 1e-3 for li in nonprot),
          "non-protected loop recomputed away from +Z")

def test_bake_zeros_slider():
    # The morph's live slider, if left nonzero (proportions.apply_shapekeys sets it),
    # would add the delta again on top of the now-baked Basis -> double-application.
    # The bake must zero the baked key's slider.
    from avatarprep.core import shapekey_bake as B
    ob = _grid_mesh()
    k = _add_key(ob, "Chest", lambda i, co: co + Vector((0,0,1)) if co.x > 0.5 else co)
    k.value = 0.5  # left nonzero, as apply_shapekeys would
    B.bake_shapekey_to_basis(ob, "Chest", 0.6)
    check(abs(ob.data.shape_keys.key_blocks["Chest"].value) < 1e-9,
          "baked key's live slider must be zeroed (else double-application)")

def test_refuses_head_mesh_suffix():
    # Blender auto-suffix / case must not slip past the never-bake-the-head rail.
    from avatarprep.core import shapekey_bake as B
    ob = _grid_mesh(name="Body.001")
    _add_key(ob, "Chest", lambda i, co: co)
    expect_raises(lambda: B.bake_shapekey_to_basis(ob, "Chest", 1.0), "head", "head suffix refusal")

def test_baked_map_accumulates():
    from avatarprep.core import shapekey_bake as B
    from avatarprep.core import scene_utils
    ob = _grid_mesh()
    _add_key(ob, "Chest", lambda i, co: co + Vector((0,0,1)) if co.x > 0.5 else co)
    rep1 = B.bake_shapekey_to_basis(ob, "Chest", 0.4)
    check(abs(rep1.get("baked_cumulative", -99) - 0.4) < 1e-6,
          "first bake baked_cumulative should be 0.4, got %r" % rep1.get("baked_cumulative"))
    m = dict(ob.get(scene_utils.STAMP_BAKED))
    check(abs(m.get("Chest", -99) - 0.4) < 1e-6, "baked map should record Chest=0.4, got %r" % m)
    check("Chest" in ob.data.shape_keys.key_blocks, "morph block must survive first bake")
    rep2 = B.bake_shapekey_to_basis(ob, "Chest", 0.25)
    check(abs(rep2.get("baked_cumulative", -99) - 0.65) < 1e-6,
          "repeat bake should accumulate to 0.65, got %r" % rep2.get("baked_cumulative"))
    m2 = dict(ob.get(scene_utils.STAMP_BAKED))
    check(abs(m2.get("Chest", -99) - 0.65) < 1e-6, "baked map should accumulate Chest=0.65, got %r" % m2)
    check("Chest" in ob.data.shape_keys.key_blocks, "morph block must survive repeat bake")


def test_baked_map_snaps_zero():
    from avatarprep.core import shapekey_bake as B
    ob = _grid_mesh()
    _add_key(ob, "Chest", lambda i, co: co + Vector((0,0,1)) if co.x > 0.5 else co)
    B.bake_shapekey_to_basis(ob, "Chest", 0.5)
    rep = B.bake_shapekey_to_basis(ob, "Chest", -0.5)  # cumulative → 0.0
    cum = rep.get("baked_cumulative")
    check(cum == 0.0, "±-reversal should snap cumulative to exactly 0.0, got %r" % cum)


def test_baked_map_nonmap_raises():
    from avatarprep.core import shapekey_bake as B
    from avatarprep.core import scene_utils
    ob = _grid_mesh()
    # A morph that would visibly move Basis IF the fold ran (right column rises z+1).
    _add_key(ob, "Chest", lambda i, co: co + Vector((0,0,1)) if co.x > 0.5 else co)
    basis = ob.data.shape_keys.key_blocks["Basis"]
    before = [Vector(basis.data[i].co) for i in range(len(ob.data.vertices))]
    ob[scene_utils.STAMP_BAKED] = "not-a-map"  # corrupt: a scalar where a map is expected
    expect_raises(lambda: B.bake_shapekey_to_basis(ob, "Chest", 1.0),
                  "not a map", "non-map baked stamp")
    # Fail-loud ordering: the guard fires BEFORE the fold, so Basis is untouched.
    after = [Vector(basis.data[i].co) for i in range(len(ob.data.vertices))]
    drift = max((after[i] - before[i]).length for i in range(len(before)))
    check(drift < 1e-9, "non-map guard must fire before folding Basis (drift=%g)" % drift)


def main():
    _clear(); _enable()
    test_bake_moves_basis_and_retains_morph()
    test_protect_group_case_insensitive()
    test_refuses_head_mesh()
    test_refuses_head_mesh_suffix()
    test_missing_key()
    test_protected_normals_retained()
    test_bake_zeros_slider()
    test_baked_map_accumulates()
    test_baked_map_snaps_zero()
    test_baked_map_nonmap_raises()
    if FAILURES:
        for f in FAILURES: print("BAKE_TEST FAIL:", f)
        sys.exit(1)
    print("BAKE_TEST OK")

if __name__ == "__main__":
    main()
