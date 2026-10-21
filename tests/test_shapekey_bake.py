"""Synthetic headless test for avatarprep.core.shapekey_bake, cli/bake_shapekey.py and cli/bake_shapekey_normals.py.

Run: blender --background --factory-startup --python tests/test_shapekey_bake.py
Prints BAKE_TEST OK / BAKE_TEST FAIL: <reason>.
"""
import math
import os
import subprocess
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def _smooth_grid(name="Garment", n=10):
    """A smooth-shaded unit grid in z=0, normals +Z."""
    verts = [(j / n, i / n, 0.0) for i in range(n + 1) for j in range(n + 1)]
    faces = [(i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j + 1, (i + 1) * (n + 1) + j)
             for i in range(n) for j in range(n)]
    md = bpy.data.meshes.new(name + "Data")
    md.from_pydata(verts, [], faces)
    md.shade_smooth()
    md.update()
    ob = bpy.data.objects.new(name, md)
    bpy.context.collection.objects.link(ob)
    return ob

def _dome(i, co):
    """A dome of height 0.1 and radius 0.3 at the grid's centre; exactly flat outside it."""
    r = math.hypot(co.x - 0.5, co.y - 0.5)
    return co + Vector((0, 0, 0.1 * (1 - (r / 0.3) ** 2) ** 2 if r < 0.3 else 0.0))

def _co(seq):
    a = np.empty(len(seq) * 3)
    seq.foreach_get("co", a)
    return a.reshape(-1, 3)

def _deg(u, v):
    from avatarprep.core import shapekey_bake as B
    return np.degrees(B._angle(u, v))

def _tilt(ob, deg=10.0):
    """Tilt the custom normal of every corner on the grid's left half (x < 0.45) by ``deg``
    about Y; returns that corner mask."""
    from avatarprep.core import shapekey_bake as B
    me = ob.data
    lv = np.array([l.vertex_index for l in me.loops])
    x = _co(me.vertices)[:, 0]
    tilt = x[lv] < 0.45
    R = np.array(Matrix.Rotation(math.radians(deg), 3, 'Y'))
    n = B._computed_normals(ob)
    n[tilt] = n[tilt] @ R.T
    me.normals_split_custom_set(n.tolist())
    me.update()
    return tilt

def test_bake_carries_normal_offsets():
    from avatarprep.core import shapekey_bake as B
    _clear()
    ob = _smooth_grid()
    _add_key(ob, "Dome", _dome)
    me = ob.data
    tilt = _tilt(ob)
    before = B._corner_normals(me)
    computed_before = B._computed_normals(ob)
    dev_before = _deg(before, computed_before)
    rep = B.bake_shapekey_to_basis(ob, "Dome", 1.0, protect_group="")
    after = B._corner_normals(me)
    computed_after = B._computed_normals(ob)
    dev_after = _deg(after, computed_after)
    check(me.has_custom_normals and rep["had_custom_normals"], "the baked mesh keeps custom normals")
    want = B._unit(computed_after + (before - computed_before))
    check(_deg(after, want).max() < 0.5, "every corner is computed-after plus its authored offset (max %.3f deg)"
          % _deg(after, want).max())
    check(np.abs(dev_after - dev_before)[tilt].max() < 0.5,
          "a 10 deg authored deviation survives the bake: %.2f..%.2f deg after"
          % (dev_after[tilt].min(), dev_after[tilt].max()))
    turned = _deg(computed_after, computed_before) > 5.0
    check((turned & tilt).any(), "the fixture's dome turns some tilted corners' computed normals")
    check(dev_after[~tilt].max() < 0.5, "a corner with no authored offset takes the recomputed normal")
    check(rep["carried_loops"] == int(tilt.sum()), "carried_loops counts the deviated corners: %d of %d"
          % (rep["carried_loops"], tilt.sum()))

def test_bake_without_custom_normals_adds_none():
    from avatarprep.core import shapekey_bake as B
    _clear()
    ob = _smooth_grid()
    _add_key(ob, "Dome", _dome)
    rep = B.bake_shapekey_to_basis(ob, "Dome", 1.0)
    check(not ob.data.has_custom_normals and rep["carried_loops"] == 0,
          "a mesh without custom normals is left without them")

def test_bake_shapekey_normals():
    from avatarprep.core import shapekey_bake as B
    _clear()
    from avatarprep.core import scene_utils
    ob = _smooth_grid()
    _add_key(ob, "Dome", _dome)
    me = ob.data
    kbs = me.shape_keys.key_blocks
    co0, basis0, dome0 = _co(me.vertices), _co(kbs["Basis"].data), _co(kbs["Dome"].data)
    before = B._corner_normals(me)
    keyed = B._computed_normals(ob, dome0)
    turn = _deg(keyed, before)
    rep = B.bake_shapekey_normals(ob, "Dome")
    after = B._corner_normals(me)
    off, whole = turn < 0.1, turn >= 1.0
    check(off.any() and whole.any(), "the fixture has corners off and on the dome's footprint")
    check(_deg(after, before)[off].max() < 0.05, "corners off the footprint are unchanged (max %.4f deg)"
          % _deg(after, before)[off].max())
    check(_deg(after, keyed)[whole].max() < 0.05, "footprint corners take the keyed normal (max %.4f deg)"
          % _deg(after, keyed)[whole].max())
    check(rep["whole"] == int(whole.sum()), "whole counts the footprint corners: %d" % rep["whole"])
    check(np.array_equal(_co(me.vertices), co0) and np.array_equal(_co(kbs["Basis"].data), basis0)
          and np.array_equal(_co(kbs["Dome"].data), dome0), "no vertex moves, on the mesh or any key")
    line = "bake_shapekey_normals --mesh Garment --key Dome"
    check(rep["recipe"] == line and dict(ob[scene_utils.STAMP_NORMALS]) == {"Dome": line},
          "the recipe is recorded per key: %r" % dict(ob.get(scene_utils.STAMP_NORMALS) or {}))
    entry = [m for m in scene_utils.report_stamps(bpy.context.scene)["unbound"] if m["name"] == "Garment"]
    check(entry and entry[0].get("normals") == {"Dome": line}, "report_stamps reads it as normals: %r" % entry)
    expect_raises(lambda: B.bake_shapekey_normals(ob, "Dome"), "already baked", "a second bake of one key")
    expect_raises(lambda: B.bake_shapekey_normals(ob, "Basis"), "is the basis", "the Basis")
    expect_raises(lambda: B.bake_shapekey_normals(ob, "Dome", fade=(1.0, 0.5)), "two angles", "a reversed fade")

    # a mesh with authored normals keeps them off the footprint
    ob2 = _smooth_grid("Garment2")
    _add_key(ob2, "Dome", _dome)
    _tilt(ob2)
    before = B._corner_normals(ob2.data)
    turn = _deg(B._computed_normals(ob2, _co(ob2.data.shape_keys.key_blocks["Dome"].data)),
                B._computed_normals(ob2))
    B.bake_shapekey_normals(ob2, "Dome")
    after = B._corner_normals(ob2.data)
    check(_deg(after, before)[turn < 0.1].max() < 0.05, "authored normals off the footprint are unchanged")
    check((_deg(after, before)[turn >= 1.0] > 1.0).all(), "authored normals on the footprint turn")
    rel = ob2.shape_key_add(name="OnDome", from_mix=False)
    rel.relative_key = ob2.data.shape_keys.key_blocks["Dome"]
    expect_raises(lambda: B.bake_shapekey_normals(ob2, "OnDome"), "relative to", "a key relative to another key")

    head = _smooth_grid("Body")
    _add_key(head, "Dome", _dome)
    expect_raises(lambda: B.bake_shapekey_normals(head, "Dome"), "head mesh", "the head mesh")

def _blender(tool, args):
    cmd = [bpy.app.binary_path, "--background", "--factory-startup", "--python",
           os.path.join(REPO, "cli", tool + ".py"), "--"] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr

def test_cli(tmp):
    from avatarprep.core import scene_utils
    _clear()
    ob = _smooth_grid()
    _add_key(ob, "Dome", _dome)
    _tilt(ob)
    src = os.path.join(tmp, "dome.blend")
    bpy.ops.wm.save_as_mainfile(filepath=src)

    out = os.path.join(tmp, "baked.blend")
    rc, txt = _blender("bake_shapekey", ["--in", src, "--out", out, "--mesh", "Garment", "--key", "Dome"])
    check(rc == 0 and "carried_loops=" in txt and "carried_loops=0" not in txt,
          "bake_shapekey prints the carried loops (rc=%d)\n%s" % (rc, txt))

    mtime = os.path.getmtime(src)
    rc, txt = _blender("bake_shapekey_normals", ["--in", src, "--whatif", "--mesh", "Garment", "--key", "Dome"])
    check(rc == 0 and "whatif; nothing saved" in txt and os.path.getmtime(src) == mtime,
          "--whatif bakes in memory and saves nothing\n%s" % txt)
    ghost = os.path.join(tmp, "ghost.blend")
    rc, txt = _blender("bake_shapekey_normals", ["--in", src, "--whatif", "--out", ghost, "--mesh", "Garment",
                                                 "--key", "Dome"])
    check(rc == 2 and not os.path.exists(ghost) and "--whatif" in txt, "--whatif with --out is bad args\n%s" % txt)
    rc, txt = _blender("bake_shapekey_normals", ["--in", src, "--whatif", "--mesh", "Garment", "--key", "Dome",
                                                 "--fade-deg", "1"])
    check(rc == 2 and "two angles" in txt, "a one-angle --fade-deg is bad args\n%s" % txt)
    normals = os.path.join(tmp, "normals.blend")
    rc, txt = _blender("bake_shapekey_normals", ["--in", src, "--out", normals, "--mesh", "Garment", "--key", "Dome",
                                                 "--fade-deg", "0.2,2"])
    line = "bake_shapekey_normals --mesh Garment --key Dome --fade-deg 0.2,2"
    check(rc == 0 and ("recipe: " + line) in txt and os.path.exists(normals),
          "a saving run prints the recipe and saves (rc=%d)\n%s" % (rc, txt))
    if rc == 0:
        rc, txt = _blender("report_stamps", ["--in", normals])
        check(rc == 0 and ("mesh Garment normals Dome=%s" % line) in txt, "report_stamps prints it\n%s" % txt)
        rc, txt = _blender("bake_shapekey_normals", ["--in", normals, "--out", ghost, "--mesh", "Garment",
                                                     "--key", "Dome"])
        check(rc == 1 and "already baked" in txt and not os.path.exists(ghost), "a rerun refuses\n%s" % txt)

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
    test_bake_carries_normal_offsets()
    test_bake_without_custom_normals_adds_none()
    test_bake_shapekey_normals()
    with tempfile.TemporaryDirectory() as tmp:
        test_cli(tmp)
    if FAILURES:
        for f in FAILURES: print("BAKE_TEST FAIL:", f)
        sys.exit(1)
    print("BAKE_TEST OK")

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "BAKE_TEST")
