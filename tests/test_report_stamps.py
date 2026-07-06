"""Synthetic headless test for avatarprep.core.scene_utils.report_stamps.

Run: blender --background --factory-startup --python tests/test_report_stamps.py
Prints REPORT_TEST OK / REPORT_TEST FAIL: <reason>.

Asserts the armature-scoped schema: each baked mesh's per-mesh entry (unchanged in
shape from the old flat list) is partitioned under its SINGLE owning armature's
``meshes[]``; a mesh owned by zero or by >=2 armatures lands in top-level
``unbound[]`` — a true partition (every baked mesh appears exactly once).
"""
import os
import sys

import bpy
from mathutils import Vector

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def _enable():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def _clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def _make_arm(name):
    ad = bpy.data.armatures.new(name + "Data")
    ao = bpy.data.objects.new(name, ad)
    bpy.context.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao
    ao.select_set(True)
    from avatarprep.core import scene_utils
    ctx = {'active_object': ao, 'object': ao}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    b = ao.data.edit_bones.new("Hips")
    b.head = Vector((0, 0, 1.0)); b.tail = Vector((0, 0, 1.1))
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')
    return ao


def _make_mesh(name):
    md = bpy.data.meshes.new(name + "Data")
    md.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [], [(0, 1, 2)])
    md.update()
    mo = bpy.data.objects.new(name, md)
    bpy.context.collection.objects.link(mo)
    return mo


def _parent_bind(mesh, arm):
    """Bind via parenting (CATS' mode-0 path)."""
    mesh.parent = arm


def _mod_bind(mesh, arm):
    """Bind via an ARMATURE modifier (the modifier-target path of the union)."""
    mod = mesh.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = arm


def _by_name(entries, name):
    return next((e for e in entries if e["name"] == name), None)


def _all_mesh_names(rep):
    names = []
    for a in rep["armatures"]:
        names += [m["name"] for m in a["meshes"]]
    names += [m["name"] for m in rep["unbound"]]
    return names


def test_partition():
    """Cases 1,2,3,4,6,8 — separation, entry shape, unbound, ambiguity, corrupt, modifier bind."""
    from avatarprep.core import scene_utils as S
    _clear()

    dress = _make_arm("Armature.Dress")
    base = _make_arm("Base")

    m_dress = _make_mesh("DressBody")          # parented to Dress (case 1, 2)
    _parent_bind(m_dress, dress)
    m_dress[S.STAMP_BAKED] = {"Chest": 0.8}

    m_base = _make_mesh("BaseBody")            # modifier-bound to Base (case 8)
    _mod_bind(m_base, base)
    m_base[S.STAMP_BAKED] = {"Chest": 0.5}

    m_free = _make_mesh("Floating")            # no armature (case 3)
    m_free[S.STAMP_BAKED] = {"Hips": 0.3}

    m_ambi = _make_mesh("Ambi")                # parent Dress + modifier Base (case 4)
    _parent_bind(m_ambi, dress)
    _mod_bind(m_ambi, base)
    m_ambi[S.STAMP_BAKED] = {"Waist": 0.2}

    m_cbound = _make_mesh("CorruptBound")      # non-map, bound to Dress (case 6)
    _parent_bind(m_cbound, dress)
    m_cbound[S.STAMP_BAKED] = "not-a-map"

    m_cfree = _make_mesh("CorruptFree")        # non-map, unbound (case 6)
    m_cfree[S.STAMP_BAKED] = "bad"

    _make_mesh("Plain")                        # no baked map → never listed

    rep = S.report_stamps(bpy.context.scene)   # must not raise

    d = _by_name(rep["armatures"], "Armature.Dress")
    b = _by_name(rep["armatures"], "Base")

    # (1) Two-armature separation — each rig holds only its own bound baked mesh.
    dm = _by_name(d["meshes"], "DressBody")
    check(dm is not None and dm.get("baked") == {"Chest": 0.8},
          "Dress should hold its own DressBody={Chest:0.8}, got %r" % dm)
    check(_by_name(d["meshes"], "BaseBody") is None,
          "Dress.meshes must NOT contain Base's mesh")
    bm = _by_name(b["meshes"], "BaseBody")
    check(bm is not None and bm.get("baked") == {"Chest": 0.5},
          "Base should hold its own BaseBody={Chest:0.5} (modifier-bound), got %r" % bm)
    check(_by_name(b["meshes"], "DressBody") is None,
          "Base.meshes must NOT contain Dress's mesh")

    # (2) Per-mesh entry shape unchanged — exactly {name, baked}.
    check(dm is not None and set(dm.keys()) == {"name", "baked"},
          "clean baked entry must be exactly {name, baked}, got keys %r"
          % (None if dm is None else sorted(dm.keys())))

    # (8) Modifier binding — BaseBody bound only by modifier appears under Base (asserted above).

    # (3) Unbound bucket — Floating, bound to no armature.
    fm = _by_name(rep["unbound"], "Floating")
    check(fm is not None and fm.get("baked") == {"Hips": 0.3},
          "Floating (no armature) should be in unbound with its map, got %r" % fm)
    check(_by_name(d["meshes"], "Floating") is None and _by_name(b["meshes"], "Floating") is None,
          "Floating must not appear under any armature")

    # (4) Ambiguous ownership (>=2 owners) → unbound, under neither rig, not duplicated.
    am = _by_name(rep["unbound"], "Ambi")
    check(am is not None and am.get("baked") == {"Waist": 0.2},
          "double-bound Ambi should route to unbound, got %r" % am)
    check(_by_name(d["meshes"], "Ambi") is None and _by_name(b["meshes"], "Ambi") is None,
          "double-bound Ambi must appear under NEITHER rig (true partition)")

    # (6) Per-mesh corrupt — flagged not raised, in sole-owner meshes[] and in unbound.
    cb = _by_name(d["meshes"], "CorruptBound")
    check(cb is not None and cb.get("baked") is None and cb.get("corrupt") is not None,
          "bound corrupt mesh should be flagged in its owner's meshes[], got %r" % cb)
    cf = _by_name(rep["unbound"], "CorruptFree")
    check(cf is not None and cf.get("baked") is None and cf.get("corrupt") is not None,
          "unbound corrupt mesh should be flagged in unbound, got %r" % cf)

    # Plain (no baked map) never listed anywhere.
    check("Plain" not in _all_mesh_names(rep),
          "a mesh with no baked map must be omitted entirely")

    # True partition — every baked mesh appears exactly once across the whole report.
    names = _all_mesh_names(rep)
    check(sorted(names) == sorted(set(names)),
          "every baked mesh must appear exactly once, got %r" % names)
    check(set(names) == {"DressBody", "BaseBody", "Floating", "Ambi",
                         "CorruptBound", "CorruptFree"},
          "unexpected baked-mesh set %r" % sorted(names))


def test_present_empty_buckets():
    """Case 5 — meshes:[] and unbound:[] are always present, never absent."""
    from avatarprep.core import scene_utils as S
    _clear()

    _make_arm("Empty")                         # no bound baked mesh → meshes: []
    owner = _make_arm("Owner")
    m = _make_mesh("Owned")                     # single-owned → unbound stays []
    _parent_bind(m, owner)
    m[S.STAMP_BAKED] = {"Chest": 0.4}

    rep = S.report_stamps(bpy.context.scene)
    e = _by_name(rep["armatures"], "Empty")
    check(e is not None and e.get("meshes") == [],
          "a rig with no solely-bound baked mesh must have meshes:[] present, got %r" % e)
    check("unbound" in rep and rep["unbound"] == [],
          "a fully-claimed report must still carry unbound:[] present, got %r" % rep.get("unbound"))


def test_armature_fields():
    """Case 7 — base/state/state_kind unchanged by the grouping."""
    from avatarprep.core import scene_utils as S
    _clear()

    _make_arm("Bare")
    a_set = _make_arm("Set")
    S.write_stamp(a_set, S.STAMP_BASE, "shinano")
    S.write_stamp(a_set, S.STAMP_STATE, "shinano-tall")
    a_int = _make_arm("Interrupted")
    S.write_stamp(a_int, S.STAMP_STATE, S.STATE_APPLYING)

    rep = S.report_stamps(bpy.context.scene)
    bare = _by_name(rep["armatures"], "Bare")
    check(bare is not None and bare["base"] is None and bare["state_kind"] == "absent"
          and bare.get("meshes") == [],
          "unstamped rig should read base=None/absent with meshes:[], got %r" % bare)
    seta = _by_name(rep["armatures"], "Set")
    check(seta is not None and seta["base"] == "shinano"
          and seta["state"] == "shinano-tall" and seta["state_kind"] == "value",
          "set rig should surface base+state, got %r" % seta)
    inta = _by_name(rep["armatures"], "Interrupted")
    check(inta is not None and inta["state_kind"] == "interrupted",
          "sentinel state should read state_kind=interrupted, got %r" % inta)


def main():
    _enable()
    test_partition()
    test_present_empty_buckets()
    test_armature_fields()
    if FAILURES:
        for f in FAILURES:
            print("REPORT_TEST FAIL:", f)
        sys.exit(1)
    print("REPORT_TEST OK")


if __name__ == "__main__":
    main()
