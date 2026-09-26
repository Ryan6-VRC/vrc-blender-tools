"""Synthetic headless test for avatarprep.core.fold_bones and cli/fold_bones.py.

Run: blender --background --factory-startup --python tests/test_fold_bones.py
Prints FOLD_TEST OK / FOLD_TEST FAIL: <reason>.

Fixture (rebuilt fresh per test — ``_build_scene`` — so one test's mutation never
leaks into another): an armature ``Hips -> Bun``, ``Hips -> Strand_1 -> Strand_2``,
``Hips -> Extra``, and the doomed chain ``Bun -> Ribbon -> Ribbon_1 -> Ribbon_2``.
``RibbonMesh`` is a hair-ribbon island weighted only to the doomed chain (plus one
untouched vertex on ``Hips`` carrying a non-bone ``Mask`` group, and a Basis+Pucker
shape key pair) — the natural gradient across ``Ribbon_1``/``Ribbon_2`` and a
5-group vertex that forces the cap are both on it. ``SecondMesh`` is bound to the
same armature but never a fold target, weighted to ``Hips`` only in the baseline —
the "other mesh in the file" gate-2 fixture.
"""
import os
import sys

import bpy
from mathutils import Vector


def _repo_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for arm in list(bpy.data.armatures):
        bpy.data.armatures.remove(arm)


# {survivor bone: fully-mapped doomed bone(s)} — Ribbon_1 -> Strand_1 and
# Ribbon_2 -> Strand_2 each map fully to ONE neighbour (the "one measured
# neighbour maps fully to it" case from the venue script), so a vertex split
# 50/50 across Ribbon_1/Ribbon_2 lands split 50/50 across Strand_1/Strand_2 —
# the gradient-survives assertion.
HAND_MAP = {
    "Ribbon":   {"Bun": 1.0},
    "Ribbon_1": {"Strand_1": 1.0},
    "Ribbon_2": {"Strand_2": 1.0},
}

BONE_RESTS = {
    # name: (head, tail, parent)
    "Hips":     ((0.0, 0.0, 1.0), (0.0, 0.0, 1.1), None),
    "Bun":      ((0.0, 0.0, 1.1), (0.0, 0.0, 1.3), "Hips"),
    "Strand_1": ((0.1, 0.0, 1.0), (0.1, 0.0, 0.8), "Hips"),
    "Strand_2": ((0.1, 0.0, 0.8), (0.1, 0.0, 0.6), "Strand_1"),
    "Extra":    ((0.2, 0.0, 1.0), (0.2, 0.0, 0.9), "Hips"),
    "Ribbon":   ((0.0, 0.05, 1.3), (0.0, 0.05, 1.5), "Bun"),
    "Ribbon_1": ((0.0, 0.05, 1.5), (0.0, 0.05, 1.7), "Ribbon"),
    "Ribbon_2": ((0.0, 0.05, 1.7), (0.0, 0.05, 1.9), "Ribbon_1"),
}


def _build_armature(add_charm_child=False):
    arm_data = bpy.data.armatures.new("TestArmatureData")
    arm_obj = bpy.data.objects.new("TestArmature", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    from avatarprep.core import scene_utils
    ctx = {'active_object': arm_obj, 'object': arm_obj}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    ebs = arm_obj.data.edit_bones

    made = {}
    # BONE_RESTS values are listed parent-before-child, so a single pass suffices.
    for name, (head, tail, parent) in BONE_RESTS.items():
        b = ebs.new(name)
        b.head = Vector(head)
        b.tail = Vector(tail)
        if parent:
            b.parent = made[parent]
        made[name] = b

    if add_charm_child:
        charm = ebs.new("Charm")
        charm.head = Vector((0.0, 0.05, 1.9))
        charm.tail = Vector((0.0, 0.05, 2.0))
        charm.parent = made["Ribbon_2"]

    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')
    return arm_obj


def _bind(mesh_obj, arm_obj):
    mod = mesh_obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm_obj


def _build_ribbon_mesh(arm_obj):
    """6 verts: v0 pure Ribbon, v1 pure Ribbon_1, v2 the 50/50 Ribbon_1/Ribbon_2
    gradient vertex, v3 pure Ribbon_2, v4 a 6-group vertex (Ribbon_1 doomed plus 5
    survivors) that forces the cap, v5 untouched (Hips only) carrying a non-bone
    "Mask" group — the invariant fixtures."""
    verts = [
        (0.0, 0.05, 1.4),
        (0.0, 0.05, 1.6),
        (0.0, 0.05, 1.65),
        (0.0, 0.05, 1.85),
        (0.02, 0.05, 1.6),
        (0.3, 0.0, 1.0),
    ]
    faces = [(0, 1, 2), (2, 3, 4), (4, 5, 0)]
    md = bpy.data.meshes.new("RibbonMeshData")
    md.from_pydata(verts, [], faces)
    md.update()
    mo = bpy.data.objects.new("RibbonMesh", md)
    bpy.context.collection.objects.link(mo)

    def add(name, pairs):
        for vidx, w in pairs:
            vg = mo.vertex_groups.get(name) or mo.vertex_groups.new(name=name)
            vg.add([vidx], w, 'REPLACE')

    add("Ribbon",   [(0, 1.0)])
    add("Ribbon_1", [(1, 1.0), (2, 0.5), (4, 0.30)])
    add("Ribbon_2", [(2, 0.5), (3, 1.0)])
    add("Strand_1", [(4, 0.15)])
    add("Hips",     [(4, 0.16), (5, 1.0)])
    add("Bun",      [(4, 0.15)])
    add("Strand_2", [(4, 0.14)])
    add("Extra",    [(4, 0.10)])
    add("Mask",     [(5, 0.7)])   # non-bone group — must be bit-identical after the fold

    mo.shape_key_add(name="Basis")
    mo.shape_key_add(name="Pucker")

    _bind(mo, arm_obj)
    return mo


def _build_second_mesh(arm_obj, weight_ribbon=False):
    verts = [(0.3, 0.0, 1.0), (0.35, 0.0, 1.0), (0.35, 0.05, 1.0)]
    md = bpy.data.meshes.new("SecondMeshData")
    md.from_pydata(verts, [], [(0, 1, 2)])
    md.update()
    mo = bpy.data.objects.new("SecondMesh", md)
    bpy.context.collection.objects.link(mo)
    vg = mo.vertex_groups.new(name="Hips")
    vg.add([0, 1, 2], 1.0, 'REPLACE')
    if weight_ribbon:
        rib = mo.vertex_groups.new(name="Ribbon")
        rib.add([0], 0.4, 'REPLACE')
    _bind(mo, arm_obj)
    return mo


def _build_scene(add_charm_child=False, second_mesh_weights_ribbon=False):
    _clear_scene()
    arm = _build_armature(add_charm_child=add_charm_child)
    ribbon = _build_ribbon_mesh(arm)
    second = _build_second_mesh(arm, weight_ribbon=second_mesh_weights_ribbon)
    return arm, ribbon, second


def _weights_of(mesh_obj, names):
    """Independent read-back: {vertex index: {group name: weight}}, restricted to
    ``names`` (bone or not — the caller decides what it's checking)."""
    idx_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups if vg.name in names}
    out = {}
    for v in mesh_obj.data.vertices:
        row = {}
        for g in v.groups:
            name = idx_to_name.get(g.group)
            if name is not None and g.weight > 1e-9:
                row[name] = row.get(name, 0.0) + g.weight
        if row:
            out[v.index] = row
    return out


def _close(a, b, eps=1e-6):
    if set(a) != set(b):
        return False
    return all(abs(a[k] - b[k]) <= eps for k in a)


FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def test_hand_map_fold_and_gradient():
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene()
    all_names = {"Hips", "Bun", "Strand_1", "Strand_2", "Extra",
                "Ribbon", "Ribbon_1", "Ribbon_2"}

    doomed = core.resolve_doomed(arm, ["Ribbon*"])
    check(doomed == ["Ribbon", "Ribbon_1", "Ribbon_2"],
          "resolve_doomed should return the ribbon chain in armature order, got %r" % doomed)

    # whatif must not mutate anything.
    bones_before = {b.name for b in arm.data.bones}
    weights_before = _weights_of(ribbon, all_names)
    preview = core.fold_bones(arm, [ribbon], doomed, HAND_MAP, whatif=True)
    check(preview.get("whatif") is True, "whatif result should carry whatif=True")
    check({b.name for b in arm.data.bones} == bones_before, "whatif mutated the armature")
    check(_weights_of(ribbon, all_names) == weights_before, "whatif mutated the mesh")
    check(preview["touched"] == 5, "expected 5 touched vertices (v0-v4) in whatif, got %r"
          % preview["touched"])
    check(preview["capped"] == 1, "expected 1 capped vertex (v4) in whatif, got %r"
          % preview["capped"])

    result = core.fold_bones(arm, [ribbon], set(doomed), HAND_MAP, whatif=False)

    check(sorted(result["bones_removed"]) == doomed,
          "expected bones_removed == %r, got %r" % (doomed, sorted(result["bones_removed"])))
    remaining = {b.name for b in arm.data.bones}
    check(remaining == {"Hips", "Bun", "Strand_1", "Strand_2", "Extra"},
          "unexpected surviving bone set: %r" % remaining)
    for n in doomed:
        check(ribbon.vertex_groups.get(n) is None, "doomed vertex group %r not removed" % n)

    final = _weights_of(ribbon, {"Hips", "Bun", "Strand_1", "Strand_2", "Extra"})
    check(_close(final.get(0, {}), {"Bun": 1.0}), "v0 (pure Ribbon) should fold to Bun=1.0, got %r"
          % final.get(0))
    check(_close(final.get(1, {}), {"Strand_1": 1.0}),
          "v1 (pure Ribbon_1) should fold to Strand_1=1.0, got %r" % final.get(1))
    check(_close(final.get(2, {}), {"Strand_1": 0.5, "Strand_2": 0.5}),
          "v2 (50/50 gradient) should land 50/50 Strand_1/Strand_2, got %r" % final.get(2))
    check(_close(final.get(3, {}), {"Strand_2": 1.0}),
          "v3 (pure Ribbon_2) should fold to Strand_2=1.0, got %r" % final.get(3))

    v4 = final.get(4, {})
    check(v4 is not None and len(v4) <= core.MAX_BONE_GROUPS,
          "v4 must have <= %d bone groups after capping, got %r" % (core.MAX_BONE_GROUPS, v4))
    check("Extra" not in v4, "v4's smallest pre-cap group (Extra) should have been dropped, got %r" % v4)
    expected_v4 = {"Strand_1": 0.5, "Hips": 0.16 / 0.90, "Bun": 0.15 / 0.90, "Strand_2": 0.14 / 0.90}
    check(_close(v4, expected_v4, eps=1e-4),
          "v4 capped+renormalised mismatch: expected %r, got %r" % (expected_v4, v4))
    check(abs(sum(v4.values()) - 1.0) < 1e-6, "v4 weights should renormalise to 1.0, got sum=%r"
          % sum(v4.values()))

    # v5: untouched (no doomed weight) — bone weight AND the non-bone Mask group unchanged.
    check(_close(final.get(5, {}), {"Hips": 1.0}), "v5 (untouched) changed: %r" % final.get(5))
    mask_after = _weights_of(ribbon, {"Mask"})
    check(_close(mask_after.get(5, {}), {"Mask": 0.7}), "non-bone Mask group changed: %r"
          % mask_after.get(5))

    sk = ribbon.data.shape_keys
    check(sk is not None and [k.name for k in sk.key_blocks] == ["Basis", "Pucker"],
          "shape-key metadata changed")

    dt = result["destination_totals"]
    check(set(dt) >= {"Bun", "Strand_1", "Strand_2"}, "destination_totals missing an expected bone: %r" % dt)
    # Mass: every fixture vertex sums to 1, so the fold moves weight without losing
    # any — the survivors' "after" total equals the "before" total over all bones.
    total_before = sum(r["before"] for r in dt.values())
    total_after = sum(r["after"] for n, r in dt.items() if n not in doomed)
    check(abs(total_before - 6.0) < 1e-6 and abs(total_after - total_before) < 1e-6,
          "mass not conserved: before=%r after=%r (6 unit vertices)" % (total_before, total_after))
    check(all(dt[n]["after"] == 0.0 for n in doomed if n in dt),
          "a doomed bone keeps weight in destination_totals: %r" % dt)
    check(result["touched"] == 5 and result["capped"] == 1,
          "expected touched=5 capped=1 on the real run, got touched=%r capped=%r"
          % (result["touched"], result["capped"]))


def test_auto_map_deterministic_readonly_and_reusable():
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene()
    all_names = {"Hips", "Bun", "Strand_1", "Strand_2", "Extra",
                "Ribbon", "Ribbon_1", "Ribbon_2"}
    doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))

    bones_before = {b.name for b in arm.data.bones}
    weights_before = _weights_of(ribbon, all_names)

    table1, info1 = core.auto_map(arm, [ribbon], doomed)
    table2, info2 = core.auto_map(arm, [ribbon], doomed)

    check(table1 == table2, "auto_map is not deterministic: %r vs %r" % (table1, table2))
    info1_cmp = {k: (v["ratio"], v["ambiguous"]) for k, v in info1.items()}
    info2_cmp = {k: (v["ratio"], v["ambiguous"]) for k, v in info2.items()}
    check(info1_cmp == info2_cmp, "auto_map info not deterministic: %r vs %r" % (info1_cmp, info2_cmp))
    check({b.name for b in arm.data.bones} == bones_before, "auto_map mutated the armature")
    check(_weights_of(ribbon, all_names) == weights_before, "auto_map mutated the mesh")

    # Table shape: exactly {doomed: {survivor: fraction}}, directly --map-consumable.
    survivors = {"Hips", "Bun", "Strand_1", "Strand_2", "Extra"}
    for bone, dests in table1.items():
        check(bone in doomed, "auto table names a non-doomed bone %r" % bone)
        check(set(dests) <= survivors, "auto row for %s names a non-survivor: %r" % (bone, dests))
        check(abs(sum(dests.values()) - 1.0) < 1e-6,
              "auto row for %s does not sum to 1.0: %r" % (bone, dests))
        i = info1[bone]
        if i["ratio"] is not None:
            check(i["ambiguous"] == (i["ratio"] <= core.AMBIGUOUS_RATIO),
                  "ambiguous flag disagrees with its own ratio for %s: %r" % (bone, i))

    # The written table is directly reusable as --map: feed it into a real run on a
    # FRESH copy of the fixture and confirm it clears every gate.
    arm2, ribbon2, second2 = _build_scene()
    doomed2 = set(core.resolve_doomed(arm2, ["Ribbon*"]))
    result = core.fold_bones(arm2, [ribbon2], doomed2, table1, whatif=False)
    check(sorted(result["bones_removed"]) == sorted(doomed2),
          "auto table replayed as --map did not remove the doomed bones: %r" % result)


def test_gate_surviving_child():
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene(add_charm_child=True)
    doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))
    check("Charm" not in doomed, "fixture bug: Charm must not match the Ribbon* glob")

    try:
        core.fold_bones(arm, [ribbon], doomed, HAND_MAP, whatif=True)
    except core.FoldRefused as refused:
        check(refused.kind == "surviving_child", "wrong gate kind: %r" % refused.kind)
        offenders = refused.offenders
        check(any(o["bone"] == "Ribbon_2" and o["child"] == "Charm" for o in offenders),
              "expected Ribbon_2/Charm named, got %r" % offenders)
    else:
        FAILURES.append("expected FoldRefused(surviving_child), the fold ran")


def test_gate_foreign_mesh():
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene(second_mesh_weights_ribbon=True)
    doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))

    try:
        core.fold_bones(arm, [ribbon], doomed, HAND_MAP, whatif=True)
    except core.FoldRefused as refused:
        check(refused.kind == "foreign_mesh", "wrong gate kind: %r" % refused.kind)
        check(any(o["mesh"] == "SecondMesh" for o in refused.offenders),
              "expected SecondMesh named, got %r" % refused.offenders)
    else:
        FAILURES.append("expected FoldRefused(foreign_mesh), the fold ran")


def test_gate_bad_destination():
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene()
    doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))
    bad_map = {"Ribbon": {"Bun": 1.0}, "Ribbon_1": {"NoSuchBone": 1.0}, "Ribbon_2": {"Strand_2": 1.0}}

    try:
        core.fold_bones(arm, [ribbon], doomed, bad_map, whatif=True)
    except core.FoldRefused as refused:
        check(refused.kind == "bad_destination", "wrong gate kind: %r" % refused.kind)
        check(any(o["bone"] == "Ribbon_1" and o["destination"] == "NoSuchBone"
                 for o in refused.offenders),
              "expected Ribbon_1 -> NoSuchBone named, got %r" % refused.offenders)
    else:
        FAILURES.append("expected FoldRefused(bad_destination), the fold ran")


def test_gate_unmapped_bone():
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene()
    doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))
    incomplete_map = {"Ribbon": {"Bun": 1.0}, "Ribbon_2": {"Strand_2": 1.0}}  # Ribbon_1 dropped

    try:
        core.fold_bones(arm, [ribbon], doomed, incomplete_map, whatif=True)
    except core.FoldRefused as refused:
        check(refused.kind == "unmapped_bone", "wrong gate kind: %r" % refused.kind)
        check("Ribbon_1" in refused.offenders, "expected Ribbon_1 named, got %r" % refused.offenders)
    else:
        FAILURES.append("expected FoldRefused(unmapped_bone), the fold ran")


def test_gate_zero_sum():
    """A row that fails to sum to 1 is now caught earlier, by
    ``check_fraction_sums`` (see ``test_gate_fraction_sum``) — so a *valid*
    (sum-to-1) map can no longer zero-sum a touched vertex through the public
    ``fold_bones()`` gate sequence: with every weighted row summing to exactly 1,
    a vertex's post-blend total is algebraically ``sum(w_b for b in its doomed
    bones)``, which is > 0 for any vertex ``fold_bones`` calls "touched". The
    zero-sum check survives as a defensive backstop, so this test exercises it directly at the unit the gate reads from —
    ``_fold_mesh`` — rather than fabricating a public-API map that can no longer
    reach it."""
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene()
    doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))
    all_bone_names = {b.name for b in arm.data.bones}
    weights = core._vertex_bone_weights(ribbon, all_bone_names)
    zero_map = {"Ribbon": {"Bun": 0.0}, "Ribbon_1": {"Strand_1": 1.0}, "Ribbon_2": {"Strand_2": 1.0}}

    res = core._fold_mesh(weights, doomed, zero_map)
    check(res["zero_sum"] == [0], "expected vertex 0 to zero-sum, got %r" % res["zero_sum"])


def test_gate_fraction_sum():
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene()
    doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))
    partial_map = {"Ribbon": {"Bun": 1.0}, "Ribbon_1": {"Strand_1": 0.5}, "Ribbon_2": {"Strand_2": 1.0}}

    try:
        core.fold_bones(arm, [ribbon], doomed, partial_map, whatif=True)
    except core.FoldRefused as refused:
        check(refused.kind == "bad_fraction_sum", "wrong gate kind: %r" % refused.kind)
        check(any(o["bone"] == "Ribbon_1" and abs(o["sum"] - 0.5) < 1e-9
                 for o in refused.offenders),
              "expected Ribbon_1 sum=0.5 named, got %r" % refused.offenders)
    else:
        FAILURES.append("expected FoldRefused(bad_fraction_sum), the fold ran")


def test_rerun_refuses():
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene()
    doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))
    core.fold_bones(arm, [ribbon], doomed, HAND_MAP, whatif=False)

    try:
        core.resolve_doomed(arm, ["Ribbon*"])
    except core.NoBonesMatched as e:
        check(e.pattern == "Ribbon*", "expected the pattern named, got %r" % e.pattern)
    else:
        FAILURES.append("expected NoBonesMatched on a rerun (the doomed bones are gone)")


def test_folded_stamp_readable_by_report_stamps():
    """The contract requires the stamp be readable by report_stamps — exercised at
    the core level (the CLI writes the identical stamp via the same helper)."""
    _repo_root()
    from avatarprep.core import scene_utils

    arm, ribbon, second = _build_scene()
    line = "fold_bones --targets RibbonMesh --bones 'Ribbon*' --map blend.json"
    scene_utils.write_stamp(ribbon, scene_utils.STAMP_FOLDED, line)
    rep = scene_utils.report_stamps(bpy.context.scene)
    entries = [m for a in rep["armatures"] for m in a["meshes"]] + rep["unbound"]
    hit = next((e for e in entries if e["name"] == "RibbonMesh"), None)
    check(hit is not None and hit.get("folded") == line,
          "avatarprep_folded stamp not surfaced by report_stamps: %r" % hit)


def _refused(fn, kind, label):
    """Run ``fn``; return its FoldRefused when it carries ``kind``, else record a
    failure (a different exception included) and return None."""
    from avatarprep.core import fold_bones as core
    try:
        fn()
    except core.FoldRefused as refused:
        if refused.kind == kind:
            return refused
        FAILURES.append("%s: wrong gate kind %r (wanted %r)" % (label, refused.kind, kind))
        return None
    except Exception as e:
        FAILURES.append("%s: expected FoldRefused(%s), got %r" % (label, kind, e))
        return None
    FAILURES.append("%s: expected FoldRefused(%s), the fold ran" % (label, kind))
    return None


def test_untouched_vertex_over_cap():
    """An untouched vertex that already carries 5 bone groups is not the fold's to
    cap: whatif and the real run both succeed and agree, and it is left as it was."""
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene()
    for name in ("Bun", "Strand_1", "Strand_2", "Extra"):
        ribbon.vertex_groups[name].add([5], 0.1, 'REPLACE')
    survivors = {"Hips", "Bun", "Strand_1", "Strand_2", "Extra"}
    v5_before = _weights_of(ribbon, survivors)[5]
    doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))

    preview = core.fold_bones(arm, [ribbon], doomed, HAND_MAP, whatif=True)
    try:
        result = core.fold_bones(arm, [ribbon], doomed, HAND_MAP, whatif=False)
    except AssertionError as e:
        FAILURES.append("real run refused an untouched 5-group vertex whatif passed: %s" % e)
        return
    check((preview["touched"], preview["capped"]) == (result["touched"], result["capped"]),
          "whatif and the real run disagree: %r vs %r" % (preview, result))
    check(_close(_weights_of(ribbon, survivors).get(5, {}), v5_before),
          "untouched 5-group vertex changed: %r" % _weights_of(ribbon, survivors).get(5))


def test_gate_bad_fraction():
    _repo_root()
    from avatarprep.core import fold_bones as core

    rows = {
        "nan": {"Strand_1": float("nan")},
        "negative": {"Strand_1": 1.5, "Strand_2": -0.5},   # sums to 1
        "over-1": {"Strand_1": 2.0},
        "non-numeric": {"Strand_1": "1.0"},
    }
    for label, row in rows.items():
        arm, ribbon, second = _build_scene()
        doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))
        bad_map = dict(HAND_MAP, Ribbon_1=row)
        refused = _refused(lambda: core.fold_bones(arm, [ribbon], doomed, bad_map, whatif=False),
                           "bad_fraction", "bad_fraction %s" % label)
        if refused is not None:
            check(all(o["bone"] == "Ribbon_1" for o in refused.offenders)
                  and {o["destination"] for o in refused.offenders} <= set(row),
                  "bad_fraction %s: expected Ribbon_1 -> its destination named, got %r"
                  % (label, refused.offenders))
        check({b.name for b in arm.data.bones} >= doomed,
              "bad_fraction %s: the armature was written before the refusal" % label)


def test_resolve_overlapping_patterns():
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene()
    try:
        doomed = core.resolve_doomed(arm, ["Ribbon*", "Ribbon_1"])
    except core.NoBonesMatched as e:
        FAILURES.append("a pattern whose bones an earlier pattern already took refused: %s" % e)
        return
    check(doomed == ["Ribbon", "Ribbon_1", "Ribbon_2"],
          "overlapping patterns should list each bone once, got %r" % doomed)


def test_gate_bone_parented():
    _repo_root()
    from avatarprep.core import fold_bones as core

    arm, ribbon, second = _build_scene()
    charm = bpy.data.objects.new("CharmEmpty", None)
    bpy.context.collection.objects.link(charm)
    charm.parent = arm
    charm.parent_type = 'BONE'
    charm.parent_bone = "Ribbon_2"
    doomed = set(core.resolve_doomed(arm, ["Ribbon*"]))
    all_names = {b.name for b in arm.data.bones}
    weights_before = _weights_of(ribbon, all_names)

    refused = _refused(lambda: core.fold_bones(arm, [ribbon], doomed, HAND_MAP, whatif=False),
                       "bone_parented", "bone_parented")
    if refused is not None:
        check(any(o["object"] == "CharmEmpty" and o["bone"] == "Ribbon_2" for o in refused.offenders),
              "expected CharmEmpty on Ribbon_2 named, got %r" % refused.offenders)
    check({b.name for b in arm.data.bones} == all_names and _weights_of(ribbon, all_names) == weights_before,
          "bone_parented: something was written before the refusal")


def _door(script, args):
    import subprocess
    cmd = [bpy.app.binary_path, "--background", "--factory-startup", "--python",
           os.path.join(_repo_root(), "cli", script), "--"] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_cli(tmp):
    """The door end to end: auto writes the table and exits 1; the table replayed as
    --map exits 0, prints the canonical recipe and stamps it; report_stamps reads it;
    a target with two Armature modifiers exits 2."""
    import json
    import shlex
    from avatarprep.core import scene_utils

    arm, ribbon, second = _build_scene()
    src = os.path.join(tmp, "in.blend")
    bpy.ops.wm.save_as_mainfile(filepath=src)
    table = os.path.join(tmp, "table.json")
    out = os.path.join(tmp, "out.blend")

    rc, txt = _door("fold_bones.py", ["--in", src, "--targets", "RibbonMesh", "--bones", "Ribbon*",
                                      "--map", "auto", "--map-out", table])
    check(rc == 1 and os.path.isfile(table) and "recipe:" not in txt,
          "--map auto should exit 1 having written the table, no recipe (rc=%d)\n%s" % (rc, txt))
    if os.path.isfile(table):
        with open(table, encoding="utf-8") as fh:
            check(set(json.load(fh)) == {"Ribbon", "Ribbon_1", "Ribbon_2"}, "auto table rows wrong")

    rc, txt = _door("fold_bones.py", ["--in", src, "--targets", "RibbonMesh", "--bones", "Ribbon*",
                                      "--map", table, "--report", os.path.join(tmp, "r.json"),
                                      "--out", out])
    line = shlex.join(["fold_bones", "--targets", "RibbonMesh", "--bones", "Ribbon*", "--map", table])
    check(rc == 0 and "=> OK" in txt and ("AVATARPREP: recipe: %s" % line) in txt,
          "a real run should exit 0 and print the recipe %r (rc=%d)\n%s" % (line, rc, txt))
    if os.path.isfile(out):
        bpy.ops.wm.open_mainfile(filepath=out)
        stamp = bpy.data.objects["RibbonMesh"].get(scene_utils.STAMP_FOLDED)
        check(stamp == line, "the stamp is the printed recipe: %r" % stamp)
    rc, txt = _door("report_stamps.py", ["--in", out])
    check(rc == 0 and "mesh RibbonMesh folded=" in txt and "Traceback" not in txt,
          "report_stamps should print the folded stamp (rc=%d)\n%s" % (rc, txt))

    arm, ribbon, second = _build_scene()
    extra = ribbon.modifiers.new("Armature.001", 'ARMATURE')
    extra.object = arm
    two = os.path.join(tmp, "two_mods.blend")
    bpy.ops.wm.save_as_mainfile(filepath=two)
    rc, txt = _door("fold_bones.py", ["--in", two, "--targets", "RibbonMesh", "--bones", "Ribbon*",
                                      "--map", table, "--whatif"])
    check(rc == 2 and "2 ARMATURE modifiers" in txt,
          "a target with two Armature modifiers should exit 2 (rc=%d)\n%s" % (rc, txt))


def main():
    test_hand_map_fold_and_gradient()
    test_auto_map_deterministic_readonly_and_reusable()
    test_gate_surviving_child()
    test_gate_foreign_mesh()
    test_gate_bad_destination()
    test_gate_unmapped_bone()
    test_gate_fraction_sum()
    test_gate_zero_sum()
    test_rerun_refuses()
    test_folded_stamp_readable_by_report_stamps()
    test_untouched_vertex_over_cap()
    test_gate_bad_fraction()
    test_resolve_overlapping_patterns()
    test_gate_bone_parented()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_cli(tmp)

    if FAILURES:
        for f in FAILURES:
            print("FOLD_TEST FAIL:", f)
        sys.exit(1)
    print("FOLD_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "FOLD_TEST")
