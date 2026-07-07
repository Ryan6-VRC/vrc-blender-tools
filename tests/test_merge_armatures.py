"""Synthetic headless tests for avatarprep.core.merge_armatures.

Run::

    blender --background --factory-startup --python tests/test_merge_armatures.py

Prints ``MERGE_TEST OK`` and exits 0 if all cases pass; prints
``MERGE_TEST FAIL: <reason>`` and exits 1 otherwise.
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
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _make_armature(name, bones):
    """bones: list of (bone_name, head Vector, parent_name_or_None)."""
    from avatarprep.core import scene_utils
    arm_data = bpy.data.armatures.new(name + "Data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    ctx = {'active_object': arm_obj, 'object': arm_obj}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    ebs = arm_obj.data.edit_bones
    for bname, head, parent in bones:
        b = ebs.new(bname)
        b.head = head
        b.tail = head + Vector((0.0, 0.0, 0.1))
    for bname, head, parent in bones:
        if parent:
            ebs[bname].parent = ebs[parent]
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')
    return arm_obj


def _make_mesh(name, arm_obj, bone_name, head):
    """Quad mesh near `head`, fully weighted to `bone_name`, bound by modifier."""
    z = head.z
    verts = [(head.x - 0.05, head.y, z), (head.x + 0.05, head.y, z),
             (head.x + 0.05, head.y, z + 0.1), (head.x - 0.05, head.y, z + 0.1)]
    md = bpy.data.meshes.new(name + "Data")
    md.from_pydata(verts, [], [(0, 1, 2, 3)])
    md.update()
    mo = bpy.data.objects.new(name, md)
    bpy.context.collection.objects.link(mo)
    vg = mo.vertex_groups.new(name=bone_name)
    vg.add(list(range(4)), 1.0, 'REPLACE')
    mod = mo.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm_obj
    mo.parent = arm_obj
    return mo


FAILURES = []


def _fail(msg):
    FAILURES.append(msg)


def _stamp(obj, **kv):
    """Set raw custom props on an object (bypasses write_stamp so a test can plant a
    non-str corrupt value too)."""
    for k, v in kv.items():
        obj[k] = v


def _twin_pair():
    """Two structurally-identical (clean) skeletons — the fixture for stamp-only
    gate cases."""
    base = _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Spine", Vector((0, 0, 1.2)), "Hips"),
    ])
    merge = _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Spine", Vector((0, 0, 1.2)), "Hips"),
    ])
    return base, merge


def test_stamp_equal_pass():
    _clear_scene()
    base, merge = _twin_pair()
    _stamp(base, avatarprep_base="shinano", avatarprep_state="unproportioned")
    _stamp(merge, avatarprep_base="shinano", avatarprep_state="unproportioned")
    from avatarprep.core.merge_armatures import armature_compat, merge_armatures
    rep = armature_compat(base, merge)
    if not rep["clean"] or not rep["stamp_clean"] or rep["stamp_mismatches"]:
        _fail("stamp_equal: expected clean/stamp_clean with no mismatches, got %r" % rep)
    res = merge_armatures(base, merge)
    if res["verdict"] != "PASS":
        _fail("stamp_equal: expected PASS, got %r" % res)


def test_stamp_base_different_fail():
    _clear_scene()
    base, merge = _twin_pair()
    _stamp(base, avatarprep_base="shinano")
    _stamp(merge, avatarprep_base="plum")
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge)
    if res["verdict"] != "FAIL":
        _fail("base_diff: expected FAIL, got %r" % res["verdict"])
    if not any("base mismatch: base=" in o for o in res.get("offenders") or []):
        _fail("base_diff: expected 'base mismatch: base=…' offender, got %r"
              % res.get("offenders"))
    if len(_scene_armatures()) != 2:
        _fail("base_diff: scene must be unmutated (2 armatures), got %d"
              % len(_scene_armatures()))


def test_stamp_state_different_fail():
    _clear_scene()
    base, merge = _twin_pair()
    _stamp(base, avatarprep_base="shinano", avatarprep_state="shinano-tall")
    _stamp(merge, avatarprep_base="shinano", avatarprep_state="unproportioned")
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge)
    if res["verdict"] != "FAIL":
        _fail("state_diff: expected FAIL, got %r" % res["verdict"])
    if not any("state mismatch: base=" in o for o in res.get("offenders") or []):
        _fail("state_diff: expected 'state mismatch: base=…' offender, got %r"
              % res.get("offenders"))


def test_stamp_interrupted_fail():
    from avatarprep.core import scene_utils
    _clear_scene()
    base, merge = _twin_pair()
    _stamp(base, avatarprep_state="unproportioned")
    _stamp(merge, avatarprep_state=scene_utils.STATE_APPLYING)
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge)
    if res["verdict"] != "FAIL":
        _fail("interrupted: expected FAIL, got %r" % res["verdict"])
    if not any("state interrupted: base=" in o for o in res.get("offenders") or []):
        _fail("interrupted: expected 'state interrupted:' offender, got %r"
              % res.get("offenders"))


def test_stamp_corrupt_fail():
    _clear_scene()
    base, merge = _twin_pair()
    _stamp(base, avatarprep_state="unproportioned")
    _stamp(merge, avatarprep_state=5)  # non-str → corrupt
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge)
    if res["verdict"] != "FAIL":
        _fail("corrupt: expected FAIL, got %r" % res["verdict"])
    if not any("state corrupt: base=" in o for o in res.get("offenders") or []):
        _fail("corrupt: expected 'state corrupt:' offender, got %r"
              % res.get("offenders"))


def test_stamp_missing_warns_and_passes():
    _clear_scene()
    base, merge = _twin_pair()
    _stamp(base, avatarprep_base="shinano")  # merge has none → missing on one side
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge)
    if res["verdict"] != "PASS":
        _fail("missing: expected warn-and-proceed PASS, got %r" % res["verdict"])
    warns = (res.get("report") or {}).get("warnings", [])
    if not any("base stamp missing" in w for w in warns):
        _fail("missing: expected a base-missing warning in report warnings, got %r" % warns)


def test_force_stamps_split():
    from avatarprep.core.merge_armatures import merge_armatures

    # (1) stamp-only FAIL cleared by force_stamps → PASS carrying forced_stamp.
    _clear_scene()
    base, merge = _twin_pair()
    _stamp(base, avatarprep_base="shinano")
    _stamp(merge, avatarprep_base="plum")
    res = merge_armatures(base, merge, force_stamps=True)
    if res["verdict"] != "PASS":
        _fail("force_stamps: expected PASS with force_stamps, got %r" % res["verdict"])
    if not any("base mismatch" in o for o in res.get("forced_stamp") or []):
        _fail("force_stamps: PASS must carry forced_stamp breach line, got %r"
              % res.get("forced_stamp"))

    # Co-occurring structural + stamp mismatch: a rename (structural) + base diff.
    def _mixed():
        b = _make_armature("Base", [
            ("Hips", Vector((0, 0, 1.0)), None),
            ("Breast_L", Vector((0.1, 0, 1.3)), "Hips"),
        ])
        m = _make_armature("Merge", [
            ("Hips", Vector((0, 0, 1.0)), None),
            ("breast_L", Vector((0.1, 0, 1.3)), "Hips"),  # rename → structural
        ])
        _stamp(b, avatarprep_base="shinano")
        _stamp(m, avatarprep_base="plum")                 # → stamp mismatch
        return b, m

    # (2) force_stamps alone: structural still FAILs.
    _clear_scene()
    b, m = _mixed()
    res = merge_armatures(b, m, force_stamps=True)
    if res["verdict"] != "FAIL":
        _fail("force_stamps: co-occurring structural must still FAIL, got %r" % res["verdict"])
    if not any("suspected rename" in o for o in res.get("offenders") or []):
        _fail("force_stamps: expected structural offender to remain, got %r"
              % res.get("offenders"))

    # (3) force (structural) alone: stamp mismatch still FAILs — proves the split.
    _clear_scene()
    b, m = _mixed()
    res = merge_armatures(b, m, force=True)
    if res["verdict"] != "FAIL":
        _fail("force_split: structural force must NOT clear a stamp mismatch, got %r"
              % res["verdict"])
    if not any("base mismatch" in o for o in res.get("offenders") or []):
        _fail("force_split: expected stamp offender to remain under force, got %r"
              % res.get("offenders"))

    # (4) both forces → PASS.
    _clear_scene()
    b, m = _mixed()
    res = merge_armatures(b, m, force=True, force_stamps=True)
    if res["verdict"] != "PASS":
        _fail("force_both: force+force_stamps should PASS, got %r (offenders=%r)"
              % (res["verdict"], res.get("offenders")))


def test_compat_flags_rename():
    _clear_scene()
    base = _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Breast_L", Vector((0.1, 0, 1.3)), "Hips"),
    ])
    merge = _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("breast_L", Vector((0.1, 0, 1.3)), "Hips"),  # renamed, same position
    ])
    from avatarprep.core.merge_armatures import armature_compat
    rep = armature_compat(base, merge)
    if rep["clean"]:
        _fail("compat: expected clean=False for a rename")
    sr = rep["suspected_renames"]
    if not any(x["merge"] == "breast_L" and x["base"] == "Breast_L" for x in sr):
        _fail("compat: expected suspected_renames breast_L->Breast_L, got %r" % sr)
    if "Hips" not in rep["matched"]:
        _fail("compat: expected Hips matched, got %r" % rep["matched"])


def _scene_armatures():
    return [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']


def test_guard_unmutated():
    _clear_scene()
    base = _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Breast_L", Vector((0.1, 0, 1.3)), "Hips"),
    ])
    merge = _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("breast_L", Vector((0.1, 0, 1.3)), "Hips"),  # rename -> not clean
    ])
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge)  # no rename_map, no force
    if res["verdict"] != "FAIL":
        _fail("guard: expected FAIL, got %r" % res["verdict"])
    if len(_scene_armatures()) != 2:
        _fail("guard: expected scene unmutated (2 armatures), got %d"
              % len(_scene_armatures()))
    if "breast_L" not in [b.name for b in merge.data.bones]:
        _fail("guard: rename_map rollback failed; merge bone names=%r"
              % [b.name for b in merge.data.bones])


def test_same_armature_guard():
    _clear_scene()
    base = _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Spine", Vector((0, 0, 1.2)), "Hips"),
    ])
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, base)  # same object for base and merge
    if res.get("verdict") != "FAIL" or res.get("reason") != "same-armature":
        _fail("same_armature: expected FAIL/reason=same-armature, got %r" % res)
    if res.get("offenders") != [base.name]:
        _fail("same_armature: expected offenders=[%r], got %r" % (base.name, res.get("offenders")))
    if len(_scene_armatures()) != 1:
        _fail("same_armature: scene should be unmutated (1 armature), got %d"
              % len(_scene_armatures()))
    if {b.name for b in base.data.bones} != {"Hips", "Spine"}:
        _fail("same_armature: base bones should be unmutated, got %r"
              % {b.name for b in base.data.bones})


def test_clean_merge_weights():
    _clear_scene()
    base = _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Spine", Vector((0, 0, 1.2)), "Hips"),
    ])
    _make_mesh("BaseBody", base, "Spine", Vector((0, 0, 1.2)))
    merge = _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Tail", Vector((0, -0.1, 1.0)), "Hips"),  # unique additive bone
    ])
    tail_mesh = _make_mesh("TailMesh", merge, "Tail", Vector((0, -0.1, 1.0)))
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge)
    if res["verdict"] != "PASS":
        _fail("clean: expected PASS, got %r (postcheck=%r)"
              % (res["verdict"], res.get("postcheck")))
        return
    arms = _scene_armatures()
    if len(arms) != 1 or arms[0].name != "Armature":
        _fail("clean: expected one armature named 'Armature', got %r"
              % [a.name for a in arms])
        return
    arm = arms[0]
    names = {b.name for b in arm.data.bones}
    if names != {"Hips", "Spine", "Tail"}:
        _fail("clean: expected union {Hips,Spine,Tail}, got %r" % names)
    # Tail mesh weight preserved and bound to the unified armature.
    vg = tail_mesh.vertex_groups.get("Tail")
    if vg is None:
        _fail("clean: Tail vertex group lost")
    else:
        w = max((g.weight for v in tail_mesh.data.vertices
                 for g in v.groups if g.group == vg.index), default=0.0)
        if w <= 0.0:
            _fail("clean: Tail weight not preserved (w=%r)" % w)
    if not any(m.type == 'ARMATURE' and m.object == arm
               for m in tail_mesh.modifiers):
        _fail("clean: Tail mesh not rebound to unified armature")


def test_rename_map():
    _clear_scene()
    base = _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Breast_L", Vector((0.1, 0, 1.3)), "Hips"),
    ])
    merge = _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("breast_L", Vector((0.1, 0, 1.3)), "Hips"),
    ])
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge, rename_map={"breast_L": "Breast_L"})
    if res["verdict"] != "PASS":
        _fail("rename_map: expected PASS, got %r (offenders=%r)"
              % (res["verdict"], res.get("offenders")))
        return
    arm = _scene_armatures()[0]
    names = {b.name for b in arm.data.bones}
    if names != {"Hips", "Breast_L"}:
        _fail("rename_map: expected {Hips,Breast_L}, got %r" % names)


def test_force_parent_mismatch():
    _clear_scene()
    base = _make_armature("Base", [
        ("Root", Vector((0, 0, 0.0)), None),
        ("Hips", Vector((0, 0, 1.0)), "Root"),
        ("Spine", Vector((0, 0, 1.2)), "Hips"),     # parent = Hips
    ])
    merge = _make_armature("Merge", [
        ("Root", Vector((0, 0, 0.0)), None),
        ("Hips", Vector((0, 0, 1.0)), "Root"),
        ("Spine", Vector((0, 0, 1.2)), "Root"),     # parent = Root (mismatch)
    ])
    from avatarprep.core.merge_armatures import armature_compat, merge_armatures
    rep = armature_compat(base, merge)
    if not any(p["bone"] == "Spine" for p in rep["parent_mismatches"]):
        _fail("force: expected Spine in parent_mismatches, got %r"
              % rep["parent_mismatches"])
    res_fail = merge_armatures(base, merge)  # unforced
    if res_fail["verdict"] != "FAIL":
        _fail("force: expected unforced FAIL, got %r" % res_fail["verdict"])
    if len(_scene_armatures()) != 2:
        _fail("force: unforced FAIL should leave scene unmutated (2 armatures), got %d"
              % len(_scene_armatures()))
    merge_spine = merge.data.bones.get("Spine")
    if merge_spine is None or merge_spine.parent is None or merge_spine.parent.name != "Root":
        _fail("force: unforced FAIL should leave merge Spine parent as 'Root', got %r"
              % (merge_spine.parent.name if merge_spine and merge_spine.parent else None))
    # Rebuild (the failed call rolled back; scene still has both) and force.
    res_ok = merge_armatures(base, merge, force=True)
    if res_ok["verdict"] != "PASS":
        _fail("force: expected forced PASS, got %r (postcheck=%r)"
              % (res_ok["verdict"], res_ok.get("postcheck")))
        return
    # force = "keep base's copy anyway": base Spine's hierarchy must NOT be
    # rewritten to the merge's topology (parent stays Hips, not Root).
    arm = _scene_armatures()[0]
    spine = arm.data.bones.get("Spine")
    if spine is None or spine.parent is None or spine.parent.name != "Hips":
        _fail("force: base Spine hierarchy corrupted — parent=%r (expected 'Hips')"
              % (spine.parent.name if spine and spine.parent else None))


def test_vgroup_fold_branch():
    # Directly exercise _process_vertex_groups' fold branch (a mesh carrying both
    # 'X' and 'X.merge' groups) — the no-Join-Meshes merge flow only ever hits the
    # rename-back branch, so this path is otherwise untested.
    _clear_scene()
    md = bpy.data.meshes.new("FoldData")
    md.from_pydata([(0, 0, 0), (0.1, 0, 0), (0.1, 0, 0.1), (0, 0, 0.1)], [], [(0, 1, 2, 3)])
    md.update()
    mo = bpy.data.objects.new("FoldMesh", md)
    bpy.context.collection.objects.link(mo)
    mo.vertex_groups.new(name="Bone").add([0, 1, 2, 3], 0.4, 'REPLACE')
    mo.vertex_groups.new(name="Bone.merge").add([0, 1, 2, 3], 0.5, 'REPLACE')
    from avatarprep.core.merge_armatures import _process_vertex_groups
    _process_vertex_groups([mo])
    names = {g.name for g in mo.vertex_groups}
    if "Bone.merge" in names:
        _fail("fold: 'Bone.merge' should be removed after fold, groups=%r" % names)
    if "Bone" not in names:
        _fail("fold: 'Bone' should remain after fold, groups=%r" % names)
        return
    vg = mo.vertex_groups.get("Bone")
    w = max((g.weight for v in mo.data.vertices for g in v.groups
             if g.group == vg.index), default=0.0)
    if abs(w - 0.9) > 1e-4:
        _fail("fold: expected combined weight ~0.9 (0.4+0.5), got %r" % w)


def test_rename_map_dup_target():
    # A many-to-one rename_map (two sources -> same target) must FAIL before any
    # mutation — applying it would auto-suffix the second to X.001 and break the
    # rollback guarantee.
    _clear_scene()
    base = _make_armature("Base", [("Hips", Vector((0, 0, 1.0)), None)])
    merge = _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("a", Vector((0.2, 0, 1.0)), "Hips"),
        ("b", Vector((0.3, 0, 1.0)), "Hips"),
    ])
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge, rename_map={"a": "X", "b": "X"})
    if res["verdict"] != "FAIL" or res.get("reason") != "rename_map":
        _fail("dup_target: expected FAIL/reason=rename_map, got %r" % res)
    names = {bn.name for bn in merge.data.bones}
    if names != {"Hips", "a", "b"}:
        _fail("dup_target: merge armature should be unmutated, got %r" % names)


def test_rename_map_swap():
    # A swap/chain (target is also a source) must FAIL before any mutation —
    # _edit_rename can't permute, so applying it would corrupt the rename and
    # break rollback.
    _clear_scene()
    base = _make_armature("Base", [("Hips", Vector((0, 0, 1.0)), None)])
    merge = _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("L", Vector((0.2, 0, 1.0)), "Hips"),
        ("R", Vector((-0.2, 0, 1.0)), "Hips"),
    ])
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge, rename_map={"L": "R", "R": "L"})
    if res["verdict"] != "FAIL" or res.get("reason") != "rename_map":
        _fail("swap: expected FAIL/reason=rename_map, got %r" % res)
    names = {bn.name for bn in merge.data.bones}
    if names != {"Hips", "L", "R"}:
        _fail("swap: merge armature should be unmutated, got %r" % names)


def test_scene_scoping():
    # A correct merge must PASS even when the scene holds unrelated objects: a
    # static prop with no armature modifier, a duplicate-named pair (Decor /
    # Decor.001), and a second unrelated armature. None should force a FAIL.
    _clear_scene()
    base = _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Spine", Vector((0, 0, 1.2)), "Hips"),
    ])
    _make_mesh("BaseBody", base, "Spine", Vector((0, 0, 1.2)))
    merge = _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Tail", Vector((0, -0.1, 1.0)), "Hips"),
    ])
    _make_mesh("TailMesh", merge, "Tail", Vector((0, -0.1, 1.0)))
    # Unrelated scene state:
    for nm in ("Prop", "Decor", "Decor.001"):
        d = bpy.data.meshes.new(nm + "D")
        d.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [], [(0, 1, 2)])
        d.update()
        bpy.context.collection.objects.link(bpy.data.objects.new(nm, d))
    _make_armature("Other", [("Bone", Vector((5, 0, 0)), None)])
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge)
    if res["verdict"] != "PASS":
        _fail("scoping: clean merge should PASS despite unrelated scene objects, "
              "got %r (postcheck=%r)" % (res["verdict"], res.get("postcheck")))


def _call_op_expect_loud_cancel(op, ctx, needle):
    """Call ``op`` via ``op_override`` and assert it fails loud.

    Per Blender's own contract (bpy.ops.rst, "Calling Operators"): when an
    operator reports an ``{'ERROR'}``-level message, calling it through
    ``bpy.ops`` raises ``RuntimeError`` with that message *regardless of the
    operator's return status* — even though ``execute()`` itself really does
    return ``{'CANCELLED'}``. So the Python-visible signal for "loud CANCELLED
    with a named message" is this RuntimeError, not a returned value. A bare
    ``{'CANCELLED'}`` return (no exception) would mean the guard did NOT
    report an error — i.e. failed silently — which is exactly the anti-pattern
    the spec forbids.
    """
    from avatarprep.core import scene_utils
    try:
        scene_utils.op_override(op, ctx)
    except RuntimeError as exc:
        if needle not in str(exc):
            _fail("op_guard: expected %r in error, got %r" % (needle, str(exc)))
        return
    _fail("op_guard: expected a loud RuntimeError (named ERROR report), "
          "operator returned normally instead")


def test_merge_operator_guard():
    # The merge operator's genuinely-new logic: the exactly-two-armatures check
    # runs in execute() and returns CANCELLED (loud), NOT a silent greyed poll.
    # Register locally so the core-only tests stay register-free.
    _clear_scene()
    import avatarprep
    avatarprep.register()
    try:
        base = _make_armature("Base", [("Hips", Vector((0, 0, 1.0)), None)])
        m1 = _make_armature("M1", [("Hips", Vector((0, 0, 1.0)), None)])
        m2 = _make_armature("M2", [("Hips", Vector((0, 0, 1.0)), None)])

        # 0 others selected -> loud CANCELLED (RuntimeError), no mutation.
        for o in bpy.data.objects:
            o.select_set(False)
        base.select_set(True)
        bpy.context.view_layer.objects.active = base
        ctx = {'active_object': base, 'object': base, 'selected_objects': [base]}
        _call_op_expect_loud_cancel(bpy.ops.avatarprep.merge_armatures, ctx,
                                    "found 0 other selected armature")
        if len(_scene_armatures()) != 3:
            _fail("op_guard: 0-others CANCEL must not mutate, got %d armatures"
                  % len(_scene_armatures()))

        # 2 others selected -> loud CANCELLED (RuntimeError), no mutation.
        for o in (base, m1, m2):
            o.select_set(True)
        bpy.context.view_layer.objects.active = base
        ctx = {'active_object': base, 'object': base,
               'selected_objects': [base, m1, m2]}
        _call_op_expect_loud_cancel(bpy.ops.avatarprep.merge_armatures, ctx,
                                    "found 2 other selected armature")
        if len(_scene_armatures()) != 3:
            _fail("op_guard: 2-others CANCEL must not mutate, got %d armatures"
                  % len(_scene_armatures()))
    finally:
        avatarprep.unregister()


def test_postcheck_fail_named_offenders():
    # A postcheck FAIL must name offenders (fail-loud), same as a pre-mutation
    # FAIL. Two rigs each with a mesh named 'Body': Blender auto-suffixes the
    # second object to 'Body.001' at creation, so after a (compat-clean) merge
    # both live under the unified armature and the postcheck flags
    # duplicate_objects -> verdict FAIL with a named offender.
    _clear_scene()
    base = _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Spine", Vector((0, 0, 1.2)), "Hips"),
    ])
    _make_mesh("Body", base, "Spine", Vector((0, 0, 1.2)))
    merge = _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Tail", Vector((0, -0.1, 1.0)), "Hips"),
    ])
    _make_mesh("Body", merge, "Tail", Vector((0, -0.1, 1.0)))  # -> 'Body.001'
    from avatarprep.core.merge_armatures import merge_armatures
    res = merge_armatures(base, merge)
    if res.get("verdict") != "FAIL":
        _fail("postcheck_offenders: expected FAIL, got %r (postcheck=%r)"
              % (res.get("verdict"), res.get("postcheck")))
        return
    offenders = res.get("offenders")
    if not offenders:
        _fail("postcheck_offenders: a FAIL must carry non-empty named offenders, "
              "got %r (postcheck=%r)" % (offenders, res.get("postcheck")))
        return
    if not any("duplicate object" in o for o in offenders):
        _fail("postcheck_offenders: expected a 'duplicate object' offender, got %r"
              % offenders)


def test_whatif():
    _clear_scene()
    base = _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Breast_L", Vector((0.1, 0, 1.3)), "Hips"),
    ])
    merge = _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("breast_L", Vector((0.1, 0, 1.3)), "Hips"),  # rename -> gate FAILs
        ("Tail", Vector((0, -0.1, 1.0)), "Hips"),
    ])
    from avatarprep.core.merge_armatures import merge_armatures
    # Incompatible without a rename_map: whatif predicts the same FAIL.
    res = merge_armatures(base, merge, whatif=True)
    if res["verdict"] != "FAIL":
        _fail("whatif: expected predicted FAIL, got %r" % res["verdict"])
    # Rename resolved: whatif predicts PASS; scene keeps BOTH armatures and the
    # merge armature's ORIGINAL bone names (rename rolled back).
    res2 = merge_armatures(base, merge, rename_map={"breast_L": "Breast_L"},
                           whatif=True)
    if res2["verdict"] != "PASS" or res2.get("reason") != "whatif":
        _fail("whatif: expected predicted PASS/whatif, got %r/%r"
              % (res2.get("verdict"), res2.get("reason")))
    if res2.get("bones_added") != 1:
        _fail("whatif: expected bones_added=1 (Tail), got %r"
              % res2.get("bones_added"))
    if len(_scene_armatures()) != 2:
        _fail("whatif: expected scene unmutated (2 armatures), got %d"
              % len(_scene_armatures()))
    if "breast_L" not in [b.name for b in merge.data.bones]:
        _fail("whatif: rename_map not rolled back; merge bones=%r"
              % [b.name for b in merge.data.bones])


def main():
    _enable_avatarprep()
    test_compat_flags_rename()
    test_guard_unmutated()
    test_same_armature_guard()
    test_clean_merge_weights()
    test_rename_map()
    test_rename_map_dup_target()
    test_rename_map_swap()
    test_whatif()
    test_force_parent_mismatch()
    test_vgroup_fold_branch()
    test_scene_scoping()
    test_postcheck_fail_named_offenders()
    test_merge_operator_guard()
    test_stamp_equal_pass()
    test_stamp_base_different_fail()
    test_stamp_state_different_fail()
    test_stamp_interrupted_fail()
    test_stamp_corrupt_fail()
    test_stamp_missing_warns_and_passes()
    test_force_stamps_split()
    if FAILURES:
        for f in FAILURES:
            print("MERGE_TEST FAIL:", f)
        sys.exit(1)
    print("MERGE_TEST OK")


if __name__ == "__main__":
    main()
