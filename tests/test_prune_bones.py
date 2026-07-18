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


def _add_repo_root_to_path():
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

        Spine  (weighted) ─── Chest      (weighted) ─── Cloth1 (zero) ─── Cloth2 (zero)
               └──────────── Upper      (weighted) ─── Upper_end  (zero-weight leaf)
        Skirt  (zero)     ─── Skirt_end  (zero-weight leaf)
        Hook   (zero)     ← Empty object parented here via BONE parent type
        Scalp  (zero)     ─── Hair1 (zero) ─── Hair2 (zero) ─── Hair3 (zero)
                              (fully zero-weight chain — deleted whole)

    ``Hook`` carries the Empty that trips the refusal gate. ``Cloth1→Cloth2`` is the
    only chain under a WEIGHTED parent, so it is the sole case asserting
    ``parent_weighted=True``; ``Cloth1`` is not spared by rule (b) because it has a
    child, so it is not a depth-1 leaf.
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

    # Zero-weight chain off a WEIGHTED parent — the parent_weighted over-prune case.
    cloth1 = ebs.new("Cloth1")
    cloth1.head = Vector((0.0, -0.2, 0.6))
    cloth1.tail = Vector((0.0, -0.2, 0.4))
    cloth1.parent = chest

    cloth2 = ebs.new("Cloth2")
    cloth2.head = Vector((0.0, -0.2, 0.4))
    cloth2.tail = Vector((0.0, -0.2, 0.2))
    cloth2.parent = cloth1

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
    _add_repo_root_to_path()

    from avatarprep.core.prune_bones import prune_zero_weight_bones, PruneRefused

    arm_obj = _build_armature()
    _build_mesh(arm_obj)
    _attach_empty(arm_obj)

    failures = []

    # ── what-if FIRST, on the intact armature ────────────────────────────────
    bones_before = {b.name for b in arm_obj.data.bones}
    try:
        preview = prune_zero_weight_bones(arm_obj, whatif=True)
    except Exception as e:
        print("PRUNE_TEST FAIL: whatif exception:", e)
        sys.exit(1)

    # The whole point of the preview: it must not touch the armature.
    if {b.name for b in arm_obj.data.bones} != bones_before:
        failures.append("whatif MUTATED the armature (bones changed)")
    if preview.get("whatif") is not True:
        failures.append("expected preview['whatif'] is True, got %r" % preview.get("whatif"))

    # Tripwire: the Empty rides Hook, and Hook is on the chopping block.
    bpo = preview.get("bone_parented_objects") or []
    hook_rows = [o for o in bpo if o.get("bone") == "Hook"]
    if len(hook_rows) != 1:
        failures.append("expected exactly 1 bone-parented object on Hook, got %r" % bpo)
    elif not hook_rows[0].get("bone_pruned"):
        failures.append("expected the Hook tripwire row to report bone_pruned=True, got %r"
                        % hook_rows[0])

    # Chains must partition the removals exactly — no bone in two chains or none.
    chains = preview.get("chains") or []
    chained = [n for ch in chains for n in ch["bones"]]
    if sorted(chained) != sorted(preview.get("deleted_bones") or []):
        failures.append("chains do not partition deleted_bones: %r vs %r"
                        % (sorted(chained), sorted(preview.get("deleted_bones") or [])))
    if len(chained) != len(set(chained)):
        failures.append("a bone appears in more than one chain: %r" % chained)
    chain_roots = {ch["root"] for ch in chains}
    if chain_roots != {"Skirt", "Hook", "Scalp", "Cloth1"}:
        failures.append("expected chain roots {Skirt, Hook, Scalp, Cloth1}, got %r" % sorted(chain_roots))

    # Both polarities — a flag only ever checked False proves nothing.
    by_root = {ch["root"]: ch for ch in chains}
    if by_root.get("Cloth1", {}).get("parent_weighted") is not True:
        failures.append("expected Cloth1 chain parent_weighted=True (parent Chest is weighted), got %r"
                        % by_root.get("Cloth1"))
    if by_root.get("Cloth1", {}).get("parent") != "Chest":
        failures.append("expected Cloth1 chain parent == 'Chest', got %r" % by_root.get("Cloth1"))
    if sorted(by_root.get("Cloth1", {}).get("bones") or []) != ["Cloth1", "Cloth2"]:
        failures.append("expected Cloth1 chain to hold both cloth bones, got %r" % by_root.get("Cloth1"))
    for root in ("Skirt", "Hook", "Scalp"):
        if by_root.get(root, {}).get("parent_weighted") is not False:
            failures.append("expected %s chain parent_weighted=False, got %r" % (root, by_root.get(root)))

    # Upper_end is the only rule-(b) keep.
    tips = {t["bone"] for t in (preview.get("kept_tips") or [])}
    if tips != {"Upper_end"}:
        failures.append("expected kept_tips == {Upper_end}, got %r" % sorted(tips))

    # The preview must carry the GATE verdict, not just the plan: HookAttachment
    # rides the doomed Hook, so a real run refuses.
    if preview.get("would_refuse") is not True:
        failures.append("expected preview['would_refuse'] is True (Hook is doomed and ridden), got %r"
                        % preview.get("would_refuse"))

    # ── the gate: an unforced run must REFUSE and mutate nothing ─────────────
    bones_pre_gate = {b.name for b in arm_obj.data.bones}
    try:
        prune_zero_weight_bones(arm_obj)
    except PruneRefused as refused:
        if {o["object"] for o in refused.offenders} != {"HookAttachment"}:
            failures.append("expected PruneRefused to name HookAttachment, got %r" % refused.offenders)
    except Exception as e:
        failures.append("expected PruneRefused, got %s: %s" % (type(e).__name__, e))
    else:
        failures.append("expected PruneRefused (HookAttachment rides the doomed Hook), but the prune ran")
    # A gate that mutates is a warning.
    if {b.name for b in arm_obj.data.bones} != bones_pre_gate:
        failures.append("a REFUSED prune must mutate nothing, but the armature changed")

    # ── execute under force, and hold it to the plan the preview published ───
    try:
        result = prune_zero_weight_bones(arm_obj, force=True)
    except Exception as e:
        print("PRUNE_TEST FAIL: exception:", e)
        sys.exit(1)
    bones_remaining = {b.name for b in arm_obj.data.bones}

    # Preview fidelity: a preview that can disagree with the run is worthless.
    if sorted(preview.get("deleted_bones") or []) != sorted(result.get("deleted_bones") or []):
        failures.append("preview plan != actual removals: %r vs %r"
                        % (sorted(preview.get("deleted_bones") or []),
                           sorted(result.get("deleted_bones") or [])))
    if (preview.get("kept"), preview.get("deleted")) != (result.get("kept"), result.get("deleted")):
        failures.append("preview counts != actual counts: %r vs %r"
                        % ((preview.get("kept"), preview.get("deleted")),
                           (result.get("kept"), result.get("deleted"))))

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
    expect_absent("Hook")         # holding an attachment is not a keep reason
    expect_absent("Skirt")        # zero-weight, no weighted descendants
    expect_absent("Skirt_end")    # zero-weight leaf, parent not weighted
    expect_absent("Cloth1")       # zero-weight under a weighted parent, but has a child
    expect_absent("Cloth2")       # zero-weight leaf, parent (Cloth1) not weighted
    # Fully zero-weight chain off an unweighted root: deleted whole.
    expect_absent("Scalp")
    expect_absent("Hair1")
    expect_absent("Hair2")
    expect_absent("Hair3")

    # Under force the orphan is deliberate, and still has to be reported.
    exec_bpo = result.get("bone_parented_objects")
    if not isinstance(exec_bpo, list):
        failures.append("execute result must carry bone_parented_objects, got %r" % exec_bpo)
    else:
        exec_hook = [o for o in exec_bpo if o.get("bone") == "Hook"]
        if len(exec_hook) != 1 or not exec_hook[0].get("bone_pruned"):
            failures.append("execute path must report the orphaned Hook attachment, got %r" % exec_bpo)
        if exec_bpo != preview.get("bone_parented_objects"):
            failures.append("execute tripwire != preview tripwire: %r vs %r"
                            % (exec_bpo, preview.get("bone_parented_objects")))

    # Guard the return dict so a miscounted result is caught too.
    if result.get("kept") != 4:
        failures.append("expected result['kept'] == 4, got %r" % result.get("kept"))
    if result.get("deleted") != 9:
        failures.append("expected result['deleted'] == 9, got %r" % result.get("deleted"))

    expected_deleted = {"Skirt", "Skirt_end", "Hook", "Cloth1", "Cloth2",
                        "Scalp", "Hair1", "Hair2", "Hair3"}
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
