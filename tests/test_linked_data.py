"""Headless test for library data across the doors: linked references, linked
collection instances, and library overrides.

Run:
  blender --background --factory-startup --python tests/test_linked_data.py

Prints LINKED_TEST OK / LINKED_TEST FAIL: <reason>.

The link target is a temp ``.blend`` this test writes first (a rig, a bound mesh
with a baked stamp, and a collection holding both), so no fixture ships. Every
case pins one measured defect or one branch of the three predicates in
``scene_utils`` (``is_linked`` / ``instances_linked`` / ``is_editable``):

  * the default armature pick skipped a linked rig (it was picked first, and a
    real prune then crashed with ``Cannot edit library linked``);
  * the whole-scene export refuses a linked object and a linked collection
    instance by name, above the multi-armature refusal;
  * a scoped export ignores a linked reference and refuses a linked scope;
  * the scale bake refuses shared or library data ONLY when a bake reaches it —
    the narrowing that lets a unit-scale override head export, pinned both ways;
  * prune reports ``would_refuse`` + ``refusal`` on a non-editable rig and raises
    ``PruneTargetNotEditable`` (force does not bypass) on a real run;
  * ``report_stamps`` carries ``library`` / ``data_library`` on every entry.
"""
import os
import sys
import tempfile

import bpy
from mathutils import Vector

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def _repo_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _make_arm(name, bone="Hips"):
    from avatarprep.core import scene_utils
    ad = bpy.data.armatures.new(name + "Data")
    ao = bpy.data.objects.new(name, ad)
    bpy.context.scene.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao
    ao.select_set(True)
    ctx = {'active_object': ao, 'object': ao}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    b = ao.data.edit_bones.new(bone)
    b.head = Vector((0, 0, 0)); b.tail = Vector((0, 0, 0.2))
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')
    return ao


def _make_mesh(name, arm, bone="Hips", data=None):
    md = data or bpy.data.meshes.new(name + "Data")
    if data is None:
        md.from_pydata([(-0.05, -0.05, 0.0), (0.05, -0.05, 0.0), (0.0, 0.05, 0.2)],
                       [], [(0, 1, 2)])
        md.update()
    mo = bpy.data.objects.new(name, md)
    bpy.context.scene.collection.objects.link(mo)
    vg = mo.vertex_groups.new(name=bone)
    vg.add([0, 1, 2], 1.0, 'REPLACE')
    mod = mo.modifiers.new("Armature", 'ARMATURE'); mod.object = arm
    mo.parent = arm
    return mo


def _write_library(tmp):
    """The link target: rig ``RefArm`` + baked mesh ``RefBody`` inside collection
    ``RefColl``. Returns its path."""
    from avatarprep.core import scene_utils
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm = _make_arm("RefArm")
    body = _make_mesh("RefBody", arm)
    body[scene_utils.STAMP_BAKED] = {"Chest": 0.6}
    scene_utils.write_stamp(arm, scene_utils.STAMP_BASE, "refbase")
    coll = bpy.data.collections.new("RefColl")
    bpy.context.scene.collection.children.link(coll)
    for o in (arm, body):
        coll.objects.link(o)
        bpy.context.scene.collection.objects.unlink(o)
    path = os.path.join(tmp, "lib.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return path


def _fresh_with_local(name="ZOwn", link_first=None):
    """Empty scene with one LOCAL rig + bound mesh. ``link_first=(lib, names)``
    links those objects BEFORE the local rig is built, so ``scene.objects`` (creation
    order) puts the linked rig first — where a first-in-scene pick would land."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    linked = _link_objects(*link_first) if link_first else []
    arm = _make_arm("Armature." + name, bone="Root")
    mesh = _make_mesh(name, arm, bone="Root")
    return (arm, mesh) + tuple(linked)


def _link_objects(lib, names):
    with bpy.data.libraries.load(lib, link=True, relative=False) as (df, dt):
        dt.objects = list(names)
    for o in dt.objects:
        bpy.context.scene.collection.objects.link(o)
    return dt.objects


def _link_collection_instance(lib, name="RefColl"):
    with bpy.data.libraries.load(lib, link=True, relative=False) as (df, dt):
        dt.collections = [name]
    coll = dt.collections[0]
    empty = bpy.data.objects.new("RefInstance", None)
    empty.instance_type = 'COLLECTION'
    empty.instance_collection = coll
    bpy.context.scene.collection.objects.link(empty)
    return empty


def _export(arm, tmp, tag, **kw):
    from avatarprep.core import fbx_export
    out = os.path.join(tmp, "%s.fbx" % tag)
    try:
        fbx_export.export_unity_fbx(out, armature_obj=arm, **kw)
    except Exception as e:  # noqa: BLE001
        return e, False
    return None, os.path.exists(out)


def main():
    _repo_root()
    from avatarprep.core import scene_utils as S
    from avatarprep.core.prune_bones import (prune_zero_weight_bones,
                                             PruneTargetNotEditable)
    print("LINKED_TEST scene_utils from", S.__file__)
    tmp = tempfile.mkdtemp(prefix="avatarprep_linked_")
    lib = _write_library(tmp)

    # ── 1. Linked reference beside a local rig ───────────────────────────────
    own, own_mesh, ref_arm, ref_body = _fresh_with_local(link_first=(lib, ["RefArm", "RefBody"]))
    names = [o.name for o in bpy.context.scene.objects]
    check(names.index("RefArm") < names.index("Armature.ZOwn"),
          "fixture: linked rig must sort before the local one, got %r" % names)

    check(S.is_linked(ref_arm) and S.is_linked(ref_body), "linked objects read is_linked")
    check(not S.is_linked(own), "local object is not is_linked")
    check(not S.is_editable(ref_arm) and S.is_editable(own),
          "is_editable: linked False, local True")
    check(S.library_path(ref_arm) and S.library_path(ref_arm).endswith("lib.blend"),
          "library_path names the library, got %r" % S.library_path(ref_arm))
    check(S.library_path(own) is None, "library_path of local data is None")

    # default pick skips the linked rig even though it comes first
    bpy.context.view_layer.objects.active = None
    check(S.find_armature() is own, "find_armature default pick must skip the linked rig")
    check(S.find_armature("RefArm") is ref_arm, "explicit name still resolves a linked rig")
    picked, err = S.resolve_target_armature(bpy.context.scene, None)
    check(picked is own and err is None, "resolve_target_armature picks the sole LOCAL rig")
    check(S.linked_armature_count() == 1, "linked_armature_count reads 1")

    # report_stamps marks the reference
    rep = S.report_stamps(bpy.context.scene)
    ra = next((a for a in rep["armatures"] if a["name"] == "RefArm"), None)
    la = next((a for a in rep["armatures"] if a["name"] == "Armature.ZOwn"), None)
    check(ra is not None and ra["library"] and ra["data_library"] and ra["base"] == "refbase",
          "linked armature entry carries library + data_library + its stamps, got %r" % ra)
    check(la is not None and la["library"] is None and la["data_library"] is None,
          "local armature entry reads library=None, data_library=None, got %r" % la)
    rb = next((m for m in ra["meshes"] if m["name"] == "RefBody"), None) if ra else None
    check(rb is not None and rb["library"] and rb["baked"] == {"Chest": 0.6},
          "linked baked mesh grouped under its linked rig with library set, got %r" % rb)

    # prune: whatif reports, real run refuses, force does not bypass
    bones_before = {b.name for b in ref_arm.data.bones}
    pre = prune_zero_weight_bones(ref_arm, whatif=True)
    check(pre["would_refuse"] and pre.get("refusal") and "library data" in pre["refusal"],
          "whatif on a linked rig: would_refuse + refusal naming library data, got %r"
          % {k: pre.get(k) for k in ("would_refuse", "refusal")})
    for force in (False, True):
        try:
            prune_zero_weight_bones(ref_arm, force=force)
            check(False, "prune on a linked rig must raise (force=%s)" % force)
        except PruneTargetNotEditable as e:
            check(e.armature == "RefArm" and e.library, "refusal names rig + library")
        except Exception as e:  # noqa: BLE001
            check(False, "wrong exception on linked prune (force=%s): %r" % (force, e))
    check({b.name for b in ref_arm.data.bones} == bones_before, "linked rig untouched")
    own_pre = prune_zero_weight_bones(own, whatif=True)
    check(own_pre["refusal"] is None and not own_pre["would_refuse"],
          "local rig whatif has no refusal")

    # export: scoped to the local rig ignores the link; whole-scene and linked scope refuse
    exc, written = _export(own, tmp, "scoped_with_link")
    check(exc is None and written, "scoped export with a linked reference present: %r" % exc)
    exc, written = _export(None, tmp, "whole_with_link")
    check(isinstance(exc, ValueError) and "linked object" in str(exc) and "RefArm" in str(exc),
          "whole-scene export must refuse naming the linked objects, got %r" % exc)
    check("delete the extra armature" not in str(exc),
          "the linked refusal must fire above the multi-armature one")
    exc, written = _export(ref_arm, tmp, "scoped_to_link")
    check(isinstance(exc, ValueError) and "linked reference" in str(exc),
          "scoping to a linked rig must refuse, got %r" % exc)

    # ── 2. Linked collection instance ────────────────────────────────────────
    own, _ = _fresh_with_local()
    inst = _link_collection_instance(lib)
    check(not S.is_linked(inst) and S.instances_linked(inst),
          "a collection instance is local but instances_linked")
    check(S.find_armature() is own and S.linked_armature_count() == 0,
          "an instance holds no scene armature; the local rig is the pick")
    exc, written = _export(None, tmp, "whole_with_instance")
    check(isinstance(exc, ValueError) and "RefInstance" in str(exc) and "UNRIGGED" in str(exc),
          "whole-scene export must refuse a linked collection instance by name, got %r" % exc)
    exc, written = _export(own, tmp, "scoped_with_instance")
    check(exc is None and written, "scoped export beside an instance still works: %r" % exc)

    # ── 3. Override head over linked data (the body-swap shape) ──────────────
    own, _ = _fresh_with_local()
    (ref_body,) = _link_objects(lib, ["RefBody"])
    ov = ref_body.override_create(remap_local_usages=True)
    ov.parent = own
    ov.matrix_parent_inverse.identity()
    ov.vertex_groups[0].name = "Root"
    for m in ov.modifiers:
        if m.type == 'ARMATURE':
            m.object = own
    check(ov.library is None and ov.override_library is not None and ov.data.library is not None,
          "override: local object over linked data")
    check(not S.is_linked(ov) and not S.is_editable(ov), "override is not linked, not editable")
    check(ov in S.get_bound_meshes(own), "override binds to the local rig")
    rep = S.report_stamps(bpy.context.scene)
    ent = next((m for a in rep["armatures"] for m in a["meshes"] if m["name"] == ov.name), None)
    check(ent is not None and ent["library"] is None and ent["data_library"],
          "override entry: library=None, data_library set, got %r" % ent)
    exc, written = _export(own, tmp, "override_unit")
    check(exc is None and written,
          "unit-scale override head must export (the narrowed refusal): %r" % exc)
    own.scale = (2.0, 2.0, 2.0)
    exc, written = _export(own, tmp, "override_reached")
    check(isinstance(exc, ValueError) and "library data" in str(exc),
          "a bake that reaches the override must refuse naming library data, got %r" % exc)

    # ── 4. Override ARMATURE: prune gates on is_editable, not is_linked ──────
    own, _ = _fresh_with_local()
    (ref_arm,) = _link_objects(lib, ["RefArm"])
    ov_arm = ref_arm.override_create(remap_local_usages=True)
    check(ov_arm.library is None and not S.is_editable(ov_arm),
          "override armature: local object, not editable")
    pre = prune_zero_weight_bones(ov_arm, whatif=True)
    check(pre["would_refuse"] and pre.get("refusal"), "override armature: whatif would_refuse")
    try:
        prune_zero_weight_bones(ov_arm)
        check(False, "prune on an override armature must raise")
    except PruneTargetNotEditable:
        pass
    except Exception as e:  # noqa: BLE001
        check(False, "wrong exception on override-armature prune: %r" % e)

    # ── 5. Shared local data: the narrowed refusal, pinned both ways ─────────
    own, own_mesh = _fresh_with_local()
    twin = _make_mesh("ZOwnTwin", own, bone="Root", data=own_mesh.data)
    check(own_mesh.data.users == 2, "fixture: two local objects share one mesh")
    exc, written = _export(own, tmp, "shared_unit")
    check(exc is None and written, "shared data no bake reaches must export: %r" % exc)
    own.scale = (2.0, 2.0, 2.0)
    exc, written = _export(own, tmp, "shared_reached")
    check(isinstance(exc, ValueError) and "shares its object data" in str(exc),
          "shared data a bake reaches must still refuse, got %r" % exc)

    if FAILURES:
        for f in FAILURES:
            print("LINKED_TEST FAIL:", f)
        sys.exit(1)
    print("LINKED_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "LINKED_TEST")
