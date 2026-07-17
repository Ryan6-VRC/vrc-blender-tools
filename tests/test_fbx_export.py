"""Regression oracle for export_unity_fbx's OBJECT-mode guard (G67).

Run:
  blender --background --factory-startup --python tests/test_fbx_export.py

Prints FBXEXPORT_TEST OK and exits 0 on success; FBXEXPORT_TEST FAIL: <reason>
and exits 1 otherwise. apply_proportion_edge exits in POSE mode on its object-only
edge path, so an apply-then-export in one script left the scene in POSE and crashed
``select_all.poll() failed, context is incorrect``. export_unity_fbx now forces
OBJECT mode itself; this asserts export succeeds from POSE (both the scoped
--armature and whole-scene branches) and does not regress from OBJECT.
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


def _add_repo_root_to_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _make_rig():
    """A one-bone armature with a single bound mesh, active + selected, in OBJECT mode."""
    _clear_scene()
    from avatarprep.core import scene_utils
    arm_data = bpy.data.armatures.new("ArmData")
    arm = bpy.data.objects.new("Armature", arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    ctx = {'active_object': arm, 'object': arm}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    b = arm.data.edit_bones.new("Root")
    b.head = Vector((0, 0, 0)); b.tail = Vector((0, 0, 0.2))
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')

    md = bpy.data.meshes.new("BodyData")
    md.from_pydata([(-0.05, -0.05, 0.0), (0.05, -0.05, 0.0), (0.0, 0.05, 0.2)], [], [(0, 1, 2)])
    md.update()
    ob = bpy.data.objects.new("Body", md)
    bpy.context.collection.objects.link(ob)
    vg = ob.vertex_groups.new(name="Root")
    vg.add([0, 1, 2], 1.0, 'REPLACE')
    mod = ob.modifiers.new("Armature", 'ARMATURE'); mod.object = arm
    ob.parent = arm
    return arm


def _enter_pose(arm):
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')


def _export(arm, scoped, tag):
    """Return (raised_exception_or_None, file_written_bool)."""
    from avatarprep.core import fbx_export
    out = os.path.join(tempfile.mkdtemp(), "%s.fbx" % tag)
    try:
        fbx_export.export_unity_fbx(out, armature_obj=(arm if scoped else None))
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return e, False
    return None, os.path.exists(out)


def main():
    _add_repo_root_to_path()

    # 1. The regression: scoped (--armature) export from POSE mode must not crash.
    arm = _make_rig()
    _enter_pose(arm)
    check(arm.mode == 'POSE', "fixture sanity: rig must be in POSE before the scoped export")
    exc, written = _export(arm, scoped=True, tag="scoped_from_pose")
    check(exc is None, "scoped export from POSE raised: %s" % exc)
    check(written, "scoped export from POSE wrote no file")

    # 2. Whole-scene export (armature_obj=None) from POSE must also be robust.
    arm = _make_rig()
    _enter_pose(arm)
    exc, written = _export(arm, scoped=False, tag="scene_from_pose")
    check(exc is None, "whole-scene export from POSE raised: %s" % exc)
    check(written, "whole-scene export from POSE wrote no file")

    # 3. No regression: a scoped export already in OBJECT mode still works.
    arm = _make_rig()
    check(arm.mode == 'OBJECT', "fixture sanity: rig must start in OBJECT")
    exc, written = _export(arm, scoped=True, tag="scoped_from_object")
    check(exc is None, "scoped export from OBJECT raised: %s" % exc)
    check(written, "scoped export from OBJECT wrote no file")

    if FAILURES:
        print("FBXEXPORT_TEST FAIL:", "; ".join(FAILURES))
        sys.exit(1)
    print("FBXEXPORT_TEST OK")
    sys.exit(0)


main()
