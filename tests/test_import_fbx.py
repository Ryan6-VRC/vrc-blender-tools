"""Headless: round-trip a cube through FBX to prove core.import_fbx imports meshes,
plus two invariants the repo leans on: the snapshot reports the source file's
``unit_scale_factor``, and dotted bone names (``UpperArm.L`` — the dominant
library convention) survive import VERBATIM. The latter is a sentinel: a fitting
run reported the importer sanitizing ``.``->``_`` (which would silently break MA
MergeArmature's match-by-name downstream); that did not reproduce on Blender
5.1.2 against the recorded vendor files, so no restore code ships — this trips
loudly if a future Blender re-introduces sanitization.

Run: blender --background --factory-startup --python tests/test_import_fbx.py
"""
import os
import sys
import tempfile

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avatarprep.core import import_fbx as import_mod


def main():
    tmp = tempfile.mkdtemp()
    fbx = os.path.join(tmp, "cube.fbx")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.export_scene.fbx(filepath=fbx)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    snap = import_mod.import_fbx(fbx)
    assert snap["meshes"] >= 1, "expected >=1 imported mesh, got %r" % snap
    assert snap.get("unit_scale_factor") in (1.0, 100.0), \
        "snapshot unit_scale_factor unreadable: %r" % snap.get("unit_scale_factor")

    # Dotted-bone-name sentinel.
    from avatarprep.core import fbx_export
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm_data = bpy.data.armatures.new("A")
    arm = bpy.data.objects.new("Armature", arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    names = ("UpperArm.L", "UpperArm.R", "Spine")
    for i, n in enumerate(names):
        b = arm_data.edit_bones.new(n)
        b.head = Vector((0.1 * i, 0, 0.5))
        b.tail = Vector((0.1 * i, 0, 0.7))
    bpy.ops.object.mode_set(mode='OBJECT')
    dotted = os.path.join(tmp, "dotted.fbx")
    fbx_export.export_unity_fbx(dotted, armature_obj=arm)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_mod.import_fbx(dotted)
    arm2 = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    got = sorted(b.name for b in arm2.data.bones)
    assert got == sorted(names), \
        "importer no longer preserves bone names verbatim: %r" % got

    print("IMPORT_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "IMPORT_TEST")
