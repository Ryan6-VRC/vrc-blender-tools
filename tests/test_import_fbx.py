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

    # An empty mesh must not be MEASURED, though it is still counted. bound_box on a
    # zero-vertex mesh is eight all-zero corners, so including it reports that object's
    # origin as geometry: a cube 10 m up reads as 10.5 m tall instead of 1.0 m.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 10))
    solo = import_mod.observe_import()["height_m"]
    empty = bpy.data.objects.new("Empty", bpy.data.meshes.new("EmptyData"))
    bpy.context.collection.objects.link(empty)
    both = import_mod.observe_import()
    assert both["meshes"] == 2, "the empty mesh should still be counted, got %r" % both
    assert abs(both["height_m"] - solo) < 1e-6, \
        "an empty mesh changed height_m from %r to %r" % (solo, both["height_m"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    only_empty = bpy.data.objects.new("Empty", bpy.data.meshes.new("EmptyData"))
    bpy.context.collection.objects.link(only_empty)
    assert import_mod.observe_import()["height_m"] == 0, \
        "a scene of only empty meshes should report height_m 0"

    # height_m measures the EVALUATED result. bound_box -- what this used to read --
    # reports the control cage: 2.0 for a subsurf-L2 size-2 cube whose real height is
    # 1.679013. measure.py's docstring holds the full set of shapes bound_box gets
    # wrong and test_proportions.py pins them on the shared helper; this is the one
    # assertion that the SNAPSHOT is wired to it, which is what own-base Phase 2 reads.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    bpy.context.object.modifiers.new("s", 'SUBSURF').levels = 2
    h = import_mod.observe_import()["height_m"]
    assert abs(h - 1.6790) < 1e-3, \
        "height_m should measure the evaluated mesh (~1.679), got %r -- 2.0 means it " \
        "is reading bound_box again" % h

    # A mesh the depsgraph will not evaluate is measured as its UNDEFORMED shape, by
    # every measure in this repo. Nothing here can fix that, so it must not read as a
    # clean measurement: the snapshot names the offender instead.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    hidden = bpy.context.object
    hidden.name = "HiddenCube"
    hidden.modifiers.new("s", 'SUBSURF').levels = 2
    snap = import_mod.observe_import()
    assert snap["unevaluated_meshes"] == [], \
        "a visible mesh must not be reported unevaluated, got %r" % snap["unevaluated_meshes"]
    hidden.hide_viewport = True
    snap = import_mod.observe_import()
    assert snap["unevaluated_meshes"] == ["HiddenCube"], \
        "a hide_viewport mesh must be named unevaluated, got %r" % snap["unevaluated_meshes"]
    assert abs(snap["height_m"] - 2.0) < 1e-3, \
        "the honest reading of an unevaluated mesh is its undeformed 2.0, got %r -- if " \
        "this ever reads 1.679 the blind spot closed and the key can go" % snap["height_m"]

    # Which checkout actually ran: an editable install records one absolute path, so a
    # second worktree can import the first one's source and pass regardless of its own
    # changes (dispatched-work.md, Worktree mechanics).
    print("IMPORT_TEST module:", import_mod.__file__)
    print("IMPORT_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "IMPORT_TEST")
