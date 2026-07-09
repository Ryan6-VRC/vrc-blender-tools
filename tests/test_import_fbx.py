"""Headless: round-trip a cube through FBX to prove core.import_fbx imports meshes.

Run: blender --background --factory-startup --python tests/test_import_fbx.py
"""
import os
import sys
import tempfile

import bpy

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
    print("IMPORT_TEST OK")


if __name__ == "__main__":
    main()
