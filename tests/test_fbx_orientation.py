"""File-level regression oracle for export_unity_fbx's orientation contract.

Run:
  blender --background --factory-startup --python tests/test_fbx_orientation.py

The defect this pins: ``wm.fbx_import`` represents a source FBX's axis
convention as an object rotation on the armature (Felis parks -180 deg Z); the
exporter re-derives its own conversion, so carrying that rotation through
double-counts it and the exported file gains an extra 180 deg (avatar faces
backwards in Unity). A Blender re-import comparison CANNOT detect this — the
importer symmetrically undoes the exporter (measured: identity to ~1e-7 even on
the defective export) — so the oracle parses the written FBX instead: the
armature node's ``Lcl Rotation`` must be pure axis conversion (-90, 0, 0), the
orientation vendor meter-unit files ship (identity node in a Z-up file).

Prints FBXORIENT_TEST OK / FBXORIENT_TEST FAIL: <reason>.
"""
import math
import os
import sys
import tempfile

import bpy

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def _add_repo_root_to_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _armature_node_rotation(path):
    """The exported Armature Null node's Lcl Rotation (degrees), (0,0,0) if the
    property is absent (identity)."""
    from io_scene_fbx import parse_fbx
    root, _ = parse_fbx.parse(path)
    objects = next(e for e in root.elems if e.id == b"Objects")
    for e in objects.elems:
        if e.id != b"Model":
            continue
        name = e.props[1].decode("utf-8", "replace").split("\x00")[0]
        if name != "Armature":
            continue
        for p70 in (c for c in e.elems if c.id == b"Properties70"):
            for p in p70.elems:
                if p.props[0] == b"Lcl Rotation":
                    return tuple(float(v) for v in p.props[4:7])
        return (0.0, 0.0, 0.0)
    return None


def _close(a, b, eps=0.1):
    return abs(a - b) <= eps


def main():
    _add_repo_root_to_path()
    from avatarprep.core import import_fbx, fbx_export

    felis = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fixtures", "Felis", "Felis.fbx")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    import_fbx.import_fbx(felis)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')

    # Witness precondition: the fixture's import parks a nonzero Z rotation on
    # the armature object — without it this test exercises nothing.
    park_z = math.degrees(arm.rotation_euler[2])
    check(abs(park_z) > 90.0,
          "fixture no longer parks an object rotation (z=%.2f deg) — "
          "the defect scenario is gone, re-derive this test" % park_z)

    tmp = tempfile.mkdtemp()

    # 1. Default export: node rotation must be pure axis conversion (-90, 0, 0).
    out = os.path.join(tmp, "felis_default.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=arm)
    rot = _armature_node_rotation(out)
    check(rot is not None, "default export: no Armature node found")
    if rot:
        check(_close(rot[0], -90.0) and _close(rot[1], 0.0) and _close(rot[2], 0.0),
              "default export carries extra rotation: node rot=%s (want ~(-90,0,0))"
              % (rot,))

    # ...and the scene is left as found (rotation restored after export).
    check(_close(math.degrees(arm.rotation_euler[2]), park_z),
          "export did not restore the armature object rotation")

    # 2. keep_object_rotation=True preserves the park (the defect stays
    # representable, so the oracle can tell the two apart).
    out_keep = os.path.join(tmp, "felis_keep.fbx")
    fbx_export.export_unity_fbx(out_keep, armature_obj=arm,
                                keep_object_rotation=True)
    rot_keep = _armature_node_rotation(out_keep)
    check(rot_keep is not None, "keep export: no Armature node found")
    if rot_keep:
        check(not (_close(rot_keep[1], 0.0) and _close(abs(rot_keep[0]), 90.0)
                   and _close(rot_keep[2], 0.0)),
              "keep_object_rotation=True produced the same node rotation as the "
              "cleared export (%s) — the flag does nothing" % (rot_keep,))

    if FAILURES:
        print("FBXORIENT_TEST FAIL:", "; ".join(FAILURES))
        sys.exit(1)
    print("FBXORIENT_TEST OK")
    sys.exit(0)


main()
