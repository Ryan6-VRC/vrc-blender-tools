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
armature node's ``Lcl Rotation`` must be pure axis conversion (-90, 0, 0), what
Blender's exporter writes into the Y-up file it declares. The Felis fixture
encodes the same world layout differently — an identity node in a self-declared
Z-up file — so its node rotation is not the value ours must match.

Scope limit: this oracle sees only the file. The same 180 deg also has a switch
in the consumer, Unity's per-asset ``bakeAxisConversion``, so a green run here
does not mean the avatar faces forward in a project. ``fbx_export`` documents
the polarity; checking it takes a Unity import at a named setting, not a parse.

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


def _model_node_rotation(path, model_name):
    """A Model node's Lcl Rotation (degrees), (0,0,0) if the property is absent
    (identity), None if the node is missing."""
    from io_scene_fbx import parse_fbx
    root, _ = parse_fbx.parse(path)
    objects = next(e for e in root.elems if e.id == b"Objects")
    for e in objects.elems:
        if e.id != b"Model":
            continue
        name = e.props[1].decode("utf-8", "replace").split("\x00")[0]
        if name != model_name:
            continue
        for p70 in (c for c in e.elems if c.id == b"Properties70"):
            for p in p70.elems:
                if p.props[0] == b"Lcl Rotation":
                    return tuple(float(v) for v in p.props[4:7])
        return (0.0, 0.0, 0.0)
    return None


def _armature_node_rotation(path):
    return _model_node_rotation(path, "Armature")


def _vertex_y_extremes(path):
    """(ymin, ymax) over every Geometry's raw vertex array. Mesh vertices are
    written in raw Blender/source coordinates (the axis conversion lives in the
    node rotations), so a baked 180° Z flip negates the y extremes — the
    geometry-level oracle the node rotation alone cannot provide."""
    from io_scene_fbx import parse_fbx
    root, _ = parse_fbx.parse(path)
    objects = next(e for e in root.elems if e.id == b"Objects")
    ymin, ymax = 1e9, -1e9
    for e in objects.elems:
        if e.id != b"Geometry":
            continue
        for c in e.elems:
            if c.id == b"Vertices":
                ys = c.props[0][1::3]
                ymin = min(ymin, min(ys))
                ymax = max(ymax, max(ys))
    return ymin, ymax


def _close(a, b, eps=0.1):
    return abs(a - b) <= eps


def _clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _parked_rig(arm_name, bones, mesh_name=None, mesh_y=0.0, park_z=math.pi):
    """Armature (object rotation parked at ``park_z`` about Z) + optionally a
    child mesh whose quad sits at local y=``mesh_y``, bound by modifier."""
    from mathutils import Vector
    arm_data = bpy.data.armatures.new(arm_name + "Data")
    arm = bpy.data.objects.new(arm_name, arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = arm_data.edit_bones
    for bname, head, parent in bones:
        b = ebs.new(bname)
        b.head = Vector(head)
        b.tail = Vector(head) + Vector((0, 0, 0.1))
    for bname, head, parent in bones:
        if parent:
            ebs[bname].parent = ebs[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.rotation_euler = (0.0, 0.0, park_z)
    if mesh_name:
        md = bpy.data.meshes.new(mesh_name + "Data")
        md.from_pydata([(-0.05, mesh_y, 0.9), (0.05, mesh_y, 0.9),
                        (0.0, mesh_y, 1.1)], [], [(0, 1, 2)])
        md.update()
        mo = bpy.data.objects.new(mesh_name, md)
        bpy.context.collection.objects.link(mo)
        mo.vertex_groups.new(name=bones[-1][0]).add([0, 1, 2], 1.0, 'REPLACE')
        mod = mo.modifiers.new("Armature", 'ARMATURE')
        mod.object = arm
        mo.parent = arm
    return arm


def test_merge_path_geometry():
    import tempfile
    from avatarprep.core import fbx_export
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1)
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 0, 1.0), None), ("Tail", (0, -0.2, 1.0), "Hips")],
                        mesh_name="TailM", mesh_y=-0.2)
    res = merge_armatures(base, merge)
    check(res["verdict"] == "PASS", "merge-path fixture: merge FAILed: %r"
          % res.get("offenders"))
    out = os.path.join(tempfile.mkdtemp(), "merged.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=base)
    rot = _armature_node_rotation(out)
    check(rot is not None and _close(rot[0], -90.0) and _close(rot[1], 0.0)
          and _close(rot[2], 0.0),
          "merged export node rotation: %s (want ~(-90,0,0))" % (rot,))
    ymin, ymax = _vertex_y_extremes(out)
    # Authored local frame: TailM at y=-0.2, BodyM at y=+0.1. A baked park flip
    # negates both — the exact defect the node rotation cannot show.
    check(abs(ymin - (-0.2)) < 1e-4 and abs(ymax - 0.1) < 1e-4,
          "merge baked the object rotation into geometry: y extremes "
          "%.4f..%.4f (want -0.2..0.1)" % (ymin, ymax))


def test_unparented_bound_mesh():
    import tempfile
    from mathutils import Vector
    from avatarprep.core import fbx_export
    _clear_scene()
    arm = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)])
    md = bpy.data.meshes.new("LooseData")
    md.from_pydata([(-0.05, -0.2, 0.9), (0.05, -0.2, 0.9), (0.0, -0.2, 1.1)],
                   [], [(0, 1, 2)])
    md.update()
    mo = bpy.data.objects.new("Loose", md)
    bpy.context.collection.objects.link(mo)
    mo.vertex_groups.new(name="Hips").add([0, 1, 2], 1.0, 'REPLACE')
    mod = mo.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    mo.rotation_euler = (0.0, 0.0, math.pi)  # carries the same park, unparented
    out = os.path.join(tempfile.mkdtemp(), "loose.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=arm)
    arot = _model_node_rotation(out, "Armature")
    mrot = _model_node_rotation(out, "Loose")
    check(mrot is not None, "unparented bound mesh missing from the export")
    if arot and mrot:
        check(all(_close(a, m) for a, m in zip(arot, mrot)),
              "bound mesh disagrees with the skeleton in the file: mesh %s vs "
              "armature %s" % (mrot, arot))
    check(_close(abs(math.degrees(mo.rotation_euler[2])), 180.0),  # ±180 are the same rotation
          "export did not restore the bound mesh's transform")


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

    # 1b. Geometry oracle: exported vertex data must sit in the VENDOR frame,
    # not a park-rotated one. The vendor file's own vertex extremes are the
    # truth; a baked 180° flip negates them.
    v_ymin, v_ymax = _vertex_y_extremes(felis)
    o_ymin, o_ymax = _vertex_y_extremes(out)
    check(abs(v_ymin - v_ymax) > 5e-3 and abs(abs(v_ymin) - abs(v_ymax)) > 5e-3,
          "fixture geometry no longer y-asymmetric (%.4f..%.4f) — this oracle "
          "cannot discriminate a flip; re-derive it" % (v_ymin, v_ymax))
    check(abs(o_ymin - v_ymin) < 2e-3 and abs(o_ymax - v_ymax) < 2e-3,
          "exported geometry not in the vendor frame: y extremes %.4f..%.4f "
          "(vendor %.4f..%.4f)" % (o_ymin, o_ymax, v_ymin, v_ymax))

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

    # 3. The merge path: merge_armatures(apply_transforms=True) must not bake
    # the park into data — an import→merge→export flow has an IDENTITY object
    # rotation at export time, so only geometry can reveal a baked flip.
    test_merge_path_geometry()

    # 4. A modifier-bound UNPARENTED mesh (a bound shape get_bound_meshes
    # supports) must be carried by the clear — else the file ships its geometry
    # 180° off the skeleton.
    test_unparented_bound_mesh()

    if FAILURES:
        print("FBXORIENT_TEST FAIL:", "; ".join(FAILURES))
        sys.exit(1)
    print("FBXORIENT_TEST OK")
    sys.exit(0)


main()
