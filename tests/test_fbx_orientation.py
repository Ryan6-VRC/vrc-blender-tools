"""File-level regression oracle for export_unity_fbx's orientation contract.

Run:
  blender --background --factory-startup --python tests/test_fbx_orientation.py

``wm.fbx_import`` represents a source FBX's axis convention as an object rotation
on the armature, and the exporter re-derives its own -90 X presuming Z-up data.
Two defects live in that composition and they pull in OPPOSITE directions — this
file pins both, because a fix for either alone re-breaks the other.

**Carrying an up-axis-PRESERVING rotation through** (Felis parks -180 deg Z)
double-counts it: the file gains an extra 180 deg and the avatar faces backwards
in Unity. A Blender re-import CANNOT detect this — the importer symmetrically
undoes the exporter (measured: identity to ~1e-7 even on the defective export) —
so the oracle parses the written FBX: the armature node's ``Lcl Rotation`` must be
pure axis conversion (-90, 0, 0). The Felis fixture encodes the same world layout
differently, as an identity node in a self-declared Z-up file, so its own node
rotation is not the value ours must match.

**Clearing an up-axis-MOVING rotation** (a Y-up source with an identity root node
parks 90 deg X) double-counts the up-axis conversion instead, and the rig exports
tipped onto its face — measured on Chocolat as re-import height 1.1992 -> 0.4574 m.
A re-import DOES see this one, so its oracle is a world-bbox round-trip. That
class is preserved rather than cleared, so it writes an IDENTITY node rotation:
(-90, 0, 0) is the up-axis-preserving case's contract, not a universal one.

The gate deciding between them is ``scene_utils.clear_axis_convention_rotation``;
``fbx_export``'s orientation docstring is the canon.

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


def _world_bbox():
    """(lo, hi) per axis over every mesh's world-space bound box."""
    from mathutils import Vector
    lo = [1e9] * 3
    hi = [-1e9] * 3
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for a in range(3):
                lo[a] = min(lo[a], w[a])
                hi[a] = max(hi[a], w[a])
    return lo, hi


def _write_yup_source(path, cm_unit):
    """Write a Maya/Max-style source FBX: Y-up header, IDENTITY root node
    rotation, raw Y-up vertex data. Built by parking the rig at Rx(+90) so the
    exporter's own -90 X cancels to an identity node.

    ``cm_unit`` picks the half of the class: the survey found it split 35 cm-unit
    to 18 meter-unit, and only the cm half also parks a 0.01 object scale — so
    both halves are pinned, and neither is described as the class.
    ``export_unity_fbx`` refuses a non-1 ``scale_length``, so the fixture is
    written with the raw operator and the scene restored after."""
    from mathutils import Vector
    _clear_scene()
    scene = bpy.context.scene
    arm_data = bpy.data.armatures.new("ArmatureData")
    arm = bpy.data.objects.new("Armature", arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = arm_data.edit_bones
    # Y-up content: "up" is +Y. Asymmetric on Y and Z so a tip is detectable.
    u = 100.0 if cm_unit else 1.0
    hips = ebs.new("Hips")
    hips.head = Vector((0.0, 0.6 * u, 0.0))
    hips.tail = Vector((0.0, 0.8 * u, 0.0))
    head = ebs.new("Head")
    head.head = Vector((0.0, 1.0 * u, 0.0))
    head.tail = Vector((0.0, 1.2 * u, 0.0))
    head.parent = hips
    bpy.ops.object.mode_set(mode='OBJECT')
    md = bpy.data.meshes.new("BodyData")
    md.from_pydata([(-0.2 * u, 0.0, -0.1 * u), (0.2 * u, 0.0, -0.1 * u),
                    (0.0, 1.2 * u, 0.3 * u)], [], [(0, 1, 2)])
    md.update()
    mo = bpy.data.objects.new("Body", md)
    bpy.context.collection.objects.link(mo)
    mo.vertex_groups.new(name="Hips").add([0, 1, 2], 1.0, 'REPLACE')
    mo.modifiers.new("Armature", 'ARMATURE').object = arm
    mo.parent = arm
    arm.rotation_euler = (math.radians(90), 0.0, 0.0)
    old_len = scene.unit_settings.scale_length
    if cm_unit:
        scene.unit_settings.scale_length = 0.01
    bpy.context.view_layer.update()
    try:
        bpy.ops.export_scene.fbx(
            filepath=path, object_types={'EMPTY', 'ARMATURE', 'MESH', 'OTHER'},
            use_mesh_modifiers=False, add_leaf_bones=False, bake_anim=False,
            apply_scale_options='FBX_SCALE_ALL', path_mode='STRIP',
            embed_textures=False)
    finally:
        scene.unit_settings.scale_length = old_len
    return path


def test_yup_source_preserved(cm_unit, scoped):
    """A Y-up source's 90 deg X residue must be PRESERVED, both export doors and
    both unit halves. Clearing it exports the rig tipped onto its face."""
    import io
    import contextlib
    import tempfile
    from avatarprep.core import import_fbx, fbx_export
    label = "%s/%s" % ("cm" if cm_unit else "m",
                       "armature" if scoped else "whole-scene")
    tmp = tempfile.mkdtemp()
    src = _write_yup_source(os.path.join(tmp, "yup_src.fbx"), cm_unit)

    _clear_scene()
    import_fbx.import_fbx(src)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    bpy.context.view_layer.update()

    # Witness precondition: without the 90 deg X park this test exercises nothing.
    park_x = math.degrees(arm.rotation_euler[0])
    check(abs(park_x - 90.0) < 0.5,
          "%s: fixture no longer parks 90 deg X (x=%.2f) — the defect scenario "
          "is gone, re-derive this test" % (label, park_x))
    src_lo, src_hi = _world_bbox()
    src_height = src_hi[2] - src_lo[2]
    # Both halves import to the SAME world height: the importer normalizes a
    # cm-unit file into a 0.01 object scale over 100x data, so world space is
    # unit-agnostic. World height therefore cannot distinguish the halves — the
    # object scale below is what pins which one this is.
    check(src_height > 0.5,
          "%s: source imports at height %.4f, expected upright" % (label, src_height))

    # Pin that the cm half really IS cm — otherwise both halves are the same
    # fixture wearing different labels and only the meter class is covered. The
    # 0.01 object scale is the half of the class the survey found paired with a
    # UnitScaleFactor=1 file; the identity class parks 1.0.
    want_scale = 0.01 if cm_unit else 1.0
    got_scale = arm.matrix_world.to_scale()[0]
    check(abs(got_scale - want_scale) < want_scale * 0.05,
          "%s: imported object scale %.5f, expected ~%.2f — the fixture is not "
          "the unit class it claims" % (label, got_scale, want_scale))

    out = os.path.join(tmp, "out.fbx")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fbx_export.export_unity_fbx(out, armature_obj=arm if scoped else None)
    emitted = buf.getvalue()

    # The diagnostic IS the oracle for the silent-preserve failure: the bbox
    # assertions below stay green with the message deleted, because a preserved
    # residue returns an identity delta exactly like a rig that never had one.
    check("preserved object rotation" in emitted,
          "%s: export preserved the residue but said nothing (emitted: %r)"
          % (label, emitted.strip()[:200]))

    rot = _armature_node_rotation(out)
    check(rot is not None and all(_close(v, 0.0) for v in rot),
          "%s: preserved export node rotation %s (want ~(0,0,0) — the parked "
          "+90 X must cancel the exporter's -90 X)" % (label, rot))

    _clear_scene()
    import_fbx.import_fbx(out)
    bpy.context.view_layer.update()
    ri_lo, ri_hi = _world_bbox()
    for a, axis in enumerate("xyz"):
        check(abs(ri_lo[a] - src_lo[a]) < 1e-3 * max(1.0, abs(src_lo[a]))
              and abs(ri_hi[a] - src_hi[a]) < 1e-3 * max(1.0, abs(src_hi[a])),
              "%s: round-trip moved the %s bounds: %.4f..%.4f (source "
              "%.4f..%.4f) — the rig exported tipped"
              % (label, axis, ri_lo[a], ri_hi[a], src_lo[a], src_hi[a]))


def test_merge_path_yup_geometry():
    """The merge path must BAKE an up-axis-moving rotation, not clear it. Clearing
    first leaves raw Y-up data under an identity object rotation — the rig lying
    down in the .blend, where the export's residue gate can no longer see it."""
    import tempfile
    from avatarprep.core import fbx_export
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    # Y-up CONTENT, like the real class: bones authored along +Y, then parked at
    # Rx(+90) so the rig stands up in Blender. (_parked_rig authors Z-up content,
    # so parking it 90 X would lay it DOWN — the mirror image of this scenario,
    # where a bbox assertion can pass on mesh spread rather than on uprightness.)
    base = _parked_rig("Armature", [("Hips", (0, 1.0, 0), None)],
                       mesh_name="BodyM", mesh_y=0.0, park_z=0.0)
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 1.0, 0), None), ("Tail", (0, 1.0, -0.2), "Hips")],
                        mesh_name="TailM", mesh_y=0.0, park_z=0.0)
    for a in (base, merge):
        a.rotation_euler = (math.radians(90), 0.0, 0.0)
    bpy.context.view_layer.update()
    res = merge_armatures(base, merge)
    check(res["verdict"] == "PASS", "yup merge FAILed: %r" % res.get("offenders"))
    bpy.context.view_layer.update()
    # Assert on the BONE, not a bbox: after a correct bake the Y-up content has
    # become Z-up data, so Hips' world head sits up the world Z axis. A wrongly
    # cleared rotation leaves the raw Y-up data in place, putting it along +Y.
    hips = base.data.bones["Hips"]
    head = base.matrix_world @ hips.head_local
    check(abs(head[2] - 1.0) < 1e-3 and abs(head[1]) < 1e-3,
          "merge baked the wrong frame: Hips world head %s (want ~(0,0,1) — the "
          "up-axis-moving rotation must be BAKED, not cleared)"
          % (tuple(round(v, 4) for v in head),))
    out = os.path.join(tempfile.mkdtemp(), "merged_yup.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=base)
    rot = _armature_node_rotation(out)
    check(rot is not None and _close(rot[0], -90.0) and _close(rot[2], 0.0),
          "merged yup export node rotation: %s (want ~(-90,0,0) — after the bake "
          "the data is Z-up and the exporter's own conversion applies)" % (rot,))


def test_merge_uncleared_front_axis_warns():
    """Two up-axis-PRESERVING rigs whose origins differ skip the pre-clear, so the
    apply bakes their front-axis residue permanently — the merged rig ends 180 deg
    off its vendor frame under an identity object rotation, which neither the
    export gate nor a re-import can afterwards detect. The warning is the only
    signal that exists, so it must name the stranded rig rather than reassure."""
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1)          # parks -180 Z
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 0, 1.0), None), ("Tail", (0, -0.2, 1.0), "Hips")],
                        mesh_name="TailM", mesh_y=-0.2)        # parks -180 Z
    merge.location = (0.5, 0.0, 0.0)                           # origins differ
    bpy.context.view_layer.update()
    # force=True because offsetting the origin also moves the bones in world
    # space, which the compat gate correctly refuses. The gate is not what is
    # under test here — reaching the APPLY with differing origins is, and force
    # overrides only the structural category.
    res = merge_armatures(base, merge, force=True)
    warns = " ".join((res.get("report") or {}).get("warnings", []))
    check("will NOT match either source's vendor frame" in warns,
          "uncleared front-axis bake did not warn that the vendor frame is lost "
          "(warnings: %r)" % warns)
    check("origins differ" in warns,
          "warning blamed rotation for an origins-only divergence (warnings: %r)"
          % warns)


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

    # 5. The OTHER direction: an up-axis-moving residue must be PRESERVED, not
    # cleared. Both export doors (the sighting hit whole-scene and --armature
    # alike) and both unit halves of the class (only the cm half also parks a
    # 0.01 object scale, so neither half alone is "the class").
    for cm_unit in (False, True):
        for scoped in (True, False):
            test_yup_source_preserved(cm_unit, scoped)

    # 6. And the merge path, where a wrongly-baked frame is unrecoverable.
    test_merge_path_yup_geometry()

    # 7. The other way into the merge path's else branch: an up-axis-preserving
    # rotation that could not be cleared gets baked, losing the vendor frame
    # silently. Only the warning can say so.
    test_merge_uncleared_front_axis_warns()

    if FAILURES:
        print("FBXORIENT_TEST FAIL:", "; ".join(FAILURES))
        sys.exit(1)
    print("FBXORIENT_TEST OK")
    sys.exit(0)


main()
