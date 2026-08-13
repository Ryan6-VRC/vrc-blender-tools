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


def _world_heads(arm):
    """{bone name: world head} — the frame-sensitive quantity the merge moves."""
    bpy.context.view_layer.update()
    return {b.name: arm.matrix_world @ b.head_local for b in arm.data.bones}


def _clear_delta(arm):
    """The world delta ``clear_axis_convention_rotation`` will apply to ``arm``:
    a rotation by R^-1 about the rig's OWN origin, T(o) @ R^-1 @ T(-o)."""
    from mathutils import Matrix
    bpy.context.view_layer.update()
    wm = arm.matrix_world
    to_origin = Matrix.Translation(wm.translation)
    return to_origin @ wm.to_quaternion().to_matrix().to_4x4().inverted() \
        @ to_origin.inverted()


def _offset_rig(arm, origin):
    """Move ``arm``'s object origin to ``origin`` and compensate its bone and mesh
    data so world layout is unchanged — the shape a source FBX whose ARMATURE node
    carries a translation imports as (measured), and the only way into the
    equal-rotation/differing-origin case with the compat gate still passing."""
    from mathutils import Vector
    bpy.context.view_layer.update()
    rot_inv = arm.matrix_world.to_quaternion().to_matrix().inverted()
    shift = rot_inv @ Vector(origin)
    with_edit = arm.data.edit_bones if arm.mode == 'EDIT' else None
    assert with_edit is None
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    for eb in arm.data.edit_bones:
        eb.head = eb.head - shift
        eb.tail = eb.tail - shift
    bpy.ops.object.mode_set(mode='OBJECT')
    # Only PARENTED meshes need compensating: they ride the armature, so the
    # move has to be cancelled in their data. An unparented modifier-bound mesh
    # does not ride it at all — touching it here would displace it from the
    # skeleton, which is the opposite of what this function promises.
    for m in [o for o in bpy.data.objects
              if o.type == 'MESH' and o.parent == arm]:
        for v in m.data.vertices:
            v.co = v.co - shift
    arm.location = Vector(arm.location) + Vector(origin)
    bpy.context.view_layer.update()


def test_merge_differing_origins_keeps_vendor_frame():
    """Two up-axis-PRESERVING rigs whose origins differ must still bake the VENDOR
    frame. Clearing each rig about its own origin would pull them apart by
    (I - R^-1)(o_base - o_merge), so the merge replays the base's clear delta onto
    the merge rig: one rigid map for both.

    The exported-geometry oracle alone cannot police this — measured, a naive
    per-rig clear writes byte-identical y extremes to the correct fix. The
    discriminator is that EVERY bone on BOTH rigs lands at delta @ its pre-merge
    world head; that is true only when one delta moved the whole assembly."""
    import tempfile
    from avatarprep.core import fbx_export
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1)          # parks -180 Z
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 0, 1.0), None), ("Tail", (0, -0.2, 1.0), "Hips")],
                        mesh_name="TailM", mesh_y=-0.2)        # parks -180 Z
    # An UNPARENTED modifier-bound mesh: without it the whole mesh-carry loop in
    # apply_world_delta can be deleted and every other assertion here still
    # passes (measured) — the merge path's counterpart to test_unparented_bound_mesh.
    tail_mesh = bpy.data.objects["TailM"]
    tail_mesh.parent = None
    tail_mesh.rotation_euler = (0.0, 0.0, math.pi)
    _offset_rig(merge, (0.5, 0.0, 0.0))

    pre_base, pre_merge = _world_heads(base), _world_heads(merge)
    # Preconditions. _parked_rig writes rotation_euler without a depsgraph update,
    # so a fixture built without one silently ends up 1.0 apart in world space and
    # the compat gate FAILs on a fixture bug rather than on the code under test.
    check((base.matrix_world.translation - merge.matrix_world.translation).length > 1e-3,
          "fixture no longer has differing origins — this test exercises nothing")
    for bone, head in pre_base.items():
        # 1e-5, tighter than the 1e-4 the post-merge assertion demands: a fixture
        # drifting into the band between them would fail there instead, blaming
        # the code under test for a fixture defect.
        check((head - pre_merge[bone]).length < 1e-5,
              "fixture bones are not world-aligned (%s: %s vs %s) — the compat "
              "gate would refuse this for a fixture bug, not the defect"
              % (bone, tuple(round(v, 4) for v in head),
                 tuple(round(v, 4) for v in pre_merge[bone])))

    delta = _clear_delta(base)
    merge_name = merge.name  # the join removes the object; keep the name for the warning check
    res = merge_armatures(base, merge)
    check(res["verdict"] == "PASS",
          "differing-origin merge FAILed unforced: %r" % res.get("offenders"))

    # The discriminating assertion: one rigid map moved everything. Shared bone
    # names are unified into the base's copy by the merge, so iterating the merge
    # rig's shared names would re-assert the BASE's bone and police nothing —
    # only merge-ONLY bones carry the merge rig's frame, and the check demands at
    # least one so a fixture that lost them cannot pass vacuously.
    post = _world_heads(base)
    merge_only = {b: h for b, h in pre_merge.items() if b not in pre_base}
    check(merge_only, "fixture has no merge-only bone — nothing polices the "
                      "merge rig's frame, only the base's")
    for pre in (pre_base, merge_only):
        for bone, head in pre.items():
            want = delta @ head
            got = post.get(bone, post.get(bone + ".merge"))
            check(got is not None and (got - want).length < 1e-4,
                  "bone %r landed at %s, want %s (delta @ its pre-merge head) — "
                  "the two rigs were not moved by ONE delta"
                  % (bone, got and tuple(round(v, 4) for v in got),
                     tuple(round(v, 4) for v in want)))

    out = os.path.join(tempfile.mkdtemp(), "merged_origins.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=base)
    ymin, ymax = _vertex_y_extremes(out)
    check(abs(ymin - (-0.2)) < 1e-4 and abs(ymax - 0.1) < 1e-4,
          "merge baked a flipped frame: y extremes %.4f..%.4f (want -0.2..0.1)"
          % (ymin, ymax))
    warns = " ".join((res.get("report") or {}).get("warnings", []))
    check("carried by" in warns and repr(merge_name) in warns,
          "the cleared-with-carry warning does not say the merge rig was carried "
          "(warnings: %r)" % warns)
    # The DISTANCE, not just the sentence: a 180 deg turn about the base's origin
    # moves this rig's origin the full 1.0 (2 x 0.5). Measuring the base's origin
    # instead would print 0.0000 here — it is the delta's fixed point — so the
    # one disclosure of a permanent world-space move would say nothing moved.
    check("moves 'Armature.Out''s origin 1.0000" in warns,
          "the warning does not disclose the true world-space displacement "
          "(want 1.0000; warnings: %r)" % warns)


def test_export_parented_gate_sees_non_euler_rotation_modes():
    """The parented-armature gate must read the rotation the object actually has,
    not ``rotation_euler``.

    Those are separate RNA fields. Measured: an armature in QUATERNION mode
    carrying 180 deg reads ``rotation_euler == (0,0,0)`` while ``matrix_world``
    reads -180, so a euler-keyed gate calls it unrotated and lets through exactly
    the parent/object rotation split it declares unjudgeable. AXIS_ANGLE behaves
    identically."""
    import tempfile
    from avatarprep.core import fbx_export
    from mathutils import Quaternion
    for mode in ('QUATERNION', 'AXIS_ANGLE'):
        _clear_scene()
        holder = bpy.data.objects.new("Holder", None)
        bpy.context.collection.objects.link(holder)
        holder.rotation_euler = (math.pi / 2, 0.0, 0.0)
        arm = _parked_rig("ArmA", [("Hips", (0, 0, 1.0), None)], park_z=0.0)
        arm.parent = holder
        arm.rotation_mode = mode
        if mode == 'QUATERNION':
            arm.rotation_quaternion = Quaternion((0.0, 0.0, 1.0), math.pi)
        else:
            arm.rotation_axis_angle = (math.pi, 0.0, 0.0, 1.0)
        bpy.context.view_layer.update()
        check(arm.rotation_euler.to_quaternion().angle < 1e-6,
              "fixture(%s): rotation_euler must read identity, or this case tests "
              "nothing" % mode)
        raised = None
        try:
            fbx_export.export_unity_fbx(os.path.join(tempfile.mkdtemp(), "qm.fbx"))
        except ValueError as e:
            raised = e
        check(raised is not None and "ArmA" in str(raised),
              "a parented armature carrying its rotation in %s mode must refuse — "
              "a rotation_euler-keyed gate reads it as unrotated; got %r"
              % (mode, raised))


def test_export_constrained_candidate_refuses():
    """A constraint makes matrix_world depsgraph-derived, so the clear's carry
    does not stick and silently does nothing of what it says. merge_armatures
    preflights this on its own apply path. check_scale_normalizable does not
    cover it — that refuses only scale-affecting constraints, and only at a
    non-unit evaluated scale."""
    import tempfile
    from avatarprep.core import fbx_export
    _clear_scene()
    tgt = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(tgt)
    a = _parked_rig("ArmA", [("Hips", (0, 0, 1.0), None)], park_z=math.pi)
    con = a.constraints.new('COPY_ROTATION')      # not a _SCALE_CONSTRAINT
    con.target = tgt
    bpy.context.view_layer.update()
    raised = None
    try:
        fbx_export.export_unity_fbx(os.path.join(tempfile.mkdtemp(), "con.fbx"))
    except ValueError as e:
        raised = e
    check(raised is not None and "ArmA" in str(raised) and "constraint" in str(raised),
          "a constrained candidate must refuse — the carry write would not stick; "
          "got %r" % raised)


def test_export_sole_parented_candidate_still_exports():
    """The carve-out that keeps item 3's win. One armature under a rotated EMPTY —
    the cm-unit root-Null shape — has no second rig to disagree with, so it must
    still export rather than being swept up by the refusal above."""
    import tempfile
    from avatarprep.core import fbx_export
    _clear_scene()
    holder = bpy.data.objects.new("Holder", None)
    bpy.context.collection.objects.link(holder)
    holder.rotation_euler = (math.pi / 2, 0.0, 0.0)
    arm = _parked_rig("ArmA", [("Hips", (0, 0, 1.0), None)], mesh_name="BodyA",
                      park_z=0.0)
    arm.parent = holder
    bpy.context.view_layer.update()
    out = os.path.join(tempfile.mkdtemp(), "sole_parented.fbx")
    raised = None
    try:
        fbx_export.export_unity_fbx(out)
    except ValueError as e:
        raised = e
    check(raised is None,
          "a SOLE parented candidate must still export — this is the cm-unit "
          "root-Null class the narrowing exists for; got %r" % raised)
    check(os.path.exists(out), "the sole-parented export wrote no file")


def test_merge_deep_cross_bound_mesh_moves_once():
    """The merge path's half of the ride-along seed hole.

    ``merge_armatures`` seeds ``already_moved`` from ``carried_by_parenting`` too,
    so the same TWO-level reach in get_bound_meshes stranded a mesh buried deeper
    than that under the merge rig while modifier-bound to the base rig: the base's
    clear moved it explicitly AND it rode the apply_world_delta carry, landing at
    delta**2. A mesh bound to NEITHER rig cannot police this — nothing ever moves
    it explicitly, so such a test passes with the hole wide open."""
    from avatarprep.core import scene_utils
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1)
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 0, 1.0), None), ("Tail", (0, -0.2, 1.0), "Hips")],
                        mesh_name="TailM", mesh_y=-0.2)
    _offset_rig(merge, (0.5, 0.0, 0.0))
    # BOTH directions. The seed used to come from the merge rig alone, so only the
    # first of these was covered; the mirror — deep under the BASE, bound to the
    # MERGE rig — rode the base's clear and was then moved again by the carry.
    deep = _deep_cross_bound_mesh("DeepMerge", merge, base, depth=2)
    deep_b = _deep_cross_bound_mesh("DeepBase", base, merge, depth=2)
    for m, under, bound_to in ((deep, merge, base), (deep_b, base, merge)):
        check(m not in scene_utils.get_bound_meshes(under),
              "fixture: %r must be past get_bound_meshes' 2-level parent reach on "
              "%r, or it never exercised the hole" % (m.name, under.name))
        check(m in scene_utils.get_bound_meshes(bound_to),
              "fixture: %r must be modifier-bound to %r — a mesh bound to neither "
              "rig is never moved explicitly and would pass regardless"
              % (m.name, bound_to.name))
    pre = {m.name: _world_verts(m) for m in (deep, deep_b)}

    delta = _clear_delta(base)
    res = merge_armatures(base, merge)
    check(res["verdict"] == "PASS",
          "deep cross-bound merge FAILed: %r" % res.get("offenders"))
    for m in (deep, deep_b):
        for got, was in zip(_world_verts(m), pre[m.name]):
            want = delta @ was
            check((got - want).length < 1e-4,
                  "%s vertex landed at %s, want %s — it rode one rig's motion AND "
                  "was moved explicitly by the other (delta**2)"
                  % (m.name, tuple(round(v, 4) for v in got),
                     tuple(round(v, 4) for v in want)))


def _deep_cross_bound_mesh(name, chain_root, modifier_arm, depth=2):
    """A mesh buried ``depth`` EMPTY levels under ``chain_root`` and modifier-bound
    to ``modifier_arm``. Past get_bound_meshes' TWO-level parent reach, which is
    what made this shape invisible to the ride-along seed."""
    from mathutils import Vector
    parent = chain_root
    for i in range(depth):
        e = bpy.data.objects.new("%s_E%d" % (name, i), None)
        bpy.context.collection.objects.link(e)
        e.parent = parent
        e.matrix_parent_inverse = parent.matrix_world.inverted()
        parent = e
    md = bpy.data.meshes.new(name + "Data")
    md.from_pydata([(-0.05, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.1)],
                   [], [(0, 1, 2)])
    md.update()
    mo = bpy.data.objects.new(name, md)
    bpy.context.collection.objects.link(mo)
    mo.modifiers.new("Armature", 'ARMATURE').object = modifier_arm
    mo.parent = parent
    mo.matrix_parent_inverse = parent.matrix_world.inverted()
    bpy.context.view_layer.update()
    return mo


def test_export_multi_rig_refuses():
    """A whole-scene export with two armatures in scope must refuse up front,
    naming the rigs and all three remedies. The machinery that once served this
    path is deleted; the refusal is also what stops an appended disposable
    reference body from silently shipping (own-mergeable's accident, previously
    only warned about)."""
    import tempfile
    from avatarprep.core import fbx_export
    _clear_scene()
    _parked_rig("ArmA", [("Hips", (0, 0, 1.0), None)], mesh_name="BodyA")
    _parked_rig("ArmB", [("Hips", (0, 0, 1.0), None)], mesh_name="BodyB")
    bpy.context.view_layer.update()
    raised = None
    try:
        fbx_export.export_unity_fbx(os.path.join(tempfile.mkdtemp(), "multi.fbx"))
    except ValueError as e:
        raised = e
    check(raised is not None and "ArmA" in str(raised) and "ArmB" in str(raised),
          "a two-armature whole-scene export must refuse naming both rigs; "
          "got %r" % raised)
    for remedy in ("merge_armatures", "armature_obj", "keep_object_rotation"):
        check(raised is not None and remedy in str(raised),
              "the refusal must name remedy %r; got %r" % (remedy, raised))


def test_export_scoped_from_two_armature_scene_exports():
    """The own-mergeable shape: an owned rig beside an appended disposable
    reference body, exported SCOPED. Scoping pins candidates to the one named
    rig, so the multi-rig refusal must not fire — and the reference body must
    stay out of the written file."""
    import tempfile
    from avatarprep.core import fbx_export
    _clear_scene()
    a = _parked_rig("ArmA", [("Hips", (0, 0, 1.0), None)], mesh_name="BodyA")
    _parked_rig("ArmB", [("Hips", (0, 0, 1.0), None)], mesh_name="BodyB")
    bpy.context.view_layer.update()
    out = os.path.join(tempfile.mkdtemp(), "scoped_two_arm.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=a)
    names = _model_node_names(out)
    check("ArmA" in names and "BodyA" in names,
          "scoped export must carry the scoped rig and its mesh; got %r"
          % (sorted(names),))
    check("ArmB" not in names and "BodyB" not in names,
          "scoped export must NOT ship the reference rig; got %r"
          % (sorted(names),))


def _model_node_names(path):
    """Names of every Model node in the written file."""
    from io_scene_fbx import parse_fbx
    root, _ = parse_fbx.parse(path)
    objects = next(e for e in root.elems if e.id == b"Objects")
    return {e.props[1].decode("utf-8", "replace").split("\x00")[0]
            for e in objects.elems if e.id == b"Model"}


def _cross_bound_mesh(name, parent_arm, modifier_arm, at):
    """A mesh PARENTED to one rig and modifier-bound to the other — bound to both
    as far as get_bound_meshes is concerned, but moved by only one of them."""
    from mathutils import Vector
    md = bpy.data.meshes.new(name + "Data")
    md.from_pydata([(-0.05, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.1)],
                   [], [(0, 1, 2)])
    md.update()
    mo = bpy.data.objects.new(name, md)
    bpy.context.collection.objects.link(mo)
    mod = mo.modifiers.new("Armature", 'ARMATURE')
    mod.object = modifier_arm
    mo.parent = parent_arm
    mo.matrix_parent_inverse = parent_arm.matrix_world.inverted()
    mo.location = Vector(at)
    bpy.context.view_layer.update()
    return mo


def _world_verts(mesh_obj):
    bpy.context.view_layer.update()
    return [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]


def test_merge_cross_bound_mesh_moves_once():
    """A mesh bound to BOTH rigs must be moved by the delta exactly once. Parented
    to one rig it rides along with that rig; if the other rig's clear or carry also
    moves it explicitly it lands at delta**2 — measured at 180 deg off the skeleton
    it is bound to, in both parent/modifier orientations."""
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1)
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 0, 1.0), None), ("Tail", (0, -0.2, 1.0), "Hips")],
                        mesh_name="TailM", mesh_y=-0.2)
    _offset_rig(merge, (0.5, 0.0, 0.0))
    # Built after the offset so _offset_rig's compensation does not touch them.
    a = _cross_bound_mesh("ParentedToBase", base, merge, (0.1, 0.3, 1.0))
    b = _cross_bound_mesh("ParentedToMerge", merge, base, (-0.1, 0.3, 1.0))
    pre = {m.name: _world_verts(m) for m in (a, b)}

    delta = _clear_delta(base)
    res = merge_armatures(base, merge)
    check(res["verdict"] == "PASS",
          "cross-bound merge FAILed: %r" % res.get("offenders"))
    for m in (a, b):
        for got, was in zip(_world_verts(m), pre[m.name]):
            want = delta @ was
            check((got - want).length < 1e-4,
                  "%s vertex landed at %s, want %s — a mesh bound to both rigs "
                  "was moved twice (delta**2) or not at all"
                  % (m.name, tuple(round(v, 4) for v in got),
                     tuple(round(v, 4) for v in want)))


def test_merge_differing_origins_identity_rotation_noop():
    """The plurality class through the collapsed gate: identity rotations, origins
    differing. There is no residue to clear, so nothing may move."""
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1, park_z=0.0)
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 0, 1.0), None), ("Tail", (0, -0.2, 1.0), "Hips")],
                        mesh_name="TailM", mesh_y=-0.2, park_z=0.0)
    _offset_rig(merge, (0.5, 0.0, 0.0))
    pre = dict(_world_heads(base), **_world_heads(merge))
    res = merge_armatures(base, merge)
    check(res["verdict"] == "PASS",
          "identity-rotation differing-origin merge FAILed: %r" % res.get("offenders"))
    post = _world_heads(base)
    for bone, head in pre.items():
        got = post.get(bone, post.get(bone + ".merge"))
        check(got is not None and (got - head).length < 1e-4,
              "noop path moved bone %r: %s -> %s"
              % (bone, tuple(round(v, 4) for v in head),
                 got and tuple(round(v, 4) for v in got)))


def test_merge_differing_rotations_warns():
    """The other branch, now reachable ONLY with differing rotations: no single
    delta clears both rigs, so their frames are baked as they stand and the
    warning is the only signal that exists. It must name the stranded rig, and its
    remedy must be one the operator can actually execute — 'align the origins' is
    not (measured: doing it by hand FAILs the compat gate on the re-run)."""
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1)              # parks -180 Z
    # Bones authored in the merge rig's OWN frame (R^-1 @ world target) so the two
    # rigs are world-aligned despite differing rotations — the gate passes on the
    # seam and the apply branch is what gets tested, no force needed.
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 0, 1.0), None), ("Tail", (-0.2, 0, 1.0), "Hips")],
                        mesh_name="TailM", mesh_y=-0.2, park_z=math.pi / 2)
    bpy.context.view_layer.update()
    res = merge_armatures(base, merge)
    check(res["verdict"] == "PASS",
          "differing-rotation merge FAILed unforced: %r" % res.get("offenders"))
    warns = " ".join((res.get("report") or {}).get("warnings", []))
    check("will NOT match either source's vendor frame" in warns,
          "uncleared front-axis bake did not warn that the vendor frame is lost "
          "(warnings: %r)" % warns)
    check("rotations differ by 90.0" in warns,
          "warning did not name the true rotation gap (warnings: %r)" % warns)
    check("Align the origins" not in warns,
          "warning still offers the unfollowable align-the-origins remedy "
          "(warnings: %r)" % warns)


def test_merge_antipodal_quaternion_is_equal_rotation():
    """Two rigs parked at +180 and -180 Z are the SAME rotation, but their
    quaternions are antipodal and rotation_difference().angle reads 2*pi — measured
    on 3x3s differing by 1.7e-07. Keyed on that, the gate misses the very
    (0,0,-180) front-axis class the clear exists for and bakes it."""
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1, park_z=math.pi)
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 0, 1.0), None), ("Tail", (0, -0.2, 1.0), "Hips")],
                        mesh_name="TailM", mesh_y=-0.2, park_z=-math.pi)
    bpy.context.view_layer.update()
    check(base.matrix_world.to_quaternion().dot(merge.matrix_world.to_quaternion()) < 0,
          "fixture quaternions are no longer antipodal — this test exercises nothing")
    res = merge_armatures(base, merge)
    check(res["verdict"] == "PASS", "antipodal merge FAILed: %r" % res.get("offenders"))
    warns = " ".join((res.get("report") or {}).get("warnings", []))
    check("cleared before the apply" in warns,
          "antipodal-but-equal rotations were treated as differing, so the "
          "front-axis residue got baked (warnings: %r)" % warns)


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

    # 7. Differing ORIGINS with a shared rotation: one delta must move both rigs,
    # so the vendor frame survives and the two stay rigid. Plus the identity
    # (noop) path through the same gate, and the antipodal-quaternion trap that
    # would route the front-axis class away from the clear.
    test_merge_differing_origins_keeps_vendor_frame()
    test_merge_cross_bound_mesh_moves_once()
    test_merge_deep_cross_bound_mesh_moves_once()
    test_merge_differing_origins_identity_rotation_noop()
    test_merge_antipodal_quaternion_is_equal_rotation()

    # 8. Differing ROTATIONS — the only way into the else branch now. Nothing can
    # clear both, so the warning is the whole contract.
    test_merge_differing_rotations_warns()

    # 9. The EXPORT path serves exactly ONE rig. A multi-rig whole-scene export
    # refuses up front (the machinery that once served it is deleted); scoped
    # export from a two-armature scene — the own-mergeable shape — must keep
    # working and keep the reference rig out of the file.
    test_export_parented_gate_sees_non_euler_rotation_modes()
    test_export_constrained_candidate_refuses()
    test_export_sole_parented_candidate_still_exports()
    test_export_multi_rig_refuses()
    test_export_scoped_from_two_armature_scene_exports()

    if FAILURES:
        print("FBXORIENT_TEST FAIL:", "; ".join(FAILURES))
        sys.exit(1)
    print("FBXORIENT_TEST OK")
    sys.exit(0)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "FBXORIENT_TEST")
