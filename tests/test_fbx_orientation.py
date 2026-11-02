"""File-level regression oracle for export_unity_fbx's orientation contract.

Two defects pull in opposite directions (``fbx_export``'s orientation docstring is
the canon), so a fix for either alone re-breaks the other and this file pins both.

A re-import cannot see an up-axis-preserving rotation carried through, because the
importer undoes the exporter, so that oracle parses the written FBX: the armature
node's ``Lcl Rotation`` must read (-90, 0, 0). The Felis fixture's own identity node
is not the value to match.

A re-import does see an up-axis-moving rotation cleared, so that oracle is a
world-bbox round-trip. That class writes an IDENTITY node rotation, so (-90, 0, 0)
is not universal.

Scope limit: this oracle sees only the file, so a green run does not mean the avatar
faces forward in a project (``fbx_export`` documents the polarity).
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

    ``cm_unit`` picks the half of the class (``fbx_export``'s scale section), so
    both are pinned. ``export_unity_fbx`` refuses a non-1 ``scale_length``, so the
    fixture is written with the raw operator and the scene restored after."""
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
    both unit halves."""
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

    park_x = math.degrees(arm.rotation_euler[0])
    check(abs(park_x - 90.0) < 0.5,
          "%s: fixture no longer parks 90 deg X (x=%.2f) — the defect scenario "
          "is gone, re-derive this test" % (label, park_x))
    src_lo, src_hi = _world_bbox()
    src_height = src_hi[2] - src_lo[2]
    # The importer normalizes a cm-unit file into a 0.01 object scale over 100x
    # data, so world height cannot distinguish the halves; the object scale below does.
    check(src_height > 0.5,
          "%s: source imports at height %.4f, expected upright" % (label, src_height))

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
    # assertions below stay green with the message deleted.
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
    """The merge path must BAKE an up-axis-moving rotation, not clear it."""
    import tempfile
    from avatarprep.core import fbx_export
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    # Y-up content: bones authored along +Y and parked at Rx(+90) so the rig stands
    # up. Z-up bones parked the same way would lie down, the mirror of this scenario.
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
    # Assert on the bone, not a bbox, which can pass on mesh spread rather than on
    # uprightness: a wrongly cleared rotation leaves Hips' head along +Y.
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
    """{bone name: world head}."""
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
    data so world layout is unchanged: the shape a source FBX whose ARMATURE node
    carries a translation imports as, and the only way into the
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
    # Only parented meshes ride the armature, so only their data is compensated;
    # touching an unparented modifier-bound mesh would displace it from the skeleton.
    for m in [o for o in bpy.data.objects
              if o.type == 'MESH' and o.parent == arm]:
        for v in m.data.vertices:
            v.co = v.co - shift
    arm.location = Vector(arm.location) + Vector(origin)
    bpy.context.view_layer.update()


def test_merge_differing_origins_keeps_vendor_frame():
    """Two up-axis-PRESERVING rigs whose origins differ must still bake the VENDOR
    frame.

    The exported-geometry oracle alone cannot police this: a naive per-rig clear
    writes byte-identical y extremes to the correct fix. The discriminator is that
    EVERY bone on BOTH rigs lands at delta @ its pre-merge world head, true only when
    one delta moved the whole assembly."""
    import tempfile
    from avatarprep.core import fbx_export
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1)
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 0, 1.0), None), ("Tail", (0, -0.2, 1.0), "Hips")],
                        mesh_name="TailM", mesh_y=-0.2)
    # An unparented modifier-bound mesh: without it the mesh-carry loop in
    # apply_world_delta can be deleted and every other assertion here still passes.
    tail_mesh = bpy.data.objects["TailM"]
    tail_mesh.parent = None
    tail_mesh.rotation_euler = (0.0, 0.0, math.pi)
    _offset_rig(merge, (0.5, 0.0, 0.0))

    pre_base, pre_merge = _world_heads(base), _world_heads(merge)
    # _parked_rig writes rotation_euler without a depsgraph update; a fixture built
    # without one ends up 1.0 apart in world space and the compat gate FAILs on it.
    check((base.matrix_world.translation - merge.matrix_world.translation).length > 1e-3,
          "fixture no longer has differing origins — this test exercises nothing")
    for bone, head in pre_base.items():
        # 1e-5, tighter than the post-merge 1e-4, so a drifting fixture fails here
        # rather than blaming the code under test.
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

    # The merge unifies shared bone names into the base's copy, so only merge-only
    # bones carry the merge rig's frame.
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
    # A 180 deg turn about the base's origin moves this rig's origin 1.0 (2 x 0.5);
    # measuring the base's origin would print 0.0000, the delta's fixed point.
    check("moves 'Armature.Out''s origin 1.0000" in warns,
          "the warning does not disclose the true world-space displacement "
          "(want 1.0000; warnings: %r)" % warns)


def test_export_parented_gate_sees_non_euler_rotation_modes():
    """The parented-armature gate must read the rotation the object actually has,
    not ``rotation_euler``."""
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
    """A constrained candidate refuses: check_scale_normalizable covers only
    scale-affecting constraints at a non-unit evaluated scale."""
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


def test_export_parented_no_own_rotation_still_exports():
    """One armature with no rotation of its own under a rotated EMPTY must still
    export, not be swept up by the parented-armature refusal."""
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
    """A mesh buried past get_bound_meshes' two-level reach under one rig and
    modifier-bound to the other must move exactly once."""
    from avatarprep.core import scene_utils
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1)
    merge = _parked_rig("Armature.Out",
                        [("Hips", (0, 0, 1.0), None), ("Tail", (0, -0.2, 1.0), "Hips")],
                        mesh_name="TailM", mesh_y=-0.2)
    _offset_rig(merge, (0.5, 0.0, 0.0))
    # BOTH directions.
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
    to ``modifier_arm``."""
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
    naming the rigs and all three remedies."""
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
    """An owned rig beside an appended disposable reference body (the own-mergeable
    shape), exported scoped: the multi-rig refusal must not fire and the reference
    body must stay out of the written file."""
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
    """A mesh bound to BOTH rigs must be moved by the delta exactly once."""
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
    """Identity rotations with differing origins have no residue to clear, so nothing
    may move."""
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
    """With differing rotations no single delta clears both rigs, so their frames are
    baked as they stand and the warning is the only signal. It must not offer 'align
    the origins': doing that by hand FAILs the compat gate on the re-run."""
    from avatarprep.core.merge_armatures import merge_armatures
    _clear_scene()
    base = _parked_rig("Armature", [("Hips", (0, 0, 1.0), None)],
                       mesh_name="BodyM", mesh_y=0.1)
    # Bones authored in the merge rig's OWN frame (R^-1 @ world target) so the two
    # rigs are world-aligned despite differing rotations and the compat gate passes
    # with no force.
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
    check("'Armature.Out'" in warns,
          "warning did not name the stranded rig (warnings: %r)" % warns)
    check("Align the origins" not in warns,
          "warning still offers the unfollowable align-the-origins remedy "
          "(warnings: %r)" % warns)


def test_merge_antipodal_quaternion_is_equal_rotation():
    """Two rigs parked at +180 and -180 Z are the SAME rotation, so the merge must
    clear them, not bake."""
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

    park_z = math.degrees(arm.rotation_euler[2])
    check(abs(park_z) > 90.0,
          "fixture no longer parks an object rotation (z=%.2f deg) — "
          "the defect scenario is gone, re-derive this test" % park_z)

    tmp = tempfile.mkdtemp()

    out = os.path.join(tmp, "felis_default.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=arm)
    rot = _armature_node_rotation(out)
    check(rot is not None, "default export: no Armature node found")
    if rot:
        check(_close(rot[0], -90.0) and _close(rot[1], 0.0) and _close(rot[2], 0.0),
              "default export carries extra rotation: node rot=%s (want ~(-90,0,0))"
              % (rot,))

    check(_close(math.degrees(arm.rotation_euler[2]), park_z),
          "export did not restore the armature object rotation")

    # The vendor file's own vertex extremes are the truth; a baked 180° flip negates them.
    v_ymin, v_ymax = _vertex_y_extremes(felis)
    o_ymin, o_ymax = _vertex_y_extremes(out)
    check(abs(v_ymin - v_ymax) > 5e-3 and abs(abs(v_ymin) - abs(v_ymax)) > 5e-3,
          "fixture geometry no longer y-asymmetric (%.4f..%.4f) — this oracle "
          "cannot discriminate a flip; re-derive it" % (v_ymin, v_ymax))
    check(abs(o_ymin - v_ymin) < 2e-3 and abs(o_ymax - v_ymax) < 2e-3,
          "exported geometry not in the vendor frame: y extremes %.4f..%.4f "
          "(vendor %.4f..%.4f)" % (o_ymin, o_ymax, v_ymin, v_ymax))

    # keep_object_rotation=True preserves the park, so the defect stays representable
    # and the oracle can tell the two apart.
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

    # An import, merge, export flow has an identity object rotation at export time,
    # so only geometry can reveal a flip the merge baked.
    test_merge_path_geometry()

    test_unparented_bound_mesh()

    for cm_unit in (False, True):
        for scoped in (True, False):
            test_yup_source_preserved(cm_unit, scoped)

    test_merge_path_yup_geometry()

    test_merge_differing_origins_keeps_vendor_frame()
    test_merge_cross_bound_mesh_moves_once()
    test_merge_deep_cross_bound_mesh_moves_once()
    test_merge_differing_origins_identity_rotation_noop()
    test_merge_antipodal_quaternion_is_equal_rotation()

    test_merge_differing_rotations_warns()

    test_export_parented_gate_sees_non_euler_rotation_modes()
    test_export_constrained_candidate_refuses()
    test_export_parented_no_own_rotation_still_exports()
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
