"""Regression oracle for export_unity_fbx's OBJECT-mode guard.

Asserts export succeeds from POSE on both the scoped and whole-scene branches and
does not regress from OBJECT.
"""
import math
import os
import sys
import tempfile

import bpy
from mathutils import Matrix, Quaternion, Vector

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

    # 4. Scale layout: pins the layout the export_unity_fbx docstring documents
    # (FBX_SCALE_ALL against FBX_SCALE_NONE).
    from avatarprep.core import fbx_export
    from io_scene_fbx import parse_fbx

    def _layout(path):
        root, _ = parse_fbx.parse(path)
        gs = next(e for e in root.elems if e.id == b"GlobalSettings")
        usf = None
        for p70 in (c for c in gs.elems if c.id == b"Properties70"):
            for p in p70.elems:
                if p.props[0] == b"UnitScaleFactor":
                    usf = float(p.props[4])
        objects = next(e for e in root.elems if e.id == b"Objects")
        arm_scale = (1.0, 1.0, 1.0)
        for e in objects.elems:
            if e.id != b"Model":
                continue
            name = e.props[1].decode("utf-8", "replace").split("\x00")[0]
            if name != "Armature":
                continue
            for p70 in (c for c in e.elems if c.id == b"Properties70"):
                for p in p70.elems:
                    if p.props[0] == b"Lcl Scaling":
                        arm_scale = tuple(float(v) for v in p.props[4:7])
        return usf, arm_scale

    for opt, want_usf, want_scale in (('FBX_SCALE_ALL', 100.0, 1.0),
                                      ('FBX_SCALE_NONE', 1.0, 100.0)):
        arm = _make_rig()
        out = os.path.join(tempfile.mkdtemp(), "scale_%s.fbx" % opt)
        fbx_export.export_unity_fbx(out, armature_obj=arm,
                                    apply_scale_options=opt)
        usf, arm_scale = _layout(out)
        check(usf == want_usf, "%s: UnitScaleFactor %s (want %s)"
              % (opt, usf, want_usf))
        check(all(abs(s - want_scale) < 1e-3 for s in arm_scale),
              "%s: armature node scale %s (want %s)" % (opt, arm_scale, want_scale))

    # 5. A non-unit scene scale refuses.
    arm = _make_rig()
    us = bpy.context.scene.unit_settings
    us.system, us.scale_length = 'METRIC', 0.01
    try:
        raised = None
        try:
            fbx_export.export_unity_fbx(
                os.path.join(tempfile.mkdtemp(), "sl.fbx"), armature_obj=arm)
        except ValueError as e:
            raised = e
        check(raised is not None and "scale_length" in str(raised),
              "non-unit scale_length must refuse loud, got %r" % raised)
    finally:
        us.scale_length = 1.0

    # --- Parked object scale ---------------------------------------------------
    def _all_node_scales(path):
        root, _ = parse_fbx.parse(path)
        objects = next(e for e in root.elems if e.id == b"Objects")
        out = []
        for e in objects.elems:
            if e.id != b"Model":
                continue
            name = e.props[1].decode("utf-8", "replace").split("\x00")[0]
            s = (1.0, 1.0, 1.0)
            for p70 in (c for c in e.elems if c.id == b"Properties70"):
                for p in p70.elems:
                    if p.props[0] == b"Lcl Scaling":
                        s = tuple(float(v) for v in p.props[4:7])
            out.append((name, s))
        return out

    def _offenders(path):
        return [(n, s) for n, s in _all_node_scales(path)
                if any(abs(c - 1.0) > 1e-4 for c in s)]

    def _world_span(obj):
        lo = [1e9] * 3
        hi = [-1e9] * 3
        for c in obj.bound_box:
            w = obj.matrix_world @ Vector(c)
            for i in range(3):
                lo[i] = min(lo[i], w[i])
                hi[i] = max(hi[i], w[i])
        return [hi[i] - lo[i] for i in range(3)]

    # 6. A parked object scale is baked into the data; world layout survives the bake.
    arm = _make_rig()
    body = bpy.data.objects["Body"]
    for v in body.data.vertices:
        v.co *= 100.0
    for o in (arm, body):
        o.scale = (0.01, 0.01, 0.01)
    bpy.context.view_layer.update()
    want_span = _world_span(body)
    out = os.path.join(tempfile.mkdtemp(), "parked_scale.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=arm)
    check(not _offenders(out),
          "a parked 0.01 must be baked into the data, not written as node scale; "
          "offenders: %r" % _offenders(out))
    got_span = _world_span(body)
    check(all(abs(a - b) <= 1e-5 for a, b in zip(want_span, got_span)),
          "normalizing must be world-preserving: span %r -> %r" % (want_span, got_span))

    # 7. Parents are applied before children (hierarchy_ordered).
    arm = _make_rig()
    arm.scale = (0.5, 0.5, 0.5)
    bpy.data.objects["Body"].scale = (2.0, 2.0, 2.0)
    bpy.context.view_layer.update()
    out = os.path.join(tempfile.mkdtemp(), "reciprocal.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=arm)
    check(not _offenders(out),
          "reciprocal parent/child scales must both normalize (parent first); "
          "offenders: %r" % _offenders(out))

    # 8. The scope reaches non-mesh descendants (hierarchy_ordered).
    arm = _make_rig()
    empty = bpy.data.objects.new("geo_grp", None)
    bpy.context.collection.objects.link(empty)
    empty.parent = arm
    bpy.data.objects["Body"].parent = empty
    arm.scale = (0.01, 0.01, 0.01)
    bpy.context.view_layer.update()
    out = os.path.join(tempfile.mkdtemp(), "empty_between.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=arm)
    check(not _offenders(out),
          "an EMPTY between armature and mesh must not strand the scale; "
          "offenders: %r" % _offenders(out))

    # 9. Every refusal precedes the irreversible bake: assert on the scene, not just
    # on the raise.
    #
    # The armature carries its own rotation as well as a parent, or this fixture
    # stops refusing (9b pins the rotation-less case).
    from avatarprep.core import scene_utils
    arm = _make_rig()
    parent = bpy.data.objects.new("RootEmpty", None)
    bpy.context.collection.objects.link(parent)
    arm.parent = parent
    arm.rotation_euler = (0.0, 0.0, math.pi)
    arm.scale = (0.01, 0.01, 0.01)
    bpy.context.view_layer.update()
    vert_before = bpy.data.objects["Body"].data.vertices[2].co.copy()
    raised = None
    try:
        fbx_export.export_unity_fbx(
            os.path.join(tempfile.mkdtemp(), "refused.fbx"), armature_obj=arm)
    except ValueError as e:
        raised = e
    check(raised is not None, "a parented+rotated armature must still refuse")
    check(all(abs(c - 0.01) < 1e-6 for c in arm.scale),
          "a refused export must not have baked the armature scale; got %r"
          % (tuple(arm.scale),))
    check((bpy.data.objects["Body"].data.vertices[2].co - vert_before).length < 1e-9,
          "a refused export must not have rewritten mesh data")

    # 9b. A parent with no rotation of its own is a noop and must export (fbx_export's
    # parented-armature comment).
    _clear_scene()
    arm = _make_rig()
    parent = bpy.data.objects.new("RootEmpty", None)
    bpy.context.collection.objects.link(parent)
    arm.parent = parent                      # parent at IDENTITY scale and rotation
    bpy.context.view_layer.update()
    out = os.path.join(tempfile.mkdtemp(), "parented_noop.fbx")
    raised = None
    try:
        fbx_export.export_unity_fbx(out, armature_obj=arm)
    except ValueError as e:
        raised = e
    check(raised is None,
          "a parented armature with identity rotation must export, not refuse: %r"
          % raised)
    # Guarded so a regression reports instead of crashing on the missing file.
    if os.path.exists(out):
        check(not _offenders(out),
              "the parented-but-unrotated export must still ship identity node "
              "scales; offenders: %r" % _offenders(out))
    else:
        check(False, "the parented-but-unrotated export wrote no file")

    # 10. Scope does not escape the caller's set (hierarchy_ordered).
    _clear_scene()
    arm = _make_rig()
    root = bpy.data.objects.new("SceneRoot", None)
    bpy.context.collection.objects.link(root)
    arm.parent = root
    prop = bpy.data.objects.new("UnrelatedProp", bpy.data.meshes.new("PropData"))
    bpy.context.collection.objects.link(prop)
    prop.parent = root
    prop.scale = (3.0, 3.0, 3.0)
    bpy.context.view_layer.update()
    scene_utils.normalize_object_scale([arm] + scene_utils.get_bound_meshes(arm))
    check(all(abs(c - 3.0) < 1e-6 for c in prop.scale),
          "normalize must not reach an unrelated sibling; prop scale is now %r"
          % (tuple(prop.scale),))

    # 11. Refusals where the bake is not world-preserving (check_scale_normalizable).
    def _refuses(build, want, label):
        _clear_scene()
        a = _make_rig()
        build(a)
        bpy.context.view_layer.update()
        err = None
        try:
            scene_utils.check_scale_normalizable(
                [a] + scene_utils.get_bound_meshes(a))
        except ValueError as e:
            err = str(e)
        check(err is not None and want in err,
              "%s must refuse with %r; got %r" % (label, want, err))

    def _posed(a):
        a.scale = (0.01, 0.01, 0.01)
        a.pose.bones["Root"].location = (0, 0, 10.0)
    _refuses(_posed, "pose translation", "a posed, scaled armature")

    def _sheared(a):
        a.scale = (2.0, 1.0, 1.0)
        bpy.data.objects["Body"].rotation_euler = (0, 0, 0.785398)
    _refuses(_sheared, "composed shear",
             "non-uniform scale over a rotated descendant")

    # 11a. Representation independence. All three shapes below compose the same
    # shear as _sheared above while reading ``rotation_euler == (0,0,0)``. Each
    # fixture asserts the identity euler read first, or the case tests nothing.
    def _sheared_mode(mode):
        def build(a):
            a.scale = (2.0, 1.0, 1.0)
            body = bpy.data.objects["Body"]
            body.rotation_mode = mode
            if mode == 'QUATERNION':
                body.rotation_quaternion = Quaternion((0.0, 0.0, 1.0), 0.785398)
            else:
                body.rotation_axis_angle = (0.785398, 0.0, 0.0, 1.0)
            check(body.rotation_euler.to_quaternion().angle < 1e-6,
                  "fixture(%s): rotation_euler must read identity, or this case "
                  "tests nothing" % mode)
        return build
    for _mode in ('QUATERNION', 'AXIS_ANGLE'):
        _refuses(_sheared_mode(_mode), "composed shear",
                 "non-uniform scale over a %s-mode rotated descendant" % _mode)

    # The rotation twin of the delta_scale refusal below: matrix_basis includes
    # delta rotation, rotation_euler does not.
    def _sheared_delta(a):
        a.scale = (2.0, 1.0, 1.0)
        body = bpy.data.objects["Body"]
        body.delta_rotation_euler = (0.0, 0.785398, 0.0)
        check(body.rotation_euler.to_quaternion().angle < 1e-6,
              "fixture(delta): rotation_euler must read identity, or this case "
              "tests nothing")
    _refuses(_sheared_delta, "composed shear",
             "non-uniform scale over a delta_rotation-only descendant")

    # 11b. ...and must not shadow the refusals naming a mirrored or degenerate descendant;
    # ordering keeps that true. Asserts which refusal answered, not just that something
    # refused.
    def _mirrored_under_nonuniform(a):
        a.scale = (2.0, 1.0, 1.0)
        bpy.data.objects["Body"].scale = (-1.0, 1.0, 1.0)
    _refuses(_mirrored_under_nonuniform, "negative (mirrored) scale",
             "a mirrored descendant under a non-uniform parent")

    def _degenerate_under_nonuniform(a):
        a.scale = (2.0, 1.0, 1.0)
        bpy.data.objects["Body"].scale = (0.0, 1.0, 1.0)
    _refuses(_degenerate_under_nonuniform, "zero scale component",
             "a degenerate descendant under a non-uniform parent")

    # The rotated variants 11b's guards defend against, and the (0,0,0) shape.
    def _mirrored_rotated(a):
        a.scale = (2.0, 1.0, 1.0)
        b = bpy.data.objects["Body"]
        b.scale = (-1.0, 1.0, 1.0)
        b.rotation_euler = (0, 0, 0.785398)
    _refuses(_mirrored_rotated, "negative (mirrored) scale",
             "a mirrored AND rotated descendant (composes 2.75e-1 of real shear)")

    def _degenerate_rotated(a):
        a.scale = (2.0, 1.0, 1.0)
        b = bpy.data.objects["Body"]
        b.scale = (0.0, 1.0, 1.0)
        b.rotation_euler = (0, 0, 0.785398)
    _refuses(_degenerate_rotated, "zero scale component",
             "a degenerate AND rotated descendant (composes 3.17e-1 of real shear)")

    def _fully_degenerate(a):
        bpy.data.objects["Body"].scale = (0.0, 0.0, 0.0)
    _refuses(_fully_degenerate, "zero scale component",
             "a fully degenerate descendant (must diagnose, not ZeroDivisionError)")

    def _zero(a):
        bpy.data.objects["Body"].scale = (0.0, 1.0, 1.0)
    _refuses(_zero, "zero scale component", "a zero scale component")

    def _delta(a):
        a.delta_scale = (0.01, 0.01, 0.01)
    _refuses(_delta, "delta_scale", "a delta_scale the bake cannot consume")

    # 11d. Shapes composing no shear must export.
    def _accepts(build, label):
        _clear_scene()
        a = _make_rig()
        build(a)
        bpy.context.view_layer.update()
        err = None
        try:
            scene_utils.check_scale_normalizable(
                [a] + scene_utils.get_bound_meshes(a))
        except ValueError as e:
            err = str(e)
        check(err is None, "%s must be accepted; got %r" % (label, err))

    # A rotation about the axis whose two PERPENDICULAR scale components are equal
    # commutes with that scale, so it composes exactly zero shear at any angle.
    for _deg in (30.0, 45.0, 180.0):
        def _commuting(a, d=_deg):
            a.scale = (2.0, 1.0, 1.0)
            bpy.data.objects["Body"].rotation_euler = (math.radians(d), 0, 0)
        _accepts(_commuting, "%g deg about the singular axis under (2,1,1)" % _deg)

    # ...and any 90/180 deg rotation about ANY axis: a signed permutation matrix
    # maps the scale frame onto itself, so the composition stays diagonal.
    def _axis_aligned_quarter(a):
        a.scale = (2.0, 1.0, 1.0)
        bpy.data.objects["Body"].rotation_euler = (0, 0, math.radians(90))
    _accepts(_axis_aligned_quarter, "90 deg about Z under (2,1,1)")

    # The real vendor shape: exporter float noise (2.9e-06 of spread) under a
    # genuine 45 deg rotation composes ~1e-06.
    def _noise_band(a):
        a.scale = (1.000002921, 1.0, 1.0)
        bpy.data.objects["Body"].rotation_euler = (0, 0, 0.785398)
    _accepts(_noise_band, "noise-band non-uniform scale over a rotated descendant")

    # 11e. Shear through matrix_parent_inverse refuses: the parent is uniform and the
    # child unrotated, while the bake moves geometry (measured 0.077 m).
    def _sheared_parent_inverse(a):
        a.scale = (2.0, 2.0, 2.0)
        body = bpy.data.objects["Body"]
        body.matrix_parent_inverse = Matrix(((1.0, 0.5, 0.0, 0.0),
                                             (0.0, 1.0, 0.0, 0.0),
                                             (0.0, 0.0, 1.0, 0.0),
                                             (0.0, 0.0, 0.0, 1.0)))
        check(scene_utils._is_unit_scale(tuple(body.scale))
              and not scene_utils.has_own_rotation(body),
              "fixture(parent-inverse): the child must read unrotated and unscaled, "
              "or this case does not test what the old proxy was blind to")
    _refuses(_sheared_parent_inverse, "composed shear",
             "shear via a sheared parent inverse under a UNIFORM parent")

    # 11f. Depth 3: the shear read reaches a descendant that is not a direct child.
    def _sheared_depth3(a):
        a.scale = (2.0, 1.0, 1.0)
        mid = bpy.data.objects.new("Mid", None)
        bpy.context.scene.collection.objects.link(mid)
        mid.parent = a
        body = bpy.data.objects["Body"]
        body.parent = mid
        body.rotation_euler = (0, 0, 0.785398)
    _refuses(_sheared_depth3, "composed shear",
             "a rotated grandchild under a non-uniform root")

    # 11h. The shear reading must not be deflated by an unrelated large axis (_composed_shear).
    def _sheared_anisotropic(a):
        a.scale = (2.0, 1.0, 100000.0)
        bpy.data.objects["Body"].rotation_euler = (0, 0, 0.785398)
    _refuses(_sheared_anisotropic, "composed shear",
             "a rotated descendant under a wildly anisotropic parent")

    # 11i. Nothing in scope is baked, so this exports; only that separates it from 11e.
    def _unbaked_parent_inverse(a):
        a.scale = (1.0, 1.0, 1.0)
        bpy.data.objects["Body"].matrix_parent_inverse = Matrix(
            ((1.0, 0.5, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    _accepts(_unbaked_parent_inverse,
             "a sheared parent inverse where the bake applies nothing")

    # 11j. A degenerate composed frame returns a float, not a ZeroDivisionError.
    _clear_scene()
    _arm2 = _make_rig()
    _arm2.scale = (2.0, 1.0, 1.0)
    _arm2.pose.bones["Root"].scale = (0.0, 0.0, 0.0)
    _child = bpy.data.objects.new("BoneChild", None)
    bpy.context.scene.collection.objects.link(_child)
    _child.parent = _arm2
    _child.parent_type = 'BONE'
    _child.parent_bone = "Root"
    bpy.context.view_layer.update()
    check(scene_utils._is_unit_scale(tuple(_child.scale))
          and max(abs(v) for v in _child.matrix_world.decompose()[2]) < 1e-9,
          "fixture(composed-degenerate): local scale must read unit and the composed "
          "frame degenerate, or this case tests nothing")
    _crash = None
    try:
        scene_utils.check_scale_normalizable([_arm2])
    except ValueError:
        _crash = "ValueError"
    except ZeroDivisionError:
        _crash = "ZeroDivisionError"
    check(_crash != "ZeroDivisionError",
          "a degenerate composed frame must not raise ZeroDivisionError out of a "
          "ValueError-only gate; got %s" % _crash)

    # 11g. The gate evaluates the depsgraph itself. Deliberately omits view_layer.update().
    _clear_scene()
    _arm = _make_rig()
    _arm.scale = (2.0, 1.0, 1.0)
    bpy.data.objects["Body"].rotation_euler = (0, 0, 0.785398)
    _stale = None
    try:
        scene_utils.check_scale_normalizable(
            [_arm] + scene_utils.get_bound_meshes(_arm))
    except ValueError as e:
        _stale = str(e)
    check(_stale is not None and "composed shear" in _stale,
          "the shear gate must evaluate the depsgraph itself, not trust the "
          "caller; on an un-flushed scene got %r" % _stale)

    # 11b. The gate's verdict is scale-invariant: it classifies a rig the same
    # whether or not the parked scale has been baked. This is independent of the
    # normalize-then-clear ordering (11c) and matters on the
    # ``bake_object_scale=False`` path, where the gate always receives a
    # scale-carrying scene. Caught mutation: switching the gate to read
    # ``matrix_world``'s un-normalized 3x3 (``old_world.to_3x3()``), whose up vector
    # is the Z scale, 0.03 unbaked against 1.0 baked, either side of ``_UP_AXIS_EPS``.
    verdicts = []
    for normalize_first in (True, False):
        _clear_scene()
        a = _make_rig()
        a.rotation_euler = (0.0, 0.0, math.pi)
        a.scale = (0.01, 0.02, 0.03)
        bpy.context.view_layer.update()
        if normalize_first:
            scene_utils.normalize_object_scale([a] + scene_utils.get_bound_meshes(a))
        status, _d, _u = scene_utils.clear_axis_convention_rotation(a, set())
        verdicts.append(status)
    check(verdicts == ['cleared', 'cleared'],
          "the axis-convention gate must classify a non-uniformly scaled rig the "
          "same whether or not the scale was baked first; got normalize-first=%r "
          "clear-first=%r" % tuple(verdicts))

    # 11c. normalize_object_scale must run before the rotation gate (fbx_export's
    # irreversible-bake comment owns why). A fixture whose meshes are parented to the rig
    # never reaches the gate's write, which is why this one is bound and not parented.
    #
    # (a) Scene restoration.
    _clear_scene()
    arm = _make_rig()
    arm.rotation_euler = (0.0, 0.0, math.pi)      # front-axis park: gate will clear it
    arm.scale = (0.01, 0.02, 0.03)
    loose = bpy.data.objects.new("Loose", bpy.data.meshes.new("LooseData"))
    loose.data.from_pydata([(0.0, 0.05, 0.2)], [], [])
    loose.data.update()
    bpy.context.collection.objects.link(loose)
    loose.vertex_groups.new(name="Root").add([0], 1.0, 'REPLACE')
    loose.modifiers.new("Armature", 'ARMATURE').object = arm   # bound, NOT parented
    loose.scale = (2.0, 3.0, 4.0)
    bpy.context.view_layer.update()
    want_world = (loose.matrix_world @ loose.data.vertices[0].co).copy()
    out = os.path.join(tempfile.mkdtemp(), "loose_bound.fbx")
    fbx_export.export_unity_fbx(out)
    check(not _offenders(out),
          "a non-uniform scale on a loose bound mesh must bake to identity node "
          "scales like any other; offenders: %r" % _offenders(out))
    bpy.context.view_layer.update()
    check(all(abs(c - 1.0) < 1e-5 for c in loose.scale),
          "the export left the loose bound mesh's object scale at %r — the undo "
          "replayed a pre-bake basis over baked data, so the mesh is now "
          "double-scaled. normalize_object_scale must run BEFORE the rotation gate."
          % (tuple(round(c, 4) for c in loose.scale),))
    got_world = loose.matrix_world @ loose.data.vertices[0].co
    check((got_world - want_world).length < 1e-5,
          "the export moved the loose bound mesh: %s -> %s (world space)"
          % (tuple(round(v, 4) for v in want_world),
             tuple(round(v, 4) for v in got_world)))

    # (b) Refusals stay ahead of the first mutation.
    _clear_scene()
    arm = _make_rig()
    arm.rotation_euler = (0.0, 0.0, math.pi)
    arm.scale = (0.01, 0.02, 0.03)
    holder = bpy.data.objects.new("MeshRoot", None)
    bpy.context.collection.objects.link(holder)
    holder.scale = (2.0, 1.0, 1.0)
    loose = bpy.data.objects.new("Loose2", bpy.data.meshes.new("Loose2Data"))
    loose.data.from_pydata([(0.0, 0.05, 0.2)], [], [])
    loose.data.update()
    bpy.context.collection.objects.link(loose)
    loose.vertex_groups.new(name="Root").add([0], 1.0, 'REPLACE')
    loose.modifiers.new("Armature", 'ARMATURE').object = arm
    loose.parent = holder                          # parented ELSEWHERE, not to the rig
    bpy.context.view_layer.update()
    raised = None
    try:
        fbx_export.export_unity_fbx(os.path.join(tempfile.mkdtemp(), "shear_order.fbx"))
    except ValueError as e:
        raised = e
    check(raised is None,
          "a bound mesh under a non-uniformly scaled parent must export cleanly "
          "when the scale is baked first; raised %r. Baking after the gate sees "
          "the rotation the gate just wrote and refuses as sheared — after the "
          "scene has already been mutated." % raised)

    # 12. bake_object_scale=False is the opt-out, and keep_object_rotation is not.
    _clear_scene()
    arm = _make_rig()
    arm.scale = (0.01, 0.01, 0.01)
    bpy.context.view_layer.update()
    fbx_export.export_unity_fbx(
        os.path.join(tempfile.mkdtemp(), "optout.fbx"), armature_obj=arm,
        bake_object_scale=False)
    check(all(abs(c - 0.01) < 1e-6 for c in arm.scale),
          "bake_object_scale=False must leave the scale alone; got %r"
          % (tuple(arm.scale),))

    # --- 13. Out-of-scope ancestors --------------------------------------------
    # check_scale_normalizable's out-of-scope comment owns why these refuse.
    def _loose_under_holder(holder_scale, child_rot=(0.0, 0.0, 0.0),
                            grandparent_scale=None, delta_scale=None):
        """Armature at root; its modifier-bound mesh parented to a Holder EMPTY
        that is OUTSIDE the armature's scope. Returns (arm, holder)."""
        _clear_scene()
        a = _make_rig()
        body = bpy.data.objects["Body"]
        holder = bpy.data.objects.new("Holder", None)
        bpy.context.collection.objects.link(holder)
        holder.scale = holder_scale
        if delta_scale is not None:
            holder.delta_scale = delta_scale
        if grandparent_scale is not None:
            gp = bpy.data.objects.new("GrandHolder", None)
            bpy.context.collection.objects.link(gp)
            gp.scale = grandparent_scale
            holder.parent = gp
        body.parent = holder            # bound to `a` by modifier, parented elsewhere
        body.rotation_euler = child_rot
        bpy.context.view_layer.update()
        return a, holder

    def _scoped_raises(arm, tag):
        err = None
        try:
            fbx_export.export_unity_fbx(
                os.path.join(tempfile.mkdtemp(), tag + ".fbx"), armature_obj=arm)
        except ValueError as e:
            err = e
        return err

    # 13a. The leak refuses rather than widening scope.
    arm, holder = _loose_under_holder((2.0, 2.0, 2.0))
    vert_before = bpy.data.objects["Body"].data.vertices[0].co.copy()
    err = _scoped_raises(arm, "oos_uniform")
    check(err is not None and "outside this export's scope" in str(err),
          "a scoped export must refuse an out-of-scope scaled ancestor; got %r" % err)
    check(err is not None and "Holder" in str(err),
          "the refusal must name the ancestor; got %r" % err)
    # Same invariant case 9 asserts: refusals precede the irreversible bake.
    check(all(abs(c - 2.0) < 1e-6 for c in holder.scale),
          "a refused export must not have baked the ancestor scale; got %r"
          % (tuple(holder.scale),))
    check((bpy.data.objects["Body"].data.vertices[0].co - vert_before).length < 1e-9,
          "a refused export must not have rewritten mesh data")

    # 13b. Scope-specific, not blanket: the SAME scene exports clean whole-scene,
    # where the ancestor is in scope and simply gets baked like any other object.
    arm, holder = _loose_under_holder((2.0, 2.0, 2.0))
    out = os.path.join(tempfile.mkdtemp(), "oos_wholescene.fbx")
    err = None
    try:
        fbx_export.export_unity_fbx(out)
    except ValueError as e:
        err = e
    check(err is None, "the whole-scene export of the same scene must not "
                       "refuse — the ancestor is in scope there; got %r" % err)
    check(not _offenders(out),
          "whole-scene export must still ship identity node scales; offenders: %r"
          % _offenders(out))

    # 13c. A non-uniform out-of-scope ancestor over a rotated child refuses.
    arm, _h = _loose_under_holder((2.0, 1.0, 1.0), child_rot=(0.0, 0.0, 0.785398))
    err = _scoped_raises(arm, "oos_shear")
    check(err is not None and "outside this export's scope" in str(err),
          "a non-uniform out-of-scope ancestor must refuse, not export sheared; "
          "got %r" % err)

    # 13d. Over-refusal guard. An out-of-scope ancestor at IDENTITY scale changes
    # no node scale, so it must stay silent — otherwise every mesh parented to a
    # plain grouping EMPTY stops exporting.
    arm, _h = _loose_under_holder((1.0, 1.0, 1.0))
    err = _scoped_raises(arm, "oos_identity")
    check(err is None,
          "an out-of-scope ancestor at identity scale must NOT refuse; got %r" % err)

    # 13d-bis. Over-refusal guard: a Keep Transform child under a 2.0 ancestor exports.
    _clear_scene()
    arm = _make_rig()
    body = bpy.data.objects["Body"]
    keeper = bpy.data.objects.new("KeepHolder", None)
    bpy.context.collection.objects.link(keeper)
    keeper.scale = (2.0, 2.0, 2.0)
    bpy.context.view_layer.update()
    body.parent = keeper
    body.matrix_parent_inverse = keeper.matrix_world.inverted()   # keep transform
    bpy.context.view_layer.update()
    check(all(abs(c - 1.0) < 1e-6 for c in body.matrix_world.to_scale()),
          "fixture: the child must sit at world scale 1.0, got %r"
          % (tuple(round(c, 4) for c in body.matrix_world.to_scale()),))
    err = _scoped_raises(arm, "oos_keep_transform")
    check(err is None,
          "an out-of-scope ancestor whose scale is cancelled by "
          "matrix_parent_inverse must NOT refuse — the child inherits unit scale; "
          "got %r" % err)

    # 13e. The read must be the ancestor's COMPOSED EVALUATED scale, not its own
    # `scale` field. Here the immediate out-of-scope parent reads (1,1,1) and only
    # its own parent carries the 2.0 — a `p.scale` implementation goes silent.
    arm, _h = _loose_under_holder((1.0, 1.0, 1.0), grandparent_scale=(2.0, 2.0, 2.0))
    err = _scoped_raises(arm, "oos_grandparent")
    check(err is not None and "outside this export's scope" in str(err),
          "a scaled out-of-scope GRANDparent must refuse — the check reads the "
          "composed evaluated scale, not p.scale; got %r" % err)

    # 13f. Same claim, second limb: `delta_scale` is not in `scale` either, and
    # transform_apply cannot consume it (case 11's `_delta` covers the in-scope
    # half of that).
    arm, _h = _loose_under_holder((1.0, 1.0, 1.0), delta_scale=(3.0, 3.0, 3.0))
    err = _scoped_raises(arm, "oos_delta_scale")
    check(err is not None and "outside this export's scope" in str(err),
          "an out-of-scope ancestor carrying only a delta_scale must refuse; "
          "got %r" % err)

    # 13h. Staleness: deliberately no view_layer.update(), asserted against
    # check_scale_normalizable directly because the export path's select_all flushes the
    # depsgraph and would hide it.
    _clear_scene()
    arm = _make_rig()
    body = bpy.data.objects["Body"]
    holder_stale = bpy.data.objects.new("StaleHolder", None)
    bpy.context.collection.objects.link(holder_stale)
    holder_stale.scale = (2.0, 2.0, 2.0)
    body.parent = holder_stale
    # NO view_layer.update() here — that is the point of this case.
    err = None
    try:
        scene_utils.check_scale_normalizable(
            [arm] + scene_utils.get_bound_meshes(arm))
    except ValueError as e:
        err = e
    check(err is not None and "outside this export's scope" in str(err),
          "the ancestor refusal must evaluate the depsgraph itself — without an "
          "update matrix_world.to_scale() reads (1,1,1) on a freshly-set scale "
          "and the refusal silently passes; got %r" % err)

    # 13g. keep_object_rotation=True skips the armature preflight entirely
    # (candidates=[]), and this cm-unit root-Null shape must still refuse: the
    # armature and its meshes hang off one scaled EMPTY.
    _clear_scene()
    arm = _make_rig()
    root_empty = bpy.data.objects.new("Root", None)
    bpy.context.collection.objects.link(root_empty)
    root_empty.scale = (0.01, 0.01, 0.01)
    arm.parent = root_empty
    bpy.context.view_layer.update()
    err = None
    try:
        fbx_export.export_unity_fbx(
            os.path.join(tempfile.mkdtemp(), "esc.fbx"), armature_obj=arm,
            keep_object_rotation=True)
    except ValueError as e:
        err = e
    check(err is not None and "outside this export's scope" in str(err),
          "keep_object_rotation=True must not smuggle an out-of-scope scaled "
          "ancestor past the bake; got %r" % err)

    # 14. The eps bands: the bake skips the noise band and still catches authored scale
    # (_SCALE_EPS's comment).
    _clear_scene()
    arm = _make_rig()
    body = scene_utils.get_bound_meshes(arm)[0]
    body.scale = (1.0000029, 1.0, 0.9999987)   # the measured exporter noise band
    bpy.context.view_layer.update()
    applied = scene_utils.normalize_object_scale(
        [arm] + scene_utils.get_bound_meshes(arm))
    check(applied == [],
          "a noise-band scale must not be baked (permanent mutation over float "
          "noise); got %r" % (applied,))
    body.scale = (0.9, 0.9, 0.9)               # nearest authored uniform value
    bpy.context.view_layer.update()
    applied = scene_utils.normalize_object_scale(
        [arm] + scene_utils.get_bound_meshes(arm))
    check(any(n == body.name for n, _ in applied),
          "an authored 0.9 scale must still be baked; got %r" % (applied,))

    # --- Visibility: the scope is caller-named, so visibility never filters it. ----
    # Exports assert the written file carries the mesh, and the caller's visibility flags
    # come back as authored (fbx_export's **Visibility**).
    def _vis(obj):
        return (obj.hide_get(), obj.hide_viewport, obj.hide_select)

    def _collect(arm, mesh, name):
        """Move the rig into its own collection; return that LayerCollection."""
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
        for o in (arm, mesh):
            # _make_rig links into the ACTIVE collection, which is not always the
            # scene root once an earlier case has added one — unlink from wherever
            # the object actually is.
            for c in list(o.users_collection):
                c.objects.unlink(o)
            col.objects.link(o)
        return bpy.context.view_layer.layer_collection.children[name]

    def _exports_with_geometry(arm, mesh, tag):
        """Export scoped, assert the mesh actually shipped, assert visibility restored."""
        bpy.context.view_layer.update()
        before = {o.name: _vis(o) for o in (arm, mesh)}
        out = os.path.join(tempfile.gettempdir(), "avatarprep_vis_%s.fbx" % tag)
        if os.path.exists(out):
            os.remove(out)
        try:
            fbx_export.export_unity_fbx(out, armature_obj=arm, embed_textures=False)
        except Exception as e:
            check(False, "%s: export raised %r; a caller-named scope must export "
                         "regardless of visibility" % (tag, e))
            return
        # Read the Model node names rather than grepping the bytes: the mesh object
        # is "Body" and its datablock "BodyData", so a substring test cannot tell a
        # shipped Model node from an incidental mention of the mesh data.
        nodes = [n for n, _ in _all_node_scales(out)]
        check(mesh.name in nodes,
              "%s: wrote %d bytes with no %r Model node (nodes=%r) — the "
              "empty-export defect" %
              (tag, os.path.getsize(out), mesh.name, nodes))
        after = {o.name: _vis(o) for o in (arm, mesh)}
        check(before == after,
              "%s: visibility not restored: %r -> %r" % (tag, before, after))
        os.remove(out)

    def _refuses(arm, mesh, tag, needle):
        """Export scoped, assert an in-grammar ValueError and an untouched scene."""
        bpy.context.view_layer.update()
        # hide_get() is only meaningful for an object the view layer holds; the two
        # object-level flags are readable regardless, and are what makes this
        # assertion non-vacuous for the excluded case (where in_vl is empty).
        in_vl = [o for o in (arm, mesh) if o.name in bpy.context.view_layer.objects]
        before = ({o.name: _vis(o) for o in in_vl},
                  {o.name: (o.hide_viewport, o.hide_select) for o in (arm, mesh)})
        out = os.path.join(tempfile.gettempdir(), "avatarprep_vis_%s.fbx" % tag)
        if os.path.exists(out):
            os.remove(out)
        try:
            fbx_export.export_unity_fbx(out, armature_obj=arm, embed_textures=False)
            check(False, "%s: expected a refusal, but the export succeeded" % tag)
        except ValueError as e:
            check(needle in str(e),
                  "%s: refusal must name %r; got %s" % (tag, needle, e))
        except Exception as e:
            # An excluded collection makes select_set RAISE RuntimeError. That is
            # loud but out-of-grammar: run_cli turns it into exit 2 "crashed before
            # reaching a verdict" rather than a named refusal at exit 1.
            check(False, "%s: refusal must be a ValueError, got %r" % (tag, e))
        check(not os.path.exists(out),
              "%s: a refused export must not leave an --out file" % tag)
        after = ({o.name: _vis(o) for o in in_vl},
                 {o.name: (o.hide_viewport, o.hide_select) for o in (arm, mesh)})
        check(before == after,
              "%s: a refused export must leave the scene untouched: %r -> %r" %
              (tag, before, after))

    # 1. Each of the three object-level flags, separately: all three silently
    #    no-op select_set, and each alone is enough to produce the empty file.
    for flag in ("hide_set", "hide_viewport", "hide_select"):
        arm = _make_rig(); mesh = bpy.data.objects["Body"]
        for o in (arm, mesh):
            if flag == "hide_set":
                o.hide_set(True)
            else:
                setattr(o, flag, True)
        _exports_with_geometry(arm, mesh, flag)

    # 2. The PARTIAL case: only the mesh hidden, a plausible rig-only file.
    arm = _make_rig(); mesh = bpy.data.objects["Body"]
    mesh.hide_set(True)
    _exports_with_geometry(arm, mesh, "partial_mesh_hidden")

    # 3. Collection-level hiding refuses, naming the objects, and pins the select_get() trap.
    for tag, apply in (
            ("layercol_hide_viewport", lambda lc: setattr(lc, "hide_viewport", True)),
            ("col_hide_viewport", lambda lc: setattr(lc.collection, "hide_viewport", True)),
            ("col_hide_select", lambda lc: setattr(lc.collection, "hide_select", True))):
        arm = _make_rig(); mesh = bpy.data.objects["Body"]
        # Author a hide the door WILL clear before it refuses, so the restore
        # assertion has something to prove: a fixture with every flag already False
        # compares all-False to all-False and passes even against a no-op restore.
        mesh.hide_set(True)
        arm.hide_select = True
        apply(_collect(arm, mesh, "Hidden_" + tag))
        _refuses(arm, mesh, tag, "could not be selected")

    # 3b. The unhide must stay inside the named scope. Every other case hides only
    #     objects that are in scope, so a regression that unhid the whole file, or a
    #     get_bound_meshes that widened, would pass all of them. A second hidden rig
    #     stays hidden and stays out of the file.
    arm = _make_rig(); mesh = bpy.data.objects["Body"]
    other_arm = bpy.data.objects.new("Armature_Other", bpy.data.armatures.new("OtherData"))
    other_mesh_data = bpy.data.meshes.new("OtherData_M")
    other_mesh_data.from_pydata([(1, 0, 0), (1.1, 0, 0), (1, 0.1, 0)], [], [(0, 1, 2)])
    other_mesh_data.update()
    other_mesh = bpy.data.objects.new("Other_Body", other_mesh_data)
    for o in (other_arm, other_mesh):
        bpy.context.collection.objects.link(o)
    other_mesh.parent = other_arm
    for o in (other_arm, other_mesh):
        o.hide_set(True)
        o.hide_viewport = True
    out = os.path.join(tempfile.gettempdir(), "avatarprep_vis_scope.fbx")
    if os.path.exists(out):
        os.remove(out)
    fbx_export.export_unity_fbx(out, armature_obj=arm, embed_textures=False,
                                keep_object_rotation=True)
    for o in (other_arm, other_mesh):
        check(o.hide_get() and o.hide_viewport,
              "out-of-scope %r was unhidden: the unhide must touch only the named "
              "scope" % o.name)
    nodes = [n for n, _ in _all_node_scales(out)]
    check("Body" in nodes, "scope containment: the named mesh must still ship "
                           "(nodes=%r)" % nodes)
    check("Other_Body" not in nodes,
          "scope containment: an out-of-scope mesh shipped (nodes=%r)" % nodes)
    os.remove(out)

    # 4. An excluded collection raises at select_set and hide_get() lies: membership runs
    #    before the snapshot.
    arm = _make_rig(); mesh = bpy.data.objects["Body"]
    _collect(arm, mesh, "Excluded").exclude = True
    _refuses(arm, mesh, "excluded", "not in view layer")

    # 5. Whole-scene: an excluded collection refuses rather than shipping a partial file.
    arm = _make_rig(); mesh = bpy.data.objects["Body"]
    bpy.ops.mesh.primitive_uv_sphere_add(location=(3, 0, 0))
    _collect(arm, mesh, "ExcludedWhole").exclude = True
    out = os.path.join(tempfile.gettempdir(), "avatarprep_vis_whole.fbx")
    if os.path.exists(out):
        os.remove(out)
    try:
        fbx_export.export_unity_fbx(out, embed_textures=False)
        check(False, "whole-scene: an excluded collection must refuse, not ship a "
                     "partial file at exit 0")
    except ValueError as e:
        check("not in view layer" in str(e),
              "whole-scene: refusal must name the view layer; got %s" % e)
    check(not os.path.exists(out),
          "whole-scene: a refused export must not leave an --out file")

    # 6. A view layer with no objects at all would write an empty FBX and report
    #    success — the same defect with no offender to name.
    _clear_scene()
    out = os.path.join(tempfile.gettempdir(), "avatarprep_vis_empty.fbx")
    if os.path.exists(out):
        os.remove(out)
    try:
        fbx_export.export_unity_fbx(out, embed_textures=False)
        check(False, "an empty view layer must refuse, not write an empty FBX")
    except ValueError as e:
        check("no objects" in str(e),
              "empty-scene refusal must say so plainly; got %s" % e)

    # Which checkout actually ran (test_import_fbx.py says why).
    print("FBXEXPORT_TEST module:", fbx_export.__file__)

    if FAILURES:
        print("FBXEXPORT_TEST FAIL:", "; ".join(FAILURES))
        sys.exit(1)
    print("FBXEXPORT_TEST OK")
    sys.exit(0)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "FBXEXPORT_TEST")
