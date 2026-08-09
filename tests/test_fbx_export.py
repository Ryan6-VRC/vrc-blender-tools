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

    # 4. Scale layout: FBX_SCALE_ALL (the canonical default) writes a meter-unit
    # file (UnitScaleFactor=100, no compensating node scales) — what meter-unit
    # vendors ship; FBX_SCALE_NONE instead writes a cm-unit file with 100x root
    # scales. Pins the layout the export_unity_fbx docstring documents.
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

    # 5. A non-unit scene scale silently rewrites the layout (measured:
    # scale_length=0.01 writes USF~1) — the exporter must refuse, not comply.
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

    # --- Parked object scale (D2A) ---------------------------------------------
    # Shared readers for cases 6-8.
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

    # 6. A parked object scale is baked into the data, so the written file carries
    # identity node scales whatever the source's unit class was. A cm-unit source
    # reaches the exporter as 0.01 over cm-magnitude geometry; before this, the
    # file shipped that 0.01 on every root-level node (measured on Chocolat: 21 of
    # 289) and Unity had nothing left to normalize it with, because our own file
    # honestly declares meters. World layout is what must survive the bake.
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

    # 7. Parents are applied before children. Reciprocal parent/child scales are
    # the case that cannot converge child-first — applying the child first strands
    # the parent's scale on it (measured on Sio_AFK: armature 0.498056 against
    # mesh 2.007806, which is why the ordering is a guarantee and not an accident).
    arm = _make_rig()
    arm.scale = (0.5, 0.5, 0.5)
    bpy.data.objects["Body"].scale = (2.0, 2.0, 2.0)
    bpy.context.view_layer.update()
    out = os.path.join(tempfile.mkdtemp(), "reciprocal.fbx")
    fbx_export.export_unity_fbx(out, armature_obj=arm)
    check(not _offenders(out),
          "reciprocal parent/child scales must both normalize (parent first); "
          "offenders: %r" % _offenders(out))

    # 8. The scope reaches non-mesh descendants. An EMPTY between the armature and
    # its meshes is not in get_bound_meshes, and applying a parent only relocates
    # its scale onto the child's local matrix — so a bound-set-only walk moves the
    # number onto the EMPTY rather than removing it (measured on Monoteiru's
    # ``geo_grp`` tree: non-unit node count 1/7 before and 1/7 after).
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

    if FAILURES:
        print("FBXEXPORT_TEST FAIL:", "; ".join(FAILURES))
        sys.exit(1)
    print("FBXEXPORT_TEST OK")
    sys.exit(0)


main()
