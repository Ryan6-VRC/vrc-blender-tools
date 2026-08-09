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
import math
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

    # 9. Every refusal precedes the irreversible bake. A refused export must leave
    # the scene EXACTLY as it found it — a mutated-then-refused scene is strictly
    # worse than either outcome, because the bake has no undo and no file is
    # written to show for it. Parented armature is the refusal that sat after the
    # bake; assert on the scene, not just on the raise.
    from avatarprep.core import scene_utils
    arm = _make_rig()
    parent = bpy.data.objects.new("RootEmpty", None)
    bpy.context.collection.objects.link(parent)
    arm.parent = parent
    arm.scale = (0.01, 0.01, 0.01)
    bpy.context.view_layer.update()
    vert_before = bpy.data.objects["Body"].data.vertices[2].co.copy()
    raised = None
    try:
        fbx_export.export_unity_fbx(
            os.path.join(tempfile.mkdtemp(), "refused.fbx"), armature_obj=arm)
    except ValueError as e:
        raised = e
    check(raised is not None, "a parented armature must still refuse")
    check(all(abs(c - 0.01) < 1e-6 for c in arm.scale),
          "a refused export must not have baked the armature scale; got %r"
          % (tuple(arm.scale),))
    check((bpy.data.objects["Body"].data.vertices[2].co - vert_before).length < 1e-9,
          "a refused export must not have rewritten mesh data")

    # 10. Scope does not escape the caller's set. Seeding the walk from each
    # object's topmost ancestor reached siblings that were never selected and are
    # never exported, and baked their authored scale permanently.
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

    # 11. Refusals for the cases where the bake is NOT world-preserving, each
    # measured. A posed armature: the bake rescales rest bones but not pose
    # translation channels (9.9 m on a 0.01 rig). Shear: non-uniform scale over a
    # rotated descendant re-decomposes without the shear term (0.041 m).
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
    _refuses(_sheared, "sheared", "non-uniform scale over a rotated descendant")

    def _zero(a):
        bpy.data.objects["Body"].scale = (0.0, 1.0, 1.0)
    _refuses(_zero, "zero scale component", "a zero scale component")

    def _delta(a):
        a.delta_scale = (0.01, 0.01, 0.01)
    _refuses(_delta, "delta_scale", "a delta_scale the bake cannot consume")

    # 11b. The gate's VERDICT is scale-invariant — it classifies a rig the same
    # whether or not the parked scale has been baked yet.
    #
    # This does NOT police the normalize-then-clear ordering; 11c does, and the
    # two are independent. It earns its place on the ``bake_object_scale=False``
    # path, where nothing bakes and the gate always receives a scale-carrying
    # scene. Caught mutation: switching the gate to read ``matrix_world``'s
    # un-normalized 3x3 (``old_world.to_3x3()``), whose up vector is the Z scale —
    # 0.03 unbaked against 1.0 baked, either side of ``_UP_AXIS_EPS``.
    # ``delta.to_3x3()`` is NOT caught and cannot be: the delta carries no scale,
    # since S cancels in ``(T*S)(T*R*S)^-1 = T*R^-1*T^-1``.
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

    # 11c. normalize_object_scale MUST run before the rotation gate, and this is
    # what pins it. The gate is not a pure predicate: for a modifier-bound mesh
    # that is NOT the armature's descendant it also writes ``m.matrix_world`` and
    # snapshots that mesh's ``matrix_basis`` for the undo replayed in the export's
    # ``finally``. Both side effects are order-sensitive, and neither is reachable
    # from a fixture whose meshes are parented to the rig (``_make_rig`` parents
    # ``Body``, so the gate takes its descendant early-out and never gets here) —
    # which is exactly why an earlier version of this test saw no difference.
    #
    # (a) Scene restoration. Clear-first snapshots the mesh's PRE-bake basis,
    # normalize then bakes the scale into its data, and the restore replays the
    # old basis over baked data — measured, a (2,3,4) bound mesh comes back
    # 2x3x4 too large, silently, with a byte-equivalent file. Baking first takes
    # the snapshot at scale 1.
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

    # (b) Refusals must stay ahead of the first mutation. The gate's write gives
    # the bound mesh a local rotation; under a non-uniformly scaled PARENT that is
    # check_scale_normalizable's shear case, which normalize re-validates. Baking
    # first evaluates that guard before the rotation exists and after the parent
    # is already uniform, so the scene is correctly accepted — measured, the
    # reordered run raises 'sheared' instead, with the clear having already
    # mutated the scene and no file written.
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

    if FAILURES:
        print("FBXEXPORT_TEST FAIL:", "; ".join(FAILURES))
        sys.exit(1)
    print("FBXEXPORT_TEST OK")
    sys.exit(0)


main()
