"""Synthetic headless test for avatarprep.core.proportions.

Run:
  blender --background --factory-startup --python tests/test_proportions.py

Prints PROP_TEST OK and exits 0 on success; PROP_TEST FAIL: <reason> exit 1 otherwise.
"""
import math
import os
import sys

import bpy
import mathutils
from mathutils import Vector

FAILURES = []

def check(cond, msg):
    if not cond:
        FAILURES.append(msg)

def expect_raises(fn, substr, label):
    try:
        fn()
    except Exception as e:
        if substr.lower() not in str(e).lower():
            FAILURES.append("%s: raised but message %r lacked %r" % (label, str(e), substr))
        return
    FAILURES.append("%s: expected an exception mentioning %r, none raised" % (label, substr))

def _add_repo_root_to_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

def _clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def test_load_edge():
    from avatarprep.core.proportions import load_edge, EdgeError
    good = {"source": "a", "target": "b", "source_base": "base",
            "object": {"pivot": "bbox_center", "scale": 0.99, "translate": [0, 0.03, 0.01]},
            "no_inherit_scale": ["Head"],
            "scales": [{"bones": ["Spine"], "value": [1.05, 1.0, 1.03]}],
            "shapekeys": {"Big": 0.2}}
    e = load_edge(good)
    check(e["scales"][0]["space"] == "local", "default space should be local")
    check(e["scales"][0]["pivot"] == "individual", "default pivot should be individual")
    check(e["source_base"] == "base", "source_base should round-trip")
    check(e["target_base"] == "base", "target_base should default to source_base")

    expect_raises(lambda: load_edge({"target": "b"}), "source", "missing source")
    expect_raises(lambda: load_edge({"source": "a"}), "target", "missing target")
    expect_raises(lambda: load_edge({"source": "a", "target": "b"}),
                  "source_base", "missing source_base")
    # origin and bbox_center are both valid object pivots; anything else raises.
    check(load_edge({"source": "a", "target": "b", "source_base": "base",
                     "object": {"pivot": "origin", "scale": 1.0}})
          ["object"]["pivot"] == "origin", "origin pivot should be accepted")
    check(load_edge({"source": "a", "target": "b", "source_base": "base",
                     "object": {"scale": 1.0}})
          ["object"]["pivot"] == "origin", "object pivot should default to origin")
    expect_raises(lambda: load_edge({"source": "a", "target": "b", "source_base": "base",
                  "object": {"pivot": "world", "scale": 1.0}}), "pivot", "bad object pivot")
    expect_raises(lambda: load_edge({"source": "a", "target": "b", "source_base": "base",
                  "object": {"scale": 0}}), "degenerate", "zero object scale")
    expect_raises(lambda: load_edge({"source": "a", "target": "b", "source_base": "base",
                  "scales": [{"bones": ["X"], "value": [1, 1, 1], "pivot": "world"}]}),
                  "pivot", "bad scale pivot")
    expect_raises(lambda: load_edge({"source": "a", "target": "b", "source_base": "base",
                  "scales": [{"bones": ["X"], "value": [1, 0, 1]}]}), "degenerate", "zero scale")
    expect_raises(lambda: load_edge({"source": "a", "target": "b", "source_base": "base",
                  "scales": [{"bones": ["X"], "value": [1, 1, 1], "rotate": [1, 0, 0]}]}),
                  "unknown", "rotation key rejected")
    expect_raises(lambda: load_edge({"source": "a", "target": "b", "source_base": "base",
                  "scales": [{"bones": ["X"], "value": [1, 1, 1], "space": "world"}]}),
                  "space", "bad space")


def _make_arm(name="Armature", bones=(("Root", (0,0,0), (0,0,0.1)),)):
    arm_data = bpy.data.armatures.new(name + "Data")
    arm = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    from avatarprep.core import scene_utils
    ctx = {'active_object': arm, 'object': arm}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    ebs = arm.data.edit_bones
    for bn, head, tail in bones:
        b = ebs.new(bn)
        b.head = Vector(head); b.tail = Vector(tail)
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')
    return arm

def _make_mesh(arm, name="Body", groups=("Root",), shapekeys=()):
    md = bpy.data.meshes.new(name + "Data")
    md.from_pydata([(-0.05,-0.05,0.05),(0.05,-0.05,0.05),(0.05,0.05,0.05),(-0.05,0.05,0.05)],
                   [], [(0,1,2,3)])
    md.update()
    ob = bpy.data.objects.new(name, md)
    bpy.context.collection.objects.link(ob)
    for g in groups:
        vg = ob.vertex_groups.new(name=g)
        vg.add([0,1,2,3], 1.0, 'REPLACE')
    if shapekeys:
        ob.shape_key_add(name="Basis")
        for sk in shapekeys:
            ob.shape_key_add(name=sk)
    mod = ob.modifiers.new("Armature", 'ARMATURE'); mod.object = arm
    ob.parent = arm
    return ob

def test_validate():
    from avatarprep.core import proportions as P
    arm = _make_arm(bones=(("Root",(0,0,0),(0,0,0.1)),("Spine",(0,0,0.1),(0,0,0.3))))
    mesh = _make_mesh(arm, groups=("Root","Spine"), shapekeys=("Big",))
    arm["avatarprep_base"] = "a"
    edge = P.load_edge({"source":"a","target":"b","source_base":"a",
        "scales":[{"bones":["Spine"],"value":[1.1,1,1]},{"bones":["Ghost"],"value":[1.1,1,1]}],
        "shapekeys":{"Big":0.2,"Missing":0.2}})
    rep = P.validate_proportion_edge(arm, [mesh], edge, bone_overrides={}, shapekey_overrides={})
    joined = " ".join(rep["offenders"])
    check("Ghost" in joined, "missing bone Ghost should be an offender")
    check("Missing" in joined, "missing shapekey should be an offender")
    check("Spine" not in joined, "present bone Spine must not be an offender")
    rep2 = P.validate_proportion_edge(arm, [mesh], edge,
        bone_overrides={"Ghost":"Spine"}, shapekey_overrides={"Missing": None})
    check(not rep2["offenders"], "overrides should clear offenders: %r" % rep2["offenders"])
    # skip_shapekeys suppresses the missing-shapekey offender (body edge onto an outfit)
    rep_skip = P.validate_proportion_edge(arm, [mesh], edge,
        bone_overrides={"Ghost":"Spine"}, skip_shapekeys=True)
    check(not any("shapekey" in o for o in rep_skip["offenders"]),
          "skip_shapekeys should suppress missing-shapekey offenders: %r" % rep_skip["offenders"])
    arm["avatarprep_state"] = "wrong"
    rep3 = P.validate_proportion_edge(arm, [mesh], edge,
        bone_overrides={"Ghost":"Spine"}, shapekey_overrides={"Missing": None})
    check(any("state" in o.lower() for o in rep3["offenders"]), "state mismatch should be an offender")
    # A rig at the reserved 'unproportioned' origin validates clean against an
    # unproportioned-source edge (exact match); base must also match source_base.
    arm["avatarprep_base"] = "a"
    arm["avatarprep_state"] = "unproportioned"
    edge_u = dict(edge); edge_u["source"] = "unproportioned"; edge_u["source_base"] = "a"
    rep_u = P.validate_proportion_edge(arm, [mesh], P.load_edge(edge_u), skip_shapekeys=True)
    check(not any("state" in o.lower() or "base" in o.lower() for o in rep_u["offenders"]),
          "unproportioned+matching-base must not offend: %r" % rep_u["offenders"])

    # A named-source-state edge on an 'unproportioned' rig now OFFENDS (wildcard removed).
    edge_named = dict(edge); edge_named["source"] = "custom"; edge_named["source_base"] = "a"
    rep_named = P.validate_proportion_edge(arm, [mesh], P.load_edge(edge_named), skip_shapekeys=True)
    check(any("state mismatch" in o.lower() for o in rep_named["offenders"]),
          "named-source on unproportioned rig must offend: %r" % rep_named["offenders"])

    # base absent -> offender.
    del arm["avatarprep_base"]
    rep_nobase = P.validate_proportion_edge(arm, [mesh], P.load_edge(edge_u), skip_shapekeys=True)
    check(any("base absent" in o.lower() for o in rep_nobase["offenders"]),
          "base-absent must offend: %r" % rep_nobase["offenders"])
    # base corrupt (present but not a str) -> distinct offender, not conflated with 'mismatch'.
    arm["avatarprep_base"] = 123
    rep_badbase = P.validate_proportion_edge(arm, [mesh], P.load_edge(edge_u), skip_shapekeys=True)
    check(any("base corrupt" in o.lower() for o in rep_badbase["offenders"]),
          "base-corrupt must offend distinctly: %r" % rep_badbase["offenders"])
    del arm["avatarprep_base"]
    # A rig left at the mid-apply sentinel (a crashed apply_proportion_edge) hard-FAILs distinctly.
    from avatarprep.core import scene_utils
    arm["avatarprep_state"] = scene_utils.STATE_APPLYING
    rep_int = P.validate_proportion_edge(arm, [mesh], edge,
        bone_overrides={"Ghost":"Spine"}, shapekey_overrides={"Missing": None})
    check(any("interrupted" in o.lower() for o in rep_int["offenders"]),
          "mid-apply sentinel should be an 'interrupted' offender: %r" % rep_int["offenders"])
    edge_med = P.load_edge({"source":"a","target":"b","source_base":"a",
        "scales":[{"bones":["Spine"],"value":[1.1,1,1],"pivot":"median"}], "shapekeys":{}})
    del arm["avatarprep_state"]
    rep_med = P.validate_proportion_edge(arm, [mesh], edge_med)
    check(any("median" in o for o in rep_med["offenders"]), "median pivot with 1 bone should offend")
    # this arm has two parentless bones (Root, Spine); an object edge must offend pre-mutation
    edge_obj = P.load_edge({"source":"a","target":"b","source_base":"a","object":{"scale":1.1}})
    rep_obj = P.validate_proportion_edge(arm, [mesh], edge_obj, skip_shapekeys=True)
    check(any("root bone" in o for o in rep_obj["offenders"]),
          "object edge on a multi-root armature should offend: %r" % rep_obj["offenders"])


def test_local_scale():
    from avatarprep.core import proportions as P
    arm = _make_arm(bones=(("UpperLeg.L",(0.1,0,0.5),(0.1,0,0.1)),
                           ("UpperLeg.R",(-0.1,0,0.5),(-0.1,0,0.1))))
    _make_mesh(arm, groups=("UpperLeg.L","UpperLeg.R"))
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    P.apply_local_scale(arm.pose.bones["UpperLeg.L"], [1.0, 1.5, 1.0])
    P.apply_local_scale(arm.pose.bones["UpperLeg.R"], [1.0, 1.5, 1.0])
    bpy.context.view_layer.update()
    check(abs(arm.pose.bones["UpperLeg.L"].scale.y - 1.5) < 1e-6, "L scale.y should be 1.5")
    check(abs(arm.pose.bones["UpperLeg.R"].scale.y - 1.5) < 1e-6, "R scale.y should be 1.5")
    bpy.ops.object.mode_set(mode='OBJECT')


def test_framed_scale():
    from avatarprep.core import proportions as P
    arm = _make_arm(bones=(("Breast.L",(0.1,0,1.0),(0.1,-0.1,1.0)),
                           ("Breast.R",(-0.1,0,1.0),(-0.1,-0.1,1.0))))
    _make_mesh(arm, groups=("Breast.L","Breast.R"))
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    pbs = [arm.pose.bones["Breast.L"], arm.pose.bones["Breast.R"]]
    xl0 = abs(pbs[0].head.x); xr0 = abs(pbs[1].head.x)
    P.apply_framed_scale(arm, pbs, [0.9, 1.0, 1.0], space="normal", pivot="median")
    bpy.context.view_layer.update()
    xl1 = abs(arm.pose.bones["Breast.L"].head.x)
    check(xl1 < xl0, "median scale 0.9 should pull breast head toward midline (%f -> %f)" % (xl0, xl1))
    check(abs(xl1 - 0.9 * xl0) < 1e-3, "x-distance should shrink ~10%%: %f vs %f" % (xl1, 0.9*xl0))
    bpy.ops.object.mode_set(mode='OBJECT')


def test_object_transform():
    from avatarprep.core import proportions as P
    arm = _make_arm(bones=(("Hips",(0,0,0.5),(0,0,0.6)),("Spine",(0,0,0.6),(0,0,0.8))))
    bpy.context.view_layer.objects.active = arm
    from avatarprep.core import scene_utils
    ctx = {'active_object': arm, 'object': arm}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    arm.data.edit_bones["Spine"].parent = arm.data.edit_bones["Hips"]
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')
    mesh = _make_mesh(arm, groups=("Spine",))
    bpy.ops.object.mode_set(mode='POSE')
    P.pose_object_transform(arm, [mesh], 2.0, [0.0, 0.0, 0.0])
    bpy.context.view_layer.update()
    check(arm.pose.bones["Hips"].matrix.to_scale().x > 1.9, "root scale should be ~2x")
    bpy.ops.object.mode_set(mode='OBJECT')


def test_shapekeys():
    from avatarprep.core import proportions as P
    arm = _make_arm()
    mesh = _make_mesh(arm, shapekeys=("Breasts_Big",))
    rep = P.apply_shapekeys([mesh], {"Breasts_Big": -0.2})
    kb = mesh.data.shape_keys.key_blocks["Breasts_Big"]
    check(abs(kb.value + 0.2) < 1e-6, "value should be -0.2")
    check(kb.slider_min <= -0.2, "slider_min should widen to <= -0.2")
    check(any(r["widened"] for r in rep), "report should note widening")


def test_apply_proportion_edge():
    from avatarprep.core import proportions as P
    arm = _make_arm(bones=(("Hips",(0,0,0.5),(0,0,0.6)),("Spine",(0,0,0.6),(0,0,0.9))))
    bpy.context.view_layer.objects.active = arm
    from avatarprep.core import scene_utils
    ctx = {'active_object': arm, 'object': arm}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    arm.data.edit_bones["Spine"].parent = arm.data.edit_bones["Hips"]
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')
    mesh = _make_mesh(arm, groups=("Spine",), shapekeys=("Big",))
    arm["avatarprep_base"] = "a"
    z0 = (mesh.matrix_world @ mathutils.Vector(mesh.data.vertices[0].co)).z
    edge = {"source":"unproportioned","target":"custom","source_base":"a",
            "object":{"pivot":"bbox_center","scale":2.0,"translate":[0,0,0]},
            "no_inherit_scale":["Spine"],
            "scales":[{"bones":["Spine"],"value":[1.0,1.5,1.0]}],
            "shapekeys":{"Big":0.5}}
    rep = P.apply_proportion_edge(arm, [mesh], edge)
    check(rep["state"] == "custom", "state should be stamped 'custom'")
    z1 = (mesh.matrix_world @ mathutils.Vector(mesh.data.vertices[0].co)).z
    check(abs(z1 - z0) > 1e-4, "geometry should have moved")
    check(mesh.data.shape_keys.key_blocks["Big"].value == 0.5, "shapekey value set")
    raised = []
    try:
        P.apply_proportion_edge(arm, [mesh], edge)
    except Exception as e:
        raised.append(str(e))
    check(raised and "state" in raised[0].lower(), "re-apply should fail on state mismatch")


def test_apply_proportion_edge_skip_shapekeys():
    # Covers skip_shapekeys (a body edge applied to a mesh lacking its shape keys)
    # AND a baked geometric check that an origin-pivot object scale maps z -> 2z.
    from avatarprep.core import proportions as P
    arm = _make_arm(bones=(("Hips",(0,0,0.0),(0,0,0.6)),))
    mesh = _make_mesh(arm, groups=("Hips",))   # NO shape keys
    arm["avatarprep_base"] = "a"
    z0 = (mesh.matrix_world @ mathutils.Vector(mesh.data.vertices[0].co)).z
    edge = {"source":"unproportioned","target":"custom","source_base":"a",
            "object":{"pivot":"origin","scale":2.0,"translate":[0,0,0]},
            "shapekeys":{"Big":0.5}}           # mesh lacks 'Big'
    raised = []
    try:
        P.apply_proportion_edge(arm, [mesh], edge)     # aborts before any mutation
    except Exception as e:
        raised.append(str(e))
    check(raised and "shapekey" in raised[0].lower(), "missing shapekey should abort without skip")
    rep = P.apply_proportion_edge(arm, [mesh], edge, skip_shapekeys=True)
    check(rep["state"] == "custom", "skip_shapekeys run should complete and stamp target")
    z1 = (mesh.matrix_world @ mathutils.Vector(mesh.data.vertices[0].co)).z
    check(abs(z1 - 2.0 * z0) < 1e-3,
          "origin scale 2x should map z=%.3f to ~%.3f, got %.3f" % (z0, 2 * z0, z1))

def test_apply_proportion_edge_median():
    from avatarprep.core import proportions as P
    arm = _make_arm(bones=(("Breast.L",(0.1,0,1.0),(0.1,-0.1,1.0)),
                           ("Breast.R",(-0.1,0,1.0),(-0.1,-0.1,1.0))))
    mesh = _make_mesh(arm, groups=("Breast.L","Breast.R"))
    arm["avatarprep_base"] = "a"
    edge = {"source":"unproportioned","target":"custom","source_base":"a",
            "scales":[{"bones":["Breast.L","Breast.R"],"value":[1.4,1.4,1.4],
                       "space":"normal","pivot":"median"}]}
    # value is uniform 1.4, so vertex distance from the median pivot must scale x1.4
    # (frame-independent) -- a geometric check that catches a pivot/scale regression.
    piv = mathutils.Vector((0.0, 0.0, 1.0))   # median of the two breast bone heads
    d0 = ((mesh.matrix_world @ mathutils.Vector(mesh.data.vertices[0].co)) - piv).length
    rep = P.apply_proportion_edge(arm, [mesh], edge)
    check(rep["state"] == "custom", "median path should stamp target")
    check(rep["scales_applied"] == 1 and len(rep["bakes"]) == 1,
          "median path should record one scale + one bake")
    d1 = ((mesh.matrix_world @ mathutils.Vector(mesh.data.vertices[0].co)) - piv).length
    check(abs(d1 - 1.4 * d0) < 1e-3,
          "uniform 1.4 about median pivot should scale vertex dist %.4f -> ~%.4f, got %.4f"
          % (d0, 1.4 * d0, d1))

    # apply transitions (base, state): reproportion keeps base, equivalency moves it.
    arm["avatarprep_base"] = "shinano"; arm["avatarprep_state"] = "unproportioned"
    repro = {"source": "unproportioned", "target": "custom",
             "source_base": "shinano", "target_base": "shinano"}
    r1 = P.apply_proportion_edge(arm, [mesh], repro, skip_shapekeys=True)
    check(arm["avatarprep_base"] == "shinano" and arm["avatarprep_state"] == "custom",
          "reproportion: base kept, state=custom; got (%r,%r)"
          % (arm.get("avatarprep_base"), arm.get("avatarprep_state")))

    arm["avatarprep_state"] = "unproportioned"
    equiv = {"source": "unproportioned", "target": "unproportioned",
             "source_base": "shinano", "target_base": "chiffon"}
    r2 = P.apply_proportion_edge(arm, [mesh], equiv, skip_shapekeys=True)
    check(arm["avatarprep_base"] == "chiffon",
          "equivalency: base moved to chiffon; got %r" % arm.get("avatarprep_base"))


def test_baked_coupling():
    from avatarprep.core import proportions as P
    arm = _make_arm()
    mesh = _make_mesh(arm, shapekeys=("Big",))
    mesh["avatarprep_baked"] = {"Big": 0.5}   # pretend 'Big' is already baked
    arm["avatarprep_base"] = "shinano"; arm["avatarprep_state"] = "unproportioned"
    edge_bk = {"source": "unproportioned", "target": "custom",
               "source_base": "shinano", "target_base": "shinano",
               "shapekeys": {"Big": 0.3}}
    rep_bk = P.validate_proportion_edge(arm, [mesh], P.load_edge(edge_bk))
    check(any("already baked" in w.lower() for w in rep_bk["warnings"]),
          "driving an already-baked key must warn: %r" % rep_bk["warnings"])
    # a key with no baked entry must NOT produce the warning
    arm2 = _make_arm()
    mesh2 = _make_mesh(arm2, shapekeys=("Big",))
    arm2["avatarprep_base"] = "shinano"; arm2["avatarprep_state"] = "unproportioned"
    edge_ok = {"source": "unproportioned", "target": "custom",
               "source_base": "shinano", "target_base": "shinano",
               "shapekeys": {"Big": 0.3}}
    rep_ok = P.validate_proportion_edge(arm2, [mesh2], P.load_edge(edge_ok))
    check(not any("already baked" in w.lower() for w in rep_ok["warnings"]),
          "unbaked key must not warn: %r" % rep_ok["warnings"])



# --- staged --whatif geometry (R8) ---------------------------------------------

def _sk_with_delta(ob, name, dz):
    """Add a shape key that actually MOVES geometry. ``shape_key_add`` alone makes a
    zero-delta key, which cannot exercise a measurement."""
    if ob.data.shape_keys is None:
        ob.shape_key_add(name="Basis")
    kb = ob.shape_key_add(name=name)
    for p in kb.data:
        p.co = p.co + Vector((0.0, 0.0, dz))
    kb.slider_min = min(kb.slider_min, 0.0)
    kb.slider_max = max(kb.slider_max, 1.0)
    # shape_key_add returns the block ALREADY at value 1.0 (measured on 5.2.0), so a
    # morph left as-is is live before the edge touches it -- and an edge driving it to
    # 1.0 would then move nothing, making the stage look inert when it is not.
    kb.value = 0.0
    return kb


def _rig_for_measure(no_inherit=False):
    """Root -> Spine, genuinely parented so the edge's object transform has exactly one
    root bone to drive, plus a bound mesh with a bounds-moving morph."""
    from avatarprep.core import scene_utils
    arm = _make_arm(bones=(("Root", (0, 0, 0), (0, 0, 0.1)),
                           ("Spine", (0, 0, 0.1), (0, 0, 0.3))))
    with scene_utils.edit_mode(arm) as ebs:
        ebs["Spine"].parent = ebs["Root"]
        if no_inherit:
            ebs["Spine"].inherit_scale = 'NONE'
    mesh = _make_mesh(arm, groups=("Root", "Spine"))
    _sk_with_delta(mesh, "Tall", 0.25)
    arm["avatarprep_base"] = "a"
    arm["avatarprep_state"] = "s0"
    return arm, mesh


def test_measure_geometry_skips_empty_meshes():
    """bound_box on a zero-vertex mesh is eight all-zero corners, which would inject
    world z=0 as if it were geometry. measure_geometry reads evaluated vertices and
    skips such a mesh; this pins that it contributes nothing."""
    _clear_scene()
    from avatarprep.core import measure
    arm, mesh = _rig_for_measure()
    empty = bpy.data.objects.new("Empty", bpy.data.meshes.new("EmptyData"))
    bpy.context.collection.objects.link(empty)
    empty.parent = arm
    m = measure.measure_geometry(arm, [mesh, empty])
    check(m["per_mesh"]["Empty"] is None, "a zero-vertex mesh should map to None")
    check(m["aggregate"]["min"][2] > 0.01,
          "an empty mesh must not drag aggregate min_z to the origin, got %r"
          % m["aggregate"]["min"][2])
    check(abs(m["aggregate"]["min"][2] - m["per_mesh"]["Body"]["min"][2]) < 1e-9,
          "aggregate should equal the only real mesh's bounds")
    # No mesh with vertices at all -> no aggregate, rather than a fabricated zero box.
    m2 = measure.measure_geometry(arm, [empty])
    check(m2["aggregate"] is None, "no measurable mesh should give aggregate None")


def test_stage_hook_order():
    _clear_scene()
    from avatarprep.core import proportions as P
    arm, mesh = _rig_for_measure()
    seen = []
    P.apply_proportion_edge(arm, [mesh], {
        "source": "s0", "target": "s1", "source_base": "a",
        "object": {"scale": 1.0, "translate": [0, 0, 0.05]},
        "scales": [{"bones": ["Spine"], "value": [1.0, 1.2, 1.0]}],
        "shapekeys": {"Tall": 1.0}}, stage_hook=lambda n: seen.append(n))
    check(seen == ["pre", "object", "scales", "shapekeys"],
          "stage hook should fire once per stage in order, got %s" % seen)

    # A stage the edge does not have must not fire.
    _clear_scene()
    arm, mesh = _rig_for_measure()
    seen2 = []
    P.apply_proportion_edge(arm, [mesh], {
        "source": "s0", "target": "s1", "source_base": "a",
        "scales": [{"bones": ["Spine"], "value": [1.0, 1.2, 1.0]}]},
        skip_shapekeys=True, stage_hook=lambda n: seen2.append(n))
    check(seen2 == ["pre", "scales"],
          "absent stages must not fire, got %s" % seen2)


def test_whatif_geometry_equals_real_apply():
    """The claim the whole feature rests on: numbers measured by the in-memory trial
    equal the numbers a real apply produces. Uses an edge with a NON-EMPTY shapekeys
    block whose morph reaches the bounds -- both shipped longlimb edges carry no
    shapekeys at all, so an edge without one cannot validate this stage."""
    from avatarprep.core import proportions as P, measure
    edge = {"source": "s0", "target": "s1", "source_base": "a",
            "object": {"scale": 1.0, "translate": [0, 0, 0.05]},
            "scales": [{"bones": ["Spine"], "value": [1.0, 1.3, 1.0]}],
            "shapekeys": {"Tall": 1.0}}

    _clear_scene()
    arm, mesh = _rig_for_measure()
    stages = []
    P.apply_proportion_edge(arm, [mesh], dict(edge),
                            stage_hook=lambda n: stages.append(
                                (n, measure.measure_geometry(arm, [mesh]))))
    trial_final = stages[-1][1]
    check([n for n, _ in stages] == ["pre", "object", "scales", "shapekeys"],
          "all four stages should have measured")
    check(stages[-1][1]["aggregate"]["max"][2] > stages[-2][1]["aggregate"]["max"][2],
          "the shapekeys stage must actually move the bounds, or it proves nothing")

    _clear_scene()
    arm2, mesh2 = _rig_for_measure()
    P.apply_proportion_edge(arm2, [mesh2], dict(edge))
    real_final = measure.measure_geometry(arm2, [mesh2])

    for key in ("min", "max"):
        for i in range(3):
            a = trial_final["aggregate"][key][i]
            b = real_final["aggregate"][key][i]
            check(abs(a - b) < 1e-9,
                  "trial aggregate %s[%d] %.12f != real %.12f" % (key, i, a, b))
    for bn, a in trial_final["bones"].items():
        b = real_final["bones"][bn]
        check(abs(a["length"] - b["length"]) < 1e-9,
              "trial bone %s length %.12f != real %.12f" % (bn, a["length"], b["length"]))


def test_collateral_lengths():
    """The spread report: a child that inherits its parent's scale changes length
    without being named, and inherit_scale NONE is supposed to stop it. Nothing before
    this proved either."""
    from avatarprep.core import proportions as P, measure
    edge = {"source": "s0", "target": "s1", "source_base": "a",
            "scales": [{"bones": ["Root"], "value": [1.0, 1.5, 1.0]}]}

    _clear_scene()
    arm, mesh = _rig_for_measure()
    pre = measure.measure_geometry(arm, [mesh])
    P.apply_proportion_edge(arm, [mesh], dict(edge), skip_shapekeys=True)
    post = measure.measure_geometry(arm, [mesh])
    spread = measure.collateral_lengths(pre, post, {"Root"})
    check([r["bone"] for r in spread] == ["Spine"],
          "Spine should inherit Root's scale and be reported, got %s"
          % [r["bone"] for r in spread])

    _clear_scene()
    arm2, mesh2 = _rig_for_measure(no_inherit=True)
    pre2 = measure.measure_geometry(arm2, [mesh2])
    P.apply_proportion_edge(arm2, [mesh2], dict(edge), skip_shapekeys=True)
    post2 = measure.measure_geometry(arm2, [mesh2])
    spread2 = measure.collateral_lengths(pre2, post2, {"Root"})
    check(spread2 == [],
          "inherit_scale NONE should stop the spread, got %s"
          % [r["bone"] for r in spread2])

    named = measure.bone_length_deltas(pre, post, ["Root"])
    # Bone coords are float32, so the achieved percentage lands within ~1e-5 of nominal,
    # not on it. Tightening this past float32 would make the suite flaky, not stricter.
    check(len(named) == 1 and abs(named[0]["pct"] - 50.0) < 1e-3,
          "the named bone's achieved pct should be +50%%, got %s" % named)
    check(measure.bone_length_deltas(pre, post, ["NoSuchBone"]) == [],
          "an absent bone should be skipped, not faked")


def test_local_scale_op_does_not_compound_down_a_chain():
    """One op naming a whole chain lands its value once per bone, as Blender's resize on
    a selected chain does — not compounded through inheritance. Measured on a hand-made
    reference: every finger segment at 1.02, where the naive apply gave 1.02/1.04/1.06."""
    from avatarprep.core import proportions as P, scene_utils
    chain = (("Root", (0, 0, 0), (0, 0, 0.1)),
             ("Mid", (0, 0, 0.1), (0, 0, 0.2)),
             ("Tip", (0, 0, 0.2), (0, 0, 0.3)))
    edge = {"source": "s0", "target": "s1", "source_base": "a",
            "scales": [{"bones": ["Root", "Mid", "Tip"], "value": [1.0, 1.5, 1.0]}]}

    def build(tip_mode='FULL', mid_mode='FULL'):
        _clear_scene()
        arm = _make_arm(bones=chain)
        with scene_utils.edit_mode(arm) as ebs:
            ebs["Mid"].parent = ebs["Root"]; ebs["Mid"].use_connect = True
            ebs["Tip"].parent = ebs["Mid"]; ebs["Tip"].use_connect = True
            ebs["Mid"].inherit_scale = mid_mode
            ebs["Tip"].inherit_scale = tip_mode
        mesh = _make_mesh(arm, groups=("Root", "Mid", "Tip"))
        arm["avatarprep_base"] = "a"; arm["avatarprep_state"] = "s0"
        return arm, mesh

    # Every inherit mode Blender offers, on the tip and on the middle link: the vector
    # modes carry the parent's scale, AVERAGE carries only its uniform volume-equivalent,
    # and the two NONE modes carry nothing. In all of them each bone must land at 1.5x once.
    modes = ('FULL', 'FIX_SHEAR', 'ALIGNED', 'AVERAGE', 'NONE', 'NONE_LEGACY')
    for tip_mode in modes:
        for mid_mode in ('FULL', 'AVERAGE', 'NONE_LEGACY'):
            arm, mesh = build(tip_mode, mid_mode)
            P.apply_proportion_edge(arm, [mesh], dict(edge), skip_shapekeys=True)
            lens = [arm.data.bones[n].length for n in ("Root", "Mid", "Tip")]
            check(all(abs(l - 0.15) < 1e-5 for l in lens),
                  "tip=%s mid=%s: every bone should be 1.5x once, got %s"
                  % (tip_mode, mid_mode, lens))

    # Separate ops compound, as separate manual resizes do.
    arm, mesh = build()
    two = dict(edge); two["scales"] = [{"bones": ["Root"], "value": [1.0, 1.5, 1.0]},
                                       {"bones": ["Mid"], "value": [1.0, 1.5, 1.0]}]
    P.apply_proportion_edge(arm, [mesh], two, skip_shapekeys=True)
    check(abs(arm.data.bones["Mid"].length - 0.225) < 1e-5,
          "a second op on the child should compound on the first, got %s"
          % arm.data.bones["Mid"].length)


def test_cli_whatif_writes_nothing_and_reports_geometry():
    """Drive the door. The byte-identity assertion is the guard that an in-memory trial
    can never become a write, however the code around it is refactored later."""
    import hashlib
    import json
    import subprocess
    import tempfile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    door = os.path.join(root, "cli", "apply_proportion_edge.py")
    tmp = tempfile.mkdtemp(prefix="avatarprep_whatif_")
    src = os.path.join(tmp, "src.blend")
    edge_path = os.path.join(tmp, "edge.json")
    bad_edge_path = os.path.join(tmp, "bad_edge.json")

    _clear_scene()
    _rig_for_measure()
    bpy.ops.wm.save_as_mainfile(filepath=src)
    digest = hashlib.sha256(open(src, "rb").read()).hexdigest()

    with open(edge_path, "w", encoding="utf-8") as fh:
        json.dump({"source": "s0", "target": "s1", "source_base": "a",
                   "object": {"scale": 1.0, "translate": [0, 0, 0.05]},
                   "scales": [{"bones": ["Spine"], "value": [1.0, 1.3, 1.0]}],
                   "shapekeys": {"Tall": 1.0}}, fh)
    # A real state mismatch: the rig is stamped s0, this edge demands sX.
    with open(bad_edge_path, "w", encoding="utf-8") as fh:
        json.dump({"source": "sX", "target": "s1", "source_base": "a",
                   "scales": [{"bones": ["Spine"], "value": [1.0, 1.3, 1.0]}]}, fh)

    def run(extra):
        proc = subprocess.run([bpy.app.binary_path, "--background", "--factory-startup",
                               "--python", door, "--", "--in", src] + extra,
                              capture_output=True, text=True, timeout=600)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    report = os.path.join(tmp, "r.json")
    rc, out = run(["--edge", edge_path, "--whatif", "--report", report])
    check(rc == 0, "clean whatif should exit 0, got %s (%r)" % (rc, out[-400:]))
    check(hashlib.sha256(open(src, "rb").read()).hexdigest() == digest,
          "--whatif must leave --in BYTE-IDENTICAL")
    check(not os.path.exists(os.path.join(tmp, "src.blend1")),
          "--whatif must not leave a Blender backup beside --in")

    data = json.load(open(report, encoding="utf-8"))
    check("geometry" in data, "--report should carry a geometry block")
    check("offenders" in data and "warnings" in data,
          "the pre-existing report shape must survive")
    geo = data["geometry"]
    check([s["stage"] for s in geo["stages"]] == ["pre", "object", "scales", "shapekeys"],
          "stages should be present and ordered, got %s"
          % [s["stage"] for s in geo["stages"]])
    check(geo["stages"][0].get("delta_from_previous") is None,
          "the pre stage has no previous stage to differ from")
    for s in geo["stages"][1:]:
        check(isinstance(s["delta_from_previous"]["d_height"], float),
              "each later stage should carry a numeric delta")
    check(geo["bones"]["pre"] and geo["bones"]["post"],
          "both ends' bone positions should be reported")
    check("head" in list(geo["bones"]["pre"].values())[0],
          "bones should carry head/tail so any span is derivable without applying")
    check(geo["scale_ops"] and geo["scale_ops"][0]["lengths"],
          "per-op achieved bone lengths should be reported")
    check("collateral_lengths" in geo, "the spread report should be present")
    check(geo["loaded_with_repairs"] is None,
          "a clean load should record no repair against these numbers")
    check("whatif stage" in out and "whatif total" in out,
          "the printed tier should carry staged aggregates, got %r" % out[-400:])
    check("per_mesh" not in out,
          "per-mesh detail belongs in --report, not in the printed tier")

    # Offenders must short-circuit: no trial runs on a rig that failed the gate, so no
    # geometry is reported and nothing claims to know what the edge would do.
    report2 = os.path.join(tmp, "r2.json")
    rc, out = run(["--edge", bad_edge_path, "--whatif", "--report", report2])
    check(rc == 1, "offenders under whatif should exit 1, got %s" % rc)
    check("OFFENDER" in out, "offenders should be named")
    check("whatif stage" not in out,
          "no geometry may be reported when the gate failed, got %r" % out[-400:])
    data2 = json.load(open(report2, encoding="utf-8"))
    check("geometry" not in data2,
          "a failed gate must not attach a geometry block")

    rc, out = run(["--edge", edge_path, "--whatif", "--out", os.path.join(tmp, "o.blend")])
    check(rc == 2, "--out under --whatif should exit 2, got %s" % rc)


# --- absent bones: a mergeable rig pruned to what it skins ---------------------

_ABSENT_RIG = (("Hips", None, (0, 0, 0.9), (0, 0, 1.0)),
               ("Spine", "Hips", (0, 0, 1.0), (0, 0, 1.3)),
               ("UpperArm.L", "Spine", (0.1, 0, 1.3), (0.35, 0, 1.3)),
               ("LowerArm.L", "UpperArm.L", (0.35, 0, 1.3), (0.6, 0, 1.3)),
               ("Hand.L", "LowerArm.L", (0.6, 0, 1.3), (0.7, 0, 1.3)),
               ("Breast.L", "Spine", (0.08, 0, 1.15), (0.08, -0.1, 1.15)),
               ("Breast.R", "Spine", (-0.08, 0, 1.15), (-0.08, -0.1, 1.15)),
               ("UpperLeg.L", "Hips", (0.1, 0, 0.9), (0.1, 0, 0.5)),
               ("Toe.L", "UpperLeg.L", (0.1, 0, 0.5), (0.1, -0.1, 0.5)))

# scales[0] partial local/individual once Hand.L is gone; scales[1] all-absent once
# Toe.L is gone; Toe.L in no_inherit_scale is dropped with it.
_ABSENT_EDGE = {"source": "s0", "target": "s1", "source_base": "a",
                "no_inherit_scale": ["UpperArm.L", "Toe.L"],
                "scales": [{"bones": ["UpperArm.L", "LowerArm.L", "Hand.L"],
                            "value": [1.0, 1.2, 1.1]},
                           {"bones": ["Toe.L"], "value": [1.3, 1.3, 1.3]},
                           {"bones": ["UpperLeg.L"], "value": [1.0, 1.1, 1.0]}]}


def _absent_rig(delete=()):
    """The full rig, minus ``delete`` (leaf bones only, so nothing present loses an
    ancestor), with a mesh skinned to bones present in every variant."""
    from avatarprep.core import scene_utils
    _clear_scene()
    arm = _make_arm(bones=[(n, h, t) for n, _, h, t in _ABSENT_RIG])
    with scene_utils.edit_mode(arm) as ebs:
        for n, parent, _, _ in _ABSENT_RIG:
            if parent:
                ebs[n].parent = ebs[parent]
        for n in delete:
            check(not ebs[n].children, "fixture deletes leaf bones only, %r has children" % n)
            ebs.remove(ebs[n])
    md = bpy.data.meshes.new("SleeveData")
    verts = [(0.2, 0, 1.3), (0.45, 0, 1.3), (0.55, 0.02, 1.32), (0.1, 0, 0.7), (0, 0, 1.2)]
    md.from_pydata(verts, [], [])
    md.update()
    ob = bpy.data.objects.new("Sleeve", md)
    bpy.context.collection.objects.link(ob)
    for g, idx in (("UpperArm.L", [0]), ("LowerArm.L", [1, 2]), ("UpperLeg.L", [3]),
                   ("Spine", [4])):
        ob.vertex_groups.new(name=g).add(idx, 1.0, 'REPLACE')
    ob.modifiers.new("Armature", 'ARMATURE').object = arm
    ob.parent = arm
    arm["avatarprep_base"] = "a"
    arm["avatarprep_state"] = "s0"
    return arm, ob


def _absent_geometry(arm, mesh):
    bones = {b.name: (tuple(b.head_local), tuple(b.tail_local)) for b in arm.data.bones}
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    verts = [tuple(mesh.matrix_world @ v.co) for v in ev.data.vertices]
    return bones, verts


def _close(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_absent_bones_drop():
    from avatarprep.core import proportions as P
    import copy

    # Full rig: the reference the dropped apply must reproduce on the bones it keeps.
    arm, mesh = _absent_rig()
    rep_full = P.apply_proportion_edge(arm, [mesh], copy.deepcopy(_ABSENT_EDGE),
                                       skip_shapekeys=True)
    full_bones, full_verts = _absent_geometry(arm, mesh)
    check(rep_full["scales_applied"] == 3 and rep_full["dropped"] == [],
          "full rig applies every op and drops nothing: %r" % rep_full["dropped"])

    # Default refuse is unchanged: the absent bones offend, nothing is dropped.
    arm, mesh = _absent_rig(delete=("Hand.L", "Toe.L"))
    val = P.validate_proportion_edge(arm, [mesh], P.load_edge(copy.deepcopy(_ABSENT_EDGE)),
                                     skip_shapekeys=True)
    joined = " ".join(val["offenders"])
    check("'Hand.L'" in joined and "'Toe.L'" in joined and not val["dropped"],
          "refuse must offend on each absent bone and drop nothing: %r" % val["offenders"])
    expect_raises(lambda: P.apply_proportion_edge(arm, [mesh], copy.deepcopy(_ABSENT_EDGE),
                                                  skip_shapekeys=True),
                  "bone not found", "default apply on an absent bone")

    # Drop: the all-absent op and the absent no_inherit_scale entry go, the partial
    # local/individual op applies to its present bones.
    arm, mesh = _absent_rig(delete=("Hand.L", "Toe.L"))
    rep = P.apply_proportion_edge(arm, [mesh], copy.deepcopy(_ABSENT_EDGE),
                                  skip_shapekeys=True, absent_bones="drop")
    got = sorted((d["where"], tuple(d["bones"]), d["whole_op"]) for d in rep["dropped"])
    want = sorted([("scales[0]", ("Hand.L",), False), ("scales[1]", ("Toe.L",), True),
                   ("no_inherit_scale", ("Toe.L",), False)])
    check(got == want, "dropped record: got %r, want %r" % (got, want))
    check(rep["scales_applied"] == 2, "scales_applied counts effective ops, got %d"
          % rep["scales_applied"])
    bones, verts = _absent_geometry(arm, mesh)
    for n, (h, t) in bones.items():
        fh, ft = full_bones[n]
        check(_close(h, fh) and _close(t, ft),
              "present bone %s must match the full-rig apply: %r/%r vs %r/%r"
              % (n, h, t, fh, ft))
    for i, (v, fv) in enumerate(zip(verts, full_verts)):
        check(_close(v, fv), "vertex %d must match the full-rig apply: %r vs %r" % (i, v, fv))
    check(not _close(full_verts[1], (0.45, 0, 1.3)),
          "the fixture must actually move the LowerArm vertex, or the equality proves nothing")

    # A partly-absent median op stays an offender under drop, and so does a framed
    # individual one: both are computed over the whole bone set. The local/median case
    # keeps two present bones, so only the partial-op rule can refuse it.
    for space, pivot, names, gone in (
            ("normal", "median", ["Breast.L", "Breast.R"], "Breast.R"),
            ("normal", "individual", ["Breast.L", "Breast.R"], "Breast.R"),
            ("local", "median", ["Breast.L", "Breast.R", "Toe.L"], "Toe.L")):
        arm, mesh = _absent_rig(delete=(gone,))
        edge = {"source": "s0", "target": "s1", "source_base": "a",
                "scales": [{"bones": names, "value": [1.2, 1.2, 1.2],
                            "space": space, "pivot": pivot}]}
        val = P.validate_proportion_edge(arm, [mesh], P.load_edge(edge),
                                         skip_shapekeys=True, absent_bones="drop")
        check(any(gone in o and "partly-absent" in o for o in val["offenders"])
              and not val["dropped"],
              "partial %s/%s op must refuse under drop: %r" % (space, pivot, val["offenders"]))

    # A --bone-override onto an absent bone is never dropped.
    arm, mesh = _absent_rig(delete=("Hand.L",))
    edge = {"source": "s0", "target": "s1", "source_base": "a",
            "scales": [{"bones": ["Paw.L"], "value": [1.1, 1.1, 1.1]}]}
    val = P.validate_proportion_edge(arm, [mesh], P.load_edge(edge),
                                     bone_overrides={"Paw.L": "Hand.L"},
                                     skip_shapekeys=True, absent_bones="drop")
    check(any("Hand.L" in o and "--bone-override" in o for o in val["offenders"])
          and not val["dropped"],
          "an override onto an absent bone must offend under drop: %r" % val["offenders"])


def test_cli_whatif_reports_dropped():
    import json
    import subprocess
    import tempfile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    door = os.path.join(root, "cli", "apply_proportion_edge.py")
    tmp = tempfile.mkdtemp(prefix="avatarprep_absent_")
    src = os.path.join(tmp, "src.blend")
    edge_path = os.path.join(tmp, "edge.json")
    _absent_rig(delete=("Hand.L", "Toe.L"))
    bpy.ops.wm.save_as_mainfile(filepath=src)
    with open(edge_path, "w", encoding="utf-8") as fh:
        json.dump(_ABSENT_EDGE, fh)

    def run(extra):
        proc = subprocess.run([bpy.app.binary_path, "--background", "--factory-startup",
                               "--python", door, "--", "--in", src, "--edge", edge_path,
                               "--whatif"] + extra,
                              capture_output=True, text=True, timeout=600)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    rc, out = run([])
    check(rc == 1 and "bone not found" in out, "default --whatif must refuse, got %s" % rc)

    report = os.path.join(tmp, "r.json")
    rc, out = run(["--absent-bones", "drop", "--report", report])
    check(rc == 0, "drop --whatif should exit 0, got %s (%r)" % (rc, out[-600:]))
    check("AVATARPREP: DROPPED scales[1] 'Toe.L'" in out
          and "AVATARPREP: DROPPED scales[0] 'Hand.L'" in out
          and "AVATARPREP: DROPPED no_inherit_scale 'Toe.L'" in out,
          "one DROPPED line per dropped op or bone, got %r" % out[-900:])
    data = json.load(open(report, encoding="utf-8"))
    check(len(data.get("dropped", [])) == 3, "report should carry dropped: %r"
          % data.get("dropped"))
    ops = data["geometry"]["scale_ops"]
    check([o["index"] for o in ops] == [0, 2],
          "geometry should list effective ops under their edge index, got %r"
          % [o["index"] for o in ops])
    check(ops and ops[0]["bones"] == ["UpperArm.L", "LowerArm.L"],
          "the partial op should list its present bones only, got %r"
          % (ops[0]["bones"] if ops else None))


def test_bbox_center_skips_empty_meshes():
    """pivot 'bbox_center' scales about the meshes' own centre. A mesh with no geometry
    must contribute nothing: counting it drags that centre toward the object's origin
    and the whole avatar scales about the wrong point. (The centre now comes from
    measure._world_bounds; this pins the skip behaviour across that move.)"""
    from avatarprep.core import proportions as P
    _clear_scene()
    arm = _make_arm()
    mesh = _make_mesh(arm)
    mesh.location = Vector((0, 0, 10))
    solo = P._world_bbox_center([mesh]).copy()

    empty = bpy.data.objects.new("Empty", bpy.data.meshes.new("EmptyData"))
    bpy.context.collection.objects.link(empty)
    both = P._world_bbox_center([mesh, empty])
    check((both - solo).length < 1e-9,
          "an empty mesh moved the bbox centre from %r to %r" % (solo[:], both[:]))

    # Only empty meshes is not a centre at the origin -- it is no centre at all.
    expect_raises(lambda: P._world_bbox_center([empty]), "no mesh geometry",
                  "bbox centre of only empty meshes")


def test_bbox_center_refuses_unevaluated_meshes():
    """The pivot refuses what the measuring doors only report. _world_bounds does not
    skip an unevaluated mesh — it measures it, off unevaluated geometry and a
    matrix_world that never re-evaluated — so a pivot computed from it moves the avatar
    by an unbounded amount. apply_proportion_edge's preflight refuses on the same
    predicate first, so this guard is unreachable on every shipped path; it is pinned
    here because the core is built to be called directly and a second caller must not
    inherit the trap by omission."""
    from avatarprep.core import proportions as P
    _clear_scene()
    arm = _make_arm()
    mesh = _make_mesh(arm)
    hidden = _make_mesh(arm, name="HiddenMesh")
    hidden.hide_viewport = True
    bpy.context.view_layer.update()
    expect_raises(lambda: P._world_bbox_center([mesh, hidden]), "HiddenMesh",
                  "bbox centre with an unevaluated mesh")
    expect_raises(lambda: P._world_bbox_center([mesh, hidden]), "not evaluated",
                  "bbox centre refusal names the predicate")

    # The negative direction, because the tempting predicate gets it wrong: visible_get()
    # reads False for hide_set, which evaluates fine, so a "simplification" of
    # _world_bounds' flag to a visibility test would start refusing working files. This
    # door consumes _world_bounds' own list, so pinning the predicate in test_rest_pose
    # does not cover it. (That suite pins the LAYER-collection eye against the same
    # mutation; one case here kills it at this door too.)
    eyed = _make_mesh(arm, name="EyeHidden")
    eyed.hide_set(True)
    bpy.context.view_layer.update()
    try:
        P._world_bbox_center([mesh, eyed])
    except Exception as e:
        FAILURES.append("hide_set evaluates fine and must not be refused: %s" % e)
    bpy.data.objects.remove(eyed, do_unlink=True)

    # The two cases must not collapse into one another. A VISIBLE zero-vertex mesh is
    # skipped silently and does not raise (test_bbox_center_skips_empty_meshes owns why),
    # so it is visible here deliberately: _world_bounds appends to `unevaluated` BEFORE
    # the zero-vertex skip, which makes a HIDDEN empty mesh refuse instead.
    empty = bpy.data.objects.new("EmptyMesh", bpy.data.meshes.new("EmptyData"))
    bpy.context.collection.objects.link(empty)
    bpy.context.view_layer.update()
    try:
        P._world_bbox_center([mesh, empty])
    except Exception as e:
        FAILURES.append("a visible zero-vertex mesh must be skipped, not refused: %s" % e)

    empty.hide_viewport = True
    bpy.context.view_layer.update()
    expect_raises(lambda: P._world_bbox_center([mesh, empty]), "EmptyMesh",
                  "a hidden empty mesh refuses (unevaluated is checked first)")


def _bounds_height(meshes):
    from avatarprep.core import measure
    b = measure._world_bounds(meshes)
    return None if b["min"] is None else b["max"][2] - b["min"][2]


def test_world_bounds_reads_the_evaluated_result():
    """The five shapes bound_box gets wrong. Each is a size-2 cube whose TRUE height is
    known, so a cage read is a specific wrong number rather than 'some drift'."""
    def _subsurf(o):
        o.modifiers.new("s", 'SUBSURF').levels = 2

    def _solidify(o):
        sm = o.modifiers.new("s", 'SOLIDIFY'); sm.thickness = 0.3; sm.offset = 1.0

    def _array(o):
        am = o.modifiers.new("a", 'ARRAY'); am.count = 4
        am.relative_offset_displace = (0, 0, 1)

    def _rotate(o):
        o.rotation_euler = (math.radians(45), 0, 0)

    # Every setup lives in the table and is called unconditionally: a dispatch chain in
    # the body would silently drop a row added here into whichever branch was last.
    for label, setup, want in (
        ("subsurf L2", _subsurf, 1.679013),
        ("solidify", _solidify, 2.346410),
        ("array x4 on Z", _array, 8.0),
        ("rotated 45 deg X", _rotate, 2.828427),
    ):
        _clear_scene()
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        o = bpy.context.object
        setup(o)
        got = _bounds_height([o])
        check(got is not None and abs(got - want) < 1e-5,
              "%s: bound_box would read 2.0; expected evaluated %.6f, got %r"
              % (label, want, got))

    # A CENTRED mirror is exact through bound_box too -- pinned so nobody "fixes" the
    # cases above by keying on 'has a generative modifier'.
    _clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    o = bpy.context.object
    mm = o.modifiers.new("m", 'MIRROR'); mm.use_axis = (False, False, True)
    check(abs(_bounds_height([o]) - 2.0) < 1e-6,
          "a centred mirror adds no geometry; height should stay 2.0")


def test_world_bounds_transform_is_the_full_affine():
    """ABSOLUTE bounds under translation AND a non-axis-aligned rotation.

    The helper hand-rolls the object transform as ``co @ mw[:3,:3].T + mw[:3,3]``, and
    every other assertion in this file is blind to getting that wrong: they measure a
    HEIGHT, which is translation-invariant, about an ORIGIN-CENTRED cube, whose bounds
    are identical under R and R-transpose. Both mutations -- dropping the translation
    (``mw[3,:3]``) and transposing the rotation -- pass the rest of the suite. This is
    the assertion that fails on them, and it matters most for
    ``proportions._world_bbox_center``, whose result MOVES the avatar.

    The fixture has to be built for this and cannot be a plain cube at an angle: a cube
    is symmetric about its own origin, and transposing a single-axis rotation is the same
    as negating it, so R and R-transpose give it identical bounds. It needs geometry
    asymmetric about the object origin AND rotation about two axes. The test proves that
    discrimination below rather than assuming it."""
    _clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1, 2, 10))
    o = bpy.context.object
    for v in o.data.vertices:                 # break symmetry about the object origin
        v.co.x += 0.7; v.co.y += 0.35; v.co.z += 1.3
    o.rotation_euler = (math.radians(35), math.radians(50), 0)

    from avatarprep.core import measure
    got = measure._world_bounds([o])

    # Ground truth from the evaluated vertices, via mathutils rather than the helper's
    # numpy -- an independent path, so a shared error cannot cancel out.
    dg = bpy.context.evaluated_depsgraph_get()
    ev = o.evaluated_get(dg)
    mw = o.matrix_world
    ws = [mw @ v.co for v in ev.data.vertices]
    want_min = [min(w[i] for w in ws) for i in range(3)]
    want_max = [max(w[i] for w in ws) for i in range(3)]
    for i, axis in enumerate("xyz"):
        check(abs(got["min"][i] - want_min[i]) < 1e-6,
              "min.%s: expected %.6f, got %.6f" % (axis, want_min[i], got["min"][i]))
        check(abs(got["max"][i] - want_max[i]) < 1e-6,
              "max.%s: expected %.6f, got %.6f" % (axis, want_max[i], got["max"][i]))

    # Prove the fixture can SEE each mutation: recompute bounds the wrong ways and
    # require them to differ from the truth. Without this, a later fixture edit could
    # quietly restore the symmetry that made the suite blind in the first place.
    R = mw.to_3x3()
    t = mw.translation
    for label, bad in (
        ("transposed rotation", [R.transposed() @ v.co + t for v in ev.data.vertices]),
        ("dropped translation", [R @ v.co for v in ev.data.vertices]),
    ):
        bad_min = [min(w[i] for w in bad) for i in range(3)]
        drift = max(abs(bad_min[i] - want_min[i]) for i in range(3))
        check(drift > 1e-3,
              "fixture cannot discriminate '%s' (drift %.9f) -- it needs geometry "
              "asymmetric about the object origin and rotation about two axes" % (label, drift))


def test_world_bounds_is_not_a_lagging_cache():
    """bound_box returns the PREVIOUS shape-key value's box: set 1.0 and it reads the
    0.0 box, set 0.5 and it reads the 1.0 box. The evaluated read must track the value
    it is asked about, on the first read, with no intervening evaluation."""
    _clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    o = bpy.context.object
    o.shape_key_add(name="Basis", from_mix=False)
    sk = o.shape_key_add(name="Tall", from_mix=False)
    for v in sk.data:
        v.co.z *= 3.0
    for value, want in ((1.0, 6.0), (0.5, 4.0), (0.0, 2.0)):
        sk.value = value
        got = _bounds_height([o])
        check(abs(got - want) < 1e-6,
              "shape key at %.1f should measure %.1f, got %.6f (a lagging cache reads "
              "the previous value's box)" % (value, want, got))


def test_world_bounds_forces_the_view_layer_update():
    """matrix_world is stale after a reparent until something forces an evaluation, and
    reading it unforced reported -50% on a real avatar.

    This pins the OUTCOME, not the mechanism: it stays green if the explicit
    view_layer.update() is deleted, because the evaluated_depsgraph_get() beside it
    forces the evaluation too. That is measured, not assumed -- and it is why the helper
    must fetch its own depsgraph per call rather than accept one from a caller, since a
    handle fetched before the mutation reads stale."""
    _clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0)
    o = bpy.context.object
    check(abs(_bounds_height([o]) - 2.0) < 1e-6, "baseline cube should measure 2.0")
    holder = bpy.data.objects.new("Holder", None)
    bpy.context.collection.objects.link(holder)
    holder.scale = Vector((1, 1, 3))
    o.parent = holder            # exactly what merge_armatures does to merge meshes
    got = _bounds_height([o])
    check(abs(got - 6.0) < 1e-6,
          "a reparent onto a z*3 holder should measure 6.0; got %.6f (stale "
          "matrix_world reads the pre-reparent 2.0)" % got)


def test_world_bounds_measurability_is_the_evaluated_count():
    """'Has vertices' means the EVALUATED count. A mesh with zero authored verts under a
    generative modifier has real geometry, and the original count calls it empty --
    which is what the two bound_box readers used to do."""
    _clear_scene()
    o = bpy.data.objects.new("Generated", bpy.data.meshes.new("GenData"))
    bpy.context.collection.objects.link(o)
    check(len(o.data.vertices) == 0, "fixture must start with zero authored verts")

    # A node group that generates a 2 m cube from nothing. Built explicitly rather than
    # via the default group, which outputs the (empty) input geometry unchanged.
    ng = bpy.data.node_groups.new("GenCube", 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    out = ng.nodes.new("NodeGroupOutput")
    cube = ng.nodes.new("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (2.0, 2.0, 2.0)
    ng.links.new(cube.outputs["Mesh"], out.inputs[0])
    o.modifiers.new("gn", 'NODES').node_group = ng

    dg = bpy.context.evaluated_depsgraph_get()
    check(len(o.evaluated_get(dg).data.vertices) == 8,
          "fixture must evaluate to real geometry, got %d verts"
          % len(o.evaluated_get(dg).data.vertices))
    got = _bounds_height([o])
    check(got is not None and abs(got - 2.0) < 1e-6,
          "a mesh generating geometry must be measured (expected 2.0, got %r) -- the "
          "original vertex count calls it empty" % got)
    bpy.data.node_groups.remove(ng)   # _clear_scene() does not purge node groups


# --- moves: rest-pose bone moves --------------------------------------------------

def test_load_moves():
    from avatarprep.core.proportions import load_edge
    base = {"source": "a", "target": "b", "source_base": "base"}
    e = load_edge(dict(base, moves=[{"bones": ["K.L", "K.R"], "value": [0, 0.01, 0]}]))
    op = e["moves"][0]
    check((op["space"], op["tail"], op["children"], op["parent_tail"], op["from"])
          == ("world", "keep", "keep", "keep", {}),
          "a move's defaults: world space, everything but the head kept, no from: %r" % op)
    check(load_edge(base)["moves"] == [], "moves defaults to empty")

    def bad(moves, substr, label, **extra):
        expect_raises(lambda: load_edge(dict(base, moves=moves, **extra)), substr, label)
    bad([{"bones": ["K"], "value": [0, 0, 0]}], "all zero", "zero move")
    bad([{"bones": ["K"], "value": [0, 1, 0], "space": "local"}], "space", "unknown space")
    bad([{"bones": ["K"], "value": [0, 1, 0], "tail": "stay"}], "tail", "bad tail policy")
    bad([{"bones": ["K"], "value": [0, 1, 0], "rotate": 1}], "unknown", "unknown key")
    bad([{"bones": ["K", "K"], "value": [0, 1, 0]}], "distinct", "duplicate bone")
    bad([{"bones": ["K"], "value": [0, 1, 0], "from": {"J": [0, 0, 0]}}], "does not move",
        "from names a bone the op does not move")
    bad([{"bones": ["K"], "value": [0, 1, 0], "from": {"K": [0, 0, 0]}}], "object/scales",
        "from in an edge with a scales stage", scales=[{"bones": ["X"], "value": [1, 1.1, 1]}])
    bad([{"bones": ["K"], "value": [0, 1, 0]},
         {"bones": ["K"], "value": [0, 1, 0], "from": {"K": [0, 0, 0]}}], "earlier move",
        "from on a bone an earlier move moves")


# Hips -> Thigh -> Shin -> Foot. The left shin and foot are connected, as a vendor import
# carries them; the right side is unconnected, as the owned body carries it.
_MOVE_RIG = (("Hips", None, False, (0, 0, 0.9), (0, 0, 1.0)),
             ("Thigh.L", "Hips", False, (0.1, 0.01, 0.85), (0.1, 0.0, 0.5)),
             ("Shin.L", "Thigh.L", True, (0.1, 0.0, 0.5), (0.1, 0.03, 0.1)),
             ("Foot.L", "Shin.L", True, (0.1, 0.03, 0.1), (0.1, -0.06, 0.03)),
             ("Thigh.R", "Hips", False, (-0.1, 0.01, 0.85), (-0.1, 0.0, 0.5)),
             ("Shin.R", "Thigh.R", False, (-0.1, 0.0, 0.5), (-0.1, 0.03, 0.1)),
             ("Foot.R", "Shin.R", False, (-0.1, 0.03, 0.1), (-0.1, -0.06, 0.03)))
_KNEE = [0.0, 0.0105, 0.0]


def _move_rig(delete=()):
    """The rig above, its world rest where the table puts it, under a vendor-import
    object transform (0.01 scale, +90 X), so a world delta must go through it to land
    in metres."""
    from avatarprep.core import scene_utils
    _clear_scene()
    arm = _make_arm(bones=[(n, h, t) for n, _, _, h, t in _MOVE_RIG])
    with scene_utils.edit_mode(arm) as ebs:
        for n, parent, conn, _, _ in _MOVE_RIG:
            ebs[n].roll = 0.25
            if parent:
                ebs[n].parent = ebs[parent]
                ebs[n].use_connect = conn
        for n in delete:
            ebs.remove(ebs[n])
    M = mathutils.Matrix.Rotation(math.radians(90), 4, 'X') @ mathutils.Matrix.Scale(0.01, 4)
    arm.data.transform(M.inverted())
    arm.matrix_world = M
    md = bpy.data.meshes.new("LegData")
    md.from_pydata([(0.1, 0.0, 0.7), (0.1, 0.02, 0.3), (0.1, 0.0, 0.05), (0, 0, 0.95)], [], [])
    md.update()
    ob = bpy.data.objects.new("Leg", md)
    bpy.context.collection.objects.link(ob)
    for g, idx in (("Thigh.L", [0]), ("Shin.L", [1]), ("Foot.L", [2]), ("Hips", [3])):
        if g in arm.data.bones:
            ob.vertex_groups.new(name=g).add(idx, 1.0, 'REPLACE')
    ob.modifiers.new("Armature", 'ARMATURE').object = arm
    bpy.context.view_layer.update()
    arm["avatarprep_base"] = "a"
    arm["avatarprep_state"] = "s0"
    return arm, ob


def _rest(arm):
    """World head and tail, connect flag and parent per bone."""
    A = arm.matrix_world
    return {b.name: {"head": A @ b.head_local, "tail": A @ b.tail_local,
                     "connect": b.use_connect, "parent": b.parent.name if b.parent else None}
            for b in arm.data.bones}


def _rolls(arm):
    from avatarprep.core import scene_utils
    with scene_utils.edit_mode(arm) as ebs:
        return {eb.name: eb.roll for eb in ebs}


def _verts(ob):
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [ob.matrix_world @ v.co for v in ev.data.vertices]


def _vclose(a, b, tol=1e-6):
    return (a - b).length < tol


def _moves_edge(moves, target="s1"):
    return {"source": "s0", "target": target, "source_base": "a", "moves": moves}


def test_moves_apply():
    from avatarprep.core import proportions as P
    d = Vector(_KNEE)
    arm, ob = _move_rig()
    before, rolls0, verts0 = _rest(arm), _rolls(arm), _verts(ob)
    weights0 = [[(g.group, g.weight) for g in v.groups] for v in ob.data.vertices]
    seen = []
    rep = P.apply_proportion_edge(arm, [ob], _moves_edge([{"bones": ["Shin.L", "Shin.R"],
                                                           "value": _KNEE}]),
                                  skip_shapekeys=True, stage_hook=lambda n: seen.append(n))
    after, rolls1 = _rest(arm), _rolls(arm)
    check(seen == ["pre", "moves"], "a moves-only edge stages pre then moves: %r" % seen)
    check(rep["moves_applied"] == 1 and len(rep["moves"]) == 2 and not rep["bakes"],
          "one op, two bone records, no bake: %r" % rep)
    for side in ("L", "R"):
        shin = "Shin." + side
        check(_vclose(after[shin]["head"], before[shin]["head"] + d),
              "%s head moves by the world delta through the 0.01 rig: %r -> %r"
              % (shin, before[shin]["head"], after[shin]["head"]))
        check(_vclose(after[shin]["tail"], before[shin]["tail"]), "%s tail is kept" % shin)
        check(abs(rolls1[shin] - rolls0[shin]) < 1e-6, "%s roll is kept" % shin)
        check(after[shin]["parent"] == "Thigh." + side, "%s keeps its parent" % shin)
        for other in ("Hips", "Thigh." + side, "Foot." + side):
            check(_vclose(after[other]["head"], before[other]["head"])
                  and _vclose(after[other]["tail"], before[other]["tail"])
                  and abs(rolls1[other] - rolls0[other]) < 1e-6,
                  "%s rest is untouched by the move" % other)
    check(not after["Shin.L"]["connect"] and after["Foot.L"]["connect"],
          "the connected shin is disconnected (thigh kept), the foot stays connected: %r"
          % {n: after[n]["connect"] for n in ("Shin.L", "Foot.L")})
    recs = {r["bone"]: r for r in rep["moves"]}
    check(recs["Shin.L"]["disconnected"] == ["Shin.L"] and recs["Shin.R"]["disconnected"] == [],
          "the record names what the move disconnected: %r"
          % {n: r["disconnected"] for n, r in recs.items()})
    for i, (v0, v1) in enumerate(zip(verts0, _verts(ob))):
        check(_vclose(v0, v1), "vertex %d stays put under a rest move: %r -> %r" % (i, v0, v1))
    check(weights0 == [[(g.group, g.weight) for g in v.groups] for v in ob.data.vertices],
          "weights are untouched")
    check(arm.get("avatarprep_state") == "s1" and arm.get("avatarprep_base") == "a",
          "the stamps transition as for any edge")

    # parent_tail "follow" keeps the connection and re-aims the parent.
    arm, ob = _move_rig()
    before = _rest(arm)
    P.apply_proportion_edge(arm, [ob], _moves_edge([{"bones": ["Shin.L"], "value": _KNEE,
                                                     "parent_tail": "follow"}]),
                            skip_shapekeys=True)
    after = _rest(arm)
    check(_vclose(after["Thigh.L"]["tail"], before["Shin.L"]["head"] + d)
          and _vclose(after["Thigh.L"]["head"], before["Thigh.L"]["head"])
          and after["Shin.L"]["connect"],
          "parent_tail follow moves the thigh's tail with the head and stays connected")

    # tail + children "follow" translate the subtree rigidly, connections intact.
    arm, ob = _move_rig()
    before = _rest(arm)
    P.apply_proportion_edge(arm, [ob], _moves_edge([{"bones": ["Shin.L"], "value": _KNEE,
                                                     "tail": "follow", "children": "follow"}]),
                            skip_shapekeys=True)
    after = _rest(arm)
    for n in ("Shin.L", "Foot.L"):
        check(_vclose(after[n]["head"], before[n]["head"] + d)
              and _vclose(after[n]["tail"], before[n]["tail"] + d),
              "%s translates rigidly under tail+children follow" % n)
    check(after["Foot.L"]["connect"], "the foot stays connected when tail and foot co-move")

    # tail "follow", children "keep": the foot stays and its connection is cleared.
    arm, ob = _move_rig()
    before = _rest(arm)
    rep = P.apply_proportion_edge(arm, [ob], _moves_edge([{"bones": ["Shin.L"], "value": _KNEE,
                                                           "tail": "follow"}]),
                                  skip_shapekeys=True)
    after = _rest(arm)
    check(_vclose(after["Foot.L"]["head"], before["Foot.L"]["head"])
          and not after["Foot.L"]["connect"]
          and "Foot.L" in rep["moves"][0]["disconnected"],
          "a kept foot under a moved tail is disconnected and reported")


def test_moves_validate():
    from avatarprep.core import proportions as P
    arm, ob = _move_rig()
    A = arm.matrix_world
    knee = [list(A @ arm.data.bones[n].head_local) for n in ("Shin.L", "Shin.R")]
    guarded = _moves_edge([{"bones": ["Shin.L", "Shin.R"], "value": _KNEE,
                            "from": {"Shin.L": knee[0], "Shin.R": knee[1]}}], target="s0")
    val = P.validate_proportion_edge(arm, [ob], P.load_edge(guarded), skip_shapekeys=True)
    check(not val["offenders"], "from at the rig's knee passes: %r" % val["offenders"])
    P.apply_proportion_edge(arm, [ob], guarded, skip_shapekeys=True)
    # A same-state edge passes the state guard a second time; from is what refuses it.
    val = P.validate_proportion_edge(arm, [ob], P.load_edge(guarded), skip_shapekeys=True)
    check(len(val["offenders"]) == 2 and all("already applied" in o for o in val["offenders"]),
          "a second apply refuses as already applied: %r" % val["offenders"])
    expect_raises(lambda: P.apply_proportion_edge(arm, [ob], guarded, skip_shapekeys=True),
                  "already applied", "second apply of a guarded same-state edge")
    off = _moves_edge([{"bones": ["Shin.L"], "value": _KNEE,
                        "from": {"Shin.L": [0.1, 0.05, 0.5]}}], target="s0")
    val = P.validate_proportion_edge(arm, [ob], P.load_edge(off), skip_shapekeys=True)
    check(any("not at the edge's source rest" in o for o in val["offenders"]),
          "a head away from both from and from + value names the mismatch: %r" % val["offenders"])

    arm, ob = _move_rig()
    for moves, substr, label in (
            ([{"bones": ["Shin.L"], "value": [0, 0.03, -0.4]}], "0.000000 m long",
             "a move onto its own tail"),
            ([{"bones": ["Thigh.L", "Shin.L"], "value": _KNEE, "children": "follow"}],
             "twice", "a bone and its ancestor in one children-follow op"),
            ([{"bones": ["Shin.L"], "value": _KNEE, "children": "follow"},
              {"bones": ["Foot.L"], "value": _KNEE, "from": {"Foot.L": [0.1, 0.03, 0.1]}}],
             "moves first", "from on a bone an earlier children-follow move carries")):
        val = P.validate_proportion_edge(arm, [ob], P.load_edge(_moves_edge(moves)),
                                         skip_shapekeys=True)
        check(any(substr in o for o in val["offenders"]),
              "%s offends: %r" % (label, val["offenders"]))
    val = P.validate_proportion_edge(arm, [ob], P.load_edge(_moves_edge(
        [{"bones": ["Shin.L", "Shin.R"], "value": [0.01, 0, 0]}])), skip_shapekeys=True)
    check(any("mirror" in w for w in val["warnings"]),
          "a pair moved by one delta with nonzero X warns: %r" % val["warnings"])


def test_moves_absent_bones():
    """A hair has no legs: drop and stub make a knee move a no-op; refuse offends."""
    from avatarprep.core import proportions as P
    edge = _moves_edge([{"bones": ["Shin.L", "Shin.R"], "value": _KNEE,
                         "from": {"Shin.L": [0.1, 0.0, 0.5], "Shin.R": [-0.1, 0.0, 0.5]}}],
                       target="s0")
    legless = ("Foot.L", "Foot.R", "Shin.L", "Shin.R", "Thigh.L", "Thigh.R")
    arm, ob = _move_rig(delete=legless)
    val = P.validate_proportion_edge(arm, [ob], P.load_edge(edge), skip_shapekeys=True)
    check(sum("bone not found" in o for o in val["offenders"]) == 2,
          "refuse offends on each absent moved bone: %r" % val["offenders"])
    for mode in ("drop", "stub"):
        arm, ob = _move_rig(delete=legless)
        before = _rest(arm)
        rep = P.apply_proportion_edge(arm, [ob], edge, skip_shapekeys=True, absent_bones=mode)
        check(rep["moves_applied"] == 0 and rep["moves"] == [] and rep["stubs"] == [],
              "%s: the move drops whole and nothing is stubbed: %r" % (mode, rep))
        check([(d["where"], d["whole_op"]) for d in rep["dropped"]] == [("moves[0]", True)],
              "%s: one whole-op DROPPED record: %r" % (mode, rep["dropped"]))
        check(all(_vclose(before[n]["head"], r["head"]) for n, r in _rest(arm).items()),
              "%s: the rig is unchanged" % mode)
        check(arm.get("avatarprep_state") == "s0", "%s: the stamps still transition" % mode)

    # One side present: the present bone moves under its from guard; the absent one drops.
    arm, ob = _move_rig(delete=("Foot.R", "Shin.R"))
    before = _rest(arm)
    rep = P.apply_proportion_edge(arm, [ob], edge, skip_shapekeys=True, absent_bones="drop")
    check([(d["bones"], d["whole_op"]) for d in rep["dropped"]] == [(["Shin.R"], False)]
          and [r["bone"] for r in rep["moves"]] == ["Shin.L"]
          and _vclose(_rest(arm)["Shin.L"]["head"], before["Shin.L"]["head"] + Vector(_KNEE)),
          "a partly-absent move applies to its present bone: %r" % rep["dropped"])

    # A --bone-override onto an absent bone is never dropped, under drop or stub.
    for mode in ("drop", "stub"):
        arm, ob = _move_rig(delete=legless)
        val = P.validate_proportion_edge(
            arm, [ob], P.load_edge(_moves_edge([{"bones": ["Knee.L"], "value": _KNEE}])),
            bone_overrides={"Knee.L": "Shin.L"}, skip_shapekeys=True, absent_bones=mode)
        check(any("Shin.L" in o and "--bone-override" in o for o in val["offenders"]),
              "%s: an override onto an absent moved bone offends: %r" % (mode, val["offenders"]))


def test_cli_whatif_reports_moves():
    """The door: --whatif writes nothing and reports each moved bone and the moves stage;
    the apply saves; a guarded second whatif exits 1 at the gate."""
    import hashlib
    import json
    import subprocess
    import tempfile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    door = os.path.join(root, "cli", "apply_proportion_edge.py")
    tmp = tempfile.mkdtemp(prefix="avatarprep_moves_")
    src, out_blend = os.path.join(tmp, "src.blend"), os.path.join(tmp, "out.blend")
    edge_path = os.path.join(tmp, "edge.json")
    arm, _ = _move_rig()
    A = arm.matrix_world
    knee = {n: list(A @ arm.data.bones[n].head_local) for n in ("Shin.L", "Shin.R")}
    bpy.ops.wm.save_as_mainfile(filepath=src)
    digest = hashlib.sha256(open(src, "rb").read()).hexdigest()
    with open(edge_path, "w", encoding="utf-8") as fh:
        json.dump(_moves_edge([{"bones": ["Shin.L", "Shin.R"], "value": _KNEE, "from": knee}],
                              target="s0"), fh)

    def run(blend, extra):
        proc = subprocess.run([bpy.app.binary_path, "--background", "--factory-startup",
                               "--python", door, "--", "--in", blend, "--edge", edge_path] + extra,
                              capture_output=True, text=True, timeout=600)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    report = os.path.join(tmp, "r.json")
    rc, out = run(src, ["--whatif", "--report", report])
    check(rc == 0, "moves whatif exits 0: %s %r" % (rc, out[-600:]))
    check(hashlib.sha256(open(src, "rb").read()).hexdigest() == digest,
          "--whatif leaves --in byte-identical")
    check("whatif moves[0] bone Shin.L head" in out and "tail kept" in out
          and "disconnected Shin.L" in out and "no bone outside the edge's moves moved" in out,
          "the printed tier shows each move: %r" % out[-1200:])
    geo = json.load(open(report, encoding="utf-8"))["geometry"]
    check([s["stage"] for s in geo["stages"]] == ["pre", "moves", "shapekeys"],
          "stages: %r" % [s["stage"] for s in geo["stages"]])
    ops = geo["move_ops"]
    check(len(ops) == 1 and [r["bone"] for r in ops[0]["bones"]] == ["Shin.L", "Shin.R"]
          and ops[0]["tail"] == "keep" and geo["collateral_moves"] == []
          and geo["collateral_lengths"] == [],
          "move_ops carries the per-bone records and the moved bones are not collateral: %r"
          % {k: geo.get(k) for k in ("move_ops", "collateral_moves", "collateral_lengths")})
    if ops and ops[0]["bones"]:
        rec = ops[0]["bones"][0]
        moved = Vector(rec["head"][1]) - Vector(rec["head"][0])
        check(_vclose(moved, Vector(_KNEE)), "the reported head delta is the edge's: %r" % moved)

    rc, out = run(src, ["--out", out_blend])
    check(rc == 0 and "moved Shin.L head" in out, "the apply: %s %r" % (rc, out[-600:]))
    rc, out = run(out_blend, ["--whatif"])
    check(rc == 1 and "already applied" in out and "whatif moves" not in out,
          "a second whatif refuses at the gate: %s %r" % (rc, out[-600:]))

def test_moves_cumulative():
    """Ops are judged in sequence: two that each pass alone cannot collapse a bone between
    them, an alias cannot slip a second move past from, and siblings moved together under
    parent_tail "follow" pass."""
    from avatarprep.core import proportions as P
    from avatarprep.core import scene_utils
    arm, ob = _move_rig()
    half_shin = [0.0, 0.015, -0.2]          # half of Shin.L's head-to-tail vector
    half_thigh = [0.0, 0.005, 0.175]        # half of Thigh.L's tail-to-head vector
    for moves, bone, label in (
            ([{"bones": ["Shin.L"], "value": half_shin}] * 2, "'Shin.L'",
             "two ops each moving a head halfway to its kept tail"),
            ([{"bones": ["Shin.L"], "value": half_thigh, "parent_tail": "follow"}] * 2,
             "'Thigh.L'", "two parent_tail follow ops collapsing the unnamed parent")):
        val = P.validate_proportion_edge(arm, [ob], P.load_edge(_moves_edge(moves)),
                                         skip_shapekeys=True)
        check(any(bone in o and "after every earlier move" in o for o in val["offenders"]),
              "%s offends: %r" % (label, val["offenders"]))
        one = P.validate_proportion_edge(arm, [ob], P.load_edge(_moves_edge(moves[:1])),
                                         skip_shapekeys=True)
        check(not one["offenders"], "%s: one op alone passes: %r" % (label, one["offenders"]))

    # The in-edit-mode backstop refuses the same collapse when validation is bypassed.
    arm, ob = _move_rig()
    moves = P.effective_edge(arm, P.load_edge(_moves_edge(
        [{"bones": ["Shin.L"], "value": half_shin}] * 2)))["edge"]["moves"]
    expect_raises(lambda: P.apply_moves(arm, moves), "half-applied",
                  "apply_moves on a collapsing op list")

    # A --bone-override alias does not get a second move past from.
    arm, ob = _move_rig()
    knee = list(arm.matrix_world @ arm.data.bones["Shin.L"].head_local)
    edge = _moves_edge([{"bones": ["Knee.L"], "value": _KNEE},
                        {"bones": ["Shin.L"], "value": _KNEE, "from": {"Shin.L": knee}}])
    val = P.validate_proportion_edge(arm, [ob], P.load_edge(edge),
                                     bone_overrides={"Knee.L": "Shin.L"}, skip_shapekeys=True)
    check(any("moves first" in o for o in val["offenders"]),
          "from on a bone an aliased earlier move moves offends: %r" % val["offenders"])

    # Two ops along one chain apply in sequence.
    arm, ob = _move_rig()
    before = _rest(arm)
    P.apply_proportion_edge(arm, [ob], _moves_edge(
        [{"bones": ["Shin.L"], "value": _KNEE},
         {"bones": ["Foot.L"], "value": _KNEE, "tail": "follow"}]), skip_shapekeys=True)
    after = _rest(arm)
    d = Vector(_KNEE)
    check(_vclose(after["Shin.L"]["head"], before["Shin.L"]["head"] + d)
          and _vclose(after["Shin.L"]["tail"], before["Shin.L"]["tail"])
          and _vclose(after["Foot.L"]["head"], before["Foot.L"]["head"] + d)
          and _vclose(after["Foot.L"]["tail"], before["Foot.L"]["tail"] + d)
          and not after["Foot.L"]["connect"],
          "two ops on one chain land in sequence: %r" % {n: after[n] for n in ("Shin.L", "Foot.L")})

    # Siblings connected to one parent tail, moved together under parent_tail "follow".
    _clear_scene()
    arm = _make_arm(bones=(("P", (0, 0, 1), (0, 0, 0.5)), ("C1", (0, 0, 0.5), (0.1, 0, 0)),
                           ("C2", (0, 0, 0.5), (-0.1, 0, 0))))
    with scene_utils.edit_mode(arm) as ebs:
        for c in ("C1", "C2"):
            ebs[c].parent = ebs["P"]
            ebs[c].use_connect = True
    arm["avatarprep_base"] = "a"
    arm["avatarprep_state"] = "s0"
    both = _moves_edge([{"bones": ["C1", "C2"], "value": [0, 0.01, 0], "parent_tail": "follow"}])
    val = P.validate_proportion_edge(arm, [], P.load_edge(both), skip_shapekeys=True)
    check(not val["offenders"], "siblings moved together pass: %r" % val["offenders"])
    val = P.validate_proportion_edge(arm, [], P.load_edge(_moves_edge(
        [{"bones": ["C1"], "value": [0, 0.01, 0], "parent_tail": "follow"}])), skip_shapekeys=True)
    check(any("'C2'" in o and "in this op too" in o for o in val["offenders"]),
          "one sibling alone offends naming the other: %r" % val["offenders"])
    P.apply_proportion_edge(arm, [], both, skip_shapekeys=True)
    after = _rest(arm)
    check(_vclose(after["P"]["tail"], Vector((0, 0.01, 0.5)))
          and after["C1"]["connect"] and after["C2"]["connect"],
          "the parent's tail follows both siblings, which stay connected: %r" % after)


def test_moves_from_tolerance():
    from avatarprep.core import proportions as P
    arm, ob = _move_rig()
    knee = arm.matrix_world @ arm.data.bones["Shin.L"].head_local
    for off, ok in ((0.0009, True), (0.0011, False)):
        edge = _moves_edge([{"bones": ["Shin.L"], "value": _KNEE,
                             "from": {"Shin.L": list(knee + Vector((off, 0, 0)))}}], target="s0")
        val = P.validate_proportion_edge(arm, [ob], P.load_edge(edge), skip_shapekeys=True)
        check(bool(val["offenders"]) != ok, "from %.4f m off should %s: %r"
              % (off, "pass" if ok else "refuse", val["offenders"]))
    expect_raises(lambda: P.load_edge(_moves_edge(
        [{"bones": ["Shin.L"], "value": [0, 0.0008, 0], "from": {"Shin.L": list(knee)}}])),
        "twice its", "from on a move within twice the tolerance")


def main():
    _clear_scene()
    _add_repo_root_to_path()
    test_load_edge()
    test_validate()
    test_local_scale()
    test_framed_scale()
    test_object_transform()
    test_shapekeys()
    test_apply_proportion_edge()
    test_apply_proportion_edge_skip_shapekeys()
    test_apply_proportion_edge_median()
    test_baked_coupling()
    test_measure_geometry_skips_empty_meshes()
    test_stage_hook_order()
    test_whatif_geometry_equals_real_apply()
    test_collateral_lengths()
    test_local_scale_op_does_not_compound_down_a_chain()
    test_cli_whatif_writes_nothing_and_reports_geometry()
    test_absent_bones_drop()
    test_cli_whatif_reports_dropped()
    test_bbox_center_skips_empty_meshes()
    test_bbox_center_refuses_unevaluated_meshes()
    test_world_bounds_reads_the_evaluated_result()
    test_world_bounds_transform_is_the_full_affine()
    test_world_bounds_is_not_a_lagging_cache()
    test_world_bounds_forces_the_view_layer_update()
    test_world_bounds_measurability_is_the_evaluated_count()
    test_load_moves()
    test_moves_apply()
    test_moves_validate()
    test_moves_absent_bones()
    test_cli_whatif_reports_moves()
    test_moves_cumulative()
    test_moves_from_tolerance()
    if FAILURES:
        for f in FAILURES:
            print("PROP_TEST FAIL:", f)
        sys.exit(1)
    print("PROP_TEST OK")

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "PROP_TEST")
