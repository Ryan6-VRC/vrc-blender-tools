"""Synthetic headless test for avatarprep.core.proportions.

Run:
  blender --background --factory-startup --python tests/test_proportions.py

Prints PROP_TEST OK and exits 0 on success; PROP_TEST FAIL: <reason> exit 1 otherwise.
"""
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


def test_bbox_center_skips_empty_meshes():
    """pivot 'bbox_center' scales about the meshes' own centre. A zero-vertex mesh's
    bound_box is eight all-zero corners, so counting it drags that centre toward the
    object's origin and the whole avatar scales about the wrong point."""
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
    test_cli_whatif_writes_nothing_and_reports_geometry()
    test_bbox_center_skips_empty_meshes()
    if FAILURES:
        for f in FAILURES:
            print("PROP_TEST FAIL:", f)
        sys.exit(1)
    print("PROP_TEST OK")

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "PROP_TEST")
