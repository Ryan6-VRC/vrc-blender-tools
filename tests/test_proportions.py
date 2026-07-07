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

def _enable_avatarprep():
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
    rep = P.validate_profile(arm, [mesh], edge, bone_overrides={}, shapekey_overrides={})
    joined = " ".join(rep["offenders"])
    check("Ghost" in joined, "missing bone Ghost should be an offender")
    check("Missing" in joined, "missing shapekey should be an offender")
    check("Spine" not in joined, "present bone Spine must not be an offender")
    rep2 = P.validate_profile(arm, [mesh], edge,
        bone_overrides={"Ghost":"Spine"}, shapekey_overrides={"Missing": None})
    check(not rep2["offenders"], "overrides should clear offenders: %r" % rep2["offenders"])
    # skip_shapekeys suppresses the missing-shapekey offender (body edge onto an outfit)
    rep_skip = P.validate_profile(arm, [mesh], edge,
        bone_overrides={"Ghost":"Spine"}, skip_shapekeys=True)
    check(not any("shapekey" in o for o in rep_skip["offenders"]),
          "skip_shapekeys should suppress missing-shapekey offenders: %r" % rep_skip["offenders"])
    arm["avatarprep_state"] = "wrong"
    rep3 = P.validate_profile(arm, [mesh], edge,
        bone_overrides={"Ghost":"Spine"}, shapekey_overrides={"Missing": None})
    check(any("state" in o.lower() for o in rep3["offenders"]), "state mismatch should be an offender")
    # A rig at the reserved 'unproportioned' origin validates clean against an
    # unproportioned-source edge (exact match); base must also match source_base.
    arm["avatarprep_base"] = "a"
    arm["avatarprep_state"] = "unproportioned"
    edge_u = dict(edge); edge_u["source"] = "unproportioned"; edge_u["source_base"] = "a"
    rep_u = P.validate_profile(arm, [mesh], P.load_edge(edge_u), skip_shapekeys=True)
    check(not any("state" in o.lower() or "base" in o.lower() for o in rep_u["offenders"]),
          "unproportioned+matching-base must not offend: %r" % rep_u["offenders"])

    # A named-source-state edge on an 'unproportioned' rig now OFFENDS (wildcard removed).
    edge_named = dict(edge); edge_named["source"] = "custom"; edge_named["source_base"] = "a"
    rep_named = P.validate_profile(arm, [mesh], P.load_edge(edge_named), skip_shapekeys=True)
    check(any("state mismatch" in o.lower() for o in rep_named["offenders"]),
          "named-source on unproportioned rig must offend: %r" % rep_named["offenders"])

    # base absent -> offender.
    del arm["avatarprep_base"]
    rep_nobase = P.validate_profile(arm, [mesh], P.load_edge(edge_u), skip_shapekeys=True)
    check(any("base absent" in o.lower() for o in rep_nobase["offenders"]),
          "base-absent must offend: %r" % rep_nobase["offenders"])
    # A rig left at the mid-apply sentinel (a crashed apply_profile) hard-FAILs distinctly.
    from avatarprep.core import scene_utils
    arm["avatarprep_state"] = scene_utils.STATE_APPLYING
    rep_int = P.validate_profile(arm, [mesh], edge,
        bone_overrides={"Ghost":"Spine"}, shapekey_overrides={"Missing": None})
    check(any("interrupted" in o.lower() for o in rep_int["offenders"]),
          "mid-apply sentinel should be an 'interrupted' offender: %r" % rep_int["offenders"])
    edge_med = P.load_edge({"source":"a","target":"b","source_base":"a",
        "scales":[{"bones":["Spine"],"value":[1.1,1,1],"pivot":"median"}], "shapekeys":{}})
    del arm["avatarprep_state"]
    rep_med = P.validate_profile(arm, [mesh], edge_med)
    check(any("median" in o for o in rep_med["offenders"]), "median pivot with 1 bone should offend")
    # this arm has two parentless bones (Root, Spine); an object edge must offend pre-mutation
    edge_obj = P.load_edge({"source":"a","target":"b","source_base":"a","object":{"scale":1.1}})
    rep_obj = P.validate_profile(arm, [mesh], edge_obj, skip_shapekeys=True)
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


def test_apply_profile():
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
    rep = P.apply_profile(arm, [mesh], edge)
    check(rep["state"] == "custom", "state should be stamped 'custom'")
    z1 = (mesh.matrix_world @ mathutils.Vector(mesh.data.vertices[0].co)).z
    check(abs(z1 - z0) > 1e-4, "geometry should have moved")
    check(mesh.data.shape_keys.key_blocks["Big"].value == 0.5, "shapekey value set")
    raised = []
    try:
        P.apply_profile(arm, [mesh], edge)
    except Exception as e:
        raised.append(str(e))
    check(raised and "state" in raised[0].lower(), "re-apply should fail on state mismatch")


def test_apply_profile_skip_shapekeys():
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
        P.apply_profile(arm, [mesh], edge)     # aborts before any mutation
    except Exception as e:
        raised.append(str(e))
    check(raised and "shapekey" in raised[0].lower(), "missing shapekey should abort without skip")
    rep = P.apply_profile(arm, [mesh], edge, skip_shapekeys=True)
    check(rep["state"] == "custom", "skip_shapekeys run should complete and stamp target")
    z1 = (mesh.matrix_world @ mathutils.Vector(mesh.data.vertices[0].co)).z
    check(abs(z1 - 2.0 * z0) < 1e-3,
          "origin scale 2x should map z=%.3f to ~%.3f, got %.3f" % (z0, 2 * z0, z1))

def test_apply_profile_median():
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
    rep = P.apply_profile(arm, [mesh], edge)
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
    r1 = P.apply_profile(arm, [mesh], repro, skip_shapekeys=True)
    check(arm["avatarprep_base"] == "shinano" and arm["avatarprep_state"] == "custom",
          "reproportion: base kept, state=custom; got (%r,%r)"
          % (arm.get("avatarprep_base"), arm.get("avatarprep_state")))

    arm["avatarprep_state"] = "unproportioned"
    equiv = {"source": "unproportioned", "target": "unproportioned",
             "source_base": "shinano", "target_base": "chiffon"}
    r2 = P.apply_profile(arm, [mesh], equiv, skip_shapekeys=True)
    check(arm["avatarprep_base"] == "chiffon",
          "equivalency: base moved to chiffon; got %r" % arm.get("avatarprep_base"))


def main():
    _clear_scene()
    _enable_avatarprep()
    test_load_edge()
    test_validate()
    test_local_scale()
    test_framed_scale()
    test_object_transform()
    test_shapekeys()
    test_apply_profile()
    test_apply_profile_skip_shapekeys()
    test_apply_profile_median()
    if FAILURES:
        for f in FAILURES:
            print("PROP_TEST FAIL:", f)
        sys.exit(1)
    print("PROP_TEST OK")

if __name__ == "__main__":
    main()
