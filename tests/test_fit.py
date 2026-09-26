"""Synthetic headless test for avatarprep.core.fit and cli/report_fit.py, compare_fit.py, push_garment.py.

Run: blender --background --factory-startup --python tests/test_fit.py
Prints FIT_TEST OK / FIT_TEST FAIL: <reason>.

Needs no provisioned wheels: the fit doors run on Blender's own numpy and mathutils.
Fixture: ``tests/_weight_fixture.py`` (the tail and the thigh bands are this suite's).
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOKEN = "FIT_TEST"
FAILURES = []
LEG = [("UpperLeg.L", None, None, None)]


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def refuses(fn, substr, label, exc=None):
    from avatarprep.core.fit import FitError
    try:
        fn()
    except (exc or FitError) as e:
        check(substr.lower() in str(e).lower(), "%s: refused but %r lacked %r" % (label, str(e), substr))
        return
    FAILURES.append("%s: expected a refusal mentioning %r" % (label, substr))


def _sweep(data, sweeps=None, regions=None):
    from avatarprep.core import fit
    p = fit.plan(data, sweeps)
    return p, fit.sweep(data, p, fit.region_sets(data, regions or {}))


def _by_step(result, garment, metric, key, axis=None):
    return [s["garments"][garment]["regions"]["all"][metric][key] for s in result["steps"]
            if axis is None or s["axis"] == axis]


def test_slide_and_penetration():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    thigh = F.add_band(s, "ThighBand", {"UpperLeg.L": 1.0})
    hips = F.add_band(s, "HipsBand", {"Hips": 1.0})
    d = fit.load(s["body"], [thigh, hips])
    p, r = _sweep(d, LEG)
    check(len(p["steps"]) == 12, "the hip row sweeps 3 steps each way on 2 axes (%d)" % len(p["steps"]))
    w = r["worst"]["ThighBand"]["all"]
    check(w["slide"]["max"] < 0.01, "a thigh-weighted band does not slide under a thigh sweep: %r" % w["slide"])
    check(abs(w["stretch"]["max"] - 1.0) < 1e-4, "a rigidly following band does not stretch: %r" % w["stretch"])
    check(sum(_by_step(r, "ThighBand", "new_pen", "count", "lateral")) == 0,
          "a thigh-weighted band takes no new penetration under flexion")
    w = r["worst"]["HipsBand"]["all"]
    check(w["new_pen"]["count"] > 0 and w["new_pen"]["max_mm"] > 1.0,
          "a Hips-only band takes new penetration under a thigh sweep: %r" % w["new_pen"])
    check(w["slide"]["p95"] > 10.0, "a Hips-only band slides against the thigh: %r" % w["slide"])
    check(all(r["rest"][n]["regions"]["all"]["pen"] == 0 for n in ("ThighBand", "HipsBand")),
          "neither band penetrates at rest")


def test_proximity_rule():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    hips = F.add_band(s, "HipsBand", {"Hips": 1.0})
    thigh = F.add_band(s, "ThighBand", {"UpperLeg.L": 1.0})
    p = fit.plan(fit.load(s["body"], [hips]))
    bones = {b["bone"]: b for b in p["bones"]}
    check("Hips" not in bones, "the armature root is never swept: %r" % sorted(bones))
    leg = bones.get("UpperLeg.L")
    check(leg is not None and leg["garment_share"] == 0.0 and leg["body_share"] > 0.9,
          "a Hips-only band draws UpperLeg.L in by proximity alone: %r" % leg)
    check("LowerLeg.L" not in bones, "bones under the floor share are not swept: %r" % sorted(bones))
    p = fit.plan(fit.load(s["body"], [thigh]))
    leg = {b["bone"]: b for b in p["bones"]}.get("UpperLeg.L")
    check(leg is not None and leg["garment_share"] == 1.0, "a thigh-weighted band draws UpperLeg.L by weight: %r" % leg)
    refuses(lambda: fit.plan(fit.load(s["body"], [thigh]), [("LowerLeg.L", None, None, None)]),
            "moves no body vertex", "a named bone with no skin under the garment")
    refuses(lambda: fit.plan(fit.load(s["body"], [thigh]), [("Nope", None, None, None)]),
            "not a deform bone", "a named bone the body lacks")
    refuses(lambda: fit.plan(fit.load(s["body"], [thigh]), [("UpperLeg.L", "up", None, None)]),
            "give MIN..MAX", "an axis the row lacks without a range")


def test_rows_and_flexion():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    thigh = F.add_band(s, "ThighBand", {"UpperLeg.L": 1.0})
    shin = F.add_band(s, "ShinBand", {"LowerLeg.L": 1.0}, z0=0.20, z1=0.35)
    d = fit.load(s["body"], [thigh, shin])
    F_ = d["frame"]
    p = fit.plan(d, [("UpperLeg.L", None, None, None), ("UpperLeg.R", None, None, None),
                     ("LowerLeg.L", None, None, None)])
    rows = {b["bone"]: b["row"] for b in p["bones"]}
    check(rows == {"UpperLeg.L": "hip", "UpperLeg.R": "hip", "LowerLeg.L": "knee"}, "joint rows: %r" % rows)
    bones = d["body"]["bones"]

    def moved(step):
        b = bones[step["bone"]]
        return (b["child"] - b["head"]) @ fit.rotation(step["u"], step["angle"]).T + b["head"] - b["child"]

    for st in p["steps"]:
        mv = moved(st)
        if st["axis"] == "lateral" and st["bone"].startswith("Upper") and st["angle"] > 0:
            check(mv @ F_["forward"] > 0, "hip flexion carries the knee forward (%s)" % st["name"])
        if st["axis"] == "forward" and st["angle"] > 0:
            side = 1 if st["bone"].endswith(".L") else -1
            check(mv @ (side * F_["lateral"]) > 0, "abduction carries the knee outward (%s)" % st["name"])
        if st["bone"] == "LowerLeg.L":
            check(st["angle"] > 0 and mv @ F_["forward"] < 0, "knee flexion carries the foot back (%s)" % st["name"])
    check(sorted(round(st["angle"]) for st in p["steps"] if st["bone"] == "LowerLeg.L") == [43, 87, 130],
          "the knee row sweeps +130 in three steps and nothing negative")


def test_symmetric_other():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build(tail=True)
    d = fit.load(s["body"], [s["tail_cover"]])
    p, r = _sweep(d)
    check([b["bone"] for b in p["bones"]] == ["Tail"] and p["bones"][0]["row"] == "other",
          "a tail cover sweeps the tail on the other row: %r" % p["bones"])
    for ax in ("a", "b"):
        got = sorted(round(st["angle"], 2) for st in p["steps"] if st["axis"] == ax)
        check(got == [-20.0, -13.33, -6.67, 6.67, 13.33, 20.0], "the other row is symmetric about %s: %r" % (ax, got))
    ua = next(st["u"] for st in p["steps"] if st["axis"] == "a")
    ub = next(st["u"] for st in p["steps"] if st["axis"] == "b")
    down = np.array([0.0, 0.0, -1.0])
    check(abs(ua @ down) < 1e-6 and abs(ub @ down) < 1e-6 and abs(ua @ ub) < 1e-6,
          "a and b are perpendicular to the bone and to each other: %r %r" % (ua, ub))
    w = r["worst"]["TailCover"]["all"]
    check(w["slide"]["max"] < 0.01 and w["new_pen"]["count"] == 0, "a tail-weighted cover follows the tail: %r" % w)


def test_regions():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build(tail=True)
    g = s["garment"]
    d = fit.load(s["body"], [g, s["tail_cover"]], groups=["WriteMask"])
    V = np.array([v.co[:] for v in g.data.vertices])
    wm = np.zeros(len(V))
    for i, v in enumerate(g.data.vertices):
        for x in v.groups:
            if g.vertex_groups[x.group].name == "WriteMask":
                wm[i] = x.weight
    tags = np.array(s["tags"])
    expect = {
        "group:WriteMask": wm > 0.5,
        "zone:front": V[:, 1] < 0,
        "zone:left": V[:, 0] > 0,
        "zone:below:UpperLeg.L": V[:, 2] < 0.899,
        "zone:front+zone:left": (V[:, 1] < 0) & (V[:, 0] > 0),
        "bone:Flap": (tags == "strip") | np.array([vv.co.y > 0.12 and abs(vv.co.z - 0.90) < 1e-4 and t == "band"
                                                   for vv, t in zip(g.data.vertices, tags)]),
    }
    regions = fit.resolve_regions(d, list(expect))
    for spec, want in expect.items():
        got = regions[spec][0]
        check(np.array_equal(got, want), "region %s: %d vertices, expected %d" % (spec, got.sum(), want.sum()))
    check(not regions["group:WriteMask"][1].any(), "a group a garment lacks is empty on it")
    tail = fit.resolve_regions(d, ["bone:Tail"])["bone:Tail"]
    check(not tail[0].any() and tail[1].all(), "bone:Tail holds the tail cover only")
    p = fit.plan(d, None, tail)
    check([b["bone"] for b in p["bones"]] == ["Tail"], "a region-derived sweep takes the region's bones: %r"
          % [b["bone"] for b in p["bones"]])
    check({"UpperLeg.L", "Tail"} <= {b["bone"] for b in fit.plan(d)["bones"]}, "the whole-garment sweep is wider")
    for bad in ("zone:up", "zone:Y", "foo:bar", "zone:above", "group"):
        refuses(lambda: fit.parse_region(bad), "", "region syntax %s" % bad, exc=ValueError)
    refuses(lambda: fit.resolve_regions(d, ["bone:Nope"]), "none of the garments", "bone: the rigs lack")
    refuses(lambda: fit.resolve_regions(d, ["zone:above:Nope"]), "names no bone", "zone: bone the body lacks")
    refuses(lambda: fit.resolve_regions(d, ["zone:front+zone:back"]), "holds no vertex", "an empty region")
    refuses(lambda: fit.load(s["body"], [g], groups=["Nope"]), "group:Nope", "group: on no garment")


def test_shape_and_cut():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    g = s["garment"]
    d = fit.load(s["body"], [g], shapes=[(None, "Bulk", 1.0)])
    check(d["garments"][0]["rsd"].min() < -0.005, "--shape Bulk=1 swells the pelvis through the band at rest")
    check(s["body"].data.shape_keys.key_blocks["Bulk"].value == 0.0, "the body's live value is untouched")
    d = fit.load(s["body"], [g], cut_shapes=["Bulk"], cut_threshold=0.005)
    kb = s["body"].data.shape_keys.key_blocks
    pelvis = sum(1 for x, y in zip(kb["Bulk"].data, kb["Basis"].data) if (x.co - y.co).length > 0.005)
    check(d["body"]["cut"].sum() == pelvis, "--cut-shape removes the vertices Bulk moves over the threshold (%d of %d)"
          % (d["body"]["cut"].sum(), pelvis))
    check(not d["body"]["cut"][d["body"]["F"]].any(), "no kept body triangle touches a cut vertex")
    refuses(lambda: fit.load(s["body"], [g], cut_shapes=["Nope"]), "is on none of", "an unknown cut shape")
    refuses(lambda: fit.load(s["body"], [g], shapes=[(None, "Nope", 1.0)]), "is on none of", "an unknown shape")


def test_simulated_transfer():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    hips = F.add_band(s, "HipsBand", {"Hips": 1.0})
    d = fit.load(s["body"], [hips])
    sim = fit.simulated(d)
    p = fit.plan(d, LEG)
    sets = fit.region_sets(d, {})
    a, b = fit.sweep(d, p, sets), fit.sweep(sim, p, sets)
    wa, wb = a["worst"]["HipsBand"]["all"], b["worst"]["HipsBand"]["all"]
    check(wa["new_pen"]["count"] > 0 and wb["new_pen"]["count"] == 0,
          "the simulated transfer removes the weights defect: %r -> %r" % (wa["new_pen"], wb["new_pen"]))
    check(wb["slide"]["p95"] < 0.5 < wa["slide"]["p95"], "and its slide: %r -> %r" % (wa["slide"], wb["slide"]))
    check(np.array_equal(d["garments"][0]["W"][:, d["body"]["names"].index("Hips")], np.ones(len(hips.data.vertices))),
          "the simulation leaves the measured data alone")
    d = fit.load(s["body"], [s["garment"]])
    g, sg = d["garments"][0], fit.simulated(d)["garments"][0]
    go = len(g["garment_only"])
    kept = [i for i in range(len(g["V"])) if g["raw"][i, g["garment_only"]].sum() >= 1 - 1e-4] + [s["four"]]
    check(all(np.allclose(sg["W"][i], g["W"][i]) for i in kept),
          "vertices at p = 1 and the one with no allowance keep their own weights, as transfer_weights keeps them")
    half = [i for i in range(len(g["V"])) if abs(g["raw"][i, g["garment_only"]].sum() - 0.5) < 1e-6]
    check(half and all(abs(sg["own"][i, go:].sum() - 0.5) < 1e-6 and (sg["own"][i, go:] > 0).sum() <= 2 for i in half),
          "a p = 0.5 vertex takes body weight 0.5 on at most 4 - 2 bones")


def test_push_core():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    static = F.add_band(s, "Static", {"UpperLeg.L": 1.0}, gap=0.0003)
    d = fit.load(s["body"], [static])
    p = fit.plan(d, [("UpperLeg.L", "lateral", None, None)])
    refuses(lambda: fit.plan_push(d, 0, p, amount=0.0005), "already there at rest", "static contact")
    pl = fit.plan_push(d, 0, p, amount=0.0005, allow_static=True)
    check(pl["kind"] == "static" and pl["seeds"] > 0, "--allow-static pushes the rest contact: %r" % pl["kind"])


def _add_bone(rig, name, head, tail, parent):
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones.new(name)
    eb.head, eb.tail = head, tail
    if parent:
        eb.parent = rig.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')


def _one_step(d, bone="UpperLeg.L", axis="lateral", angle=90.0):
    from avatarprep.core import fit
    return fit.plan(d, [(bone, axis, (0.0, angle) if angle > 0 else (angle, 0.0), 1)])["steps"][0]


def test_pose_math():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    d = fit.load(s["body"], [F.add_band(s, "ThighBand", {"UpperLeg.L": 1.0})])
    st = _one_step(d)
    h, R = st["h"], fit.rotation(st["u"], st["angle"])
    k = d["body"]["bones"]["LowerLeg.L"]["head"]
    got = fit.pose_points(np.array([h, k]), np.ones(2), h, R)
    check(np.allclose(got[0], h, atol=1e-12), "the swept bone's head stays fixed")
    check(np.allclose(got[1], h + R @ (k - h), atol=1e-12), "the knee lands at h + R(k - h)")
    BV, _ = fit.posed(d, d["garments"][0], st)
    share = d["body"]["W"][:, st["cols"]].sum(1)
    full, none = share > 1 - 1e-9, share < 1e-12
    V0 = d["body"]["V"]
    check(full.any() and np.allclose(BV[full], (V0[full] - h) @ R.T + h, atol=1e-9),
          "vertices wholly on the swept chain turn rigidly about its head")
    check(none.any() and np.array_equal(BV[none], V0[none]), "vertices off the swept chain do not move")


def test_garment_bones():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    rig = s["garment_rig"]
    _add_bone(rig, "Strap", (F.LX, -0.08, 0.70), (F.LX, -0.08, 0.60), "UpperLeg.L")
    _add_bone(rig, "Free", (0.3, 0, 0.7), (0.3, 0, 0.6), None)
    strap = F.add_band(s, "StrapBand", {"Strap": 1.0})
    free = F.add_band(s, "FreeBand", {"Free": 1.0})
    toes = F.add_band(s, "ToesBand", {"Toes.L": 1.0})
    d = fit.load(s["body"], [strap, free, toes])
    st = _one_step(d)
    h, R = st["h"], fit.rotation(st["u"], st["angle"])
    gs, gf, gt = d["garments"]
    _, GV = fit.posed(d, gs, st)
    check(np.allclose(GV, (gs["V"] - h) @ R.T + h, atol=1e-9),
          "a garment bone under UpperLeg.L moves rigidly with the thigh, so it does not slide")
    _, GV = fit.posed(d, gf, st)
    check(np.array_equal(GV, gf["V"]), "a garment bone with no body ancestor stays still")
    check(gs["cls"]["physbone"].all() and gf["cls"]["physbone"].all(), "garment-only bands are physbone")
    check(not gt["cls"]["physbone"].any(),
          "a body armature deform bone the body mesh leaves unweighted is not garment-only (Toes.L)")
    refuses(lambda: fit.plan(d, bones=["Nope"]), "lacks Nope", "a union sweep bone this rig lacks")


def test_top4_in_fit():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    d = fit.load(s["body"], [s["garment"]])
    g = d["garments"][0]
    row = g["own"][s["four"]]
    check(int((row > 0).sum()) == 4 and np.allclose(row[row > 0], 0.25),
          "five equal groups skin on four at 0.25 each: %r" % row)
    check(abs(g["phys"][s["four"]] - 0.75) < 1e-9, "three of the four kept are garment bones (%g)" % g["phys"][s["four"]])
    check(abs(g["W"][s["four"], d["body"]["names"].index("Hips")] - 1.0) < 1e-9,
          "Flap bones move with their ancestor Hips")


def test_body_through():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    hips = F.add_band(s, "HipsBand", {"Hips": 1.0})
    thigh = F.add_band(s, "ThighBand", {"UpperLeg.L": 1.0})
    d = fit.load(s["body"], [hips, thigh])
    p, r = _sweep(d, [("UpperLeg.L", None, None, None)])
    w = r["worst"]["HipsBand"]["all"]["body_through"]
    check(w["count"] > 0 and 0 < w["max_mm"] <= 5.0,
          "the thigh leaving a still band comes through it, within 5 mm: %r" % w)
    check(sum(_by_step(r, "ThighBand", "body_through", "count")) == 0,
          "a band that follows its thigh has no body-through, however close the other thigh comes")
    g = d["garments"][1]
    adduct = next(st for st in p["steps"] if st["name"] == "UpperLeg.L:forward:-10")
    BV, GV = fit.posed(d, g, adduct)
    saved = g["bh0"]
    g["bh0"] = np.full_like(saved, -1.0)
    try:
        _, tri, height, _ = fit._depth_and_through(d, g, BV, GV, np.zeros(len(GV), bool),
                                                   np.flatnonzero(g["tok"]), fit.THROUGH)
    finally:
        g["bh0"] = saved
    check(((height > 0) & (height <= fit.THROUGH)).any(),
          "without the rest-inside rule the other thigh would read as body-through (the rule is doing work)")


def test_push_seeds_follow_contact():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    d = fit.load(s["body"], [F.add_band(s, "ThighBand", {"UpperLeg.L": 1.0})])
    g = d["garments"][0]
    p = fit.plan(d, [("UpperLeg.L", "forward", None, None)])
    near = 0.0005
    pop = ~g["cls"]["physbone"] & ~g["cut"]
    for st in p["steps"]:
        seeds = fit._seeds(d, g, st, near, ~g["cut"])
        BV, GV = fit.posed(d, g, st)
        depth, _, _, _ = fit._depth_and_through(d, g, BV, GV, pop, np.zeros(0, np.int64), 0)
        contact = pop & (depth > -near * 1000.0)
        check(np.array_equal(seeds, contact), "%s: a band following its thigh seeds only where it touches the "
              "body (%d seeds, %d touching)" % (st["name"], seeds.sum(), contact.sum()))
        if st["angle"] > 0:
            check(not seeds.any(), "%s: abduction away from the other thigh seeds nothing" % st["name"])


def _hex_band(s, name, inradius, inward=False):
    import _weight_fixture as F
    verts, faces, tags = [], [], []
    F.tube(F.LX, 0, inradius / np.cos(np.pi / 6), 0.60, 0.76, 4, 6, name, verts, faces, tags)
    if inward:
        faces = [f[::-1] for f in faces]
    ob = F.mesh(name, verts, faces, s["garment_rig"])
    ob.vertex_groups.new(name="Hips").add(list(range(len(verts))), 1.0, 'REPLACE')
    return ob


def test_push_profile():
    import _weight_fixture as F
    from avatarprep.core import fit
    s = F.build()
    hips = F.add_band(s, "HipsBand", {"Hips": 1.0})
    flipped = F.add_band(s, "Flipped", {"Hips": 1.0})
    me = flipped.data
    for poly in me.polygons:
        poly.flip()
    me.update()
    hexb = _hex_band(s, "Hex", F.RL + 0.0015)
    d = fit.load(s["body"], [hips, flipped, hexb])
    p = fit.plan(d, [("UpperLeg.L", "lateral", None, None)])
    amount, falloff = 0.0005, 0.025
    pl = fit.plan_push(d, 0, p, amount=amount, falloff=falloff)
    g = d["garments"][0]
    V, seeds = g["V"], pl["seeded"]
    dist = np.min(np.linalg.norm(V[:, None, :] - V[seeds][None, :, :], axis=2), axis=1)
    want = amount * fit.smoothstep(1.0 - dist / falloff)
    got = np.linalg.norm(pl["delta"], axis=1)
    check(np.allclose(got, want, atol=1e-9), "the push follows amount * smoothstep(1 - d / falloff)")
    check(pl["rim_moved"] > 0, "without --rim-hold the boundary moves")
    held = fit.plan_push(d, 0, p, amount=amount, falloff=falloff, rim_hold=0.01)
    check(held["rim_moved"] == 0 and held["moved"] > 0, "--rim-hold keeps the boundary on the skin: %r"
          % {k: held[k] for k in ("moved", "rim_moved")})
    pf = fit.plan_push(d, 1, p, amount=amount, falloff=falloff)
    radial = d["garments"][1]["V"][:, :2] - np.array([F.LX, 0.0])
    moved = np.linalg.norm(pf["delta"], axis=1) > 0
    check(pf["turned"] == pf["moved"] > 0 and ((pf["delta"][:, :2] * radial).sum(1)[moved] > 0).all(),
          "an inward-wound band's normals are turned, and it still moves outward")
    gh = d["garments"][2]
    pop = ~gh["cls"]["physbone"]
    corner_only = False
    for st in p["steps"]:
        seeds = fit._seeds(d, gh, st, 0.0005, ~gh["cut"])
        BV, GV = fit.posed(d, gh, st)
        depth, _, _, _ = fit._depth_and_through(d, gh, BV, GV, pop, np.zeros(0, np.int64), 0)
        corner_only |= bool((seeds & ~(depth > -0.5)).any())
    check(corner_only, "a coarse band seeds the corners of a triangle the body comes through between its vertices")


def test_apply_push_units():
    from avatarprep.core import fit
    me = bpy.data.meshes.new("CmMesh")
    rng = np.random.default_rng(1)
    co = (rng.random((200, 3)) * [30, 20, 170]).tolist()
    me.from_pydata(co, [], [(i, i + 1, i + 2) for i in range(0, 198, 3)])
    ob = bpy.data.objects.new("CmMesh", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.shape_key_add(name="Basis")
    a = ob.shape_key_add(name="A", from_mix=False)
    b = ob.shape_key_add(name="B", from_mix=False)
    b.relative_key = a
    for i in range(len(co)):
        a.data[i].co = a.data[i].co + type(a.data[i].co)((0.0, 0.0, 1.3))
        b.data[i].co = a.data[i].co + type(a.data[i].co)((0.7, 0.0, 0.0))
    ab = np.array([(y.co - x.co)[:] for x, y in zip(a.data, b.data)])
    delta = rng.random((len(co), 3)) * 0.05
    n = fit.apply_push(ob, delta)
    ab2 = np.array([(y.co - x.co)[:] for x, y in zip(a.data, b.data)])
    check(n == 3 and np.abs(ab2 - ab).max() < 1e-4, "a centimetre-scale mesh takes the push; a key relative to "
          "another key keeps its offset from it")


def test_sweep_grammar():
    from cli import _fit

    def err(msg):
        raise ValueError(msg)
    check(_fit.parse_sweep("mixamorig:LeftUpLeg:lateral:-30..90:4", err) == ("mixamorig:LeftUpLeg", "lateral", (-30.0, 90.0), 4),
          "a colon in the bone name survives")
    check(_fit.parse_sweep("Tail", err) == ("Tail", None, None, None), "a bare bone")
    check(_fit.parse_sweep("UpperLeg.L:0..90", err) == ("UpperLeg.L", None, (0.0, 90.0), None), "a range alone")
    from avatarprep.core import fit
    rows = {n: fit.joint_row(n)[0] for n in ("J_Bip_L_UpperLeg", "mixamorig:LeftUpLeg", "mixamorig:LeftFoot", "mixamorig:LeftForeArm",
                                              "Head", "HeadTop_End", "Tail")}
    check(rows == {"J_Bip_L_UpperLeg": "hip", "mixamorig:LeftUpLeg": "hip", "mixamorig:LeftFoot": "ankle", "mixamorig:LeftForeArm": "elbow",
                   "Head": "head", "HeadTop_End": "other", "Tail": "other"}, "joint rows by glob: %r" % rows)
    for bad, why in (("UpperLeg.L:X", "world axis"), ("UpperLeg.L:lateral:30..90", "through 0"),
                     ("UpperLeg.L:lateral:0..90:0", "at least 1")):
        refuses(lambda: _fit.parse_sweep(bad, err), why, "sweep grammar %s" % bad, exc=ValueError)


def _blender(tool, args):
    cmd = [bpy.app.binary_path, "--background", "--factory-startup", "--python",
           os.path.join(REPO, "cli", tool + ".py"), "--"] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def _keyed_band(s, name, weights, gap):
    import _weight_fixture as F
    ob = F.add_band(s, name, weights, gap=gap)
    ob.shape_key_add(name="Basis")
    kb = ob.shape_key_add(name="Loose", from_mix=False)
    kb.value = 0.0
    for i, v in enumerate(ob.data.vertices):
        kb.data[i].co = (F.LX + (v.co.x - F.LX) * 1.05, v.co.y * 1.05, v.co.z)
    return ob


def _coords(ob):
    me = ob.data
    out = {"mesh": np.array([v.co[:] for v in me.vertices])}
    for kb in me.shape_keys.key_blocks:
        out[kb.name] = np.array([d.co[:] for d in kb.data])
    return out


def test_cli(tmp):
    import _weight_fixture as F
    from avatarprep.core import scene_utils
    s = F.build()
    ob = _keyed_band(s, "Band", {"Hips": 1.0}, F.GAP)
    a_blend = os.path.join(tmp, "a.blend")
    bpy.ops.wm.save_as_mainfile(filepath=a_blend)
    ob.vertex_groups["Hips"].name = "UpperLeg.L"
    b_blend = os.path.join(tmp, "b.blend")
    bpy.ops.wm.save_as_mainfile(filepath=b_blend)
    rdir = os.path.join(tmp, "renders")

    rep = os.path.join(tmp, "r.json")
    rc, txt = _blender("report_fit", ["--in", a_blend, "--targets", "Band", "--sweep", "UpperLeg.L",
                                      "--report", rep])
    check(rc == 0 and "=> OK" in txt and "Traceback" not in txt, "report_fit exits 0 with OK (rc=%d)\n%s" % (rc, txt))
    for want in ("sweep UpperLeg.L row=hip side=left (named)", "extents in the frame", "legend: classes",
                 "Band [all] new-pen", "Band [all] slide p95", "forward=(0,-1,0)"):
        check(want in txt, "report_fit prints %r\n%s" % (want, txt))
    if os.path.exists(rep):
        r = json.load(open(rep))
        check(len(r["steps"]) == 12 and r["worst"]["Band"]["all"]["new_pen"]["count"] > 0,
              "the report holds every step and the worst new penetration")
        print("%s timing: report_fit sweep of %d steps on %d garment vertices took %.2fs"
              % (TOKEN, len(r["steps"]), len(ob.data.vertices), r["seconds"]))
    rc, txt = _blender("report_fit", ["--in", a_blend, "--targets", "Band", "--whatif"])
    check(rc == 0 and "estimate=" in txt and "steps=24" in txt, "--whatif estimates the derived sweep\n%s" % txt)
    rc, txt = _blender("report_fit", ["--in", a_blend, "--targets", "Band", "--sweep", "UpperLeg.L:Y"])
    check(rc == 2 and "world axis" in txt, "a world axis is bad args\n%s" % txt)
    rc, txt = _blender("report_fit", ["--in", a_blend, "--targets", "Band", "--sweep", "LowerLeg.L"])
    check(rc == 1 and "=> FAIL:" in txt and "moves no body vertex" in txt, "a skinless sweep bone refuses\n%s" % txt)
    rc, txt = _blender("report_fit", ["--in", a_blend, "--targets", "Band", "--region", "zone:front",
                                      "--render", rdir])
    pngs = [l.rsplit(": ", 1)[1] for l in txt.splitlines() if l.startswith("AVATARPREP: render ")]
    check(rc == 0 and "Band [zone:front] new-pen" in txt and pngs and all(os.path.exists(p) for p in pngs),
          "a region report renders its worst steps (rc=%d)\n%s" % (rc, txt))

    cmp = os.path.join(tmp, "c.json")
    rc, txt = _blender("compare_fit", ["--in", a_blend, "--in", b_blend, "--targets", "Band", "--sweep",
                                       "UpperLeg.L:lateral", "--render", rdir, "--report", cmp])
    check(rc == 0 and set(re.findall(r"=> ([A-Z]+)", txt)) == {"OK"},
          "compare_fit's only summary token is OK, never a verdict\n%s" % txt)
    if os.path.exists(cmp):
        c = json.load(open(cmp))
        dn = c["deltas"]["Band"]["all"]
        check(dn["new_pen"]["delta"]["b"] < 0 and dn["slide"]["delta"]["b"] < 0,
              "the thigh-weighted input has less penetration and slide: %r" % dn)
        check(c["renders"] and all(os.path.exists(x["png"]) and x["columns"] == ["a", "b"] for x in c["renders"]),
              "stitched sheets carry both inputs: %r" % c["renders"])
    rc, txt = _blender("compare_fit", ["--in", a_blend, "--simulate-transfer", "--targets", "Band", "--sweep",
                                       "UpperLeg.L:lateral", "--report", cmp])
    check(rc == 0 and "a:simulated" in txt, "--simulate-transfer compares one input with itself\n%s" % txt)
    if rc == 0:
        dn = json.load(open(cmp))["deltas"]["Band"]["all"]
        check(dn["new_pen"]["delta"]["a:simulated"] < 0, "the simulation removes new penetration: %r" % dn["new_pen"])
    rc, txt = _blender("compare_fit", ["--in", a_blend, "--targets", "Band"])
    check(rc == 2, "one --in without --simulate-transfer is bad args\n%s" % txt)

    # push: static contact, posed contact, the stamp, --whatif
    s = F.build()
    _keyed_band(s, "Band", {"UpperLeg.L": 1.0}, 0.0003)
    static = os.path.join(tmp, "static.blend")
    bpy.ops.wm.save_as_mainfile(filepath=static)
    rc, txt = _blender("push_garment", ["--in", static, "--targets", "Band", "--amount", "0.0005",
                                        "--sweep", "UpperLeg.L:lateral", "--whatif"])
    check(rc == 1 and "--allow-static" in txt, "static contact refuses\n%s" % txt)
    rc, txt = _blender("push_garment", ["--in", static, "--targets", "Band", "--amount", "0.0005", "--whatif"])
    check(rc == 2 and "--sweep or" in txt, "a push without --sweep or --region is bad args\n%s" % txt)
    rc, txt = _blender("push_garment", ["--in", static, "--targets", "Band", "--amount", "0.5", "--sweep",
                                        "UpperLeg.L", "--whatif"])
    check(rc == 2 and "metres" in txt, "an --amount over --falloff is bad args naming metres\n%s" % txt)

    before_mtime = os.path.getmtime(a_blend)
    rc, txt = _blender("push_garment", ["--in", a_blend, "--targets", "Band", "--amount", "0.0005",
                                        "--sweep", "UpperLeg.L:lateral", "--whatif"])
    check(rc == 0 and "whatif; nothing saved" in txt and os.path.getmtime(a_blend) == before_mtime,
          "--whatif plans and writes nothing\n%s" % txt)
    pushed = os.path.join(tmp, "pushed.blend")
    rc, txt = _blender("push_garment", ["--in", a_blend, "--targets", "Band", "--amount", "0.0005",
                                        "--sweep", "UpperLeg.L:lateral", "--out", pushed])
    line = "push_garment --targets Band --amount 0.0005 --sweep UpperLeg.L:lateral"
    check(rc == 0 and ("recipe: " + line) in txt, "a posed-contact push saves (rc=%d)\n%s" % (rc, txt))
    if rc == 0:
        bpy.ops.wm.open_mainfile(filepath=a_blend)
        old = _coords(bpy.data.objects["Band"])
        check(bpy.data.objects["Band"].get(scene_utils.STAMP_PUSHED) is None, "--whatif left no stamp")
        bpy.ops.wm.open_mainfile(filepath=pushed)
        ob = bpy.data.objects["Band"]
        new = _coords(ob)
        deltas = {k: new[k] - old[k] for k in old}
        ref = deltas["Basis"]
        check(all(np.abs(dv - ref).max() < 1e-7 for dv in deltas.values()),
              "the mesh, Basis and every key move by the same delta")
        mag = np.linalg.norm(ref, axis=1)
        check(0 < mag.max() <= 0.0005 + 1e-7, "the push is at most --amount: %g" % mag.max())
        radial = old["Basis"][:, :2] - np.array([F.LX, 0.0])
        check(((ref[:, :2] * radial).sum(1)[mag > 0] > 0).all(), "the push is outward from the thigh")
        check(ob.get(scene_utils.STAMP_PUSHED) == line, "the object carries the recipe: %r" % ob.get(scene_utils.STAMP_PUSHED))
        rep = scene_utils.report_stamps(bpy.context.scene)
        entry = [m for a in rep["armatures"] for m in a["meshes"] + rep["unbound"] if m["name"] == "Band"]
        check(entry and entry[0].get("pushed") == line, "report_stamps surfaces the push: %r" % rep)
        rc, txt = _blender("report_stamps", ["--in", pushed])
        check(rc == 0 and ("mesh Band pushed=%s" % line) in txt, "report_stamps prints the push\n%s" % txt)
        rc, txt = _blender("push_garment", ["--in", pushed, "--targets", "Band", "--amount", "0.0005",
                                            "--sweep", "UpperLeg.L:lateral", "--whatif"])
        check(rc == 1 and "already pushed by `%s`" % line in txt, "a stamped mesh refuses and names its line\n%s" % txt)


def main():
    for p in (REPO, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    import avatarprep
    try:
        avatarprep.register()
    except Exception:
        pass
    test_slide_and_penetration()
    test_proximity_rule()
    test_rows_and_flexion()
    test_symmetric_other()
    test_regions()
    test_shape_and_cut()
    test_simulated_transfer()
    test_push_core()
    test_pose_math()
    test_garment_bones()
    test_top4_in_fit()
    test_body_through()
    test_push_seeds_follow_contact()
    test_push_profile()
    test_apply_push_units()
    test_sweep_grammar()
    with tempfile.TemporaryDirectory() as tmp:
        test_cli(tmp)
    if FAILURES:
        for f in FAILURES:
            print("%s FAIL: %s" % (TOKEN, f))
        sys.exit(1)
    print("%s OK" % TOKEN)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    from _harness import run
    run(main, TOKEN)
