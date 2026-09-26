"""Synthetic headless test for avatarprep.core.weight_transfer and cli/transfer_weights.py.

Run: blender --background --factory-startup --python tests/test_weight_transfer.py
Prints WEIGHTS_TEST OK / WEIGHTS_TEST FAIL: <reason>.

Needs scipy and robust_laplacian from ``tools/provision_deps.py``. Without them the suite
FAILS rather than skipping, so an unprovisioned checkout cannot pass the gate.

Fixture: ``tests/_weight_fixture.py``. Each refusal is one tweak of it.
"""
import hashlib
import json
import os
import shlex
import shutil
import site
import subprocess
import sys
import tempfile

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOKEN = "WEIGHTS_TEST"
FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def refuses(fn, substr, label):
    from avatarprep.core.weight_transfer import WeightTransferError
    try:
        fn()
    except WeightTransferError as e:
        check(substr.lower() in str(e).lower(), "%s: refused but %r lacked %r" % (label, str(e), substr))
        return
    FAILURES.append("%s: expected a refusal mentioning %r" % (label, substr))


def _run(s, **kw):
    from avatarprep.core import weight_transfer as WT
    return WT.transfer_weights(s["body"], [s["garment"]], **kw)["targets"][0]


def _bone_sum(ob, arm, i):
    v = ob.data.vertices[i]
    names = [g.name for g in ob.vertex_groups]
    return sum(g.weight for g in v.groups if names[g.group] in arm.data.bones)


def test_invariants():
    import _weight_fixture as F
    s = F.build()
    g, tags = s["garment"], s["tags"]
    before = F.weights_by_name(g)
    row = _run(s)
    after = F.weights_by_name(g)
    for n in ("Flap", "Flap2", "Flap3", "Flap4", "Detail", "WriteMask"):
        check(after[n] == before[n], "group %s must be bit-identical" % n)
    kept = [i for i in range(len(tags)) if before["Flap"][i] >= 1.0 - 1e-4 or i == s["four"]]
    check(all(after[n][i] == before.get(n, [0.0] * len(tags))[i] for i in kept for n in after),
          "vertices at p=1 and the no-allowance vertex must be unchanged")
    check(row["kept_full"] == 1, "one vertex has no allowance (kept_full=%d)" % row["kept_full"])
    check(row["capped"] >= 1, "the two-garment-group midline must cap (capped=%d)" % row["capped"])
    m = row["matched_mask"]
    check(all(m[i] for i, t in enumerate(tags) if t == "flipped"), "the flipped patch matches by flip")
    check(row["matched_flipped"] >= 9, "matched_flipped counts the patch (%d)" % row["matched_flipped"])
    fl = [i for i, t in enumerate(tags) if t == "float"]
    check(not any(m[i] for i in fl), "the floating island is unmatched")
    check(any(abs(p["centre_mm"][1] + 280) < 1 and p["verts"] == 9 for p in row["unmatched_parts"]),
          "the floating island is reported as an unmatched part: %r" % row["unmatched_parts"])
    arm = s["garment_rig"]
    names = [x.name for x in g.vertex_groups]
    for i, v in enumerate(g.data.vertices):
        if i in kept:
            continue
        bones = [x for x in v.groups if names[x.group] in arm.data.bones and x.weight > 0]
        check(len(bones) <= 4, "vertex %d carries %d bone groups" % (i, len(bones)))
        if tags[i] != "float":
            check(abs(_bone_sum(g, arm, i) - 1.0) < 1e-5, "vertex %d bone weights sum to %g"
                  % (i, _bone_sum(g, arm, i)))
    check(all(abs(_bone_sum(g, arm, i) - 1.0) < 1e-5 for i in fl), "inpainted island is fully weighted")
    cuff = [i for i, t in enumerate(tags) if t == "cuff_L"]
    check(all(after["UpperLeg.L"][i] > 0.5 for i in cuff), "the left cuff follows UpperLeg.L")
    check(all(after.get("UpperLeg.R", [0.0] * len(tags))[i] == 0.0 for i in cuff), "no right-leg weight on the left cuff")
    half = [i for i, t in enumerate(tags) if before["Flap"][i] == 0.25]
    check(half and all(abs(_bone_sum(g, arm, i) - 1.0) < 1e-5 and
                       abs(sum(after[n][i] for n in after if n.startswith(("Hips", "UpperLeg", "LowerLeg"))) - 0.5) < 1e-5
                       for i in half), "a p=0.5 vertex takes body weight 0.5")


def test_rerun_reproduces():
    import _weight_fixture as F
    s = F.build()
    _run(s)
    first = F.weights_by_name(s["garment"])
    _run(s)
    second = F.weights_by_name(s["garment"])
    worst = max(abs(a - b) for n in first for a, b in zip(first[n], second.get(n, [0.0] * len(first[n]))))
    check(set(first) == set(second), "a rerun must not create groups")
    check(worst <= 1e-7, "a rerun reproduces its weights (worst %g)" % worst)


def test_no_flip_and_mask_and_smooth():
    import _weight_fixture as F
    s = F.build()
    row = _run(s, flip=False)
    check(not any(row["matched_mask"][i] for i, t in enumerate(s["tags"]) if t == "flipped"),
          "with flip off the inward patch is unmatched")
    s = F.build()
    before = F.weights_by_name(s["garment"])
    _run(s, mask="WriteMask")
    after = F.weights_by_name(s["garment"])
    out = [i for i in range(len(s["tags"])) if before["WriteMask"][i] <= 0.5]
    check(all(after[n][i] == before.get(n, [0.0] * len(s["tags"]))[i] for i in out for n in after),
          "vertices outside --mask are unchanged")
    check(any(after["UpperLeg.L"][i] > 0 for i in range(len(s["tags"])) if s["tags"][i] == "cuff_L"),
          "vertices inside --mask are written")
    s = F.build()
    row = _run(s, smooth=(4, 0.2))
    check(row["smoothed"] and row["smoothed"] > 0, "smoothing touches vertices near inpainted ones")


def test_shapes():
    import _weight_fixture as F
    s = F.build()
    _run(s, shapes=[(None, "Bulk", 1.0)])
    check(s["body"].data.shape_keys.key_blocks["Bulk"].value == 0.0, "the body's live value is untouched")
    refuses(lambda: _run(F.build(), shapes=[(None, "Nope", 1.0)]), "is on none of", "unknown bare shape")
    refuses(lambda: _run(F.build(), shapes=[("Other", "Bulk", 1.0)]), "neither the source nor a target",
            "scoped shape on an unlisted mesh")
    refuses(lambda: _run(F.build(), shapes=[(None, "Bulk", 2.0)]), "outside the slider range [0, 1]",
            "a --shape value Blender would clamp")
    s = F.build()
    s["body"].data.shape_keys.key_blocks["Bulk"].slider_max = 2.0
    _run(s, shapes=[(None, "Bulk", 2.0)])  # inside a widened range

    def keyed(body_bulk):
        s = F.build()
        s["garment"].shape_key_add(name="Basis")
        s["garment"].shape_key_add(name="Bulk", from_mix=False).value = 0.0
        s["body"].data.shape_keys.key_blocks["Bulk"].value = body_bulk
        return s
    _run(keyed(0.5), shapes=[(None, "Bulk", 0.5)])  # a bare --shape sets both: seated for the run
    refuses(lambda: _run(keyed(0.0), shapes=[("Body_Base", "Bulk", 1.0)]), "not seated",
            "a scoped --shape that unseats the run")


def _islanded_target(s):
    """A band on the left thigh and, 50 cm in front, a 64-vertex patch: more vertices than the
    point-cloud Laplacian's neighbour count, so its graph leaves the patch disconnected."""
    import _weight_fixture as F
    verts, faces, tags = [], [], []
    F.tube(F.LX, 0, F.RL + F.GAP, 0.60, 0.76, 8, 24, "band", verts, faces, tags)
    F.patch(-0.05, 0.05, -0.6, 0.95, 1.05, 7, False, "far", verts, faces, tags)
    ob = F.mesh("Islanded", verts, faces, s["garment_rig"])
    ob.vertex_groups.new(name="Hips").add(list(range(len(verts))), 1.0, 'REPLACE')
    ob.vertex_groups.new(name="BandOnly").add([i for i, t in enumerate(tags) if t == "band"], 1.0, 'REPLACE')
    return ob


def test_disconnected_island():
    import _weight_fixture as F
    from avatarprep.core import weight_transfer as WT
    s = F.build()
    ob = _islanded_target(s)
    refuses(lambda: WT.transfer_weights(s["body"], [ob]), "Islanded: the point-cloud Laplacian leaves 64 verts",
            "an island the Laplacian leaves with no matched vertex")
    WT.transfer_weights(s["body"], [ob], mask="BandOnly")  # masked out, the island is not written


def test_blend_and_frame():
    import _weight_fixture as F
    from avatarprep.core import weight_transfer as WT
    s = F.build()
    fr = WT.body_frame(s["body_rig"])
    check(abs(fr["forward"][1] + 1) < 1e-6 and abs(fr["lateral"][0] - 1) < 1e-6,
          "forward is -Y and lateral +X on this rig: %r %r" % (fr["forward"], fr["lateral"]))
    refuses(lambda: WT.body_frame(s["body_rig"], forward_bone="Nose"), "--forward Nose", "missing --forward bone")
    rep = WT.transfer_weights(s["body"], [s["garment"]], exclude=["upper*leg*"], exclude_max=0.35,
                              blend=("back", 0.0, 0.04), blend_lateral=(0.06, 0.04),
                              blend_smooth=(2, 0.5, 0.04))
    b = rep["targets"][0]["blend"]
    check(rep["excluded"]["bones"] == sorted(["UpperLeg.L", "UpperLeg.R", "LowerLeg.L", "LowerLeg.R",
                                              "Foot.L", "Foot.R", "Toes.L", "Toes.R"]),
          "a glob takes its descendants: %r" % rep["excluded"]["bones"])
    check(b["loops"] >= 4 and b["smoothed"] > 0, "leg-hole loops found and smoothed: %r" % b)
    check(b["narrowed"] + b["whole"] + b["ramp"] == rep["targets"][0]["verts"], "the blend split covers every vertex")
    check(b["narrowed"] > 0 and b["whole"] > 0, "both transfers contribute: %r" % b)
    refuses(lambda: WT.transfer_weights(s["body"], [s["garment"]], exclude=["upper*leg*"], exclude_max=0.35,
                                        blend=("Y", 0.0, 0.04)), "not one of", "a world axis")
    for bone in ("Foot.L", "Foot.R"):
        s["body_rig"].data.bones[bone].use_deform = False
    refuses(lambda: WT.transfer_weights(s["body"], [s["garment"]], exclude=["upper*leg*"], exclude_max=0.35,
                                        blend=("back", 0.0, 0.04)), "matches *foot*", "no foot bone")


def _closed_target(s):
    """A capped tube around the pelvis: no boundary loop at all."""
    import _weight_fixture as F
    verts, faces, tags = [], [], []
    F.tube(0, 0, F.RP + F.GAP, 0.92, 1.04, 6, 32, "suit", verts, faces, tags)
    faces.append(tuple(range(31, -1, -1)))
    faces.append(tuple(range(6 * 32, 7 * 32)))
    ob = F.mesh("Suit", verts, faces, s["garment_rig"])
    ob.vertex_groups.new(name="Hips").add(list(range(len(verts))), 1.0, 'REPLACE')
    return ob


def test_refusals():
    import _weight_fixture as F
    from avatarprep.core import weight_transfer as WT
    from avatarprep.core import scene_utils
    refuses(lambda: _run(F.build(), exclude=["Tail*"], exclude_max=0.35), "matches no deform bone", "no-match glob")
    refuses(lambda: _run(F.build(), exclude=["Hips"], exclude_max=0.0), "drops every source triangle",
            "--source-exclude-max 0 over every bone")

    s = F.build()
    suit = _closed_target(s)
    refuses(lambda: WT.transfer_weights(s["body"], [suit], exclude=["upper*leg*"], exclude_max=0.35,
                                        blend=("back", 0.0, 0.04), blend_smooth=(2, 0.5, 0.04)),
            "no boundary loop", "closed garment with smoothing")
    WT.transfer_weights(s["body"], [suit], exclude=["upper*leg*"], exclude_max=0.35,
                        blend=("back", 0.0, 0.04), blend_smooth=(2, 0.5, 0.04), allow_no_loops=True)

    for variant in ("plain", "masked"):
        s = F.build()
        rig = s["garment_rig"]
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        rig.data.edit_bones.remove(rig.data.edit_bones["UpperLeg.L"])
        bpy.ops.object.mode_set(mode='OBJECT')
        kw = {}
        if variant == "masked":
            vg = s["garment"].vertex_groups.new(name="FrontR")
            vg.add([i for i, v in enumerate(s["garment"].data.vertices)
                    if s["tags"][i] == "band" and v.co.x < -0.05 and v.co.z > 1.0], 1.0, 'REPLACE')
            kw["mask"] = "FrontR"
        before = F.weights_by_name(s["garment"])
        refuses(lambda: _run(s, **kw), "UpperLeg.L", "target armature missing a body bone (%s)" % variant)
        check(F.weights_by_name(s["garment"]) == before, "a refusal leaves the weights untouched (%s)" % variant)

    s = F.build()
    pelvis = [i for i, v in enumerate(s["body"].data.vertices) if v.co.z > 0.95 and abs(v.co.x) < 0.2
              and (v.co.x ** 2 + v.co.y ** 2) > 0.15 ** 2]
    for grp in s["body"].vertex_groups:
        grp.remove(pelvis)
    refuses(lambda: _run(s), "carry no transferred body weight", "unweighted source under written vertices")

    s = F.build()
    s["garment"].modifiers.new("Tri", 'TRIANGULATE')
    refuses(lambda: _run(s), "modifier changes the polygon count", "topology modifier")

    s = F.build()
    s["body"][scene_utils.STAMP_BAKED] = {"Bulk": 0.2}
    refuses(lambda: _run(s), "not seated", "baked-map disagreement")
    _run(s, allow_unseated=True)


def test_blend_regions():
    """Past the ramp the blend is exactly one of its two transfers: back-midline vertices
    draw at most exclude_max from the leg bones, and front vertices and side vertices past
    the lateral fade equal a plain whole-source run."""
    import numpy as np
    import _weight_fixture as F
    from avatarprep.core import weight_transfer as WT
    blend, lateral, xmax = ("back", 0.0, 0.04), (0.06, 0.04), 0.35
    s = F.build()
    row = WT.transfer_weights(s["body"], [s["garment"]], exclude=["upper*leg*"], exclude_max=xmax,
                              blend=blend, blend_lateral=lateral)["targets"][0]
    got = F.weights_by_name(s["garment"])
    frame = WT.body_frame(s["body_rig"])
    TV = np.array([tuple(s["garment"].matrix_world @ v.co) for v in s["garment"].data.vertices])
    mix = WT.blend_mix(TV, frame, *blend, lateral=lateral)
    s2 = F.build()
    WT.transfer_weights(s2["body"], [s2["garment"]])
    whole = F.weights_by_name(s2["garment"])
    n = len(TV)
    legs = [k for k in got if k.startswith(("UpperLeg", "LowerLeg", "Foot", "Toes"))]
    garment = [k for k in got if k.startswith("Flap")]
    back = [i for i in range(n) if mix[i] >= 0.99 and row["matched_mask"][i] and not any(got[k][i] for k in garment)]
    check(len(back) > 10, "the back midline has narrowed vertices (%d)" % len(back))
    worst = max((sum(got[k][i] for k in legs) / (sum(got[k][i] for k in legs) + got["Hips"][i]) for i in back),
                default=0.0)
    check(worst <= xmax + 1e-6, "back-midline vertices draw at most %g from the legs (worst %.4f)" % (xmax, worst))
    d = TV - frame["origin"]
    front = [i for i in range(n) if d[i] @ frame["back"] < blend[1] - blend[2] / 2]
    side = [i for i in range(n) if d[i] @ frame["back"] > blend[1] + blend[2] / 2
            and abs(d[i] @ frame["lateral"]) > lateral[0] + lateral[1] / 2]
    check(front and side, "front (%d) and side (%d) vertices exist past the ramps" % (len(front), len(side)))
    diff = max(abs(got.get(k, [0.0] * n)[i] - whole.get(k, [0.0] * n)[i])
               for i in front + side for k in set(got) | set(whole))
    check(diff <= 1e-5, "front and side vertices equal the whole-source run (worst %g)" % diff)


def test_more_refusals():
    import _weight_fixture as F
    from avatarprep.core import weight_transfer as WT
    s = F.build()
    s["garment"].hide_viewport = True
    refuses(lambda: _run(s), "is not evaluated", "a hidden target")
    s = F.build()
    s["body"].hide_viewport = True
    s["body"].data.shape_keys.key_blocks["Bulk"].value = 1.0
    hidden = _run(s, allow_unseated=True)
    s = F.build()
    s["body"].data.shape_keys.key_blocks["Bulk"].value = 1.0
    shown = _run(s, allow_unseated=True)
    check(hidden["contact_mean_dw"] == shown["contact_mean_dw"],
          "a hidden body evaluates with its live keys like a shown one (%r vs %r)"
          % (hidden["contact_mean_dw"], shown["contact_mean_dw"]))
    s = F.build()
    s["garment"].vertex_groups.new(name="Nothing")
    refuses(lambda: _run(s, mask="Nothing"), "no vertex to write", "an empty write set")

    s = F.build()
    kb = s["body"].shape_key_add(name="Blink", from_mix=False)
    kb.value = 1.0
    _run(s)  # a live key only the body carries is not a seat
    s = F.build()
    s["garment"].shape_key_add(name="Basis")
    s["garment"].shape_key_add(name="Bulk", from_mix=False).value = 0.0
    s["body"].data.shape_keys.key_blocks["Bulk"].value = 0.5
    refuses(lambda: _run(s), "not seated", "a live key both carry at different values")

    s = F.build()
    refuses(lambda: _run(s, exclude=["upper*leg*"], exclude_max=0.35, blend=("back", 0.0, 0.04),
                         reference_bone="Pelvis"), "--reference-bone Pelvis", "a missing reference bone")
    for bone in ("Foot.L", "Foot.R"):
        s["body_rig"].data.bones[bone].use_deform = False
    rep_ = WT.transfer_weights(s["body"], [s["garment"]], exclude=["upper*leg*"], exclude_max=0.35,
                               blend=("back", 0.0, 0.04), forward_bone="Foot.L")
    check(abs(rep_["frame"]["forward"][1] + 1) < 1e-6 and rep_["frame"]["from"] == "--forward Foot.L",
          "--forward reads the named bone: %r" % rep_["frame"])


def test_recipe_round_trip():
    """A recipe with a negative value re-parses to itself."""
    from cli import transfer_weights as D
    a = D._parse_args(["--in", "x.blend", "--targets", "Shorts", "--source-exclude", "upper*leg*",
                       "--source-exclude-max", "0.35", "--exclude-blend", "back,-0.06,0.03",
                       "--exclude-blend-lateral=-0.02,0.04", "--whatif"])
    line = D.recipe(a)
    check("--exclude-blend-lateral=-0.02,0.04" in line, "a negative value is joined with '=': %s" % line)
    again = D.recipe(D._parse_args(shlex.split(line)[1:] + ["--in", "x.blend", "--whatif"]))
    check(again == line, "the recipe re-parses to itself: %s vs %s" % (line, again))


def _blender(args, env=None):
    cmd = [bpy.app.binary_path, "--background", "--factory-startup", "--python",
           os.path.join(REPO, "cli", "transfer_weights.py"), "--"] + args
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def test_cli(tmp):
    import _weight_fixture as F
    from avatarprep.core import scene_utils
    F.build()
    src = os.path.join(tmp, "costume.blend")
    bpy.ops.wm.save_as_mainfile(filepath=src)
    out = os.path.join(tmp, "out.blend")
    viz = os.path.join(tmp, "viz.blend")
    rc, txt = _blender(["--in", src, "--targets", "Shorts", "--max-distance", "0.04", "--viz", viz,
                        "--report", os.path.join(tmp, "r.json"), "--out", out])
    check(rc == 0 and "=> OK" in txt and "saved=" in txt, "a saving run exits 0 with OK (rc=%d)\n%s" % (rc, txt))
    check("recipe: transfer_weights --targets Shorts --max-distance 0.04" in txt, "the recipe line is printed\n%s" % txt)
    check("Traceback" not in txt, "no traceback in a clean run\n%s" % txt)
    bpy.ops.wm.open_mainfile(filepath=out)
    ob = bpy.data.objects["Shorts"]
    me = ob.data
    check(ob.get(scene_utils.STAMP_WEIGHTS) == "transfer_weights --targets Shorts --max-distance 0.04",
          "the target object carries the recipe: %r" % ob.get(scene_utils.STAMP_WEIGHTS))
    rep = scene_utils.report_stamps(bpy.context.scene)
    entry = [m for a in rep["armatures"] for m in a["meshes"] if m["name"] == "Shorts"]
    check(entry and entry[0].get("weights") == ob.get(scene_utils.STAMP_WEIGHTS) and "baked" not in entry[0],
          "report_stamps lists a weights-only mesh under its armature: %r" % rep)
    check(me.color_attributes.get("avatarprep_matched") is None, "the deliverable carries no review layer")
    p = subprocess.run([bpy.app.binary_path, "--background", "--factory-startup", "--python",
                        os.path.join(REPO, "cli", "report_stamps.py"), "--", "--in", out],
                       capture_output=True, text=True)
    txt = p.stdout + p.stderr
    check(p.returncode == 0 and "mesh Shorts weights=transfer_weights --targets Shorts" in txt
          and "Traceback" not in txt, "report_stamps prints the weights-only mesh\n%s" % txt)
    bpy.ops.wm.open_mainfile(filepath=viz)
    check(bpy.data.objects["Shorts"].data.color_attributes.get("avatarprep_matched") is not None,
          "the --viz copy carries the review layer")
    with open(os.path.join(tmp, "r.json"), encoding="utf-8") as fh:
        r = json.load(fh)
    row = r["targets"][0] if r.get("targets") else {}
    check(r.get("recipe") == "transfer_weights --targets Shorts --max-distance 0.04" and r.get("saved") == out
          and row.get("mesh") == "Shorts" and row.get("written", 0) > 0 and "matched_mask" not in row
          and r.get("viz") == viz, "the --report carries the recipe, the paths and the per-mesh row: %r" % r)

    def digest(path):
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    before = digest(src)
    rc, txt = _blender(["--in", src, "--targets", "Shorts", "--whatif"])
    check(digest(src) == before, "--whatif leaves --in byte-identical")
    inplace = os.path.join(tmp, "inplace.blend")
    shutil.copyfile(src, inplace)
    rc, txt = _blender(["--in", inplace, "--targets", "Shorts", "--in-place"])
    check(rc == 0 and "saved=%s" % inplace in txt, "--in-place saves over --in\n%s" % txt)
    bpy.ops.wm.open_mainfile(filepath=inplace)
    check(bpy.data.objects["Shorts"].get(scene_utils.STAMP_WEIGHTS) == "transfer_weights --targets Shorts",
          "the in-place save carries the stamp")
    rc, txt = _blender(["--in", src, "--targets", "Shorts", "--viz", out, "--out", out])
    check(rc == 2 and "--viz" in txt, "--viz naming --out is bad args\n%s" % txt)
    rc, txt = _blender(["--in", src, "--targets", "Shorts", "--source-exclude", "upper*leg*",
                        "--source-exclude-max", "0.35", "--exclude-blend", "back,0,0.04",
                        "--exclude-blend-smooth", "2,2.0", "--whatif"])
    check(rc == 2 and "ALPHA" in txt, "an out-of-range --exclude-blend-smooth alpha is bad args\n%s" % txt)

    ghost = os.path.join(tmp, "ghost.blend")
    rc, txt = _blender(["--in", src, "--targets", "Shorts", "--whatif"])
    check(rc == 0 and "whatif; nothing saved" in txt, "--whatif exits 0 and saves nothing\n%s" % txt)
    rc, txt = _blender(["--in", src, "--targets", "Shorts", "--whatif", "--out", ghost])
    check(rc == 2 and not os.path.exists(ghost), "--whatif with --out is a bad-args exit 2\n%s" % txt)
    rc, txt = _blender(["--in", src, "--targets", "Shorts", "--source-exclude", "upper*leg*",
                        "--source-exclude-max", "0.35", "--exclude-blend=-Y,-0.06,0.03", "--whatif"])
    check(rc == 2 and "world axis" in txt, "a world axis is refused as bad args\n%s" % txt)
    rc, txt = _blender(["--in", src, "--targets", "Shorts", "--source-exclude", "upper*leg*", "--whatif"])
    check(rc == 2 and "go together" in txt, "--source-exclude without its max is bad args\n%s" % txt)
    rc, txt = _blender(["--in", src, "--targets", "Shorts", "--source-exclude", "Tail*",
                        "--source-exclude-max", "0.3", "--whatif"])
    check(rc == 1 and "=> FAIL:" in txt and "matches no deform bone" in txt, "a refusal exits 1\n%s" % txt)
    rc, txt = _blender(["--in", src, "--targets", "Nope", "--whatif"])
    check(rc == 2, "an unknown target exits 2\n%s" % txt)


def main():
    site.addsitedir(os.environ.get("AVATARPREP_DEPS") or os.path.join(REPO, "deps"))
    try:
        import scipy.sparse.linalg  # noqa: F401
        import robust_laplacian  # noqa: F401
    except ImportError as e:
        print("%s FAIL: scipy/robust_laplacian do not import (%s); run python tools/provision_deps.py "
              "--blender <blender.exe>" % (TOKEN, e))
        sys.exit(1)
    for p in (REPO, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    test_invariants()
    test_rerun_reproduces()
    test_no_flip_and_mask_and_smooth()
    test_shapes()
    test_disconnected_island()
    test_blend_and_frame()
    test_blend_regions()
    test_refusals()
    test_more_refusals()
    test_recipe_round_trip()
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
