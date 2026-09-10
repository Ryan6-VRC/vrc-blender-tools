"""Synthetic headless test for avatarprep.core.coverage and the mark_coverage door.

Run: blender --background --factory-startup --python tests/test_coverage.py
Prints COVERAGE_TEST OK / COVERAGE_TEST FAIL: <reason>; exit 1 on any failure.

Fixtures are a cylinder "body" and tubes around it built in memory, weighted by name to
bones of a small armature, so every case pins one clause of the criterion or of the
carrier rule:

  * a closed tube covers the cylinder band it wraps (the base case);
  * a double-walled tube whose inner wall's normals face the body still covers — nothing
    reads a garment normal;
  * a short open tube leaves the skin beyond and at its edge uncovered (rays escape
    through the opening); at 75 degrees the row under the edge is reachable at a grazing
    angle and counts as rest-visible, at 85 it does not; a tube standing far off the skin
    is seen into;
  * a tube band on a bone the body lacks is transparent (declines as swings); a
    blend-ratio difference on the body's own bones still covers;
  * a farther co-moving garment covers where a nearer swinging one is transparent;
  * kin: a band on a child bone declines at kin 0 and covers at kin 1; fold: a band on a
    'Breast_*' bone covers at kin 0 once folded onto its parent;
  * shapes: a body key that swells the top half out past the tube uncovers it when set,
    on a stand-in — the body's own key value is untouched; an unknown shape refuses;
  * the carrier is polygon-level: one uncovered corner keeps its whole quad;
  * a one-quad-wide covered strip is residue: covered, carrier empty there;
  * --cut-shape excludes what a shape already moves, and an unknown cut shape refuses;
  * write_carrier lands the key on a local body, refuses a moved Basis and a same-named
    key; save_marked writes the inspect copy; the door's --whatif / --out / --out-marked
    paths and exit codes hold via subprocess, on a LINKED body.
"""
import math
import os
import subprocess
import sys
import tempfile

import bpy
from mathutils import Vector

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def _repo_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


BONES = {"Hips": None, "Spine": "Hips", "Chest": "Spine", "Breast_Root": "Chest", "Breast_1": "Breast_Root",
         "Skirt_Root": "Hips", "Skirt_1": "Skirt_Root", "Other": None}


def _armature(name="Rig", bones=BONES):
    arm = bpy.data.armatures.new(name)
    ob = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = {}
    for b in bones:
        eb = arm.edit_bones.new(b)
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, 0.0, 0.1)
        ebs[b] = eb
    for b, p in bones.items():
        if p:
            ebs[b].parent = ebs[p]
    bpy.ops.object.mode_set(mode='OBJECT')
    return ob


def _rig(ob, arm):
    m = ob.modifiers.new("Armature", 'ARMATURE')
    m.object = arm


def _cylinder(name, radius, z0, z1, segs=24, rings=8, weights=None, flip=False):
    """Open cylinder along +Z, outward normals (inward when ``flip``). ``weights`` is a
    function ring_index -> {group: w} or None for a single 'Spine' group."""
    verts, faces = [], []
    for r in range(rings + 1):
        z = z0 + (z1 - z0) * r / rings
        for s in range(segs):
            a = 2 * math.pi * s / segs
            verts.append((radius * math.cos(a), radius * math.sin(a), z))
    for r in range(rings):
        for s in range(segs):
            a = r * segs + s
            b = r * segs + (s + 1) % segs
            c = (r + 1) * segs + (s + 1) % segs
            d = (r + 1) * segs + s
            faces.append((a, d, c, b) if flip else (a, b, c, d))
    me = bpy.data.meshes.new(name + "Data")
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    groups = {}
    for r in range(rings + 1):
        w = weights(r) if weights else {"Spine": 1.0}
        for s in range(segs):
            for g, val in w.items():
                if g not in groups:
                    groups[g] = ob.vertex_groups.new(name=g)
                groups[g].add([r * segs + s], val, 'REPLACE')
    return ob


def _scene(body_weights=None, tube_kw=None, rig=True):
    """A body cylinder and a closed tube around it, both rigged to one armature."""
    _clear()
    arm = _armature() if rig else None
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20, weights=body_weights)
    kw = dict(radius=0.11, z0=-0.2, z1=1.2, rings=8)
    kw.update(tube_kw or {})
    tube = _cylinder("Tube", kw["radius"], kw["z0"], kw["z1"], rings=kw["rings"], weights=kw.get("weights"),
                     flip=kw.get("flip", False))
    if rig:
        _rig(body, arm)
        _rig(tube, arm)
    return body, tube


def _measure(body, garments, **kw):
    from avatarprep.core import coverage
    args = dict(cone_deg=75.0, share=0.5, reach=1.0, kin=0, cut_threshold=0.01)
    args.update(kw)
    return coverage.measure(body, garments, **args)


def test_closed_tube_covers():
    body, tube = _scene()
    r = _measure(body, [tube])
    check(r["covered_triangles"] == r["triangles"],
          "closed tube should cover every body triangle, got %d/%d (declined %s)"
          % (r["covered_triangles"], r["triangles"], r["declined"]))
    check(r["realised_triangles"] == r["triangles"] and r["residue_triangles"] == 0,
          "closed tube: realised %d residue %d" % (r["realised_triangles"], r["residue_triangles"]))
    check(r["rest_visible_triangles"] == 0, "closed tube: nothing removed is rest-visible, got %d"
          % r["rest_visible_triangles"])
    check(r["by_garment"]["Tube"] == r["vertex_count"], "every vertex claimed by the tube")


def test_double_wall_ignores_garment_normals():
    body, outer = _scene(tube_kw=dict(radius=0.112))
    inner = _cylinder("Inner", 0.105, -0.2, 1.2, rings=8, flip=True)  # lining, normals toward the body
    with bpy.context.temp_override(active_object=outer, selected_editable_objects=[outer, inner]):
        bpy.ops.object.join()
    r = _measure(body, [outer])
    check(r["covered_triangles"] == r["triangles"],
          "double wall should still cover fully, got %d/%d (declined %s)"
          % (r["covered_triangles"], r["triangles"], r["declined"]))


def test_open_tube_and_loose_collar():
    body, tube = _scene(tube_kw=dict(z1=0.5))   # open top at z=0.5, mid-body
    r = _measure(body, [tube])
    check(r["declined"].get("escaped", 0) > 0, "rays should escape through the opening, declined=%s" % r["declined"])
    check(0 < r["covered_triangles"] < r["triangles"], "open tube should cover part, got %d/%d"
          % (r["covered_triangles"], r["triangles"]))
    above = [i for i, c in enumerate(r["covered"]) if c and body.data.vertices[i].co.z > 0.5]
    check(not above, "vertices beyond the opening were marked covered: %d" % len(above))
    ring_at = [i for i, c in enumerate(r["covered"]) if c and abs(body.data.vertices[i].co.z - 0.5) < 1e-6]
    check(not ring_at, "the ring at the opening is seen into and must not be covered: %d" % len(ring_at))
    deep = [i for i, c in enumerate(r["covered"]) if c and body.data.vertices[i].co.z < 0.2]
    check(len(deep) > 0, "skin well below the opening should still be covered")
    # the row just under the edge is reachable at a grazing angle the 75-degree cone does
    # not sweep: the cost of the narrower cone, and what the count is for
    check(r["rest_visible_triangles"] == 24 * 2, "open tube at 75: the row under the edge is rest-visible, got %d"
          % r["rest_visible_triangles"])
    r85 = _measure(body, [tube], cone_deg=85.0)
    check(r85["rest_visible_triangles"] == 0 and r85["realised_triangles"] < r["realised_triangles"],
          "at 85 the sweep and the criterion agree: rest-visible %d, realised %d vs %d"
          % (r85["rest_visible_triangles"], r85["realised_triangles"], r["realised_triangles"]))
    # a collar standing 40 mm off the skin over a short band is seen into from both openings
    body, collar = _scene(tube_kw=dict(radius=0.14, z0=0.6, z1=0.9, rings=4))
    r = _measure(body, [collar])
    check(r["covered_triangles"] < r["triangles"] * 0.2,
          "a loose collar should leave most of the skin under it visible, covered %d/%d"
          % (r["covered_triangles"], r["triangles"]))


def test_swinging_bones_and_blend_ratio():
    body, tube = _scene(tube_kw=dict(weights=lambda r: {"Spine": 1.0} if r < 4 else {"Skirt_1": 1.0}))
    r = _measure(body, [tube], kin=1)      # kin 1 from Spine reaches Hips and Chest, not Skirt_1
    check(r["declined"].get("swings", 0) > 0, "the skirt-boned band is transparent, declined=%s" % r["declined"])
    check(0 < r["covered_triangles"] < r["triangles"], "skirt case should cover part only, got %d/%d"
          % (r["covered_triangles"], r["triangles"]))
    check(r["rest_visible_triangles"] == 0, "swinging band: removed set stays rest-invisible, got %d"
          % r["rest_visible_triangles"])
    body, tube = _scene(body_weights=lambda r: {"Spine": 0.9, "Hips": 0.1},
                        tube_kw=dict(weights=lambda r: {"Spine": 0.4, "Hips": 0.6}))
    r = _measure(body, [tube])
    check(r["covered_triangles"] == r["triangles"],
          "a blend-ratio difference on body bones must not decline, got %d/%d (declined %s)"
          % (r["covered_triangles"], r["triangles"], r["declined"]))


def test_farther_comoving_garment_covers():
    body, bad = _scene(tube_kw=dict(radius=0.105, weights=lambda r: {"Other": 1.0}))  # nearer, wrong bone
    good = _cylinder("Good", 0.112, -0.2, 1.2, rings=8)
    r = _measure(body, [bad, good])
    check(r["covered_triangles"] == r["triangles"],
          "a farther co-moving garment must cover where the nearer one swings, got %d/%d"
          % (r["covered_triangles"], r["triangles"]))
    check(r["by_garment"]["Good"] == r["vertex_count"] and r["by_garment"]["Tube"] == 0,
          "the co-moving garment claims, the transparent one never does: %s" % r["by_garment"])


def test_kin_and_fold():
    from avatarprep.core import coverage
    body, tube = _scene(body_weights=lambda r: {"Chest": 1.0},
                        tube_kw=dict(weights=lambda r: {"Breast_1": 1.0}))
    r0 = _measure(body, [tube], kin=0)
    r1 = _measure(body, [tube], kin=1)
    r2 = _measure(body, [tube], kin=2)
    check(r0["covered_triangles"] == 0 and r1["covered_triangles"] == 0,
          "a grandchild bone is not kin at 0 or 1 step: %d, %d" % (r0["covered_triangles"], r1["covered_triangles"]))
    check(r2["covered_triangles"] == r2["triangles"], "two steps reach Chest -> Breast_Root -> Breast_1: %d/%d"
          % (r2["covered_triangles"], r2["triangles"]))
    rf = _measure(body, [tube], kin=0, fold=["Breast_*"])
    check(rf["covered_triangles"] == rf["triangles"], "folding Breast_* onto Chest covers at kin 0: %d/%d"
          % (rf["covered_triangles"], rf["triangles"]))
    # kin needs a rig; unrigged meshes must say so rather than silently act as kin 0
    body, tube = _scene(rig=False)
    try:
        _measure(body, [tube], kin=2)
        check(False, "kin without an armature should refuse")
    except coverage.CoverageError as e:
        check("kin" in str(e), "kin refusal should name kin, got %s" % e)
    check(_measure(body, [tube], kin=0)["covered_triangles"] > 0, "kin 0 works unrigged")


def test_shapes_on_stand_in():
    from avatarprep.core import coverage
    body, tube = _scene()
    body.shape_key_add(name="Basis", from_mix=False)
    swell = body.shape_key_add(name="Swell", from_mix=False)
    swell.value = 0.0                              # a fresh key starts live in this Blender
    for i, v in enumerate(body.data.vertices):
        if v.co.z > 0.5:
            swell.data[i].co = v.co * 1.3          # top half out past the tube (0.13 > 0.11)
    r_plain = _measure(body, [tube])
    r_swell = _measure(body, [tube], shapes={"Swell": 1.0})
    check(r_plain["covered_triangles"] == r_plain["triangles"], "unset shape: fully covered")
    check(r_swell["covered_triangles"] < r_plain["covered_triangles"] * 0.6,
          "with the swell set the top half stands outside the tube and must uncover: %d vs %d"
          % (r_swell["covered_triangles"], r_plain["covered_triangles"]))
    check(body.data.shape_keys.key_blocks["Swell"].value == 0.0, "the body's own key value must be untouched")
    check(not any(o.name.startswith("__coverage_") for o in bpy.data.objects), "stand-ins must be removed")
    check(r_swell["settings"]["shapes"] == {"Swell": 1.0}, "shapes echo in settings")
    try:
        _measure(body, [tube], shapes={"Nope": 1.0})
        check(False, "a shape on no listed mesh should refuse")
    except coverage.CoverageError as e:
        check("Nope" in str(e), "shape refusal should name it, got %s" % e)


def test_polygon_carrier_keeps_uncovered_corner():
    """One body vertex made unweighted declines; every quad around it must survive, so
    the removed set never contains a polygon with an uncovered corner."""
    body, tube = _scene()
    victim = 10 * 24 + 5                     # a mid-body vertex
    for g in body.vertex_groups:
        g.remove([victim])
    r = _measure(body, [tube])
    check(not r["covered"][victim] and r["declined"].get("unweighted") == 1, "the victim declines as unweighted")
    polys = [tuple(p.vertices) for p in body.data.polygons]
    around = {pi for pi, p in enumerate(polys) if victim in p}
    removed = set(r["removed_polygons"])
    check(not (around & removed), "quads around an uncovered corner must not be removed: %d were"
          % len(around & removed))
    check(all(all(r["covered"][v] for v in polys[pi]) for pi in removed),
          "every removed polygon must be fully covered")
    check(r["covered_triangles"] == r["triangles"] - 2 * len(around) and r["residue_triangles"] == 0,
          "the ring of quads around the victim is not covered at all, got covered %d residue %d"
          % (r["covered_triangles"], r["residue_triangles"]))


def test_residue_strip():
    """A one-quad-wide covered band: the tube rides a bone the body lacks except over a
    narrow band on the body's bone, so exactly two vertex rows are covered, one row of
    quads is covered, and no vertex of it has every surrounding quad covered. Covered,
    carrier empty, all of it residue."""
    body, tube = _scene(tube_kw=dict(rings=28))
    for g in list(tube.vertex_groups):
        tube.vertex_groups.remove(g)
    tspine = tube.vertex_groups.new(name="Spine")
    tother = tube.vertex_groups.new(name="Other")
    for i, v in enumerate(tube.data.vertices):
        (tspine if 0.44 < v.co.z < 0.52 else tother).add([i], 1.0, 'REPLACE')
    r = _measure(body, [tube])
    check(r["covered_triangles"] == 24 * 2, "band should cover one ring of quads, got %d (declined %s)"
          % (r["covered_triangles"], r["declined"]))
    check(r["realised_triangles"] == 0 and r["residue_triangles"] == 24 * 2 and not r["carrier"],
          "a one-quad band is all residue, got realised %d residue %d carrier %d"
          % (r["realised_triangles"], r["residue_triangles"], len(r["carrier"])))


def test_cut_shape_exclusion_and_unknown():
    from avatarprep.core import coverage
    body, tube = _scene()
    body.shape_key_add(name="Basis", from_mix=False)
    cutk = body.shape_key_add(name="Lower_OFF", from_mix=False)
    cutk.value = 0.0
    for i, v in enumerate(body.data.vertices):
        if v.co.z < 0.3:
            cutk.data[i].co = v.co * 0.5   # moves > 10 mm
    r = _measure(body, [tube], cut_shapes=["Lower_OFF"])
    check(r["already_cut_triangles"] > 0, "cut shape should mark triangles already cut")
    check(r["covered_triangles"] + r["already_cut_triangles"] == r["triangles"],
          "cut + covered should partition a fully covered body: %d + %d != %d"
          % (r["covered_triangles"], r["already_cut_triangles"], r["triangles"]))
    try:
        _measure(body, [tube], cut_shapes=["Nope_OFF"])
        check(False, "unknown cut shape should refuse")
    except coverage.CoverageError as e:
        check("Nope_OFF" in str(e), "unknown cut shape refusal should name it, got %s" % e)


def test_write_carrier_and_gates():
    from avatarprep.core import coverage
    body, tube = _scene()
    r = _measure(body, [tube])
    kb = coverage.write_carrier(body.data, r, shape_name="Cover_T", delta=0.02)
    check(kb.name == "Cover_T" and kb.value == 0.0, "carrier key should exist at value 0")
    basis = body.data.shape_keys.key_blocks[0]
    moved = sum(1 for i in range(len(body.data.vertices))
                if (kb.data[i].co - basis.data[i].co).length > 0.01)
    check(moved == len(r["carrier"]), "carrier key should move exactly the carrier: %d vs %d"
          % (moved, len(r["carrier"])))
    try:
        coverage.write_carrier(body.data, r, shape_name="Cover_T", delta=0.02)
        check(False, "same-named key should refuse without replace")
    except coverage.CoverageError:
        pass
    coverage.write_carrier(body.data, r, shape_name="Cover_T", delta=0.02, replace=True)
    basis.data[0].co = basis.data[0].co + Vector((0.001, 0, 0))
    try:
        coverage.write_carrier(body.data, r, shape_name="Cover_U", delta=0.02)
        check(False, "moved Basis should refuse")
    except coverage.CoverageError as e:
        check("Basis" in str(e), "moved-Basis refusal should say so, got %s" % e)


def test_marked_copy_and_save(tmp):
    from avatarprep.core import coverage
    body, tube = _scene(tube_kw=dict(z1=0.5))
    r = _measure(body, [tube])
    ob = coverage.marked_copy(body, r, name="M", garment_order=["Tube"])
    check(ob.data.color_attributes.active_color is not None, "marked copy needs an active colour")
    check(len(ob.data.vertices) == len(body.data.vertices), "marked copy keeps geometry")
    coverage.remove_marked_copy(ob)
    ob2 = coverage.marked_copy(body, r, name="R", garment_order=["Tube"], remove_carrier=True)
    check(len(ob2.data.polygons) == len(body.data.polygons) - len(r["removed_polygons"]),
          "removed copy should drop exactly the removed polygons: %d vs %d - %d"
          % (len(ob2.data.polygons), len(body.data.polygons), len(r["removed_polygons"])))
    coverage.remove_marked_copy(ob2)
    path = os.path.join(tmp, "marked.blend")
    coverage.save_marked(path, body, r, [tube], label="Cover_T")
    check(os.path.exists(path), "save_marked writes the file")
    check(not bpy.data.filepath, "save_marked leaves the open file's path alone (copy)")
    check(bpy.data.objects.get("Cover_T_removed") is None, "save_marked removes its copies afterwards")
    check(not body.hide_get(), "save_marked restores hidden state")
    with bpy.data.libraries.load(path) as (df, dt):
        names = list(df.objects)
    check("Cover_T_removed" in names and "Cover_T_marked" in names and "Tube" in names,
          "the saved file holds the two copies and the garment, got %s" % names)


def test_cli(tmp):
    """The door through a fresh blender: --whatif writes nothing, --shape on a LINKED body
    measures on a stand-in, --out writes the key onto that body's own blend, --out-marked
    saves the inspect copy, bad args exit 2 in-grammar."""
    root = _repo_root()
    blender = bpy.app.binary_path
    # base blend: the rigged body with a swell key
    _clear()
    arm = _armature()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    _rig(body, arm)
    body.shape_key_add(name="Basis", from_mix=False)
    swell = body.shape_key_add(name="Swell", from_mix=False)
    swell.value = 0.0
    for i, v in enumerate(body.data.vertices):
        if v.co.z > 0.5:
            swell.data[i].co = v.co * 1.3
    base = os.path.join(tmp, "base.blend")
    bpy.ops.wm.save_as_mainfile(filepath=base)
    # costume blend: the tube on its own rig, with the body linked in
    _clear()
    arm = _armature()
    tube = _cylinder("Tube", 0.11, -0.2, 1.2, rings=8)
    _rig(tube, arm)
    with bpy.data.libraries.load(base, link=True, relative=False) as (df, dt):
        dt.objects = ["Body"]
    for o in dt.objects:
        bpy.context.scene.collection.objects.link(o)
    costume = os.path.join(tmp, "costume.blend")
    bpy.ops.wm.save_as_mainfile(filepath=costume)

    def run(args):
        cmd = [blender, "--background", "--factory-startup", "--python",
               os.path.join(root, "cli", "mark_coverage.py"), "--"] + args
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode, (p.stdout or "") + (p.stderr or "")

    base_args = ["--in", costume, "--body", "Body", "--garments", "Tube", "--shape-name", "Cover_T"]
    report = os.path.join(tmp, "r.json")
    marked = os.path.join(tmp, "marked", "m.blend")
    rc, out = run(base_args + ["--whatif", "--report", report, "--out-marked", marked])
    check(rc == 0 and "markcoverage Cover_T" in out and "=> OK" in out, "whatif should pass: %d\n%s" % (rc, out))
    check(os.path.exists(report), "whatif should write the report")
    check(os.path.exists(marked) and "marked=" in out, "whatif should save --out-marked and name it")
    check("png=" not in out, "no render requested, no png= trailer")
    check("restvisible=0" in out, "closed tube: restvisible=0 on the result line, got\n%s" % out)
    full = [l for l in out.splitlines() if "=> OK" in l][0]
    rc, out = run(base_args + ["--whatif", "--shape", "Swell=1"])
    check(rc == 0 and "=> OK" in out, "shape on a linked body should measure on a stand-in: %d\n%s" % (rc, out))
    swelled = [l for l in out.splitlines() if "=> OK" in l][0]
    def realised(line):
        return int(line.split("realised=")[1].split()[0])
    check(realised(swelled) < realised(full), "the swell must uncover the top half: %s vs %s" % (swelled, full))

    out_blend = os.path.join(tmp, "base_marked.blend")
    rc, out = run(base_args + ["--out", out_blend])
    check(rc == 0 and "saved=" in out, "write should pass and name the save: %d\n%s" % (rc, out))
    check(os.path.exists(out_blend), "write should save --out")
    bpy.ops.wm.open_mainfile(filepath=out_blend)
    me = bpy.data.objects["Body"].data
    check(me.shape_keys is not None and "Cover_T" in me.shape_keys.key_blocks,
          "the saved base copy should carry the carrier key")
    check(me.library is None, "the key must land on the base's own local mesh")

    rc, out = run(["--in", costume, "--body", "Body", "--garments", "Nope", "--shape-name", "X", "--whatif"])
    check(rc == 2 and "ERROR" in out, "missing garment should ERROR exit 2: %d\n%s" % (rc, out))
    rc, out = run(base_args + ["--whatif", "--shape", "Nope=1"])
    check(rc == 2 and "Nope" in out, "unknown shape should ERROR exit 2 naming it: %d\n%s" % (rc, out))
    rc, out = run(base_args)
    check(rc == 2 and "=> FAIL: bad args" in out, "no mode flag should be a bad-args FAIL exit 2: %d\n%s" % (rc, out))
    rc, out = run(base_args + ["--whatif", "--delta-m", "0.005"])
    check(rc == 2 and "must exceed" in out, "delta under threshold should refuse: %d\n%s" % (rc, out))
    rc, out = run(base_args + ["--whatif", "--shape", "Swell"])
    check(rc == 2 and "NAME=VALUE" in out, "malformed --shape should refuse: %d\n%s" % (rc, out))


def main():
    _repo_root()
    from avatarprep.core import coverage
    print("COVERAGE_TEST module:", coverage.__file__)
    test_closed_tube_covers()
    test_double_wall_ignores_garment_normals()
    test_open_tube_and_loose_collar()
    test_swinging_bones_and_blend_ratio()
    test_farther_comoving_garment_covers()
    test_kin_and_fold()
    test_shapes_on_stand_in()
    test_polygon_carrier_keeps_uncovered_corner()
    test_residue_strip()
    test_cut_shape_exclusion_and_unknown()
    test_write_carrier_and_gates()
    with tempfile.TemporaryDirectory() as tmp:
        test_marked_copy_and_save(tmp)
        test_cli(tmp)
    if FAILURES:
        for f in FAILURES:
            print("COVERAGE_TEST FAIL:", f)
        sys.exit(1)
    print("COVERAGE_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _harness
    _harness.run(main, "COVERAGE_TEST")
