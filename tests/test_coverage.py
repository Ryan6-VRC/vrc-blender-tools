"""Synthetic headless test for avatarprep.core.coverage and the mark_coverage door.

Run: blender --background --factory-startup --python tests/test_coverage.py
Prints COVERAGE_TEST OK / COVERAGE_TEST FAIL: <reason>; exit 1 on any failure.

Fixtures are a cylinder "body" weighted to one bone and tubes around it built in memory,
so every case pins one clause of the criterion or of the carrier rule:

  * a closed tube 10 mm out covers the cylinder band it wraps (the base case);
  * a double-walled tube whose inner wall's normals face the body still covers — nothing
    reads a garment normal;
  * a short open tube leaves the skin near and beyond its edge uncovered (the peek rays
    escape through the opening), and a tube standing far off the skin is seen into;
  * a tube band weighted to a bone the body lacks declines as cloth and reports a low
    weight share; a blend-ratio difference on the body's own bones still covers;
  * a one-triangle-wide covered strip is residue: covered, carrier empty there;
  * --cut-shape excludes what a shape already moves, and an unknown cut shape refuses;
  * write_carrier lands the key on a local body, refuses a moved Basis and a same-named
    key, and the door's --whatif / --out paths and exit codes hold via subprocess.
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


def _cylinder(name, radius, z0, z1, segs=24, rings=8, weights=None, flip=False, cap=False):
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
                groups.setdefault(g, ob.vertex_groups.new(name=g)) if g not in groups else None
                groups[g].add([r * segs + s], val, 'REPLACE')
    return ob


def _measure(body, garments, **kw):
    from avatarprep.core import coverage
    args = dict(distance=0.015, body_bone_share=0.7, peek_deg=60.0, peek_reach=0.5,
                cut_threshold=0.01)
    args.update(kw)
    return coverage.measure(body, garments, **args)


def test_closed_tube_covers():
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    tube = _cylinder("Tube", 0.11, -0.2, 1.2, rings=8)   # overhangs both ends: no hem near the body
    r = _measure(body, [tube])
    check(r["covered_triangles"] == r["triangles"],
          "closed tube should cover every body triangle, got %d/%d" % (r["covered_triangles"], r["triangles"]))
    check(r["realised_triangles"] == r["triangles"] and r["residue_triangles"] == 0,
          "closed tube: realised %d residue %d" % (r["realised_triangles"], r["residue_triangles"]))
    check(r["per_garment"]["Tube"]["weight_share_on_body_groups"] == 1.0, "weight share should be 1.0")


def test_double_wall_ignores_garment_normals():
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    outer = _cylinder("Outer", 0.112, -0.2, 1.2, rings=8)
    inner = _cylinder("Inner", 0.105, -0.2, 1.2, rings=8, flip=True)  # lining, normals toward the arm
    # one garment object holding both walls
    with bpy.context.temp_override(active_object=outer, selected_editable_objects=[outer, inner]):
        bpy.ops.object.join()
    r = _measure(body, [outer])
    check(r["covered_triangles"] == r["triangles"],
          "double wall should still cover fully, got %d/%d (declined %s)"
          % (r["covered_triangles"], r["triangles"], r["per_garment"]["Outer"]["declined"]))


def test_peek_declines_edges_and_gaps():
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    tube = _cylinder("Tube", 0.11, -0.2, 0.5, rings=8)   # open top at z=0.5, mid-body
    r = _measure(body, [tube])
    d = r["per_garment"]["_all"]["declined"]
    check(d.get("peek", 0) > 0, "rays should escape through the opening, declined=%s" % d)
    check(0 < r["covered_triangles"] < r["triangles"], "open tube should cover part, got %d/%d"
          % (r["covered_triangles"], r["triangles"]))
    above = [i for i, c in enumerate(r["covered"]) if c and body.data.vertices[i].co.z > 0.5]
    check(not above, "vertices beyond the opening were marked covered: %d" % len(above))
    ring_below = [i for i, c in enumerate(r["covered"]) if c and abs(body.data.vertices[i].co.z - 0.5) < 1e-6]
    check(not ring_below, "the ring at the opening is seen into and must not be covered: %d" % len(ring_below))
    deep = [i for i, c in enumerate(r["covered"]) if c and body.data.vertices[i].co.z < 0.3]
    check(len(deep) > 0, "skin well below the opening should still be covered")
    # a collar standing 40 mm off the skin: within distance, but the gap is seen into
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    collar = _cylinder("Collar", 0.14, 0.6, 0.9, rings=4)
    r = _measure(body, [collar], distance=0.05)
    check(r["covered_triangles"] < r["triangles"] * 0.2,
          "a loose collar should leave most of the skin under it visible, covered %d/%d"
          % (r["covered_triangles"], r["triangles"]))


def test_cloth_bones_decline_and_ratio_does_not():
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    tube = _cylinder("Tube", 0.11, -0.2, 1.2, rings=8,
                     weights=lambda r: {"Spine": 1.0} if r < 4 else {"Skirt": 1.0})
    r = _measure(body, [tube])
    d = r["per_garment"]["Tube"]["declined"]
    check(d.get("cloth", 0) > 0, "skirt-weighted band should decline as cloth, declined=%s" % d)
    check(0 < r["covered_triangles"] < r["triangles"], "cloth case should cover part only")
    share = r["per_garment"]["Tube"]["weight_share_on_body_groups"]
    check(0.3 < share < 0.7, "weight share should be about half, got %r" % share)
    # a pants leg at a different blend ratio on the body's own bones still covers
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20,
                     weights=lambda r: {"Spine": 0.9, "Hips": 0.1})
    tube = _cylinder("Tube", 0.11, -0.2, 1.2, rings=8,
                     weights=lambda r: {"Spine": 0.4, "Hips": 0.6})
    r = _measure(body, [tube])
    check(r["covered_triangles"] == r["triangles"],
          "a blend-ratio difference on body bones must not decline, got %d/%d (declined %s)"
          % (r["covered_triangles"], r["triangles"], r["per_garment"]["Tube"]["declined"]))


def test_any_garment_may_cover():
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    bad = _cylinder("Bad", 0.105, -0.2, 1.2, rings=8, weights=lambda r: {"Other": 1.0})  # nearer, wrong bone
    good = _cylinder("Good", 0.112, -0.2, 1.2, rings=8)
    r = _measure(body, [bad, good])
    check(r["covered_triangles"] == r["triangles"],
          "a farther passing garment must cover where the nearest fails, got %d/%d"
          % (r["covered_triangles"], r["triangles"]))


def test_residue_strip():
    """A thin covered band: the tube rides a bone the body lacks except over a narrow band
    on the body's bone, so only a few rings of quads are covered and their border vertices
    cannot carry. Residue must be non-zero and realised must fall short of covered."""
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20, segs=24)
    tube = _cylinder("Tube", 0.11, -0.2, 1.2, rings=28)
    for g in list(tube.vertex_groups):
        tube.vertex_groups.remove(g)
    tspine = tube.vertex_groups.new(name="Spine")
    tskirt = tube.vertex_groups.new(name="Skirt")
    for i, v in enumerate(tube.data.vertices):
        (tspine if 0.44 < v.co.z < 0.61 else tskirt).add([i], 1.0, 'REPLACE')
    r = _measure(body, [tube])
    check(0 < r["covered_triangles"] <= 24 * 2 * 3, "band should cover at most three rings of quads, got %d"
          % r["covered_triangles"])
    check(r["residue_triangles"] > 0 and r["realised_triangles"] < r["covered_triangles"],
          "a thin band must leave residue, got realised %d residue %d"
          % (r["realised_triangles"], r["residue_triangles"]))


def test_cut_shape_exclusion_and_unknown():
    from avatarprep.core import coverage
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    tube = _cylinder("Tube", 0.11, -0.2, 1.2, rings=8)
    body.shape_key_add(name="Basis", from_mix=False)
    cutk = body.shape_key_add(name="Lower_OFF", from_mix=False)
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
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    tube = _cylinder("Tube", 0.11, -0.2, 1.2, rings=8)
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


def test_marked_copy_and_removed():
    from avatarprep.core import coverage
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    tube = _cylinder("Tube", 0.11, -0.2, 0.5, rings=8)
    r = _measure(body, [tube])
    ob = coverage.marked_copy(body, r, name="M", garment_order=["Tube"])
    check(ob.data.color_attributes.active_color is not None, "marked copy needs an active colour")
    check(len(ob.data.vertices) == len(body.data.vertices), "marked copy keeps geometry")
    coverage.remove_marked_copy(ob)
    ob2 = coverage.marked_copy(body, r, name="R", garment_order=["Tube"], remove_carrier=True)
    n_faces = len(ob2.data.polygons)
    # quads: realised triangles / 2 quads removed (no cut shapes in this fixture)
    check(n_faces == len(body.data.polygons) - r["realised_triangles"] // 2,
          "removed copy should drop the realised quads: %d vs %d - %d/2"
          % (n_faces, len(body.data.polygons), r["realised_triangles"]))
    coverage.remove_marked_copy(ob2)


def test_cli(tmp):
    """The door through a fresh blender: --whatif writes nothing, --out writes the key onto
    a LINKED body's own blend, bad args exit 2 in-grammar."""
    root = _repo_root()
    blender = bpy.app.binary_path
    # base blend: the body
    _clear()
    _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    base = os.path.join(tmp, "base.blend")
    bpy.ops.wm.save_as_mainfile(filepath=base)
    # costume blend: the tube, with the body linked in
    _clear()
    _cylinder("Tube", 0.11, -0.2, 1.2, rings=8)
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

    report = os.path.join(tmp, "r.json")
    rc, out = run(["--in", costume, "--body", "Body", "--garments", "Tube", "--shape-name", "Cover_T",
                   "--whatif", "--report", report])
    check(rc == 0 and "markcoverage Cover_T" in out and "=> OK" in out, "whatif should pass: %d\n%s" % (rc, out))
    check(os.path.exists(report), "whatif should write the report")
    check("png=" not in out, "no render requested, no png= trailer")

    out_blend = os.path.join(tmp, "base_marked.blend")
    rc, out = run(["--in", costume, "--body", "Body", "--garments", "Tube", "--shape-name", "Cover_T",
                   "--out", out_blend])
    check(rc == 0 and "saved=" in out, "write should pass and name the save: %d\n%s" % (rc, out))
    check(os.path.exists(out_blend), "write should save --out")
    bpy.ops.wm.open_mainfile(filepath=out_blend)
    me = bpy.data.objects["Body"].data
    check(me.shape_keys is not None and "Cover_T" in me.shape_keys.key_blocks,
          "the saved base copy should carry the carrier key")
    check(me.library is None, "the key must land on the base's own local mesh")

    rc, out = run(["--in", costume, "--body", "Body", "--garments", "Nope", "--shape-name", "X", "--whatif"])
    check(rc == 2 and "ERROR" in out, "missing garment should ERROR exit 2: %d\n%s" % (rc, out))
    rc, out = run(["--in", costume, "--body", "Body", "--garments", "Tube", "--shape-name", "X"])
    check(rc == 2 and "=> FAIL: bad args" in out, "no mode flag should be a bad-args FAIL exit 2: %d\n%s" % (rc, out))
    rc, out = run(["--in", costume, "--body", "Body", "--garments", "Tube", "--shape-name", "X",
                   "--whatif", "--delta-m", "0.005"])
    check(rc == 2 and "must exceed" in out, "delta under threshold should refuse: %d\n%s" % (rc, out))


def main():
    _repo_root()
    from avatarprep.core import coverage
    print("COVERAGE_TEST module:", coverage.__file__)
    test_closed_tube_covers()
    test_double_wall_ignores_garment_normals()
    test_peek_declines_edges_and_gaps()
    test_cloth_bones_decline_and_ratio_does_not()
    test_any_garment_may_cover()
    test_residue_strip()
    test_cut_shape_exclusion_and_unknown()
    test_write_carrier_and_gates()
    test_marked_copy_and_removed()
    with tempfile.TemporaryDirectory() as tmp:
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
