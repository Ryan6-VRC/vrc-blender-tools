"""Synthetic headless test for avatarprep.core.coverage and the mark_coverage door.

Run: blender --background --factory-startup --python tests/test_coverage.py
Prints COVERAGE_TEST OK / COVERAGE_TEST FAIL: <reason>; exit 1 on any failure.

Fixtures are a cylinder "body" weighted to one bone and tubes around it built in memory,
so every case pins one clause of the criterion or of the carrier rule:

  * a closed tube 10 mm out covers the cylinder band it wraps (the base case);
  * a double-walled tube whose inner wall's normals face the body still covers — nothing
    reads a garment normal;
  * a short open tube leaves the ring beyond its hem uncovered (boundary-edge margin);
  * a tube band weighted to another bone declines by weight, and a tube weighted to a
    bone the body lacks reports a low weight share;
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
    args = dict(distance=0.015, weight_tol=0.35, angle_deg=60.0, hem_margin=0.015,
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


def test_hem_leaves_ring_beyond_edge():
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    tube = _cylinder("Tube", 0.11, -0.2, 0.5, rings=8)   # hem at z=0.5, mid-body
    r = _measure(body, [tube])
    # rings above the hem: far. The ring within 15 mm below the hem: hem. Below that: covered.
    d = r["per_garment"]["Tube"]["declined"]
    check(d.get("hem", 0) > 0, "hem margin should decline vertices near the edge, declined=%s" % d)
    check(0 < r["covered_triangles"] < r["triangles"], "hem case should cover part, got %d/%d"
          % (r["covered_triangles"], r["triangles"]))
    # no covered vertex within the margin below the hem edge (z in [0.485, 0.5])
    leak = [i for i, c in enumerate(r["covered"]) if c and 0.485 <= body.data.vertices[i].co.z <= 0.5]
    check(not leak, "vertices within the hem margin were marked covered: %d" % len(leak))
    # and no vertex above the hem at all
    above = [i for i, c in enumerate(r["covered"]) if c and body.data.vertices[i].co.z > 0.5]
    check(not above, "vertices beyond the hem were marked covered: %d" % len(above))


def test_weight_mismatch_declines():
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20)
    tube = _cylinder("Tube", 0.11, -0.2, 1.2, rings=8,
                     weights=lambda r: {"Spine": 1.0} if r < 4 else {"Skirt": 1.0})
    r = _measure(body, [tube])
    d = r["per_garment"]["Tube"]["declined"]
    check(d.get("weight", 0) > 0, "re-weighted band should decline by weight, declined=%s" % d)
    check(0 < r["covered_triangles"] < r["triangles"], "weight case should cover part only")
    share = r["per_garment"]["Tube"]["weight_share_on_body_groups"]
    check(0.3 < share < 0.7, "weight share should be about half, got %r" % share)


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
    """A body whose covered region is a single ring of triangles: every covered vertex has
    an uncovered incident triangle, so the carrier is empty and residue = covered."""
    _clear()
    body = _cylinder("Body", 0.10, 0.0, 1.0, rings=20, segs=24)
    # a tube covering only z in [0.47, 0.53] with overhang far beyond the hem margin is
    # impossible (it IS the hem), so build the strip by weights: only ring 10's vertices
    # and ring 11's share the tube's bone
    tube = _cylinder("Tube", 0.11, -0.2, 1.2, rings=8)
    for g in body.vertex_groups:
        body.vertex_groups.remove(g)
    spine = body.vertex_groups.new(name="Spine")
    other = body.vertex_groups.new(name="Other")
    for i, v in enumerate(body.data.vertices):
        ring = i // 24
        (spine if ring in (10, 11) else other).add([i], 1.0, 'REPLACE')
    r = _measure(body, [tube])
    check(r["covered_triangles"] == 24 * 2, "strip should cover one ring of quads (48 tris), got %d"
          % r["covered_triangles"])
    check(r["realised_triangles"] == 0 and r["residue_triangles"] == r["covered_triangles"],
          "strip should be all residue, realised %d residue %d"
          % (r["realised_triangles"], r["residue_triangles"]))
    check(r["carrier"] == [], "carrier should be empty on a strip")


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
    test_hem_leaves_ring_beyond_edge()
    test_weight_mismatch_declines()
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
