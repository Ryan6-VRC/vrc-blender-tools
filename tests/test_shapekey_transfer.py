"""Synthetic headless test for avatarprep.core.shapekey_transfer.

Run: blender --background --factory-startup --python tests/test_shapekey_transfer.py
Prints TRANSFER_TEST OK / TRANSFER_TEST FAIL: <reason>.

Fixture: a dense "body" grid with a bump key, baked half in (Basis raised by half the
bump, avatarprep_baked records 0.5); a "garment" grid hovering a fixed 5 mm above the
UNBAKED body — the vendor-neutral cut. The transfer must seat the garment over the baked
body at the same 5 mm, leave vertices off the bump alone, and carry the key over when asked.
"""
import os
import sys
import math

import bpy
from mathutils import Vector

FAILURES = []
GAP = 0.005
BUMP = 0.03
BAKED = 0.5


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def expect_raises(fn, substr, label):
    try:
        fn()
    except Exception as e:
        if substr.lower() not in str(e).lower():
            FAILURES.append("%s: raised but %r lacked %r" % (label, str(e), substr))
        return
    FAILURES.append("%s: expected exception mentioning %r" % (label, substr))


def _enable():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def _clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def _bump(x, y):
    r = math.hypot(x - 0.5, y - 0.5)
    return BUMP * max(0.0, 1.0 - r / 0.25) ** 2


def _grid(name, n, z_fn):
    verts, faces = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            x, y = i / n, j / n
            verts.append((x, y, z_fn(x, y)))
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            faces.append((a, a + 1, a + n + 2, a + n + 1))
    md = bpy.data.meshes.new(name + "Data")
    md.from_pydata(verts, [], faces); md.update()
    ob = bpy.data.objects.new(name, md)
    bpy.context.collection.objects.link(ob)
    return ob


def _body():
    from avatarprep.core import scene_utils
    # Basis carries half the bump (the bake); the key keeps its full delta from Basis,
    # exactly as bake_shapekey leaves a body.
    ob = _grid("Body_Base", 40, lambda x, y: BAKED * _bump(x, y))
    ob.shape_key_add(name="Basis")
    kb = ob.shape_key_add(name="Bump", from_mix=False)
    kb.value = 0.0
    for i, v in enumerate(ob.data.vertices):
        kb.data[i].co = Vector(v.co) + Vector((0, 0, _bump(v.co.x, v.co.y)))
    ob[scene_utils.STAMP_BAKED] = {"Bump": BAKED}
    return ob


def _garment():
    return _grid("Top", 24, lambda x, y: GAP)  # cut over the unbaked body: flat, 5 mm up


def _gap_over_body(garment, body):
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(body, dg)
    ev = garment.evaluated_get(dg).data
    gaps = []
    for v in ev.vertices:
        loc, n, fi, dist = tree.find_nearest(garment.matrix_world @ v.co)
        gaps.append((garment.matrix_world @ v.co - loc).dot(n))
    return gaps


def test_seat_only_folds_into_basis_and_stamps():
    from avatarprep.core import shapekey_transfer as T
    from avatarprep.core import scene_utils
    body = _body(); top = _garment()
    rep = T.transfer_shapekeys(body, [top], keys=[], authored={"Bump": 0.0})
    row = rep["targets"][0]
    check(abs(rep["seat"]["Bump"] - BAKED) < 1e-9, "seat should be state-authored = 0.5")
    gaps = _gap_over_body(top, body)
    worst = max(abs(g - GAP) for g in gaps)
    check(worst < 0.0015, "seated garment should keep its 5 mm gap over the bump (worst dev %.2f mm)" % (worst * 1000))
    corner = top.data.vertices[0].co
    check(abs(corner.z - GAP) < 1e-6, "a vertex off the bump must not move (z=%g)" % corner.z)
    m = dict(top.get(scene_utils.STAMP_BAKED))
    check(abs(m.get("Bump", -9) - BAKED) < 1e-9, "garment baked map should record Bump=0.5, got %r" % m)
    check(top.data.shape_keys is None or "Bump" not in top.data.shape_keys.key_blocks,
          "no key requested, none added")
    check(row["leak_outside_footprint"] == 0, "no movement outside the footprint (leak=%d)" % row["leak_outside_footprint"])


def test_add_key_carries_seat_as_live_value():
    from avatarprep.core import shapekey_transfer as T
    from avatarprep.core import scene_utils
    body = _body(); top = _garment()
    T.transfer_shapekeys(body, [top], keys=["Bump"], authored={"Bump": 0.0})
    kb = top.data.shape_keys.key_blocks
    check("Bump" in kb, "Bump key should be added")
    check(abs(kb["Bump"].value - BAKED) < 1e-9, "added key rests at the body's state 0.5, got %g" % kb["Bump"].value)
    centre = min(range(len(top.data.vertices)), key=lambda i: (Vector(top.data.vertices[i].co).xy - Vector((0.5, 0.5))).length)
    basis_z = kb["Basis"].data[centre].co.z
    # The live key carries the seat; Basis keeps only the residual between the single-shot
    # seat and the key's linearised delta, a couple of millimetres at most on this bump.
    check(abs(basis_z - GAP) < 0.003, "Basis carries only the seat residual (z=%g)" % basis_z)
    full = kb["Bump"].data[centre].co.z - basis_z
    check(abs(full - BUMP) < 0.002, "added key's full delta should be the bump (%.1f mm vs %.1f)" % (full * 1000, BUMP * 1000))
    check(top.get(scene_utils.STAMP_BAKED) is None or "Bump" not in dict(top.get(scene_utils.STAMP_BAKED)),
          "an added key writes no baked entry")
    gaps = _gap_over_body(top, body)
    worst = max(abs(g - GAP) for g in gaps)
    check(worst < 0.0015, "evaluated garment keeps its gap (worst dev %.2f mm)" % (worst * 1000))


def test_authored_offset_unbakes_negative():
    # The garment was cut against the body at Bump=1.0 (a bigger bust); un-baking to state 0.5
    # records the offset as a negative-going move recorded in the baked map.
    from avatarprep.core import shapekey_transfer as T
    from avatarprep.core import scene_utils
    body = _body()
    top = _grid("Top", 24, lambda x, y: GAP + _bump(x, y))  # cut over the full bump
    rep = T.transfer_shapekeys(body, [top], keys=[], authored={"Bump": 1.0})
    check(abs(rep["seat"]["Bump"] + 0.5) < 1e-9, "seat should be 0.5-1.0 = -0.5, got %r" % rep["seat"])
    m = dict(top.get(scene_utils.STAMP_BAKED))
    check(abs(m.get("Bump", 9) + 0.5) < 1e-9, "baked map should record Bump=-0.5, got %r" % m)
    gaps = _gap_over_body(top, body)
    worst = max(abs(g - GAP) for g in gaps)
    check(worst < 0.0015, "un-baked garment keeps its gap (worst dev %.2f mm)" % (worst * 1000))


def test_add_key_with_authored_offset():
    # Cut at Bump=1.0, body at 0.5, key added: live value is the body's state (0.5) and the
    # authored 1.0 is un-baked into Basis, recorded as -1.0.
    from avatarprep.core import shapekey_transfer as T
    from avatarprep.core import scene_utils
    body = _body()
    top = _grid("Top", 24, lambda x, y: GAP + _bump(x, y))
    T.transfer_shapekeys(body, [top], keys=["Bump"], authored={"Bump": 1.0})
    kb = top.data.shape_keys.key_blocks
    check(abs(kb["Bump"].value - BAKED) < 1e-9, "live value is the body's state")
    m = dict(top.get(scene_utils.STAMP_BAKED))
    check(abs(m.get("Bump", 9) + 1.0) < 1e-9, "baked map records the un-baked authored offset -1.0, got %r" % m)
    gaps = _gap_over_body(top, body)
    worst = max(abs(g - GAP) for g in gaps)
    check(worst < 0.0015, "evaluated garment keeps its gap (worst dev %.2f mm)" % (worst * 1000))


def test_smooth_keeps_gap():
    from avatarprep.core import shapekey_transfer as T
    body = _body(); top = _garment()
    T.transfer_shapekeys(body, [top], keys=[], authored={"Bump": 0.0}, smooth=5)
    gaps = _gap_over_body(top, body)
    worst = max(abs(g - GAP) for g in gaps)
    check(worst < 0.01, "smoothed seat trades gap fidelity, within bounds (worst dev %.2f mm)" % (worst * 1000))
    check(abs(top.data.vertices[0].co.z - GAP) < 1e-6, "smoothing stays inside the footprint")


def test_whatif_writes_nothing():
    from avatarprep.core import shapekey_transfer as T
    body = _body(); top = _garment()
    before = [Vector(v.co) for v in top.data.vertices]
    rep = T.transfer_shapekeys(body, [top], keys=["Bump"], authored={"Bump": 0.0}, whatif=True)
    check(rep["targets"][0]["after"]["n"] > 0, "whatif still measures")
    check(top.data.shape_keys is None, "whatif adds no key")
    drift = max((Vector(v.co) - b).length for v, b in zip(top.data.vertices, before))
    check(drift == 0.0, "whatif moves nothing (drift=%g)" % drift)


def test_refusals():
    from avatarprep.core import shapekey_transfer as T
    body = _body(); top = _garment()
    top.shape_key_add(name="Basis"); top.shape_key_add(name="Bump")
    expect_raises(lambda: T.transfer_shapekeys(body, [top], keys=["Bump"]), "already carries", "existing key")
    top2 = _garment()
    expect_raises(lambda: T.transfer_shapekeys(body, [top2], keys=["Ghost"]), "not found", "missing source key")
    expect_raises(lambda: T.transfer_shapekeys(body, [top2], keys=[]), "nothing to do", "empty request")
    expect_raises(lambda: T.transfer_shapekeys(body, [top2], keys=[], authored={"Bump": 0.0}, seat=False), "nothing to do", "no-seat without keys")
    expect_raises(lambda: T.transfer_shapekeys(body, [body], keys=[], authored={"Bump": 0.0}), "source body", "source as target")
    expect_raises(lambda: T.scan_authored(body, top2, "Ghost"), "not found", "scan of a missing key")
    top2["avatarprep_baked"] = "not-a-map"
    before = [Vector(v.co) for v in top2.data.vertices]
    expect_raises(lambda: T.transfer_shapekeys(body, [top2], keys=[], authored={"Bump": 0.0}), "not a map", "corrupt baked stamp")
    drift = max((Vector(v.co) - b).length for v, b in zip(top2.data.vertices, before))
    check(drift == 0.0, "a corrupt stamp is refused before anything moves (drift=%g)" % drift)


def test_scan_reads_authored_value():
    from avatarprep.core import shapekey_transfer as T
    body = _body()
    top = _grid("Top", 24, lambda x, y: GAP + _bump(x, y))  # cut at Bump=1.0
    sc = T.scan_authored(body, top, "Bump", footprint=0.005)
    check(sc["core_verts"] > 20, "scan finds the bump core")
    rows = sc["scan"]
    best = min(rows, key=lambda k: abs(rows[k]["min_mm"] - GAP * 1000))
    check(best == "1", "scan should read the authored value 1.0 off the gap table, got %r (%r)" % (best, rows))


def test_static_island_is_dropped():
    # A flush overlay the key leaves in place — a pasty — sits 0.5 mm above the full bump's
    # apex as its own island, so it is the nearest surface to a garment cut at Bump=1.0.
    # Garment vertices over it must still move with the skin under it.
    from avatarprep.core import shapekey_transfer as T
    import bmesh
    body = _body(); top = _grid("Top", 24, lambda x, y: GAP + _bump(x, y))
    me = body.data
    base_n = len(me.vertices)
    verts = [(0.5 + 0.02 * dx, 0.5 + 0.02 * dy, _bump(0.5 + 0.02 * dx, 0.5 + 0.02 * dy) + 0.0005)
             for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    faces = [(j * 3 + i, j * 3 + i + 1, (j + 1) * 3 + i + 1, (j + 1) * 3 + i) for j in range(2) for i in range(2)]
    bm = bmesh.new(); bm.from_mesh(me)
    nv = [bm.verts.new(co) for co in verts]
    for f in faces:
        bm.faces.new([nv[i] for i in f])
    bm.to_mesh(me); bm.free(); me.update()
    for kb in me.shape_keys.key_blocks:
        for i, co in enumerate(verts):
            kb.data[base_n + i].co = co  # the overlay is static under every key
    rep = T.transfer_shapekeys(body, [top], keys=["Bump"], authored={"Bump": 1.0})
    check([d["verts"] for d in rep["static_islands_dropped"]] == [9],
          "the 9-vertex overlay is reported dropped, got %r" % rep["static_islands_dropped"])
    row = rep["targets"][0]
    check(row["verts_over_static_island"] > 0, "garment vertices over the overlay are counted")
    kb = top.data.shape_keys.key_blocks
    centre = min(range(len(top.data.vertices)), key=lambda i: (Vector(top.data.vertices[i].co).xy - Vector((0.5, 0.5))).length)
    full = kb["Bump"].data[centre].co.z - kb["Basis"].data[centre].co.z
    check(abs(full - BUMP) < 0.002, "the key still reaches the vertex over the overlay (%.1f mm vs %.1f)" % (full * 1000, BUMP * 1000))


def test_tear_heals():
    # A key that pulls the two halves of the body apart, sharply at x=0.5 — the cleavage,
    # where each breast recedes toward its own side. A garment bridging the midline maps
    # each vertex to whichever half is nearest and shears along the line unless the move
    # is healed across it.
    from avatarprep.core import shapekey_transfer as T
    from avatarprep.core import scene_utils
    import math
    SPREAD = 0.015

    def spread_body():
        body = _grid("Body_Base", 160, lambda x, y: 0.0)
        body.shape_key_add(name="Basis")
        kb = body.shape_key_add(name="Spread", from_mix=False)
        for i, v in enumerate(body.data.vertices):
            kb.data[i].co = Vector(v.co) + Vector((SPREAD * math.tanh((v.co.x - 0.5) / 0.003), 0, 0))
        body[scene_utils.STAMP_BAKED] = {"Spread": 0.0}
        kb.value = 1.0  # the body wears the spread live
        return body

    def strain(top):
        return max(((Vector(top.data.vertices[a].co) - ORIG[a]) - (Vector(top.data.vertices[b].co) - ORIG[b])).length
                   / (ORIG[a] - ORIG[b]).length for a, b in (tuple(e.vertices) for e in top.data.edges))

    body = spread_body(); top = _grid("Top", 80, lambda x, y: GAP)
    ORIG = [Vector(v.co) for v in top.data.vertices]
    T.transfer_shapekeys(body, [top], keys=[], authored={"Spread": 0.0}, heal=0.0)
    raw = strain(top)
    check(raw > 1.0, "without healing the seat shears along the midline (strain %.2f)" % raw)
    _clear(); body = spread_body(); top2 = _grid("Top", 80, lambda x, y: GAP)
    ORIG = [Vector(v.co) for v in top2.data.vertices]
    rep = T.transfer_shapekeys(body, [top2], keys=[], authored={"Spread": 0.0}, heal=0.05)
    row = rep["targets"][0]
    check(row["torn_edges"] > 0 and row["healed_verts"] > 0, "healing reports the tear (%r)" % row)
    healed = strain(top2)
    check(healed < 1.0 and healed < 0.6 * raw, "healed seat gathers across the midline (strain %.2f vs raw %.2f)" % (healed, raw))
    gaps = _gap_over_body(top2, body)
    check(min(gaps) > GAP - 1e-3, "healing keeps the garment off the skin (min gap %.2f mm)" % (min(gaps) * 1000))
    edge = top2.data.vertices[0].co
    check(abs(edge.x - ORIG[0].x + SPREAD) < 1e-4, "far from the midline the seat is the skin's own move (dx=%g)" % (edge.x - ORIG[0].x))


def main():
    _clear(); _enable()
    for t in (test_seat_only_folds_into_basis_and_stamps, test_add_key_carries_seat_as_live_value,
              test_authored_offset_unbakes_negative, test_add_key_with_authored_offset, test_smooth_keeps_gap, test_whatif_writes_nothing, test_refusals,
              test_scan_reads_authored_value, test_static_island_is_dropped, test_tear_heals):
        _clear()
        t()
    from avatarprep.core import shapekey_transfer
    print("TRANSFER_TEST module", shapekey_transfer.__file__)
    if FAILURES:
        for f in FAILURES:
            print("TRANSFER_TEST FAIL:", f)
        sys.exit(1)
    print("TRANSFER_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "TRANSFER_TEST")
