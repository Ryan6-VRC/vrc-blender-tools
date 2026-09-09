"""Co-moving coverage: mark body triangles a garment set hides, as a Delete carrier.

A body triangle is *covered* when every vertex of it sits just behind a named garment
surface that rides the body's own bones, because geometry skinned to the same skeleton
deforms together, so a rest-pose distance holds under pose; a garment surface skinned to
cloth, physbone or helper bones the body never uses swings away from the skin and covers
nothing. Nothing here casts a ray or renders to decide; a garment normal is read only as
one of two side branches, never alone — the prototype that read it alone declared a
double-walled sleeve's arm "outside" because the nearest point was the lining.

Per body vertex, a garment passes when ALL hold (the first failure is the decline reason):

  far      no point on the garment within ``distance``
  side     the nearest point is neither on the skin's outward side by the BODY normal
           nor in front of the garment face the skin sits behind — the second branch is
           what a loose panel standing off a concave region (the side torso under the
           arm) passes by; a lining whose normals face the skin fails it and passes the
           first, which is why neither branch alone would do
  hem      the nearest point lies within ``hem_margin`` of a garment boundary edge (an
           edge with exactly one face) — pure topology, so it holds on zero-thickness and
           unculled meshes; blind to two coincident duplicate sheets. This is the
           peek-under-a-cuff guard and the one number worth care
  angle    the skin-to-point direction is more than ``angle`` off the body normal (skipped
           when the skin sits behind the garment face — that branch already says enclosed)
  unweighted  the body vertex's raw weight sum is under ``min_raw_weight``
  cloth    the nearest face's weight mass (mean of its corners, top ``max_bones`` groups,
           renormalised) on bone names the BODY mesh has a vertex group for is under
           ``body_bone_share`` — the face rides skirt/ribbon/helper bones, not the body's

A vertex is covered when ANY listed garment passes. A pants leg skinned to the leg bones
at a different blend ratio than the skin still covers: that mismatch clips in game and is
not a reason to keep the triangle.

The consumer is a Modular Avatar ShapeChanger Delete: a triangle is removed when ANY of
its vertices moves more than the component's threshold under the shape. So the carrier
vertex set is every vertex whose incident triangles are ALL covered or already cut; the
carrier then removes exactly the triangles touching it, a subset of the covered set by
construction. ``residue`` is what is covered but has no interior vertex to carry it.

Pure ``bpy`` data access, no operators, no UI. Door: ``cli/mark_coverage.py``.
"""
import hashlib
import math
from collections import defaultdict
from typing import Dict, List, Sequence

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

REASONS = ("far", "side", "hem", "angle", "unweighted", "cloth")


class CoverageError(Exception):
    """A refusal the door prints in-grammar (unresolvable input, changed body)."""


# --- mesh access ---------------------------------------------------------------------

def evaluated_mesh(obj, depsgraph):
    """The object's depsgraph-evaluated mesh (shape-key values applied, modifiers
    included) as ``(evaluated_object, mesh)``; the caller clears it with
    ``evaluated_object.to_mesh_clear()``. Refuses a modifier that changes the vertex
    count: weights and the carrier are indexed against the original mesh."""
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    if len(me.vertices) != len(obj.data.vertices):
        n_ev, n_orig = len(me.vertices), len(obj.data.vertices)
        ev.to_mesh_clear()
        raise CoverageError("%s: a modifier changes the vertex count (%d -> %d); coverage "
                            "is indexed against the original mesh, so disable it first"
                            % (obj.name, n_orig, n_ev))
    return ev, me


def world_verts_normals(obj, me):
    mw = obj.matrix_world
    rot = mw.to_3x3()
    return ([mw @ v.co for v in me.vertices],
            [(rot @ v.normal).normalized() for v in me.vertices])


def normalized_weights(obj, me, max_bones: int, floor: float = 0.01):
    """Per vertex: ``(raw_sum, {group_name: w})`` with the top ``max_bones`` groups kept
    and renormalised to sum 1 — the vector Unity skins with. Groups under ``floor`` are
    dropped before truncation."""
    names = [g.name for g in obj.vertex_groups]
    out = []
    for v in me.vertices:
        raw = {}
        for g in v.groups:
            if g.weight >= floor and g.group < len(names):
                raw[names[g.group]] = raw.get(names[g.group], 0.0) + g.weight
        raw_sum = sum(raw.values())
        top = sorted(raw.items(), key=lambda kv: -kv[1])[:max_bones]
        s = sum(w for _, w in top) or 1.0
        out.append((raw_sum, {k: w / s for k, w in top}))
    return out


def tv_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Total-variation distance between two normalised weight vectors: 0 identical,
    1 disjoint bone sets."""
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _face_weights(g_w, corners):
    """Mean of the corners' normalised weight vectors — the garment's weights over the
    face the body vertex is nearest to."""
    out = defaultdict(float)
    for vi in corners:
        for k, w in g_w[vi][1].items():
            out[k] += w / len(corners)
    return out


def boundary_edges_world(me, mw):
    """World-space segments of every edge with exactly one face — the garment's hems,
    cuffs and necklines, whatever its normals or culling say."""
    count = defaultdict(int)
    for p in me.polygons:
        for ek in p.edge_keys:
            count[ek] += 1
    return [(mw @ me.vertices[a].co, mw @ me.vertices[b].co)
            for (a, b), n in count.items() if n == 1]


def _segment_bvh(segs):
    """A BVH over slivers (each edge plus a point a hair off it) so nearest-edge queries
    reuse ``find_nearest``. None for a closed mesh with no boundary."""
    if not segs:
        return None
    verts, polys = [], []
    for a, b in segs:
        c = (a + b) * 0.5 + Vector((0.0, 0.0, 1e-7))
        i = len(verts)
        verts += [a, b, c]
        polys.append((i, i + 1, i + 2))
    return BVHTree.FromPolygons(verts, polys)


def basis_hash(me) -> str:
    """Order-sensitive hash of the mesh's Basis coordinates (shape key 0 when present,
    else the vertex coordinates), so a reordered or edited body is caught before an
    index-list carrier lands on it."""
    h = hashlib.sha256()
    src = me.shape_keys.key_blocks[0].data if me.shape_keys else me.vertices
    for d in src:
        h.update(("%.6f %.6f %.6f;" % tuple(d.co)).encode())
    return h.hexdigest()


def triangles(me) -> List[tuple]:
    """Fan-triangulated polygon vertex triples, the granularity the consumer removes at."""
    tris = []
    for p in me.polygons:
        vs = p.vertices
        for k in range(1, len(vs) - 1):
            tris.append((vs[0], vs[k], vs[k + 1]))
    return tris


def moved_vertices(me, shape_names: Sequence[str], threshold: float):
    """Vertices any named shape key moves more than ``threshold`` — the consumer's
    already-cut set — plus the names that are not on the mesh (never silently ignored)."""
    moved = [False] * len(me.vertices)
    if not me.shape_keys:
        return moved, list(shape_names)
    basis = me.shape_keys.key_blocks[0]
    unknown = []
    for name in shape_names:
        kb = me.shape_keys.key_blocks.get(name)
        if kb is None:
            unknown.append(name)
            continue
        for i in range(len(me.vertices)):
            if (kb.data[i].co - basis.data[i].co).length > threshold:
                moved[i] = True
    return moved, unknown


# --- the measurement -----------------------------------------------------------------

def measure(body, garments, *, distance: float, body_bone_share: float, angle_deg: float,
            hem_margin: float, cut_threshold: float, cut_shapes: Sequence[str] = (),
            max_bones: int = 4, min_raw_weight: float = 0.5, depsgraph=None) -> Dict:
    """Measure coverage of ``body`` by ``garments`` (mesh objects). Returns per-vertex
    ``covered`` / ``claimed_by`` / ``cut`` / ``near`` lists, the carrier, triangle counts,
    per-garment decline counts and the body's basis hash. Raises ``CoverageError`` on an
    unresolvable input; never mutates the scene."""
    if body is None or body.type != 'MESH':
        raise CoverageError("body is not a mesh object")
    if not garments:
        raise CoverageError("no garments named; the cover set is an explicit input")
    for g in garments:
        if g is None or g.type != 'MESH':
            raise CoverageError("garment %r is not a mesh object" % getattr(g, "name", g))
        if g == body:
            raise CoverageError("%s is both the body and a garment" % body.name)

    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    cos_angle = math.cos(math.radians(angle_deg))

    b_ev, b_me = evaluated_mesh(body, dg)
    try:
        b_verts, b_norms = world_verts_normals(body, b_me)
        b_w = normalized_weights(body, b_me, max_bones)
        tris = triangles(b_me)
    finally:
        b_ev.to_mesh_clear()
    b_hash = basis_hash(body.data)
    cut, unknown_cuts = moved_vertices(body.data, cut_shapes, cut_threshold)
    if unknown_cuts:
        raise CoverageError("cut shape(s) not on %s: %s" % (body.name, ", ".join(unknown_cuts)))
    body_groups = set(g.name for g in body.vertex_groups)

    n = len(b_verts)
    covered = [False] * n
    near = [False] * n
    claimed_by = [None] * n
    per_garment = {}

    for g in garments:
        g_ev, g_me = evaluated_mesh(g, dg)
        try:
            gmw = g.matrix_world
            g_verts = [gmw @ v.co for v in g_me.vertices]
            polys = [tuple(p.vertices) for p in g_me.polygons]
            tree = BVHTree.FromPolygons(g_verts, polys)
            g_w = normalized_weights(g, g_me, max_bones)
            hems = boundary_edges_world(g_me, gmw)
            hem_tree = _segment_bvh(hems)
        finally:
            g_ev.to_mesh_clear()

        # share of the garment's weight mass on names the body has: a low share is a
        # rig-name mismatch, and the honest reading of "nothing covered"
        mass_total = mass_known = 0.0
        for _, w in g_w:
            for k, v in w.items():
                mass_total += v
                if k in body_groups:
                    mass_known += v

        reasons = defaultdict(int)
        passed = 0
        for i in range(n):
            p = b_verts[i]
            loc, _nor, idx, dist = tree.find_nearest(p, distance)
            if loc is None:
                reasons["far"] += 1
                continue
            near[i] = True
            d = loc - p
            outward = d.dot(b_norms[i]) > 0.0          # garment above the skin
            enclosed = _nor is not None and d.dot(_nor) > 0.0   # skin behind the garment face
            if not (outward or enclosed):
                reasons["side"] += 1
                continue
            if hem_tree is not None and hem_tree.find_nearest(loc, hem_margin)[0] is not None:
                reasons["hem"] += 1
                continue
            if (not enclosed and dist > 1e-9
                    and d.normalized().dot(b_norms[i]) < cos_angle):
                reasons["angle"] += 1
                continue
            raw_sum, bw = b_w[i]
            if raw_sum < min_raw_weight:
                reasons["unweighted"] += 1
                continue
            fw = _face_weights(g_w, polys[idx])
            if sum(w for k, w in fw.items() if k in body_groups) < body_bone_share:
                reasons["cloth"] += 1
                continue
            passed += 1
            if not covered[i]:
                covered[i] = True
                claimed_by[i] = g.name
        per_garment[g.name] = {
            "passed_vertices": passed,
            "declined": {r: reasons[r] for r in REASONS if reasons[r]},
            "boundary_edges": len(hems),
            "weight_share_on_body_groups": round(mass_known / mass_total, 3) if mass_total else 0.0,
        }

    # triangle granularity against the any-vertex rule
    incident = defaultdict(list)
    for ti, t in enumerate(tris):
        for v in t:
            incident[v].append(ti)
    tri_covered = [all(covered[v] for v in t) for t in tris]
    tri_cut = [any(cut[v] for v in t) for t in tris]
    carrier = [i for i in range(n)
               if covered[i] and not cut[i] and incident[i]
               and all(tri_covered[ti] or tri_cut[ti] for ti in incident[i])]
    carrier_set = set(carrier)
    n_cut = sum(tri_cut)
    n_covered = sum(1 for ti, t in enumerate(tris) if tri_covered[ti] and not tri_cut[ti])
    n_realised = sum(1 for ti, t in enumerate(tris)
                     if not tri_cut[ti] and any(v in carrier_set for v in t))

    return {
        "body": body.name,
        "body_mesh": body.data.name,
        "body_library": body.data.library.filepath if body.data.library else None,
        "vertex_count": n,
        "basis_hash": b_hash,
        "triangles": len(tris),
        "already_cut_triangles": n_cut,
        "covered_triangles": n_covered,
        "realised_triangles": n_realised,
        "residue_triangles": n_covered - n_realised,
        "carrier": carrier,
        "covered": covered,
        "claimed_by": claimed_by,
        "cut": cut,
        "near": near,
        "per_garment": per_garment,
        "settings": {"distance_m": distance, "body_bone_share": body_bone_share, "angle_deg": angle_deg,
                     "hem_margin_m": hem_margin, "cut_threshold_m": cut_threshold,
                     "max_bones": max_bones, "min_raw_weight": min_raw_weight,
                     "cut_shapes": list(cut_shapes)},
    }


def declined_summary(result: Dict) -> Dict[str, int]:
    """Decline reasons summed over garments, in ``REASONS`` order, zeros dropped."""
    out = defaultdict(int)
    for g in result["per_garment"].values():
        for r, c in g["declined"].items():
            out[r] += c
    return {r: out[r] for r in REASONS if out[r]}


# --- outputs -------------------------------------------------------------------------

# vertex colours (sRGB bytes) for the review sheet; garments cycle through the palette
COLOR_KEPT = (190, 190, 190, 255)
COLOR_CUT = (40, 40, 40, 255)
COLOR_DECLINED = (245, 170, 40, 255)      # near a garment, not covered
COLOR_GARMENT = [(230, 40, 40, 255), (40, 100, 230, 255), (40, 180, 80, 255),
                 (200, 40, 200, 255), (40, 200, 200, 255), (230, 120, 40, 255)]


def marked_copy(body, result: Dict, *, name: str, garment_order: Sequence[str],
                remove_carrier: bool = False):
    """A LOCAL copy of the body mesh (geometry, vertex groups, no library data) linked
    into the scene collection as ``name``, carrying a ``coverage`` colour attribute:
    kept grey, near-but-declined amber, covered coloured by claiming garment. Already-cut
    triangles are removed in every copy; with ``remove_carrier`` the triangles the carrier
    would delete are gone too, the closest proxy to the built result. The caller removes it; nothing is saved."""
    me = body.data.copy()
    me.name = name
    ob = bpy.data.objects.new(name, me)
    ob.matrix_world = body.matrix_world.copy()
    bpy.context.scene.collection.objects.link(ob)
    palette = {g: COLOR_GARMENT[k % len(COLOR_GARMENT)] for k, g in enumerate(garment_order)}
    attr = me.color_attributes.new("coverage", 'BYTE_COLOR', 'POINT')
    for i in range(len(me.vertices)):
        if result["cut"][i]:
            col = COLOR_CUT
        elif result["covered"][i]:
            col = palette.get(result["claimed_by"][i], COLOR_GARMENT[0])
        elif result["near"][i]:
            col = COLOR_DECLINED
        else:
            col = COLOR_KEPT
        attr.data[i].color_srgb = [c / 255.0 for c in col]
    me.color_attributes.active_color = attr
    me.color_attributes.render_color_index = list(me.color_attributes).index(attr)
    # already-cut triangles are gone in every sheet (they are gone in the build too), so
    # the review reads the body as it ships; ``remove_carrier`` drops the carrier's as well
    import bmesh
    carrier = set(result["carrier"]) if remove_carrier else set()
    cut = result["cut"]
    bm = bmesh.new()
    bm.from_mesh(me)
    doomed = [f for f in bm.faces
              if any(cut[v.index] for v in f.verts) or any(v.index in carrier for v in f.verts)]
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context='FACES')
    bm.to_mesh(me)
    bm.free()
    me.update()
    return ob


def remove_marked_copy(ob):
    me = ob.data
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.meshes.remove(me)


def write_carrier(me, result: Dict, *, shape_name: str, delta: float, replace: bool = False):
    """Add the carrier shape key to mesh ``me`` (the body's own, editable datablock):
    relative to Basis, value 0, each carrier vertex displaced ``delta`` along its inverted
    normal so the consumer's threshold catches it. Refuses a vertex-count or basis-hash
    mismatch against the measurement, and a same-named key unless ``replace``."""
    if not me.is_editable or me.library is not None:
        raise CoverageError("%s is library data and cannot take a shape key" % me.name)
    if len(me.vertices) != result["vertex_count"]:
        raise CoverageError("%s has %d vertices, the measured body %d: re-run the mark"
                            % (me.name, len(me.vertices), result["vertex_count"]))
    if basis_hash(me) != result["basis_hash"]:
        raise CoverageError("%s Basis differs from the measured body: re-run the mark"
                            % me.name)
    ob = next((o for o in bpy.data.objects if o.data == me), None)
    if ob is None:
        raise CoverageError("no object uses mesh %s; a shape key needs one" % me.name)
    if me.shape_keys is None:
        ob.shape_key_add(name="Basis", from_mix=False)
    existing = me.shape_keys.key_blocks.get(shape_name)
    if existing is not None:
        if not replace:
            raise CoverageError("shape key %r already exists on %s (pass --replace)"
                                % (shape_name, me.name))
        kb = existing
    else:
        kb = ob.shape_key_add(name=shape_name, from_mix=False)
    basis = me.shape_keys.key_blocks[0]
    kb.relative_key = basis
    kb.value = 0.0
    for i in range(len(me.vertices)):
        kb.data[i].co = basis.data[i].co
    for i in result["carrier"]:
        kb.data[i].co = basis.data[i].co - me.vertices[i].normal * delta
    return kb
