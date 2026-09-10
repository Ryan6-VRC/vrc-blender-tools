"""Co-moving coverage: mark body triangles a garment set hides, as a Delete carrier.

A body vertex is *covered* when no line of sight reaches it that a co-moving garment does
not block. Rays leave the skin point in a cone around the skin normal; a ray is blocked
only by a listed garment face that rides the same bones as the skin point, because
geometry skinned to the same skeleton deforms together, so what hides the skin in rest
pose hides it in every pose. A face on other bones (a skirt on its physbone chain over a
thigh) swings away and is transparent to the ray, which marches on. The body never blocks
its own rays: an intelligently authored body has no triangle that hides behind another
part of itself in every pose.

Per body vertex (the first failure is the decline reason):

  unweighted  the vertex's raw weight sum is under ``min_raw_weight``
  swings      some ray met only garment faces on bones the vertex does not ride
  escaped     some ray reached ``reach`` without meeting any listed garment at all

"Same bones" is bone-name membership with two allowances: ``kin`` parent-or-child steps
of the armature count as the same bone (chest to breast bone through its root; hips to
the skirt's first ring), and ``fold`` globs name bones that read as their nearest unfolded
ancestor, for a physbone the operator knows is stiff. The hit face's weight share on the
allowed set must reach ``share``; blend ratio is otherwise ignored, since a mismatch on
shared bones clips in game and is not a reason to keep a triangle. Only vertex groups
named for a bone of the armature count as skinning when there is one.

The consumer is a Modular Avatar ShapeChanger Delete: a triangle is removed when ANY of
its vertices moves more than the component's threshold under the shape. The carrier is
therefore every vertex whose every incident POLYGON is fully covered or already cut —
polygons, not fan triangles, because neither the FBX exporter nor Unity promises Blender's
split, and a quad with one uncovered corner must never go. The carrier removes exactly
the polygons touching it; ``residue`` is covered but has no interior vertex to carry it.
``rest_visible`` counts removed triangles a rest-pose ray can still reach from the
centroid (to 5 degrees short of edge-on), with every listed garment blocking regardless
of bones: the cost of a cone narrower than the hemisphere, reported so it is never silent.

Pure ``bpy`` data access, no operators, no UI. Door: ``cli/mark_coverage.py``.
"""
import contextlib
import fnmatch
import hashlib
import math
from collections import defaultdict
from typing import Dict, Iterable, Optional, Sequence

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

REASONS = ("unweighted", "swings", "escaped")


class CoverageError(Exception):
    """A refusal the door prints in-grammar (unresolvable input, changed body)."""


# --- mesh access ---------------------------------------------------------------------

def evaluated_mesh(obj, depsgraph):
    """The object's depsgraph-evaluated mesh (shape-key values applied, modifiers
    included) as ``(evaluated_object, mesh)``; the caller clears it with
    ``evaluated_object.to_mesh_clear()``. Refuses a modifier that changes the vertex or
    polygon count: weights, the carrier and the removed polygons are indexed against the
    original mesh, and a Triangulate keeps every vertex while rewriting every face."""
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    for what, n_ev, n_orig in (("vertex", len(me.vertices), len(obj.data.vertices)),
                               ("polygon", len(me.polygons), len(obj.data.polygons))):
        if n_ev != n_orig:
            ev.to_mesh_clear()
            raise CoverageError("%s: a modifier changes the %s count (%d -> %d); coverage "
                                "is indexed against the original mesh, so disable it first"
                                % (obj.name, what, n_orig, n_ev))
    return ev, me


def normal_matrix(mw):
    """The transform for normals under ``mw``: the inverse transpose, so non-uniform
    scale keeps them perpendicular and a mirrored object keeps them pointing out."""
    return mw.inverted_safe().transposed().to_3x3()


def world_verts_normals(obj, me):
    mw = obj.matrix_world
    rot = normal_matrix(mw)
    return ([mw @ v.co for v in me.vertices],
            [(rot @ v.normal).normalized() for v in me.vertices])


def normalized_weights(obj, me, max_bones: int, floor: float = 0.01, bones: Optional[set] = None):
    """Per vertex: ``(raw_sum, {group_name: w})`` with the top ``max_bones`` groups kept
    and renormalised to sum 1 — the vector Unity skins with. Groups under ``floor`` are
    dropped before truncation. With ``bones`` given, only groups named for a bone count:
    a mask or helper group two meshes happen to share is not skinning."""
    names = [g.name for g in obj.vertex_groups]
    out = []
    for v in me.vertices:
        raw = {}
        for g in v.groups:
            if g.weight >= floor and g.group < len(names) and (bones is None or names[g.group] in bones):
                raw[names[g.group]] = raw.get(names[g.group], 0.0) + g.weight
        raw_sum = sum(raw.values())
        top = sorted(raw.items(), key=lambda kv: -kv[1])[:max_bones]
        s = sum(w for _, w in top) or 1.0
        out.append((raw_sum, {k: w / s for k, w in top}))
    return out


def _face_weights(g_w, corners):
    """Mean of the corners' normalised weight vectors — the garment's weights over the
    face a ray hit — over the corners that carry any weight, so an unweighted corner
    does not thin the face's share."""
    out = defaultdict(float)
    weighted = [vi for vi in corners if g_w[vi][1]]
    for vi in weighted:
        for k, w in g_w[vi][1].items():
            out[k] += w / len(weighted)
    return out


def cone_directions(normal, cone_deg: float, ring_deg: float = 15.0):
    """The rays: the normal itself plus rings of directions at even fractions of
    ``cone_deg`` off it, one ring per ``ring_deg``, with azimuth count growing with the
    ring's circumference (8 to 24) so a wide cone is not sparser. World-space unit
    vectors."""
    n = Vector(normal).normalized()
    up = Vector((0.0, 0.0, 1.0)) if abs(n.z) < 0.9 else Vector((1.0, 0.0, 0.0))
    u = n.cross(up).normalized()
    v = n.cross(u).normalized()
    out = [n]
    rings = max(1, int(round(cone_deg / ring_deg)))
    step = math.radians(cone_deg) / rings
    for r in range(1, rings + 1):
        theta = step * r
        st, ct = math.sin(theta), math.cos(theta)
        azimuths = min(24, max(8, int(round(8 * st / math.sin(step)))))
        for k in range(azimuths):
            phi = 2.0 * math.pi * k / azimuths
            out.append((n * ct + (u * math.cos(phi) + v * math.sin(phi)) * st).normalized())
    return out


def basis_hash(me) -> str:
    """Order-sensitive hash of the mesh's Basis coordinates (shape key 0 when present,
    else the vertex coordinates), so a reordered or edited body is caught before an
    index-list carrier lands on it."""
    h = hashlib.sha256()
    src = me.shape_keys.key_blocks[0].data if me.shape_keys else me.vertices
    for d in src:
        h.update(("%.6f %.6f %.6f;" % tuple(d.co)).encode())
    return h.hexdigest()


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


# --- bones ---------------------------------------------------------------------------

def bone_parents(objects: Iterable) -> Dict[str, Optional[str]]:
    """``{bone: parent}`` merged by name over every armature the objects' Armature
    modifiers target — merging by name is what MA's zip does at build. Empty when no
    object has one."""
    out = {}
    for ob in objects:
        for m in getattr(ob, "modifiers", []):
            if m.type == 'ARMATURE' and m.object is not None:
                for b in m.object.data.bones:
                    out.setdefault(b.name, b.parent.name if b.parent else None)
    return out


class BoneKin:
    """The bone-name allowance: ``fold`` globs collapse a bone onto its nearest unfolded
    ancestor, then ``kin`` parent-or-child steps count as the same bone."""

    def __init__(self, parents: Dict[str, Optional[str]], kin: int, fold: Sequence[str] = ()):
        self.parents = parents
        self.kin = kin
        self.fold = list(fold)
        self.children = defaultdict(list)
        for b, p in parents.items():
            if p:
                self.children[p].append(b)
        self._fold_cache = {}
        self._set_cache = {}

    def folded(self, bone: str) -> Optional[str]:
        if bone in self._fold_cache:
            return self._fold_cache[bone]
        b = bone
        while b is not None and any(fnmatch.fnmatchcase(b, pat) for pat in self.fold):
            b = self.parents.get(b)
        self._fold_cache[bone] = b
        return b

    def allowed(self, bones: Iterable[str]) -> frozenset:
        key = frozenset(bones)
        if key in self._set_cache:
            return self._set_cache[key]
        seen = {self.folded(b) for b in key} - {None}
        frontier = set(seen)
        for _ in range(self.kin):
            nxt = set()
            for b in frontier:
                p = self.parents.get(b)
                if p:
                    nxt.add(p)
                nxt.update(self.children.get(b, ()))
            nxt -= seen
            seen |= nxt
            frontier = nxt
        out = frozenset(seen)
        self._set_cache[key] = out
        return out

    def share(self, face_weights: Dict[str, float], allowed: frozenset) -> float:
        return sum(w for k, w in face_weights.items() if self.folded(k) in allowed)


# --- configuration shapes --------------------------------------------------------------

@contextlib.contextmanager
def shaped(objects: Sequence, shapes: Dict[str, float]):
    """Set each shape's value on every listed mesh that carries the key, for the block's
    duration, then restore what was there. Library data takes the value in memory (it is
    never saved), so the linked body measures, renders and saves in its configuration with
    every modifier it really has. Refuses a shape no listed mesh has."""
    saved = []
    try:
        for name, value in shapes.items():
            hit = False
            for ob in objects:
                keys = ob.data.shape_keys
                kb = keys.key_blocks.get(name) if keys else None
                if kb is None:
                    continue
                hit = True
                saved.append((kb, kb.value))
                kb.value = value
            if not hit:
                raise CoverageError("shape %r is on none of %s" % (name, ", ".join(o.name for o in objects)))
        yield
    finally:
        for kb, value in reversed(saved):
            kb.value = value


# --- the measurement -----------------------------------------------------------------

def measure(body, garments, *, cone_deg: float = 75.0, share: float = 0.5, reach: float = 1.0,
            kin: int = 2, fold: Sequence[str] = (), shapes: Optional[Dict[str, float]] = None,
            cut_threshold: float = 0.01, cut_shapes: Sequence[str] = (), ring_deg: float = 15.0,
            max_bones: int = 4, min_raw_weight: float = 0.5) -> Dict:
    """Measure coverage of ``body`` by ``garments`` (mesh objects). Returns per-vertex
    ``covered`` / ``claimed_by`` / ``cut`` / ``near`` lists, the carrier, the removed
    polygons, triangle counts, decline counts and the body's basis hash. Raises
    ``CoverageError`` on an unresolvable input; leaves the scene as it found it."""
    if body is None or body.type != 'MESH':
        raise CoverageError("body is not a mesh object")
    if not garments:
        raise CoverageError("no garments named; the cover set is an explicit input")
    for g in garments:
        if g is None or g.type != 'MESH':
            raise CoverageError("garment %r is not a mesh object" % getattr(g, "name", g))
        if g == body:
            raise CoverageError("%s is both the body and a garment" % body.name)

    parents = bone_parents([body] + list(garments))
    if kin > 0 and not parents:
        raise CoverageError("kin=%d needs an Armature modifier on the body or a garment; none found "
                            "(pass kin 0 for unrigged meshes)" % kin)
    kinship = BoneKin(parents, kin, fold)

    with shaped([body] + list(garments), shapes or {}):
        dg = bpy.context.evaluated_depsgraph_get()
        return _measure(body, garments, dg, kinship, cone_deg, share, reach, ring_deg,
                        cut_threshold, cut_shapes, max_bones, min_raw_weight, shapes or {})


def _measure(body, garments, dg, kinship, cone_deg, share, reach, ring_deg,
             cut_threshold, cut_shapes, max_bones, min_raw_weight, shapes):
    bones = set(kinship.parents) if kinship.parents else None
    b_ev, b_me = evaluated_mesh(body, dg)
    try:
        b_verts, b_norms = world_verts_normals(body, b_me)
        b_w = normalized_weights(body, b_me, max_bones, bones=bones)
        polys = [tuple(p.vertices) for p in b_me.polygons]
        rot = normal_matrix(body.matrix_world)
        poly_normals = [(rot @ p.normal).normalized() for p in b_me.polygons]
    finally:
        b_ev.to_mesh_clear()
    b_hash = basis_hash(body.data)
    cut, unknown_cuts = moved_vertices(body.data, cut_shapes, cut_threshold)
    if unknown_cuts:
        raise CoverageError("cut shape(s) not on %s: %s" % (body.name, ", ".join(unknown_cuts)))

    trees = []   # (name, BVHTree, per-face weights)
    for g in garments:
        g_ev, g_me = evaluated_mesh(g, dg)
        try:
            gmw = g.matrix_world
            g_verts = [gmw @ v.co for v in g_me.vertices]
            g_polys = [tuple(p.vertices) for p in g_me.polygons]
            tree = BVHTree.FromPolygons(g_verts, g_polys)
            g_w = normalized_weights(g, g_me, max_bones, bones=bones)
        finally:
            g_ev.to_mesh_clear()
        trees.append((g.name, tree, [_face_weights(g_w, p) for p in g_polys]))

    n = len(b_verts)
    covered = [False] * n
    near = [False] * n
    claimed_by = [None] * n
    reasons = defaultdict(int)
    eps = 1e-4

    def march(origin, direction, allowed):
        """Follow one ray past transparent faces: (blocking garment or None, met anything)."""
        o, left, hit_any = origin, reach, False
        for _ in range(8):
            best = None
            for name, tree, fw in trees:
                loc, _nor, idx, dist = tree.ray_cast(o, direction, left)
                if loc is not None and (best is None or dist < best[3]):
                    best = (name, fw[idx], loc, dist)
            if best is None:
                return None, hit_any
            hit_any = True
            name, fw, loc, dist = best
            if kinship.share(fw, allowed) >= share:
                return name, True
            o = loc + direction * eps
            left -= dist + eps
            if left <= 0:
                return None, hit_any
        return None, hit_any

    for i in range(n):
        raw_sum, bw = b_w[i]
        if raw_sum < min_raw_weight:
            reasons["unweighted"] += 1
            continue
        allowed = kinship.allowed(bw.keys())
        origin = b_verts[i] + b_norms[i] * eps
        ok = True
        for d in cone_directions(b_norms[i], cone_deg, ring_deg):
            blocker, hit_any = march(origin, d, allowed)
            if hit_any:
                near[i] = True
            if blocker is None:
                reasons["swings" if hit_any else "escaped"] += 1
                ok = False
                break
            if claimed_by[i] is None:
                claimed_by[i] = blocker
        covered[i] = ok
        if not ok:
            claimed_by[i] = None

    # the carrier, at polygon granularity against the consumer's any-vertex rule
    vpolys = defaultdict(list)
    for pi, p in enumerate(polys):
        for v in p:
            vpolys[v].append(pi)
    poly_covered = [all(covered[v] for v in p) for p in polys]
    poly_cut = [any(cut[v] for v in p) for p in polys]
    carrier = [i for i in range(n)
               if covered[i] and not cut[i] and vpolys[i]
               and all(poly_covered[pi] or poly_cut[pi] for pi in vpolys[i])]
    carrier_set = set(carrier)
    removed = [pi for pi, p in enumerate(polys) if not poly_cut[pi] and any(v in carrier_set for v in p)]

    def tri_count(indices):
        return sum(len(polys[pi]) - 2 for pi in indices)

    n_cut = tri_count(pi for pi in range(len(polys)) if poly_cut[pi])
    n_covered = tri_count(pi for pi in range(len(polys)) if poly_covered[pi] and not poly_cut[pi])
    n_realised = tri_count(removed)

    # rest-pose visibility of what goes: every listed garment blocks, bones ignored; the
    # sweep stops 5 degrees short of the tangent, where a face is edge-on and a ray runs
    # along an open tube without ever meeting it
    rest_visible = []
    for pi in removed:
        p = polys[pi]
        c = sum((b_verts[v] for v in p), Vector()) / len(p)
        o = c + poly_normals[pi] * eps
        for d in cone_directions(poly_normals[pi], 85.0, 10.0):
            if all(tree.ray_cast(o, d, reach)[0] is None for _, tree, _ in trees):
                rest_visible.append(pi)
                break

    return {
        "body": body.name,
        "body_mesh": body.data.name,
        "body_library": body.data.library.filepath if body.data.library else None,
        "vertex_count": n,
        "basis_hash": b_hash,
        "triangles": tri_count(range(len(polys))),
        "already_cut_triangles": n_cut,
        "covered_triangles": n_covered,
        "realised_triangles": n_realised,
        "residue_triangles": n_covered - n_realised,
        "rest_visible_triangles": tri_count(rest_visible),
        "carrier": carrier,
        "removed_polygons": removed,
        "covered": covered,
        "claimed_by": claimed_by,
        "cut": cut,
        "near": near,
        "declined": {r: reasons[r] for r in REASONS if reasons[r]},
        "by_garment": {g.name: sum(1 for c in claimed_by if c == g.name) for g in garments},
        "settings": {"cone_deg": cone_deg, "share": share, "reach_m": reach, "kin": kinship.kin,
                     "fold": list(kinship.fold), "shapes": dict(shapes), "ring_deg": ring_deg,
                     "cut_threshold_m": cut_threshold, "cut_shapes": list(cut_shapes),
                     "max_bones": max_bones, "min_raw_weight": min_raw_weight,
                     "bone_graph": len(kinship.parents)},
    }


# --- outputs -------------------------------------------------------------------------

# vertex colours (sRGB bytes) for the review sheet; garments cycle through the palette
COLOR_KEPT = (190, 190, 190, 255)
COLOR_CUT = (40, 40, 40, 255)
COLOR_DECLINED = (245, 170, 40, 255)      # a garment was met, none co-moving on every ray
COLOR_GARMENT = [(230, 40, 40, 255), (40, 100, 230, 255), (40, 180, 80, 255),
                 (200, 40, 200, 255), (40, 200, 200, 255), (230, 120, 40, 255)]


def marked_copy(body, result: Dict, *, name: str, garment_order: Sequence[str],
                remove_carrier: bool = False, shapes: Optional[Dict[str, float]] = None):
    """A LOCAL copy of the body mesh (geometry, vertex groups, no library data) linked
    into the scene collection as ``name``, carrying a ``coverage`` colour attribute:
    kept grey, near-but-declined amber, covered coloured by claiming garment. Already-cut
    polygons are removed in every copy; with ``remove_carrier`` the polygons the carrier
    deletes are gone too, the built result. ``shapes`` are set on the copy so it shows the
    configuration that was measured. The caller removes it; nothing is saved."""
    me = body.data.copy()
    me.name = name
    ob = bpy.data.objects.new(name, me)
    ob.matrix_world = body.matrix_world.copy()
    bpy.context.scene.collection.objects.link(ob)
    if shapes and me.shape_keys:
        for k, v in shapes.items():
            kb = me.shape_keys.key_blocks.get(k)
            if kb is not None:
                kb.value = v
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
    import bmesh
    doomed_idx = set(result["removed_polygons"]) if remove_carrier else set()
    cut = result["cut"]
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    doomed = [f for f in bm.faces if f.index in doomed_idx or any(cut[v.index] for v in f.verts)]
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


def save_marked(path: str, body, result: Dict, garments: Sequence, *, label: str,
                shapes: Optional[Dict[str, float]] = None) -> str:
    """Save a COPY of the open file to ``path`` holding the marked body and the
    carrier-removed body beside the garments, everything else hidden, the removed body
    visible and the viewport in plain solid shading so hems read at the cut. The open
    file's own path and contents are untouched. ``shapes`` are set on the garments too,
    so the saved copy shows the configuration that was measured."""
    names = [g.name for g in garments]
    with shaped([body] + list(garments), shapes or {}):
        marked = marked_copy(body, result, name=label + "_marked", garment_order=names, shapes=shapes)
        removed = marked_copy(body, result, name=label + "_removed", garment_order=names,
                              remove_carrier=True, shapes=shapes)
        keep = {marked.name, removed.name} | set(names)
        hidden = []
        for ob in bpy.context.scene.objects:
            if ob.name not in keep and not ob.hide_get():
                ob.hide_set(True)
                hidden.append(ob)
        marked.hide_set(True)
        try:
            for area in (bpy.context.screen.areas if bpy.context.screen else ()):
                if area.type == 'VIEW_3D':
                    for sp in area.spaces:
                        if sp.type == 'VIEW_3D':
                            sp.shading.type = 'SOLID'
                            sp.shading.color_type = 'SINGLE'
            bpy.ops.wm.save_as_mainfile(filepath=path, copy=True, relative_remap=True)
        finally:
            for ob in hidden:
                ob.hide_set(False)
            remove_marked_copy(marked)
            remove_marked_copy(removed)
    return path


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
    if shape_name == me.shape_keys.key_blocks[0].name:
        raise CoverageError("%r is the Basis of %s; the carrier needs its own key" % (shape_name, me.name))
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
