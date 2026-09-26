"""Transfer a body's skin weights onto skinned garments while every garment bone keeps its weight exactly.

Algorithm: Abdrashitov, Raichstat, Monsen & Hill, *Robust Skin Weights Transfer via Weight
Inpainting* (SIGGRAPH Asia 2023), ported from its MIT reference code
(RobustSkinWeightsTransferCode). The inpaint's Laplacian is robust-laplacian's point-cloud
Laplacian (Nicholas Sharp, MIT). sentfromspacevr's Robust Weight Transfer add-on (GPL)
first combined the two with flipped-normal matching; both additions are rebuilt here from
their MIT sources and no line of the add-on is used. The influence limit is this module's
own design.

Per target mesh, in world space at the current shape-key mix plus ``shapes``:

1. **Match.** Each target vertex takes the closest point on the source's loop triangles
   (a BVH picks the triangle, the closest point and barycentrics are recomputed in float64),
   interpolating the source's deform weights and vertex normal there. It is matched when
   within ``max_distance`` and within ``normal_angle`` of the source normal, or of its
   negation unless ``flip`` is off.
2. **Inpaint.** Unmatched vertices take the minimiser of ``w^T Q w`` with matched weights
   fixed, ``Q = L + L M^-1 L`` over the point-cloud Laplacian of every target vertex, solved
   by sparse LU; clipped to [0, 1] after the solve, since the biharmonic fill overshoots.
   A failed solve refuses and names the loose parts that hold no matched vertex.
3. **Smooth** (optional). ``steps`` Jacobi passes of ``(1 - f) w + f * mean(one-ring incl.
   self)`` on the vertices an edge walk from an unmatched vertex reaches within
   ``max_distance`` of it. Jacobi is the add-on's form rather than the paper's Gauss-Seidel
   sweep, kept so its recorded ``--smooth`` lines stay meaningful; the walk is the paper's,
   without its dependence on seed order.
4. **Narrow the source** (optional). Case-insensitive globs over the source armature's
   deform bones, each hit taking its descendants; a source vertex whose share on those
   bones exceeds ``exclude_max`` is dropped with every triangle touching it.
5. **Blend** (optional). Whole and narrowed transfers, each normalised, mixed by a
   smoothstep of rest position along a frame axis from the reference bone's head, times an
   optional lateral fade; then Laplacian steps on the band near boundary loops whose mean
   share on the excluded bones exceeds ``exclude_max`` (leg holes). No such loop refuses
   unless ``allow_no_loops``.
6. **Limit.** A weight at or under ``EPS`` is zero; each vertex keeps its top
   ``UNITY_BONES - garment_groups`` body weights, ties broken by column order.
7. **Write.** Garment bones are the target armature's deform bones the source armature
   lacks. A vertex's garment total ``p`` is never written; its body groups are replaced by
   the transfer normalised to ``1 - p``. A vertex at ``p ~ 1``, with no allowance left, or
   outside ``mask`` is left alone. Groups naming no source deform bone are untouched.
8. **Refuse / assert.** Refused before anything is written: a written vertex with no body
   weight; transferred weight on a bone the target armature lacks, anywhere in the matched
   or written region; a target whose baked shape state disagrees with the source's; no
   leg-hole loop for the smoothing. Asserted after writing (nothing is saved on a failure):
   garment and non-bone groups bit-identical, unwritten vertices identical, no written
   vertex over ``UNITY_BONES`` bone groups, shape-key metadata and datablock counts unchanged.

The source is evaluated from an in-memory copy with its Armature modifier disabled, because
a linked body armature evaluates at whatever pose its library saved. Targets evaluate with
their Armature modifiers disabled and their editable armatures at rest, both restored.
scipy and robust_laplacian import inside the functions that use them, so the package
registers without them. Pure ``bpy`` data access. Door: ``cli/transfer_weights.py``.
"""
import contextlib
import fnmatch
import math
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import scene_utils

EPS = 1e-4
UNITY_BONES = 4
AXES = ("forward", "back", "lateral", "up")
VIZ_LAYER = "avatarprep_matched"


class WeightTransferError(ValueError):
    """A refusal the door prints in-grammar; names the offender and the fix."""


# --- scene access --------------------------------------------------------------------

def armature_of(ob):
    """The one armature ``ob``'s single Armature modifier targets, and that modifier."""
    mods = [m for m in ob.modifiers if m.type == 'ARMATURE']
    if len(mods) != 1 or mods[0].object is None or mods[0].object.type != 'ARMATURE':
        raise WeightTransferError("%s needs exactly one Armature modifier with an armature object "
                                  "(found %d)" % (ob.name, len(mods)))
    return mods[0].object, mods[0]


def deform_bones(arm) -> set:
    return {b.name for b in arm.data.bones if b.use_deform}


def mesh_arrays(ob, me):
    """World vertices (float32), loop triangles (int64) and world vertex normals (float32,
    unnormalised) of ``me`` evaluated for ``ob``: the arrays both match sides use."""
    n = len(me.vertices)
    co = np.empty((n, 3), np.float32)
    nrm = np.empty((n, 3), np.float32)
    me.vertices.foreach_get("co", co.ravel())
    me.vertices.foreach_get("normal", nrm.ravel())
    me.calc_loop_triangles()
    tris = np.empty((len(me.loop_triangles), 3), np.int64)
    me.loop_triangles.foreach_get("vertices", tris.ravel())
    M = np.array(ob.matrix_world, np.float64)
    V = (np.c_[co.astype(np.float64), np.ones(n)] @ M.T)[:, :3].astype(np.float32)
    N = (nrm.astype(np.float64) @ np.linalg.inv(M[:3, :3])).astype(np.float32)
    return V, tris, N


def group_matrix(ob) -> Tuple[List[str], np.ndarray]:
    """``(names, W)``: every vertex group of ``ob`` and its stored weights, float32 (V, G)."""
    names = [g.name for g in ob.vertex_groups]
    W = np.zeros((len(ob.data.vertices), len(names)), np.float32)
    for i, v in enumerate(ob.data.vertices):
        for g in v.groups:
            if g.group < len(names):
                W[i, g.group] = g.weight
    return names, W


def _evaluated(ob, dg):
    """``(evaluated_object, mesh)``; refuses a modifier that changes the vertex or polygon
    count, since weights are indexed against the original mesh."""
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    for what, a, b in (("vertex", len(ob.data.vertices), len(me.vertices)),
                       ("polygon", len(ob.data.polygons), len(me.polygons))):
        if a != b:
            ev.to_mesh_clear()
            raise WeightTransferError("%s: a modifier changes the %s count (%d -> %d); weights are "
                                      "indexed against the original mesh, so disable or apply it first"
                                      % (ob.name, what, a, b))
    return ev, me


def adjacency(me, include_self=True):
    """Symmetric vertex adjacency over the mesh's edges, float64 CSR."""
    import scipy.sparse as sp
    e = np.empty((len(me.edges), 2), np.int64)
    me.edges.foreach_get("vertices", e.ravel())
    n = len(me.vertices)
    A = sp.csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n))
    A.data[:] = 1.0
    if include_self:
        A = (A + sp.identity(n, format="csr")).tocsr()
        A.data[:] = 1.0
    return A


def boundary_loops(me) -> List[np.ndarray]:
    """Vertex index arrays of the mesh's boundary loops (edges used by one polygon), loops
    sharing a vertex merged."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    count = {}
    for p in me.polygons:
        for k in p.edge_keys:
            count[k] = count.get(k, 0) + 1
    edges = np.array([k for k, c in count.items() if c == 1], np.int64).reshape(-1, 2)
    if not len(edges):
        return []
    n = len(me.vertices)
    G = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    on = np.unique(edges)
    return [on[lab[on] == c] for c in np.unique(lab[on])]


def loose_parts(A, V, matched) -> List[Dict]:
    """Connected parts of the mesh holding no matched vertex: size and centre (mm)."""
    from scipy.sparse.csgraph import connected_components
    _, lab = connected_components(A, directed=False)
    out = []
    for c in np.unique(lab):
        vs = np.flatnonzero(lab == c)
        if not matched[vs].any():
            out.append({"verts": int(len(vs)),
                        "centre_mm": [round(float(x) * 1000.0, 1) for x in V[vs].mean(0)]})
    return out


# --- configuration shapes --------------------------------------------------------------

def resolve_shapes(shapes: Sequence[Tuple[Optional[str], str, float]], meshes: Sequence) -> Dict[str, Dict[str, float]]:
    """``{mesh_name: {key: value}}`` from ``(mesh_or_None, key, value)`` triples. A bare key
    lands on every listed mesh carrying it and is refused when none does; a scoped key is
    refused when its mesh is not listed or lacks the key."""
    by = {m.name: m for m in meshes}
    out = {m.name: {} for m in meshes}

    def has(m, k):
        return m.data.shape_keys is not None and k in m.data.shape_keys.key_blocks

    for mesh, key, value in shapes:
        if mesh is not None:
            if mesh not in by:
                raise WeightTransferError("--shape %s:%s names a mesh that is neither the source nor a "
                                          "target (%s)" % (mesh, key, ", ".join(by)))
            if not has(by[mesh], key):
                raise WeightTransferError("--shape %s:%s: %s has no shape key %r" % (mesh, key, mesh, key))
            out[mesh][key] = value
            continue
        hit = [m for m in meshes if has(m, key)]
        if not hit:
            raise WeightTransferError("--shape %s is on none of %s" % (key, ", ".join(by)))
        for m in hit:
            out[m.name][key] = value
    return out


def _key_meta(me):
    if not me.shape_keys:
        return []
    return [(k.name, k.value, k.relative_key.name, k.vertex_group, k.mute, k.slider_min, k.slider_max)
            for k in me.shape_keys.key_blocks]


def _baked(ob) -> Dict[str, float]:
    raw = ob.get(scene_utils.STAMP_BAKED)
    if raw is None:
        return {}
    try:
        return {k: float(v) for k, v in dict(raw).items()}
    except (TypeError, ValueError):
        raise WeightTransferError("%s on %s is not a {shapekey: value} map (%r)"
                                  % (scene_utils.STAMP_BAKED, ob.name, raw))


def seat_disagreement(source, target) -> List[Tuple[str, float, float]]:
    """``(key, source_state, target_state)`` for each of the source's keys on which the
    target's baked-plus-live state differs from the source's by more than 1e-6. A key the
    target records as a negative bake is skipped: that is ``transfer_shapekeys`` un-baking
    an authored offset, seated by construction. Keys only the target carries are its own."""
    def state(ob):
        baked = _baked(ob)
        live = {k.name: float(k.value) for k in ob.data.shape_keys.key_blocks[1:]} if ob.data.shape_keys else {}
        return baked, {k: baked.get(k, 0.0) + live.get(k, 0.0) for k in set(baked) | set(live)}
    _, s = state(source)
    tb, t = state(target)
    out = []
    for k in sorted(s):
        if tb.get(k, 0.0) < -1e-6:
            continue
        if abs(s[k] - t.get(k, 0.0)) > 1e-6:
            out.append((k, round(s[k], 6), round(t.get(k, 0.0), 6)))
    return out


# --- step 1: match -------------------------------------------------------------------------

def closest_on_triangles(P, A, B, C):
    """Barycentrics (n, 3) of the closest point to each ``P`` on triangle ``(A, B, C)``,
    float64 (Ericson, *Real-Time Collision Detection* 5.1.5)."""
    P, A, B, C = (np.asarray(x, np.float64) for x in (P, A, B, C))
    ab, ac = B - A, C - A
    ap, bp, cp = P - A, P - B, P - C
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    out = np.zeros((len(P), 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        v, w = vb / den, vc / den
        out[:] = np.c_[1 - v - w, v, w]
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        r = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        out[r] = np.c_[np.zeros(r.sum()), 1 - t[r], t[r]]
        t = d2 / (d2 - d6)
        r = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out[r] = np.c_[1 - t[r], np.zeros(r.sum()), t[r]]
        r = (d6 >= 0) & (d5 <= d6)
        out[r] = (0.0, 0.0, 1.0)
        t = d1 / (d1 - d3)
        r = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out[r] = np.c_[1 - t[r], t[r], np.zeros(r.sum())]
        r = (d3 >= 0) & (d4 <= d3)
        out[r] = (0.0, 1.0, 0.0)
        r = (d1 <= 0) & (d2 <= 0)
        out[r] = (1.0, 0.0, 0.0)
    out[~np.isfinite(out).all(1)] = (1.0, 0.0, 0.0)  # a degenerate triangle: its first corner
    return out


def match(SV, SF, SN, SW, TV, TN, max_distance=0.05, normal_angle=30.0, flip=True) -> Dict:
    """Step 1. Plain arrays in (source vertices, triangles, normals, weights; target
    vertices, normals), dict out: ``matched`` (bool V), ``weights`` (V, G) float64,
    ``distance`` (m) and ``angle`` (degrees, the unflipped one), for every target vertex."""
    tree = BVHTree.FromPolygons(SV.tolist(), SF.tolist())
    tri = np.empty(len(TV), np.int64)
    for i, co in enumerate(TV.tolist()):
        tri[i] = tree.find_nearest(co)[2]
    F = SF[tri]
    bary = closest_on_triangles(TV, SV[F[:, 0]], SV[F[:, 1]], SV[F[:, 2]])
    SV64 = SV.astype(np.float64)
    near = np.einsum("nk,nkd->nd", bary, SV64[F])
    W = np.einsum("nk,nkg->ng", bary, SW.astype(np.float64)[F])
    Ns = np.einsum("nk,nkd->nd", bary, SN.astype(np.float64)[F])
    Nt = TN.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (Ns * Nt).sum(1) / (np.linalg.norm(Ns, axis=1) * np.linalg.norm(Nt, axis=1))
    angle = np.degrees(np.arccos(np.clip(np.nan_to_num(cos, nan=0.0), -1.0, 1.0)))
    dist = np.sqrt(((TV.astype(np.float64) - near) ** 2).sum(1))
    ok_angle = angle <= normal_angle
    if flip:
        ok_angle |= (180.0 - angle) <= normal_angle
    return {"matched": (dist <= max_distance) & ok_angle, "weights": W, "distance": dist, "angle": angle}


# --- step 2: inpaint -----------------------------------------------------------------------

def inpaint(TV, W, matched) -> np.ndarray:
    """Step 2. Unmatched rows of ``W`` replaced by the biharmonic fill over the point-cloud
    Laplacian of ``TV``; matched rows returned as given. Unclipped, float64. Raises
    ``WeightTransferError`` when the system cannot be solved."""
    import robust_laplacian
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    matched = np.asarray(matched, bool)
    if not matched.any():
        raise WeightTransferError("no matched vertex to inpaint from")
    out = np.array(W, np.float64)
    u, b = np.flatnonzero(~matched), np.flatnonzero(matched)
    if not len(u):
        return out
    L, M = robust_laplacian.point_cloud_laplacian(np.asarray(TV))
    L = sp.csr_matrix(L, dtype=np.float64)
    with np.errstate(divide="ignore"):
        Minv = sp.diags(1.0 / M.diagonal().astype(np.float64))
    Q = (L + L @ Minv @ L).tocsr()
    Qu = Q[u]
    try:
        X = splu(Qu[:, u].tocsc()).solve(-(Qu[:, b] @ out[b]))
    except (RuntimeError, ValueError) as e:
        raise WeightTransferError("the inpaint solve failed (%s)" % e)
    if not np.isfinite(X).all():
        raise WeightTransferError("the inpaint solve returned non-finite weights")
    out[u] = X.reshape(len(u), -1)
    return out


# --- step 3: smooth ------------------------------------------------------------------------

def smooth_band(TV, matched, A, radius):
    """The vertices an edge walk from each unmatched vertex reaches without leaving the
    ball of ``radius`` around it, the seeds included (the paper's reference selection, made
    independent of seed order)."""
    V = np.asarray(TV, np.float64)
    ptr, nbr = A.indptr, A.indices
    band = ~np.asarray(matched, bool)
    for seed in np.flatnonzero(band):
        c, seen, stack = V[seed], {int(seed)}, [int(seed)]
        while stack:
            v = stack.pop()
            for nb in nbr[ptr[v]:ptr[v + 1]]:
                nb = int(nb)
                if nb not in seen and np.linalg.norm(V[nb] - c) < radius:
                    seen.add(nb)
                    stack.append(nb)
        band[list(seen)] = True
    return band


def smooth(TV, W, matched, A, steps, factor, radius):
    """Step 3. ``steps`` Jacobi passes over self-inclusive adjacency ``A`` on
    ``smooth_band``; every other row held. Returns ``(W, band)``."""
    import scipy.sparse as sp
    band = np.zeros(len(TV), bool)
    if steps <= 0 or matched.all():
        return W, band
    band = smooth_band(TV, matched, A, radius)
    S = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    W0, W = W, W.copy()
    for _ in range(steps):
        W = (1.0 - factor) * W + factor * (S @ W)
        W[~band] = W0[~band]
    return W, band


# --- step 4: narrow the source ---------------------------------------------------------------

def excluded_bones(arm, globs: Sequence[str]) -> List[str]:
    """Deform bones of ``arm`` hit by a case-insensitive glob, with their descendants. A glob
    that hits no deform bone refuses."""
    deform = [b for b in arm.data.bones if b.use_deform]
    roots, empty = set(), []
    for g in globs:
        hit = [b.name for b in deform if fnmatch.fnmatchcase(b.name.lower(), g.lower())]
        if not hit:
            empty.append(g)
        roots.update(hit)
    if empty:
        raise WeightTransferError("--source-exclude %s matches no deform bone of %s"
                                  % (",".join(empty), arm.name))

    def under(b):
        while b is not None:
            if b.name in roots:
                return True
            b = b.parent
        return False
    return sorted(b.name for b in deform if under(b))


def narrow_source(SF, SW, names, excluded, exclude_max):
    """Step 4. ``(kept_triangles, dropped_vertex_mask)``: triangles touching a source vertex
    whose share on ``excluded`` exceeds ``exclude_max`` are dropped."""
    cols = [j for j, n in enumerate(names) if n in set(excluded)]
    tot = SW.sum(1).astype(np.float64)
    share = np.where(tot > 0, SW[:, cols].sum(1) / np.where(tot > 0, tot, 1.0), 0.0)
    drop = share > exclude_max
    keep = ~drop[SF].any(1)
    if not keep.any():
        raise WeightTransferError("--source-exclude %s at --source-exclude-max %g drops every source "
                                  "triangle" % (",".join(excluded), exclude_max))
    return SF[keep], drop


# --- the body frame ----------------------------------------------------------------------------

def body_frame(arm, reference_bone="Hips", forward_bone=None) -> Dict:
    """Frame axes as words, in world space. ``up`` is world +Z; ``forward`` is the rest toe
    head minus the foot head over every deform ``*foot*`` bone (its tail minus head where it
    has no ``*toe*`` child), or ``forward_bone``'s tail minus head, flattened to the ground;
    ``lateral`` is up x forward, the body's left (the ``.L`` side); origin is the reference
    bone's head."""
    bones = arm.data.bones
    M = arm.matrix_world
    ref = bones.get(reference_bone)
    if ref is None:
        raise WeightTransferError("--reference-bone %s is not a bone of %s" % (reference_bone, arm.name))

    def flat(v):
        return Vector((v.x, v.y, 0.0))

    if forward_bone:
        b = bones.get(forward_bone)
        if b is None:
            raise WeightTransferError("--forward %s is not a bone of %s" % (forward_bone, arm.name))
        vec, how = flat(M @ b.tail_local - M @ b.head_local), "--forward %s" % forward_bone
    else:
        feet = [b for b in bones if b.use_deform and fnmatch.fnmatchcase(b.name.lower(), "*foot*")]
        if not feet:
            raise WeightTransferError("no deform bone of %s matches *foot*, which forward is read from; "
                                      "pass --forward BONE naming a bone whose tail points the way the "
                                      "body faces" % arm.name)
        vec, parts = Vector(), []
        for f in feet:
            toes = [c for c in f.children if fnmatch.fnmatchcase(c.name.lower(), "*toe*")]
            if toes:
                vec += flat(M @ toes[0].head_local - M @ f.head_local)
                parts.append("%s->%s" % (f.name, toes[0].name))
            else:
                vec += flat(M @ f.tail_local - M @ f.head_local)
                parts.append("%s tail" % f.name)
        how = ", ".join(parts)
    if vec.length < 1e-6:
        raise WeightTransferError("forward from %s lies along up; pass --forward BONE" % how)
    up = Vector((0.0, 0.0, 1.0))
    fwd = vec.normalized()
    return {"origin": np.array(M @ ref.head_local), "forward": np.array(fwd), "back": -np.array(fwd),
            "lateral": np.array(up.cross(fwd)), "up": np.array(up), "from": how,
            "reference_bone": reference_bone}


# --- step 5: blend -----------------------------------------------------------------------------

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def blend_mix(TV, frame, axis, center, width, lateral=None):
    """The narrowed-transfer share per target vertex: 0 at ``center - width/2`` along
    ``axis`` from the frame origin, 1 at ``center + width/2``; times, with ``lateral`` as
    ``(center, width)``, a fade that is 1 up to ``center - width/2`` from the midline and
    0 from ``center + width/2``, the same on both sides."""
    d = TV.astype(np.float64) - frame["origin"]
    m = smoothstep((d @ frame[axis] - (center - width / 2.0)) / width)
    if lateral is not None:
        lc, lw = lateral
        m = m * (1.0 - smoothstep((np.abs(d @ frame["lateral"]) - (lc - lw / 2.0)) / lw))
    return m


def leg_hole_band(loops, TV, share, exclude_max, radius):
    """``(band, loop_count)``: vertices within ``radius`` (straight line; ``inf`` is every
    vertex) of a boundary loop whose mean ``share`` exceeds ``exclude_max``."""
    from scipy.spatial import cKDTree
    rims = [l for l in loops if share[l].mean() > exclude_max]
    if not rims:
        return np.zeros(len(TV), bool), 0
    if not math.isfinite(radius):
        return np.ones(len(TV), bool), len(rims)
    d, _ = cKDTree(TV[np.concatenate(rims)]).query(TV)
    return d <= radius, len(rims)


def laplacian_steps(W, A, band, steps, alpha):
    import scipy.sparse as sp
    S = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    W0, W = W, W.copy()
    for _ in range(steps):
        W = (1.0 - alpha) * W + alpha * (S @ W)
        W[~band] = W0[~band]
    return W


def _normalised(T):
    s = T.sum(1, keepdims=True)
    return np.where(s > 0, T / np.where(s > 0, s, 1.0), 0.0)


# --- step 6: limit -----------------------------------------------------------------------------

def limit(T, allow):
    """Step 6. ``(T, capped)``: weights at or under ``EPS`` zeroed, then each row cut to its
    ``allow`` largest (stable, so ties keep column order); ``capped`` marks the rows cut."""
    T = np.where(T > EPS, T, 0.0)
    allow = np.asarray(allow, np.int64)
    capped = (T > 0).sum(1) > allow
    for i in np.flatnonzero(capped):
        keep = np.argsort(-T[i], kind="stable")[:allow[i]]
        row = np.zeros_like(T[i])
        row[keep] = T[i, keep]
        T[i] = row
    return T, capped


# --- one transfer ------------------------------------------------------------------------------

def _transfer(TV, TN, src, SF, knobs, A):
    m = match(src["V"], SF, src["N"], src["W"], TV, TN, knobs["max_distance"], knobs["normal_angle"],
              knobs["flip"])
    if not m["matched"].any():
        raise WeightTransferError("no vertex matched the source within %g m and %g degrees"
                                  % (knobs["max_distance"], knobs["normal_angle"]))
    try:
        T = inpaint(TV, m["weights"], m["matched"])
    except WeightTransferError as e:
        parts = loose_parts(A, TV, m["matched"])
        raise WeightTransferError("%s; loose parts with no matched vertex: %s"
                                  % (e, _fmt_parts(parts) or "none"))
    T = np.clip(T, 0.0, 1.0)
    band = np.zeros(len(TV), bool)
    if knobs["smooth"]:
        T, band = smooth(TV, T, m["matched"], A, knobs["smooth"][0], knobs["smooth"][1],
                         knobs["max_distance"])
    return m, T, band


def _fmt_parts(parts):
    return ", ".join("%d verts at %s mm" % (p["verts"], ",".join("%.0f" % c for c in p["centre_mm"]))
                     for p in parts)


@contextlib.contextmanager
def _rest_state(targets, target_shapes):
    """Targets at rest for evaluation: Armature modifiers off, editable armatures at REST,
    ``--shape`` values on; all restored."""
    saved = []
    try:
        for t in targets:
            arm, mod = armature_of(t)
            saved.append((mod, "show_viewport", mod.show_viewport))
            mod.show_viewport = False
            if scene_utils.is_editable(arm) and arm.data.pose_position != 'REST':
                saved.append((arm.data, "pose_position", arm.data.pose_position))
                arm.data.pose_position = 'REST'
            for k, v in target_shapes.get(t.name, {}).items():
                kb = t.data.shape_keys.key_blocks[k]
                saved.append((kb, "value", kb.value))
                kb.value = v
        yield
    finally:
        for owner, attr, value in reversed(saved):
            setattr(owner, attr, value)


def source_arrays(source, shapes: Dict[str, float]) -> Dict:
    """Source vertices, loop triangles, normals (world) and deform weights, from an
    in-memory copy with ``shapes`` set and its Armature modifier disabled; the copy is
    removed before returning."""
    arm, _ = armature_of(source)
    deform = deform_bones(arm)
    tmp = source.copy()
    tmp.data = source.data.copy()
    bpy.context.scene.collection.objects.link(tmp)
    try:
        for m in tmp.modifiers:
            if m.type == 'ARMATURE':
                m.show_viewport = False
        for k, v in shapes.items():
            tmp.data.shape_keys.key_blocks[k].value = v
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        ev, me = _evaluated(tmp, dg)
        try:
            V, F, N = mesh_arrays(ev, me)
        finally:
            ev.to_mesh_clear()
    finally:
        me_copy = tmp.data
        bpy.data.objects.remove(tmp, do_unlink=True)
        bpy.data.meshes.remove(me_copy)
    names, Wall = group_matrix(source)
    cols = [j for j, n in enumerate(names) if n in deform]
    return {"V": V, "F": F, "N": N, "W": Wall[:, cols], "names": [names[j] for j in cols],
            "armature": arm, "deform": deform}


# --- the transfer ------------------------------------------------------------------------------

def transfer_weights(source, targets: Sequence, *, shapes=(), mask: Optional[str] = None,
                     max_distance=0.05, normal_angle=30.0, flip=True, smooth=None,
                     exclude: Sequence[str] = (), exclude_max: Optional[float] = None,
                     blend=None, blend_lateral=None, blend_smooth=None, allow_no_loops=False,
                     allow_unseated=False, reference_bone="Hips", forward_bone=None) -> Dict:
    """Transfer ``source``'s skin weights onto ``targets`` (mesh objects), in memory; the
    caller saves. See the module docstring for the steps.

    ``shapes``: ``(mesh_name_or_None, key, value)`` triples. ``smooth``: ``(steps, factor)``.
    ``blend``: ``(axis, center, width)`` with ``axis`` in ``AXES``; ``blend_lateral``:
    ``(center, width)``; ``blend_smooth``: ``(steps, alpha, radius)``. Distances in metres.

    Returns the report: ``source``, ``excluded`` (bones, dropped vertices and triangles),
    ``frame`` when a blend ran, and ``targets``, one row per mesh whose ``matched_mask``
    (numpy bool) the caller drops before serialising. Raises ``WeightTransferError``."""
    targets = list(targets)
    if source is None or source.type != 'MESH':
        raise WeightTransferError("the source is not a mesh object")
    if not targets:
        raise WeightTransferError("no targets named")
    src_arm, _ = armature_of(source)
    for t in targets:
        if t is None or t.type != 'MESH':
            raise WeightTransferError("target %r is not a mesh object" % getattr(t, "name", t))
        if t is source or t.data is source.data:
            raise WeightTransferError("%s is the source; the source is never written" % t.name)
        if scene_utils.is_linked(t) or not scene_utils.is_editable(t):
            raise WeightTransferError("%s is linked or override data and cannot be written" % t.name)
        if t.data.users > 1:
            raise WeightTransferError("%s shares its mesh with %d other user(s); make it single-user first"
                                      % (t.name, t.data.users - 1))
        armature_of(t)
        if mask is not None and mask not in t.vertex_groups:
            raise WeightTransferError("%s has no vertex group %r for --mask" % (t.name, mask))
        if not allow_unseated:
            bad = seat_disagreement(source, t)
            if bad:
                raise WeightTransferError(
                    "%s is not seated on %s: shape state differs on %s; seat it with transfer_shapekeys "
                    "first, or pass --allow-unseated to transfer onto it as it sits"
                    % (t.name, source.name, ", ".join("%s (source %g, target %g)" % b for b in bad)))
    if (exclude_max is None) != (not exclude):
        raise WeightTransferError("--source-exclude and --source-exclude-max go together")
    if exclude and not 0.0 <= exclude_max < 1.0:
        raise WeightTransferError("--source-exclude-max %g is outside [0, 1)" % exclude_max)
    if blend is not None:
        if not exclude:
            raise WeightTransferError("--exclude-blend mixes in the narrowed transfer; it needs --source-exclude")
        if blend[0] not in AXES:
            raise WeightTransferError("--exclude-blend axis %r is not one of %s" % (blend[0], ", ".join(AXES)))
        if not blend[2] > 0:
            raise WeightTransferError("--exclude-blend width must be positive")
        if blend_lateral is not None and blend[0] == "lateral":
            raise WeightTransferError("--exclude-blend-lateral fades across the lateral axis; the blend "
                                      "axis must be forward, back or up")
    elif blend_lateral is not None or blend_smooth is not None:
        raise WeightTransferError("--exclude-blend-lateral and --exclude-blend-smooth need --exclude-blend")
    if blend_lateral is not None and not blend_lateral[1] > 0:
        raise WeightTransferError("--exclude-blend-lateral width must be positive")
    if smooth is not None and (smooth[0] < 1 or not 0.0 < smooth[1] <= 1.0):
        raise WeightTransferError("--smooth wants N >= 1 passes at a factor in (0, 1]")

    per_mesh = resolve_shapes(shapes, [source] + targets)
    knobs = {"max_distance": max_distance, "normal_angle": normal_angle, "flip": flip, "smooth": smooth}
    ids0 = (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.shape_keys))
    keys0 = {o.name: _key_meta(o.data) for o in [source] + targets}

    src = source_arrays(source, per_mesh[source.name])
    report = {"source": source.name, "source_armature": src_arm.name, "excluded": None,
              "frame": None, "targets": []}
    SFx, xcols = None, []
    if exclude:
        xb = excluded_bones(src_arm, exclude)
        SFx, drop = narrow_source(src["F"], src["W"], src["names"], xb, exclude_max)
        xcols = [j for j, n in enumerate(src["names"]) if n in set(xb)]
        report["excluded"] = {"bones": xb, "max": exclude_max, "verts": int(drop.sum()),
                              "tris": int(len(src["F"]) - len(SFx)), "of_tris": int(len(src["F"]))}
    frame = None
    if blend is not None and (blend[0] != "up" or blend_lateral is not None):
        frame = body_frame(src_arm, reference_bone, forward_bone)
    elif blend is not None:
        frame = body_frame_up(src_arm, reference_bone)
    if frame is not None:
        report["frame"] = {k: ((v.round(6) + 0.0).tolist() if isinstance(v, np.ndarray) else v) for k, v in frame.items()}

    plans = []
    with _rest_state(targets, per_mesh):
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        for t in targets:
            ev, me = _evaluated(t, dg)
            try:
                TV, _, TN = mesh_arrays(ev, me)
            finally:
                ev.to_mesh_clear()
            plans.append(_plan(t, TV, TN, src, SFx, xcols, knobs, frame, blend, blend_lateral,
                               blend_smooth, exclude_max, allow_no_loops, mask))

    for plan in plans:
        _write(plan, src)
        report["targets"].append(plan["row"])

    if {o.name: _key_meta(o.data) for o in [source] + targets} != keys0:
        raise WeightTransferError("invariant broken (nothing saved): shape-key metadata changed")
    if (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.shape_keys)) != ids0:
        raise WeightTransferError("invariant broken (nothing saved): the source copy left datablocks behind")
    return report


def body_frame_up(arm, reference_bone):
    """The frame an ``up`` blend needs: origin and up only, no foot bones read."""
    ref = arm.data.bones.get(reference_bone)
    if ref is None:
        raise WeightTransferError("--reference-bone %s is not a bone of %s" % (reference_bone, arm.name))
    return {"origin": np.array(arm.matrix_world @ ref.head_local), "up": np.array((0.0, 0.0, 1.0)),
            "from": "world +Z", "reference_bone": reference_bone}


def _plan(t, TV, TN, src, SFx, xcols, knobs, frame, blend, blend_lateral, blend_smooth,
          exclude_max, allow_no_loops, mask):
    """Everything for one target short of writing it: the transfer, the limit, the new
    body weights and every refusal."""
    arm, _ = armature_of(t)
    garment = deform_bones(arm) - src["deform"]
    gnames, W0 = group_matrix(t)
    gcols = [i for i, n in enumerate(gnames) if n in garment]
    p = W0[:, gcols].sum(1).astype(np.float64)
    allow = np.clip(UNITY_BONES - (W0[:, gcols] > 0).sum(1), 0, UNITY_BONES)
    A = adjacency(t.data)
    row = {"mesh": t.name, "verts": len(TV)}

    na = knobs["normal_angle"]
    if blend is None:
        m, T, sband = _transfer(TV, TN, src, src["F"] if SFx is None else SFx, knobs, A)
        matched = m["matched"]
        flipped = matched & (m["angle"] > na)
    else:
        m, Tw, sband = _transfer(TV, TN, src, src["F"], knobs, A)
        mx, Tn, sbx = _transfer(TV, TN, src, SFx, knobs, A)
        Tw, Tn = _normalised(Tw), _normalised(Tn)
        mix = blend_mix(TV, frame, blend[0], blend[1], blend[2], blend_lateral)
        T = mix[:, None] * Tn + (1.0 - mix[:, None]) * Tw
        lband, nloops = np.zeros(len(TV), bool), 0
        if blend_smooth is not None and blend_smooth[0] > 0:
            loops = boundary_loops(t.data)
            lband, nloops = leg_hole_band(loops, TV, Tw[:, xcols].sum(1), exclude_max, blend_smooth[2])
            if nloops == 0 and not allow_no_loops:
                raise WeightTransferError(
                    "%s: --exclude-blend-smooth found no boundary loop whose mean share on the excluded "
                    "bones exceeds %g (%d boundary loops in all); pass --allow-no-loops to blend without "
                    "the smoothing" % (t.name, exclude_max, len(loops)))
            if nloops:
                T = laplacian_steps(T, A, lband, blend_smooth[0], blend_smooth[1])
        row["blend"] = {"narrowed": int((mix >= 0.99).sum()), "whole": int((mix <= 0.01).sum()),
                        "ramp": int(((mix > 0.01) & (mix < 0.99)).sum()), "smoothed": int(lband.sum()),
                        "loops": nloops}
        matched = m["matched"] | mx["matched"]
        flipped = (m["matched"] & (m["angle"] > na)) | (mx["matched"] & (mx["angle"] > na))
        sband = sband | sbx
    row["matched_flipped"] = int(flipped.sum())

    active = p < 1.0 - EPS
    kept_full = active & (allow == 0)
    active &= allow > 0
    if mask is not None:
        active &= W0[:, t.vertex_groups[mask].index] > 0.5
    region = matched | active
    lost = [n for j, n in enumerate(src["names"]) if n not in arm.data.bones and (T[region, j] > EPS).any()]
    if lost:
        raise WeightTransferError("%s: transferred weight lands on %s, which %s lacks, so the export "
                                  "would drop it; add the bone to %s or narrow the source away from it"
                                  % (t.name, ", ".join(lost), arm.name, arm.name))

    T, capped = limit(T, allow)

    s = T.sum(1)
    empty = np.flatnonzero(active & (s <= 0))
    if len(empty):
        raise WeightTransferError("%s: %d written vertices carry no transferred body weight (first %s); "
                                  "the source there is unweighted or excluded"
                                  % (t.name, len(empty), empty[:5].tolist()))
    new = np.where(s[:, None] > 0, T / np.where(s > 0, s, 1.0)[:, None], 0.0) * (1.0 - p)[:, None]

    loops = boundary_loops(t.data)
    edge = np.zeros(len(TV), bool)
    if loops:
        edge[np.concatenate(loops)] = True
        edge = (A @ edge.astype(np.float64)) > 0
    row.update({
        "matched": int(matched.sum()), "inpainted": int((~matched).sum()),
        "matched_pct_boundary_band": round(100.0 * matched[edge].mean(), 1) if edge.any() else None,
        "matched_pct_interior": round(100.0 * matched[~edge].mean(), 1) if (~edge).any() else None,
        "unmatched_parts": loose_parts(A, TV, matched),
        "smoothed": int(sband.sum()) if knobs["smooth"] else None,
        "written": int(active.sum()), "kept_garment": int((p >= 1.0 - EPS).sum()),
        "kept_full": int(kept_full.sum()), "capped": int((capped & active).sum()),
    })
    return {"target": t, "armature": arm, "gnames": gnames, "W0": W0, "active": active,
            "new": new, "matched": matched, "row": row}


def _write(plan, src):
    """Step 7 and the step-8 asserts for one target."""
    t, arm, gnames, W0, active, new = (plan[k] for k in ("target", "armature", "gnames", "W0", "active", "new"))
    names = src["names"]
    col = {n: j for j, n in enumerate(names)}
    carried = {names[j] for j in range(len(names)) if (new[active, j] > 0).any()}
    idx = np.flatnonzero(active)
    created = []
    for n in sorted({g for g in gnames if g in src["deform"]} | carried):
        w = new[:, col[n]] if n in col else np.zeros(len(new))
        g = t.vertex_groups.get(n)
        if g is None:
            g = t.vertex_groups.new(name=n)
            created.append(n)
        g.remove(idx[w[idx] <= 0].tolist())
        for i in idx[w[idx] > 0]:
            g.add([int(i)], float(w[i]), 'REPLACE')

    _, W1 = group_matrix(t)
    old = np.zeros_like(W1)
    old[:, :W0.shape[1]] = W0
    keep = [i for i, n in enumerate(gnames) if n not in src["deform"]]
    changed = [gnames[i] for i in keep if not np.array_equal(W1[:, i], old[:, i])]
    if changed:
        raise WeightTransferError("invariant broken (nothing saved): %s: garment or non-bone groups "
                                  "changed: %s" % (t.name, ", ".join(changed)))
    if not np.array_equal(W1[~active], old[~active]):
        raise WeightTransferError("invariant broken (nothing saved): %s: a vertex outside the write "
                                  "changed" % t.name)
    bone_cols = [i for i, g in enumerate(t.vertex_groups) if g.name in arm.data.bones]
    over = int(((W1[active][:, bone_cols] > 0).sum(1) > UNITY_BONES).sum())
    if over:
        raise WeightTransferError("invariant broken (nothing saved): %s: %d written vertices exceed %d "
                                  "bone groups" % (t.name, over, UNITY_BONES))
    d = np.abs(W1 - old).max(1)
    contact = plan["matched"] & active
    plan["row"].update({
        "touched": int((d > 1e-6).sum()), "groups_created": created,
        "contact_max_dw": round(float(d[contact].max()), 4) if contact.any() else 0.0,
        "contact_mean_dw": round(float(d[contact].mean()), 4) if contact.any() else 0.0,
        "matched_mask": plan["matched"],
    })


# --- review layer --------------------------------------------------------------------------------

def paint_matched(ob, matched):
    """A corner colour layer ``VIZ_LAYER`` on ``ob``: white matched, magenta inpainted, set
    active for ``render_mesh --shading vertexcolor``."""
    me = ob.data
    a = me.color_attributes.get(VIZ_LAYER) or me.color_attributes.new(VIZ_LAYER, 'BYTE_COLOR', 'CORNER')
    lv = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", lv)
    c = np.ones((len(me.loops), 4), np.float32)
    c[~np.asarray(matched, bool)[lv]] = (234 / 255, 0.0, 1.0, 1.0)
    a.data.foreach_set("color", c.ravel())
    i = me.color_attributes.find(VIZ_LAYER)
    me.color_attributes.render_color_index = i
    me.color_attributes.active_color_index = i


def clear_matched(ob):
    a = ob.data.color_attributes.get(VIZ_LAYER)
    if a is not None:
        ob.data.color_attributes.remove(a)
