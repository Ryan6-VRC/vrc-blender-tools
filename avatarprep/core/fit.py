"""Measure how skinned garments fit their body under motion, and push a garment off the body where motion pulls skin through it.

Everything is read from ``bpy`` data once (``load``) and then posed analytically, so no
armature is copied, posed or switched between rest and pose:

- **Rest.** Body and garments are evaluated from in-memory copies with their Armature
  modifiers off and the ``--shape`` values on (``weight_transfer.source_arrays``); the
  ``--cut-shape`` Deletes drop the vertices they move (``coverage.moved_vertices``).
- **Skinning** is Unity's: each vertex keeps its four largest bone weights over its own
  mesh's deform bones, normalised. A garment bone the body armature lacks moves with its
  nearest ancestor the body has, which is what Modular Avatar's merge does; one with no
  such ancestor stays still.
- **A step** turns one body bone by an angle about a world axis through its head: the
  transform ``T(h) R T(-h)`` for the bone and its descendants, identity elsewhere, so a
  vertex lands at ``p + s (R (p - h) + h - p)`` with ``s`` its weight share on the moved
  bones. Dynamics (PhysBones) are out of scope; that is why ``physbone`` vertices are
  excluded from every metric.
- **The sweep set**, when no bone is named: body deform bones weighting the garment, plus
  bones weighting body vertices within ``NEAR_BODY`` of it at rest, minus the armature's
  root bones and any bone whose largest share on either set is under ``FLOOR_SHARE``.
  Each bone takes the first ``JOINT_TABLE`` row whose glob matches its lowercased name
  (else ``other``); positive angles are flexion, fixed by where a 30 degree test turn
  moves the child's head, so bone roll and mirroring cannot flip it.

Metric meanings are ``LEGEND``, which the doors print; that text is the canon. The body is
cropped to the garment's posed bounds plus ``CROP`` before every BVH build.

Doors: ``cli/report_fit.py``, ``cli/compare_fit.py``, ``cli/push_garment.py``.
"""
import fnmatch
import math
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import coverage
from . import scene_utils
from . import weight_transfer as WT

UNITY_BONES = 4
FLOOR_SHARE = 0.05      # a bone under this largest share on the garment and the near body is not swept
NEAR_BODY = 0.020       # m: body vertices this close to the garment at rest pull their bones into the sweep
CROP = 0.03             # m: the body is cropped to the garment's bounds plus this before a BVH build
TIGHT = 0.005           # m: interior vertices this close to the body at rest are skin-tight
EDGE_NEAR = 0.010       # m: edge vertices this close to the body at rest count for edge poke and slide
PEN_MM = 1.0            # mm: depth that counts as penetration
THROUGH = 0.005         # m: body vertices this close outside a garment triangle count as through
MIN_EDGE = 0.0005       # m: shorter rest edges (and anchor edges) are too short to read a ratio off
STRETCH_FLAG = 1.3
PHYS_SHARE = 0.5
DEFAULT_STEPS = 3
TEST_DEG = 30.0
SEED_THROUGH = 0.005    # m: push_garment's body-vertex reach for the through test

AXIS_WORDS = ("forward", "back", "lateral", "up", "own", "a", "b")
ZONE_WORDS = ("front", "back", "left", "right")

# row, globs (matched against the lowercased bone name), axes: (axis, +max, -max, flexion test)
# The flexion test names where a positive turn carries the child's head.
JOINT_TABLE = (
    ("hip", ("upper*leg*", "thigh*"), (("lateral", 120, 30, "forward"), ("forward", 45, 15, "outward"))),
    ("knee", ("lower*leg*", "shin*", "calf*", "knee*"), (("lateral", 130, 0, "back"),)),
    ("ankle", ("foot*", "ankle*"), (("lateral", 20, 40, "up"),)),
    ("shoulder", ("upper*arm*",), (("forward", 75, 60, "down"), ("up", 90, 30, "forward"))),
    ("clavicle", ("shoulder*", "clavicle*"), (("forward", 20, 20, "down"), ("up", 15, 15, "forward"))),
    ("elbow", ("lower*arm*", "fore*arm*", "elbow*"), (("up", 130, 0, "forward"), ("own", 60, 60, None))),
    ("wrist", ("hand*", "wrist*"), (("a", 45, 45, None), ("b", 45, 45, None))),
    ("spine", ("spine*", "chest*", "upperchest*", "neck*"),
     (("lateral", 25, 15, "forward"), ("forward", 15, 15, None), ("own", 20, 20, None))),
    ("head", ("head",), (("lateral", 30, 30, "forward"), ("forward", 20, 20, None))),
)
OTHER_ROW = ("other", (), (("a", 20, 20, None), ("b", 20, 20, None)))

LEGEND = (
    "classes, fixed at rest, first match wins: physbone = garment-bone share >= 0.5 (moved by dynamics, "
    "so left out of every metric); edge = a boundary-loop vertex or one ring in; skin-tight = within 5 mm "
    "of the body; loose = the rest. Cut vertices (--cut-shape) are in no class",
    "a step turns one body bone and everything under it by the named angle about an axis through its "
    "head (forward, lateral or up from the frame; own = along the bone; a and b = across it), the rest "
    "of the body at rest; positive is flexion. Named Bone:axis:angle; --sweep Bone:axis:0..angle:1 (or "
    "angle..0:1 when negative) measures that step alone",
    "depth = mm inside the body along the nearest body triangle's normal. new-pen = skin-tight vertices "
    "deeper than 1 mm that were not at rest; edge-poke = the same over edge vertices within 10 mm of the "
    "body at rest. Rest penetration is reported beside them and is never counted as new",
    "body-through = body vertices that were beneath a garment triangle at rest and are within 5 mm on "
    "its outer side at the step (outer = facing away from the body at rest; triangles with an edge, "
    "physbone or cut corner excluded): skin poking out between garment vertices, which no vertex "
    "depth sees",
    "stretch = posed / rest length of edges with no physbone or cut end and rest length >= 0.5 mm; "
    "excess = stretch divided by the stretch of the edge between the two ends' body anchors (anchor edge "
    ">= 0.5 mm), so skin stretching under the garment does not count. Both: max, p95, count over 1.3",
    "anchor = a vertex's nearest body point at rest, carried on its triangle. slide = how far (mm) a "
    "vertex moves relative to its anchor, measured in the anchor triangle's own frame, over contact "
    "vertices (skin-tight, plus edge within 10 mm)",
    "self-x = body triangle pairs crossing each other (sharing no vertex) in the cropped body: where it "
    "rises, depth signs near that crease are unreliable. Every count here is context for the operator's "
    "eye, never a pass or a fail",
)


class FitError(ValueError):
    """A refusal the door prints in-grammar; names the offender and the fix."""


# --- small numerics ---------------------------------------------------------------------------

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def top4(W) -> np.ndarray:
    """Each row's ``UNITY_BONES`` largest weights, normalised to sum 1 (zero rows stay zero)."""
    W = np.array(W, np.float64)
    if W.shape[1] > UNITY_BONES:
        drop = np.argsort(-W, axis=1, kind="stable")[:, UNITY_BONES:]
        np.put_along_axis(W, drop, 0.0, axis=1)
    s = W.sum(1, keepdims=True)
    return np.where(s > 0, W / np.where(s > 0, s, 1.0), 0.0)


def rotation(u, deg) -> np.ndarray:
    """3x3 rotation by ``deg`` about unit axis ``u`` (right hand)."""
    u = np.asarray(u, np.float64)
    u = u / np.linalg.norm(u)
    t = math.radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


def pose_points(P, s, h, R):
    """``P`` skinned by a share ``s`` on bones turned by ``R`` about the point ``h``."""
    Q = (P - h) @ R.T + h
    return P + s[:, None] * (Q - P)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def _pct(x, q):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else None


def _nearest(tree, P, maxd=None):
    """``(loc, normal, face, dist)`` arrays for each point; face -1 and dist inf where none."""
    n = len(P)
    loc = np.zeros((n, 3))
    nor = np.zeros((n, 3))
    face = np.full(n, -1, np.int64)
    dist = np.full(n, np.inf)
    for i, co in enumerate(P.tolist()):
        r = tree.find_nearest(co) if maxd is None else tree.find_nearest(co, maxd)
        if r[0] is not None:
            loc[i] = r[0]
            nor[i] = r[1]
            face[i] = r[2]
            dist[i] = r[3]
    return loc, nor, face, dist


def _crop(V, F, lo, hi):
    """``(Vc, Fc, sel)``: triangles of ``F`` with a corner inside the box, compacted; ``sel``
    indexes them in ``F``."""
    inside = ((V >= lo) & (V <= hi)).all(1)
    sel = np.flatnonzero(inside[F].any(1))
    if not len(sel):
        return None, None, sel
    used, inv = np.unique(F[sel], return_inverse=True)
    return V[used], inv.reshape(-1, 3), sel


def _tree(V, F):
    return BVHTree.FromPolygons(V.tolist(), F.tolist())


def _self_crossings(tree, F) -> int:
    pairs = np.array(tree.overlap(tree), np.int64).reshape(-1, 2)
    if not len(pairs):
        return 0
    pairs = np.sort(pairs, 1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    if not len(pairs):
        return 0
    shared = (F[pairs[:, 0]][:, :, None] == F[pairs[:, 1]][:, None, :]).any((1, 2))
    return int((~shared).sum())


# --- reading the scene ---------------------------------------------------------------------------

def _cut_masks(meshes, cut_shapes, threshold) -> Dict[str, np.ndarray]:
    """Per mesh name, the vertices a ``--cut-shape`` Delete removes. A bare key lands on every
    listed mesh carrying it and is refused when none does."""
    out = {m.name: np.zeros(len(m.data.vertices), bool) for m in meshes}
    for key in cut_shapes:
        hit = False
        for m in meshes:
            keys = m.data.shape_keys
            if keys is None or keys.key_blocks.get(key) is None:
                continue
            hit = True
            moved, _ = coverage.moved_vertices(m.data, [key], threshold)
            out[m.name] |= np.asarray(moved, bool)
        if not hit:
            raise FitError("--cut-shape %s is on none of %s" % (key, ", ".join(m.name for m in meshes)))
    return out


def _bones(arm, deform_names) -> Dict[str, Dict]:
    """Rest data of every bone of ``arm`` in world space, with its deform descendants."""
    M = np.array(arm.matrix_world, np.float64)
    col = {n: j for j, n in enumerate(deform_names)}

    def w(v):
        return (M @ np.r_[np.array(v, np.float64), 1.0])[:3]

    out = {}
    for b in arm.data.bones:
        kids = [c for c in b.children if c.use_deform] or list(b.children)
        desc = [b.name] + [c.name for c in b.children_recursive]
        out[b.name] = {"head": w(b.head_local), "tail": w(b.tail_local),
                       "child": w(kids[0].head_local) if kids else w(b.tail_local),
                       "parent": b.parent.name if b.parent else None,
                       "cols": np.array([col[n] for n in desc if n in col], np.int64)}
    return out


def _edges(me):
    E = np.empty((len(me.edges), 2), np.int64)
    me.edges.foreach_get("vertices", E.ravel())
    le = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("edge_index", le)
    count = np.bincount(le, minlength=len(me.edges))
    boundary = np.zeros(len(me.vertices), bool)
    boundary[E[count == 1].ravel()] = True
    ring = boundary.copy()
    ring[E[boundary[E].any(1)].ravel()] = True
    return E, boundary, ring


def _dominant(names, own, arm):
    """Per vertex, the garment-armature bone carrying its largest weight, and a map from each
    bone name to the set of bones at or under it."""
    dom = np.full(len(own), None, dtype=object)
    if len(names):
        dom[:] = np.array(names, dtype=object)[np.argmax(own, 1)]
        dom[own.sum(1) <= 0] = None
    under = {}
    for b in arm.data.bones:
        under[b.name] = {b.name} | {c.name for c in b.children_recursive}
    return dom, under


def _garment(ob, shapes, body_names, cut):
    """One garment's rest arrays, weights mapped onto the body's deform bones, and classes."""
    g = WT.source_arrays(ob, shapes)
    arm = g["armature"]
    body_set = set(body_names)
    idx = {n: j for j, n in enumerate(body_names)}
    names = g["names"]
    garment_only = [j for j, n in enumerate(names) if n not in body_set]
    mapping = np.zeros((len(names), len(body_names)))
    for j, n in enumerate(names):
        b = arm.data.bones.get(n)
        while b is not None and b.name not in body_set:
            b = b.parent
        if b is not None:
            mapping[j, idx[b.name]] = 1.0
    own = top4(g["W"])
    E, boundary, ring = _edges(ob.data)
    F = g["F"]
    keepF = F[~cut[F].any(1)]
    dom, under = _dominant(names, own, arm)
    return {"name": ob.name, "object": ob, "V": g["V"].astype(np.float64), "N": g["N"].astype(np.float64),
            "F": keepF, "E": E[~cut[E].any(1)], "boundary": boundary, "ring": ring, "cut": cut,
            "names": names, "garment_only": garment_only, "mapping": mapping, "own": own,
            "W": own @ mapping, "phys": own[:, garment_only].sum(1) if garment_only else np.zeros(len(own)),
            "armature": arm.name, "dominant": dom, "under": under, "groups": {}}


def load(source, targets: Sequence, *, shapes=(), cut_shapes=(), cut_threshold=0.01,
         reference_bone="Hips", forward_bone=None, groups=()) -> Dict:
    """Read everything the sweep needs from ``bpy`` data, once. ``shapes``: ``(mesh_or_None,
    key, value)`` triples as ``weight_transfer.resolve_shapes`` takes them. ``groups``: garment
    vertex-group names a ``group:`` region reads. Returns plain arrays; nothing in the scene
    changes. Raises ``FitError``."""
    targets = list(targets)
    if source is None or source.type != 'MESH':
        raise FitError("the source is not a mesh object")
    if not targets:
        raise FitError("no targets named")
    for t in targets:
        if t is source or t.data is source.data:
            raise FitError("%s is the source; measure a garment against it" % t.name)
    try:
        per_mesh = WT.resolve_shapes(shapes, [source] + targets)
        src = WT.source_arrays(source, per_mesh[source.name])
        frame = WT.body_frame(src["armature"], reference_bone, forward_bone)
        for t in targets:
            WT.armature_of(t)
    except WT.WeightTransferError as e:
        raise FitError(str(e))
    cut = _cut_masks([source] + targets, cut_shapes, cut_threshold)
    names = src["names"]
    arm = src["armature"]
    bcut = cut[source.name]
    body = {"name": source.name, "armature": arm.name, "V": src["V"].astype(np.float64),
            "N": src["N"], "F_all": src["F"], "F": src["F"][~bcut[src["F"]].any(1)], "cut": bcut,
            "Wraw": src["W"], "W": top4(src["W"]), "names": names, "bones": _bones(arm, names),
            "roots": {b.name for b in arm.data.bones if b.parent is None}}
    if not len(body["F"]):
        raise FitError("--cut-shape removes every triangle of %s" % source.name)
    garments = []
    try:
        for t in targets:
            garments.append(_garment(t, per_mesh[t.name], names, cut[t.name]))
    except WT.WeightTransferError as e:
        raise FitError(str(e))
    missing = [n for n in groups if not any(n in t.vertex_groups for t in targets)]
    if missing:
        raise FitError("group:%s is a vertex group on none of %s" % (",".join(missing),
                                                                     ", ".join(t.name for t in targets)))
    for g, t in zip(garments, targets):
        if groups:
            gnames, GW = WT.group_matrix(t)
            for n in groups:
                if n in gnames:
                    g["groups"][n] = GW[:, gnames.index(n)]
    data = {"body": body, "garments": garments, "frame": frame, "cut_threshold": cut_threshold}
    for g in garments:
        _rest(data, g)
    return data


def _rest(data, g):
    """The rest step for one garment: anchors, rest distances, classes, the near body."""
    body = data["body"]
    V = g["V"]
    live = ~g["cut"]
    lo, hi = V[live].min(0) - CROP, V[live].max(0) + CROP
    Vc, Fc, sel = _crop(body["V"], body["F"], lo, hi)
    if Vc is None:
        raise FitError("%s is farther than %g m from %s everywhere; nothing to measure against"
                       % (g["name"], CROP, body["name"]))
    loc, nor, face, dist = _nearest(_tree(Vc, Fc), V)
    anc = sel[np.maximum(face, 0)]
    T = body["F"][anc]
    BV = body["V"]
    bary = WT.closest_on_triangles(V, BV[T[:, 0]], BV[T[:, 1]], BV[T[:, 2]])
    g["anchor"], g["bary"] = anc, bary
    g["rsd"] = ((V - loc) * nor).sum(1)
    g["rd"] = dist
    g["bn0"] = nor
    tn = np.cross(V[g["F"][:, 1]] - V[g["F"][:, 0]], V[g["F"][:, 2]] - V[g["F"][:, 0]])
    g["gsign"] = np.where((tn * nor[g["F"]].mean(1)).sum(1) >= 0, 1.0, -1.0)
    g["out0"] = np.nan_to_num(_unit(tn)) * g["gsign"][:, None]
    phys = live & (g["phys"] >= PHYS_SHARE)
    edge = live & ~phys & g["ring"]
    tight = live & ~phys & ~edge & (dist <= TIGHT)
    g["cls"] = {"physbone": phys, "edge": edge, "skin-tight": tight, "loose": live & ~phys & ~edge & ~tight}
    g["contact"] = tight | (edge & (dist <= EDGE_NEAR))
    E = g["E"]
    g["L0"] = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    g["AP0"] = np.einsum("nk,nkd->nd", bary, BV[T])
    g["AL0"] = np.linalg.norm(g["AP0"][E[:, 0]] - g["AP0"][E[:, 1]], axis=1)
    g["eok"] = ~phys[E].any(1) & (g["L0"] >= MIN_EDGE)
    F = g["F"]
    g["tok"] = ~(phys | g["ring"])[F].any(1)
    g["frame0"] = _tri_frames(BV, T)
    g["off0"] = _local(g["frame0"], V - g["AP0"])


def _tri_frames(BV, T):
    A, B, C = BV[T[:, 0]], BV[T[:, 1]], BV[T[:, 2]]
    e = _unit(B - A)
    n = _unit(np.cross(B - A, C - A))
    return e, n, np.cross(n, e)


def _local(frame, d):
    e, n, t = frame
    return np.stack([(d * e).sum(1), (d * n).sum(1), (d * t).sum(1)], 1)


def extents(data, g) -> Dict[str, List[float]]:
    """The garment's rest extent along each frame axis, metres from the reference bone's head."""
    f = data["frame"]
    d = g["V"][~g["cut"]] - f["origin"]
    return {k: [round(float((d @ f[k]).min()), 4), round(float((d @ f[k]).max()), 4)]
            for k in ("forward", "lateral", "up")}


# --- regions ------------------------------------------------------------------------------------

def parse_region(text) -> List[Tuple[str, str]]:
    """``[(kind, value), ...]`` from ``KIND:VALUE[+KIND:VALUE]...``; raises ``ValueError`` naming
    the form. The terms intersect."""
    terms = []
    for part in text.split("+"):
        kind, sep, value = part.partition(":")
        kind = kind.strip().lower()
        if not sep or not value or kind not in ("group", "bone", "zone"):
            raise ValueError("--region wants group:NAME, bone:NAME or zone:NAME (terms joined by + "
                             "intersect), got %r" % text)
        if kind == "zone":
            word, _, bone = value.partition(":")
            word = word.lower()
            if word.strip("+-") in ("x", "y", "z"):
                raise ValueError("zone:%s is a world axis, which reads differently on a rig facing -Y than "
                                 "+Y; zones are front, back, left, right, above:BONE, below:BONE" % value)
            if not ((word in ZONE_WORDS and not bone) or (word in ("above", "below") and bone)):
                raise ValueError("zone:%s is not one of front, back, left, right, above:BONE, below:BONE"
                                 % value)
        terms.append((kind, value))
    return terms


def region_mask(data, g, terms) -> np.ndarray:
    """Vertices of ``g`` in the region: ``group:`` weight over 0.5; ``bone:`` largest influence
    on the named garment-armature bone or a bone under it; ``zone:`` rest position on that side
    of the reference bone's head (front/back along forward, left/right along lateral) or above
    or below the named body bone's head. Cut vertices are never in a region."""
    f = data["frame"]
    d = g["V"] - f["origin"]
    m = ~g["cut"]
    for kind, value in terms:
        if kind == "group":
            w = g["groups"].get(value)
            m = m & (w > 0.5) if w is not None else np.zeros_like(m)
        elif kind == "bone":
            under = g["under"].get(value)
            m = m & np.array([x in under for x in g["dominant"]], bool) if under else np.zeros_like(m)
        else:
            word, _, bone = value.partition(":")
            word = word.lower()
            if word in ZONE_WORDS:
                axis, sign = {"front": ("forward", 1), "back": ("forward", -1),
                              "left": ("lateral", 1), "right": ("lateral", -1)}[word]
                m = m & (sign * (d @ f[axis]) > 0)
            else:
                b = data["body"]["bones"].get(bone)
                if b is None:
                    raise FitError("zone:%s names no bone of %s" % (value, data["body"]["armature"]))
                up = (g["V"] - b["head"]) @ f["up"]
                m = m & ((up > 0) if word == "above" else (up < 0))
    return m


def region_groups(specs) -> List[str]:
    """The vertex-group names the ``group:`` terms of ``specs`` read, for ``load``."""
    return sorted({v for spec in specs for k, v in parse_region(spec) if k == "group"})


def resolve_regions(data, specs: Sequence[str]) -> Dict[str, List[np.ndarray]]:
    """``{spec: [mask per garment]}``; a bone: term naming no garment armature's bone, or a region
    empty on every garment, refuses."""
    out = {}
    for spec in specs:
        terms = parse_region(spec)
        for kind, value in terms:
            if kind == "bone" and not any(value in g["under"] for g in data["garments"]):
                raise FitError("--region %s: %s is a bone of none of the garments' armatures (%s)"
                               % (spec, value, ", ".join(sorted({g["armature"] for g in data["garments"]}))))
        masks = [region_mask(data, g, terms) for g in data["garments"]]
        if not any(m.any() for m in masks):
            raise FitError("--region %s holds no vertex of %s" % (spec, ", ".join(g["name"] for g in data["garments"])))
        out[spec] = masks
    return out


# --- the sweep plan ---------------------------------------------------------------------------------

def _near_body(data, g, focus) -> np.ndarray:
    """Body vertices within ``NEAR_BODY`` of the garment triangles touching ``focus`` at rest."""
    body = data["body"]
    F = g["F"][focus[g["F"]].any(1)]
    near = np.zeros(len(body["V"]), bool)
    if not len(F):
        return near
    V = g["V"]
    used = np.unique(F)
    lo, hi = V[used].min(0) - NEAR_BODY, V[used].max(0) + NEAR_BODY
    cand = np.flatnonzero(((body["V"] >= lo) & (body["V"] <= hi)).all(1) & ~body["cut"])
    _, _, face, _ = _nearest(_tree(V, F), body["V"][cand], NEAR_BODY)
    near[cand[face >= 0]] = True
    return near


def shares(data, focus_masks=None) -> Dict[str, Tuple[float, float]]:
    """``{bone: (garment_share, near_body_share)}``: the largest skinning share each body deform
    bone has on the focused garment vertices and on the body vertices near them."""
    body = data["body"]
    gmax = np.zeros(len(body["names"]))
    bmax = np.zeros(len(body["names"]))
    for i, g in enumerate(data["garments"]):
        focus = (~g["cut"]) if focus_masks is None else focus_masks[i]
        g["near"] = np.zeros(len(body["V"]), bool)
        if not focus.any():
            continue
        gmax = np.maximum(gmax, g["W"][focus].max(0))
        g["near"] = _near_body(data, g, focus)
        if g["near"].any():
            bmax = np.maximum(bmax, body["W"][g["near"]].max(0))
    return {n: (float(gmax[j]), float(bmax[j])) for j, n in enumerate(body["names"])}


def joint_row(bone):
    low = bone.lower()
    for row in JOINT_TABLE:
        if any(fnmatch.fnmatchcase(low, glob) for glob in row[1]):
            return row
    return OTHER_ROW


def _axis_vector(data, bone, axis):
    f = data["frame"]
    b = data["body"]["bones"][bone]
    d = _unit(b["tail"] - b["head"])
    if axis in ("forward", "lateral", "up"):
        return f[axis]
    if axis == "back":
        return -f["forward"]
    if axis == "own":
        return d
    a = np.cross(d, f["up"])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(d, f["forward"])
    a = _unit(a)
    return a if axis == "a" else _unit(np.cross(d, a))


def _side(data, bone):
    f = data["frame"]
    x = float((data["body"]["bones"][bone]["head"] - f["origin"]) @ f["lateral"])
    return (1 if x >= 0 else -1), ("left" if x > 1e-3 else "right" if x < -1e-3 else "mid")


def _flexion_sign(data, bone, u, test):
    """+1 or -1 so a positive turn about ``u`` moves the child's head toward ``test``."""
    if test is None:
        return 1
    f = data["frame"]
    b = data["body"]["bones"][bone]
    side, _ = _side(data, bone)
    vec = {"forward": f["forward"], "back": -f["forward"], "up": f["up"], "down": -f["up"],
           "outward": side * f["lateral"]}[test]
    moved = (b["child"] - b["head"]) @ rotation(u, TEST_DEG).T + b["head"]
    return -1 if float((moved - b["child"]) @ vec) < -1e-9 else 1


def angles(lo, hi, steps) -> List[float]:
    out = [hi * k / steps for k in range(1, steps + 1)] if hi > 0 else []
    return out + ([lo * k / steps for k in range(1, steps + 1)] if lo < 0 else [])


def step_name(bone, axis, angle):
    return "%s:%s:%+g" % (bone, axis, round(angle, 2))


def plan(data, sweeps=None, focus_masks=None, steps=DEFAULT_STEPS, bones=None) -> Dict:
    """The bones to sweep, the row each takes, and the flat step list.

    ``sweeps``: ``[(bone, axis_or_None, (lo, hi)_or_None, steps_or_None), ...]`` as the door
    parsed them; ``None`` derives the sweep set (see the module docstring) over ``focus_masks``
    (per garment; ``None`` is every vertex). ``bones`` replaces the derived set with a given one
    (``compare_fit`` sweeps every input over the union of their derived sets)."""
    body = data["body"]
    sh = shares(data, focus_masks)
    entries = []
    if sweeps is None and bones is not None:
        entries = [(n, None, None, None, False) for n in bones]
    elif sweeps is None:
        for n in body["names"]:
            gs, bs = sh[n]
            if n in body["roots"] or max(gs, bs) < FLOOR_SHARE:
                continue
            entries.append((n, None, None, None, False))
        if not entries:
            raise FitError("no body bone other than the root weights the garment or the body near it at "
                           "a share of %g or more, so nothing would move; name one with --sweep" % FLOOR_SHARE)
    else:
        near = np.zeros(len(body["V"]), bool)
        for g in data["garments"]:
            near |= g["near"]
        for bone, axis, rng, n in sweeps:
            if bone not in body["bones"] or not len(body["bones"][bone]["cols"]):
                raise FitError("--sweep %s is not a deform bone of %s (or has none under it)"
                               % (bone, body["armature"]))
            if not (body["W"][near][:, body["bones"][bone]["cols"]].sum(1) > 0).any():
                raise FitError("--sweep %s moves no body vertex within %g m of the garment; sweep a bone "
                               "whose skin is under it" % (bone, NEAR_BODY))
            entries.append((bone, axis, rng, n, True))

    bones, flat = [], []
    for bone, axis, rng, n, explicit in entries:
        row = joint_row(bone)
        side, side_word = _side(data, bone)
        axes = []
        table = {a[0]: a for a in row[2]}
        if axis is None:
            use = list(row[2])
        elif axis in table:
            use = [table[axis]]
        else:
            if rng is None:
                raise FitError("--sweep %s:%s: the %s row has no %s axis (it has %s); give MIN..MAX"
                               % (bone, axis, row[0], axis, ", ".join(table)))
            use = [(axis, 0, 0, None)]
        for ax, pos, neg, test in use:
            u = _axis_vector(data, bone, ax)
            sign = _flexion_sign(data, bone, u, test)
            lo, hi = (-neg, pos) if rng is None else rng
            k = n or steps
            axes.append({"axis": ax, "lo": lo, "hi": hi, "steps": k, "flexion": test})
            for a in angles(lo, hi, k):
                flat.append({"bone": bone, "axis": ax, "angle": a, "name": step_name(bone, ax, a),
                             "u": sign * u, "h": body["bones"][bone]["head"],
                             "cols": body["bones"][bone]["cols"]})
        gs, bs = sh.get(bone, (0.0, 0.0))
        bones.append({"bone": bone, "row": row[0], "side": side_word, "garment_share": round(gs, 3),
                      "body_share": round(bs, 3), "explicit": explicit, "axes": axes})
    return {"bones": bones, "steps": flat}


# --- one step ------------------------------------------------------------------------------------

def posed(data, g, step):
    """``(body V, garment V)`` world positions at ``step`` (``None`` is rest)."""
    body = data["body"]
    if step is None:
        return body["V"], g["V"]
    R = rotation(step["u"], step["angle"])
    c = step["cols"]
    return (pose_points(body["V"], body["W"][:, c].sum(1), step["h"], R),
            pose_points(g["V"], g["W"][:, c].sum(1), step["h"], R))


def _depth_and_through(data, g, BV, GV, query, through_tris, through_reach, crossing=False):
    """Depth (mm, positive inside) for the ``query`` vertices; body vertices within
    ``through_reach`` of ``through_tris``: their triangle index and outward height (m), with
    ``crossing`` only those that were beneath that triangle's plane at rest (so another body
    part that was already outside, a neighbouring limb, is not counted); the self-crossing
    count of the cropped body."""
    body = data["body"]
    live = ~g["cut"]
    lo, hi = GV[live].min(0) - CROP, GV[live].max(0) + CROP
    Vc, Fc, _ = _crop(BV, body["F"], lo, hi)
    depth = np.full(len(GV), -np.inf)
    selfx = 0
    if Vc is not None:
        tree = _tree(Vc, Fc)
        q = np.flatnonzero(query)
        loc, nor, face, _ = _nearest(tree, GV[q])
        ok = face >= 0
        depth[q[ok]] = -((GV[q[ok]] - loc[ok]) * nor[ok]).sum(1) * 1000.0
        selfx = _self_crossings(tree, Fc)
    hit_tri = np.zeros(0, np.int64)
    height = np.zeros(0)
    if len(through_tris):
        lo, hi = GV[live].min(0) - through_reach, GV[live].max(0) + through_reach
        cand = np.flatnonzero(((BV >= lo) & (BV <= hi)).all(1) & ~body["cut"])
        if len(cand):
            loc, nor, face, _ = _nearest(_tree(GV, g["F"][through_tris]), BV[cand], through_reach)
            ok = face >= 0
            tri = through_tris[face[ok]]
            h = ((BV[cand[ok]] - loc[ok]) * nor[ok]).sum(1) * g["gsign"][tri]
            if crossing:
                was = ((body["V"][cand[ok]] - g["V"][g["F"][tri, 0]]) * g["out0"][tri]).sum(1) < 0
                tri, h = tri[was], h[was]
            hit_tri, height = tri, h
    return depth, hit_tri, height, selfx


def _stats(values, flag=None):
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    out = {"max": round(float(values.max()), 3), "p95": round(_pct(values, 95), 3), "n": int(len(values))}
    if flag is not None:
        out["over"] = int((values > flag).sum())
    return out


def measure_step(data, g, step, regions) -> Dict:
    """Every metric of ``LEGEND`` for one garment at one step, per region (``regions``:
    ``{label: vertex mask}``)."""
    BV, GV = posed(data, g, step)
    tris = np.flatnonzero(g["tok"])
    depth, hit_tri, height, selfx = _depth_and_through(data, g, BV, GV, g["contact"], tris, THROUGH,
                                                       crossing=True)
    rdep = -g["rsd"] * 1000.0
    E = g["E"]
    ratio = np.linalg.norm(GV[E[:, 0]] - GV[E[:, 1]], axis=1) / np.maximum(g["L0"], 1e-12)
    T = data["body"]["F"][g["anchor"]]
    AP = np.einsum("nk,nkd->nd", g["bary"], BV[T])
    aratio = np.linalg.norm(AP[E[:, 0]] - AP[E[:, 1]], axis=1) / np.maximum(g["AL0"], 1e-12)
    excess = ratio / np.maximum(aratio, 1e-6)
    off = _local(_tri_frames(BV, T), GV - AP)
    slide = np.linalg.norm(off - g["off0"], axis=1) * 1000.0
    out = {"self_x": selfx, "regions": {}}
    tight, edge = g["cls"]["skin-tight"], g["cls"]["edge"] & (g["rd"] <= EDGE_NEAR)
    newly = (depth > PEN_MM) & (rdep <= PEN_MM)
    through = height > 0
    for label, R in regions.items():
        pen = tight & R & newly
        poke = edge & R & newly
        th = through & R[g["F"][hit_tri]].any(1) if len(hit_tri) else np.zeros(0, bool)
        er = g["eok"] & R[E].any(1)
        ex = er & (g["AL0"] >= MIN_EDGE)
        sl = g["contact"] & R
        out["regions"][label] = {
            "new_pen": {"count": int(pen.sum()), "max_mm": round(float(depth[pen].max()), 2) if pen.any() else 0.0},
            "edge_poke": {"count": int(poke.sum()), "max_mm": round(float(depth[poke].max()), 2) if poke.any() else 0.0},
            "body_through": {"count": int(th.sum()), "max_mm": round(float(height[th].max()) * 1000.0, 2) if th.any() else 0.0},
            "stretch": _stats(ratio[er], STRETCH_FLAG),
            "excess": _stats(excess[ex], STRETCH_FLAG),
            "slide": _stats(slide[sl]),
        }
    return out


def rest_context(data, g, regions) -> Dict:
    """Rest-state counts per region: skin-tight and near-edge vertices already deeper than 1 mm
    (the seat's, a push's or a shape's to fix, never the weights'), and the body's
    self-crossings."""
    BV, GV = posed(data, g, None)
    _, _, _, selfx = _depth_and_through(data, g, BV, GV, np.zeros(len(GV), bool), np.zeros(0, np.int64), 0)
    rdep = -g["rsd"] * 1000.0
    edge = g["cls"]["edge"] & (g["rd"] <= EDGE_NEAR)
    out = {"self_x": selfx, "regions": {}}
    for label, R in regions.items():
        out["regions"][label] = {"pen": int((g["cls"]["skin-tight"] & R & (rdep > PEN_MM)).sum()),
                                 "edge_pen": int((edge & R & (rdep > PEN_MM)).sum())}
    return out


WORST_KEYS = (("new_pen", "count"), ("edge_poke", "count"), ("body_through", "count"),
              ("stretch", "max"), ("excess", "max"), ("slide", "p95"))


def worst(steps_out, gname, label) -> Dict:
    """Per metric, the step with the largest value for one garment and region, with that
    step's body self-crossing count."""
    out = {}
    for metric, key in WORST_KEYS:
        best = None
        for s in steps_out:
            v = s["garments"][gname]["regions"][label][metric]
            if v is None:
                continue
            tie = v.get("max_mm", v.get("max", 0.0))
            if best is None or (v[key], tie) > (best[1][key], best[2]):
                best = (s["name"], v, tie, s["garments"][gname]["self_x"])
        out[metric] = None if best is None else {"step": best[0], **best[1], "self_x": best[3]}
    sx = max(steps_out, key=lambda s: s["garments"][gname]["self_x"], default=None)
    out["self_x"] = None if sx is None else {"step": sx["name"], "count": sx["garments"][gname]["self_x"]}
    return out


def region_sets(data, region_masks) -> List[Dict[str, np.ndarray]]:
    """Per garment ``{label: mask}``: ``all`` first, then each ``--region``."""
    out = []
    for i, g in enumerate(data["garments"]):
        d = {"all": ~g["cut"]}
        for spec, masks in (region_masks or {}).items():
            d[spec] = masks[i]
        out.append(d)
    return out


def sweep(data, the_plan, regions) -> Dict:
    """Run every step of ``the_plan`` over every garment. Returns ``rest``, ``steps`` (name and
    per-garment metrics) and ``worst`` per garment and region, plus the wall time."""
    t0 = time.perf_counter()
    rest = {g["name"]: rest_context(data, g, regions[i]) for i, g in enumerate(data["garments"])}
    steps_out = []
    for st in the_plan["steps"]:
        steps_out.append({"name": st["name"], "bone": st["bone"], "axis": st["axis"], "angle": st["angle"],
                          "garments": {g["name"]: measure_step(data, g, st, regions[i])
                                       for i, g in enumerate(data["garments"])}})
    w = {g["name"]: {label: worst(steps_out, g["name"], label) for label in regions[i]}
         for i, g in enumerate(data["garments"])}
    return {"rest": rest, "steps": steps_out, "worst": w, "seconds": round(time.perf_counter() - t0, 2)}


def estimate(data, the_plan, regions) -> float:
    """Seconds the sweep would take, from one timed step (the plan's first)."""
    if not the_plan["steps"]:
        return 0.0
    t0 = time.perf_counter()
    st = the_plan["steps"][0]
    for i, g in enumerate(data["garments"]):
        measure_step(data, g, st, regions[i])
    one = time.perf_counter() - t0
    return one * (len(the_plan["steps"]) + 1)


# --- the simulated transfer ----------------------------------------------------------------------

def simulated(data) -> Dict:
    """A copy of ``data`` whose garments carry the body's weights interpolated at each vertex's
    nearest body point, garment bones keeping their share as ``transfer_weights`` keeps them
    (the body part scaled to ``1 - p``), then Unity's top four. Separates a weights defect from
    a geometry one: a posed defect this removes is the weights'."""
    body = data["body"]
    out = dict(data)
    out["garments"] = []
    for g in data["garments"]:
        m = WT.match(body["V"].astype(np.float32), body["F"], body["N"], body["Wraw"],
                     g["V"].astype(np.float32), g["N"].astype(np.float32), max_distance=np.inf,
                     normal_angle=180.0, flip=True)
        s = m["weights"].sum(1, keepdims=True)
        interp = np.where(s > 0, m["weights"] / np.where(s > 0, s, 1.0), 0.0)
        go = g["garment_only"]
        p = g["own"][:, go].sum(1) if go else np.zeros(len(g["V"]))
        own = np.concatenate([g["own"][:, go], interp * (1.0 - p)[:, None]], 1)
        mapping = np.concatenate([g["mapping"][go], np.eye(len(body["names"]))], 0)
        own = top4(own)
        h = dict(g)
        h["own"] = own
        h["W"] = own @ mapping
        h["phys"] = own[:, :len(go)].sum(1)
        out["garments"].append(h)
    return out


# --- push ------------------------------------------------------------------------------------------

def _seeds(data, g, step, near, focus):
    """Seed vertices at one step: posed vertices inside the body or within ``near`` of it, and
    the corners of garment triangles a body vertex within ``SEED_THROUGH`` comes through (or
    reaches within ``near`` of from beneath); physbone and cut vertices never seed."""
    BV, GV = posed(data, g, step)
    pop = focus & ~g["cls"]["physbone"] & ~g["cut"]
    tris = np.arange(len(g["F"]))
    depth, hit_tri, height, _ = _depth_and_through(data, g, BV, GV, pop, tris, SEED_THROUGH)
    hit = pop & (depth > -near * 1000.0)
    if len(hit_tri):
        corners = np.zeros(len(GV), bool)
        corners[g["F"][hit_tri[height > -near]].ravel()] = True
        hit |= corners & pop
    return hit


def plan_push(data, gi, the_plan, *, amount, near=0.0005, falloff=0.025, rim_hold=0.0,
              allow_static=False, focus=None) -> Dict:
    """Seeds and the per-vertex push for one garment; nothing is written. Dynamic seeds are the
    seeds of any step minus the rest seeds; none refuses unless ``allow_static``, which pushes
    the rest seeds instead. Every vertex within ``falloff`` (rest, straight line) of a seed moves
    along its rest normal, turned to face away from its nearest body triangle, by
    ``amount * smoothstep(1 - d / falloff)``, times ``smoothstep(e / rim_hold)`` when set (``e``
    its distance to the nearest boundary vertex). Returns the object-space delta and counts."""
    g = data["garments"][gi]
    focus = (~g["cut"]) if focus is None else focus
    rest = _seeds(data, g, None, near, focus)
    posed_any = np.zeros(len(g["V"]), bool)
    per = []
    for st in the_plan["steps"]:
        s = _seeds(data, g, st, near, focus)
        per.append({"step": st["name"], "seeds": int(s.sum()), "dynamic": int((s & ~rest).sum())})
        posed_any |= s
    dynamic = posed_any & ~rest
    if dynamic.any():
        seeds, kind = dynamic, "dynamic"
    elif allow_static and rest.any():
        seeds, kind = rest, "static"
    elif allow_static:
        raise FitError("%s: no vertex is in contact with %s at rest or at any step, so there is nothing to "
                       "push" % (g["name"], data["body"]["name"]))
    else:
        raise FitError("%s: every contact the sweep finds (%d vertices) is already there at rest, so pushing "
                       "would only move static contact, which is the seat's or a shape's to fix; pass "
                       "--allow-static to push it anyway" % (g["name"], int(rest.sum())))
    V = g["V"]
    kd = KDTree(int(seeds.sum()))
    for i, co in zip(np.flatnonzero(seeds), V[seeds].tolist()):
        kd.insert(co, int(i))
    kd.balance()
    d = np.array([kd.find(co)[2] for co in V.tolist()])
    f = smoothstep(1.0 - d / falloff)
    if rim_hold > 0 and g["boundary"].any():
        kb = KDTree(int(g["boundary"].sum()))
        for i, co in zip(np.flatnonzero(g["boundary"]), V[g["boundary"]].tolist()):
            kb.insert(co, int(i))
        kb.balance()
        e = np.array([kb.find(co)[2] for co in V.tolist()])
        f = f * smoothstep(e / rim_hold)
    n = _unit(g["N"])
    turned = (n * g["bn0"]).sum(1) < 0
    n[turned] *= -1.0
    n = np.nan_to_num(n)
    dw = n * (amount * f)[:, None]
    M3 = np.array(g["object"].matrix_world.to_3x3(), np.float64)
    moved = f > 0
    return {"delta": dw @ np.linalg.inv(M3).T, "kind": kind, "rest_seeds": int(rest.sum()),
            "posed_seeds": int(posed_any.sum()), "dynamic_seeds": int(dynamic.sum()), "seeds": int(seeds.sum()),
            "moved": int(moved.sum()), "full": int((f >= 0.99).sum()),
            "max_mm": round(float(np.linalg.norm(dw, axis=1).max()) * 1000.0, 4),
            "rim_moved": int((moved & g["boundary"]).sum()), "turned": int((turned & moved).sum()),
            "per_step": per}


def apply_push(ob, delta) -> int:
    """Add the object-space ``delta`` to the mesh and to every shape key, Basis included, so every
    key keeps its offset from Basis. Returns the key count. Raises ``FitError`` if an offset or
    the vertex count changed."""
    me = ob.data
    n = len(me.vertices)
    if delta.shape != (n, 3):
        raise FitError("%s: the push was planned for %d vertices, the mesh has %d" % (ob.name, len(delta), n))
    keys = list(me.shape_keys.key_blocks) if me.shape_keys else []

    def co(src):
        a = np.empty(n * 3)
        src.foreach_get("co", a)
        return a.reshape(-1, 3)

    before = [co(k.data) - co(keys[0].data) for k in keys] if keys else []
    for k in keys:
        k.data.foreach_set("co", (co(k.data) + delta).ravel())
    me.vertices.foreach_set("co", (co(me.vertices) + delta).ravel())
    me.update()
    after = [co(k.data) - co(keys[0].data) for k in keys] if keys else []
    if any(np.abs(a - b).max() > 1e-6 for a, b in zip(before, after)) or len(me.vertices) != n:
        raise FitError("invariant broken (nothing saved): %s: a shape key's offset from Basis changed" % ob.name)
    return len(keys)


# --- renders -----------------------------------------------------------------------------------------

def _view_angles(frame):
    """render_mesh's world-axis angle names for the body's front, back, left and right."""
    def name(v):
        k = int(np.argmax(np.abs(v[:2])))
        return {(0, 1): "right", (0, -1): "left", (1, 1): "front", (1, -1): "back"}[(k, 1 if v[k] > 0 else -1)]
    return [name(frame["forward"]), name(-frame["forward"]), name(frame["lateral"]), name(-frame["lateral"])]


def vertex_stretch(data, g, GV):
    E = g["E"]
    ratio = np.linalg.norm(GV[E[:, 0]] - GV[E[:, 1]], axis=1) / np.maximum(g["L0"], 1e-12)
    vm = np.ones(len(GV))
    r = np.where(g["eok"], ratio, 1.0)
    np.maximum.at(vm, E[:, 0], r)
    np.maximum.at(vm, E[:, 1], r)
    return vm


def render_step(data, gi, step, out_dir, label, focus=None, resolution=512):
    """A close-up sheet of one garment at one step: the garment coloured by stretch (white 1.0,
    red 1.3, magenta 1.6 and over; physbone slate), the posed body grey, both cropped to the
    focused vertices' posed bounds plus ``CROP``, from the body's front, back, left and right.
    Temporary objects are removed. Returns ``render_mesh.render``'s line."""
    from . import render_mesh
    g = data["garments"][gi]
    body = data["body"]
    BV, GV = posed(data, g, step)
    focus = (~g["cut"]) if focus is None else focus
    lo, hi = GV[focus].min(0) - CROP, GV[focus].max(0) + CROP
    vm = vertex_stretch(data, g, GV)
    t = np.clip((vm - 1.0) / 0.3, 0, 1)
    t2 = np.clip((vm - 1.3) / 0.3, 0, 1)
    gcol = np.stack([np.ones_like(t), 1 - t, np.maximum(1 - t, t2), np.ones_like(t)], 1)
    gcol[g["cls"]["physbone"]] = (0.35, 0.45, 0.6, 1.0)
    made = []
    try:
        for name, V, F, col in (("__fit_body", BV, body["F"], None), ("__fit_garment", GV, g["F"], gcol)):
            inside = ((V >= lo) & (V <= hi)).all(1)
            used, inv = np.unique(F[inside[F].any(1)], return_inverse=True)
            me = bpy.data.meshes.new(name)
            me.from_pydata(V[used].tolist(), [], inv.reshape(-1, 3).tolist())
            me.update()
            c = np.tile([0.62, 0.62, 0.62, 1.0], (len(used), 1)) if col is None else col[used]
            a = me.color_attributes.new("fit", 'FLOAT_COLOR', 'POINT')
            a.data.foreach_set("color", c.astype(np.float32).ravel())
            ob = bpy.data.objects.new(name, me)
            bpy.context.scene.collection.objects.link(ob)
            made.append(ob)
        return render_mesh.render(label=label, only=[o.name for o in made], angles=_view_angles(data["frame"]),
                                  shading="vertexcolor", resolution=resolution, out_dir=out_dir)
    finally:
        for ob in made:
            me = ob.data
            bpy.data.objects.remove(ob, do_unlink=True)
            bpy.data.meshes.remove(me)


def png_of(line) -> Optional[str]:
    m = re.search(r"\| png=(.+)$", line or "")
    return m.group(1).strip() if m else None
