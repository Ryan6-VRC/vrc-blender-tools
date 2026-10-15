"""Mark a part of a garment as a Delete or Hide carrier: a shape key that moves the part past the ShapeChanger threshold, rigidly or collapsed to a point.

A **selection** is the fit family's ``--region`` grammar (``fit.parse_region``), read on the
mesh's Basis in world space: ``group:``, ``bone:``, ``slot:`` and ``zone:`` terms, ``+``
intersecting and repeated regions adding together. ``bone:`` reads each vertex's largest
influence over the deform bones of the mesh's own armature, and ``zone:`` the body frame of
that armature (``weight_transfer.body_frame``).

The selection is then quantified over **islands**, the mesh's edge-linked parts: an island
is taken whole when ``any``, at least ``half``, or ``all`` of its vertices are in the
selection, and no vertex outside a taken island moves. Each taken island moves **rigidly** by
default, every vertex by the one vector that carries the island's centroid to its target, or
with ``collapse`` every vertex onto the target point. The target is the head of the island's
dominant bone, the deform bone carrying its largest summed weight (a tie going to the first in
the mesh's vertex-group order), unless ``to`` names a bone or a world point for every island.

The consumer is a Modular Avatar ShapeChanger Delete, which keeps a vertex its shape moves by
the threshold or less, so a key moving any vertex at or under ``threshold`` (world metres,
what the export's scale bake leaves the mesh in) refuses rather than leaving it standing. A
Hide toggle animating the same key shows the part wherever the key puts it: a rigid move keeps
the part's shape around its target, so a part larger than the body there stays visible, and
``collapse`` is the form that hides it. A part on a PhysBone chain is carried away by the
chain wherever the key moves it, so its key only marks it for deletion.

The key is written by ``coverage.add_key``, the carrier writer ``mark_coverage`` shares, and
each target object records ``{key: recipe}`` in ``avatarprep_authored``. ``replace`` rewrites
only a key that stamp lists: a vendor key, or another door's, is never overwritten.

Pure ``bpy`` data access, no operators beyond leaving Edit Mode. Door: ``cli/mark_parts.py``.
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import coverage, fit, scene_utils
from . import weight_transfer as WT

ISLAND_MODES = ("any", "half", "all")
POSE_TOL = 1e-4


class PartsError(ValueError):
    """A refusal the door prints in-grammar; names the offender and the fix."""


def object_mode() -> None:
    """Leave Edit (or any non-Object) Mode, so mesh data reads what the edit holds and a saved
    blend reopens in Object Mode."""
    active = bpy.context.view_layer.objects.active
    if active is not None and active.mode != 'OBJECT':
        scene_utils.op_override(bpy.ops.object.mode_set, {'active_object': active, 'object': active},
                                mode='OBJECT')


def _armature(ob):
    """``(armature, None)`` or ``(None, reason)``; only a term or target that reads bones needs one."""
    try:
        arm, _ = WT.armature_of(ob)
    except WT.WeightTransferError as e:
        return None, str(e)
    return arm, None


def check_target(ob) -> None:
    """Refuse a mesh the door cannot write a key on, or whose rig is posed."""
    if ob is None or ob.type != 'MESH':
        raise PartsError("%s is not a mesh object" % getattr(ob, "name", ob))
    if scene_utils.is_linked(ob) or not scene_utils.is_editable(ob):
        raise PartsError("%s is linked or override data and cannot take a shape key; mark the local "
                         "garment this costume owns (own-mergeable makes one)" % ob.name)
    others = ob.data.users - (1 if ob.data.use_fake_user else 0) - 1
    if others > 0:
        raise PartsError("%s shares its mesh with %d other user(s), so a key on it would land on them "
                         "too; make it single-user first" % (ob.name, others))
    arm, _ = _armature(ob)
    if arm is not None:
        ident = Matrix.Identity(4)
        for pb in arm.pose.bones:
            if max(abs(pb.matrix_basis[r][c] - ident[r][c]) for r in range(4) for c in range(4)) > POSE_TOL:
                raise PartsError("%s is posed (bone %s); the key is read off rest positions and bone heads "
                                 "at rest, so clear the pose, or bake it with apply_pose, first"
                                 % (arm.name, pb.name))


def check_key(ob, key: str, *, replace: bool = False) -> None:
    """Every refusal the write makes about the key, read-only, so ``--whatif`` refuses it too:
    the writer's (``coverage.check_key``), a non-map ``avatarprep_authored`` stamp, and
    ``replace`` on a key the stamp does not list."""
    stamp = ob.get(scene_utils.STAMP_AUTHORED)
    if stamp is not None and not hasattr(stamp, "keys"):
        raise PartsError("%s's avatarprep_authored is %r, not a {key: recipe} map; delete the property "
                         "before authoring, since this door will not overwrite it" % (ob.name, stamp))
    try:
        existing = coverage.check_key(ob, key, replace=replace)
    except coverage.CoverageError as e:
        raise PartsError(str(e))
    if existing is not None and key not in (stamp or {}):
        raise PartsError("--replace would overwrite %r on %s, which mark_parts did not author (it is not in "
                         "avatarprep_authored); a vendor key or another door's is never overwritten, so "
                         "author the part under a new name" % (key, ob.name))


def basis_world(ob) -> np.ndarray:
    """Basis vertex positions (the mesh's own coordinates when it has no keys) in world space."""
    me = ob.data
    src = me.shape_keys.key_blocks[0].data if me.shape_keys else me.vertices
    co = np.empty((len(me.vertices), 3), np.float64)
    src.foreach_get("co", co.ravel())
    M = np.array(ob.matrix_world, np.float64)
    return (np.c_[co, np.ones(len(co))] @ M.T)[:, :3]


def islands_of(me) -> np.ndarray:
    """Per vertex, the index of its edge-linked island: the lowest vertex index in it."""
    n = len(me.vertices)
    parent = np.arange(n)
    E = np.empty((len(me.edges), 2), np.int64)
    me.edges.foreach_get("vertices", E.ravel())

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for a, b in E:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)], np.int64)


def _deform_weights(ob, arm):
    """``(names, W)``: the vertex groups naming deform bones of ``arm`` and their weights."""
    names, W = WT.group_matrix(ob)
    deform = WT.deform_bones(arm) if arm is not None else set()
    cols = [j for j, n in enumerate(names) if n in deform]
    return [names[j] for j in cols], np.asarray(W[:, cols], np.float64)


def select(ob, regions: Sequence[str], *, islands: str = "half", reference_bone: str = "Hips",
           forward_bone: Optional[str] = None) -> Dict:
    """The vertices the regions select, and the islands that quantifies them to.

    Returns ``{"qualifying", "island", "taken", "selected", "islands"}``: per-vertex masks for
    the regions' union and for the taken islands, each vertex's island index, the taken island
    indices, and per island ``{"island", "vertices", "qualifying"}`` for every island holding a
    qualifying vertex. Raises ``PartsError``."""
    if islands not in ISLAND_MODES:
        raise PartsError("--islands is one of %s, got %r" % (", ".join(ISLAND_MODES), islands))
    if not regions:
        raise PartsError("name the part with --region")
    terms = []
    for spec in regions:
        try:
            terms.extend(fit.parse_region(spec))
        except ValueError as e:
            raise PartsError(str(e))
    arm, why = _armature(ob)
    if arm is None and any(k in ("bone", "zone") for k, _ in terms):
        raise PartsError("bone: and zone: terms read %s's armature, and %s" % (ob.name, why))
    names, W = _deform_weights(ob, arm)
    under = {b.name: {b.name} | {c.name for c in b.children_recursive} for b in arm.data.bones} if arm else {}
    if arm is not None:
        dominant, _ = fit._dominant(names, fit.top4(W), arm)
    else:
        dominant = np.full(len(ob.data.vertices), None, dtype=object)
    slots = [(s.material.name if s.material else s.name) for s in ob.material_slots]
    for k, v in terms:
        if k == "slot" and not fit._matching(slots, v):
            raise PartsError("slot:%s matches no material slot of %s (%s)"
                             % (v, ob.name, ", ".join(slots) or "it has none"))
    patterns = fit.region_groups(regions)
    groups = fit.group_weights(ob, patterns)
    missing = [n for n in patterns if n not in groups]
    if missing:
        raise PartsError("group:%s matches no vertex group of %s" % (",".join(missing), ob.name))
    V = basis_world(ob)
    g = {"name": ob.name, "object": ob, "V": V, "cut": np.zeros(len(V), bool), "groups": groups,
         "under": under, "dominant": dominant, "armature": arm.name if arm else "(none)"}
    data = {"garments": [g], "frame": None,
            "body": {"bones": fit._bones(arm, names) if arm else {}, "armature": arm.name if arm else "(none)"}}
    if any(k == "zone" for k, _ in terms):
        try:
            data["frame"] = WT.body_frame(arm, reference_bone, forward_bone)
        except WT.WeightTransferError as e:
            raise PartsError(str(e))
    try:
        masks = fit.resolve_regions(data, list(regions))
    except fit.FitError as e:
        raise PartsError(str(e))
    qualifying = np.zeros(len(V), bool)
    for m in masks.values():
        qualifying |= m[0]
    island = islands_of(ob.data)
    size = np.bincount(island, minlength=len(V))
    hits = np.bincount(island[qualifying], minlength=len(V))
    rows = []
    for i in np.flatnonzero(hits):
        rows.append({"island": int(i), "vertices": int(size[i]), "qualifying": int(hits[i])})
    rule = {"any": lambda r: r["qualifying"] > 0, "half": lambda r: 2 * r["qualifying"] >= r["vertices"],
            "all": lambda r: r["qualifying"] == r["vertices"]}[islands]
    taken = [r["island"] for r in rows if rule(r)]
    selected = np.isin(island, taken)
    return {"qualifying": qualifying, "island": island, "taken": taken, "selected": selected,
            "islands": rows}


def _target_point(arm, to) -> Tuple[np.ndarray, str]:
    if isinstance(to, str):
        if arm is None or arm.data.bones.get(to) is None:
            raise PartsError("--to %s is not a bone of %s; name one of its bones or a world point x,y,z"
                             % (to, arm.name if arm else "the mesh's armature (it has none)"))
        return np.array(arm.matrix_world @ arm.data.bones[to].head_local, np.float64), "%s head" % to
    return np.array(to, np.float64), "(%g, %g, %g)" % tuple(to)


def plan(ob, selection: Dict, *, to: Union[None, str, Sequence[float]] = None,
         threshold: float = 0.01, collapse: bool = False, key: Optional[str] = None,
         replace: bool = False) -> Dict:
    """Where each taken island moves. Returns ``{"islands": [...], "offsets", "vertices",
    "min_delta", "max_delta"}``: per island its vertex count, dominant bone, target and its
    vertices' smallest and largest world move (equal under a rigid move); ``offsets`` the
    object-space ``{vertex: Vector}`` ``coverage.add_key`` writes. Refuses an empty selection,
    an island with no deform weight and no ``to``, any vertex whose move is at or under
    ``threshold``, and, when ``key`` is named, whatever ``check_key`` refuses."""
    if key is not None:
        check_key(ob, key, replace=replace)
    taken = selection["taken"]
    if not taken:
        best = max(selection["islands"], key=lambda r: r["qualifying"] / r["vertices"], default=None)
        hint = (" (the best island holds %d of %d vertices in it; --islands any takes it)"
                % (best["qualifying"], best["vertices"])) if best else ""
        raise PartsError("the regions select zero vertices of %s%s" % (ob.name, hint))
    arm, why = _armature(ob)
    if to is None and arm is None:
        raise PartsError("the default target is each island's dominant bone, and %s; pass --to x,y,z" % why)
    names, W = _deform_weights(ob, arm)
    V = basis_world(ob)
    inv = np.linalg.inv(np.array(ob.matrix_world, np.float64)[:3, :3])
    island = selection["island"]
    fixed = _target_point(arm, to) if to is not None else None
    rows, offsets, short = [], {}, []
    for i in taken:
        idx = np.flatnonzero(island == i)
        sums = W[idx].sum(0) if len(names) else np.zeros(0)
        dom = names[int(np.argmax(sums))] if len(names) and sums.max() > 0 else None
        if fixed is None:
            if dom is None:
                raise PartsError("island at vertex %d of %s (%d vertices) carries no deform weight, so it "
                                 "has no dominant bone to move to; pass --to BONE or --to x,y,z"
                                 % (i, ob.name, len(idx)))
            point, label = _target_point(arm, dom)
        else:
            point, label = fixed
        if collapse:
            moves = point - V[idx]
        else:
            moves = np.repeat((point - V[idx].mean(0))[None, :], len(idx), 0)
        lengths = np.linalg.norm(moves, axis=1)
        local = moves @ inv.T
        for v, d in zip(idx, local):
            offsets[int(v)] = Vector(d)
        row = {"island": int(i), "vertices": int(len(idx)), "dominant": dom, "target": label,
               "min_m": round(float(lengths.min()), 6), "max_m": round(float(lengths.max()), 6),
               "short": int((lengths <= threshold).sum())}
        rows.append(row)
        if row["short"]:
            short.append(row)
    if short:
        listed = "; ".join("island at vertex %d (%d vertices, %d of them) moves %.4f m to %s"
                           % (r["island"], r["vertices"], r["short"], r["min_m"], r["target"])
                           for r in short[:5])
        raise PartsError("%d island(s) of %s would move a vertex %g m or less, which the Delete keeps "
                         "standing: %s%s. Pass --to a bone or a point farther from them"
                         % (len(short), ob.name, threshold, listed, ", ..." if len(short) > 5 else ""))
    deltas = [r["min_m"] for r in rows] + [r["max_m"] for r in rows]
    return {"islands": rows, "offsets": offsets, "vertices": len(offsets),
            "min_delta": min(deltas), "max_delta": max(deltas)}


def write(ob, key: str, the_plan: Dict, *, recipe: str, replace: bool = False):
    """Write the key through ``coverage.add_key`` and record ``{key: recipe}`` in the object's
    ``avatarprep_authored``; parks the active key on Basis. Raises ``PartsError``."""
    check_key(ob, key, replace=replace)
    try:
        kb = coverage.add_key(ob, key, the_plan["offsets"], replace=replace)
    except coverage.CoverageError as e:
        raise PartsError(str(e))
    stamp = ob.get(scene_utils.STAMP_AUTHORED)
    stamp = dict(stamp) if stamp is not None else {}
    stamp[key] = recipe
    ob[scene_utils.STAMP_AUTHORED] = stamp
    ob.active_shape_key_index = 0
    return kb
