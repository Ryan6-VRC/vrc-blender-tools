"""Fold a doomed bone chain's weight into surviving neighbours, then remove it.

Generalises a personal-venue one-off (a fixed per-bone blend table folding a
vendor hair asset's ribbon bones into its scalp neighbours): its hardcoded blend
table becomes the ``bone_map`` argument here, and its bone list becomes
``doomed``. The cap-then-renormalise step (:func:`_fold_mesh`) is that script's
algorithm verbatim.

Pure ``bpy`` data access, no operators, no UI. Door: ``cli/fold_bones.py``.
"""
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import bpy
import fnmatch
import mathutils

from . import scene_utils

# Unity's per-vertex bone-influence cap. A constant, never a flag (tool-design.md
# "Flags converge on the family's names").
MAX_BONE_GROUPS = 4

# Below this a weight (or a distance) reads as zero for gate/split purposes.
WEIGHT_EPS = 1e-9
DISTANCE_EPS = 1e-9

# A ratio this close to 1 means the auto split is a coin flip between the two
# nearest candidates — flagged, never blocked (the operator's eye decides).
AMBIGUOUS_RATIO = 1.2


class NoBonesMatched(ValueError):
    """``--bones`` pattern matched no bone in the target armature. Raised by
    :func:`resolve_doomed`; also what a rerun on an already-folded output hits,
    since the doomed bones no longer exist to match."""

    def __init__(self, pattern):
        self.pattern = pattern
        super().__init__("--bones %r matched no bone in the armature" % pattern)


class FoldRefused(ValueError):
    """One of fold_bones' five pre-write gates. ``kind`` tags which one (for a
    caller that wants to branch); ``offenders`` is a list of dicts/strings naming
    what tripped it, for the CLI's ``OFFENDER`` lines. Nothing is mutated before
    this raises — every gate in this module runs before any bpy write."""

    def __init__(self, kind, message, offenders):
        self.kind = kind
        self.offenders = offenders
        super().__init__(message)


def match_bones(pattern: str, names: Iterable[str]) -> List[str]:
    """Case-insensitive glob (or exact name) match of ``pattern`` over ``names``,
    preserving ``names``' order. A plain name is a valid glob with no wildcard, so
    this is the one matcher for both forms the door's ``--bones``/``--neighbours``
    accept."""
    pat = pattern.casefold()
    return [n for n in names if fnmatch.fnmatchcase(n.casefold(), pat)]


def resolve_doomed(armature: bpy.types.Object, patterns: Sequence[str]) -> List[str]:
    """Resolve ``--bones GLOB|LIST`` against ``armature``'s bones. Each pattern must
    match at least one bone or this raises :class:`NoBonesMatched` naming it — the
    same gate a rerun on a folded output hits, since the doomed names are gone.
    Preserves armature order; a bone matched by two patterns is listed once."""
    names = [b.name for b in armature.data.bones]
    doomed: List[str] = []
    seen: Set[str] = set()
    for pat in patterns:
        hits = [n for n in match_bones(pat, names) if n not in seen]
        if not hits:
            raise NoBonesMatched(pat)
        seen.update(hits)
        doomed.extend(hits)
    return doomed


def _under_doomed(bone: bpy.types.Bone, doomed: Set[str]) -> bool:
    p = bone.parent
    while p is not None:
        if p.name in doomed:
            return True
        p = p.parent
    return False


def check_no_surviving_children(armature: bpy.types.Object, doomed: Set[str]) -> None:
    """Gate: a doomed bone with a surviving (non-doomed) child. ``edit_bones.remove``
    splices such a child onto the removed bone's parent, silently re-routing a
    survivor to a new hierarchy path — the same hazard ``prune_bones`` documents."""
    offenders = []
    for name in sorted(doomed):
        b = armature.data.bones.get(name)
        if b is None:
            continue
        for c in b.children:
            if c.name not in doomed:
                offenders.append({"bone": name, "child": c.name})
    if offenders:
        raise FoldRefused(
            "surviving_child",
            "doomed bone(s) have a surviving child, which removal would misroute: %s"
            % ", ".join("%s -> %s" % (o["bone"], o["child"]) for o in offenders),
            offenders)


def _mesh_bone_weight_total(mesh_obj: bpy.types.Object, vg_index: int) -> float:
    total = 0.0
    for v in mesh_obj.data.vertices:
        for g in v.groups:
            if g.group == vg_index:
                total += g.weight
    return total


def check_no_foreign_weight(doomed: Set[str], targets: Sequence[bpy.types.Object]) -> None:
    """Gate: any mesh OTHER than ``targets``, anywhere in the file, weighting a
    doomed bone — it would be orphaned onto a bone the fold is about to delete,
    silently, since nothing outside ``targets`` is touched here."""
    target_names = {m.name for m in targets}
    offenders = []
    for ob in bpy.data.objects:
        if ob.type != 'MESH' or ob.name in target_names:
            continue
        hits = []
        for n in sorted(doomed):
            vg = ob.vertex_groups.get(n)
            if vg is None:
                continue
            if _mesh_bone_weight_total(ob, vg.index) > WEIGHT_EPS:
                hits.append(n)
        if hits:
            offenders.append({"mesh": ob.name, "bones": hits})
    if offenders:
        raise FoldRefused(
            "foreign_mesh",
            "other mesh(es) in the file weight a doomed bone: %s"
            % ", ".join("%s:%s" % (o["mesh"], o["bones"]) for o in offenders),
            offenders)


def check_map_destinations(bone_map: Dict[str, Dict[str, float]],
                           armature: bpy.types.Object, doomed: Set[str]) -> None:
    """Gate: a map row naming a destination that is not a surviving bone — missing
    from the armature entirely, or itself doomed (about to be removed too)."""
    all_names = {b.name for b in armature.data.bones}
    offenders = []
    for bone, dests in bone_map.items():
        for tgt in dests:
            if tgt not in all_names or tgt in doomed:
                offenders.append({"bone": bone, "destination": tgt})
    if offenders:
        raise FoldRefused(
            "bad_destination",
            "map names a destination that is not a surviving bone: %s"
            % ", ".join("%s -> %s" % (o["bone"], o["destination"]) for o in offenders),
            offenders)


def check_full_coverage(bone_map: Dict[str, Dict[str, float]], weighted_doomed: Set[str]) -> None:
    """Gate: a doomed bone that carries weight somewhere in ``targets`` but has no
    map row. A weightless doomed bone is legitimate and needs none — it is simply
    removed, contributing nothing."""
    missing = sorted(n for n in weighted_doomed if n not in bone_map)
    if missing:
        raise FoldRefused(
            "unmapped_bone",
            "weighted doomed bone(s) have no map row: %s" % missing,
            missing)


def _vertex_bone_weights(mesh_obj: bpy.types.Object,
                         bone_names: Set[str]) -> Dict[int, Dict[str, float]]:
    """``{vertex index: {bone-named group: weight}}``, restricted to vertex groups
    named after an armature bone (doomed or surviving) and weight over
    :data:`WEIGHT_EPS`. A non-bone group (a mask, a shape-key carrier) never
    appears here — it is read nowhere in this module and so is never touched."""
    idx_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups if vg.name in bone_names}
    out: Dict[int, Dict[str, float]] = {}
    for v in mesh_obj.data.vertices:
        row: Dict[str, float] = {}
        for g in v.groups:
            name = idx_to_name.get(g.group)
            if name is None or g.weight <= WEIGHT_EPS:
                continue
            row[name] = row.get(name, 0.0) + g.weight
        if row:
            out[v.index] = row
    return out


def _weighted_doomed_bones(per_mesh_weights: Dict[str, Dict[int, Dict[str, float]]],
                           doomed: Set[str]) -> Set[str]:
    found: Set[str] = set()
    for weights in per_mesh_weights.values():
        for row in weights.values():
            found |= (set(row) & doomed)
    return found


def _fold_mesh(weights: Dict[int, Dict[str, float]], doomed: Set[str],
               bone_map: Dict[str, Dict[str, float]]) -> dict:
    """The per-vertex fold on one mesh's pre-read ``weights``. Per touched vertex:
    ``new[t] = sum(w_old[b] * frac[b][t])`` over doomed bones ``b``, added to any
    existing (survivor) weight already on ``t``; keep the top
    :data:`MAX_BONE_GROUPS` groups by weight; renormalise the kept set back to the
    pre-cap total. This is ``reweight_ribbon_rosyloop.py``'s remap loop verbatim,
    generalised past its single-island assumption (every ORIGINAL bone-named group
    on a touched vertex, survivor or doomed, is cleared and rewritten from the
    ranked set — not only the doomed ones — so a survivor group capped out here
    cannot leave stale weight behind)."""
    touched = 0
    capped = 0
    zero_sum: List[int] = []
    new_weights: Dict[int, Dict[str, float]] = {}
    dest_before: Dict[str, float] = defaultdict(float)
    dest_after: Dict[str, float] = defaultdict(float)

    for vidx, row in weights.items():
        for name, w in row.items():
            dest_before[name] += w
        if not (set(row) & doomed):
            continue
        touched += 1
        acc: Dict[str, float] = {}
        for name, w in row.items():
            if name in doomed:
                for tgt, frac in bone_map.get(name, {}).items():
                    acc[tgt] = acc.get(tgt, 0.0) + w * frac
            else:
                acc[name] = acc.get(name, 0.0) + w
        total = sum(acc.values())
        if total <= WEIGHT_EPS:
            zero_sum.append(vidx)
            continue
        ranked = sorted(acc.items(), key=lambda kv: -kv[1])[:MAX_BONE_GROUPS]
        if len(acc) > MAX_BONE_GROUPS:
            capped += 1
        rtotal = sum(w for _, w in ranked)
        final = {g: w / rtotal for g, w in ranked}
        new_weights[vidx] = final
        for name, w in final.items():
            dest_after[name] += w

    # Untouched vertices' weight is unchanged, so fold it into the "after" total too
    # — the report's before/after pair is a whole-mesh mass check, not just a
    # touched-vertex one.
    for vidx, row in weights.items():
        if vidx in new_weights or (set(row) & doomed):
            continue
        for name, w in row.items():
            dest_after[name] += w

    return {"touched": touched, "capped": capped, "zero_sum": zero_sum,
            "new_weights": new_weights,
            "dest_before": dict(dest_before), "dest_after": dict(dest_after)}


def _combine_dest_totals(fold_results: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    before: Dict[str, float] = defaultdict(float)
    after: Dict[str, float] = defaultdict(float)
    for r in fold_results.values():
        for n, w in r["dest_before"].items():
            before[n] += w
        for n, w in r["dest_after"].items():
            after[n] += w
    names = set(before) | set(after)
    return {n: {"before": before.get(n, 0.0), "after": after.get(n, 0.0)} for n in sorted(names)}


def _snapshot_extra(mesh_obj: bpy.types.Object, bone_names: Set[str]) -> dict:
    """Non-bone vertex-group weights and shape-key names, read before mutation so
    :func:`_assert_invariants` can prove neither moved — this module never writes
    to either, but the contract asks for a real post-condition check, not a claim."""
    sk = mesh_obj.data.shape_keys
    shape_names = [k.name for k in sk.key_blocks] if sk else None
    non_bone: Dict[int, Dict[str, float]] = {}
    idx_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    for v in mesh_obj.data.vertices:
        for g in v.groups:
            name = idx_to_name.get(g.group)
            if name is not None and name not in bone_names and g.weight > WEIGHT_EPS:
                non_bone.setdefault(v.index, {})[name] = g.weight
    return {"shape_names": shape_names, "non_bone": non_bone}


def _weights_differ(a: Dict[str, float], b: Dict[str, float], eps: float = 1e-6) -> bool:
    if set(a) != set(b):
        return True
    return any(abs(a[k] - b[k]) > eps for k in a)


def _assert_invariants(mesh_obj: bpy.types.Object, weights_before: Dict[int, Dict[str, float]],
                       new_weights: Dict[int, Dict[str, float]], bone_names: Set[str],
                       extra_before: dict) -> None:
    """Post-write checks the contract asks for explicitly: untouched vertices
    identical, non-bone groups bit-identical, shape-key metadata unchanged, no
    vertex over the cap. Raises ``AssertionError`` naming the offender — these are
    real regression guards, not documentation of an invariant assumed to hold."""
    idx_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    for v in mesh_obj.data.vertices:
        current: Dict[str, float] = {}
        for g in v.groups:
            name = idx_to_name.get(g.group)
            if name in bone_names and g.weight > WEIGHT_EPS:
                current[name] = current.get(name, 0.0) + g.weight
        if len(current) > MAX_BONE_GROUPS:
            raise AssertionError("vertex %d has %d bone groups after fold (cap is %d)"
                                 % (v.index, len(current), MAX_BONE_GROUPS))
        if v.index not in new_weights:
            before = weights_before.get(v.index, {})
            if _weights_differ(current, before):
                raise AssertionError("untouched vertex %d changed: %r -> %r"
                                     % (v.index, before, current))

    extra_after = _snapshot_extra(mesh_obj, bone_names)
    if extra_after["shape_names"] != extra_before["shape_names"]:
        raise AssertionError("shape-key metadata changed: %r -> %r"
                             % (extra_before["shape_names"], extra_after["shape_names"]))
    before_nb = extra_before["non_bone"]
    after_nb = extra_after["non_bone"]
    if set(before_nb) != set(after_nb) or any(
            _weights_differ(before_nb[k], after_nb[k]) for k in before_nb):
        raise AssertionError("non-bone vertex group weights changed")


def _apply_fold(mesh_obj: bpy.types.Object, weights_before: Dict[int, Dict[str, float]],
                new_weights: Dict[int, Dict[str, float]]) -> None:
    vg = mesh_obj.vertex_groups
    dest_names = {n for final in new_weights.values() for n in final}
    for n in sorted(dest_names):
        if vg.get(n) is None:
            vg.new(name=n)
    for vidx, final in new_weights.items():
        for name in weights_before[vidx]:
            g = vg.get(name)
            if g:
                g.remove([vidx])
        for name, w in final.items():
            vg[name].add([vidx], w, 'REPLACE')


def fold_bones(armature: bpy.types.Object, meshes: Sequence[bpy.types.Object],
              doomed: Iterable[str], bone_map: Dict[str, Dict[str, float]],
              whatif: bool = False) -> dict:
    """Fold ``doomed`` bones' weight into ``bone_map``'s survivors across
    ``meshes``, then remove the doomed vertex groups and edit bones.

    ``bone_map`` must be an explicit ``{doomed_bone: {survivor: fraction}}`` dict —
    there is no in-executor "auto" mode. :func:`auto_map` only ever computes a
    suggestion table; nothing here can run off it directly, which is what makes
    the door's auto-refuses-to-run behaviour true rather than asserted.

    Runs every gate (:func:`check_no_surviving_children`,
    :func:`check_no_foreign_weight`, :func:`check_map_destinations`,
    :func:`check_full_coverage`, and the zero-sum-vertex check below) before any
    mutation, ``whatif`` or not — a preview that could disagree with the real run
    is worthless (``prune_bones``' standard).

    Returns a report dict: ``bones_removed`` (``[]`` under ``whatif``, since
    nothing is removed yet — see ``doomed`` for the plan), ``touched``, ``capped``,
    ``destination_totals`` (``{bone: {"before", "after"}}``, whole-mesh mass check),
    and ``meshes`` (per-target ``{"touched", "capped"}``). ``whatif`` adds
    ``whatif=True``.

    Raises:
        FoldRefused: a gate fired. Nothing is mutated.
    """
    if not isinstance(bone_map, dict):
        raise ValueError("fold_bones requires an explicit bone_map dict — 'auto' only "
                         "writes a suggestion table via auto_map(), it never executes")
    doomed = set(doomed)
    check_no_surviving_children(armature, doomed)
    check_no_foreign_weight(doomed, meshes)

    all_bone_names = {b.name for b in armature.data.bones}
    per_mesh_weights = {m.name: _vertex_bone_weights(m, all_bone_names) for m in meshes}
    weighted_doomed = _weighted_doomed_bones(per_mesh_weights, doomed)

    check_map_destinations(bone_map, armature, doomed)
    check_full_coverage(bone_map, weighted_doomed)

    fold_results: Dict[str, dict] = {}
    zero_sum_offenders = []
    for m in meshes:
        res = _fold_mesh(per_mesh_weights[m.name], doomed, bone_map)
        fold_results[m.name] = res
        for vidx in res["zero_sum"]:
            zero_sum_offenders.append({"mesh": m.name, "vertex": vidx})
    if zero_sum_offenders:
        raise FoldRefused(
            "zero_sum",
            "%d vertex/vertices fold to zero bone weight" % len(zero_sum_offenders),
            zero_sum_offenders)

    touched_total = sum(r["touched"] for r in fold_results.values())
    capped_total = sum(r["capped"] for r in fold_results.values())
    dest_totals = _combine_dest_totals(fold_results)
    per_mesh_report = {name: {"touched": r["touched"], "capped": r["capped"]}
                       for name, r in fold_results.items()}

    if whatif:
        return {"whatif": True, "bones_removed": [], "planned_bones": sorted(doomed),
                "touched": touched_total, "capped": capped_total,
                "destination_totals": dest_totals, "meshes": per_mesh_report}

    extras_before = {m.name: _snapshot_extra(m, all_bone_names) for m in meshes}
    for m in meshes:
        _apply_fold(m, per_mesh_weights[m.name], fold_results[m.name]["new_weights"])
        _assert_invariants(m, per_mesh_weights[m.name], fold_results[m.name]["new_weights"],
                           all_bone_names, extras_before[m.name])

    for m in meshes:
        vg = m.vertex_groups
        for n in sorted(doomed):
            g = vg.get(n)
            if g:
                vg.remove(g)

    removed_bones: List[str] = []
    with scene_utils.edit_mode(armature) as ebs:
        for n in sorted(doomed):
            b = ebs.get(n)
            if b:
                ebs.remove(b)
                removed_bones.append(n)

    return {"bones_removed": removed_bones, "touched": touched_total, "capped": capped_total,
            "destination_totals": dest_totals, "meshes": per_mesh_report}


def _point_segment_distance(p: mathutils.Vector, a: mathutils.Vector,
                            b: mathutils.Vector) -> float:
    ab = b - a
    denom = ab.length_squared
    if denom <= DISTANCE_EPS:
        return (p - a).length
    t = max(0.0, min(1.0, (p - a).dot(ab) / denom))
    return (p - (a + ab * t)).length


def auto_map(armature: bpy.types.Object, meshes: Sequence[bpy.types.Object],
            doomed: Iterable[str], neighbours: Optional[Sequence[str]] = None
            ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, dict]]:
    """Suggest a ``bone_map`` for :func:`fold_bones`: per doomed bone, the
    weight-averaged centroid (rest space) of the vertices it weights across
    ``meshes``, split by inverse distance over the nearest two candidate bones'
    rest segments (head-tail). A weightless doomed bone gets no row — it needs
    none (:func:`check_full_coverage`) — and is reported ``weightless``. Read-only:
    never mutates the armature or meshes, and never calls :func:`fold_bones` — the
    door's "auto refuses to run" is this function simply never reaching the
    executor, not a flag it checks.

    ``neighbours``, when given, is the candidate list verbatim (validated: each
    must name a surviving bone). Otherwise the candidates are the armature's
    surviving deform bones (``Bone.use_deform``) that are not themselves under a
    doomed bone.

    Deterministic: same input, same output, since the centroid and distances are
    plain arithmetic over stored rest data with no operator/RNG in the path.

    Returns ``(table, info)``:
      ``table`` — exactly the ``{doomed_bone: {survivor: fraction}}`` shape
      ``--map`` accepts, so ``--map-out``'s file is directly reusable as
      ``--map`` with no reshaping.
      ``info`` — per doomed bone, ``{"weightless", "candidates", "distances",
      "ratio", "ambiguous"}`` for the door's printed diagnostics and
      ``--report``; deliberately NOT folded into ``table``, which stays the
      strict fraction shape.

    Raises:
        FoldRefused: :func:`check_no_surviving_children` fired (auto assumes a
            clean doomed subtree, same as the executor).
        ValueError: ``neighbours`` names a missing or doomed bone, or the
            candidate set is empty.
    """
    doomed = set(doomed)
    check_no_surviving_children(armature, doomed)
    all_bones = armature.data.bones

    if neighbours:
        missing = [n for n in neighbours if all_bones.get(n) is None]
        if missing:
            raise ValueError("--neighbours name(s) not in the armature: %s" % missing)
        bad = [n for n in neighbours if n in doomed]
        if bad:
            raise ValueError("--neighbours name(s) are themselves doomed: %s" % bad)
        candidates = list(dict.fromkeys(neighbours))
    else:
        candidates = [b.name for b in all_bones
                     if b.name not in doomed and getattr(b, "use_deform", True)
                     and not _under_doomed(b, doomed)]
    if not candidates:
        raise ValueError("no surviving candidate bones for auto mapping "
                         "(pass --neighbours to name them explicitly)")

    all_bone_names = {b.name for b in all_bones}
    per_mesh_weights = {m.name: _vertex_bone_weights(m, all_bone_names) for m in meshes}
    arm_inv = armature.matrix_world.inverted()

    table: Dict[str, Dict[str, float]] = {}
    info: Dict[str, dict] = {}
    for bone in sorted(doomed):
        num = mathutils.Vector((0.0, 0.0, 0.0))
        den = 0.0
        for m in meshes:
            weights = per_mesh_weights[m.name]
            to_arm = arm_inv @ m.matrix_world
            for vidx, row in weights.items():
                w = row.get(bone)
                if not w:
                    continue
                co = to_arm @ m.data.vertices[vidx].co
                num = num + co * w
                den += w
        if den <= WEIGHT_EPS:
            info[bone] = {"weightless": True, "candidates": [], "distances": {},
                         "ratio": None, "ambiguous": False}
            continue
        centroid = num / den
        dists = sorted(
            ((c, _point_segment_distance(centroid, all_bones[c].head_local,
                                         all_bones[c].tail_local)) for c in candidates),
            key=lambda cd: cd[1])

        if len(dists) == 1:
            name1, _d1 = dists[0]
            split = {name1: 1.0}
            ratio = None
        else:
            (name1, d1), (name2, d2) = dists[0], dists[1]
            if d1 <= DISTANCE_EPS:
                split = {name1: 1.0}
                ratio = None
            else:
                inv1 = 1.0 / d1
                inv2 = 1.0 / d2 if d2 > DISTANCE_EPS else inv1
                total = inv1 + inv2
                split = {name1: inv1 / total, name2: inv2 / total}
                ratio = d2 / d1

        ambiguous = ratio is not None and ratio <= AMBIGUOUS_RATIO
        table[bone] = split
        info[bone] = {"weightless": False, "candidates": [c for c, _ in dists],
                     "distances": {c: d for c, d in dists},
                     "ratio": ratio, "ambiguous": ambiguous}

    return table, info
