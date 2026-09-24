"""Seat a keyless garment on a body it was not cut for, and optionally give it the body's keys.

A vendor garment carries no shape keys, or not the ones the venue body bakes or drives,
and was cut against some configuration of the vendor body (the vendor's shipped defaults:
Shinano's ``Breasts_big`` radial rests at 0.5, Plum's family sits at neutral). The venue
body rests somewhere else (``Breasts_flat`` baked at 0.2, ``Breasts_small`` baked at 0.6),
so the garment floats or sinks by the difference. This module moves the garment from the
configuration it was authored against to the configuration the body is in now, by
transferring each body key's delta onto the garment through a Surface Deform binding
made on the body at the authored configuration, and can leave the transferred keys on the
garment so a body morph driven later reaches it.

Mechanism: a Surface Deform binding, which weights several nearby polygons per vertex,
over a nearest-surface delta, which maps each vertex to one point; on the gap numbers the
two tie. Neither knows what cloth does: a seat that carries a band across the underbust
fold — un-baking a vendor's ``Breasts_big`` 0.5 moves the cups 25 mm — buckles the band,
because vertices on either side of the fold follow skin that moves by different amounts.
``smooth`` softens that into a gather at a small cost in gap fidelity; the render is the
judge, as it is for any fit.

Two effects, independently switchable by the caller:

* **seat** — the garment Basis moves by ``(state_k - authored_k) * T_k`` for every key
  whose authored configuration differs from the body's state, where ``T_k`` is the
  transferred full delta of key ``k`` and ``state_k`` is the body's effective amount
  (``avatarprep_baked`` cumulative plus live value). A key the caller also adds carries
  the body's state as its live value, and only the authored offset folds into Basis; a key
  not added folds the whole seat into Basis. Either fold is recorded in the garment's
  ``avatarprep_baked`` map — a negative cumulative for an authored offset un-baked
  (``Breasts_big`` -0.5), which ``compose-mergeable`` already reads as a fit-time proxy.
* **add** — the caller names which keys land on the garment as relative keys. Optional:
  not every venue wants a garment carrying body morphs, and a seat alone is a complete fix.

Movement is masked to each key's footprint — garment vertices whose nearest body point the
key moves by more than ``footprint`` — so a ribbon or a strap far from the morph stays put.
The body's disjoint islands no involved key moves — flush pasties, decals — are dropped from
the binding surface and the mask, since a garment vertex nearest to one would otherwise
freeze; the report names them. Where the mapping tears — a garment edge whose ends follow
skin on either side of the underbust crease, or on two different breasts, and so move
apart by more than the edge is long — the move within ``heal`` of the tear is re-solved as
the smooth interpolation of its surroundings and held out of the skin, so a shell cup
gathers instead of shearing into the band.

Pure ``bpy``; headless-safe; never writes the source.
"""
from typing import Any, Dict, List, Optional, Sequence

import bpy
import idprop
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import scene_utils


class TransferError(ValueError):
    """Raised on a bad transfer request. Names the offender."""


def _tris(me) -> List[tuple]:
    out = []
    for p in me.polygons:
        vs = list(p.vertices)
        for i in range(1, len(vs) - 1):
            out.append((vs[0], vs[i], vs[i + 1]))
    return out


def _moving_islands(B, D, tris, keys, footprint):
    """Keep only the source islands at least one of ``keys`` moves by more than ``footprint``.

    A body carries flush overlays as disjoint islands — chest and hip pasties, decals — that
    its morphs leave in place. A garment vertex over one has that overlay as its nearest
    surface, so a nearest-triangle mask reads the key as not moving there and a Surface
    Deform binding damps its move toward the static overlay. Both must see moving skin
    only. Returns the compacted ``(B, D, tris, keep)`` over the kept vertices, ``keep``
    mapping old index to new, and the dropped islands as ``[{verts, max_move_mm}]``."""
    par = list(range(len(B)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a

    for a, b, c in tris:
        for x, y in ((a, b), (b, c)):
            ra, rb = find(x), find(y)
            if ra != rb:
                par[ra] = rb
    move = {}
    for k in keys:
        for i, d in enumerate(D[k]):
            r = find(i)
            if d.length > move.get(r, 0.0):
                move[r] = d.length
    roots = set(find(i) for t in tris for i in t)
    static = {r for r in roots if move.get(r, 0.0) <= footprint}
    if not static:
        return B, D, tris, None, []
    if len(static) == len(roots):
        return B, D, tris, None, []  # nothing moves anywhere: the masks already say so
    size = {}
    for i in range(len(B)):
        r = find(i)
        if r in static:
            size[r] = size.get(r, 0) + 1
    dropped = [{"verts": size[r], "max_move_mm": round(move.get(r, 0.0) * 1000, 2)}
               for r in sorted(static, key=lambda r: -size[r])]
    kept_tris = [t for t in tris if find(t[0]) not in static]
    used = sorted(set(i for t in kept_tris for i in t))
    keep = {old: new for new, old in enumerate(used)}
    B2 = [B[i] for i in used]
    D2 = {k: [D[k][i] for i in used] for k in D}
    tris2 = [tuple(keep[i] for i in t) for t in kept_tris]
    return B2, D2, tris2, keep, dropped


def _baked_map(obj) -> Dict[str, float]:
    raw = obj.get(scene_utils.STAMP_BAKED)
    if raw is None:
        return {}
    if not isinstance(raw, (dict, idprop.types.IDPropertyGroup)):
        raise TransferError("%s on %r is not a map (%r)" % (scene_utils.STAMP_BAKED, obj.name, raw))
    return {k: float(v) for k, v in dict(raw).items()}


def source_state(source) -> Dict[str, float]:
    """The body's effective amount per key: baked cumulative plus live value."""
    baked = _baked_map(source)
    kb = source.data.shape_keys.key_blocks
    return {k.name: baked.get(k.name, 0.0) + float(k.value) for k in kb if k.name != "Basis"}


def _signed_gaps(coords, tree, maxdist=0.08):
    out = []
    for c in coords:
        loc, n, fi, dist = tree.find_nearest(c)
        out.append(None if loc is None or dist > maxdist else (c - loc).dot(n))
    return out


def _stats(vals) -> Dict[str, Any]:
    g = sorted(v for v in vals if v is not None)
    if not g:
        return {"n": 0}
    return {"n": len(g), "min_mm": round(g[0] * 1000, 2),
            "p5_mm": round(g[len(g) // 20] * 1000, 2),
            "median_mm": round(g[len(g) // 2] * 1000, 2),
            "penetrating": sum(1 for v in g if v < -0.0003)}


def scan_authored(source, target, key, values=(0.0, 0.25, 0.5, 0.75, 1.0),
                  authored: Optional[Dict[str, float]] = None, neutral: Sequence[str] = (),
                  footprint=0.015) -> Dict[str, Any]:
    """Which value of ``key`` was ``target`` cut against? Score the garment's Basis against
    the body at each candidate value (other keys at ``authored``), over the vertices in the
    key's core footprint. The authored value reads as the one with a near-zero minimum gap
    and no penetration; a garment that clears the skin at every value is not coupled to
    this key at all. ``neutral`` names keys held at 0 during the scan (the keys the
    transfer will involve), matching the transfer's own reading of an absent key.
    A report, never a choice."""
    authored = dict(authored or {})
    for k in neutral:
        authored.setdefault(k, 0.0)
    B, D, tris = _source_geometry(source)
    for k in [key] + list(authored):
        if k not in D:
            raise TransferError("shape key %r not found on source %r" % (k, source.name))
    state = source_state(source)
    B, D, tris, _, _ = _moving_islands(B, D, tris, [key], 0.001)
    VS = _body_now(B, D, source)
    G = [target.matrix_world @ v.co for v in target.data.vertices]
    core_tri = [max(D[key][j].length for j in t) > footprint for t in tris]
    tree0 = BVHTree.FromPolygons([tuple(v) for v in VS], tris)
    mask = []
    for c in G:
        loc, n, fi, dist = tree0.find_nearest(c)
        mask.append(loc is not None and dist < 0.06 and core_tri[fi])
    if sum(mask) < 20:
        return {"key": key, "target": target.name, "core_verts": sum(mask), "scan": {}}
    rows = {}
    for val in values:
        cfg = dict(authored); cfg[key] = val
        V = _body_at(VS, D, state, cfg)
        tree = BVHTree.FromPolygons([tuple(v) for v in V], tris)
        rows["%g" % val] = _stats(g for g, m in zip(_signed_gaps(G, tree), mask) if m)
    return {"key": key, "target": target.name, "core_verts": sum(mask), "scan": rows}


def _source_geometry(source):
    sk = source.data.shape_keys
    if source.type != 'MESH' or not sk:
        raise TransferError("source %r must be a mesh carrying shape keys" % source.name)
    kb = sk.key_blocks; M = source.matrix_world
    B = [M @ v.co for v in kb["Basis"].data]
    D = {k.name: [(M @ a.co) - b for a, b in zip(k.data, B)] for k in kb if k.name != "Basis"}
    return B, D, _tris(source.data)


def _body_now(B, D, source):
    """The body as it evaluates now: Basis plus every live key value."""
    V = list(B)
    for k in source.data.shape_keys.key_blocks:
        if k.name == "Basis" or abs(k.value) < 1e-9:
            continue
        V = [v + float(k.value) * d for v, d in zip(V, D[k.name])]
    return V


def _body_at(VS, D, state, cfg):
    """The body at configuration ``cfg``, from its current shape ``VS`` (absent keys keep
    the body's state)."""
    V = list(VS)
    for k, amt in cfg.items():
        delta = amt - state.get(k, 0.0)
        if abs(delta) < 1e-9:
            continue
        Dk = D[k]
        V = [v + delta * d for v, d in zip(V, Dk)]
    return V


def transfer_shapekeys(source, targets: Sequence, keys: Sequence[str] = (),
                       authored: Optional[Dict[str, float]] = None, *,
                       seat=True, footprint=0.001, falloff=4.0, smooth=0,
                       heal=0.020, tear=1.0, whatif=False) -> Dict[str, Any]:
    """Seat ``targets`` on ``source`` and add ``keys`` to them. See the module docstring.

    ``authored`` maps key name to the value the garment was cut against; a key absent from
    it is taken as 0 (the vendor-neutral body). Every key named in ``authored`` or ``keys``
    must exist on ``source``. A target already carrying a key in ``keys`` is refused by
    name — a real vendor key is never overwritten; delete it first if it is a synthetic one.
    ``smooth`` Laplacian-smooths each transferred displacement field over the garment's
    own edges that many times, inside the footprint, trading gap fidelity for a band that
    does not buckle where the mapping crosses a fold. ``whatif`` measures and reports,
    writing nothing.

    Returns a report: per target the gap statistics before and after over the union
    footprint, the fidelity of each vertex's gap to its authored gap, leakage outside the
    footprint, and what was written."""
    authored = {k: float(v) for k, v in (authored or {}).items()}
    keys = list(keys)
    if not keys and not authored:
        raise TransferError("nothing to do: name keys to add (--keys) or an authored "
                            "configuration to seat from (--authored)")
    if not keys and not seat:
        raise TransferError("nothing to do: seat=False needs keys to add")
    B, D, tris = _source_geometry(source)
    state = source_state(source)
    for k in list(authored) + keys:
        if k not in D:
            raise TransferError("shape key %r not found on source %r" % (k, source.name))
    involved = []
    for k in keys + list(authored):
        if k not in involved:
            involved.append(k)
    # Keys whose seat is nonzero: authored differs from the body's state.
    seat_amount = {k: state.get(k, 0.0) - authored.get(k, 0.0) for k in involved}
    for k in involved:
        if k in keys and not (0.0 <= state.get(k, 0.0) <= 1.0):
            raise TransferError("source state of %r is %g, outside the 0..1 a live key value "
                                "can hold; seat without adding it" % (k, state.get(k, 0.0)))
    target_baked = {}
    for t in targets:
        if t.type != 'MESH':
            raise TransferError("target %r is not a mesh" % t.name)
        if t is source or t.data is source.data:
            raise TransferError("target %r is the source body; the source is never written" % t.name)
        if scene_utils.is_linked(t) or not scene_utils.is_editable(t):
            raise TransferError("target %r is linked library data and cannot be written" % t.name)
        if t.data.users > 1:
            raise TransferError("target %r shares its mesh data with %d other user(s); make it "
                                "single-user first" % (t.name, t.data.users - 1))
        if t.data.shape_keys:
            if not t.data.shape_keys.use_relative:
                raise TransferError("target %r carries absolute shape keys; this door writes "
                                    "relative ones" % t.name)
            present = [k for k in keys if k in t.data.shape_keys.key_blocks]
            if present:
                raise TransferError("target %r already carries %s — a vendor key is never "
                                    "overwritten; delete a synthetic one first"
                                    % (t.name, ", ".join(present)))
        target_baked[t.name] = _baked_map(t)  # validated before anything moves

    cfg_authored = {k: authored.get(k, 0.0) for k in involved}
    VS = _body_now(B, D, source)
    VA = _body_at(VS, D, state, cfg_authored)
    tree_full = BVHTree.FromPolygons([tuple(v) for v in VA], tris)
    tris_full = tris
    B, D, tris, keep, dropped = _moving_islands(B, D, tris, involved, footprint)
    if keep is not None:
        VS = [VS[i] for i in sorted(keep, key=keep.get)]
        VA = [VA[i] for i in sorted(keep, key=keep.get)]
        static_tri = set(i for i, t in enumerate(tris_full) if t[0] not in keep)
    else:
        static_tri = set()
    tree_auth = BVHTree.FromPolygons([tuple(v) for v in VA], tris)
    tree_now = BVHTree.FromPolygons([tuple(v) for v in VS], tris)
    foot_tri = {k: [max(D[k][j].length for j in t) > footprint for t in tris] for k in involved}

    # One Surface Deform binding on the authored body, one full-key evaluation per key.
    me = bpy.data.meshes.new("avatarprep_transfer_src")
    me.from_pydata([tuple(v) for v in VA], [], tris); me.update()
    TA = bpy.data.objects.new("avatarprep_transfer_src", me)
    bpy.context.scene.collection.objects.link(TA)
    TA.shape_key_add(name="Basis")
    # One evaluation moves the body from the authored configuration to its state (the
    # seat); one per added key moves it from its state by that key's full delta. Summing
    # per-key evaluations at their extremes instead crumples a cup band: Surface Deform
    # is not linear in the deformation, so the seat is evaluated at the configuration
    # that ships and the keys are linearised around it.
    k_seat = TA.shape_key_add(name="seat")
    for i, v in enumerate(VS):
        k_seat.data[i].co = v
    kfull = {}
    for k in keys:
        kb = TA.shape_key_add(name=k)
        for i, (v, d) in enumerate(zip(VS, D[k])):
            kb.data[i].co = v + d
        kfull[k] = kb
    report = {"source": source.name, "keys_added": keys, "authored": cfg_authored,
              "state": {k: state.get(k, 0.0) for k in involved},
              "seat": {k: round(v, 6) for k, v in seat_amount.items()} if seat else {},
              "static_islands_dropped": dropped,
              "targets": []}
    scratch = [TA]
    try:
        for t in targets:
            G = [t.matrix_world @ v.co for v in t.data.vertices]
            gm = bpy.data.meshes.new("avatarprep_transfer_tgt")
            gm.from_pydata([tuple(v) for v in G], [], []); gm.update()
            W = bpy.data.objects.new("avatarprep_transfer_tgt", gm)
            bpy.context.scene.collection.objects.link(W); scratch.append(W)
            for kb in list(kfull.values()) + [k_seat]:
                kb.value = 0.0
            mod = W.modifiers.new("SD", 'SURFACE_DEFORM'); mod.target = TA; mod.falloff = falloff
            scene_utils.op_override(bpy.ops.object.surfacedeform_bind,
                                    {'active_object': W, 'object': W}, modifier="SD")
            if not mod.is_bound:
                raise TransferError("Surface Deform could not bind %r to %r" % (t.name, source.name))
            near = []
            for c in G:
                loc, n, fi, dist = tree_auth.find_nearest(c)
                near.append(None if loc is None or dist > 0.08 else fi)
            over_static = 0
            if static_tri:
                for c in G:
                    loc, n, fi, dist = tree_full.find_nearest(c)
                    if loc is not None and dist <= 0.08 and fi in static_tri:
                        over_static += 1

            def _evaluate():
                bpy.context.view_layer.update()
                ev = W.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
                out = [Vector(v.co) for v in ev.vertices]
                W.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
                return out

            adjacency = [[] for _ in G]
            for e in t.data.edges:
                a, b = e.vertices
                adjacency[a].append(b); adjacency[b].append(a)
            healed = [False] * len(G)

            def _heal(field):
                # Torn edges: the mapping moves the two ends apart by more than the edge is long.
                torn = set()
                for i, nbs in enumerate(adjacency):
                    for j in nbs:
                        if j > i:
                            L = (G[i] - G[j]).length
                            if L > 1e-9 and (field[i] - field[j]).length > tear * L:
                                torn.add(i); torn.add(j)
                if not torn:
                    return field, 0
                import heapq
                dist = {i: 0.0 for i in torn}
                heap = [(0.0, i) for i in torn]
                while heap:
                    d, i = heapq.heappop(heap)
                    if d > dist.get(i, 1e9):
                        continue
                    for j in adjacency[i]:
                        nd = d + (G[i] - G[j]).length
                        if nd <= heal and nd < dist.get(j, 1e9):
                            dist[j] = nd; heapq.heappush(heap, (nd, j))
                free = sorted(dist)
                field = list(field)
                for _ in range(500):
                    worst = 0.0
                    for i in free:
                        nbs = adjacency[i]
                        if not nbs:
                            continue
                        acc = Vector((0, 0, 0)); wsum = 0.0
                        for j in nbs:
                            w = 1.0 / max((G[i] - G[j]).length, 1e-6)
                            acc += w * field[j]; wsum += w
                        new = acc / wsum
                        worst = max(worst, (new - field[i]).length)
                        field[i] = new
                    if worst < 1e-6:
                        break
                # A healed vertex interpolates between skin that moved and skin that did not,
                # so it can land inside the body; hold it at the gap it had before the move.
                pushed = 0
                for i in free:
                    healed[i] = True
                    loc0, n0, fi0, d0 = tree_auth.find_nearest(G[i])
                    if loc0 is None or d0 > 0.08:
                        continue
                    gap0 = max((G[i] - loc0).dot(n0), 0.0)
                    p = G[i] + field[i]
                    loc, n, fi, d = tree_now.find_nearest(p)
                    if loc is None or d > 0.08:
                        continue
                    gap = (p - loc).dot(n)
                    if gap < gap0:
                        field[i] = field[i] + n * (gap0 - gap); pushed += 1
                return field, len(torn) // 2

            def _smooth(field, mask):
                for _ in range(int(smooth)):
                    nxt = []
                    for i, f in enumerate(field):
                        nb = adjacency[i]
                        if not mask[i] or not nb:
                            nxt.append(f); continue
                        nxt.append(0.5 * f + 0.5 * sum((field[j] for j in nb), Vector((0, 0, 0))) / len(nb))
                    field = nxt
                return field

            seat_keys = [k for k in involved if abs(seat_amount[k]) > 1e-9]
            seat_mask = [fi is not None and any(foot_tri[k][fi] for k in seat_keys) for fi in near]
            k_seat.value = 1.0
            at_state = _evaluate()
            k_seat.value = 0.0
            T_seat = [(p - c) if m else Vector((0, 0, 0)) for p, c, m in zip(at_state, G, seat_mask)]
            torn_total = 0
            if heal and seat:
                T_seat, torn = _heal(T_seat); torn_total += torn
            if smooth:
                T_seat = _smooth(T_seat, seat_mask)
            T = {}
            for k in keys:
                kfull[k].value = 1.0  # its coords already sit at the state plus the key
                at_key = _evaluate()
                kfull[k].value = 0.0
                mask = [fi is not None and foot_tri[k][fi] for fi in near]
                T[k] = [(a - b) if m else Vector((0, 0, 0)) for a, b, m in zip(at_key, at_state, mask)]
                if heal:
                    T[k], torn = _heal(T[k]); torn_total += torn
                if smooth:
                    T[k] = _smooth(T[k], mask)
            union = [(any(foot_tri[k][fi] for k in involved) if fi is not None else False) or h
                     for fi, h in zip(near, healed)]
            # Seat: Basis move for keys not added; live value for keys added.
            # Seat per key: an added key carries the body's state as its live value, so only
            # the authored offset folds into Basis (-authored_k); a key not added folds the
            # whole seat (state_k - authored_k). Either fold is recorded in avatarprep_baked.
            live = {}
            fold = {}
            for k in involved:
                if not seat:
                    if k in keys:
                        live[k] = 0.0
                    continue
                if k in keys:
                    live[k] = state.get(k, 0.0)
                    amt = -authored.get(k, 0.0)
                else:
                    amt = seat_amount[k]
                if abs(amt) > 1e-9:
                    fold[k] = amt
            # The garment evaluates to the single-shot seat exactly: Basis carries whatever
            # the added keys' live values do not.
            after = [c + ts for c, ts in zip(G, T_seat)] if seat else list(G)
            basis_move = [ts if seat else Vector((0, 0, 0)) for ts in T_seat]
            for k, lv in live.items():
                if abs(lv) > 1e-9:
                    basis_move = [bm - lv * tk for bm, tk in zip(basis_move, T[k])]
            ev_now = t.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
            G_now = [t.matrix_world @ v.co for v in ev_now.vertices] if len(ev_now.vertices) == len(G) else G
            g_auth = _signed_gaps(G, tree_auth)
            g_before = _signed_gaps(G_now, tree_now)
            g_after = _signed_gaps(after, tree_now)
            fid = sorted(abs(a - b) * 1000 for a, b, u in zip(g_after, g_auth, union)
                         if u and a is not None and b is not None)
            total_move = [a - c for a, c in zip(after, G)]
            row = {"target": t.name, "verts": len(G), "footprint_verts": sum(union),
                   "before": _stats(g for g, u in zip(g_before, union) if u),
                   "after": _stats(g for g, u in zip(g_after, union) if u),
                   "fidelity_to_authored_mm": {"p95": round(fid[int(len(fid) * 0.95)], 2),
                                               "max": round(fid[-1], 2)} if fid else {},
                   "leak_outside_footprint": sum(1 for m, u in zip(total_move, union)
                                                 if not u and m.length > 0.0005),
                   "verts_over_static_island": over_static,
                   "torn_edges": torn_total, "healed_verts": sum(healed),
                   "max_move_mm": round(max(m.length for m in total_move) * 1000, 2) if G else 0.0,
                   "live_values": live,
                   "baked_written": {}}
            if not whatif:
                Mi = t.matrix_world.inverted()
                if t.data.shape_keys is None:
                    t.shape_key_add(name="Basis")
                kbs = t.data.shape_keys.key_blocks
                basis = kbs["Basis"]
                # Move Basis; relative keys follow (Blender keeps each key's delta from Basis).
                moved = any(bm.length > 0 for bm in basis_move)
                if moved:
                    old = [Vector(p.co) for p in basis.data]
                    # Basis key data and the mesh's own vertex array are separate storage;
                    # shape_key_add(from_mix=False) copies the latter, so keep both moved.
                    for i, bm in enumerate(basis_move):
                        if bm.length > 0:
                            basis.data[i].co = Mi @ ((t.matrix_world @ old[i]) + bm)
                            t.data.vertices[i].co = basis.data[i].co
                    for kb in kbs:
                        if kb.name == "Basis":
                            continue
                        for i, bm in enumerate(basis_move):
                            if bm.length > 0:
                                kb.data[i].co = Mi @ ((t.matrix_world @ Vector(kb.data[i].co)) + bm)
                    bmap = dict(target_baked[t.name])
                    for k, amt in fold.items():
                        cum = bmap.get(k, 0.0) + amt
                        bmap[k] = 0.0 if abs(cum) < 1e-6 else cum
                        row["baked_written"][k] = bmap[k]
                    t[scene_utils.STAMP_BAKED] = bmap
                for k in keys:
                    kb = t.shape_key_add(name=k, from_mix=False)
                    for i, tk in enumerate(T[k]):
                        if tk.length > 0:
                            kb.data[i].co = Mi @ ((t.matrix_world @ Vector(kb.data[i].co)) + tk)
                    kb.value = live.get(k, 0.0)
                t.data.update()
            report["targets"].append(row)
    finally:
        for o in scratch:
            data = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            bpy.data.meshes.remove(data)
    return report
