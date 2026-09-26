"""Differential acceptance: the weight-transfer core against the installed Robust Weight Transfer add-on.

Local only, outside ``tests/run_all.py``. Run:

    AVATARPREP_RWT_ADDON=<add-on dir> blender --background --factory-startup \
        --python tests/acceptance/diff_addon.py [-- --report <json>]

The add-on (GPL) is executed from its install folder, never copied: its ``util`` and
``weighttransfer`` modules load by path with its own ``deps`` on ``sys.path``. The reference
pipeline is the one the venue's recorded transfers ran: the add-on's matching, point-cloud
inpaint, smoothing and smoothed influence limit, composed with the write policy the core
now owns (garment bones kept, body share ``1 - p``). Both sides run on the same in-memory
fixture (``tests/_weight_fixture.py``) in one session, one case per knob family.

Per case it prints: whether the evaluated positions and normals are identical; every vertex
whose matched/unmatched verdict differs, with its distance and angle margins; the largest
weight difference on commonly matched vertices; the inpaint solver difference on identical
inputs; and the final per-vertex max |dw| on vertices neither limit touched, beside the
capped-set delta (ours: rows cut to their allowance; the add-on's: rows its dilated
limit mask or hard cap changed). Pass is identical positions, no flips and
max |dw| <= 1e-5 off the capped sets. Three design differences are measured, never
tolerated: the core clips to [0, 1] before smoothing where the add-on clipped at 0 after it;
its smoothing walk is independent of seed order where the add-on's is not (the smoothing-set
line); and it limits per vertex where the add-on's mask dilates into neighbours (the
capped-set delta). A ``--smooth`` case therefore diverges on vertices the two sets or the
clip order separate.
"""
import importlib.util
import json
import os
import site
import sys
import sysconfig

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
REPO = os.path.dirname(TESTS)
TOKEN = "DIFF_ADDON"
TOL = 1e-5
EPS = 1e-4
LIMIT = 4


def load_addon(root):
    for name in ("util.py", "weighttransfer.py"):
        if not os.path.isfile(os.path.join(root, name)):
            print("%s FAIL: %s has no %s; AVATARPREP_RWT_ADDON must name the add-on folder" % (TOKEN, root, name))
            sys.exit(2)
    deps = os.path.join(root, "deps")
    site.addsitedir(sysconfig.get_paths(sysconfig.get_preferred_scheme("user"), vars={"userbase": deps})["purelib"])
    mods = []
    for name in ("util", "weighttransfer"):
        spec = importlib.util.spec_from_file_location("rwt_" + name, os.path.join(root, name + ".py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        mods.append(m)
    return mods


# --- the reference pipeline: add-on functions under the venue's composition --------------------

def ref_arrays(util, WT, source, target):
    """Add-on arrays for the source (copy, Armature modifier off) and target (rest state)."""
    tmp = source.copy()
    tmp.data = source.data.copy()
    bpy.context.scene.collection.objects.link(tmp)
    for m in tmp.modifiers:
        if m.type == 'ARMATURE':
            m.show_viewport = False
    try:
        with WT._rest_state([target], {}):
            dg = bpy.context.evaluated_depsgraph_get()
            dg.update()
            SV, SF, SN = (a.copy() for a in util.get_obj_arrs_world(tmp.evaluated_get(dg)))
            TV, TF, TN = (a.copy() for a in util.get_obj_arrs_world(target.evaluated_get(dg)))
    finally:
        me = tmp.data
        bpy.data.objects.remove(tmp, do_unlink=True)
        bpy.data.meshes.remove(me)
    is_deform = [util.is_vertex_group_deform_bone(source, g.name) for g in source.vertex_groups]
    SW = util.get_groups_arr(source, is_deform)
    return SV, SF, SN, SW, [g.name for g in source.vertex_groups], TV, TF, TN


def ref_one(util, wt, target, SV, SF, SN, SW, TV, TF, TN, case):
    matched, W2 = wt.find_matches_closest_surface(SV, SF, SN, TV, TN, SW, case["max_distance"] ** 2,
                                                  case["normal_angle"], case["flip"])
    T = wt.inpaint(TV, TF, W2, matched, True)
    if case.get("smooth"):
        adj = util.get_mesh_adjacency_matrix_sparse(target.data, include_self=True)
        T = np.asarray(wt.smooth_weigths(TV, T, matched, adj, util.get_mesh_adjacency_list(target.data),
                                         case["smooth"][0], case["smooth"][1], case["max_distance"]))
    return matched, W2, np.clip(np.asarray(T, np.float64), 0, None)


def ref_limit(wt, T, adj, allow):
    T = T.copy()
    T[T <= EPS] = 0
    out = T.copy()
    for L in sorted(set(allow.tolist()) - {0}):
        rows = allow == L
        m = wt.limit_mask(T, adj, limit_num=int(L))
        out[rows] = ((1 - m) * T)[rows]
    out[out <= EPS] = 0
    for i in np.flatnonzero((out > 0).sum(1) > allow):
        keep = np.argsort(out[i])[::-1][:allow[i]]
        row = np.zeros_like(out[i])
        row[keep] = out[i, keep]
        out[i] = row
    changed = np.abs(out - T).max(1) > 0
    return out, changed


def ref_pipeline(util, wt, WT, s, case):
    source, target = s["body"], s["garment"]
    SV, SF, SN, SW, snames, TV, TF, TN = ref_arrays(util, WT, source, target)
    runs = [ref_one(util, wt, target, SV, SF, SN, SW, TV, TF, TN, case)]
    xcols = []
    if case.get("exclude"):
        arm = source.modifiers["Armature"].object
        xb = WT.excluded_bones(arm, case["exclude"])
        xcols = [j for j, n in enumerate(snames) if n in xb]
        tot = SW.sum(1)
        share = np.where(tot > 0, SW[:, xcols].sum(1) / np.where(tot > 0, tot, 1), 0)
        SFx = SF[~(share > case["exclude_max"])[SF].any(1)]
        runs.append(ref_one(util, wt, target, SV, SFx, SN, SW, TV, TF, TN, case))
    adj = util.get_mesh_adjacency_matrix_sparse(target.data, include_self=True)
    if case.get("blend"):
        import scipy.sparse
        Ts = [WT._normalised(r[2]) for r in runs]
        frame = WT.body_frame(source.modifiers["Armature"].object)
        m = WT.blend_mix(TV, frame, *case["blend"], lateral=case.get("lateral"))
        T = m[:, None] * Ts[1] + (1 - m[:, None]) * Ts[0]
        n, alpha, radius = case["blend_smooth"]
        band, _ = WT.leg_hole_band(WT.boundary_loops(target.data), TV, Ts[0][:, xcols].sum(1),
                                   case["exclude_max"], radius)
        A = scipy.sparse.csr_array(adj, dtype=np.float64)
        S = scipy.sparse.diags(1 / np.asarray(A.sum(1)).ravel()) @ A
        T0 = T.copy()
        for _ in range(n):
            T = (1 - alpha) * T + alpha * (S @ T)
            T[~band] = T0[~band]
        matched = runs[0][0] | runs[1][0]
    else:
        matched, _, T = runs[-1]
    return {"SV": SV, "SN": SN, "TV": TV, "TN": TN, "snames": snames, "matched": matched,
            "W2": runs[-1][1] if not case.get("blend") else runs[0][1], "T": T, "adj": adj,
            "runs": runs, "SF": SF}


def write_policy(target, source_deform, snames, T, limiter, mask=None):
    """The shared write policy on a limited transfer: new body weights per name, by vertex."""
    arm = target.modifiers["Armature"].object
    garment = {b.name for b in arm.data.bones if b.use_deform} - source_deform
    gnames = [g.name for g in target.vertex_groups]
    W0 = np.zeros((len(target.data.vertices), len(gnames)), np.float32)
    for i, v in enumerate(target.data.vertices):
        for g in v.groups:
            W0[i, g.group] = g.weight
    gcols = [i for i, n in enumerate(gnames) if n in garment]
    p = W0[:, gcols].sum(1).astype(np.float64)
    allow = np.clip(LIMIT - (W0[:, gcols] > 0).sum(1), 0, LIMIT)
    active = (p < 1 - EPS) & (allow > 0)
    Tl, capped = limiter(T, allow)
    s = Tl.sum(1)
    new = np.where(s[:, None] > 0, Tl / np.where(s > 0, s, 1)[:, None], 0) * (1 - p)[:, None]
    final = {n: W0[:, j].copy() for j, n in enumerate(gnames)}
    for j, n in enumerate(snames):
        col = final.setdefault(n, np.zeros(len(p), np.float32))
        col[active] = new[active, j].astype(np.float32)
    for n in list(final):
        if n in source_deform and n not in snames:
            final[n][active] = 0
    return final, active, capped


# --- comparison ------------------------------------------------------------------------------------

def puff(s):
    """Flare the band's and left cuff's lower rows outward past the match distance, so a
    contiguous region is inpainted from matched neighbours (the fixture otherwise inpaints only
    detached islands, which reads nothing of the solve against a matched border)."""
    import math
    for i, v in enumerate(s["garment"].data.vertices):
        t, (x, y, z) = s["tags"][i], v.co
        cx, amount = (0.0, 0.07 * (0.94 - z) / 0.04) if t == "band" else (0.08, 0.07 * (0.72 - z) / 0.06)
        if t in ("band", "cuff_L") and amount > 0:
            r = math.hypot(x - cx, y)
            v.co.x = cx + (x - cx) * (r + amount) / r
            v.co.y = y * (r + amount) / r
    s["garment"].data.update()
    return s


def _build(F, case):
    s = F.build()
    return puff(s) if case.get("puff") else s


def run_case(util, wt, WT, F, case):
    out = {"case": case["name"]}
    # ours, end to end
    s = _build(F, case)
    kw = {k: case[k] for k in ("max_distance", "normal_angle", "flip", "smooth") if k in case}
    for k, ck in (("exclude", "exclude"), ("exclude_max", "exclude_max"), ("blend", "blend"),
                  ("blend_lateral", "lateral"), ("blend_smooth", "blend_smooth")):
        if ck in case:
            kw[k] = case[ck]
    row = WT.transfer_weights(s["body"], [s["garment"]], **kw)["targets"][0]
    ours_final = F.weights_by_name(s["garment"])
    ours_matched = row["matched_mask"]

    # ours, stage arrays on a fresh build (for positions, match and capped set)
    s = _build(F, case)
    src = WT.source_arrays(s["body"], {})
    with WT._rest_state([s["garment"]], {}):
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        ev, me = WT._evaluated(s["garment"], dg)
        TV, _, TN = WT.mesh_arrays(ev, me)
        ev.to_mesh_clear()
    SF = src["F"]
    if case.get("exclude") and not case.get("blend"):
        xb = WT.excluded_bones(src["armature"], case["exclude"])
        SF, _ = WT.narrow_source(src["F"], src["W"], src["names"], xb, case["exclude_max"])
    m = WT.match(src["V"], SF, src["N"], src["W"], TV, TN, case["max_distance"], case["normal_angle"], case["flip"])

    # reference on the same build
    ref = ref_pipeline(util, wt, WT, s, case)
    out["positions_identical"] = bool(np.array_equal(src["V"], ref["SV"]) and np.array_equal(TV, ref["TV"]))
    out["normals_identical"] = bool(np.array_equal(src["N"], ref["SN"]) and np.array_equal(TN, ref["TN"]))
    out["max_abs_dpos"] = float(max(np.abs(src["V"] - ref["SV"]).max(), np.abs(TV - ref["TV"]).max()))

    flips = np.flatnonzero(ours_matched != ref["matched"])
    out["flipped_matches"] = [{"vertex": int(i), "ours": bool(ours_matched[i]), "addon": bool(ref["matched"][i]),
                               "distance_margin_m": round(float(case["max_distance"] - m["distance"][i]), 9),
                               "angle_margin_deg": round(float(case["normal_angle"] - min(m["angle"][i], 180 - m["angle"][i]
                                                                                        if case["flip"] else 1e9)), 6)}
                              for i in flips]
    both = m["matched"] & ref["runs"][0 if case.get("blend") else -1][0]
    out["max_dw_matched_interp"] = float(np.abs(m["weights"][both] - ref["runs"][0 if case.get("blend") else -1][1][both]).max()) if both.any() else 0.0

    # inpaint solver on identical inputs
    r_matched, r_W2 = ref["runs"][0][0], ref["runs"][0][1]
    ours_T = WT.inpaint(ref["TV"], r_W2, r_matched)
    addon_T = wt.inpaint(ref["TV"], None, r_W2, r_matched, True)
    out["max_dw_inpaint_solver"] = float(np.abs(ours_T - addon_T).max())
    out["clip_above_1_vertices"] = int((ours_T > 1.0).any(1).sum())
    if case.get("smooth"):
        # The add-on's smoothing set, read off its own function: one full-strength pass over
        # random weights changes exactly the rows it smooths.
        n_, a_ = case["smooth"]
        adj_ = util.get_mesh_adjacency_matrix_sparse(s["garment"].data, include_self=True)
        R = np.random.default_rng(0).random((len(TV), 3))
        R2 = np.asarray(wt.smooth_weigths(ref["TV"], R, r_matched, adj_,
                                          util.get_mesh_adjacency_list(s["garment"].data), 1, 1.0,
                                          case["max_distance"]))
        addon_set = np.abs(R2 - R).max(1) > 0
        _, ours_set = WT.smooth(TV, R, r_matched, WT.adjacency(s["garment"].data), 1, 1.0, case["max_distance"])
        out["smooth_set_ours"] = int(ours_set.sum())
        out["smooth_set_addon"] = int(addon_set.sum())
        out["smooth_set_only_ours"] = int((ours_set & ~addon_set).sum())
        out["smooth_set_only_addon"] = int((addon_set & ~ours_set).sum())

    # final weights: ours end to end vs the reference composed through the same write policy
    deform = src["deform"]
    ref_final, active, ref_changed = write_policy(s["garment"], deform, ref["snames"], ref["T"],
                                                  lambda T, allow: ref_limit(wt, T, ref["adj"], allow))
    _, _, ours_capped = write_policy(s["garment"], deform, src["names"], _ours_T_before_limit(WT, src, TV, TN, s, case),
                                     lambda T, allow: WT.limit(T, allow))
    n = len(TV)
    dw = np.zeros(n)
    for name in set(ref_final) | set(ours_final):
        a = np.asarray(ours_final.get(name, [0.0] * n), np.float64)
        b = np.asarray(ref_final.get(name, np.zeros(n)), np.float64)
        dw = np.maximum(dw, np.abs(a - b))
    capped = (ours_capped | ref_changed) & active
    free = active & ~capped
    if len(flips):
        free &= ~np.isin(np.arange(n), flips)
    out["active"] = int(active.sum())
    out["max_dw_noncapped"] = float(dw[free].max()) if free.any() else 0.0
    out["worst_noncapped_vertex"] = int(np.flatnonzero(free)[np.argmax(dw[free])]) if free.any() else None
    out["capped_ours"] = int((ours_capped & active).sum())
    out["capped_addon"] = int((ref_changed & active).sum())
    out["capped_only_ours"] = int((ours_capped & ~ref_changed & active).sum())
    out["capped_only_addon"] = int((ref_changed & ~ours_capped & active).sum())
    out["max_dw_capped"] = float(dw[capped].max()) if capped.any() else 0.0
    out["inpainted_active"] = int((active & ~ours_matched).sum())
    out["max_dw_inpainted_noncapped"] = float(dw[free & ~ours_matched].max()) if (free & ~ours_matched).any() else None
    out["pass"] = bool(out["positions_identical"] and not len(flips) and out["max_dw_noncapped"] <= TOL)
    return out


def _ours_T_before_limit(WT, src, TV, TN, s, case):
    """The core's transfer before step 6, rebuilt from its public stages, for its capped set."""
    A = WT.adjacency(s["garment"].data)
    knobs = {"max_distance": case["max_distance"], "normal_angle": case["normal_angle"],
             "flip": case["flip"], "smooth": case.get("smooth")}
    if not case.get("exclude"):
        return WT._transfer(TV, TN, src, src["F"], knobs, A)[1]
    xb = WT.excluded_bones(src["armature"], case["exclude"])
    SFx, _ = WT.narrow_source(src["F"], src["W"], src["names"], xb, case["exclude_max"])
    if not case.get("blend"):
        return WT._transfer(TV, TN, src, SFx, knobs, A)[1]
    Tw = WT._normalised(WT._transfer(TV, TN, src, src["F"], knobs, A)[1])
    Tn = WT._normalised(WT._transfer(TV, TN, src, SFx, knobs, A)[1])
    frame = WT.body_frame(src["armature"])
    mix = WT.blend_mix(TV, frame, *case["blend"], lateral=case.get("lateral"))
    T = mix[:, None] * Tn + (1 - mix[:, None]) * Tw
    xcols = [j for j, n in enumerate(src["names"]) if n in xb]
    n, alpha, radius = case["blend_smooth"]
    band, nl = WT.leg_hole_band(WT.boundary_loops(s["garment"].data), TV, Tw[:, xcols].sum(1),
                                case["exclude_max"], radius)
    return WT.laplacian_steps(T, A, band, n, alpha) if nl else T


CASES = [
    {"name": "default", "max_distance": 0.05, "normal_angle": 30.0, "flip": True},
    {"name": "no_flip", "max_distance": 0.05, "normal_angle": 30.0, "flip": False},
    {"name": "smooth_4_0.2", "max_distance": 0.05, "normal_angle": 30.0, "flip": True, "smooth": (4, 0.2)},
    {"name": "exclude_legs", "max_distance": 0.05, "normal_angle": 30.0, "flip": True,
     "exclude": ["upper*leg*"], "exclude_max": 0.35},
    {"name": "blend_back_lateral_smooth", "max_distance": 0.05, "normal_angle": 30.0, "flip": True,
     "exclude": ["upper*leg*"], "exclude_max": 0.35, "blend": ("back", 0.0, 0.04),
     "lateral": (0.06, 0.04), "blend_smooth": (2, 0.5, 0.04)},
    {"name": "puffed", "max_distance": 0.05, "normal_angle": 30.0, "flip": True, "puff": True},
    {"name": "puffed_smooth_4_0.2", "max_distance": 0.05, "normal_angle": 30.0, "flip": True, "puff": True,
     "smooth": (4, 0.2)},
    {"name": "puffed_blend", "max_distance": 0.05, "normal_angle": 30.0, "flip": True, "puff": True,
     "exclude": ["upper*leg*"], "exclude_max": 0.35, "blend": ("back", 0.0, 0.04),
     "lateral": (0.06, 0.04), "blend_smooth": (2, 0.5, 0.04)},
]


def main():
    root = os.environ.get("AVATARPREP_RWT_ADDON")
    if not root:
        print("%s FAIL: set AVATARPREP_RWT_ADDON to the installed Robust Weight Transfer add-on folder" % TOKEN)
        sys.exit(2)
    site.addsitedir(os.environ.get("AVATARPREP_DEPS") or os.path.join(REPO, "deps"))
    util, wt = load_addon(root)
    for p in (REPO, TESTS):
        if p not in sys.path:
            sys.path.insert(0, p)
    import _weight_fixture as F
    from avatarprep.core import weight_transfer as WT
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    results = [run_case(util, wt, WT, F, c) for c in CASES]
    for r in results:
        print("%s %s: pass=%s positions_identical=%s normals_identical=%s flips=%d interp=%.2e solver=%.2e "
              "clip>1=%d noncapped max|dw|=%.2e (vertex %s) capped ours=%d addon=%d only-ours=%d only-addon=%d "
              "capped max|dw|=%.3f inpainted-written=%d (noncapped max|dw| %s)"
              % (TOKEN, r["case"], r["pass"], r["positions_identical"], r["normals_identical"],
                 len(r["flipped_matches"]), r["max_dw_matched_interp"], r["max_dw_inpaint_solver"],
                 r["clip_above_1_vertices"], r["max_dw_noncapped"], r["worst_noncapped_vertex"],
                 r["capped_ours"], r["capped_addon"], r["capped_only_ours"], r["capped_only_addon"],
                 r["max_dw_capped"], r["inpainted_active"], r["max_dw_inpainted_noncapped"]))
        if "smooth_set_ours" in r:
            print("%s   smoothing set ours=%d addon=%d only-ours=%d only-addon=%d"
                  % (TOKEN, r["smooth_set_ours"], r["smooth_set_addon"], r["smooth_set_only_ours"],
                     r["smooth_set_only_addon"]))
        for f in r["flipped_matches"]:
            print("%s   flip %r" % (TOKEN, f))
    if "--report" in argv:
        path = os.path.abspath(argv[argv.index("--report") + 1])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print("%s report -> %s" % (TOKEN, path))
    print("%s %s" % (TOKEN, "OK" if all(r["pass"] for r in results) else "DIVERGES (see the lines above)"))


if __name__ == "__main__":
    sys.path.insert(0, TESTS)
    from _harness import run
    run(main, TOKEN)
