"""Reference-verification harness for the 'Custom Plum' chain (Task 10).

Custom Plum = Plum -> Chiffon -> Custom Chiffon (two edges). This harness imports
the vendor Plum FBX, applies edges/plum-to-chiffon.json then
edges/chiffon-to-custom_chiffon.json via avatarprep.core.proportions.apply_proportion_edge, then
compares the resulting armature's bone REST positions (bone heads, in WORLD space
metres) against the hand-made BasePlum.blend reference for every commonly-named
bone. World space is used because the vendor import and the hand-made reference may
carry different armature object matrices (the vendor rig is 0.01-scaled, rotated);
world position is the frame-independent ground truth (Shinano hit 0.00000 m here).

Also reports body-mesh bbox height/width vs BasePlum and renders ortho FRONT/SIDE
of the result to a scratch dir.

Run:
  blender --background --factory-startup --python tests/verify_proportions_plum.py -- \
      --fbx <vendor_Plum.fbx> --ref <BasePlum.blend> \
      --p1 <plum-to-chiffon.json> --p2 <chiffon-to-custom_chiffon.json> --out <scratch_dir>

Prints VERIFY: lines; exits 0 always (measurement/report harness, not a gate).
"""
import os
import sys
import argparse

import bpy
from mathutils import Vector

BODY_MESH = "body_base"  # nude body skin in both vendor + ref (case-insensitive)


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="verify_proportions_plum")
    p.add_argument("--fbx", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--p1", required=True, help="plum-to-chiffon edge")
    p.add_argument("--p2", required=True, help="chiffon-to-custom_chiffon edge")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


# Structural: a fresh --background --python process has no repo path; this must
# precede any shared import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cli._common import enable_avatarprep


def _clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def _world_bbox(meshes):
    lo = Vector((1e18, 1e18, 1e18))
    hi = Vector((-1e18, -1e18, -1e18))
    for m in meshes:
        for c in m.bound_box:
            w = m.matrix_world @ Vector(c)
            for k in range(3):
                lo[k] = min(lo[k], w[k])
                hi[k] = max(hi[k], w[k])
    return lo, hi


def _find_body(meshes):
    for m in meshes:
        if m.type == 'MESH' and m.name.lower() == BODY_MESH:
            return m
    cand = [m for m in meshes if m.type == 'MESH']
    return max(cand, key=lambda m: (_world_bbox([m])[1] - _world_bbox([m])[0]).z) if cand else None


def _eval_bbox(mesh):
    """World-space bbox from the DEPSGRAPH-evaluated mesh (honours shape keys +
    any live modifiers; bound_box is stale after rest-pose foreach_set and ignores
    shape keys, so it is not trustworthy here)."""
    deps = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(deps)
    me = ev.to_mesh()
    mw = mesh.matrix_world
    lo = Vector((1e18, 1e18, 1e18))
    hi = Vector((-1e18, -1e18, -1e18))
    for v in me.vertices:
        w = mw @ v.co
        for k in range(3):
            lo[k] = min(lo[k], w[k]); hi[k] = max(hi[k], w[k])
    ev.to_mesh_clear()
    return lo, hi


def _dims(mesh):
    lo, hi = _eval_bbox(mesh)
    d = hi - lo
    return d.x, d.y, d.z  # W, D, H


def _bone_world_heads(arm):
    """name -> world-space rest head position (metres)."""
    mw = arm.matrix_world
    return {b.name: (mw @ b.head_local) for b in arm.data.bones}


def _setup_render(all_meshes):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 700
    scene.render.resolution_y = 1000
    scene.render.film_transparent = False
    lo, hi = _world_bbox(all_meshes)
    center = (lo + hi) * 0.5
    height = (hi.z - lo.z)
    cam_data = bpy.data.cameras.new("VERIFY_Cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = height * 1.15
    cam = bpy.data.objects.new("VERIFY_Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    return scene, cam, center


def _render_view(scene, cam, center, loc, out_path):
    cam.location = Vector(loc)
    direction = Vector(center) - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    return out_path


def _render_pair(out_dir, prefix, all_meshes):
    scene, cam, center = _setup_render(all_meshes)
    r = 4.0
    front = _render_view(scene, cam, center, (center.x, center.y - r, center.z),
                         os.path.join(out_dir, prefix + "_front.png"))
    side = _render_view(scene, cam, center, (center.x + r, center.y, center.z),
                        os.path.join(out_dir, prefix + "_side.png"))
    return front, side


def main():
    args = _parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    enable_avatarprep()
    from avatarprep.core import import_fbx, scene_utils, proportions

    # ---- Apply the chain to vendor Plum ----
    _clear()
    import_fbx.import_fbx(os.path.abspath(args.fbx))
    arm = scene_utils.find_armature()
    meshes = scene_utils.get_bound_meshes(arm)
    body = _find_body(meshes)
    print("VERIFY: vendor Plum armature=%r body mesh=%r (bones=%d)"
          % (arm.name, body.name, len(arm.data.bones)))

    for tag, prof in (("plum-to-chiffon", args.p1), ("chiffon-to-custom_chiffon", args.p2)):
        rep = proportions.apply_proportion_edge(arm, meshes, os.path.abspath(prof))
        print("VERIFY: applied %-16s -> state=%r scales=%d bakes=%d shapekeys=%d"
              % (tag, rep["state"], rep["scales_applied"], len(rep["bakes"]),
                 len(rep["shapekeys"])))
        for w in rep["warnings"]:
            print("VERIFY:     warn:", w)
        for sk in rep["shapekeys"]:
            print("VERIFY:     shapekey %s.%s=%s (widened=%s)"
                  % (sk["mesh"], sk["key"], sk["value"], sk["widened"]))

    res_heads = _bone_world_heads(arm)
    res_w, res_d, res_h = _dims(body)
    body_name = body.name  # capture before the ref file replaces these objects
    res_armW = [tuple(round(c, 5) for c in row) for row in arm.matrix_world]
    res_meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    front_r, side_r = _render_pair(out_dir, "plum_result", res_meshes)

    # ---- Reference BasePlum ----
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.ref))
    enable_avatarprep()
    from avatarprep.core import scene_utils as su2
    ref_arm = su2.find_armature()
    ref_heads = _bone_world_heads(ref_arm)
    ref_meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    ref_body = _find_body(ref_meshes)
    ref_w, ref_d, ref_h = _dims(ref_body)
    ref_armW = [tuple(round(c, 5) for c in row) for row in ref_arm.matrix_world]
    front_ref, side_ref = _render_pair(out_dir, "plum_ref", ref_meshes)

    # ---- Bone-rest delta report ----
    common = sorted(set(res_heads) & set(ref_heads))
    only_res = sorted(set(res_heads) - set(ref_heads))
    only_ref = sorted(set(ref_heads) - set(res_heads))

    print("=" * 70)
    print("VERIFY: armature world matrix  result=%r" % (res_armW,))
    print("VERIFY: armature world matrix  ref   =%r" % (ref_armW,))
    print("VERIFY: bone counts result=%d ref=%d common=%d (only_result=%d only_ref=%d)"
          % (len(res_heads), len(ref_heads), len(common), len(only_res), len(only_ref)))

    # Focus set: core humanoid body bones present in both.
    focus_names = ["Hips", "Spine", "Chest", "Neck", "Head",
                   "Shoulder.L", "Shoulder.R", "UpperArm.L", "UpperArm.R",
                   "LowerArm.L", "LowerArm.R", "Hand.L", "Hand.R",
                   "UpperLeg.L", "UpperLeg.R", "LowerLeg.L", "LowerLeg.R",
                   "Foot.L", "Foot.R", "Toe.L", "Toe.R",
                   "Breast_L_Root", "Breast_R_Root"]
    focus = [n for n in focus_names if n in res_heads and n in ref_heads]

    def _delta_stats(names):
        if not names:
            return None
        per_axis = {k: [] for k in range(3)}
        dist = []
        rows = []
        for n in names:
            d = res_heads[n] - ref_heads[n]
            dist.append(d.length)
            for k in range(3):
                per_axis[k].append(d[k])
            rows.append((n, d))
        stats = {}
        for k in range(3):
            vals = per_axis[k]
            stats[k] = {
                "mean": sum(vals) / len(vals),
                "absmean": sum(abs(v) for v in vals) / len(vals),
                "max": max(vals), "min": min(vals),
                "spread": max(vals) - min(vals),
            }
        stats["dist_mean"] = sum(dist) / len(dist)
        stats["dist_max"] = max(dist)
        return stats, rows

    for label, names in (("FOCUS humanoid", focus), ("ALL common", common)):
        res = _delta_stats(names)
        if not res:
            print("VERIFY: [%s] no bones" % label)
            continue
        stats, rows = res
        print("-" * 70)
        print("VERIFY: [%s] n=%d  bone-head WORLD delta (result-ref), metres:" % (label, len(names)))
        for k, ax in enumerate("XYZ"):
            s = stats[k]
            print("VERIFY:   %s: mean=%+.5f absmean=%.5f min=%+.5f max=%+.5f spread=%.5f"
                  % (ax, s["mean"], s["absmean"], s["min"], s["max"], s["spread"]))
        print("VERIFY:   |delta| mean=%.5f m  max=%.5f m" % (stats["dist_mean"], stats["dist_max"]))
        # uniform-offset vs position-dependent judgement
        max_spread = max(stats[k]["spread"] for k in range(3))
        if max_spread < 1e-4:
            print("VERIFY:   -> deltas ~UNIFORM (pure offset; spread<0.1mm)")
        elif max_spread < 2e-3:
            print("VERIFY:   -> deltas near-uniform (spread<2mm)")
        else:
            print("VERIFY:   -> deltas POSITION-DEPENDENT (spread=%.4f m) -> not a pure offset"
                  % max_spread)

    # Worst offenders (focus set if available else common)
    pick = focus if focus else common
    stats, rows = _delta_stats(pick)
    rows.sort(key=lambda r: -r[1].length)
    print("-" * 70)
    print("VERIFY: worst-offset bones (focus set):")
    for n, d in rows[:12]:
        print("VERIFY:   %-16s d=(%+.4f,%+.4f,%+.4f) |%.4f| m"
              % (n, d.x, d.y, d.z, d.length))

    print("-" * 70)
    def pct(a, b):
        return 100.0 * (a - b) / b if b else float('nan')
    print("VERIFY: result body H/W/D = %.4f / %.4f / %.4f  (mesh %r)"
          % (res_h, res_w, res_d, body_name))
    print("VERIFY: ref    body H/W/D = %.4f / %.4f / %.4f  (mesh %r)"
          % (ref_h, ref_w, ref_d, ref_body.name))
    print("VERIFY: delta  H/W/D = %+.2f%% / %+.2f%% / %+.2f%%"
          % (pct(res_h, ref_h), pct(res_w, ref_w), pct(res_d, ref_d)))

    print("-" * 70)
    print("VERIFY: only in RESULT (%d): %s" % (len(only_res), only_res[:40]))
    print("VERIFY: only in REF    (%d): %s" % (len(only_ref), only_ref[:40]))
    print("-" * 70)
    print("VERIFY: renders ->")
    for p in (front_r, side_r, front_ref, side_ref):
        print("VERIFY:   ", p)
    print("=" * 70)
    print("VERIFY: DONE")


if __name__ == "__main__":
    main()
