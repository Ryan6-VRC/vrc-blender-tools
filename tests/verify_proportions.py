"""Reference-verification harness for the Shinano proportion profile (Task 10).

Headless. Imports the vendor Shinano FBX, applies profiles/vendor-to-custom_shinano.json via
avatarprep.core.proportions.apply_profile, measures the body mesh world bbox, opens
the hand-made BaseShinano.blend reference and measures ITS body mesh, prints % deltas,
and renders orthographic FRONT/SIDE views of both to a scratch dir for visual review.

Also runs two deferred checks from the Task 3 review:
  (1) the breast median/normal op leaves the breast Y-extent ~unchanged (value Y==1.0),
  (2) a per-bone LOCAL op actually baked into the rest pose (Upper_leg length grew ~Y).

Run:
  blender --background --factory-startup --python tests/verify_proportions.py -- \
      --fbx <vendor_Shinano.fbx> --ref <BaseShinano.blend> \
      --profile <vendor-to-custom_shinano.json> --out <scratch_dir>

Prints VERIFY: lines. Exits 0 when the two deferred checks (breast-Y-unchanged,
leg-length-baked) pass; exits 1 if either lands in REVIEW. The H/W/D deltas and the
renders are for human judgement of overall proportions, not gated.
"""
import os
import sys
import argparse

import bpy
import mathutils
from mathutils import Vector

BODY_MESH = "body_base"  # case-insensitive; the nude body skin in both vendor + ref


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="verify_proportions")
    p.add_argument("--fbx", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def _enable_avatarprep():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import avatarprep
    try:
        avatarprep.register()
    except Exception:
        pass


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
    # fallback: tallest mesh
    cand = [m for m in meshes if m.type == 'MESH']
    return max(cand, key=lambda m: (_world_bbox([m])[1] - _world_bbox([m])[0]).z) if cand else None


def _dims(mesh):
    lo, hi = _world_bbox([mesh])
    d = hi - lo
    return d.x, d.y, d.z  # W, D, H  (X=width, Y=depth, Z=height)


def _setup_render(out_dir, tag, all_meshes):
    scene = bpy.context.scene
    # Workbench: deterministic headless solid shading, ideal for silhouette/proportion review.
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 700
    scene.render.resolution_y = 1000
    scene.render.film_transparent = False

    # Sun (honours the brief; Workbench uses studio light but we add it for parity with EEVEE runs).
    if "VERIFY_Sun" not in bpy.data.objects:
        sd = bpy.data.lights.new("VERIFY_Sun", 'SUN')
        sun = bpy.data.objects.new("VERIFY_Sun", sd)
        bpy.context.collection.objects.link(sun)
        sun.rotation_euler = (0.6, 0.1, 0.3)

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
    scene, cam, center = _setup_render(out_dir, prefix, all_meshes)
    r = 4.0
    front = _render_view(scene, cam, center, (center.x, center.y - r, center.z),
                         os.path.join(out_dir, prefix + "_front.png"))
    side = _render_view(scene, cam, center, (center.x + r, center.y, center.z),
                        os.path.join(out_dir, prefix + "_side.png"))
    return front, side


def _measure_breast_y(meshes):
    """Y-extent of vertices weighted to the breast bones, in world space."""
    return _region_extent(meshes, ("Breast_root", "Breast_1", "Breast_2"))


def _region_extent(meshes, bone_prefixes):
    lo = Vector((1e18, 1e18, 1e18))
    hi = Vector((-1e18, -1e18, -1e18))
    found = False
    for m in meshes:
        gi = {vg.index: vg.name for vg in m.vertex_groups}
        want = {i for i, n in gi.items() if any(n.startswith(p) for p in bone_prefixes)}
        if not want:
            continue
        mw = m.matrix_world
        for v in m.data.vertices:
            if any(g.group in want and g.weight > 0.3 for g in v.groups):
                w = mw @ v.co
                for k in range(3):
                    lo[k] = min(lo[k], w[k]); hi[k] = max(hi[k], w[k])
                found = True
    return (hi - lo) if found else None


def main():
    args = _parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    _enable_avatarprep()
    from avatarprep.core import import_fbx, scene_utils, proportions

    # ---- Deferred check setup: measure breast Y-extent on a clean import first ----
    _clear()
    import_fbx.import_fbx(os.path.abspath(args.fbx))
    arm0 = scene_utils.find_armature()
    meshes0 = scene_utils.get_bound_meshes(arm0)
    breast_before = _measure_breast_y(meshes0)
    upperleg0 = arm0.data.bones["Upper_leg.L"].length

    # ---- Phase A: apply full profile to vendor Shinano, measure + render ----
    _clear()
    import_fbx.import_fbx(os.path.abspath(args.fbx))
    arm = scene_utils.find_armature()
    meshes = scene_utils.get_bound_meshes(arm)
    body = _find_body(meshes)
    print("VERIFY: applying profile %s to %r (body mesh = %r)"
          % (os.path.basename(args.profile), arm.name, body.name))
    report = proportions.apply_profile(arm, meshes, os.path.abspath(args.profile))
    print("VERIFY: apply report state=%r scales_applied=%d bakes=%d shapekeys=%d"
          % (report["state"], report["scales_applied"], len(report["bakes"]),
             len(report["shapekeys"])))
    for w in report["warnings"]:
        print("VERIFY:   warning:", w)

    res_w, res_d, res_h = _dims(body)
    body_name = body.name
    breast_after = _measure_breast_y(meshes)
    upperleg1 = arm.data.bones["Upper_leg.L"].length

    front_r, side_r = _render_pair(out_dir, "shinano", meshes)

    # ---- Phase C: reference BaseShinano ----
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.ref))
    _enable_avatarprep()
    from avatarprep.core import scene_utils as su2
    ref_arm = su2.find_armature()
    ref_meshes = su2.get_bound_meshes(ref_arm)
    ref_body = _find_body([o for o in bpy.data.objects if o.type == 'MESH'])
    ref_w, ref_d, ref_h = _dims(ref_body)
    front_ref, side_ref = _render_pair(out_dir, "shinano_ref",
                                       [o for o in bpy.data.objects if o.type == 'MESH'])

    # ---- Report ----
    def pct(a, b):
        return 100.0 * (a - b) / b if b else float('nan')

    print("=" * 64)
    print("VERIFY: shinano result H/W/D = %.4f / %.4f / %.4f  (mesh %r)"
          % (res_h, res_w, res_d, body_name))
    print("VERIFY: ref BaseShinano H/W/D = %.4f / %.4f / %.4f  (mesh %r)"
          % (ref_h, ref_w, ref_d, ref_body.name))
    print("VERIFY: delta H/W/D = %+.2f%% / %+.2f%% / %+.2f%%"
          % (pct(res_h, ref_h), pct(res_w, ref_w), pct(res_d, ref_d)))
    print("VERIFY: (depth D is unreliable -- reference body mesh carries merged/extra "
          "geometry; judge H and W.)")

    print("-" * 64)
    if breast_before and breast_after:
        print("VERIFY: breast region extent (W,D,H) before = %.4f,%.4f,%.4f  "
              "after = %.4f,%.4f,%.4f"
              % (breast_before.x, breast_before.y, breast_before.z,
                 breast_after.x, breast_after.y, breast_after.z))
        ydelta = pct(breast_after.y, breast_before.y)
        xdelta = pct(breast_after.x, breast_before.x)
        print("VERIFY: breast Y-extent delta = %+.2f%% (profile Y=1.0 -> expect ~0%%); "
              "X-extent delta = %+.2f%% (profile X=0.95 + Chest X=1.05 propagation)"
              % (ydelta, xdelta))
        breast_y_ok = abs(ydelta) < 3.0
        print("VERIFY: breast-Y-unchanged check: %s"
              % ("PASS" if breast_y_ok else "REVIEW (>3%)"))
    else:
        breast_y_ok = True
        print("VERIFY: breast region not measurable (no weighted verts found)")

    legdelta = pct(upperleg1, upperleg0)
    leg_ok = 2.0 < legdelta < 7.0
    print("VERIFY: Upper_leg.L rest length %.5f -> %.5f (%+.2f%%; profile Y=1.055 "
          "after global 0.99 -> expect ~+4.4%%) local-op-baked check: %s"
          % (upperleg0, upperleg1, legdelta, "PASS" if leg_ok else "REVIEW"))

    print("-" * 64)
    print("VERIFY: renders ->")
    for p in (front_r, side_r, front_ref, side_ref):
        print("VERIFY:   ", p)
    print("=" * 64)
    review = [n for n, ok in (("breast-Y-unchanged", breast_y_ok), ("leg-length", leg_ok)) if not ok]
    if review:
        print("VERIFY: REVIEW NEEDED — failing checks:", ", ".join(review))
        sys.exit(1)
    print("VERIFY: DONE (human judges proportions vs renders + deltas above)")


if __name__ == "__main__":
    main()
