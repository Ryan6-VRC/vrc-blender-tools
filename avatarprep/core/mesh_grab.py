"""Headless multi-angle Workbench render of the scene's render-visible meshes — the
Blender-side sibling of Unity AvatarGrab (``vrc-unity-tools/.../Editor/AvatarGrab.cs``):
same result grammar, same refusal shape, same angle vocabulary, same
``meshgrab_<label>_<stamp>.png`` filename and terminal ``png=`` trailer.

Runs entirely under ``--background`` (no VIEW_3D, no GL). Both shading modes render
through ``BLENDER_WORKBENCH``; ``film_transparent`` carries geometry coverage as the
rendered alpha, and we composite that over a Unity-grey ``71`` plate ourselves in
numpy — so the plate byte is ``71`` by construction and the drew-guard keys off alpha
(a shading-independent "did anything draw" signal), never a colour-difference read.

No ``bpy.types.Operator`` subclasses and no UI live here (mirrors scene_utils). The
one door is the headless ``cli/mesh_grab.py``.

Two spellings, on purpose: ``mesh_grab`` is the file/key/param form; ``meshgrab`` is
the display token in the result line (mirrors AvatarGrab's concatenated ``avatargrab_``).
"""

import glob
import math
import os
import tempfile
import time
from datetime import datetime

import bpy
import numpy as np
from mathutils import Quaternion, Vector

_MARGIN = 0.15   # ortho-scale border fraction so the silhouette doesn't touch the tile edge
_EPSILON = 1e-6  # zero-extent bounds guard
_PLATE = 71      # Unity-grey plate byte (parity with AvatarGrab's (71,71,71) compose fill)
_SHEET_CAP = 2048  # composed-sheet edge ceiling — parity with AvatarGrab's SheetEdgeCap
_MAX_RES = 8192    # per-tile render ceiling: rendered full-size BEFORE the sheet downscale, so an
                   # absurd value would allocate GBs; refuse early rather than OOM late
_DREW_FLOOR = 0.005  # per-tile alpha-coverage floor: a sub-1% "nothing drew" threshold, NOT a
                     # "little drew" one — the uniform ortho_scale frames a tall avatar's top/bottom
                     # as a small footprint, so a few-percent coverage is legitimate (Felis top tile
                     # is the checked-in calibration witness).

_ANGLE_VOCAB = ["front", "back", "left", "right", "top", "bottom"]
_SHADING_VOCAB = ["solid", "vertexcolor"]

# angle -> orthographic-view rotation quaternion (w, x, y, z), calibrated to which WORLD axis
# each view puts toward the camera. Confirmed byte-exact headless (Blender 5.1) against a 6-colour
# cube AND against Felis: this maps front->+Y, back->-Y, left->-X, right->+X, top->+Z, bottom->-Z.
# A VRChat avatar imported by avatarprep's wm.fbx_import lands facing +Y (up +Z), so front shows the
# face and left/right show the avatar's own left/right — with no up/down swap. NOTE: front/back are
# SWAPPED from Blender's native front/back (native front looks +Y at the -Y face); Felis calibration
# proved the native table renders the avatar's BACK on "front", so front and back are exchanged here.
# The angle is a world axis — a target NOT on the upright-facing-front convention renders the scene's
# front, not its own (the documented AvatarGrab limitation).
_VIEW_Q = {
    "front":  (0.0, 0.0, math.sqrt(2) / 2, math.sqrt(2) / 2),
    "back":   (math.sqrt(2) / 2, math.sqrt(2) / 2, 0.0, 0.0),
    "right":  (0.5, 0.5, 0.5, 0.5),
    "left":   (0.5, 0.5, -0.5, -0.5),
    "top":    (1.0, 0.0, 0.0, 0.0),
    "bottom": (0.0, 1.0, 0.0, 0.0),
}


def _sanitize(name: str) -> str:
    """alnum + '_' only; every other char -> '_'. Used for the <label> slot of both the
    filename and the result line."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in (name or ""))


def _fail(label, reason: str) -> str:
    """Mirror AvatarGrab's Fail(): family arrow, NO png= trailer — the schema never points
    at a PNG that isn't on disk. ``label`` may be None/empty -> "?"."""
    lbl = label if label else "?"
    return "AVATARPREP: meshgrab " + lbl + " => FAIL: " + reason


def _downscale(arr, edge):
    """Area-average (box filter) resample to edge x edge via an integral image — plain
    numpy, no PIL. Never called to upscale (callers only ever pass edge <= arr's side, so
    edge >= side is a no-op). Box-filtering (not nearest) keeps a thin high-frequency
    feature from aliasing away."""
    side = arr.shape[0]
    if edge >= side:
        return arr
    bounds = np.round(np.linspace(0, side, edge + 1)).astype(np.int64)
    bounds = np.clip(bounds, 0, side)
    for i in range(1, edge + 1):  # strictly increasing (no empty bins)
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = min(bounds[i - 1] + 1, side)
    ii = np.zeros((side + 1, side + 1, arr.shape[2]), dtype=np.float64)
    ii[1:, 1:] = arr.astype(np.float64).cumsum(axis=0).cumsum(axis=1)
    y0, y1 = bounds[:-1], bounds[1:]
    x0, x1 = bounds[:-1], bounds[1:]
    block_sum = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
    area = (y1 - y0)[:, None, None] * (x1 - x0)[None, :, None]
    return np.clip(block_sum / area + 0.5, 0, 255).astype(np.uint8)  # round, matching the other sites


def _compose_sheet(tiles, edge, cols, rows):
    """Row-major in requested order, 71-plate-filled empty cells. The array is bottom-origin
    (matches the bpy image API — never flipped end to end), so requested-order row 0 lands at
    the LAST array band (mirrors AvatarGrab.Compose's ``(rows - 1 - r)``) and reads back as the
    top row of the saved sheet."""
    sheet = np.full((rows * edge, cols * edge, 4), _PLATE, dtype=np.uint8)
    sheet[:, :, 3] = 255
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y0, x0 = (rows - 1 - r) * edge, c * edge
        sheet[y0:y0 + edge, x0:x0 + edge] = tile
    return sheet


def _drawable_set(view_layer):
    """(all_meshes, visible) for the active view layer — visible = render-visible
    (``hide_render == False``). View-layer membership (not scene.objects) so a
    collection-excluded object is neither framed nor isolated."""
    all_meshes = [o for o in view_layer.objects if o.type == 'MESH']
    visible = [o for o in all_meshes if not o.hide_render]
    return all_meshes, visible


def _world_aabb(drawable, depsgraph):
    """Union of each drawable mesh's depsgraph-EVALUATED world bounding box (so modifiers —
    e.g. a mirror that doubles true extent — are included). Returns (center, max_dim)."""
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for o in drawable:
        ev = o.evaluated_get(depsgraph)
        mw = o.matrix_world
        for corner in ev.bound_box:
            w = mw @ Vector(corner)
            for i in range(3):
                if w[i] < mins[i]:
                    mins[i] = w[i]
                if w[i] > maxs[i]:
                    maxs[i] = w[i]
    center = Vector(((mins[i] + maxs[i]) / 2.0 for i in range(3)))
    max_dim = max(maxs[i] - mins[i] for i in range(3))
    return center, max_dim


def _prune_old_sheets(out_dir, days=30):
    """Delete our own ``meshgrab_*.png`` older than ``days`` in ``out_dir`` — best-effort per file
    (a locked or already-gone file is skipped). Mirrors AvatarGrab's 30-day self-prune: the persistent
    temp is never swept by the OS, so each write bounds it; the just-written sheet is newer than the
    cutoff, so it never self-deletes."""
    cutoff = time.time() - days * 86400
    for old in glob.glob(os.path.join(out_dir, "meshgrab_*.png")):
        try:
            if os.path.getmtime(old) < cutoff:
                os.remove(old)
        except OSError:
            pass


def _notes_field(notes):
    """Single ``note=`` field of space-separated key:value tokens; each value is a
    comma-joined list with no spaces, in encounter order. Empty -> ""."""
    order = ["no-color-attribute", "ambiguous-color-attribute", "only-hidden", "only-not-found"]
    toks = [("%s:%s" % (k, ",".join(notes[k]))) for k in order if notes.get(k)]
    return (" | note=%s" % " ".join(toks)) if toks else ""  # its own | field, before terminal png=


def grab(
    label=None,       # filename/result label; None/empty -> the scene name
    only=None,        # flat object-name filter; None/empty -> all render-visible meshes
    angles=None,      # subset of {front,back,left,right,top,bottom}; None/empty -> [front, back]
    shading: str = "solid",  # "solid" | "vertexcolor"
    resolution: int = 1024,  # per-tile square edge cap in px; must be >= 1; never upscales
) -> str:
    """Render the scene's render-visible meshes (optionally narrowed by ``only`` names) from
    ``angles`` to one stamped contact-sheet PNG in ``bpy.app.tempdir``; return the one-line
    summary (OK or FAIL). Never raises for an EXPECTED refusal — it returns the FAIL line;
    a genuinely unexpected error propagates (the cli maps that to exit 2).

    Operates on the CURRENT ``bpy.context.scene``. The ``.blend`` on disk is never touched,
    but shared in-memory state (object ``hide_render``, ``scene.camera``, temp datablocks) IS
    mutated, so it is snapshot at entry and restored in a ``finally`` — several ``grab()`` calls
    can run in one ``--python`` process without call N corrupting call N+1.
    """
    scene = bpy.context.scene
    label_raw = label if label else (scene.name if scene else None)
    lbl = _sanitize(label_raw) if label_raw else None

    # --- validate resolution ---------------------------------------------------------------
    try:
        resolution = int(resolution)
    except (TypeError, ValueError):
        return _fail(lbl, "resolution must be >= 1, got %r" % (resolution,))
    if resolution < 1:
        return _fail(lbl, "resolution must be >= 1, got %d" % resolution)
    if resolution > _MAX_RES:
        return _fail(lbl, "resolution must be <= %d, got %d" % (_MAX_RES, resolution))

    # --- validate angles -------------------------------------------------------------------
    if not angles:
        resolved_angles = ["front", "back"]
    else:
        resolved_angles = []
        for a in angles:
            aa = (a or "").strip().lower()
            if aa not in _ANGLE_VOCAB:
                return _fail(lbl, "unknown angle '%s' — valid: %s" % (a, ",".join(_ANGLE_VOCAB)))
            resolved_angles.append(aa)

    # --- validate shading ------------------------------------------------------------------
    shading = (shading or "").strip().lower()  # "Solid"/" solid " normalize and pass
    if shading not in _SHADING_VOCAB:
        return _fail(lbl, "unknown shading '%s' — valid: %s" % (shading, ",".join(_SHADING_VOCAB)))

    if scene is None:
        return _fail(lbl, "no active scene")

    view_layer = bpy.context.view_layer
    view_layer.update()  # so view_layer.objects reflects freshly-linked objects (needed in-process)
    all_meshes, visible = _drawable_set(view_layer)

    notes = {}
    touched_hide = {}  # object -> original hide_render, restored in finally
    saved_active_color = {}  # mesh.data -> original active_color_index, restored in finally

    # --- resolve the drawable set (visibility-based; --only narrows) -----------------------
    if only:
        requested = [n for n in (str(s).strip() for s in only) if n]
        visible_names = {o.name for o in visible}
        hidden_names = {o.name for o in all_meshes if o.hide_render}
        keep, hidden_hit, not_found = set(), [], []
        for n in requested:
            if n in visible_names:
                keep.add(n)
            elif n in hidden_names:
                hidden_hit.append(n)   # present in the .blend but hide_render -> found, stays hidden
            else:
                not_found.append(n)    # no render-visible mesh by that name at all
        if hidden_hit:
            notes["only-hidden"] = [_sanitize(n) for n in hidden_hit]
        if not_found:
            notes["only-not-found"] = [_sanitize(n) for n in not_found]
        drawable = [o for o in visible if o.name in keep]
        if not drawable:
            # fold the breakdown into the reason so the agent can tell "un-hide it" from "typo"
            parts = []
            if hidden_hit:
                parts.append("hidden:" + ",".join(_sanitize(n) for n in hidden_hit))
            if not_found:
                parts.append("not-found:" + ",".join(_sanitize(n) for n in not_found))
            detail = (" (%s)" % " ".join(parts)) if parts else ""
            return _fail(lbl, "--only matched no render-visible mesh%s" % detail)
    else:
        drawable = visible
        if not drawable:
            return _fail(lbl, "no render-visible meshes to render")

    # --- deterministic framing bounds ------------------------------------------------------
    depsgraph = bpy.context.evaluated_depsgraph_get()
    center, max_dim = _world_aabb(drawable, depsgraph)
    if max_dim <= _EPSILON:
        return _fail(lbl, "zero-extent bounds — nothing with size to frame")

    prev_camera = scene.camera
    cam_data = None
    cam_obj = None
    loaded_images = []
    written_tiles = []  # per-angle temp PNGs, deleted in finally (survives an exception mid-loop)
    delivered_edge = 1
    fail_reason = None
    path = None

    try:
        # --- isolation: hide every render-visible mesh not in the --only keep set ----------
        if only:
            keep = {o.name for o in drawable}
            for o in visible:
                if o.name not in keep:
                    touched_hide[o] = o.hide_render
                    o.hide_render = True

        # --- pin render + display settings (fresh every call; no restore needed) -----------
        r = scene.render
        r.engine = 'BLENDER_WORKBENCH'
        r.film_transparent = True
        r.use_border = False
        r.use_crop_to_border = False
        r.pixel_aspect_x = 1.0
        r.pixel_aspect_y = 1.0
        r.use_stamp = False
        r.resolution_percentage = 100
        r.resolution_x = resolution
        r.resolution_y = resolution
        img_s = r.image_settings
        img_s.file_format = 'PNG'
        img_s.color_mode = 'RGBA'
        img_s.color_depth = '8'
        scene.view_settings.view_transform = 'Standard'

        disp = scene.display.shading
        if shading == "solid":
            disp.color_type = 'SINGLE'
            disp.light = 'STUDIO'
        else:  # vertexcolor
            disp.color_type = 'VERTEX'
            disp.light = 'FLAT'
            no_attr, ambiguous = [], []
            for o in drawable:
                attrs = o.data.color_attributes
                n = len(attrs)
                if n == 0:
                    no_attr.append(o.name)  # zero attributes -> grey fallback
                    continue
                # Point Workbench at the mesh's active RENDER color attribute. Blender keeps
                # render_color_index valid (0..n-1) whenever any attribute exists; clamp defensively.
                rci = attrs.render_color_index
                if rci < 0 or rci >= n:
                    rci = 0
                saved_active_color[o.data] = attrs.active_color_index
                try:
                    attrs.active_color_index = rci
                except Exception:
                    saved_active_color.pop(o.data, None)
                # The real hazard is IDENTITY, not range: with >1 attribute the render index may point
                # at a non-RBT layer (e.g. imported vertex colors) and the schema would silently render
                # it as if it were the marker. Name what was actually rendered so the read can't lie.
                if n > 1:
                    ambiguous.append("%s=%s" % (_sanitize(o.name), _sanitize(attrs[rci].name)))
            if no_attr:
                notes["no-color-attribute"] = [_sanitize(m) for m in no_attr]
            if ambiguous:
                notes["ambiguous-color-attribute"] = ambiguous

        # --- temp orthographic camera; one ortho_scale for every angle ---------------------
        cam_data = bpy.data.cameras.new("meshgrab_cam")
        cam_data.type = 'ORTHO'
        cam_data.ortho_scale = max_dim * (1.0 + _MARGIN)
        dist = max_dim * 4.0 + 1.0
        cam_data.clip_start = 0.001
        cam_data.clip_end = dist + max_dim * 4.0 + 1.0
        cam_obj = bpy.data.objects.new("meshgrab_cam", cam_data)
        scene.collection.objects.link(cam_obj)
        cam_obj.rotation_mode = 'QUATERNION'
        scene.camera = cam_obj

        # --- delivery sizing (sheet cap; never upscale) ------------------------------------
        n = len(resolved_angles)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        cap_edge = _SHEET_CAP // max(cols, rows)
        delivered_edge = max(1, min(resolution, cap_edge))

        # --- per-angle render + read + composite -------------------------------------------
        tiles = []
        for angle in resolved_angles:
            q = Quaternion(_VIEW_Q[angle])
            cam_obj.rotation_quaternion = q
            back = q.to_matrix() @ Vector((0.0, 0.0, 1.0))  # +Z (backward), NOT -Z (would be inside)
            cam_obj.location = center + back * dist

            tmp_path = os.path.join(bpy.app.tempdir, "meshgrab_tile_%s.png" % angle)
            written_tiles.append(tmp_path)
            r.filepath = tmp_path
            bpy.ops.render.render(write_still=True)

            img = bpy.data.images.load(tmp_path)
            loaded_images.append(img)
            # load() defaults an 8-bit PNG to 'sRGB', which makes img.pixels return scene-linear
            # floats (sRGB->linear applied on read) — undoing the write-side 'Standard' pin. 'Non-Color'
            # makes the read the byte-exact inverse of the write, so determinism reaches numpy.
            img.colorspace_settings.name = 'Non-Color'
            img.alpha_mode = 'STRAIGHT'
            w, h = img.size
            arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)  # bottom-origin; no flip
            px = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
            bpy.data.images.remove(img)
            loaded_images.remove(img)

            alpha = px[:, :, 3]
            coverage = float((alpha > 0).mean())
            if coverage < _DREW_FLOOR:
                fail_reason = "blank render on '%s' — nothing drew" % angle
                break

            # composite geometry over the 71 plate (straight alpha); interior alpha=1 -> rgb exact
            a = (alpha.astype(np.float32) / 255.0)[:, :, None]
            rgb = px[:, :, :3].astype(np.float32)
            comp = rgb * a + float(_PLATE) * (1.0 - a)
            tile = np.empty((h, w, 4), dtype=np.uint8)
            tile[:, :, :3] = np.clip(comp + 0.5, 0, 255).astype(np.uint8)
            tile[:, :, 3] = 255
            tiles.append(_downscale(tile, delivered_edge))

        if fail_reason is None:
            sheet = _compose_sheet(tiles, delivered_edge, cols, rows)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # millisecond, matching AvatarGrab
            # Persistent OS temp (Python's tempfile.gettempdir — the analog of Unity AvatarGrab's
            # Application.temporaryCachePath), NOT bpy.app.tempdir: Blender NUKES its session dir on
            # process exit, so a headless cli would return a png= path that no longer exists. The
            # per-angle intermediate tiles above stay in bpy.app.tempdir — they're intra-process scratch.
            out_dir = tempfile.gettempdir()
            path = os.path.join(out_dir, "meshgrab_%s_%s.png" % (lbl, stamp))
            sh, sw = sheet.shape[0], sheet.shape[1]
            out = bpy.data.images.new("meshgrab_out", sw, sh, alpha=True)
            loaded_images.append(out)
            out.colorspace_settings.name = 'Non-Color'  # byte-exact inverse of the Non-Color read
            out.pixels.foreach_set((sheet.astype(np.float32) / 255.0).reshape(-1))
            out.filepath_raw = path
            out.file_format = 'PNG'
            out.save()
            bpy.data.images.remove(out)
            loaded_images.remove(out)
            _prune_old_sheets(out_dir)  # bound the persistent dir (the just-written sheet is newest)

    finally:
        for o, v in touched_hide.items():
            try:
                o.hide_render = v
            except Exception:
                pass
        for mesh_data, idx in saved_active_color.items():
            try:
                mesh_data.color_attributes.active_color_index = idx
            except Exception:
                pass
        try:
            scene.camera = prev_camera
        except Exception:
            pass
        if cam_obj is not None:
            try:
                bpy.data.objects.remove(cam_obj, do_unlink=True)
            except Exception:
                pass
        if cam_data is not None:
            try:
                bpy.data.cameras.remove(cam_data)
            except Exception:
                pass
        for im in loaded_images:
            try:
                bpy.data.images.remove(im)
            except Exception:
                pass
        for p in written_tiles:
            try:
                os.remove(p)
            except OSError:
                pass

    if fail_reason is not None:
        return _fail(lbl, fail_reason)

    return ("AVATARPREP: meshgrab %s angles=%s shading=%s tiles=%d res=%d => OK%s | png=%s"
            % (lbl, ",".join(resolved_angles), shading, len(resolved_angles),
               delivered_edge, _notes_field(notes), path))
