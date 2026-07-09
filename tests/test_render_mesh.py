"""Synthetic headless test for avatarprep.core.render_mesh.render().

Run: blender --background --factory-startup --python tests/test_render_mesh.py
Prints RENDERMESH_TEST OK / RENDERMESH_TEST FAIL: <reason>; sys.exit(1) on any failure.

Workbench renders headless, so BOTH the pre-render refusals AND the real render
assertions live here (no separate windowed script). The load-bearing render asserts:
a direction-keyed 3-axis orientation fixture (fails on any front/back, left/right, or
top/bottom vertical-flip swap) and an RBT-marker color round-trip (catches the sRGB
read-back defect). See the module docstring in core/render_mesh.py for the pipeline.
"""
import os
import sys
import math
import tempfile

import bpy
import numpy as np
from mathutils import Vector

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def _enable():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def _clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)


# --- world face-normal axis -> distinct color (byte); the orientation ground truth ---------
_FACE_COLORS = {
    (1, 0, 0):  (255, 0, 0, 255),      # +X red
    (-1, 0, 0): (0, 255, 255, 255),    # -X cyan
    (0, 1, 0):  (0, 255, 0, 255),      # +Y green
    (0, -1, 0): (255, 0, 255, 255),    # -Y magenta
    (0, 0, 1):  (0, 0, 255, 255),      # +Z blue
    (0, 0, -1): (255, 255, 0, 255),    # -Z yellow
}
# confirmed mapping (which world face each angle shows), matching core _VIEW_Q's Felis calibration:
# front->+Y, back->-Y, left->-X, right->+X, top->+Z, bottom->-Z. The RGB each angle must show dominant.
_ANGLE_EXPECT = {
    "front":  (0, 255, 0),     # +Y
    "back":   (255, 0, 255),   # -Y
    "left":   (0, 255, 255),   # -X
    "right":  (255, 0, 0),     # +X
    "top":    (0, 0, 255),     # +Z
    "bottom": (255, 255, 0),   # -Z
}


def _dominant_axis(n):
    ax = max(range(3), key=lambda i: abs(n[i]))
    v = [0, 0, 0]
    v[ax] = 1 if n[ax] > 0 else -1
    return tuple(v)


def _add_cube(name, size=2.0, attr_name="Test", per_face=True, solid_color=None):
    """A cube with a CORNER BYTE_COLOR attribute. per_face: a distinct colour per world face
    normal, AND on the four SIDE faces a bright top-half (high world Z) vs a dark bottom-half —
    a vertical asymmetry so the orientation test's counted assert fails on a forgotten reshape
    flip (a uniform-per-face colour is flip-invariant). Else one uniform solid_color (rgba bytes)."""
    bpy.ops.mesh.primitive_cube_add(size=size)
    ob = bpy.context.active_object
    ob.name = name
    me = ob.data
    attr = me.color_attributes.new(name=attr_name, type='BYTE_COLOR', domain='CORNER')
    me.color_attributes.active_color = attr
    me.color_attributes.render_color_index = list(me.color_attributes).index(attr)
    # author via color_srgb: BYTE_COLOR stores sRGB, so the stored byte == the authored byte
    # (the .color accessor is linear and would sRGB-encode a mid-value on store).
    for poly in me.polygons:
        axis = _dominant_axis(poly.normal)
        for li in poly.loop_indices:
            if not per_face:
                col = solid_color
            elif axis[2] != 0:  # top/bottom face — no world-Z asymmetry, uniform identity colour
                col = _FACE_COLORS[axis]
            else:               # side face — bright above the face centre, dark below (Z split)
                bright = _FACE_COLORS[axis]
                vz = me.vertices[me.loops[li].vertex_index].co.z
                col = bright if vz > 0 else tuple(int(c * 0.4) for c in bright[:3]) + (255,)
            attr.data[li].color_srgb = [c / 255.0 for c in col]
    me.update()
    return ob


def _add_plain_cube(name, size=2.0):
    bpy.ops.mesh.primitive_cube_add(size=size)
    ob = bpy.context.active_object
    ob.name = name
    return ob


def _load_png_top(path):
    """Load a saved sheet PNG byte-exact (Non-Color) and return it top-origin uint8 HxWx4."""
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = 'Non-Color'
    img.alpha_mode = 'STRAIGHT'
    w, h = img.size
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return np.clip(arr[::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)  # flip to top-origin


def _png_path(line):
    return line.split("png=")[-1].strip()


def _res(line):
    for tok in line.split():
        if tok.startswith("res="):
            return int(tok[4:])
    return None


def _dominant_nonplate(cell):
    rgb = cell[:, :, :3].reshape(-1, 3).astype(np.int32)
    mask = np.abs(rgb - 71).sum(axis=1) > 30  # drop the 71 plate
    rgb = rgb[mask]
    if len(rgb) == 0:
        return None
    colors, counts = np.unique(rgb, axis=0, return_counts=True)
    return tuple(int(x) for x in colors[counts.argmax()])


def _nearest_face(dom):
    return min(_FACE_COLORS.values(), key=lambda c: sum((c[i] - dom[i]) ** 2 for i in range(3)))


# ============================ pre-render refusals ==========================================

def test_sanitize():
    from avatarprep.core.render_mesh import _sanitize
    check(_sanitize("a.b c-d") == "a_b_c_d", "sanitize should map . space - to _, got %r" % _sanitize("a.b c-d"))
    check(_sanitize("Body_01") == "Body_01", "sanitize should keep alnum + _, got %r" % _sanitize("Body_01"))


def test_refuse_unknown_angle():
    from avatarprep.core.render_mesh import render
    _clear()
    _add_plain_cube("Cube")
    line = render(angles=["front", "sideways"], resolution=64)
    check("=> FAIL:" in line and "sideways" in line and "front,back,left,right,top,bottom" in line,
          "unknown angle should refuse listing the vocabulary, got %r" % line)
    check("png=" not in line, "a FAIL line must emit no png=, got %r" % line)


def test_refuse_unknown_shading():
    from avatarprep.core.render_mesh import render
    _clear()
    _add_plain_cube("Cube")
    line = render(shading="glow", resolution=64)
    check("=> FAIL:" in line and "solid,vertexcolor" in line,
          "unknown shading should refuse listing solid,vertexcolor, got %r" % line)
    # mixed-case / whitespace normalizes and passes
    ok = render(shading=" Solid ", resolution=64)
    check("=> OK" in ok, "' Solid ' should normalize and pass, got %r" % ok)


def test_refuse_bad_resolution():
    from avatarprep.core.render_mesh import render
    _clear()
    _add_plain_cube("Cube")
    line = render(resolution=0)
    check("=> FAIL:" in line and "resolution must be >= 1" in line and "png=" not in line,
          "resolution < 1 should refuse naming it, got %r" % line)


def test_refuse_empty_scene():
    from avatarprep.core.render_mesh import render
    _clear()
    line = render(resolution=64)
    check("=> FAIL:" in line and "no render-visible mesh" in line and "png=" not in line,
          "empty scene should refuse, got %r" % line)


def test_refuse_zero_extent():
    from avatarprep.core.render_mesh import render
    _clear()
    me = bpy.data.meshes.new("Dot")
    me.from_pydata([(0.0, 0.0, 0.0)], [], [])  # single vertex — zero extent
    me.update()
    ob = bpy.data.objects.new("Dot", me)
    bpy.context.scene.collection.objects.link(ob)
    line = render(resolution=64)
    check("=> FAIL:" in line and "zero-extent bounds" in line and "png=" not in line,
          "single-vertex mesh should refuse zero-extent, got %r" % line)


def test_only_not_found_refusal():
    from avatarprep.core.render_mesh import render
    _clear()
    _add_plain_cube("Cube")
    line = render(only=["Ghost"], resolution=64)
    check("=> FAIL:" in line and "--only matched no render-visible mesh" in line and "png=" not in line,
          "--only naming only an absent object should refuse, got %r" % line)


def test_only_notes_distinguish():
    from avatarprep.core.render_mesh import render
    _clear()
    v = _add_plain_cube("Visible")
    h = _add_plain_cube("Hidden")
    h.hide_render = True
    line = render(only=["Visible", "Hidden", "Ghost"], resolution=64)
    check("=> OK" in line, "a resolvable --only should render, got %r" % line)
    check("only-hidden:Hidden" in line, "present-but-hidden name should read only-hidden, got %r" % line)
    check("only-not-found:Ghost" in line, "absent name should read only-not-found, got %r" % line)
    # both flags live in a single note= field before png=
    check(line.index("note=") < line.index("png="), "note= must sit before terminal png=, got %r" % line)


# ============================ real Workbench render assertions =============================

def test_solid_render():
    from avatarprep.core.render_mesh import render
    _clear()
    _add_plain_cube("Cube")
    line = render(shading="solid", angles=["front"], resolution=256)
    check("=> OK" in line and "png=" in line, "solid single-angle should OK + png, got %r" % line)
    path = _png_path(line)
    check(os.path.exists(path), "solid PNG should exist at %r" % path)
    if os.path.exists(path):
        px = _load_png_top(path)
        # coverage over the plate: something drew (drew floor cleared, well above 0.5%)
        nonplate = (np.abs(px[:, :, :3].astype(np.int32) - 71).sum(axis=2) > 30).mean()
        check(nonplate > 0.05, "solid render should clear the drew floor, coverage=%.4f" % nonplate)


def _nonplate_luma(region):
    """Mean luminance (rgb sum) of the non-plate pixels in a region, 0 if none."""
    rgb = region[:, :, :3].reshape(-1, 3).astype(np.int32)
    m = np.abs(rgb - 71).sum(axis=1) > 30
    return float(rgb[m].sum(axis=1).mean()) if m.any() else 0.0


def _under(path, base):
    p = os.path.normcase(os.path.abspath(path))
    b = os.path.normcase(os.path.abspath(base))
    return p == b or p.startswith(b.rstrip(os.sep) + os.sep)


def test_output_persists_location():
    """The returned png= must live in the PERSISTENT OS temp, in the DEDICATED
    <tempfile.gettempdir>/avatarprep_rendermesh subdir (not the bare temp root — so a grab-dir
    scanner sees only rendermesh sheets, mirroring RenderAvatar's per-project cache), NOT in
    bpy.app.tempdir — Blender deletes its session dir on process exit, so a session-dir path is a
    dead path by the time a real headless cli returns. A true cross-process test isn't feasible in
    one process; this location assert is the practical regression guard."""
    from avatarprep.core.render_mesh import render
    _clear()
    _add_plain_cube("Cube")
    line = render(angles=["front"], shading="solid", resolution=128)
    path = _png_path(line)
    subdir = os.path.join(tempfile.gettempdir(), "avatarprep_rendermesh")
    check(_under(path, subdir),
          "png= should be under the dedicated subdir %r, got %r" % (subdir, path))
    check(not _under(path, bpy.app.tempdir),
          "png= must NOT be under bpy.app.tempdir (nuked on exit), got %r" % path)


def test_orientation():
    """Direction-keyed 3-axis fixture. Two counted asserts: (1) each cell's dominant colour equals
    the expected face colour — fails on any front/back, left/right, or top/bottom face swap; (2) on
    the four side tiles the bright top-half sits ABOVE the dark bottom-half — fails on a forgotten
    reshape vertical flip (which a uniform-per-face colour, being flip-invariant, would not catch)."""
    from avatarprep.core.render_mesh import render
    _clear()
    _add_cube("OrientCube", per_face=True)
    angles = ["front", "back", "left", "right", "top", "bottom"]
    line = render(angles=angles, shading="vertexcolor", resolution=256)
    check("=> OK" in line and "tiles=6" in line and "png=" in line,
          "6-angle vertexcolor should OK tiles=6 + png, got %r" % line)
    path = _png_path(line)
    edge = _res(line)
    if not (os.path.exists(path) and edge):
        check(False, "orientation PNG/res missing, line=%r" % line)
        return
    px = _load_png_top(path)
    cols = math.ceil(math.sqrt(len(angles)))
    for i, angle in enumerate(angles):
        r, c = divmod(i, cols)
        cell = px[r * edge:(r + 1) * edge, c * edge:(c + 1) * edge]
        dom = _dominant_nonplate(cell)
        if dom is None:
            check(False, "angle %s cell had no geometry" % angle)
            continue
        got = _nearest_face(dom)[:3]
        check(got == _ANGLE_EXPECT[angle],
              "angle %s should show face %s, dominant=%s nearest=%s"
              % (angle, _ANGLE_EXPECT[angle], dom, got))
        # vertical-flip guard: the four side faces are bright-top/dark-bottom (top = high world Z)
        if angle in ("front", "back", "left", "right"):
            h = cell.shape[0]
            top = _nonplate_luma(cell[:h // 2])
            bot = _nonplate_luma(cell[h // 2:])
            check(top > bot + 60.0,
                  "angle %s top-half should be brighter than bottom (upright, not flipped): "
                  "top-luma=%.0f bot-luma=%.0f" % (angle, top, bot))


def test_rbt_marker():
    """An 'RBT Matched'-style render colour attribute round-trips to the authored bytes within a
    tight tolerance — catches the sRGB read-back defect. A SINGLE colour attribute is unambiguous,
    so no ambiguous-color-attribute note fires."""
    from avatarprep.core.render_mesh import render
    _clear()
    rbt_fail = (234, 0, 255)  # robust-weight-transfer 'failed' magenta (234/255, 0, 1)
    _add_cube("RBTMesh", per_face=False, attr_name="RBT Matched",
              solid_color=(rbt_fail[0], rbt_fail[1], rbt_fail[2], 255))
    line = render(angles=["front"], shading="vertexcolor", resolution=256)
    check("=> OK" in line and "png=" in line, "RBT vertexcolor should OK + png, got %r" % line)
    check("no-color-attribute" not in line and "ambiguous-color-attribute" not in line,
          "single-attr RBT mesh should emit no missing/ambiguous note, got %r" % line)
    path = _png_path(line)
    if os.path.exists(path):
        px = _load_png_top(path)
        dom = _dominant_nonplate(px)
        check(dom is not None and all(abs(dom[i] - rbt_fail[i]) <= 3 for i in range(3)),
              "RBT marker should round-trip to %s within 3, got %s" % (rbt_fail, dom))


def test_ambiguous_color_attribute():
    """The identity hazard: a mesh with >1 colour attribute renders the one at render_color_index,
    and the OK line must NAME what was actually rendered (so a non-RBT layer can't masquerade as the
    marker). The single-attribute case must NOT emit this note."""
    from avatarprep.core.render_mesh import render
    _clear()
    # a mesh carrying an unrelated layer FIRST and the RBT marker second, marker flagged for render
    me = bpy.data.meshes.new("MultiData")
    me.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
    me.update()
    a0 = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='CORNER')
    a1 = me.color_attributes.new(name="RBT Matched", type='BYTE_COLOR', domain='CORNER')
    for d in a0.data:
        d.color_srgb = (0.0, 0.0, 1.0, 1.0)
    for d in a1.data:
        d.color_srgb = (234 / 255.0, 0.0, 1.0, 1.0)
    me.color_attributes.render_color_index = list(me.color_attributes).index(a1)
    ob = bpy.data.objects.new("Multi", me)
    bpy.context.scene.collection.objects.link(ob)
    line = render(angles=["top"], shading="vertexcolor", resolution=256)
    check("=> OK" in line, "multi-attr vertexcolor should OK, got %r" % line)
    check("ambiguous-color-attribute:Multi=RBT_Matched" in line,
          "multi-attr mesh must name the rendered attribute, got %r" % line)


# NOTE: the spec (v2) drops the unreachable no-render-color-attribute case — Blender 5.1 clamps
# render_color_index to a valid index (0) whenever any colour attribute exists, so "attributes but
# none render-flagged" cannot occur through the API. The real hazard is IDENTITY (render index points
# at a non-marker layer), covered by test_ambiguous_color_attribute above.
def test_no_color_attribute():
    from avatarprep.core.render_mesh import render
    _clear()
    _add_plain_cube("Plain")
    line = render(angles=["front"], shading="vertexcolor", resolution=256)
    check("=> OK" in line, "vertexcolor on a plain mesh should still OK, got %r" % line)
    check("no-color-attribute:Plain" in line,
          "a mesh with no colour attribute should be flagged no-color-attribute, got %r" % line)
    path = _png_path(line)
    if os.path.exists(path):
        px = _load_png_top(path)
        dom = _dominant_nonplate(px)
        # renders a neutral (grey) fallback, distinct from the 71 plate
        check(dom is not None and max(dom) - min(dom) <= 25,
              "no-colour mesh should render a neutral grey fallback, got %s" % (dom,))


def test_plate():
    """An empty grid cell / margin samples exactly the 71 plate (passes by construction — a
    compose-layout check, not a round-trip guard)."""
    from avatarprep.core.render_mesh import render
    _clear()
    _add_plain_cube("Cube")
    # 3 angles -> 2x2 grid, one empty cell filled with the plate
    line = render(angles=["front", "back", "left"], shading="solid", resolution=128)
    path = _png_path(line)
    edge = _res(line)
    if os.path.exists(path) and edge:
        px = _load_png_top(path)
        empty = px[edge:2 * edge, edge:2 * edge]  # cell (1,1) is empty for 3 tiles in a 2x2 grid
        check(np.all(empty[:, :, 0] == 71) and np.all(empty[:, :, 1] == 71) and np.all(empty[:, :, 2] == 71),
              "empty grid cell should be exactly the 71 plate, got mean=%s" % (empty[:, :, :3].mean(axis=(0, 1)),))


def test_fail_emits_no_png():
    from avatarprep.core.render_mesh import render
    _clear()
    _add_plain_cube("Cube")
    line = render(angles=["nonsense"], resolution=64)
    check("=> FAIL" in line and "png=" not in line, "FAIL path must emit no png=, got %r" % line)


def test_resolution_ceiling():
    from avatarprep.core.render_mesh import render
    _clear()
    _add_plain_cube("Cube")
    line = render(resolution=100000)
    check("=> FAIL:" in line and "resolution must be <=" in line and "png=" not in line,
          "an absurd resolution should refuse early naming the ceiling, got %r" % line)


def test_only_all_hidden_breakdown():
    from avatarprep.core.render_mesh import render
    _clear()
    h = _add_plain_cube("Cape")
    h.hide_render = True
    # every --only name unresolvable: one hidden, one absent — the FAIL must break them down
    line = render(only=["Cape", "Ghost"], resolution=64)
    check("=> FAIL:" in line and "png=" not in line, "all-hidden/absent --only should refuse, got %r" % line)
    check("hidden:Cape" in line and "not-found:Ghost" in line,
          "the refusal should distinguish hidden from not-found, got %r" % line)


def main():
    _enable()
    test_sanitize()
    test_refuse_unknown_angle()
    test_refuse_unknown_shading()
    test_refuse_bad_resolution()
    test_refuse_empty_scene()
    test_refuse_zero_extent()
    test_only_not_found_refusal()
    test_only_notes_distinguish()
    test_solid_render()
    test_output_persists_location()
    test_orientation()
    test_rbt_marker()
    test_ambiguous_color_attribute()
    test_no_color_attribute()
    test_plate()
    test_fail_emits_no_png()
    test_resolution_ceiling()
    test_only_all_hidden_breakdown()
    if FAILURES:
        for f in FAILURES:
            print("RENDERMESH_TEST FAIL:", f)
        sys.exit(1)
    print("RENDERMESH_TEST OK")


if __name__ == "__main__":
    main()
