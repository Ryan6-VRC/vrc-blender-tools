"""In-script fixture for the weight-transfer and fit suites and the add-on differential runner.

Not a suite (no ``test_`` prefix): ``tests/test_weight_transfer.py``, ``tests/test_fit.py``
and ``tests/acceptance/diff_addon.py`` build from it.

Body ``Body_Base`` on ``BodyRig``: a pelvis tube (Hips, fading to UpperLeg.L/R at its
lower rim) over two leg tubes (UpperLeg, LowerLeg below the knee, Hips at the top), Foot and
Toes bones facing -Y, and a ``Bulk`` shape key. Garment ``Shorts`` on ``GarmentRig`` (the
body's bones plus garment bones Flap..Flap4), 2 mm off the body, as islands:

  band    around the pelvis, two boundary loops; its back bottom row is Flap 1.0 (kept), the
          next Flap and Flap2 at 0.25 (the midline there caps at two body bones), and one vertex
          carries Flap..Flap4 at 0.2 each (no allowance left)
  cuff_L / cuff_R  around each thigh, two boundary loops each (leg holes)
  flipped a flat patch in front of the pelvis wound inward, so it matches only by flip
  float   a patch 12 cm in front of the body: unmatched, inpainted
  strip   a Flap-only strip hanging behind the band

Every body-bone vertex starts on a crude vendor weight (Hips at ``1 - p``). Groups beside
the bones: ``Detail`` (non-bone, front of the band) and ``WriteMask`` (the band's +X half
and the left cuff at 1.0).

``build(tail=True)`` adds a ``Tail`` bone to both rigs (a non-humanoid bone hanging behind the
pelvis), a tail tube on the body weighted to it, and a ``TailCover`` sleeve on it.
``add_band`` adds a band 2 mm around the left thigh, where the body is pure UpperLeg.L.
"""
import math

import bpy

GAP = 0.002
RP, RL, LX = 0.16, 0.07, 0.08
BODY_BONES = [  # name, head, tail, parent
    ("Hips", (0, 0, 0.95), (0, 0, 1.10), None),
    ("UpperLeg.L", (LX, 0, 0.90), (LX, 0, 0.50), "Hips"),
    ("LowerLeg.L", (LX, 0, 0.50), (LX, 0, 0.10), "UpperLeg.L"),
    ("Foot.L", (LX, 0, 0.10), (LX, -0.08, 0.02), "LowerLeg.L"),
    ("Toes.L", (LX, -0.10, 0.02), (LX, -0.15, 0.02), "Foot.L"),
    ("UpperLeg.R", (-LX, 0, 0.90), (-LX, 0, 0.50), "Hips"),
    ("LowerLeg.R", (-LX, 0, 0.50), (-LX, 0, 0.10), "UpperLeg.R"),
    ("Foot.R", (-LX, 0, 0.10), (-LX, -0.08, 0.02), "LowerLeg.R"),
    ("Toes.R", (-LX, -0.10, 0.02), (-LX, -0.15, 0.02), "Foot.R"),
]
TAIL_Y, RT = 0.20, 0.03
TAIL_BONE = [("Tail", (0, TAIL_Y, 0.88), (0, TAIL_Y, 0.60), "Hips")]
GARMENT_BONES = [("Flap%s" % s, (0, 0.17, 0.92), (0, 0.17, 0.82), "Hips") for s in ("", "2", "3", "4")]


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)


def armature(name, bones):
    arm = bpy.data.armatures.new(name)
    ob = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = {}
    for n, h, t, _ in bones:
        eb = arm.edit_bones.new(n)
        eb.head, eb.tail = h, t
        ebs[n] = eb
    for n, _, _, p in bones:
        if p:
            ebs[n].parent = ebs[p]
    bpy.ops.object.mode_set(mode='OBJECT')
    return ob


def tube(cx, cy, r, z0, z1, rings, segs, tag, verts, faces, tags):
    base = len(verts)
    for i in range(rings + 1):
        z = z0 + (z1 - z0) * i / rings
        for j in range(segs):
            a = 2 * math.pi * j / segs
            verts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
            tags.append(tag)
    for i in range(rings):
        for j in range(segs):
            a = base + i * segs + j
            b = base + i * segs + (j + 1) % segs
            faces.append((a, b, b + segs, a + segs))


def patch(x0, x1, y, z0, z1, n, inward, tag, verts, faces, tags):
    """A flat (n+1)^2 patch in the plane y; normal -Y (out of the front) unless ``inward``."""
    base = len(verts)
    for i in range(n + 1):
        for j in range(n + 1):
            verts.append((x0 + (x1 - x0) * j / n, y, z0 + (z1 - z0) * i / n))
            tags.append(tag)
    for i in range(n):
        for j in range(n):
            a = base + i * (n + 1) + j
            q = (a, a + 1, a + n + 2, a + n + 1)  # normal -Y
            faces.append(q[::-1] if inward else q)


def mesh(name, verts, faces, rig):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = rig
    m = ob.modifiers.new("Armature", 'ARMATURE')
    m.object = rig
    return ob


def body_weights(v, tag):
    x, y, z = v
    if tag == "tail":
        return {"Tail": 1.0}
    if tag == "pelvis":
        h = smoothstep((z - 0.88) / 0.10)
        left = smoothstep((x / RP + 0.3) / 0.6)
        return {"Hips": h, "UpperLeg.L": (1 - h) * left, "UpperLeg.R": (1 - h) * (1 - left)}
    side = "L" if x > 0 else "R"
    hips = 0.5 * smoothstep((z - 0.80) / 0.10)
    lower = 1 - smoothstep((z - 0.42) / 0.16)
    return {"Hips": hips, "UpperLeg." + side: (1 - hips) * (1 - lower), "LowerLeg." + side: (1 - hips) * lower}


def build(tail=False):
    """Build the scene; returns ``{"body", "body_rig", "garment", "garment_rig", "tags"}``
    where ``tags`` names each garment vertex's island (plus ``tail_cover`` with ``tail``)."""
    clear()
    bones = BODY_BONES + (TAIL_BONE if tail else [])
    body_rig = armature("BodyRig", bones)
    garment_rig = armature("GarmentRig", bones + GARMENT_BONES)

    verts, faces, tags = [], [], []
    tube(0, 0, RP, 0.88, 1.10, 11, 32, "pelvis", verts, faces, tags)
    tube(LX, 0, RL, 0.10, 0.90, 40, 24, "leg", verts, faces, tags)
    tube(-LX, 0, RL, 0.10, 0.90, 40, 24, "leg", verts, faces, tags)
    if tail:
        tube(0, TAIL_Y, RT, 0.60, 0.86, 13, 16, "tail", verts, faces, tags)
    body = mesh("Body_Base", verts, faces, body_rig)
    groups = {}
    for i, (v, t) in enumerate(zip(verts, tags)):
        for g, w in body_weights(v, t).items():
            if w > 0:
                groups.setdefault(g, body.vertex_groups.get(g) or body.vertex_groups.new(name=g)).add([i], w, 'REPLACE')
    body.shape_key_add(name="Basis")
    kb = body.shape_key_add(name="Bulk", from_mix=False)
    kb.value = 0.0
    for i, (v, t) in enumerate(zip(verts, tags)):
        if t == "pelvis":
            kb.data[i].co = (v[0] * 1.06, v[1] * 1.06, v[2])

    verts, faces, tags = [], [], []
    tube(0, 0, RP + GAP, 0.90, 1.06, 8, 32, "band", verts, faces, tags)
    tube(LX, 0, RL + GAP, 0.66, 0.84, 9, 24, "cuff_L", verts, faces, tags)
    tube(-LX, 0, RL + GAP, 0.66, 0.84, 9, 24, "cuff_R", verts, faces, tags)
    patch(-0.02, 0.02, -(RP + GAP), 1.07, 1.09, 2, True, "flipped", verts, faces, tags)
    patch(-0.02, 0.02, -(RP + 0.12), 0.98, 1.02, 2, False, "float", verts, faces, tags)
    patch(-0.03, 0.03, RP + 0.03, 0.80, 0.88, 2, True, "strip", verts, faces, tags)
    garment = mesh("Shorts", verts, faces, garment_rig)
    vg = {n: garment.vertex_groups.new(name=n) for n in
          ("Hips", "Flap", "Flap2", "Flap3", "Flap4", "Detail", "WriteMask")}
    four = None
    for i, ((x, y, z), t) in enumerate(zip(verts, tags)):
        garm = {}
        if t == "strip":
            garm = {"Flap": 1.0}
        elif t == "band" and y > 0.12:
            if abs(z - 0.90) < 1e-6:
                garm = {"Flap": 1.0}
            elif abs(z - 0.92) < 1e-6:
                garm = {"Flap": 0.25, "Flap2": 0.25}
            elif abs(z - 0.96) < 1e-6 and four is None:
                garm = {"Flap": 0.2, "Flap2": 0.2, "Flap3": 0.2, "Flap4": 0.2}
                four = i
        for g, w in garm.items():
            vg[g].add([i], w, 'REPLACE')
        rest = 1.0 - sum(garm.values())
        if rest > 1e-6:
            vg["Hips"].add([i], rest, 'REPLACE')
        if t == "band" and y < -0.1:
            vg["Detail"].add([i], 0.3 + 0.6 * (z - 0.90) / 0.16, 'REPLACE')
        if (t == "band" and x > 0) or t == "cuff_L":
            vg["WriteMask"].add([i], 1.0, 'REPLACE')
    out = {"body": body, "body_rig": body_rig, "garment": garment, "garment_rig": garment_rig,
           "tags": tags, "four": four}
    if tail:
        out["tail_cover"] = _weighted_tube("TailCover", garment_rig, (0, TAIL_Y, RT + GAP, 0.62, 0.84, 11, 16),
                                           {"Tail": 1.0})
    return out


def _weighted_tube(name, rig, shape, weights):
    verts, faces, tags = [], [], []
    cx, cy, r, z0, z1, rings, segs = shape
    tube(cx, cy, r, z0, z1, rings, segs, name, verts, faces, tags)
    ob = mesh(name, verts, faces, rig)
    for g, w in weights.items():
        ob.vertex_groups.new(name=g).add(list(range(len(verts))), w, 'REPLACE')
    return ob


def add_band(s, name, weights, gap=GAP, z0=0.60, z1=0.76):
    """A band ``gap`` off the left thigh between ``z0`` and ``z1`` on the garment rig, every
    vertex carrying ``weights`` (``{group: w}``)."""
    return _weighted_tube(name, s["garment_rig"], (LX, 0, RL + gap, z0, z1, 8, 24), weights)


def weights_by_name(ob):
    """``{group_name: [w per vertex]}`` as stored, for comparisons across rebuilds."""
    names = [g.name for g in ob.vertex_groups]
    out = {n: [0.0] * len(ob.data.vertices) for n in names}
    for i, v in enumerate(ob.data.vertices):
        for g in v.groups:
            out[names[g.group]][i] = g.weight
    return out
