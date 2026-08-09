"""Export the scene to a Unity/VRChat-correct FBX.

Thin wrapper over ``bpy.ops.export_scene.fbx`` with the parameter set Unity /
VRChat expect for avatar import (each value is documented inline below).
"""

import math
from typing import Optional

import bpy


def export_unity_fbx(filepath: str,
                     armature_obj: Optional[bpy.types.Object] = None,
                     object_types=None,
                     use_mesh_modifiers: bool = False,
                     add_leaf_bones: bool = False,
                     bake_anim: bool = False,
                     apply_scale_options: str = 'FBX_SCALE_ALL',
                     path_mode: str = 'COPY',
                     embed_textures: bool = True,
                     use_selection: bool = False,
                     keep_object_rotation: bool = False,
                     bake_object_scale: bool = True,
                     **extra) -> str:
    """Export ``filepath`` as an FBX using the CATS / Unity recipe.

    The defaults are the VRChat-correct settings:
      * ``object_types={'EMPTY', 'ARMATURE', 'MESH', 'OTHER'}``
      * ``use_mesh_modifiers=False``
      * ``add_leaf_bones=False``
      * ``bake_anim=False``
      * ``apply_scale_options='FBX_SCALE_ALL'``
      * ``embed_textures=True``
      * ``path_mode='COPY'`` (required for embedding to work)

    **Orientation — the canon; other files route here rather than re-derive it.**
    ``wm.fbx_import`` represents a source FBX's axis convention as an armature
    *object* rotation and leaves vertex data raw; this exporter writes root node
    = ``Gm(-90 X) @ matrix_world`` and leaves ``Vertices`` in object-local
    coordinates. Residue and conversion therefore compose **in the node, not the
    data**, and the exporter's own -90 X presumes the data it is handed is
    Blender-Z-up.

    So an object rotation is importer residue **only when it leaves the up axis
    fixed**, and only then is it cleared UNAPPLIED for the export (bone/mesh data
    untouched) and restored after. The three residue classes measured across ~250
    vendor files:

      * **identity** (the plurality) — a Y-up file whose root node already carries
        -90 X, i.e. Blender-exported. Nothing to do.
      * **(0, 0, -180)** — a ``+Z up / +Y front / -X coord`` file (the Felis
        fixture; the -X coord sign is what produces the 180). Leaves the up axis
        fixed, so it is a FRONT-axis convention difference: cleared. Carrying it
        through double-counts and Unity shows the avatar backwards. A Blender
        re-import cannot see that (the importer symmetrically undoes it); parsing
        the file can, which is why ``tests/test_fbx_orientation.py`` asserts on
        the written node rotation.
      * **(90, 0, 0)** — a Y-up file with an identity root node, i.e.
        Maya/Max-exported; roughly a third of the library. This MOVES the up axis,
        so it *is* the source's up-axis conversion, not residue: preserved.
        Clearing it double-counts the up-axis conversion and the rig exports
        tipped 90° onto its face (measured on Chocolat: re-import height
        1.1992 -> 0.4574 m, Y/Z bounds swapped).

    No file in the survey carried a residue that both moves the up axis and
    rotates about it; such a residue is preserved whole, and the emitted line
    names its value so the reader can see it.

    A deliberately rotated armature is the rare exception: pass
    ``keep_object_rotation=True``. That exception is now specifically a deliberate
    rotation *about the up axis* — an up-axis-moving one is preserved anyway.

    **Scope: ARMATURE objects only.** Mesh-only prop FBXs have no armature and are
    never touched here — which is exactly why they are correct today, the same
    fact this rule encodes. Note also that nothing in the ``avatarprep_`` stamp
    namespace records which frame the data is in; that is why a merge that bakes a
    wrong frame into the ``.blend`` (see ``merge_armatures``) is unrecoverable
    downstream rather than merely wrong. ``**extra`` lets a caller override
    ``axis_up``/``axis_forward``; the up-axis reasoning above is hardwired to the
    default -90 X conversion and does not follow an override.

    That 180° has a second switch on the consumer side, covered below.

    The consumer-side switch: Unity's per-asset
    ``bakeAxisConversion`` applies the same rotation, and no test here can see it
    (they parse the written file; this one lives in the Unity importer). This
    export is correct at Unity's default, OFF — turning it on for our output
    faces the avatar backwards. Vendor files declaring a non-Unity axis system
    ship it ON (the Felis fixture: Z-up/+Y-front, ``bakeAxisConversion: 1``) and
    then sit bone-for-bone on this export, so the two assets agree at *opposite*
    settings by construction: copying a vendor's importer settings onto an owned
    re-export is exactly how to break it.

    **Scale:** ``FBX_SCALE_ALL`` writes a ``UnitScaleFactor=100`` file, and this
    function **bakes every parked object scale into the data first** (via
    ``scene_utils.normalize_object_scale``), so the written file always carries
    identity node scales — the canonical layout, identical to what meter-unit
    vendors ship. Pass ``bake_object_scale=False`` to export the transforms
    as-is; ``keep_object_rotation`` governs only the rotation gate and does not
    suppress this.

    That apply is **permanent and unreported by the scene**: the scale moves out
    of the object transform and into vertices, shape keys and rest bones, the
    emitted ``AVATARPREP:`` line is the only record, and the scene is NOT
    restored afterwards (unlike the rotation gate, which clears unapplied).

    It preserves world layout **for the cases it accepts**, which is not all of
    them — ``check_scale_normalizable`` runs first and refuses the rest, because
    two were measured to move geometry by metres rather than relocate a number: a
    posed armature (the bake rescales rest bones but not pose translation
    channels — 9.9 m on a 0.01 rig) and shear from a non-uniform scale over a
    rotated descendant (0.041 m, silently dropped in the re-decomposition). Every
    refusal is raised **before** the first mutation, so a refused export leaves
    the scene untouched. ``normalize_object_scale`` owns the rest of the
    reasoning: why the accepted cases are not gated on the parked value, and why
    parents are applied before children.

    What it fixes: from a cm-unit (``UnitScaleFactor=1``) source the importer
    parks a 0.01 object scale, and without the apply the written file carried
    ``Lcl Scaling 0.01`` over centimetre-magnitude geometry on its **root-level
    nodes only** — measured on Chocolat, 21 of 289 Model nodes (the armature
    ``Null`` plus 20 root-level ``Mesh`` siblings; all 268 ``LimbNode``s were
    identity, so it never compounded). The import snapshot's
    ``unit_scale_factor`` names the source's class, but see
    ``normalize_object_scale``: the parked *value* does not, which is why this
    does not gate on it. Scale tracks the source file's unit, NOT the
    orientation class above — the two are independent, and in the survey the
    (90,0,0) class is 35 cm-unit against 18 meter-unit files.

    Why it mattered, measured in Unity on that Chocolat pair: the **vendor**
    cm-unit file imports clean, because Unity's own unit normalization does the
    work (``useFileScale=True, fileScale=0.01``, all 290 transforms at
    ``localScale`` 1). An un-normalised re-export is an honest meter-unit file,
    so Unity sets ``fileScale=1`` and has nothing left to normalize with: the
    0.01 lands as a literal ``localScale`` on 21 GameObjects with bones at
    centimetre ``localPosition``. World bounds and humanoid ``humanScale`` are
    identical either way — so this never broke an avatar, it made our owned
    re-export structurally worse than the vendor original it replaces, and put
    it 100x off any meter-clean rig it is merged or animated against.

    **Scope: this covers mesh-only prop FBXs too.** A cm-unit prop with no
    armature parks the same 0.01 on its mesh objects (measured on
    ``Telmy_Helmet.fbx``: the export wrote ``Lcl Scaling 0.01``), so props were
    never exempt from this the way they are exempt from the orientation gate
    above — do not read the two scopes as one.

    Owned exports do not otherwise mimic the source: Unity normalizes file units
    at import, and world-space parity there is the owning skill's gate. The two
    export paths now agree on unit layout as well as world layout —
    ``merge_armatures``' ``transform_apply`` bakes scale as well as rotation, and
    reaches the same place by the same means. (``FBX_SCALE_NONE`` instead writes
    a cm-unit file with 100x root node scales; measured, and not what any probed
    vendor ships.)

    ``armature_obj`` scopes the export to one rig: it selects that armature plus
    its bound meshes and exports selection-only. Because a scoped export is by
    construction an *owned* re-export (the owned meshes reuse the vendor materials
    by GUID in Unity, so nothing needs embedding) it also forces
    ``path_mode='STRIP'`` and ``embed_textures=False`` — otherwise Blender would
    re-embed textures by the vendor author's unresolvable absolute paths, emitting
    warnings and junk sub-assets. With ``armature_obj=None`` the whole scene is
    exported (``use_selection=False``) on the VRChat embed recipe, matching CATS.

    Returns the filepath written.
    """
    if object_types is None:
        object_types = {'EMPTY', 'ARMATURE', 'MESH', 'OTHER'}

    from . import scene_utils

    # A non-unit scene scale silently changes the exported unit layout
    # (measured: METRIC scale_length=0.01 writes UnitScaleFactor~1, the cm-unit
    # layout, breaking the canonical-layout contract above; system NONE ignores
    # scale_length). Refuse loud — the remedy is the scene setting, not a flag.
    us = bpy.context.scene.unit_settings
    if us.system != 'NONE' and abs(us.scale_length - 1.0) > 1e-9:
        raise ValueError(
            "scene unit_settings.scale_length=%r would change the exported unit "
            "layout away from the canonical meter-unit file (UnitScaleFactor=100); "
            "set scene.unit_settings.scale_length = 1.0 (rescale the content if it "
            "relied on it) and re-export" % us.scale_length)

    # ``select_all`` (and the FBX exporter) poll for OBJECT mode; a caller that left
    # the scene in POSE/EDIT — apply_proportion_edge exits in POSE on its object-only
    # edge path — otherwise crashes ``select_all.poll() failed, context is incorrect``.
    # Force OBJECT here so apply-then-export in one script is safe for every caller.
    active = bpy.context.view_layer.objects.active
    if active is not None and active.mode != 'OBJECT':
        scene_utils.op_override(bpy.ops.object.mode_set,
                                {'active_object': active, 'object': active},
                                mode='OBJECT')

    if armature_obj is not None:
        bpy.ops.object.select_all(action='DESELECT')
        armature_obj.select_set(True)
        for m in scene_utils.get_bound_meshes(armature_obj):
            m.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        use_selection = True
        path_mode = 'STRIP'
        embed_textures = False

    # --- Every refusal runs BEFORE the irreversible scale bake below. -----------
    # The bake has no undo, so a refusal raised after it leaves the caller with a
    # permanently rewritten scene AND no file — strictly worse than either
    # outcome. Anything that can refuse this export belongs above this line.
    if keep_object_rotation:
        candidates = []
    elif armature_obj is not None:
        candidates = [armature_obj]
    else:
        candidates = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    # Refuse a parented armature rather than guess which frame the gate should
    # judge. Under a rotated parent, the object's world rotation and the delta a
    # local clear produces are different rotations that can disagree about
    # whether the up axis moves, and no reading of one is defensible for the
    # other. Nothing here produces a parented armature (wm.fbx_import creates
    # them at root), so this closes a question rather than blocking real work —
    # and it matches the preflight merge_armatures already applies.
    parented = [o.name for o in candidates if o.parent is not None]
    if parented:
        raise ValueError(
            "armature(s) %s have a parent object; the axis-convention gate cannot "
            "judge a rotation split between parent and object. Clear or apply the "
            "parent relation, or pass keep_object_rotation=True to export the "
            "rotations as-is" % ", ".join(repr(n) for n in parented))

    if armature_obj is not None:
        scale_scope = [armature_obj] + scene_utils.get_bound_meshes(armature_obj)
    else:
        scale_scope = list(bpy.context.scene.objects)
    if bake_object_scale:
        scene_utils.check_scale_normalizable(scale_scope)

    # --- Mutation starts here. -------------------------------------------------
    # Bake parked object scale into the data (see the **Scale** section above).
    #
    # Running this BEFORE the rotation gate below is currently free rather than
    # load-bearing, and the difference matters if you touch either. The gate is
    # scale-INVARIANT today, so moving this call after it changes no field of the
    # written file — measured: node rotation, node scales and vertex data
    # identical, only the FBX's embedded timestamp differs, which already differs
    # between two identical runs. ``to_quaternion`` normalizes columns, so
    # ``T*R*S`` with diagonal S recovers R whatever S is; and the gate decides on
    # the clear DELTA, ``(T*S)(T*R*S)^-1 = T*R^-1*T^-1``, where S cancels
    # adjacently. Shear would break the first of those, but it takes a parent
    # chain and parented armatures raise above.
    #
    # Lose that invariance and the order becomes load-bearing at once: measured on
    # a gate handed the scale-carrying matrix, a non-uniformly scaled front-axis
    # rig classifies 'cleared' normalize-first and 'preserved' clear-first, and
    # the second ships the avatar backwards. So no test can pin the order (nothing
    # observable changes while the gate is invariant), but the invariance that
    # makes it safe is pinned — tests/test_fbx_export.py 11c fails before the
    # order can silently start to matter.
    applied = scene_utils.normalize_object_scale(scale_scope) if bake_object_scale else []
    if applied:
        # One line, not one per object: a cm-unit avatar parks the same scale on
        # every root (21 objects on Chocolat), and 21 identical lines would bury
        # the rotation line printed right after. The values are each object's own
        # scale at the moment it was applied — a child reads the compounded value
        # its parent's apply pushed onto it, so more than one value here is normal
        # and is NOT evidence of more than one conversion.
        values = sorted({s for _, s in applied})
        print("AVATARPREP: export applied parked object scale into object data on "
              "%d object(s) — values %s, objects %s. The written file carries "
              "identity node scales. This is PERMANENT and the scene is NOT "
              "restored afterwards (unlike the rotation gate below) — see "
              "export_unity_fbx's Scale docstring"
              % (len(applied), ", ".join(repr(v) for v in values),
                 ", ".join(repr(n) for n, _ in applied)))

    kwargs = dict(
        filepath=filepath,
        object_types=object_types,
        use_mesh_modifiers=use_mesh_modifiers,
        add_leaf_bones=add_leaf_bones,
        bake_anim=bake_anim,
        apply_scale_options=apply_scale_options,
        path_mode=path_mode,
        embed_textures=embed_textures,
        use_selection=use_selection,
    )
    kwargs.update(extra)

    # Neutralise the importer's axis-convention residue on every exported
    # armature (see docstring), restore after. The gate lives in
    # clear_axis_convention_rotation: a residue that leaves the up axis fixed is
    # cleared, one that MOVES it is preserved. Children ride along via parenting;
    # a modifier-bound NON-descendant mesh (a bound shape get_bound_meshes
    # supports) is carried by the same delta inside the helper — otherwise the
    # file would ship its geometry 180° off the skeleton.
    undo = []
    # Seeded, not empty: a mesh parented to one candidate and modifier-bound to
    # another rides along with its parent, so whichever rig the loop reaches
    # first must not also move it explicitly, or it lands at delta**2 — 180 deg
    # off both skeletons. The clear records the descendants it skips, which
    # covers parent-first; scene order decides which comes first, so seed the
    # whole set up front rather than depending on it.
    moved = set()
    for o in candidates:
        moved |= scene_utils.carried_by_parenting(o)
    try:
        for o in candidates:
            # matrix_world is stale after a direct rotation write, and this read
            # is an oracle now (the tests assert on the emitted line), so a caller
            # that set rotation_euler and exported without an update would print
            # one rotation while the gate decided on another.
            bpy.context.view_layer.update()
            old_rot = tuple(round(math.degrees(a), 3)
                            for a in o.matrix_world.to_euler())
            # Report on ``status``, never on ``delta``: a preserved residue
            # returns an IDENTITY delta, so a delta-keyed message goes silent on
            # exactly the class this gate exists for.
            status, _delta, u = scene_utils.clear_axis_convention_rotation(o, moved)
            undo += u
            if status == 'cleared':
                print("AVATARPREP: export cleared object rotation on %r "
                      "(was %s deg; axis-convention residue about the up axis — "
                      "pass keep_object_rotation=True if it was deliberate; see "
                      "export_unity_fbx's orientation docstring)"
                      % (o.name, old_rot))
            elif status == 'preserved':
                # Says the rotation is preserved WHOLE, not that it is purely an
                # up-axis conversion: a rotation that also spins about the up axis
                # keeps that spin too, and would export front-reversed. Claiming
                # purity here would be false for exactly that residue, and this
                # line is the only signal the reader gets.
                print("AVATARPREP: export preserved object rotation on %r "
                      "(%s deg) whole — it moves the up axis, so clearing it "
                      "would export the rig tipped onto its face. Any rotation "
                      "about the up axis it also carries is preserved with it, "
                      "so check facing if that value is not a pure axis swap "
                      "(see export_unity_fbx's orientation docstring)"
                      % (o.name, old_rot))
        bpy.ops.export_scene.fbx('EXEC_DEFAULT', **kwargs)
    finally:
        scene_utils.restore_transforms(undo)
    return filepath
