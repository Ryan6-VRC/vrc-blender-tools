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

    **Scope: ARMATURE objects only, and at most ONE armature per export.**
    Mesh-only prop FBXs have no armature and are never touched here — which is
    exactly why they are correct today, the same fact this rule encodes. A
    whole-scene export of a multi-armature scene REFUSES up front: the gate
    decides one clear for one rig, no sanctioned workflow exports a multi-rig
    scene whole (own-mergeable exports scoped precisely so an appended
    disposable reference body never ships; own-base merges to one rig first),
    and no surveyed vendor file imports more than one armature. Merge the rigs
    first (``merge_armatures``), scope to one (``armature_obj``), or pass
    ``keep_object_rotation=True`` to export every transform as-is.
    Note also that nothing in the ``avatarprep_`` stamp
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
    ``scene_utils.normalize_object_scale``), so the written file carries identity
    **object** scale on its Model nodes — the canonical layout, identical to what
    meter-unit vendors ship — **or the export refuses**. Pass
    ``bake_object_scale=False`` to export the transforms as-is;
    ``keep_object_rotation`` governs only the rotation gate and does not suppress
    this.

    The guarantee is about **object** scale, and only object scale. A vendor can
    also ship non-unit scale in the armature's REST BONE data, which surfaces as
    ``Lcl Scaling`` on ``LimbNode``s and which no object-scale bake can reach by
    construction (measured on Telmy: ``Breast_2_L`` 0.99785 and its ``_end``
    1.002155 survive an otherwise fully-normalised export). Read a non-identity
    ``LimbNode`` scale as vendor rest data, not as a failure of this function.

    That apply is **permanent and unreported by the scene**: the scale moves out
    of the object transform and into vertices, shape keys and rest bones, the
    emitted ``AVATARPREP:`` line is the only record, and the scene is NOT
    restored afterwards (unlike the rotation gate, which clears unapplied).

    It preserves world layout **for the cases it accepts**, which is not all of
    them — ``check_scale_normalizable`` runs first and refuses the rest. Two of
    those refusals exist because the bake was measured to move geometry by metres
    rather than relocate a number: a posed armature (the bake rescales rest bones
    but not pose translation channels — 9.9 m on a 0.01 rig) and shear from a
    non-uniform scale over a rotated descendant (0.041 m, silently dropped in the
    re-decomposition).

    A third refuses for a different reason — the layout promise above simply
    cannot be kept. An ancestor OUTSIDE this export's scope is neither baked nor
    (when scoped) written, so its scale collapses into the in-scope descendant's
    own node: measured at ``Lcl Scaling (2,2,2)`` on a synthetic scope boundary,
    and 30 of 590 Model nodes at 0.01 on a cm-unit vendor import whose meshes
    hang off a root EMPTY. Where that ancestor is also non-uniform it is the
    shear case as well, invisible to the condition above and measured exporting
    silently at (1.58114, 1.58114, 1.0) — so this one refusal covers both a
    layout break and a geometry break.

    Every refusal is raised **before** the first mutation, so a refused export
    leaves the scene untouched. ``normalize_object_scale`` owns the rest of the
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
    # At most ONE armature in scope — the docstring's rule, enforced before any
    # other reasoning about the rigs can be needed. Everything a multi-rig path
    # would have to weigh (which rig's rotation is the reference, how a shared
    # mesh moves, whether a parented rig's world rotation is its own) exists
    # only on a path no sanctioned workflow takes, and refusing is also the
    # honest fix for the real accident that path enabled: a whole-scene export
    # silently shipping an appended disposable reference body.
    if len(candidates) > 1:
        raise ValueError(
            "%d armatures are in this export's scope (%s), and this export "
            "handles exactly one. Merge the rigs into one first "
            "(merge_armatures), or delete the extra armature; from the "
            "CLI/API, scope the export to one rig (--armature / "
            "armature_obj=...); or pass keep_object_rotation=True to export "
            "every transform as-is. See export_unity_fbx's orientation "
            "docstring."
            % (len(candidates),
               ", ".join(sorted(repr(o.name) for o in candidates))))
    # Refuse a parented armature rather than guess which frame the gate should
    # judge. Under a rotated parent, the object's world rotation and the delta a
    # local clear produces are different rotations that can disagree about
    # whether the up axis moves, and no reading of one is defensible for the
    # other — matching the preflight merge_armatures applies.
    #
    # Narrowed to a parented armature that ALSO carries its own rotation, because
    # only then is there a split to judge. The gate decides on the clear DELTA
    # (see clear_axis_convention_rotation), and for an identity local rotation
    # that delta is identity: the gate returns 'noop', reads nothing ambiguous,
    # and writes nothing. Refusing there blocked an export over a decision that
    # was never being made.
    #
    # This is the whole cm-unit root-Null class, and it is not rare: wm.fbx_import
    # renders a source whose only root node is a Null as an EMPTY parent carrying
    # the axis conversion AND the unit conversion, leaving the armature itself at
    # identity. Measured across the 131-file vendor survey, 17 of the 42 cm-unit
    # files import to exactly that shape (against 0 of 89 meter-unit), all sharing
    # one signature — UpAxis=1,+1 / FrontAxis=2 / UnitScaleFactor=1 — and no root
    # Null anywhere in the corpus carries scaling without rotation. Both sampled
    # files (Lunary 591 nodes, Telmy 402) import with the parent at (90,0,0) and
    # 0.01 while the armature reads (0,0,0), so the gate is a 'noop' on every one
    # of them. Their parked scale is still refused — by the out-of-scope-ancestor
    # condition in check_scale_normalizable, which names the EMPTY and is the
    # accurate diagnosis for this shape.
    # ``has_own_rotation`` reads matrix_basis, not rotation_euler: those are
    # separate RNA fields, and a QUATERNION- or AXIS_ANGLE-mode armature carrying
    # 180 deg reads rotation_euler (0,0,0), which would sail straight through the
    # very gate this raises.
    parented = [o.name for o in candidates
                if o.parent is not None and scene_utils.has_own_rotation(o)]
    if parented:
        raise ValueError(
            "armature(s) %s have BOTH a parent object and their own object "
            "rotation; the axis-convention gate cannot judge a rotation split "
            "between parent and object. Clear or apply the parent relation, or "
            "pass keep_object_rotation=True to export the rotations as-is"
            % ", ".join(repr(n) for n in parented))

    # Constraints make matrix_world depsgraph-derived, so the clear's carry
    # below does not stick and silently does nothing of what it says.
    # merge_armatures preflights exactly this on its own apply path.
    # check_scale_normalizable does not cover it: that one refuses only
    # _SCALE_CONSTRAINTS, and only at a non-unit evaluated scale, so
    # COPY_ROTATION / COPY_TRANSFORMS / CHILD_OF pass.
    if candidates:
        constrained = [(o.name, o.constraints) for o in candidates if o.constraints]
        for c in candidates:
            constrained += [(m.name, m.constraints)
                            for m in scene_utils.get_bound_meshes(c)
                            if m.constraints and not scene_utils._is_descendant(m, c)]
        if constrained:
            seen_c = set()
            named = [(n, cons) for n, cons in constrained
                     if not (n in seen_c or seen_c.add(n))]
            raise ValueError(
                "object(s) %s carry constraint(s) that make matrix_world "
                "depsgraph-derived, so the axis-convention clear and its carry "
                "would not stick and would silently do nothing. Apply or remove "
                "them, or pass keep_object_rotation=True to skip the gate entirely"
                % ", ".join("%r (%s)" % (n, ", ".join(repr(c.name) for c in cons))
                            for n, cons in named))

    if armature_obj is not None:
        scale_scope = [armature_obj] + scene_utils.get_bound_meshes(armature_obj)
    else:
        scale_scope = list(bpy.context.scene.objects)
    if bake_object_scale:
        scene_utils.check_scale_normalizable(scale_scope)

    # --- Mutation starts here. -------------------------------------------------
    # Bake parked object scale into the data (see the **Scale** section above).
    #
    # BEFORE the rotation gate below, and load-bearing — though not for the reason
    # once given here. The gate's VERDICT is scale-invariant either way:
    # ``to_quaternion`` normalizes columns, and the gate decides on the clear
    # DELTA, ``(T*S)(T*R*S)^-1 = T*R^-1*T^-1``, where S cancels adjacently. So a
    # reorder cannot misclassify a rig, and on a rig whose meshes are all
    # descendants it changes nothing at all.
    #
    # The order matters because the gate does not only READ. For a modifier-bound
    # mesh that is NOT the armature's descendant it also writes
    # ``m.matrix_world`` and snapshots that mesh's ``matrix_basis`` for the undo
    # replayed in the ``finally`` below. Both are order-sensitive, measured
    # end-to-end through this function:
    #
    #   * The snapshot captures the mesh's own scale. Clear-first takes it PRE-bake,
    #     normalize then bakes that scale into the data, and the restore replays
    #     the old basis over baked data — a (2,3,4) bound mesh comes back 2x3x4
    #     too large, silently, with a byte-equivalent file.
    #   * The write gives the mesh a local rotation. Under a non-uniformly scaled
    #     PARENT that is ``check_scale_normalizable``'s shear case, which
    #     ``normalize_object_scale`` re-validates: clear-first raises 'sheared'
    #     with the scene already mutated and no file, breaking the refusals-before-
    #     mutation invariant stated above.
    #
    # Baking first removes both — the parent is uniform before the gate rotates
    # the child, and the snapshot is taken at scale 1. The armature preflight above
    # does not cover this: it reads ``candidates``, and these are bound meshes.
    # Pinned by tests/test_fbx_export.py 11c; 11b pins the verdict invariance
    # separately, which the ``bake_object_scale=False`` path needs regardless.
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
              "identity OBJECT scale on its Model nodes (a vendor's non-unit REST "
              "BONE scale lands on LimbNodes and is not reachable by this bake). "
              "This is PERMANENT and the scene is NOT restored afterwards (unlike "
              "the rotation gate below) — see export_unity_fbx's Scale docstring"
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

    # Neutralise the importer's axis-convention residue on the one exported
    # armature (see docstring), restore after. The gate lives in
    # clear_axis_convention_rotation: a residue that leaves the up axis fixed is
    # cleared, one that MOVES it is preserved. Children ride along via parenting;
    # a modifier-bound NON-descendant mesh (a bound shape get_bound_meshes
    # supports) is carried by the same delta inside the helper — otherwise the
    # file would ship its geometry 180° off the skeleton.
    undo = []
    try:
        if candidates:
            arm = candidates[0]
            bpy.context.view_layer.update()  # matrix_world is stale after direct writes
            old_rot = tuple(round(math.degrees(a), 3)
                            for a in arm.matrix_world.to_euler())
            # Report on ``status``, never on ``delta``: a preserved residue
            # returns an IDENTITY delta, so a delta-keyed message goes silent
            # on exactly the class this gate exists for.
            status, _delta, undo = scene_utils.clear_axis_convention_rotation(arm)
            _report_rotation(status, arm.name, old_rot)
        bpy.ops.export_scene.fbx('EXEC_DEFAULT', **kwargs)
    finally:
        scene_utils.restore_transforms(undo)
    return filepath


def _report_rotation(status, name, old_rot):
    """Emit the one ``AVATARPREP:`` line the rotation gate's verdict earns.

    Split out so the wording lives in one place — the tests assert on these
    lines."""
    if status == 'cleared':
        print("AVATARPREP: export cleared object rotation on %r "
              "(was %s deg; axis-convention residue about the up axis — "
              "pass keep_object_rotation=True if it was deliberate; see "
              "export_unity_fbx's orientation docstring)."
              % (name, old_rot))
    elif status == 'preserved':
        # Says the rotation is preserved WHOLE, not that it is purely an up-axis
        # conversion: a rotation that also spins about the up axis keeps that
        # spin too, and would export front-reversed. Claiming purity here would
        # be false for exactly that residue, and this line is the only signal the
        # reader gets.
        print("AVATARPREP: export preserved object rotation on %r "
              "(%s deg) whole — it moves the up axis, so clearing it "
              "would export the rig tipped onto its face. Any rotation "
              "about the up axis it also carries is preserved with it, "
              "so check facing if that value is not a pure axis swap "
              "(see export_unity_fbx's orientation docstring)."
              % (name, old_rot))
