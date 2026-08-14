"""Rename scene objects as a SET, so a swap is expressible and a collision refuses.

``own-base``'s canonical rename is ``Face -> Body, Body -> Body_Base``, in which ``Body``
is both a source and a destination. Renaming pair-by-pair does not express that:
Blender auto-suffixes on collision, so ``Face`` lands on ``Body.001`` and the mismatch
is silent — reported names look right while the object carries the wrong one, and it
surfaces much later as a Unity clip binding that resolves to nothing.

Object names are unique across ``bpy.data.objects`` — spanning every SCENE and every
object TYPE — so the collision scan has to be wider than the per-scene resolvers
elsewhere in this package (``resolve_arm``, ``find_armature``, ``get_bound_meshes``),
which are correctly per-scene for their own purposes. An armature named ``Body`` in
another scene collides with a mesh rename here; measured.

The mesh DATABLOCK name is deliberately untouched. It diverges from the object name
routinely on real assets (``Svak (Owned)``, a shipped owned base, carries object
``Body`` over datablock ``Plane.002``; an FBX-imported base measured 22 of 22
diverged), it is vendor modelling residue, and nothing downstream reads it: Unity's
clip paths and name-based material matching key on the object name, which is what
becomes the GameObject. Syncing them would rename residue no consumer reads and bury
the load-bearing renames in noise.

Pure bpy: no operator, no UI.
"""
from typing import Any, Dict, List

import bpy


class RenameRefused(Exception):
    """Raised before any mutation when the requested set cannot be applied as asked.
    Carries ``offenders`` (list of str) so a door can print them individually."""

    def __init__(self, message, offenders):
        super().__init__(message)
        self.offenders = offenders


def _temp_name(taken, seed):
    """A name no object holds, for breaking a rename cycle."""
    i = 0
    while True:
        candidate = "_avatarprep_rename_%s_%d" % (seed, i)
        if candidate not in taken:
            return candidate
        i += 1


def plan_renames(pairs) -> Dict[str, Any]:
    """Validate ``{old: new}`` against the scene and order it. Mutates nothing.

    Returns ``{"offenders", "warnings", "order", "cycles"}``. ``order`` is the sequence
    of ``(old, new)`` steps to apply; a member of ``cycles`` is routed through a temp
    name because no ordering can free its target first (``A -> B, B -> A``)."""
    objs = bpy.data.objects
    offenders: List[str] = []
    warnings: List[str] = []

    by_new: Dict[str, List[str]] = {}
    for old, new in pairs.items():
        by_new.setdefault(new, []).append(old)

    for old, new in sorted(pairs.items()):
        if not new:
            offenders.append("rename %r -> empty name (a dropped '=' in --rename lands "
                             "here; Blender would name the object 'Object')" % old)
            continue
        if old not in objs:
            offenders.append("object not found: %r" % old)
            continue
        if old == new:
            warnings.append("rename %r -> itself; no-op" % old)
            continue
        holder = objs.get(new)
        # A name freed by another pair in the same set is NOT a collision -- that is
        # the whole point of resolving as a set. A name held by an object nobody is
        # renaming away is.
        if holder is not None and new not in pairs:
            offenders.append("target name %r is already held by %s %r (not being "
                             "renamed away by this set)" % (new, holder.type, holder.name))

    for new, olds in sorted(by_new.items()):
        if len(olds) > 1:
            offenders.append("two or more renames target %r: %s"
                             % (new, ", ".join(sorted(olds))))

    if offenders:
        return {"offenders": offenders, "warnings": warnings, "order": [], "cycles": []}

    # Topological order: apply a pair once its target name is free. Whatever is left
    # when nothing is free is a cycle, which no ordering can solve.
    remaining = {o: n for o, n in pairs.items() if o != n}
    taken = {ob.name for ob in objs}
    order: List[Dict[str, Any]] = []
    cycles: List[str] = []
    while remaining:
        free = [o for o, n in sorted(remaining.items()) if n not in taken]
        if not free:
            old = sorted(remaining)[0]
            tmp = _temp_name(taken, old.replace(" ", "_"))
            order.append({"from": old, "to": tmp, "via_temp": True})
            taken.discard(old)
            taken.add(tmp)
            remaining[tmp] = remaining.pop(old)
            cycles.append(old)
            continue
        for old in free:
            new = remaining.pop(old)
            order.append({"from": old, "to": new, "via_temp": False})
            taken.discard(old)
            taken.add(new)

    return {"offenders": offenders, "warnings": warnings, "order": order,
            "cycles": cycles}


def rename_objects(pairs, *, whatif=False) -> Dict[str, Any]:
    """Apply ``{old: new}`` as a set. Raises :class:`RenameRefused` before touching
    anything when the plan does not hold.

    Returns ``{"renamed", "source_map", "order", "cycles", "warnings", "whatif"}``.
    ``renamed`` is read back OFF THE OBJECTS after assignment, never echoed from the
    request: Blender silently auto-suffixes on a collision, so a report built from what
    was asked for is exactly the lie this door exists to prevent. A residual suffix is
    a refusal, not a warning.

    ``source_map`` is ``{ourName: sourceName}`` — the inverse the Unity material-copy
    step consumes, emitted so nobody maintains it by hand.

    Under ``whatif`` the set-level checks and the ordering are real and nothing is
    touched; the read-back assertion cannot run, since there is nothing to read back."""
    plan = plan_renames(pairs)
    if plan["offenders"]:
        raise RenameRefused("rename_objects refused %d name(s)" % len(plan["offenders"]),
                            plan["offenders"])

    result = {"renamed": {}, "source_map": {}, "order": plan["order"],
              "cycles": plan["cycles"], "warnings": list(plan["warnings"]),
              "whatif": bool(whatif)}
    if whatif:
        # The plan's final resting name per original object, temp hops collapsed.
        chain = {}
        for step in plan["order"]:
            origin = chain.pop(step["from"], step["from"])
            chain[step["to"]] = origin
        result["renamed"] = {origin: final for final, origin in chain.items()}
        result["source_map"] = {final: origin for final, origin in chain.items()}
        return result

    objs = bpy.data.objects
    applied = {}
    for step in plan["order"]:
        obj = objs.get(step["from"])
        if obj is None:
            raise RenameRefused(
                "rename_objects lost %r mid-set" % step["from"],
                ["object %r vanished between planning and applying" % step["from"]])
        obj.name = step["to"]
        if obj.name != step["to"]:
            raise RenameRefused(
                "rename_objects could not take the name %r" % step["to"],
                ["asked for %r, Blender assigned %r — the name was taken after "
                 "planning" % (step["to"], obj.name)])
        origin = applied.pop(step["from"], step["from"])
        applied[obj.name] = origin

    result["renamed"] = {origin: final for final, origin in applied.items()}
    result["source_map"] = {final: origin for final, origin in applied.items()}
    return result
