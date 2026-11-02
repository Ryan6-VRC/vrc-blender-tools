"""Rename scene objects as a SET, so a swap is expressible and a collision refuses.

``own-base``'s canonical rename is ``Face -> Body, Body -> Body_Base``, in which ``Body``
is both a source and a destination. Renaming pair-by-pair does not express that:
Blender auto-suffixes on collision, so ``Face`` lands on ``Body.001`` and the mismatch
is silent — reported names look right while the object carries the wrong one, and it
surfaces much later as a Unity clip binding that resolves to nothing.

Object names are unique across ``bpy.data.objects``, spanning every scene and object
type, so the collision scan is file-wide: an armature named ``Body`` in another scene
collides with a mesh rename here.

The mesh datablock name is deliberately untouched. It diverges from the object name
routinely and nothing downstream reads it: Unity's clip paths and name-based material
matching key on the object name, which is what becomes the GameObject.
"""
from typing import Any, Dict, List

import bpy


class RenameRefused(Exception):
    """Raised when the requested set cannot be applied as asked: before any mutation when
    the plan refuses, mid-set when an apply step loses its object or name. Carries ``offenders`` (list of str) so a door can print them individually."""

    def __init__(self, message, offenders):
        super().__init__(message)
        self.offenders = offenders


def _temp_name(taken):
    """A name no object holds, for breaking a rename cycle. Carries no source name, which
    could push it past the name limit."""
    i = 0
    while True:
        candidate = "_avatarprep_rename_%d" % i
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
        limit = bpy.types.ID.bl_rna.properties["name"].length_max - 1
        if len(new.encode("utf-8")) > limit:
            offenders.append("target name %r is %d bytes; Blender would truncate it to %d"
                             % (new, len(new.encode("utf-8")), limit))
            continue
        holder = objs.get(new)
        # A name freed by another pair in the set is not a collision.
        if holder is not None and new not in pairs:
            offenders.append("target name %r is already held by %s %r (not being "
                             "renamed away by this set)" % (new, holder.type, holder.name))

    for new, olds in sorted(by_new.items()):
        if len(olds) > 1:
            offenders.append("two or more renames target %r: %s"
                             % (new, ", ".join(sorted(olds))))

    if offenders:
        return {"offenders": offenders, "warnings": warnings, "order": [], "cycles": []}

    remaining = {o: n for o, n in pairs.items() if o != n}
    taken = {ob.name for ob in objs}
    order: List[Dict[str, Any]] = []
    cycles: List[str] = []
    while remaining:
        free = [o for o, n in sorted(remaining.items()) if n not in taken]
        if not free:
            old = sorted(remaining)[0]
            tmp = _temp_name(taken)
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
    ``renamed`` is read back off the objects after assignment, never echoed from the
    request, because Blender silently auto-suffixes on a collision. A residual suffix is
    a refusal, not a warning.

    ``source_map`` is ``{ourName: sourceName}``, the inverse the Unity material-copy step
    consumes.

    Under ``whatif`` the checks and the ordering run and nothing is touched, so the
    read-back does not run."""
    plan = plan_renames(pairs)
    if plan["offenders"]:
        raise RenameRefused("rename_objects refused %d name(s)" % len(plan["offenders"]),
                            plan["offenders"])

    result = {"renamed": {}, "source_map": {}, "order": plan["order"],
              "cycles": plan["cycles"], "warnings": list(plan["warnings"]),
              "whatif": bool(whatif)}
    if whatif:
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
