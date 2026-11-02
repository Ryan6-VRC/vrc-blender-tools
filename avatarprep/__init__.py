"""AvatarPrep - VRChat avatar prep tools for Blender 5.x.

Independently implements techniques popularized by the dormant CATS Blender
Plugin. There is intentionally NO upper Blender-version guard here: CATS shipped
``if bpy.app.version >= (5, 1): unregister()``, which broke it on Blender 5.1+.
"""

from . import operators
from . import ui


def register():
    operators.register()
    ui.register()


def unregister():
    ui.unregister()
    operators.unregister()
