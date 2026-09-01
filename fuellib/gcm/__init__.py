"""GCM (Group Contribution Method) abstraction package."""

# Import concrete implementations to trigger their registration.
from . import gani  # noqa: F401
from .base import GCMMethod
from .registry import get_gcm, register_gcm

__all__ = ["GCMMethod", "get_gcm", "register_gcm"]
