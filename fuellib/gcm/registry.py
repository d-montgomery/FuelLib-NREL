"""Registry for GCM (Group Contribution Method) implementations."""


_registry = {}


def register_gcm(name: str, cls: type) -> None:
    """
    Register a GCM implementation under a given name.

    :param name: Unique name to register the GCM under (e.g. "gani").
    :type name: str
    :param cls: GCM class to register.
    :type cls: type[GCMMethod]
    """
    _registry[name] = cls


def get_gcm(name: str) -> type:
    """
    Retrieve a registered GCM class by name.

    :param name: Name the GCM was registered under.
    :type name: str
    :return: The registered GCM class.
    :rtype: type[GCMMethod]
    :raises KeyError: If no GCM is registered under the given name.
    """
    if name not in _registry:
        raise KeyError(
            f"No GCM registered under name '{name}'. Available: {sorted(_registry)}"
        )
    return _registry[name]


__all__ = ["get_gcm", "register_gcm"]
