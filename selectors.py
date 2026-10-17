"""Bridge module bridging stdlib selectors and PANDA node selectors.

This ensures standard library modules (e.g. socket, urllib, pandas) can import
Python's built-in selectors module without collision or circular import errors,
while lazily exposing PANDA's NodeSelectionStrategy classes upon access.
"""

import importlib.util
import os
import sys
import sysconfig

# 1. Directly load and re-export the standard library selectors module from Python Lib
_stdlib_path = os.path.join(sysconfig.get_path("stdlib"), "selectors.py")
if os.path.exists(_stdlib_path):
    _spec = importlib.util.spec_from_file_location("_stdlib_selectors", _stdlib_path)
    _stdlib_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_stdlib_mod)
    for _attr in dir(_stdlib_mod):
        if not _attr.startswith("__"):
            globals()[_attr] = getattr(_stdlib_mod, _attr)


def __getattr__(name: str):
    """Lazily resolves PANDA node selectors on demand to prevent circular import."""
    import panda_gnn.selectors as _panda_selectors

    if hasattr(_panda_selectors, name):
        val = getattr(_panda_selectors, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'selectors' has no attribute '{name}'")
