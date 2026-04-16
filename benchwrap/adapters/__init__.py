"""
Adapter auto-discovery and registry.
Scans benchwrap/adapters/ for .py files that define BenchmarkAdapter subclasses.
"""

import os
import sys
import importlib
import inspect
from typing import Optional

from benchwrap.core.adapter import BenchmarkAdapter

# Registry of discovered adapters
_registry: dict[str, type[BenchmarkAdapter]] = {}
_info: dict[str, dict] = {}


def discover_adapters():
    """Scan adapter directories and register all BenchmarkAdapter subclasses."""
    global _registry, _info
    _registry = {}
    _info = {}

    adapter_dirs = [
        os.path.join(os.path.dirname(__file__), "custom"),
        os.path.dirname(__file__),
    ]

    for adapter_dir in adapter_dirs:
        if not os.path.isdir(adapter_dir):
            continue
        for fname in os.listdir(adapter_dir):
            if fname.endswith(".py") and not fname.startswith("_"):
                fpath = os.path.join(adapter_dir, fname)
                _load_adapter_file(fpath)


def _load_adapter_file(fpath: str):
    """Load a single adapter file and register any BenchmarkAdapter subclasses."""
    module_name = f"benchwrap.adapters.{os.path.basename(fpath)[:-3]}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, fpath)
        if not spec or not spec.loader:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Warning: Failed to load adapter {fpath}: {e}", file=sys.stderr)
        return

    # Find BenchmarkAdapter subclasses
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, BenchmarkAdapter)
            and obj is not BenchmarkAdapter
            and not inspect.isabstract(obj)
        ):
            try:
                instance = obj()
                adapter_name = instance.name()
                _registry[adapter_name] = obj
                _info[adapter_name] = {
                    "datasets": instance.datasets(),
                    "description": obj.__doc__ or "",
                    "module": module_name,
                    "class": name,
                }
            except Exception as e:
                print(f"Warning: Failed to instantiate {name}: {e}", file=sys.stderr)


def get_adapter(name: str) -> Optional[BenchmarkAdapter]:
    """Get an adapter instance by name."""
    if not _registry:
        discover_adapters()
    cls = _registry.get(name)
    return cls() if cls else None


def list_adapters() -> dict[str, dict]:
    """List all registered adapters with their info."""
    if not _registry:
        discover_adapters()
    return dict(_info)


# Auto-discover on import
discover_adapters()
