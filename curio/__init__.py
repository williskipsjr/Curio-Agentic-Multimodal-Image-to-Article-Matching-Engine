"""Curio — Agentic Multimodal Image-to-Article Matching Engine."""

import sys
import types

__version__ = "0.1.0"

# Compatibility shim for sniffio:
# sniffio checks `if 'curio' in sys.modules: from curio.meta import curio_running`.
# Since our project package is named 'curio', sniffio mistakes it for the legacy
# curio async framework. Providing curio.meta.curio_running returning False resolves this.
if "curio.meta" not in sys.modules:
    _meta = types.ModuleType("curio.meta")
    _meta.curio_running = lambda: False  # type: ignore[attr-defined]
    sys.modules["curio.meta"] = _meta

