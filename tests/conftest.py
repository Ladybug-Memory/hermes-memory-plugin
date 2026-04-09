from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Stub the agent packages that only exist inside the Hermes source tree.
# Tests that exercise MemoryManager integration are skipped when these are
# absent; unit-level tool and lifecycle tests run fine with the stubs.
# ---------------------------------------------------------------------------

agent_pkg = types.ModuleType("agent")
memory_provider_mod = types.ModuleType("agent.memory_provider")


class MemoryProvider:
    pass


memory_provider_mod.MemoryProvider = MemoryProvider
sys.modules.setdefault("agent", agent_pkg)
sys.modules.setdefault("agent.memory_provider", memory_provider_mod)

# ---------------------------------------------------------------------------
# Register the root __init__.py as plugins.memory.ladybug so the test file
# can use `from plugins.memory.ladybug import LadybugMemoryProvider` without
# any changes.
# ---------------------------------------------------------------------------

for _pkg in ("plugins", "plugins.memory"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))

_spec = importlib.util.spec_from_file_location("plugins.memory.ladybug", ROOT / "__init__.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["plugins.memory.ladybug"] = _mod
_spec.loader.exec_module(_mod)
