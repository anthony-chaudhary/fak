"""Shared fixtures: mock PriskvClient that records calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest


@dataclass
class MockClient:
    """Drop-in stand-in for cama_client.PriskvClient.

    Stores everything for assertions. mget_rdma returns 0 (hit) for any key it
    has seen via mset, else -1 (miss).
    """

    stored: set[str] = field(default_factory=set)
    reg_calls: list[tuple[int, int]] = field(default_factory=list)
    dereg_calls: list[int] = field(default_factory=list)
    mset_calls: list[list[str]] = field(default_factory=list)
    mget_calls: list[list[str]] = field(default_factory=list)
    mexists_calls: list[list[str]] = field(default_factory=list)
    closed: bool = False
    _next_handle: int = 1

    def reg_memory(self, ptr: int, size: int, buf=None) -> int:
        self.reg_calls.append((ptr, size))
        h = self._next_handle
        self._next_handle += 1
        return h

    def dereg_memory(self, handle: int) -> None:
        self.dereg_calls.append(handle)

    def mset(self, keys, sgls, ttl_ms: int = 0):
        self.mset_calls.append(list(keys))
        for k in keys:
            self.stored.add(k)
        return [0] * len(keys)

    def mget_rdma(self, keys, sgls, sizes=None):
        self.mget_calls.append(list(keys))
        return [0 if k in self.stored else -1 for k in keys]

    def mexists(self, keys):
        self.mexists_calls.append(list(keys))
        return [1 if k in self.stored else 0 for k in keys]

    def mdel(self, keys):
        for k in keys:
            self.stored.discard(k)
        return 0

    def info(self):
        return {"mock": True}

    def close(self):
        self.closed = True


class MockSGL:
    def __init__(self, ptr: int, size: int, reg_handle: int = 1):
        self.ptr = ptr
        self.size = size
        self.reg_handle = reg_handle


@pytest.fixture
def mock_client():
    return MockClient()


@pytest.fixture
def patch_cama_client(monkeypatch, mock_client):
    """Patch ``cama_client.PriskvClient`` / ``SGL`` to the mock."""
    import sys
    import types
    mod = types.ModuleType("cama_client")
    mod.PriskvClient = lambda *a, **kw: mock_client
    mod.SGL = MockSGL
    mod.CamaServerOverloadError = type("CamaServerOverloadError", (Exception,), {})
    mod.CamaOOMError = type("CamaOOMError", (Exception,), {})
    mod.CamaNotReadyError = type("CamaNotReadyError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "cama_client", mod)
    monkeypatch.setitem(sys.modules, "l3_client", mod)
    return mock_client
