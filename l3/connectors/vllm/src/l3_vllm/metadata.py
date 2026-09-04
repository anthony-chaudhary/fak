"""Scheduler -> Worker metadata for l3-vllm-connector.

The scheduler-side connector populates this and the worker-side reads it on
``start_load_kv`` / ``save_kv_layer`` to know which blocks to load/save.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class LoadSpec:
    """One request's load directive."""

    request_id: str
    block_hashes: list[Any]    # vLLM block hashes (opaque to us)
    block_ids: list[int]       # vLLM internal block ids the data goes into
    num_external_tokens: int


@dataclass
class SaveSpec:
    """One request's save directive."""

    request_id: str
    block_hashes: list[Any]
    block_ids: list[int]


@dataclass
class L3KVConnectorMetadata(KVConnectorMetadata):
    loads: list[LoadSpec] = field(default_factory=list)
    saves: list[SaveSpec] = field(default_factory=list)


# Backward compatibility alias
CamaKVConnectorMetadata = L3KVConnectorMetadata
