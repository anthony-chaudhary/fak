"""Key naming for CAMA-stored vLLM KV blocks.

Mirrors the suffix scheme from `cama_module/cama_storage.py` (lines 634-640,
1179-1188 in the SGLang connector). vLLM provides its own sha256 Merkle block
hash — we do NOT re-hash; we just namespace and decorate.

Forward-compatibility hook: the `_{tp_rank}_` slice is the future remap
anchor for disagg-prefill (NixlConnector-style tp_mapping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class KeySchemeConfig:
    """Configuration controlling how block hashes are turned into CAMA keys."""

    engine_namespace: str = "vllm_l3"
    tp_rank: int = 0
    pp_rank: int = 0
    pp_size: int = 1
    is_mla: bool = False

    @property
    def enable_pp(self) -> bool:
        return self.pp_size > 1

    @property
    def mha_suffix(self) -> str:
        # Matches cama_storage.py:634-640.
        if self.enable_pp:
            return f"{self.tp_rank}_{self.pp_rank}"
        return f"{self.tp_rank}"

    @property
    def mla_suffix(self) -> str:
        # Matches cama_storage.py:634-640.
        if self.enable_pp:
            return f"{self.pp_rank}"
        return ""


class KeyScheme:
    """Builds CAMA keys from vLLM block hashes."""

    def __init__(self, config: KeySchemeConfig):
        self.config = config

    def block_hash_to_hex(self, block_hash) -> str:
        """Normalize whatever vLLM hands us to a hex string."""
        if isinstance(block_hash, bytes):
            return block_hash.hex()
        if isinstance(block_hash, int):
            return format(block_hash, "x")
        if isinstance(block_hash, str):
            return block_hash
        if hasattr(block_hash, "hash_value"):
            return self.block_hash_to_hex(block_hash.hash_value)
        if hasattr(block_hash, "hex"):
            try:
                return block_hash.hex()
            except TypeError:
                pass
        return str(block_hash)

    def keys_for_block(self, block_hash) -> list[str]:
        """One or two CAMA keys per block, depending on MHA/MLA.

        MHA -> [f"{ns}_{hash}_{suffix}_k", f"{ns}_{hash}_{suffix}_v"]
        MLA -> [f"{ns}_{hash}_{suffix}"]  (or _suffix omitted when pp_size==1)
        """
        cfg = self.config
        h = self.block_hash_to_hex(block_hash)
        base = f"{cfg.engine_namespace}_{h}"
        if cfg.is_mla:
            if cfg.mla_suffix:
                return [f"{base}_{cfg.mla_suffix}"]
            return [base]
        # MHA: always include tp_rank (and pp_rank when PP>1).
        return [
            f"{base}_{cfg.mha_suffix}_k",
            f"{base}_{cfg.mha_suffix}_v",
        ]

    def keys_for_blocks(self, block_hashes: Iterable) -> list[str]:
        out: list[str] = []
        for bh in block_hashes:
            out.extend(self.keys_for_block(bh))
        return out

    @property
    def keys_per_block(self) -> int:
        return 1 if self.config.is_mla else 2
