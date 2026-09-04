"""CamaConnectorV1 — top-level vLLM KVConnectorBase_V1 subclass.

Constructed twice per process by vLLM:
- once with role=SCHEDULER -> delegates to L3ConnectorScheduler
- once with role=WORKER    -> delegates to L3ConnectorWorker

Selection in vLLM startup:

    --kv-transfer-config '{
      "kv_connector": "CamaConnector",
      "kv_connector_module_path": "l3_vllm.connector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {"remote_addr":"...", "remote_port":18001}
    }'

The ``register()`` function is the vLLM entry_points hook (older vLLM); newer
vLLM uses ``kv_connector_module_path`` directly. We support both.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from l3_vllm.config import L3ConnectorConfig
from l3_vllm.key_scheme import KeyScheme, KeySchemeConfig
from l3_vllm.metadata import L3KVConnectorMetadata
from l3_vllm.scheduler import L3ConnectorScheduler
from l3_vllm.worker import L3ConnectorWorker

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


def register() -> None:
    """Register CamaConnector with vLLM's KVConnectorFactory.

    Called via the ``vllm.kv_connector_v1`` entry_point on older vLLM.
    Newer vLLM (with kv_connector_module_path) discovers us directly.
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        for name in ("L3Connector", "CamaConnector"):
            try:
                KVConnectorFactory.register_connector(
                    name,
                    "l3_vllm.connector",
                    "L3ConnectorV1",
                )
            except ValueError:
                pass
        logger.info("CamaConnector registered with vLLM KVConnectorFactory")
    except ValueError:
        # Already registered — fine.
        pass
    except Exception as e:  # noqa: BLE001
        logger.warning("CamaConnector register failed: %s", e)


class L3ConnectorV1(KVConnectorBase_V1):
    """Top-level connector that delegates by role."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        # Older vLLM (0.21) had a 2-arg signature; newer requires 3.
        try:
            super().__init__(
                vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
            )
        except TypeError:
            super().__init__(vllm_config=vllm_config, role=role)  # type: ignore[call-arg]

        kv_transfer = vllm_config.kv_transfer_config
        extra = getattr(kv_transfer, "kv_connector_extra_config", None) or {}
        self.config = L3ConnectorConfig.from_extra_config(extra)

        # Detect MLA from model config if user didn't pin.
        is_mla = self.config.is_mla
        if is_mla is None:
            is_mla = self._detect_mla(vllm_config)
        self.config.is_mla = is_mla

        # Pull TP/PP rank for key suffix.
        tp_rank = self._safe_attr(vllm_config, "parallel_config.tensor_parallel_rank", 0)
        pp_rank = self._safe_attr(vllm_config, "parallel_config.pipeline_parallel_rank", 0)
        pp_size = self._safe_attr(vllm_config, "parallel_config.pipeline_parallel_size", 1)

        key_cfg = KeySchemeConfig(
            engine_namespace=self.config.engine_namespace,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            pp_size=pp_size,
            is_mla=is_mla,
        )
        self.key_scheme = KeyScheme(key_cfg)

        self._scheduler: L3ConnectorScheduler | None = None
        self._worker: L3ConnectorWorker | None = None

        if role is KVConnectorRole.SCHEDULER:
            self._scheduler = L3ConnectorScheduler(self.config, self.key_scheme)
            block_size = self._safe_attr(vllm_config, "cache_config.block_size", 16)
            self._scheduler.set_block_size(block_size)
            logger.info(
                "CamaConnector[SCHEDULER] init: ns=%s tp=%d pp=%d/%d mla=%s block_size=%d",
                key_cfg.engine_namespace, tp_rank, pp_rank, pp_size, is_mla, block_size,
            )
        else:
            self._worker = L3ConnectorWorker(self.config, self.key_scheme)
            logger.info(
                "CamaConnector[WORKER] init: ns=%s tp=%d pp=%d/%d mla=%s",
                key_cfg.engine_namespace, tp_rank, pp_rank, pp_size, is_mla,
            )

    # =========================================================
    # Worker-side methods (delegate to self._worker)
    # =========================================================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._worker is None:
            return
        try:
            self._worker.register_kv_caches(kv_caches)
        except Exception as e:  # noqa: BLE001
            logger.warning("CamaConnector.register_kv_caches failed: %s", e)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if self._worker is None:
            return
        try:
            self._worker.bind_metadata(self._get_connector_metadata())  # type: ignore[arg-type]
            self._worker.start_load_kv(forward_context, **kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning("CamaConnector.start_load_kv failed: %s", e)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._worker is None:
            return
        try:
            self._worker.wait_for_layer_load(layer_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("wait_for_layer_load failed: %s", e)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if self._worker is None:
            return
        try:
            self._worker.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning("save_kv_layer failed: %s", e)

    def wait_for_save(self) -> None:
        if self._worker is None:
            return
        try:
            self._worker.wait_for_save()
        except Exception as e:  # noqa: BLE001
            logger.warning("wait_for_save failed: %s", e)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self._worker is None:
            return None, None
        try:
            return self._worker.get_finished(finished_req_ids)
        except Exception as e:  # noqa: BLE001
            logger.warning("get_finished failed: %s", e)
            return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self._worker is None:
            return set()
        try:
            return self._worker.get_block_ids_with_load_errors()
        except Exception as e:  # noqa: BLE001
            logger.warning("get_block_ids_with_load_errors failed: %s", e)
            return set()

    def shutdown(self) -> None:
        if self._worker is not None:
            try:
                self._worker.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.warning("shutdown failed: %s", e)

    # =========================================================
    # Scheduler-side methods (delegate to self._scheduler)
    # =========================================================

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if self._scheduler is None:
            return 0, False
        try:
            return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)
        except Exception as e:  # noqa: BLE001
            logger.warning("get_num_new_matched_tokens failed: %s", e)
            return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if self._scheduler is None:
            return
        try:
            self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)
        except Exception as e:  # noqa: BLE001
            logger.warning("update_state_after_alloc failed: %s", e)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        if self._scheduler is None:
            return L3KVConnectorMetadata()
        try:
            return self._scheduler.build_connector_meta(scheduler_output)
        except Exception as e:  # noqa: BLE001
            logger.warning("build_connector_meta failed: %s", e)
            return L3KVConnectorMetadata()

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._scheduler is None:
            return False, None
        try:
            return self._scheduler.request_finished(request, block_ids)
        except Exception as e:  # noqa: BLE001
            logger.warning("request_finished failed: %s", e)
            return False, None

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _safe_attr(obj: Any, path: str, default: Any) -> Any:
        cur = obj
        for part in path.split("."):
            if cur is None:
                return default
            cur = getattr(cur, part, None)
        return cur if cur is not None else default

    @staticmethod
    def _detect_mla(vllm_config: "VllmConfig") -> bool:
        """Detect MLA model from VllmConfig.

        DeepSeek-V2/V3 and MiniMax-class models expose latent attention.
        Different vLLM versions expose this differently; we check a few.
        """
        try:
            model_config = getattr(vllm_config, "model_config", None)
            if model_config is None:
                return False
            # Newer vLLM: ModelConfig.use_mla
            if getattr(model_config, "use_mla", False):
                return True
            hf_config = getattr(model_config, "hf_config", None)
            if hf_config is None:
                return False
            # DeepSeek family exposes kv_lora_rank / q_lora_rank.
            if getattr(hf_config, "kv_lora_rank", None):
                return True
            arch_list = getattr(hf_config, "architectures", []) or []
            mla_archs = {"DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "MiniMaxText01ForCausalLM"}
            return any(a in mla_archs for a in arch_list)
        except Exception:  # noqa: BLE001
            return False


# Alias so that vLLM's `kv_connector_module_path` discovery works with
# `kv_connector="CamaConnector"`. vLLM does `getattr(module, kv_connector)`,
# which means the class name in the module must equal the connector name.
CamaConnector = CamaConnectorV1


# Backward compatibility aliases
CamaConnectorV1 = L3ConnectorV1
L3Connector = L3ConnectorV1
CamaConnector = L3ConnectorV1
