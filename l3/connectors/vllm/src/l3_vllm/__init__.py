"""l3-vllm-connector: vLLM KV connector backed by L3 KV cache.

Out-of-tree connector that lets vLLM use a running L3 server as an external
prefix cache (below GPU L1 and CPU L2).

Selection at vLLM startup:

    vllm serve <model> --kv-transfer-config '{
      "kv_connector": "L3Connector",
      "kv_connector_module_path": "l3_vllm.connector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {"remote_addr":"...","remote_port":18001}
    }'
"""

try:
    from l3_vllm.connector import (
        CamaConnector,
        CamaConnectorV1,
        L3Connector,
        L3ConnectorV1,
        register,
    )
except ImportError:
    # torch or vllm may not be installed in lightweight/test environments
    CamaConnector = None  # type: ignore
    CamaConnectorV1 = None  # type: ignore
    L3Connector = None  # type: ignore
    L3ConnectorV1 = None  # type: ignore
    def register():  # type: ignore
        raise RuntimeError("vllm and torch are required to register L3Connector")

__all__ = [
    "L3Connector",
    "L3ConnectorV1",
    "CamaConnector",
    "CamaConnectorV1",
    "register",
    "__version__",
]
__version__ = "0.1.0"
