# l3-vllm-connector

A vLLM `KVConnectorBase_V1` implementation that uses L3 KV cache as
an external prefix cache below vLLM's GPU L1 and CPU L2 KV cache tiers.

## Status

v0.1.0 — prefix-cache offload (L3) only. Disaggregated-prefill is architected
for but not yet implemented.

## Install

```bash
pip install -e .
# or
pip install l3-vllm-connector
```

Requires `l3-client` and a running L3 server.

## Use

Start the L3 KV cache server:

```bash
./l3-server --tcp 0.0.0.0:18001 --rdma none
```

Start vLLM with L3 as the KV connector:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --kv-transfer-config '{
      "kv_connector": "L3Connector",
      "kv_connector_module_path": "l3_vllm.connector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "remote_addr": "127.0.0.1",
        "remote_port": 18001
      }
    }'
```

Backward compatibility: `"kv_connector": "CamaConnector"` is also supported.

## Config keys (`kv_connector_extra_config`)

| Key | Default | Description |
|-----|---------|-------------|
| `remote_addr` | `localhost` | L3 server host |
| `remote_port` | `18001` | L3 server TCP port |
| `password` | `None` | Server password (if set) |
| `engine_namespace` | `vllm_l3` | Key prefix; isolates from SGLang-written entries |
| `async_load_min_tokens` | `32` | Bypass async path under this size |
| `save_chunk_size` | `256` | Max keys per `mset` call |
| `bloom_capacity` | `10_000_000` | Bloom filter capacity |
| `bloom_fpr` | `0.01` | Bloom filter false-positive rate |
| `lru_capacity` | `100_000` | Confirmed-present LRU size |
| `cb_failure_threshold` | `10` | Circuit breaker open threshold |
| `cb_overload_rate_per_sec` | `3.0` | Overload rate that trips CB |
| `cb_probe_interval_s` | `5.0` | OPEN -> HALF_OPEN probe interval |
| `is_mla` | `None` | Auto-detect MLA vs MHA when `None` |

## Architecture

Three layers:

- `L3ConnectorV1` (alias `CamaConnectorV1`) — top-level vLLM subclass; delegates by role.
- `L3ConnectorScheduler` (alias `CamaConnectorScheduler`) — bloom filter + LRU, `get_num_new_matched_tokens`,
  `build_connector_meta`. No remote RPCs on the critical path.
- `L3ConnectorWorker` (alias `CamaConnectorWorker`) — client connection pool, RDMA registration,
  per-forward batched operations.

## Key naming

Mirrors storage buffer naming with a `vllm_l3_` namespace prefix:

```
MHA: vllm_l3_{block_hash_hex}_{tp_rank}[_{pp_rank}]_k
     vllm_l3_{block_hash_hex}_{tp_rank}[_{pp_rank}]_v
MLA: vllm_l3_{block_hash_hex}[_{pp_rank}]
```

The `vllm_l3_` prefix prevents collisions with SGLang-written keys.

## Failure handling

No L3 exception propagates above the connector. Every public method is
wrapped in `try/except` and returns a miss-equivalent on failure. A
`CircuitBreaker` short-circuits operations when the server is unreachable
or overloaded, transitioning back to CLOSED after a successful probe.

This means **vLLM never crashes on a cache hiccup** — at worst, prefix-cache
hit rate drops to zero until the server recovers.
