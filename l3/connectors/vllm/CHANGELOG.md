# Changelog

## 0.1.0 — 2026-05-27

Initial release.

- `CamaConnectorV1` implements `KVConnectorBase_V1` (vLLM v1 API).
- Three-class delegation: `CamaConnectorScheduler` (bloom + LRU + meta) /
  `CamaConnectorWorker` (PriskvClient, RDMA registration, batched mget/mset).
- Engine-namespaced keys (`vllm_l3_` prefix) — isolated from SGLang-written entries.
- MHA + MLA support; MLA auto-detected from `model_config.use_mla` or DeepSeek-family arch.
- Async loading by default; inline blocking under `async_load_min_tokens=32`.
- Circuit breaker (CLOSED / OPEN / HALF_OPEN) with success/failure/overload tracking.
- Codec module vendored from cama-connector (zero changes).
- Profiling module de-SGLanged for vLLM.
- 32 unit tests + 5 integration tests against a real `cama-server` over TCP.
- Verified end-to-end on WSL Ubuntu-24.04 with vLLM 0.21.0: save/load round-trip,
  circuit-breaker trip on server kill, HALF_OPEN recovery on restart.

### Disagg-prefill forward compatibility

- Key namespace prefix isolates from future `vllm_prefill_` / `vllm_decode_`.
- `request_finished` returns `(False, None)` — v2 will fill `kv_transfer_params`.
- `_{tp_rank}_` flagged as future TP-remap anchor (NixlConnector style).

### Known limitations

- No GDR / direct GPU-HBM RDMA — RDMA targets the CPU staging buffer in vLLM's offload path.
- Background bloom sweeper not yet implemented; bloom is populated by save success only.
- Per-layer event signalling (`signal_per_layer`) is a config knob but currently all
  events are signalled together when `mget_rdma` returns.
