---
title: "fak Server Configuration: fak serve Flags & Env Vars"
description: "Configuration reference for fak serve: command-line flags, environment variables, policy manifest schema, HTTP endpoints, and metrics for the gateway server."
---

# fak Server Configuration Reference

This document catalogs all configuration options for `fak serve`, the gateway server that fronts the fak kernel over OpenAI-compatible HTTP and MCP interfaces.

**Note:** `fak serve` is the gateway server. Other `fak` commands (`run`, `agent`, `bench`, `debug`, etc.) have their own flags documented in `fak --help`.

---

## Server Command-Line Flags

### Basic Server Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--addr` | string | `127.0.0.1:8080` | HTTP listen address for OpenAI-compatible and fak-native endpoints. Ignored with `--stdio`. Required unless `--stdio` is set. |
| `--stdio` | bool | `false` | Serve MCP over stdin/stdout (newline-delimited JSON-RPC) instead of HTTP. |

### Upstream Model Configuration (Proxy Mode)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--provider` | string | `openai` | Upstream provider transcript wire. Options: `openai`, `anthropic`, `gemini`, `xai`. |
| `--base-url` | string | `""` | Upstream provider base URL for `/v1/chat/completions` proxy (e.g., `https://api.openai.com/v1`). Empty enables the offline mock planner — unless `--gguf` is set, in which case the in-kernel model serves chat instead. |
| `--model` | string | `mock` | Model ID advertised by `/v1/models` and used for upstream calls. |
| `--api-key-env` | string | `""` | Environment variable name holding the upstream API key for proxy mode. |

### In-Kernel Model Configuration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gguf` | string | `""` | Path to GGUF weights to load into the in-kernel engine at boot. The load is measured as part of startup and exposed on `/metrics`. With no `--base-url`, both `/v1/chat/completions` and `/v1/messages` serve the in-kernel model directly, using the GGUF's embedded tokenizer (or `--tokenizer` if given). |
| `--tokenizer` | string | `""` | Path to `tokenizer.json` (or its directory) for the in-kernel CHAT planner. **Optional:** with `--gguf` and no `--base-url`, fak uses the GGUF's embedded tokenizer, so chat works without it. Pass `--tokenizer` only to override, or for a rare SPM-only GGUF that embeds no usable BPE tokenizer. |

### Kernel Engine Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--engine` | string | `inkernel` | Registered engine ID for `fak_syscall` dispatch. Default: `inkernel` (the fused in-kernel model). |

### Policy and Capability Floor

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--policy` | string | `""` | Path to capability-floor manifest file. If empty, uses the built-in `DefaultPolicy`. |
| `--policy-check` | bool | `false` | Validate `--policy` manifest and exit without binding a listener. Requires `--policy` to be set. |

### vDSO and Caching Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--vdso` | bool | `true` | Enable the vDSO dedup fast path for cross-agent read deduplication. |
| `--invalidation` | string | `global` | vDSO tier-2 invalidation granularity for the live fleet. Options: `global`, `namespace`, `resource`. |

### External Engine Cache Reset

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--engine-cache-engine` | string | `""` | Self-hosted serving engine for cache reset after quarantined proxy tool results. Options: `sglang`, `vllm`. Empty disables remote cache reset. |
| `--engine-cache-base-url` | string | `""` | Serving-engine control/base URL for cache reset. Defaults to `--base-url` when `--engine-cache-engine` is set. |
| `--engine-cache-admin-key-env` | string | `""` | Environment variable holding the serving-engine admin API key for cache reset. |
| `--engine-cache-idle-timeout` | duration | `0` | SGLang `/flush_cache` idle timeout (e.g., `30s`). Zero fails fast. |
| `--engine-cache-require-exact-span` | bool | `false` | Require exact remote K/V/index span eviction. Fail closed if the selected engine only supports whole-cache reset. |

### Authentication

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--require-key-env` | string | `""` | Environment variable holding a bearer token REQUIRED on every request (except `/healthz`). Empty means no auth. |

---

## Environment Variables

### Model and Compute

| Variable | Description |
|----------|-------------|
| `FAK_MODEL_DIR` | Path to a model export directory. When set, the in-kernel engine loads from this export instead of using the synthetic checkpoint. |
| `FAK_Q4K` | When set to `1`, uses the direct-resident-Q4_K path for Qwen3.6-27B Q4_K_M models (SDOT int8 decode GEMV path, ~10× faster load). |
| `FAK_BACKEND` | Compute backend selection. Options: `cuda`, `metal`, `vulkan`, `cpu`. Default is auto-detected. |
| `FAK_CUDA_GRAPH` | When set to `1`, enables CUDA-graph decode path on CUDA backend. |
| `FAK_CUDA_F16` | When set to `1`, enables f16 computation paths in CUDA benchmarks. |
| `FAK_CUDA_Q8` | When set to `1`, enables Q8 computation paths in CUDA benchmarks. |
| `FAK_WORKERS` | Caps matmul parallelism. Default is `GOMAXPROCS`. Set to a fixed number (e.g., `8`) to pin worker count. |

### Kernel Behavior

| Variable | Description |
|----------|-------------|
| `FAK_IFC` | IFC (Information Flow Control) toggle. Set to `off` to make both IFC gates no-ops (A/B testing ablation). Default: enabled. |
| `FAK_IFC_GATE_EXEC` | Restrictive opt-in for the untrusted-input threat model: set to `1` to also gate the EXEC (shell/Bash) sink when the session is tainted. Default OFF — the reasonable default gates only the EGRESS and DESTRUCTIVE sinks, because gating shell on the session taint high-water mark blocks normal Bash after any untrusted read (and the hard arg-rules block dangerous shell unconditionally regardless of taint). Enable it for an agent processing untrusted input. |
| `FAK_NORMGATE` | Normalization gate toggle. Set to `off` to make the admit gate a no-op. Default: enabled. |
| `FAK_VDSO_GRANULARITY` | vDSO tier-2 invalidation granularity (overrides `--invalidation`). Options: `global`, `namespace`, `resource`. |

### HTTP Server

| Variable | Description |
|----------|-------------|
| `FAK_HTTP_READ_TIMEOUT_S` | HTTP read timeout in seconds. Default: `30`. Set to `0` to disable. |
| `FAK_HTTP_WRITE_TIMEOUT_S` | HTTP write timeout in seconds. Default: `90`. Increase for slow local models. Set to `0` to disable. |
| `FAK_HTTP_IDLE_TIMEOUT_S` | HTTP idle timeout in seconds. Default: `120`. Set to `0` to disable. |

### Agent/Planner

| Variable | Description |
|----------|-------------|
| `FAK_PLANNER_TIMEOUT_S` | Per-request HTTP timeout for upstream model calls. Default: `60`. |
| `FAK_PROVIDER_EXTRA_BODY_JSON` | Optional extra JSON body to send with upstream provider requests (advanced). |
| `FAK_INKERNEL_MAX_TOKENS` | Max tokens to generate for in-kernel planner. Default: `256`. |
| `FAK_INKERNEL_TEMP` | Sampling temperature for in-kernel planner. Default: `0`. |
| `FAK_INKERNEL_SEED` | Sampling seed for in-kernel planner. Default: `0` (random). |

### Performance Tuning (Internal)

| Variable | Description |
|----------|-------------|
| `FAK_QKERNEL` | Quantization kernel selection on amd64/arm64. Options: `auto`, `scalar`, `neon`, `avx2`, `avx512`. |
| `FAK_QGEMM` | Quantized GEMM mode selection. Options: `tile`, `legacy`. |
| `FAK_QATTN_GQA` | GQA (grouped-query attention) mode selection. Options: `auto`, `scalar`, `simd`. |
| `FAK_QPROFILE` | When set to `1`, enables coarse phase timing in quantized prefill (diagnostic). |
| `FAK_HAL_Q8_BATCH_LAYERS` | Number of layers to batch in Q8 matmul (hardware-specific tuning). |

### GPU Lease

| Variable | Description |
|----------|-------------|
| `FAK_GPU_LEASE` | Path to the GPU lease lockfile. Default: system temp directory. |
| `FAK_GPU_LEASE_NOWAIT` | When set to `1`, GPU lease acquisition fails fast instead of waiting. |

### Rate Limiting

| Variable | Description |
|----------|-------------|
| `FAK_RATELIMIT_MAX_CALLS` | Per-key admitted-call quota. Default: disabled. |
| `FAK_RATELIMIT_MAX_COST` | Per-key cumulative cost budget (arg bytes ~ tokens). Default: disabled. |
| `FAK_RATELIMIT_KEY` | Dimension to bucket by: `trace`, `tool`, or `global`. Default: `trace`. |

### Diagnostics and Testing

| Variable | Description |
|----------|-------------|
| `FAK_AUDIT_JOURNAL` | Path to audit journal JSONL file for durable syscall audit. Default: disabled. |
| `FAK_APP_VERSION` | Override version string (for testing). |
| `FAK_TOKENIZER_DIR` | Directory for tokenizer test fixtures. Default: `~/.cache/fak-models/tokenizers/qwen2.5`. |
| `FAK_ORACLE_DIRS` | Colon-separated list of oracle directories for model tests. |
| `FAK_ORACLE_REQUIRED_FAMILIES` | Comma-separated list of required model families for tests. |

### Benchmark Control

| Variable | Description |
|----------|-------------|
| `FAK_BENCH_PREFIX` | Default prefix length for quantized GEMM benchmarks. Default: `256`. |
| `FAK_BENCH_BATCHES` | Comma-separated batch sizes for quantized GEMM benchmarks. Default: `8,16,32,128`. |
| `FAK_BENCH_RECT_B` | Batch dimension for rectangular benchmarks. Default: `8`. |
| `FAK_BENCH_RECT_P` | Prefix dimension for rectangular benchmarks. Default: `48`. |

---

## Policy Manifest Schema

The `--policy` flag loads a capability-floor manifest from a JSON file. Run `fak policy --dump` to emit the built-in `DefaultPolicy` as a starting template.

### Schema Version

```json
{
  "version": "fak-policy/v1"
}
```

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Manifest schema version. Optional; defaults to current binary version. |
| `posture` | string | Default-deny behavior. Options: `fail_closed` (default), `admit_and_log`. |
| `allow` | array[string] | Exact tool names to permit. Everything else is DEFAULT_DENY. |
| `allow_prefix` | array[string] | Prefix patterns to permit. A tool matching any prefix is allowed. |
| `deny` | object | Explicit denials: `{"tool_name": "REASON_CODE"}. Reason must be from the closed vocabulary. |
| `self_modify_globs` | array[string] | Path fragments that prove a SELF_MODIFY attempt in write-shaped calls. |
| `redact_fields` | array[string] | Argument keys whose values are stripped (TRANSFORM) before dispatch. |
| `arg_rules` | array[object] | Per-tool argument-value constraints (see below). |
| `safe_sinks` | array[string] | Tools exempt from IFC egress checks. |
| `authorize` | array[object] | IFC authorization rules releasing tainted flows into exact sinks. |
| `sources` | object | Tool-to-source-class mapping for provenance tracking. |

### ArgRule Schema

Each `arg_rules` entry constrains one tool's argument values:

```json
{
  "tool": "tool_name",
  "arg": "arg_name",
  "allow_glob": "./out/**",     // Path must be under this glob (exactly one matcher required)
  "deny_regex": "secret.*",     // OR: regex pattern that DENIES matching values
  "max_bytes": 1048576,         // OR: max bytes for string args
  "reason": "POLICY_BLOCK"      // Optional: closed-vocabulary reason code (default: POLICY_BLOCK)
}
```

### AuthorizeRule Schema

Each `authorize` entry releases tainted data into one sink:

```json
{
  "tool": "tool_name",
  "sink": "EGRESS"              // Sink class: EGRESS, EXEC, or DESTRUCTIVE
}
```

### Sources Schema

Maps tools to provenance source classes:

```json
{
  "sources": {
    "read_file": "trusted_local",
    "http_get": "untrusted"
  }
}
```

Valid source classes: `trusted_local`, `untrusted`.

### Example Manifest

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow_prefix": ["read_", "get_", "list_", "search_", "calc_"],
  "deny": {
    "bash": "POLICY_BLOCK",
    "write_file": "POLICY_BLOCK"
  },
  "self_modify_globs": ["*.go", "*.md", "CLAUDE.md"],
  "redact_fields": ["api_key", "token", "password"],
  "arg_rules": [
    {
      "tool": "write_file",
      "arg": "path",
      "allow_glob": "./out/**"
    }
  ],
  "authorize": [
    {
      "tool": "format_output",
      "sink": "EGRESS"
    }
  ],
  "sources": {
    "read_file": "trusted_local",
    "http_get": "untrusted"
  }
}
```

---

## HTTP Endpoints

When `--addr` is set (without `--stdio`), the gateway exposes these HTTP routes:

| Route | Method | Description |
|-------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions proxy with tool-call adjudication. |
| `/v1/messages` | POST | Anthropic-compatible messages endpoint. |
| `/v1/messages/count_tokens` | POST | Anthropic token counting endpoint. |
| `/v1/fak/syscall` | POST | Fak native: run a tool call through full syscall (adjudicate + dispatch + admit). |
| `/v1/fak/adjudicate` | POST | Fak native: run pre-execution adjudication only (no dispatch). |
| `/v1/fak/admit` | POST | Fak native: admit a client-produced tool result through quarantine. |
| `/v1/fak/changes` | GET/POST | Cross-agent "what changed" feed (vdso coherence bus). |
| `/v1/fak/revoke` | POST | Revoke a witness, evicting pooled entries admitted under it. |
| `/v1/fak/context/change` | POST | Request a context-control mutation (e.g., tombstone) on a persisted core image. |
| `/v1/fak/policy/reload` | POST | Reload the capability-floor manifest from disk (requires `--policy`). |
| `/v1/fak/trace/reset` | POST | Reset one trace's process-local taint ledger mark. |
| `/v1/models` | GET | List available models (mirrors `--model`). |
| `/mcp` | POST | MCP-over-HTTP endpoint. |
| `/healthz` | GET | Unauthenticated health check. |
| `/metrics` | GET | Prometheus metrics. |
| `/debug/vars` | GET | Expvar-style diagnostics. |

---

## Metrics

The `/metrics` endpoint exposes Prometheus metrics including:

| Metric | Type | Description |
|--------|------|-------------|
| `fak_gateway_time_to_ready_seconds` | gauge | Time from process start to ready (listener bound, model loaded). |
| `fak_gateway_startup_phase_duration_seconds` | gauge | Per-phase startup duration (flag-parse, policy-load, model-load, etc.). |
| `fak_model_load_*` | gauge | Model load profile breakdown (source, mode, total_seconds, tensors, bottleneck). |
| `fak_gateway_operations_total` | counter | Kernel operation counts by operation, verdict (ALLOW, DENY, TRANSFORM, QUARANTINE, WITNESS), reason, disposition, and deciding adjudicator. |
| `fak_gateway_operation_duration_seconds` | histogram | Operation latency by operation type (adjudicate, syscall, admit), verdict, and deciding adjudicator. |

---

## MCP Interface (stdio)

When `--stdio` is set, the gateway serves MCP over stdin/stdout with newline-delimited JSON-RPC. Exposed tools:

- `fak_adjudicate` — Pre-execution adjudication
- `fak_syscall` — Full syscall with dispatch
- `fak_admit` — Admit tool results
- `fak_changes` — Cross-agent coherence feed
- `fak_revoke` — Witness revocation
- `fak_context_change` — Context mutations

---

## Further Documentation

- `fak --help` — Full command-line reference for all `fak` verbs
- `fak policy --dump` — Emit the built-in DefaultPolicy as a starting template
- `fak policy --check FILE` — Validate a policy manifest and see its floor summary
