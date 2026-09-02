---
title: "fak learning path — L500"
description: "A staged part of the fak learning path, split out of LEARNING-PATH.md so each stage stays a bounded read."
---

# L500 — Serving, Integration, and the In-Kernel Model

**Stage 5 of the path** · prev: [L400 — The Performance Core](performance-core.md) · next: [L600 — Mastery](mastery.md) · back to the [overview and L100–L200](../../LEARNING-PATH.md)

**Read:** [`docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md`](../notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md)

**Lab:**
```bash
Reproduce the doc arithmetic for a Qwen2.5-7B geometry: compute KV bytes/token (2 x 28 x 4 x 128 x 2), a 100k-token cache size, and its ratio to A100 L2 (40MB) and one SM's SRAM (192KB).
```

**Checkpoint:** State which saturation point binds first at agent-city scale and why it is residency rather than compute; then name two meters that would prove a system actually scales.

---

## L500 — Serving, Integration, and the In-Kernel Model

**Theme.** Running and hardening the gateway, the gateway drop guarantee, repointing existing agents at one base URL, the framework cookbook, the pure-Go in-kernel model + compute HAL with oracle parity, and the GPU lease.

**Who joins here.** A platform/SRE who already runs vLLM, or an app developer who just calls an LLM API and wants governance with zero agent rewrite. Join here if you can take the security and performance cores as given and want to deploy, integrate, or understand the reference forward pass.

**Assumes you can already pass:** **FAK 105**, **FAK 301**, **FAK 304**, **FAK 310**.

| Course | Hard prerequisites |
|---|---|
| **FAK 501** — The fak serve Mental Model: One Binary, Four Tiers, Three Modes | **FAK 105**, **FAK 301** |
| **FAK 502** — Starting the Gateway: serve Flags and the Engine-vs-Upstream Axis | **FAK 501** |
| **FAK 503** — The HTTP API: OpenAI, Anthropic, fak-native, and MCP Surfaces | **FAK 502**, **FAK 310** |
| **FAK 504** — Hardening the Gateway: Bearer Auth, the Policy Floor, and Live Reload | **FAK 503**, **FAK 304** |
| **FAK 505** — Observability: Prometheus Metrics, JSON Access Log, X-Trace-Id | **FAK 503** |
| **FAK 506** — Tuning Timeouts and the serve Env Vars | **FAK 502** |
| **FAK 507** — Deploying the Gateway: Docker, Compose, Kubernetes, Bare Metal | **FAK 504**, **FAK 505** |
| **FAK 508** — Scaling and HA: Process-Local State and Sticky Routing | **FAK 507**, **FAK 407**, **FAK 314** |
| **FAK 509** — The MCP Tool-Result Wire: Refusal as a Value | **FAK 503**, **FAK 312** |
| **FAK 510** — Troubleshooting the Gateway and the fak CLI Verbs | **FAK 504** |
| **FAK 511** — The Integration Index: Repoint One Base URL | **FAK 503** |
| **FAK 512** — Claude Code / Anthropic API Through fak | **FAK 511** |
| **FAK 513** — OpenAI Codex / OpenAI SDK Through fak | **FAK 511** |
| **FAK 514** — Cursor via MCP or OpenAI Proxy | **FAK 511** |
| **FAK 515** — MCP One-Paste Setup and the fak_* Tools | **FAK 511**, **FAK 509** |
| **FAK 516** — Agent<->Kernel Architecture and the Frozen ABI Verdict Union | **FAK 511**, **FAK 208** |
| **FAK 517** — Framework Cookbook: Transparent Proxy (Mode A) vs Explicit Adjudication (Mode B) | **FAK 516**, **FAK 513**, **FAK 302** |
| **FAK 518** — Migration: Moving Existing Code by Repointing a Base URL | **FAK 516** |
| **FAK 519** — Multi-Language Client Code and Disposition-Aware Retry | **FAK 516**, **FAK 509** |
| **FAK 520** — The Adopter Playbook: Front-a-Model, Manual MCP, Embed-in-CI | **FAK 512**, **FAK 515** |
| **FAK 521** — GGUF Loading: Offsets, Dtypes, and Dequant Layout | **FAK 205** |
| **FAK 522** — Tokenizer: Lossless ByteLevel BPE With Oracle Parity | **FAK 521** |
| **FAK 523** — Normalization: RMSNorm, NormGain1p, and LayerNorm | **FAK 522** |
| **FAK 524** — RoPE: Rotary Position Embedding and Scaling Variants | **FAK 523** |
| **FAK 525** — Attention: Stable Softmax, Causal Mask, and the Attention Sink | **FAK 524** |
| **FAK 526** — MLP / SwiGLU+GeGLU, MoE Routing, and the Residual Stream | **FAK 525** |
| **FAK 527** — In-Kernel KV Cache: Slotting, Span-Exact Eviction, SWA, Prefix Reuse | **FAK 526**, **FAK 406** |
| **FAK 528** — Quantization: Q4_K/Q8_0/Q4_0 Dequant, AWQ, and Bit-Identical int8 SDOT | **FAK 521**, **FAK 526** |
| **FAK 529** — Forward-Pass Parity vs the HuggingFace Oracle | **FAK 527**, **FAK 528**, **FAK 210** |
| **FAK 530** — The Compute HAL Seam and Hardware Portability | **FAK 529**, **FAK 210** |
| **FAK 531** — Metal GPU GEMM Parity and the Stub-vs-Device Build | **FAK 530** |
| **FAK 532** — The Engine Seam: Determinism and Cache-Invalidation Binding | **FAK 529**, **FAK 206** |
| **FAK 533** — In-Kernel Model & Compute Env Knobs (FAK_* Engine Vars) | **FAK 502**, **FAK 528** |
| **FAK 534** — GPU Lease: Machine-Wide Mutual Exclusion for Model Residency | **FAK 533** |
| **FAK 535** — The Gateway Drop Guarantee: Fail-Closed on a Failed Adjudication | **FAK 510**, **FAK 314** |

### FAK 501 — The fak serve Mental Model: One Binary, Four Tiers, Three Modes

**Prerequisites:** **FAK 105**, **FAK 301**
  ·  **Background:** **FAK 302**, **FAK 403**

**You'll be able to:**
- Frame the deploy-stack-ownership claim: fak collapses the governance half of agent serving (API surface + capability gate + result containment + audit + auth) into ONE static binary that fronts, not replaces, a token engine — identical laptop to fleet
- Distinguish proxy mode (--base-url), in-kernel mode (--gguf, no --base-url), and offline mock
- Name the four escalating setup tiers (0 offline kernel, 1 front a model, 2 in-kernel synthetic, 2b real weights)
- Explain why Tier 2's in-kernel SmolLM2 is a reference forward pass and NOT a production chat server

**Read:** [`docs/explainers/one-binary-one-surface.md`](../explainers/one-binary-one-surface.md), [`GETTING-STARTED.md`](../../GETTING-STARTED.md), [`docs/fak/server-quickstart.md`](../fak/server-quickstart.md)

**Lab:**
```bash
go run ./cmd/fak run --trace testdata/tau2/tau2-smoke.json   # Tier 0: replay a trace through the kernel offline
```

**Checkpoint:** Draw the two-halves split (governance+gateway vs token engine) and explain why 'the laptop story and the fleet story are the same binary' — what changes is flags, not installed components. Then explain proxy vs in-kernel vs offline mock, and why Tier 2's in-kernel SmolLM2 is a reference forward pass and NOT a production chat server.

### FAK 502 — Starting the Gateway: serve Flags and the Engine-vs-Upstream Axis

**Prerequisites:** **FAK 501**

**You'll be able to:**
- Use the core serve flags (--addr, --provider, --base-url, --model, --gguf, --tokenizer, --engine, --stdio)
- Explain why --engine (serving /v1/fak/*) is a separate axis from --base-url (the upstream model)
- Predict what /healthz reports for the engine field in a Tier-1 proxy deployment

**Read:** [`docs/fak/server-config.md`](../fak/server-config.md), [`docs/fak/server-quickstart.md`](../fak/server-quickstart.md), [`GETTING-STARTED.md`](../../GETTING-STARTED.md)

**Lab:**
```bash
ollama serve & ; ollama pull qwen2.5:1.5b ; go run ./cmd/fak serve --addr 127.0.0.1:8080 --base-url http://localhost:11434/v1 --model qwen2.5:1.5b ; curl -s http://127.0.0.1:8080/healthz
```

**Checkpoint:** Given a Tier-1 deployment, predict what curl /healthz returns for the engine field, and explain why your upstream model is reached only via /v1/chat/completions and not via /v1/fak/syscall.

### FAK 503 — The HTTP API: OpenAI, Anthropic, fak-native, and MCP Surfaces

**Prerequisites:** **FAK 502**, **FAK 310**

**You'll be able to:**
- Identify which endpoint to call across the four wire surfaces on one port
- Explain why a policy refusal returns HTTP 200 carrying a verdict (deny-as-value, not an error) and that SSE is synthesized from the finished turn
- Distinguish /v1/fak/adjudicate from /v1/fak/syscall and /v1/fak/admit

**Read:** [`docs/fak/api-reference.md`](../fak/api-reference.md), [`GETTING-STARTED.md`](../../GETTING-STARTED.md), [`docs/fak/server-config.md`](../fak/server-config.md)

**Lab:**
```bash
curl -s -X POST http://127.0.0.1:8080/v1/fak/adjudicate -H 'Content-Type: application/json' -d '{"tool":"refund_payment","arguments":{}}'   # observe verdict DENY in a 200 response
```

**Checkpoint:** Explain why a policy refusal returns HTTP 200 (not 4xx), what the fak response extension contains for a turn with a dropped tool call, and how /v1/fak/adjudicate differs from /v1/fak/syscall and /v1/fak/admit.

### FAK 504 — Hardening the Gateway: Bearer Auth, the Policy Floor, and Live Reload

**Prerequisites:** **FAK 503**, **FAK 304**

**You'll be able to:**
- Add dual-header bearer auth with --require-key-env and pin a fail-closed --policy floor
- Reload the policy live with POST /v1/fak/policy/reload without restarting or dropping warm vDSO/IFC state
- Explain why a non-loopback bind without a key still serves (with a warning) and why that is a hazard

**Read:** [`docs/serve-config.md`](../serve-config.md), [`docs/fak/server-config.md`](../fak/server-config.md), [`docs/fak/server-quickstart.md`](../fak/server-quickstart.md)

**Lab:**
```bash
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)" ; fak policy --dump > policy.json ; fak policy --check policy.json ; fak serve --addr 0.0.0.0:8080 --base-url http://localhost:11434/v1 --model M --policy policy.json --require-key-env FAK_GATEWAY_KEY
```

**Checkpoint:** Set up auth + a custom policy, prove every route except /healthz now requires the token, then edit policy.json and reload it live with a single authenticated POST without restarting the process.

### FAK 505 — Observability: Prometheus Metrics, JSON Access Log, X-Trace-Id

**Prerequisites:** **FAK 503**

**You'll be able to:**
- Alert on fak_gateway_up, build_info, per-route latency/error rate, verdict counts, and startup-phase timings
- Correlate one request across logs/metrics/headers via X-Trace-Id
- Name which fields the access log deliberately never carries and why that lets you ship it to a SIEM

**Read:** [`docs/fak/observability.md`](../fak/observability.md), [`docs/fak/server-config.md`](../fak/server-config.md)

**Lab:**
```bash
curl -s http://127.0.0.1:8137/metrics | grep fak_gateway ; curl -si -H 'X-Trace-Id: my-req-42' http://127.0.0.1:8137/healthz | grep -i x-trace-id
```

**Checkpoint:** Write the PromQL for per-route p99 latency and per-route 5xx error rate, and explain which fields the access log deliberately never carries and why that lets you ship it to a SIEM safely.

### FAK 506 — Tuning Timeouts and the serve Env Vars

**Prerequisites:** **FAK 502**

**You'll be able to:**
- Size FAK_HTTP_*_TIMEOUT_S and FAK_PLANNER_TIMEOUT_S for a slow local CPU model vs a fast hosted upstream
- Explain why FAK_HTTP_WRITE_TIMEOUT_S must be >= FAK_PLANNER_TIMEOUT_S
- Explain what setting the write timeout to 0 does and why it is a slow-loris risk, plus the [5,3600] planner clamp

**Read:** [`docs/serve-config.md`](../serve-config.md), [`docs/fak/server-config.md`](../fak/server-config.md), [`docs/fak/advanced-topics.md`](../fak/advanced-topics.md)

**Lab:**
```bash
FAK_PLANNER_TIMEOUT_S=600 FAK_HTTP_WRITE_TIMEOUT_S=600 fak serve --addr 127.0.0.1:8080 --gguf model.gguf --policy policy.json
```

**Checkpoint:** Explain why FAK_HTTP_WRITE_TIMEOUT_S must be at least FAK_PLANNER_TIMEOUT_S, what setting the write timeout to 0 does and why it is a slow-loris risk on a network bind, and the [5,3600] clamp on the planner timeout.

### FAK 507 — Deploying the Gateway: Docker, Compose, Kubernetes, Bare Metal

**Prerequisites:** **FAK 504**, **FAK 505**

**You'll be able to:**
- Deploy the single static binary across four targets using the distroless nonroot image
- Walk the production-readiness checklist (auth on, policy pinned, intentional bind, sized timeouts, audit journal, non-root)
- Explain why /healthz is a valid readiness probe (no /readyz; GGUF loads before bind) and why readOnlyRootFilesystem is safe

**Read:** [`docs/fak/deployment-guide.md`](../fak/deployment-guide.md), [`docs/fak/server-quickstart.md`](../fak/server-quickstart.md)

**Lab:**
```bash
docker build -t fak:0.34.0 . ; docker run --rm -p 8080:8080 -e FAK_GATEWAY_KEY="$(openssl rand -hex 32)" fak:0.34.0 serve --addr 0.0.0.0:8080 --base-url http://host.docker.internal:11434/v1 --model qwen2.5:1.5b
```

**Checkpoint:** Walk the production-readiness checklist and justify each item; explain why /healthz is a valid readiness probe and why readOnlyRootFilesystem is safe for fak.

### FAK 508 — Scaling and HA: Process-Local State and Sticky Routing

**Prerequisites:** **FAK 507**, **FAK 407**, **FAK 314**

**You'll be able to:**
- Explain why the verdict path is stateless and replicates freely but the vDSO cache and per-trace IFC ledger are process-local
- Configure sticky-by-trace_id routing for IFC correctness
- Explain why scaling out dilutes the cross-agent vDSO hit rate and why rate-limit counters are per-process

**Read:** [`docs/fak/advanced-topics.md`](../fak/advanced-topics.md), [`docs/fak/observability.md`](../fak/observability.md)

**Lab:**
```bash
Configure an nginx upstream with `hash $http_x_trace_id consistent;` over three fak gateways and verify that all calls of one trace land on one replica.
```

**Checkpoint:** Explain why a multi-call IFC flow needs sticky routing by trace_id, why scaling out reduces the vDSO cross-agent hit rate, and why FAK_RATELIMIT_MAX_CALLS gives 'N per replica the trace touches' rather than a true fleet cap under round-robin.

### FAK 509 — The MCP Tool-Result Wire: Refusal as a Value

**Prerequisites:** **FAK 503**, **FAK 312**

**You'll be able to:**
- Explain why isError is always false even on a DENY (deny as successful adjudication)
- Given verdict.reason='SELF_MODIFY', derive the disposition class (RETRYABLE/WAIT/ESCALATE/TERMINAL)
- Name on which verdict kind repaired_arguments appears

**Read:** [`docs/mcp-tool-result.md`](../mcp-tool-result.md)

**Lab:**
```bash
Hand-write the SyscallResponse JSON a client would receive (a) when ctxmmu quarantines a secret-shaped result and (b) when canon repairs a path; verify each field against the tables in docs/mcp-tool-result.md.
```

**Checkpoint:** Why is isError false even on a DENY? Given verdict.reason='SELF_MODIFY', what disposition does kernel.Disposition derive, and on which verdict kind does repaired_arguments appear?

### FAK 510 — Troubleshooting the Gateway and the fak CLI Verbs

**Prerequisites:** **FAK 504**

**You'll be able to:**
- Diagnose port conflicts, OOM/model-load failures, GPU/CUDA/Vulkan errors, tokenizer fallbacks, and policy errors
- Use the debugging tools (/healthz, /metrics load phases, FAK_LOG=debug, --policy-check)
- Situate serve among the run/preflight/bench/policy/agent/recall/debug verbs that author and exercise the same capability floor

**Read:** [`docs/fak/server-troubleshooting.md`](../fak/server-troubleshooting.md), [`docs/cli-reference.md`](../cli-reference.md)

**Operator verb map:** when the troubleshooting path points at the wider CLI, use
`fak usage` for gateway/provider usage ledgers, `fak issue` for issue queue inspection,
`fak workflow-audit` for workflow-policy drift, `fak fleetcap` for fleet capacity,
`fak frontierswe` for FrontierSWE cache/score witnesses, `fak fused` for fused-turn
diagnostics, and `fak ablate-arm` when you need an explicit ablation arm in a comparison.

**Lab:**
```bash
fak serve --gguf models/qwen.gguf --policy-check   # validate model+policy load without binding a listener
```

**Checkpoint:** Given 'bind: address already in use', diagnose and fix it two ways; explain the troubleshooting step for a GGUF that embeds no usable BPE tokenizer (the offline-mock-planner fallback), and situate serve among the run/preflight/bench/policy verbs.

### FAK 511 — The Integration Index: Repoint One Base URL

**Prerequisites:** **FAK 503**

**You'll be able to:**
- Identify the one configuration value a team changes to route every proposed tool call through fak
- State what does NOT change (the agent code itself)
- Pick the right per-agent integration guide from the index

**Read:** [`docs/integrations/README.md`](../integrations/README.md)

**Lab:**
```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # expect DENY (POLICY_BLOCK); then --tool search_kb expecting ALLOW
```

**Checkpoint:** Given a team running LangChain against Ollama, name the one configuration value they change to route every proposed tool call through fak, and state what does NOT change.

### FAK 512 — Claude Code / Anthropic API Through fak

**Prerequisites:** **FAK 511**

**You'll be able to:**
- Point ANTHROPIC_BASE_URL at the gateway ORIGIN (not the /v1 path) and run the dogfood launcher
- Read the denial table and the _fak/fak response extension
- Predict the verdict for a dangerous call under the dogfood policy

**Read:** [`docs/integrations/claude.md`](../integrations/claude.md)

**Lab:**
```bash
./scripts/dogfood-claude.sh --probe "Reply with exactly the word: pong"  (Windows: .\scripts\dogfood-claude.ps1 --probe "say pong"); then ./fak preflight --tool Bash --args '{"command":"rm -rf /tmp/x"}' --policy examples/dogfood-claude-policy.json
```

**Checkpoint:** Explain why the Anthropic base URL is the gateway ORIGIN (http://127.0.0.1:8080) and not the /v1 path, and predict the verdict for git push origin master under the dogfood policy.

### FAK 513 — OpenAI Codex / OpenAI SDK Through fak

**Prerequisites:** **FAK 511**

**You'll be able to:**
- Set OPENAI_BASE_URL (or SDK base_url) to fak's /v1 origin with no code change
- Apply coding-agent policy patterns (code-review, safe-refactor, dry-run DevOps)
- Show the two-step migration from a direct OpenAI client

**Read:** [`docs/integrations/openai-codex.md`](../integrations/openai-codex.md)

**Lab:**
```bash
./fak serve --addr 127.0.0.1:8080 --base-url http://localhost:11434/v1 --model codellama:7b --policy examples/dev-agent-policy.json  &&  ./fak preflight --tool Bash --args '{"command":"git push origin main"}' --policy examples/dev-agent-policy.json
```

**Checkpoint:** Show the two-step change that adds the kernel boundary to an existing openai.OpenAI(api_key=...) client, and explain why the application code itself stays unchanged.

### FAK 514 — Cursor via MCP or OpenAI Proxy

**Prerequisites:** **FAK 511**

**You'll be able to:**
- Wire fak into Cursor as a native MCP server (ask-the-kernel) or as an OpenAI-compatible proxy
- Contrast ask-the-kernel with transparent-proxy and write the JSON config for each
- Decide when to choose MCP over the proxy integration

**Read:** [`docs/integrations/cursor.md`](../integrations/cursor.md)

**Lab:**
```bash
./fak policy --dump > cursor-policy.json  &&  ./fak policy --check cursor-policy.json  &&  ./fak preflight --tool read_file --args '{"path":"test.txt"}' --policy cursor-policy.json
```

**Checkpoint:** Describe when you would choose Cursor's MCP integration over the OpenAI-proxy integration, and what each gives you at the tool boundary.

### FAK 515 — MCP One-Paste Setup and the fak_* Tools

**Prerequisites:** **FAK 511**, **FAK 509**

**You'll be able to:**
- Run fak serve --stdio as an MCP server exposing fak_adjudicate, fak_syscall, fak_admit, fak_changes, fak_revoke
- Drop a .mcp.json at the project root and complete the stdio handshake
- Name which fak_* tool you call BEFORE running a tool vs AFTER

**Read:** [`examples/mcp/README.md`](../../examples/mcp/README.md), [`docs/integrations/adopter-playbook.md`](../integrations/adopter-playbook.md)

**Lab:**
```bash
python examples/mcp/verify.py   # PASS/FAIL, exit 0/1 — drives the real stdio transport: initialize, tools/list, git_push->DENY, git_status->ALLOW
```

**Checkpoint:** Name which fak_* tool you call BEFORE running a tool your own client executes vs which one you call AFTER, and state what each protects against.

### FAK 516 — Agent<->Kernel Architecture and the Frozen ABI Verdict Union

**Prerequisites:** **FAK 511**, **FAK 208**

**You'll be able to:**
- Name the six verdict kinds in the closed union
- Explain 'deny-as-value': which HTTP status a policy refusal carries and what an HTTP error status is reserved for
- Use the stable contract (gateway entry points, ToolCall struct, internal/abi/types.go) that every integration depends on

**Read:** [`docs/fak/agent-integration-architecture.md`](../fak/agent-integration-architecture.md)

**Lab:**
```bash
curl http://127.0.0.1:8080/v1/fak/changes?since=0  &&  curl -X POST http://127.0.0.1:8080/v1/fak/revoke -H 'Content-Type: application/json' -d '{"witness":"git-commit-abc123"}'
```

**Checkpoint:** Name the six verdict kinds in the closed union and explain what 'deny-as-value' means: which HTTP status does a policy refusal carry, and what is an HTTP error status reserved for?

### FAK 517 — Framework Cookbook: Transparent Proxy (Mode A) vs Explicit Adjudication (Mode B)

**Prerequisites:** **FAK 516**, **FAK 513**, **FAK 302**

**You'll be able to:**
- Give the smallest per-framework change for LangChain/LangGraph, LlamaIndex, AutoGen, CrewAI (plus Semantic Kernel, Haystack, Griptape)
- Write the shared guarded() wrapper that adjudicates and admits (Mode B)
- Apply the honest scope (the floor bounds tool NAMES not arguments) and choose proxy vs explicit adjudication

**Read:** [`docs/fak/agent-framework-integration.md`](../fak/agent-framework-integration.md)

**Lab:**
```bash
fak serve --addr 127.0.0.1:8080 --base-url http://localhost:11434/v1 --model qwen2.5:1.5b --policy policy.json  &&  curl -s -X POST http://127.0.0.1:8080/v1/fak/adjudicate -H 'Content-Type: application/json' -d '{"tool":"refund_payment","arguments":{}}'
```

**Checkpoint:** For LangChain, give the Mode A one-line change AND the Mode B guarded() wrapper, and explain the honest-scope caveat about why you keep irreversible operations OFF the allow-list.

### FAK 518 — Migration: Moving Existing Code by Repointing a Base URL

**Prerequisites:** **FAK 516**

**You'll be able to:**
- Migrate LangChain, AutoGen, llama.cpp, or a direct OpenAI/Anthropic client by redirecting the base URL
- State the two invariants that hold for every migration (fak never executes your tools; a refusal is a 200 carrying a value)
- Diagnose the OpenAI vs Anthropic base-URL gotcha

**Read:** [`docs/fak/migration-guide.md`](../fak/migration-guide.md)

**Lab:**
```bash
fak serve --addr 127.0.0.1:8080 --provider openai --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --model gpt-4o --policy policy.json  &&  fak preflight --policy policy.json --tool git_push --args '{}'
```

**Checkpoint:** A client gets 404 on /v1/v1/messages. Diagnose the cause and the fix, then state which two invariants hold for every migration.

### FAK 519 — Multi-Language Client Code and Disposition-Aware Retry

**Prerequisites:** **FAK 516**, **FAK 509**

**You'll be able to:**
- Call the fak-native one-POST-one-verdict surface from Python, JS/TS, Go, and Rust
- Read verdict.kind (never HTTP status alone) and branch on disposition to spend zero extra model turns
- Explain how the four dispositions change retry logic

**Read:** [`docs/fak/multi-language-examples.md`](../fak/multi-language-examples.md)

**Lab:**
```bash
curl -s -X POST http://127.0.0.1:8080/v1/fak/adjudicate -H 'Content-Type: application/json' -d '{"tool":"Bash","arguments":{"command":"rm -rf /tmp/x"}}'   # inspect verdict.kind / reason / disposition
```

**Checkpoint:** Given a DENY verdict, explain how the four dispositions (RETRYABLE, WAIT, ESCALATE, TERMINAL) change your client's retry logic, and state why you must read verdict.kind instead of the HTTP status code.

### FAK 520 — The Adopter Playbook: Front-a-Model, Manual MCP, Embed-in-CI

**Prerequisites:** **FAK 512**, **FAK 515**

**You'll be able to:**
- Run the bare-serve production loop (author policy, bind an auth-key env, start, check /healthz, repoint base URL)
- Serve all three shapes (A proxy, B stdio MCP, C offline CI gate) from one binary
- Explain why --require-key-env matters once the bind address is not loopback

**Read:** [`docs/integrations/adopter-playbook.md`](../integrations/adopter-playbook.md)

**Lab:**
```bash
fak policy --dump > policy.json  &&  fak policy --check policy.json  &&  export FAK_TOKEN=$(openssl rand -hex 32)  &&  fak serve --addr 0.0.0.0:8080 --provider openai --base-url http://127.0.0.1:11434/v1 --model qwen2.5-coder:7b --policy policy.json --require-key-env FAK_TOKEN  &&  curl -s http://127.0.0.1:8080/healthz
```

**Checkpoint:** List the five ordered steps of the bare-serve loop (Shape A), and explain why --require-key-env matters once the bind address is not loopback.

### FAK 521 — GGUF Loading: Offsets, Dtypes, and Dequant Layout

**Prerequisites:** **FAK 205**

**You'll be able to:**
- Address each tensor's own byte window off the hot path and dequantize every block format to f32
- Map GGUF tensor names to HF names
- Compute an absolute FileOffset from an in-data offset and alignment, and explain why reading tensor i can never address tensor j's bytes

**Read:** [`docs/proofs/ggufload.md`](../proofs/ggufload.md)

**Lab:**
```bash
go test ./internal/ggufload/ -count=1 -timeout 120s -run 'TestReadParsesMetadataTensorDirectoryAndConfig|TestWeightSourceReadsAndDequantizesSimpleTensors' -v
```

**Checkpoint:** Given a tensor declared at in-data offset 64 with 64-byte alignment, compute its absolute FileOffset and explain why reading tensor i can never address tensor j's bytes. Why is the strict encode-then-read involution OPEN here?

### FAK 522 — Tokenizer: Lossless ByteLevel BPE With Oracle Parity

**Prerequisites:** **FAK 521**

**You'll be able to:**
- Convert text to/from token ids via a ByteLevel byte-to-unicode bijection and lowest-rank-first BPE merges
- Explain why BPE merge selection is deterministic (a pure function of symbols + merge ranks)
- Explain why the per-model pre-tokenizer dispatch (Qwen Split regex vs GPT-2 ByteLevel) is needed for oracle parity

**Read:** [`docs/proofs/tokenizer.md`](../proofs/tokenizer.md)

**Lab:**
```bash
go test -run 'TestEncodeSmallByteLevelBPEFixture|TestDecodePreservesSplitUTF8Bytes|TestQwenOracleGolden' -v ./internal/tokenizer/ -count=1 -timeout 120s
```

**Checkpoint:** Explain why BPE merge selection is deterministic and why the per-model pre-tokenizer dispatch is needed for oracle parity.

### FAK 523 — Normalization: RMSNorm, NormGain1p, and LayerNorm

**Prerequisites:** **FAK 522**

**You'll be able to:**
- Compute RMSNorm, Gemma's (1+w) gain, and mean-subtracting LayerNorm to their closed forms
- Explain why the sum-of-squares is kept scalar in-order so f32 forward rungs stay bit-reproducible
- State the approximate input magnitude at which the f32 sum-of-squares overflows

**Read:** [`docs/proofs/model-norm.md`](../proofs/model-norm.md)

**Lab:**
```bash
go test -run 'TestNormGain1p|TestLayerNormAxis|TestProofNormNumericallyStableLargeInputs' ./internal/model/ -count=1 -timeout 120s -v
```

**Checkpoint:** Write the closed form RMSNorm computes and state why LayerNorm is shift+scale equivariant in the eps->0 limit. At roughly what input magnitude does the f32 sum-of-squares overflow?

### FAK 524 — RoPE: Rotary Position Embedding and Scaling Variants

**Prerequisites:** **FAK 523**

**You'll be able to:**
- Inject position by Givens-rotating each dim-pair by p*inv_freq and show attention depends only on (m-n)
- Apply llama3/yarn/longrope frequency rescaling
- Explain why the yarn/longrope attention-factor scale breaks per-pair norm preservation

**Read:** [`docs/proofs/model-rope.md`](../proofs/model-rope.md)

**Lab:**
```bash
go test -run 'TestProofRopePreservesPairNorm|TestProofRopeDotRelativePosition|TestRopeScalingLlama3' ./internal/model/ -count=1 -timeout 120s -v
```

**Checkpoint:** Prove <R_m q, R_n k> depends on m,n only through (m-n), and explain why the yarn/longrope attention-factor scale breaks per-pair norm preservation (cos^2+sin^2=scale^2!=1).

### FAK 525 — Attention: Stable Softmax, Causal Mask, and the Attention Sink

**Prerequisites:** **FAK 524**

**You'll be able to:**
- Compute scaled-dot-product attention with a row-stochastic shift-invariant softmax
- Explain why the score loop makes causality structural rather than after-the-fact masking
- Derive the single-visible-score sink weight 1/(1+exp(sink-s))

**Read:** [`docs/proofs/model-attention.md`](../proofs/model-attention.md)

**Lab:**
```bash
go test -run 'TestAttentionSinkSoftmaxDropsSink|TestProofSoftmaxRowStochasticAndShiftInvariant|TestProofCausalStrictlyLowerTriangular' ./internal/model/ -count=1 -timeout 120s -v
```

**Checkpoint:** Explain why the score loop `for j := lo; j <= t` makes causality structural rather than after-the-fact masking, and derive the single-visible-score sink weight.

### FAK 526 — MLP / SwiGLU+GeGLU, MoE Routing, and the Residual Stream

**Prerequisites:** **FAK 525**

**You'll be able to:**
- Compute the gated MLP down(act(gate(x))*up(x)) and top-k MoE weighted-sum routing
- Describe torch.topk's stable tie-break and NormTopKProb renormalization
- Name the four residual topologies (PreNorm/PostNorm/Sandwich/Parallel) and how each composes the sub-layer delta

**Read:** [`docs/proofs/model-mlp+residual.md`](../proofs/model-mlp+residual.md)

**Lab:**
```bash
go test -run 'TestMoEDenseNoOpIdentical|TestBlockTopologyComposition|TestMoERoutingHandComputed' ./internal/model/ -count=1 -timeout 120s -v
```

**Checkpoint:** Describe MoE top-k routing including torch.topk's stable tie-break and NormTopKProb renormalization, and name the four residual topologies and how each composes the sub-layer delta.

### FAK 527 — In-Kernel KV Cache: Slotting, Span-Exact Eviction, SWA, Prefix Reuse

**Prerequisites:** **FAK 526**, **FAK 406**

**You'll be able to:**
- Correctly slot (layer,pos,head) and Evict byte-identically to never-having-seen a span
- Explain why eviction re-rotates each survivor's K from stored pre-RoPE Kraw in a SINGLE rotation
- Explain why the sliding window keys off pos[] rather than the slice index

**Read:** [`docs/proofs/model-kv.md`](../proofs/model-kv.md)

**Lab:**
```bash
go test -run 'TestStandardLayoutNoOp|TestKVQuarantineEqualsNeverSaw|TestSWAWindowMasksOldKeys|TestKVPrefixReuseMatchesRecompute' ./internal/model/ -count=1 -timeout 180s -v
```

**Checkpoint:** Explain why eviction re-rotates each survivor's K from stored pre-RoPE Kraw in a SINGLE rotation rather than composing two, and why the sliding window keys off pos[] instead of the slice index.

### FAK 528 — Quantization: Q4_K/Q8_0/Q4_0 Dequant, AWQ, and Bit-Identical int8 SDOT

**Prerequisites:** **FAK 521**, **FAK 526**

**You'll be able to:**
- Apply affine-correct dequant of GGUF k-quant and AWQ 4-bit formats
- Explain why the int8 SDOT reduction is bit-identical across SIMD lane orders (order-independent, no overflow)
- Distinguish what the AWQ 'matches reference' claim PROVES (affine self-consistency) from what is OPEN (no HF AutoAWQ fixture)

**Read:** [`docs/proofs/model-quant.md`](../proofs/model-quant.md), [`docs/explainers/awq-quantization.md`](../explainers/awq-quantization.md)

**Lab:**
```bash
go test -run 'TestQ4KDequantSuperBlockMatchesRef|TestQ4KReduceAsmMatchesScalar|TestProofAWQMatchesReference' ./internal/model/ -count=1 -timeout 120s -v
```

**Checkpoint:** State the AWQ dequant formula scale[o]*(code-8) and explain why the int8 SDOT reduction is bit-identical across SIMD lane orders. Which part of the AWQ claim is PROVEN and which is OPEN?

### FAK 529 — Forward-Pass Parity vs the HuggingFace Oracle

**Prerequisites:** **FAK 527**, **FAK 528**, **FAK 210**

**You'll be able to:**
- Reproduce PyTorch/HF hidden-state cosine ~1, per-position argmax, and greedy ids token-for-token on smollm2
- Explain why argmax-pin at every position is a stronger witness than a logit tolerance
- Read the honest ledger: PROVEN on llama, OPEN for other families, REFUTED for Qwen3.6 hybrid-GDN (diverges at token 3)

**Read:** [`docs/proofs/model-forward-parity.md`](../proofs/model-forward-parity.md)

**Lab:**
```bash
go test -run 'Oracle|Parity|Greedy|Argmax|Forward' ./internal/model/ -count=1 -timeout 240s -v
```

**Checkpoint:** Explain why argmax-pin at every position is a stronger witness than a logit tolerance, and describe the Qwen3.6 REFUTED finding (near-tie argmax flip at token 3) without conflating it with the llama PROVEN row.

### FAK 530 — The Compute HAL Seam and Hardware Portability

**Prerequisites:** **FAK 529**, **FAK 210**

**You'll be able to:**
- Name three of the seven baked-in hardware assumptions the internal/compute Backend interface neutralizes and the type that lifts each
- Explain why adding a GPU/NPU is a registration, not a fork of the hot loop
- Explain why only a Reference backend faces max|delta|=0 while every Approx faces argmax-exact + logit-cosine

**Read:** [`docs/explainers/hardware-portability.md`](../explainers/hardware-portability.md), [`docs/proofs/compute-gemm.md`](../proofs/compute-gemm.md)

**Lab:**
```bash
go test -run 'MatMul|Reduction|Q8|Correctness|Registry|Device' ./internal/compute/ -count=1 -timeout 120s -v
```

**Checkpoint:** Name three of the seven assumptions the seam neutralizes and the type that lifts each, and explain why only a Reference backend faces max|delta|=0 while every Approx faces argmax-exact + logit-cosine.

### FAK 531 — Metal GPU GEMM Parity and the Stub-vs-Device Build

**Prerequisites:** **FAK 530**
  ·  **Background:** **FAK 534**

**You'll be able to:**
- Match Apple-Silicon Metal GEMM (f16 MPS) to the f32 CPU reference within the half-precision error model
- Explain why the witness is err/scale<1% and logit-cosine=1.0 rather than a bit-compare
- Explain how mutually-exclusive build tags guarantee the stub introduces no numerical drift

**Read:** [`docs/proofs/metalgemm.md`](../proofs/metalgemm.md)

**Lab:**
```bash
CGO_ENABLED=1 go test -run 'MatMul|Reset' ./internal/metalgemm/ -count=1 -v   # (Apple Silicon only; default build links Metal when cgo is enabled)
```

**Checkpoint:** Explain why the Metal witness is err/scale<1% and logit-cosine=1.0 rather than a bit-compare, and how the mutually-exclusive build tags guarantee the stub introduces no numerical drift.

### FAK 532 — The Engine Seam: Determinism and Cache-Invalidation Binding

**Prerequisites:** **FAK 529**, **FAK 206**

**You'll be able to:**
- Explain why greedy decode makes Complete a pure function of (tool,args) (no RNG/clock)
- Bind enginecache invalidation directives to SGLang/vLLM resets
- Explain the fail-closed gate: why Invalidate errors BEFORE issuing any reset when RequiredScope==exact_span but the engine only supports whole-prefix reset

**Read:** [`docs/proofs/engine-seam.md`](../proofs/engine-seam.md)

**Lab:**
```bash
go test ./internal/modelengine/ -run 'TestDecodeIsDeterministicAndInputDriven|TestCompleteRunsRealDecode' -count=1 -v && go test ./internal/enginecache/ -count=1 -v
```

**Checkpoint:** Explain why greedy decode makes Complete a pure function of (tool,args), and describe the fail-closed gate when RequiredScope==exact_span but the engine only supports whole-prefix reset.

### FAK 533 — In-Kernel Model & Compute Env Knobs (FAK_* Engine Vars)

**Prerequisites:** **FAK 502**, **FAK 528**

**You'll be able to:**
- Tune GPU residency budget, Q4K/Q8 load format, matmul worker budget, SIMD tiers, and generation bounds
- Distinguish FAK_WORKERS vs FAK_BUDGET for matmul parallelism
- Separate the model-engine-env vars from the serve-config vars

**Read:** [`docs/model-engine-env.md`](../model-engine-env.md), [`docs/fak/server-config.md`](../fak/server-config.md), [`GETTING-STARTED.md`](../../GETTING-STARTED.md)

**Lab:**
```bash
FAK_Q4K=1 fak serve --addr 127.0.0.1:8137 --gguf ~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf --model qwen3.6-27b-q4k
```

**Checkpoint:** Explain what FAK_Q4K changes about the load/decode path for a Qwen3.6-27B model, how FAK_WORKERS vs FAK_BUDGET differ, and which FAK_* vars belong to model-engine-env vs serve-config.

### FAK 534 — GPU Lease: Machine-Wide Mutual Exclusion for Model Residency

**Prerequisites:** **FAK 533**

**You'll be able to:**
- Explain why at most one live holder machine-wide is required before two processes both try to make a model resident on the same GPU
- Explain the three regime-D properties: fail-closed-when-busy (no-wait), bounded wait-then-acquire, and crashed-holder reclaim via flock release on process exit
- Identify this as the operational precondition for Tier-2b real-weights serving (FAK 533) and Metal modelbench (FAK 531)

**Read:** [`docs/proofs/gpulease.md`](../proofs/gpulease.md)

**Lab:**
```bash
go test ./internal/gpulease/ -count=1 -timeout 120s -run 'TestNoWaitBusyThenFree|TestWaitTimesOut|TestWaitThenSucceed|TestReleaseOnProcessExit|TestReleaseIdempotent' -v
```

**Checkpoint:** Explain why a machine-wide flock guarantees at most one live holder, why a busy lease fails closed (no-wait) rather than racing, and how a crashed holder's lease is reclaimed without a manual unlock. State why this is the precondition for the real-weights modelbench path.

### FAK 535 — The Gateway Drop Guarantee: Fail-Closed on a Failed Adjudication

**Prerequisites:** **FAK 510**, **FAK 314**

**You'll be able to:**
- State the two regime-D theorems: a wire verdict equals the in-process kernel verdict (no network bypass), and a call that fails adjudication is dropped fail-closed
- Explain why the wire never carries an abi.Ref so a client cannot smuggle a pre-trusted CAS handle to skip the IFC / self-modify rungs
