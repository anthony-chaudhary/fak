# Concept Study: Lemonade SDK (lemonade-sdk/lemonade)

**Repository:** https://github.com/lemonade-sdk/lemonade  
**Pinned Revision:** `bc8d99ccfee301742f1dd86a3f6ec406bb8a863d`  
**Study Date:** 2026-09-03  
**Status:** Studied  
**Receipt ID:** `study_cc3b4fe925b988801bccaeb92d141ee0e0f0a37cc17e8edb0e4fb3a883df1e69`  
**License:** Apache-2.0 (Permissive, compatible)  

---

## Repository Overview

Lemonade is an open-source, local multi-model AI inference and routing runtime designed to serve optimized LLMs, vision, speech, and diffusion models on consumer and workstation hardware (AMD Ryzen AI NPUs, Radeon GPUs, NVIDIA GPUs, Apple Silicon Metal, and CPUs).

**Key Architectural Subsystems:**
- **Inference Runtime (`lemond`)** — C++20 daemon abstracting heterogeneous backends (`llama.cpp`, `FastFlowLM`, `RyzenAI`, `vLLM`, `whisper.cpp`, `sd.cpp`, `Kokoro`, `Moonshine`). Backends run as managed subprocesses communicating over local HTTP sockets.
- **Model Auto-Tuning & GGUF Metadata** — Probes system VRAM, GTT, and physical RAM to calculate achievable context sizes. Calculates weighted KV-cache bytes per token directly from GGUF architecture metadata (including sliding-window attention patterns).
- **Speculative Draft Companion Discovery** — Scans HuggingFace and local checkpoints to detect companion draft GGUFs (`mtp-*`, `dflash-*`), grouping and matching them to main models by directory depth, quant tag matching, and nearest bit-distance.
- **Resilient Streaming Proxy** — Streaming forwarder that intercepts GPU compute hangs and backend crashes, discriminating between zero-byte-streamed turns (safe to reset backend and transparently replay) and active streaming turns (where replay would duplicate tokens or corrupt protocol framing).
- **Eviction Engine & VRAM Monitor** — Active memory management scoring candidate models by `idle_time / (load_duration * weight_factor)`, guaranteeing fast-loading models are sacrificed before heavy models during VRAM pressure.
- **Smart Routing Policy Engine** — Pure model selection engine evaluating AST boolean expressions (`All`, `Any`, `Not`, `Leaf`) over request features and classifier services, attaching route decisions both to HTTP response headers and in-band to the first SSE streaming event.
- **OS Suspend Inhibitor** — Refcounted lock holding systemd-logind DBus `Inhibit("sleep:idle")` during active inference so long turns and overnight runs are not killed by OS sleep.
- **Dynamic Model Alias Manager** — Thread-safe, persistent alias store supporting multi-hop alias resolution, cycle prevention, and instant active-standby hot failover.

---

## Source Classes Covered

| Class | Coverage | Notes |
|-------|----------|-------|
| `readme_docs` | ✅ | README.md, DESIGN.md, AGENTS.md, docs/dev/ |
| `architecture_design` | ✅ | C++ server architecture, router, wrapped_server, eviction_engine |
| `runtime_source` | ✅ | `src/cpp/server/` (49 files), `src/cpp/include/lemon/` (59 files), `src/cpp/cli/` (15 files) |
| `tests_fixtures` | ✅ | `test/` (55 test files) and `test/cpp/` (87 C++ unit tests) |
| `history_changelog_releases` | ✅ | Releases v11.6.0 through v11.9.0, git log at tip `bc8d99c` |
| `open_closed_issues_prs_discussions` | ✅ | Examined open issues (#3495, #3431, #3426, #3434, #3433, #3461) |
| `roadmap_todos` | ✅ | Smart router roadmap, MCP server integration RFCs |
| `license_provenance` | ✅ | Apache-2.0 root license + MIT third-party notices (aixlog) |
| `fak_selfquery_witness` | ✅ | Tested on-axis against `internal/ggufload`, `internal/compute`, `internal/polymodel`, `internal/gateway` |
| `candidate_matrix` | ✅ | 9 candidates extracted, ablated, and classified |
| `completeness_critic` | ✅ | Exhaustive fan-out across all 8 load-bearing subsystems; no gaps |
| `issue_tracking` | ✅ | 6 issues filed (#11086-#11091) |

---

## Completeness Critic

**Subsystems Inspected (Fan-Out):**
1. **Model Management & Speculative Variants** (`src/cpp/server/hf_variants.cpp`, `model_manager.cpp`) — ✅ Deep
2. **Backend Execution & Streaming Resilience** (`src/cpp/server/wrapped_server.cpp`, `streaming_proxy.cpp`) — ✅ Deep
3. **Eviction Engine & Global VRAM Monitor** (`src/cpp/server/eviction_engine.cpp`, `global_vram_monitor.cpp`) — ✅ Deep
4. **Smart Routing Policy & In-Band SSE Injection** (`src/cpp/server/routing_policy*.cpp`, `route_decision_response.cpp`) — ✅ Deep
5. **Operating System Power Management** (`src/cpp/server/platform/suspend_linux.cpp`) — ✅ Deep
6. **Dynamic Model Aliasing** (`src/cpp/server/alias_manager.cpp`) — ✅ Deep
7. **Job Engine & Expression DAG** (`src/cpp/server/jobs/job_manager.cpp`, `docs/dev/job-system.md`) — ✅ Deep
8. **Memory Allocator Configuration** (`src/cpp/server/main.cpp`) — ✅ Deep

**Subsystems Not Opened (Justified):**
- `src/app/` (Tauri desktop UI components in React/TypeScript) — GUI presentation layer; not relevant to agent kernel / gateway mechanics.
- `tools/` and installer scripts — Distribution packaging (.msi, .deb, .rpm), not runtime architecture.

**Verdict:** The completeness critic finds **no material runtime subsystem unopened**.

---

## Candidate Matrix

| # | Technique | Source Anchor | Axis | Witness Status | Disposition | Worldview Reason & Tradeoff | Filed Issue |
|---|-----------|---------------|------|----------------|-------------|-----------------------------|-------------|
| 1 | **Speculative companion discovery & bit-distance matching** | `src/cpp/server/hf_variants.cpp:137-186@bc8d99c` | Automatic GGUF draft pairing & quant tolerance | **PARTIAL** | DEFAULT | Desktop users download separate companion files (`mtp-*`, `dflash-*`); manual pairing fails on slight quant mismatches. `fak` supports in-file NextN layers but lacks companion sidecar discovery. | [#11086](https://github.com/anthony-chaudhary/fak/issues/11086) |
| 2 | **Zero-byte-streamed transparent retry for compute errors** | `src/cpp/server/wrapped_server.cpp:845-877@bc8d99c` | Accelerator fault resilience without token duplication | **PARTIAL** | DEFAULT | Consumer GPUs hit driver timeouts during prefill. If zero bytes were streamed, reloading and replaying saves the turn; if bytes streamed, replay is banned to prevent duplication. `fak` retries upstream HTTP 429s but lacks local engine zero-byte replay. | [#11087](https://github.com/anthony-chaudhary/fak/issues/11087) |
| 3 | **Reload-cost-weighted model eviction scoring** | `src/cpp/server/eviction_engine.cpp:110-118@bc8d99c` | Reload latency minimization under VRAM pressure | **PARTIAL** | DEFAULT | Alternating between a 0.5B embedder and a 14B LLM causes pure LRU to evict the heavy LLM if touched 1ms earlier. Scoring `idle_time / (load_duration * weight)` protects expensive models. `fak`'s `polymodel.Pool` currently uses unweighted LRU. | [#11088](https://github.com/anthony-chaudhary/fak/issues/11088) |
| 4 | **In-band first-chunk route decision injection in SSE streams** | `src/cpp/server/route_decision_response.cpp:138-193@bc8d99c` | In-band routing observability for streaming consumers | **ABSENT** | DEFAULT | Agent clients and streaming SDKs consume SSE tokens directly and drop HTTP response headers. Framing route decisions into the first SSE chunk ensures instant in-band visibility. `fak` emits HTTP headers but no in-band SSE decision events. | [#11089](https://github.com/anthony-chaudhary/fak/issues/11089) |
| 5 | **OS suspend and idle inhibitor during active runs** | `src/cpp/server/platform/suspend_linux.cpp:25-98@bc8d99c` | Power-management sleep prevention during long tasks | **ABSENT** | DEFAULT | Workstations sleep automatically on user inactivity. Long prefill passes, overnight agent batches (`run-it-all-night`), and serving instances get severed by sleep. `fak` currently has no OS sleep inhibitor. | [#11090](https://github.com/anthony-chaudhary/fak/issues/11090) |
| 6 | **Dynamic model alias registry with active-standby failover** | `src/cpp/server/alias_manager.cpp:23-48@bc8d99c` | Hot model re-targeting without client restarts | **ABSENT** | DEFAULT | Automated agents configure target aliases (`production-llm`). Re-targeting or active-standby failover should require zero client restarts. `fak` currently hardcodes static aliases in Go code. | [#11091](https://github.com/anthony-chaudhary/fak/issues/11091) |
| 7 | **SWA ratio-aware KV cache context sizing** | `src/cpp/include/lemon/auto_tune.h:214-239@bc8d99c` | Precise KV memory budgeting for hybrid attention | **PRESENT** | DEFAULT | Modern architectures (Gemma 2/3, Mistral, Qwen 3.5) interleave local sliding-window and global attention layers. `fak` already implements per-layer SWA bounding in `internal/compute/capacity.go:160-174` (`WindowPerLayer`). | — |
| 8 | **Server-side forward-only DAG job execution engine** | `src/cpp/server/jobs/job_manager.cpp:45-120@bc8d99c` | Multi-step benchmark execution surviving client drops | **DIVERGENT** | EXCLUDE | Lemonade built a REST-exposed DAG job engine because its Tauri/browser UI loses state on page reloads. `fak` is a CLI-native kernel and orchestrator driven by host Go CLI verbs (`fak bench`, `fak dos`, `run-it-all-night`) with durable JSONL ledgers. | — |
| 9 | **Glibc malloc mmap threshold floor pinning (`mallopt`)** | `src/cpp/server/main.cpp:35-73@bc8d99c` | Heap fragmentation mitigation for large HTTP buffers | **DIVERGENT** | EXCLUDE | In C++ with `cpp-httplib`, multi-megabyte image/audio payloads inflate glibc's dynamic malloc mmap threshold, causing RSS retention. Go's runtime manages memory arenas directly via OS mmap and spans, bypassing glibc malloc for all server buffers. | — |

---

## Registration & Durable Receipts

- **Receipt Store:** `study_cc3b4fe925b988801bccaeb92d141ee0e0f0a37cc17e8edb0e4fb3a883df1e69` (supersedes `study_2e5b46c51a0f5d71aa6755bd0a0203db752657b0eecd4c2c913a979772a3346d`)
- **Inventory Path:** `docs/research/inventory/lemonade-sdk-lemonade.json`
- **Monitored Ledger:** Registered in `docs/research/monitored-repositories.json`

## Companions

- [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md)
- [`docs/notes/CONCEPT-STUDY-OPENCODE-2026-09-02.md`](CONCEPT-STUDY-OPENCODE-2026-09-02.md)
- [`docs/notes/CONCEPT-STUDY-PORTKEY-AI-GATEWAY-2026-09-02.md`](CONCEPT-STUDY-PORTKEY-AI-GATEWAY-2026-09-02.md)
- [`docs/research/monitored-repositories.json`](../research/monitored-repositories.json)
