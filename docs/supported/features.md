---
title: "Features supported by fak — capability gate, KV cache, gateway, with honest status"
description: "Every fak capability grouped by subsystem with its honest shipped / simulated / stub status, a reader-friendly view of the CLAIMS.md ledger: adjudication and the capability floor, the tool vDSO fast path, the context-MMU result quarantine, the in-kernel model and addressable KV cache, the security substrate, the gateway, and the benchmarks."
---

# Features supported by fak

This page lists every fak capability grouped by subsystem. It is the reader-friendly view of the [claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md); it does not re-grade anything. Each row carries the same status the ledger assigns, and the ledger is the witnessed source of record.

fak tags every capability with exactly one of three honest states:

- **Shipped** — real code on the critical path, closed by a mechanical witness (a `go test`, a `go build`, a benchmark field, a file read-back). Reproducible now.
- **Simulated** — a real seam with labeled stand-in numbers. There is no GPU or live serving engine on the build box, so the seam is real and the numbers are illustrative.
- **Stub** — plumbing present, behavior deferred. Clearly labeled, returns a STUB or no-op result.

## The product (one binary)

| Feature | Status | What it does |
|---|---|---|
| One statically-linked `fak` binary | Shipped | Runs an agentic tool loop where every tool call crosses one in-process syscall boundary. |
| Process-level fusion | Shipped | Harness, reference monitor, vDSO, pre-flight, and context-MMU collapsed into one Go address space. No spawned hook, no IPC on the decide path (`TestNoOsExecOnHotPath`). |
| Frozen, additive-only ABI | Shipped | A machine-checked contract over the syscall envelope and the discriminated-union Verdict (`TestABIGoldenFreeze`). |
| Zero data races | Shipped | The whole module passes the Go race detector in the `race-detector` CI job; the Windows build box has no cgo, so it runs via WSL/Linux/CI. |
| In-process adjudication latency check | Shipped | A subsystem regression sentinel timing the in-process fold against a spawned-hook baseline on the same machine. Deliberately not a production-readiness or throughput headline. |

## Adjudication and the capability floor

| Feature | Status | What it does |
|---|---|---|
| Provable-deny / unprovable-defer | Shipped | Provable refusal returns Deny, anything unprovable returns Defer; default-deny on an empty policy. |
| Structured 12-reason refusal | Shipped | Refusals come from a closed 12-reason vocabulary with a bounded-disclosure witness (a SELF_MODIFY deny returns only the offending glob). |
| Deny-as-value | Shipped | A refusal carries a derived disposition (RETRYABLE / WAIT / ESCALATE / TERMINAL) the loop consumes. |
| Batch adjudication | Shipped | Adjudicating a set of calls in one pass equals the serial result. |
| Deployable capability floor | Shipped | The policy is a declarative, version-tagged JSON manifest loaded at runtime (`--policy FILE`); unknown fields, reasons, or versions are a fatal load error. `fak policy --dump`/`--check` authors and validates it. |
| Git-shape prefilter | Shipped | A registered rung (`internal/gitgate`) refuses the argv-decidable git hazards (force-push, `commit --amend`, `add -A`, `--no-verify`, `tag -f`, `rebase -i`) at the call boundary, defers on state-dependent laws, and opts out with `FAK_GITGATE=off`. |
| Default dev-agent floor + ship gate | Shipped | `adjudicator.DevAgentPolicy()` denies shared-history git mutations, escalates a spine write as SELF_MODIFY, and lifts a ship call to `require-witness` so an unwitnessed ship is refused and a git-corroborated ship is allowed. |

## Tool vDSO fast path

| Feature | Status | What it does |
|---|---|---|
| 3-tier local fast path | Shipped | Tier-1 pure registry (gated on read-only + idempotent hints, re-checked not trusted), tier-2 content-addressed world-versioned LRU cache, tier-3 static table. |
| Arg-order-independent content keys | Shipped | Canonicalized JSON keys, so reordered arguments hit the same cache entry. |
| Write-shaped invalidation | Shipped | A write-shaped completion bumps the world-version and invalidates the cache, so a hit equals a fresh call. |
| Real-world vDSO hit-rate | Simulated | The demo trace is deliberately cache-favorable (~50% hits); measured addressable purity on real tau2-airline is ~0.7%. The vDSO is reported as an upside secondary, never the headline. |

## Pre-flight ladder and grammar repair

| Feature | Status | What it does |
|---|---|---|
| Rung-0 static parse + rung-1 schema check | Shipped | Cheapest-first, escalate-on-pass, with hard-negative label harvesting. |
| Grammar rung (positional to named auto-repair) | Shipped | An in-syscall TRANSFORM with no model turn for arity-matched calls; unrepairable calls Deny(MISROUTE); fail-open on unknown grammar; content-addressed grammar dedup. |
| Rung-2 dry-run + rung-3 sandbox probe | Stub | The offline/sandbox escalation rungs above rung-1 are not built in v0.1. |
| Decode-time logit-mask (grammar-constrained generation) | Stub | The never-emit-a-malformed-call form requires owning the decode loop; not in v0.1. |

## Context-MMU result quarantine

| Feature | Status | What it does |
|---|---|---|
| Result-admit gate | Shipped | Secret-shaped and prompt-injection/poison results are quarantined out of context to a stub pointer; oversize benign results page out to a <2KB pointer; byte-repeat pollution is quarantined. |
| Witness-gated page-in | Shipped | Page-in is gated on an explicit witness `Clear()`, with a pollution-rate counter and a content-addressed blob store shared with the vDSO. |
| `normgate` canonicalize-and-rescan driver | Shipped | A ResultAdmitter in front of ctxmmu that strips zero-width/bidi/homoglyph evasions and decodes base64/hex before rescanning; lifts measured agent-evasion catch 0 to 20 of 24. One blank-import to enable. |
| `headroom` (Rust) page-out codec | Simulated | The default page-out is pure-Go content-addressed; the headroom backend is an optional labeled seam, not on the critical path. |
| Answer-shape degeneration witness | Shipped | `answershape.Measure` grades repeat-n, repeated-line coverage, and short-period tiling against caller thresholds; `fak answer-shape` is the pipeline gate and `fak doctor` cross-checks the kernel admit verdict. |

## Codelint at the boundary

| Feature | Status | What it does |
|---|---|---|
| Language-server packs over agent-written code | Shipped | `codelint` reports only hard parse/compile errors; ships Go + JSON in-process and Python + CUDA shelling out (no-opinion when the toolchain is absent); honors no in-content ignore comment and runs off the hot path. |
| `fak codelint PATH...` write/definition-time check | Shipped | Routes each file to its owning pack and exits 1 on a hard error; the SWE-bench fleet runs the same packs over agent file writes when `--lint-writes` is set (advisory, off by default). |
| Write-scoped codelint verdict in the adjudicator (#536) | Shipped | Under the opt-in `LintWrites` policy, a whole-file write of unparseable Go/JSON is refused with Deny(MALFORMED) and a bounded `file:line:col` witness; off by default, fail-open for languages whose checkers shell out. |

## Durability gate (context is not memory)

| Feature | Status | What it does |
|---|---|---|
| Rung-1 write-time durability classifier | Shipped | A cheap lexical/tense prior assigns a benign result a turn / session / durable class and stamps it on the additive `Verdict.Meta` map (not a model call), failing closed to `turn`. |
| Zero ABI / golden-freeze cost | Shipped | The durability tag rides the additive `Meta` map, so the frozen ABI does not move. |
| Rung-1 default-expire promotion gate | Shipped | A `PromotionMode` gates promotion into the persisted core image, so only a `durable`-classed benign fact crosses the durable boundary. Closes the benign over-promotion arm of OWASP Memory-Poisoning T1; it is not the adversarial-T1 floor. |
| Two-commit honesty posture | Shipped | `PromotionWarn` (the default) stamps the class and counts would-refusals but still persists, so existing callers stay green; `PromotionEnforce` is opt-in. |
| Rung-2 bitemporal validity (#501) | Stub | A `recall.Page` validity interval plus an as-of read gate that makes `bounded` the first temporally-enforced class. |
| Rung-3 engine-integrated TTL (#502) | Stub | A `kvmmu.Segment` TTL over the bit-exact `model.KVCache.Evict`, so a span is forgotten on a clock the fact sets. |
| Dream-time durability consolidation | Stub | Principled sleep-time promotion over the rung-1 class signal (S7 epic #496). |

## Session core-dump and context debugger

| Feature | Status | What it does |
|---|---|---|
| `recall` core image | Shipped | A finished session persists as a durable core image: a `manifest.json` page table over a `cas.json` content-addressed swap device, reloaded in a fresh process with every blob integrity-checked against its digest. |
| Durable quarantine moat | Shipped | A page sealed at write time is refused on page-in across the process boundary unless a witness `Clear()` ran AND the bytes pass a fresh re-screen. |
| `cdb` context debugger | Shipped | Turns a real Claude Code transcript into a core image through the same gate and binds an inspection surface (Info/Backtrace/Examine/WorkingSet/Grep). |
| Demand-paging the working set | Shipped | A follow-up is answered by paging in only the pages it references; measured on a real 2.8 MB session (18 KB page table over a 1.2 MB swap device). |
| Agent/requester context tombstones | Shipped | A negative-only request suppresses a page from future model-visible recall without deleting CAS bytes; exposed via `fak debug`, HTTP, and MCP. |
| Inherited detection ceiling (surfaced) | Shipped | The same run sealed 2 of 59 pages as false positives; `cdb` makes the gate's decision durable and queryable, it does not improve the decision. |

## In-kernel model and addressable KV cache

| Feature | Status | What it does |
|---|---|---|
| Pure-Go SmolLM2-135M forward pass | Shipped | Runs in-process with the KV cache as a kernel-owned Go structure; every rung proven against a HuggingFace oracle (per-layer cos=1.000000, KV-decode token-for-token identical). |
| Parity lane | Shipped | Parallel matmul, batched prefill GEMM, and an 8-accumulator `fdot`, each bit-identical to the serial reference; decode beats every same-precision HF f32 config. |
| KV-quarantine bridge | Shipped | A ctxmmu Quarantine verdict mechanically evicts that result's K/V span, leaving the cache bit-identical (max|Δ|=0) to never-having-seen it (synthetic-model witness; live agent loop not yet wired). |
| Planned-elision to KV-eviction residency bridge (#550) | Shipped | `kvmmu.Context.ApplyPlan` evicts every elided span so the cache's residency shrinks to the planner's O(1) view, bit-for-bit (synthetic-model witness; not yet wired into the live HTTP loop). |
| Context-planner candidate index | Shipped | `ctxplan.Index` bounds the planner's per-turn compute (an inverted token index + recency tail + durable set), flattening cumulative planning from Θ(N²) to Θ(c·N); pruning is a forecast miss, never a lost fact. |
| Provable-deletion certificate | Shipped | `deletioncert` binds the evicted span, count, byte-exact equivalence, a hash-chained anchor, and the trust epoch under one ed25519 signature, failing closed on any forged field (self-signed v1 receipt). |
| In-kernel engine backend (`--engine inkernel`) | Shipped | The in-kernel model is wired as a `RegisterEngine` backend; an allowed tool call is completed by a real greedy decode over the kernel-owned KV cache. Builds a synthetic checkpoint with no export; `FAK_MODEL_DIR` loads a real one. |
| Poly-model serving core | Shipped | The deterministic "host many models, share the prefill, decode one" policy/accounting brain (residency pool, serial decode lane, speculative-accept core, cross-model prefill-share gate). Runs no model and is off mainline by construction (`FAK_POLYMODEL`, default off). |
| Single-pass batched + tree-attention verify | Shipped | `model.VerifyForward` turns the accept decision into a real one-pass forward over candidate tokens, token-identical to sequential decode; CPU synthetic regime, so no tokens/sec, off mainline (`FAK_POLYMODEL`). |
| Multi-model weight-residency layer | Shipped | `internal/residency` lifts the single-model assumption with a weight-byte budget and LRU page-out, reusing `polymodel.Pool`'s policy; moves no weight bytes, off mainline (`FAK_POLYMODEL`). |
| RadixAttention parity vs SGLang | Shipped | `internal/radixkv` rebuilds SGLang's radix-tree prefix reuse over the kernel-owned KVCache; measured 77.2–88.2% cache hit rate inside SGLang's 50–99% band, reuse-through-edge-split bit-identical to recompute. |
| int8/Q8_0 SIMD lane | Simulated | Hand-written AVX2/AVX-512 lane (CPUID-gated, scalar fallback) is the active in-flight increment, witnessed green in the working tree but deliberately not given a Shipped row in the ledger until the implementation lane commits it. |

GPU device compute is witnessed real on several backends (CUDA on an RTX 4070 and a datacenter sm_80 GPU, Vulkan on an AMD RX 7600, Metal on an Apple M3 Pro). See the [hardware matrix](../HARDWARE-MATRIX.md) and the [engines page](engines.md) for those rows.

## Security substrate (the kernel stops believing the model)

| Feature | Status | What it does |
|---|---|---|
| Information-flow control (IFC) | Shipped | `Ref.Taint` is source-stamped and a tainted-to-sink flow is sink-gated at adjudication time (rank-30, pre-call). |
| Kernel-authored trust/provenance | Shipped | A classifier takes authorship of trust away from the model, with a hardened sink classifier. |
| plan-CFI | Shipped | A plan control-flow-integrity adjudicator with a `RequireApproval` verdict; `internal/harvest` folds the verdict stream into a frozen label corpus. |
| Effect-verifying witness gate | Shipped | An in-process `dos_verify` effect-verify backs a `require-witness` verdict that fails closed when unwitnessed. |
| Dynamic attack battery | Shipped | `internal/agentdojo` is an ASR-gated AgentDojo-style red-team replacing the static poison fixture; 3 of 4 arrows shipped, the RL red-team generator is a documented seam. |
| `normgate` admission driver | Shipped | The rank-5 canonicalize-and-decode ResultAdmitter; lifts agent-evasion catch 0 to 20 of 24 and cuts private-transcript false positives with 0 new FPs and 0 leaks. |

## Gateway (`fak serve` / `fak guard`)

| Feature | Status | What it does |
|---|---|---|
| `fak serve` OpenAI-compatible surface | Shipped | An HTTP surface (`/v1/chat/completions` adjudication proxy, `/v1/fak/{syscall,adjudicate}`, `/v1/models`, `/healthz`) plus MCP over stdio/HTTP; mints a tainted agent-scoped Ref so the IFC/secret/self-modify rungs stay armed; optional constant-time bearer auth. |
| Served result-side stack (`fak_admit`) | Shipped | `POST /v1/fak/admit` runs a client-produced result through the context-MMU quarantine + IFC taint ledger, with a `TraceID` threaded end-to-end. |
| `fak guard -- <agent>` front door | Shipped | Starts the in-process gateway on a private loopback port, injects its URL into the child only, execs the real agent, and prints a verdict roll-up from the same counters `/metrics` exposes. Default upstream is the Anthropic API in passthrough. |

The gateway fronts any OpenAI-compatible upstream (a local engine or a cloud provider). The exact list of harnesses, wires, and backends that have been sourced is in the [compatibility matrix](../integrations/compatibility-matrix.md); the streaming and endpoint detail is in the [gateway API reference](../fak/api-reference.md) and the [MCP tool-result wire](../mcp-tool-result.md). One streaming caveat: `stream:true` SSE on the OpenAI proxy is synthesized from the finished, already-adjudicated turn (the Anthropic `/v1/messages` passthrough does live token streaming).

## Engine and live seam

| Feature | Status | What it does |
|---|---|---|
| OpenAI-compatible client | Shipped | A base-url-swappable `/v1/chat/completions` client with bounded timeout + backoff, cassette record/replay for offline runs, token-usage extraction, and a deterministic mock engine. |
| Live seam honesty | Shipped | A run carries a real transcript hash XOR the explicit RED flag `live_seam_unverified`, never a silent skip. |
| Live seam exercised end-to-end | Shipped | `fak agent` drove the kernel with a real OpenAI-compatible model (Gemini OpenAI-compat and local Qwen2.5), each run carrying a real `transcript_sha`. |
| Hardware-aware cache placement | Shipped | `internal/cachemeta` models CXL/NUMA-far tiers, per-tier TTL, a zero-copy share descriptor, and a cost-driven `PlanPlacement` that demotes a hot prefix instead of evicting it. The payload-free policy plane; the engine adapter performs the physical movement. |
| Capacity engine adapter (PlanPlacement demote/spill executor) | Shipped | `internal/engine.CapacityAdapter.Execute` turns a `PlanPlacement` demote/spill into a real stage-to-the-colder-tier + eviction from the live KV cache, fail-safe (stage before evict) and recorded as a typed cache event (#708). |
| metrics-service scrape / KV-residency / token-per-watt | Simulated | Labeled SIMULATED telemetry; there is no watt source on the build box. |
| Zero-copy KV co-residence with an external engine | Stub | The `Ref`/`Resolver`/`RegionBackend` seam is frozen so it is a backend swap later, but the shipped path is copy-CAS. The in-kernel model owns its own KV cache. |
| Advisory adjudication model (harvest-corpus consumer) | Shipped | A small fail-closed classifier (`internal/advmodel`, #580) trained over the floor-labeled `internal/harvest` corpus; it can only corroborate a deny, never weaken the floor. Held-out P/R/F1 vs the stock reference are committed in the artifact meta. |
| Fine-tuned syscall/adjudication LLM + AsyncLM | Stub | The advisory model above is a logistic-regression bag-of-tokens model, NOT a fine-tune of the fused SmolLM2 forward pass; the tuned LLM head (GPU + weights + hours) and AsyncLM's interrupt behavior remain unbuilt. |

## Turn-tax and policy-replay benchmarks

| Feature | Status | What it does |
|---|---|---|
| `fak turntax` | Shipped | Replays a class-labeled trace through the real kernel and prices the extra error-code model turn a SOTA loop fires vs fak's 1-shot adjudication; the fak side is live kernel events, a happy-path control saves exactly 0. |
| Policy-replay spine | Shipped | Scores K policies against one recorded trajectory as model-free kernel replays (product to sum), every arm carrying an `exact` / `bounded@i` divergence witness so a counterfactual resolve-rate is refused. |
| Divergence-witness hardening + histogram | Shipped | Captures a raw per-call monitor verdict so a redact-only policy diff comes out `bounded@i`, and emits the first-divergence distribution + exact-cell fraction. |
| Payload-bearing trajectory sink | Shipped | `internal/tracesink` registers as an emitter and writes a payload-bearing, IFC-labeled trace so a content-inspecting policy can re-adjudicate a recorded run; recorder overhead ~1.5 µs/call. |
| Per-kernel adjudicator-chain injection | Shipped | `RunPolicyReplay` fans its K arms across goroutines on fresh kernels instead of mutating the process-global policy, with results identical to the serial path. |
| Fleet counterfactual replay | Shipped | Re-adjudicates a recorded corpus against candidate policies at $0 model and reports per-policy floor coverage; resolve-rate reported only for exact cells. |
| OPE calibration past the divergence frontier | Shipped | A clearly-modeled off-policy resolve-rate estimate plus CI alongside the measured floor counters; IPS is explicitly refused for deterministic policies. |
| Replay-as-fitness policy search | Shipped | A deterministic, model-free ($0) hill-climb over the policy genome scored entirely by replay on honest replayable axes; resolve-rate is not a fitness term. |
| Lever-flip causal attribution | Shipped | Replays one trace through L kernels each with one rung ablated, producing an exact per-rung attribution table. |
| World-pluggable replay + token-ledger demo | Shipped | `RunWithWorld` replays the same machinery against a different tool world; `cmd/tokendemo` scores a model-context win and a tool-side win as honestly-distinct meters. |
| `fanbench` fan-out benchmark | Shipped | Sweeps one master goal to N sub-agents (N=1…1024) and reports from real kernel events the cross-agent tool-result dedup + shared-prefix KV-reuse geometry. |
| `fanbench` token-multiplier economics | Simulated | The prefix-cache `tax_clawed_back`, latency, throughput, and saturation knee are a transparent knobbed cost model priced at documented prompt-cache multiples, reported apart from the measured halves. |
| `longctxbench` ultra-long-context work floor | Shipped | Closed-form, contention-free token and FLOP floors for the >100k-token regime; eliminates ~10× vs naive re-prefill for one session, ~40×+ for a 5-agent fleet. |
| `longctxbench` live wall-clock validation | Simulated | The live wall-clock validation at >100k needs a model resident on a bench node and is not run on the build box; the floor arithmetic stands independently. |

## Stewards and the RSI ship-gate

| Feature | Status | What it does |
|---|---|---|
| Single-invariant stewards | Shipped | Secret-in-context, lease-disjointness, kpi-regression, and vdso-soundness stewards fire only with an independently-authored witness; a meta-steward prunes never-firing stewards. |
| RSI-as-ship-gate | Shipped | Keep-or-revert on a non-forgeable keep-bit (strict metric gain AND suite-green AND truth-clean), applied in an isolated git worktree, with an escalation breaker. |
| RSI closed loop | Shipped | `internal/rsiloop` derives every keep-bit witness from a real run it performs itself, re-measuring the baseline from `main` every run; it never mutates `main`. |
| Issue-dispatch closed loop | Shipped | The witness-gated GitHub-issue backlog driver: routes each open issue to its lane, passes the dispatch preflight gate, spawns one detached worker, and closes only via a per-SHA `dos commit-audit` re-run. |

## The honest ceiling fak surfaces

The result detector these drivers feed is approximately 100% evadable on a SOTA evasion battery and false-positive-prone on private real-transcript corpora. Detection is deliberately non-load-bearing. The load-bearing guarantee is the capability floor plus containment, which never run the detector. Improving detection is additive, not the moat. fak reports this ceiling rather than hiding it, and surfaces the same gate decision durably through `cdb` without claiming it improves the decision.

For the witnessed source of record behind every row on this page, read the [claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) and the [status posture](https://github.com/anthony-chaudhary/fak/blob/main/STATUS.md).

## Related: the supported-things pages

- [What fak supports (hub)](README.md) — the index of every "supported" page
- [Models](models.md) — in-kernel architectures + any model you front
- [Clouds & hosted providers](clouds.md) — Anthropic, OpenAI, Gemini, xAI, Bedrock, Vertex, Azure, OpenRouter, Together, Groq, Fireworks
- [APIs, wires & MCP](apis-and-protocols.md) — OpenAI Chat/Responses, Anthropic Messages, Gemini, xAI, MCP, fak-native endpoints
- [Agent harnesses & frameworks](agent-harnesses.md) — Claude Code, Cursor, Codex, Aider, Cline, Roo, LangChain, LlamaIndex, CrewAI, …
- [Serving engines](engines.md) — Ollama, vLLM, SGLang, llama.cpp, LM Studio, and the in-kernel reference engine

## Reference (the witnessed sources behind this page)

- [Compatibility matrix](../integrations/compatibility-matrix.md) — 44 sourced harnesses / frameworks / backends / protocols, each with the exact repoint key
- [Integration index](../integrations/README.md) — the "repoint one base URL" recipe and the 60-second offline proof
- [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — every capability with one machine-checked tag (shipped / simulated / stub)
- [Status](https://github.com/anthony-chaudhary/fak/blob/main/STATUS.md) · [CLI reference](../cli-reference.md) · [Hardware matrix](../HARDWARE-MATRIX.md) · [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt)
