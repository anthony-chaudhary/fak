---
title: "fak product scorecard — which concepts are durable, real, useful-today products"
description: "Inward product-concept scorecard: each fak concept positioned on the durable / real / useful-today axes a person feels, with one honest verdict per concept and the most-critical areas to progress. Two driven numbers: coverage (of the concept catalog) and product-debt."
---

# Product scorecard — durable, real, useful-today

<!-- product-scorecard: 2026-06-24 @v0.30.0 · process: tools/product_scorecard.py · data: tools/product_scorecard.data/ -->

The sibling scorecards grade fak's internals (code, docs) and its competitive standing (industry). This one asks the question a *person* asks: **of the concepts fak ships, which can I actually pick up and use this afternoon — and which are still a named gap, a research seam, or an overclaim?** The source of truth is the concept catalog in [`CLAIMS.md`](../../CLAIMS.md); every number below is re-derived from `tools/product_scorecard.data/` by `tools/product_scorecard.py` and cross-checked against the real tree (the CLAIMS tag a concept carries, whether its first command resolves, whether its witness/entry paths exist). No verdict is hand-typed.

> Regenerate: `python tools/product_scorecard.py --markdown-dir docs/product-scorecard`.

## Headline

| Metric | Value |
|---|---|
| **Coverage** | **100.0%** (18/18 concept sections positioned) |
| **Product-debt** | **0** (honesty 0 + coverage 0) |
| Composite score | 100.0/100 (grade A) |
| Durable products | 10 of 35 concepts |
| As of | 2026-06-24 (fak v0.30.0) |

> **Read this right.** The score grades how *complete and honest the product map is* — not how much fak wins. A concept that is an honest `real-not-easy` subsystem or a labeled `honest-stub` is not a defect; an *overclaimed* verdict is.

## The verdict ladder

| Verdict | Means |
|---|---|
| ★ durable-product | shipped + an OFFLINE first command (no GPU/key) + a witness that exists + an entry doc that exists — use it today on a laptop |
| ● usable-today | shipped + a first command, but it needs a GPU / key / network |
| ◐ real-not-easy | shipped/real, but no copy-pasteable command (a subsystem, not a surface) |
| ○ honest-stub | a STUB / SIMULATED seam, labeled honestly |
| · concept-only | a roadmap idea, not built |

## The product concepts (best verdict first)

| | Verdict | Maturity | Category | Use today? | Concept — what you get |
|---|---|---|---|---|---|
| ★ | durable-product | shipped | memory | laptop | **Context debugger (cdb) over a real session transcript** — Point it at a real Claude Code transcript and inspect it like a core dump — Info/Backtrace/Examine/WorkingSet/Grep over the pages, driven through the same shipped admission gate. |
| ★ | durable-product | shipped | platform | laptop | **One static Go binary (the whole governed surface)** — A single dependency-free Go binary (no Python, no CUDA, no go.sum) that you `go build` and run — the gateway, policy gate, quarantine, audit log, and metrics all in one process. |
| ★ | durable-product | shipped | platform | laptop | **MCP server (fak serve --stdio)** — Expose the kernel's verdict as MCP tools (fak_adjudicate, fak_admit, ...) over stdio so Claude Code or any MCP client can ask the kernel before it runs a tool — offline, no upstream model needed. |
| ★ | durable-product | shipped | security | laptop | **Default-deny capability floor** — A reviewable JSON policy that declares which tools your agent may call; everything else is refused by structure, not by a model guess. |
| ★ | durable-product | shipped | security | laptop | **In-process adjudicator (the DOS reference monitor)** — A one-shot oracle that tells you ALLOW/DENY for any tool call against a policy, with a structured refusal reason from a closed vocabulary — no model in the loop. |
| ★ | durable-product | shipped | security | laptop | **Write-time result quarantine (context-MMU)** — Run an agent turn and watch a poisoned tool result get held out of the model's context — the injection never reaches the model, and the task still completes. |
| ★ | durable-product | shipped | tooling | laptop | **Pre-flight ladder (static parse + schema validation)** — A cheap, deterministic check that refuses a malformed tool call at the boundary — static parse first, then JSON-Schema validation, cheapest-first, before anything fires. |
| ★ | durable-product | shipped | tooling | laptop | **Answer-shape degeneration/verbosity witness** — A pipeline gate that grades the SHAPE of any text (repetition, repeated-line blocks, short-period tiling) against tunable thresholds and exits 1 when it is degenerate — catch a looping model in CI. |
| ★ | durable-product | shipped | tooling | laptop | **Doctor (operator diagnostic over answer-shape + kernel admit)** — Pipe text in and get operator recommendations plus the real kernel admit verdict on the same bytes — the fak analogue of `dos doctor`, an off-path read-only health check. |
| ★ | durable-product | shipped | tooling | laptop | **Codelint (language packs over agent-written code)** — Lint code your agent produced for HARD parse/compile errors — Go/JSON in-process, Python/CUDA via their toolchains — and exit 1 on a real error, the write-time code check at the kernel boundary. |
| ● | usable-today | shipped | model | laptop | **Persistent per-session context planner (O(1) resident view)** — Replay the heaviest real sessions through the planner and watch resident tokens stay ~13x below linear with every back-reference served as a recoverable page fault — measured, not modeled. |
| ● | usable-today | shipped | performance | laptop | **Syscall adjudication latency sentinel (fak bench)** — Run a frozen trace through the binary and watch in-process adjudication (~µs) vs a spawned-hook baseline (~ms) — a regression sentinel that the decide path never pays a per-call process boundary. |
| ● | usable-today | shipped | performance | laptop | **Turn-tax benchmark (the round-trip that never fires)** — Replay a real tool-call trace and count the extra error-recovery model round-trips a naive/tuned loop is forced into that the kernel resolves in the same syscall — with the safety floor on a separate axis. |
| ● | usable-today | shipped | performance | laptop | **Fan-out benchmark (fanbench, N=1..1024 sub-agents)** — Sweep one master goal into N sub-agents and measure the cross-agent tool-result dedup and shared-prefix KV reuse the fan-out buys — the regime no public multi-agent benchmark maps (they top out at 5-7). |
| ● | usable-today | shipped | performance | laptop | **Ultra-long-context work floor (longctxbench, >100k tokens)** — Compute the exact, contention-free reread-elimination work floor for the >100k-token regime as closed-form arithmetic from the session shape — no GPU, no live model required. |
| ● | usable-today | shipped | platform | needs gpu/key | **Governed gateway (fak serve)** — Front any OpenAI-compatible model server with one command; every tool call the model proposes is denied / repaired / quarantined before your agent runs it, with /metrics and /healthz. |
| ● | usable-today | shipped | platform | needs gpu/key | **Claude Code passthrough (fak guard -- claude)** — One command wraps a real Claude Code session with the capability floor armed — your own key and prompt-cache breakpoints pass through untouched, and you get a verdict roll-up on exit. |
| ● | usable-today | shipped | platform | laptop | **RSI ship-gate (stewards + keep-or-revert loop)** — A runnable loop that evaluates a code candidate in an isolated worktree, measures a KPI, runs the suite, and KEEPS it only on a non-forgeable keep-bit (metric gain AND suite-green AND truth-clean). |
| ◐ | real-not-easy | shipped | memory | — | **Session core-dump (recall) — durable quarantine across the process boundary** — A finished session persists as a durable core image (a page table over a content-addressed swap device), reloadable in a fresh process where a sealed page stays sealed — quarantine that survives a restart. |
| ◐ | real-not-easy | shipped | memory | — | **S7 write-time durability gate (context is not memory)** — A cheap lexical classifier tags a benign result turn/session/durable, and a promotion gate keeps only durable-classed facts in the persisted image — an ephemeral observation no longer silently becomes a persistent bias. |
| ◐ | real-not-easy | shipped | model | — | **In-kernel model (oracle-exact forward pass + kernel-owned KV)** — A pure-Go model whose KV cache is a kernel-owned Go structure, every rung proven bit-for-bit against a HuggingFace oracle — so the kernel can evict a poisoned span at the KV level. |
| ◐ | real-not-easy | shipped | model | — | **KV-quarantine bridge (quarantine evicts the K/V span)** — When a tool result is quarantined, the kernel mechanically evicts that result's K/V span, leaving the attention cache bit-identical to never having seen the poison. |
| ◐ | real-not-easy | shipped | model | — | **Parity lane (parallel matmul + batched prefill, bit-identical)** — A parallelized matmul and batched-prefill GEMM whose every output is bit-identical to the serial reference, so the model gets faster with no proven-correctness rung disturbed. |
| ◐ | real-not-easy | shipped | model | — | **Planned-elision -> KV-eviction residency bridge** — When the context planner elides a span, the kernel-owned KV residency shrinks to the planner's O(1) resident view byte-for-byte — an O(1) view becomes an O(1) KV footprint. |
| ◐ | real-not-easy | shipped | model | — | **RadixAttention parity (prefix-tree KV reuse + policy eviction)** — A radix-tree KV cache over the kernel-owned cache that reaches a 77-88% hit rate (inside SGLang's published band) with bit-identical reuse-through-an-edge-split and policy-driven eviction. |
| ◐ | real-not-easy | shipped | model | — | **Cross-engine zero-copy KV co-residence seam** — A frozen ABI seam that lets the kernel Evict/Clone an EXTERNAL engine's KV (vLLM/SGLang) zero-copy, so the per-agent quarantine holds against an engine fak does not itself run. |
| ◐ | real-not-easy | shipped | model | — | **GPU backends (AMD Vulkan / NVIDIA CUDA / Apple Metal)** — The in-kernel forward pass runs on real GPUs argmax-exact (cosine 1.0 vs cpu-ref); the NVIDIA CUDA-graph path even hits decode-speed parity with llama.cpp Q8_0 on a model that fits. |
| ◐ | real-not-easy | shipped | performance | — | **Tool vDSO (3-tier local fast path)** — A read-only/idempotent tool call can be served from a content-addressed local cache instead of re-executing, with automatic invalidation when the world changes. |
| ◐ | real-not-easy | shipped | performance | — | **Shared-prefix KV reuse (prefill once, clone bit-identically)** — The kernel prefills a shared prompt prefix once and clones the KV bit-identically into every agent, so a read-heavy fleet does the shared setup work one time, not per agent per turn. |
| ◐ | real-not-easy | shipped | performance | — | **OpenAI-compatible engine client (record/replay + mock)** — A base-url-swappable OpenAI-compatible client with bounded timeout/backoff, cassette record/replay for deterministic offline runs, and a deterministic mock engine. |
| ◐ | real-not-easy | shipped | security | — | **Normalize-and-rescan admission driver** — A deterministic, model-free screen in front of the context-MMU that strips unicode tricks and decodes base64/hex so an attacker cannot smuggle a secret past the regex floor by re-encoding it. |
| ◐ | real-not-easy | shipped | security | — | **Information-flow control + plan-CFI + effect-verifying witness gate** — The kernel stops believing the model: tainted data is sink-gated, a plan-CFI rung can require approval, and a 'ship' claim is refused unless corroborated by git evidence the agent did not author. |
| ◐ | real-not-easy | shipped | security | — | **Local-model-on-the-wire screen (semantic + PII redaction + phash dedup)** — An additive, default-inert screen consulted after the regex floor that can quarantine a semantic injection, redact PII before bytes leave the box (with byte-exact restore), and collapse duplicate screenshots. |
| ◐ | real-not-easy | shipped | tooling | — | **Grammar rung (positional->named auto-repair, in-syscall)** — An in-process, model-free transform that repairs an arity-matched positional tool call into a named one without a model turn; unrepairable calls are denied with MISROUTE. |
| ◐ | real-not-easy | shipped | tooling | — | **Write-scoped codelint verdict in the adjudicator** — An opt-in policy knob that refuses a whole-file write of unparseable Go/JSON with Deny(MALFORMED) + a bounded file:line:col witness before it lands — the in-kernel dual of the fleet's advisory lint. |

## Per-KPI (product-debt = honesty/quality of the rows that exist)

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| well-formed | `well_formed` | 100 | 0 | all 35 rows well-formed |
| honesty | `claim_honest` | 100 | 0 | every claimed maturity matches CLAIMS.md (0 unmatched section) |
| honesty | `verdict_consistency` | 100 | 0 | every verdict matches its evidence |
| usefulness | `command_resolves` | 100 | 0 | every first command resolves to a real cmd dir + documented verb |
| durability | `witnessed` | 100 | 0 | every shipped/simulated concept is witnessed by a real path |
| durability | `discoverable` | 100 | 0 | every usable concept has a real entry doc |

