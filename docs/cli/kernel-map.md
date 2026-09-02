---
title: "fak CLI reference — The kernel map — files, contract, and build history"
description: "The kernel file map, the one-breath contract, the witness-closed shipped table, honest limits, and wave-0 build history, split out of docs/cli-reference.md."
---

# The kernel map — files, contract, and build history

This is the kernel-level map that used to trail [the CLI reference](../cli-reference.md); kept verbatim so every pointer into it still lands.

## What's here

| File | What it is |
|---|---|
| `internal/abi/types.go` | The **frozen, additive-only** ABI spine: the syscall envelope, the discriminated-union Verdict, the addressable `Ref`, the async + provisional-lifecycle seams, and the core interfaces. No subsystem is named in it. |
| `internal/abi/registry.go` | The **extension mechanism**: `Register*` from a driver's `init()`, reserved number ranges for link-time disjointness, the driver interfaces (engine / region / page-out / witness / steward). |
| `ARCHITECTURE.md` | The extension model — how a new idea bakes in, the 4 seams that had to be frozen now, the bake-in walkthrough (speculative exec, async, zero-copy, the syscall-tuned model, an unforeseen idea). |
| `DIRECTION.md` | The **strongly-typed-core direction** — the request/enforcement path is Go (illegal states unrepresentable, the same thesis applied to the source); non-Go is permitted only at rare named *seams/interconnects* that sit off the path behind a typed, serialized boundary (the ML-ecosystem oracle, build/CI glue, out-of-band analysis). With the reviewer's three greps that prove it holds. |
| `DISAGGREGATED-AGENT-MEMORY.md` | The **strategy note** — fak as the MMU + reference monitor for shared agent memory: six memory semantics (S1–S6) mapped to shipped primitives, the cross-agent / cross-tenant / cross-node axes, and §2.5's four-layer distinction (routing vs addressing vs fusion vs semantics) that keeps a routing win from being mistold as a fak win. |
| `MEMORY-LAYERS-EXPLAINER.md` | The **four-layer explainer** (teaching artifact for the above §2.5): *routing* (where a cell lives), *addressing* (its name), *fusion* (zero-copy co-residence), *semantics* (coherent mutation / isolation / provenance / capability — fak's actual change), with an ASCII stack diagram, the Docker(defines the object) vs Kubernetes(routes the object) analogy, and the one-line "is this a routing claim or a fak claim" test. |
| `docs/SKILL-CONTEXT-MEMORY.md` | The **skill-context memory** note - treats `.claude/skills/` as the procedural twin of `.claude/memory/`: named, versioned, load-on-demand context capsules that can emit witnessed, cacheable `SkillContextRecord`s instead of replaying long context. |
| `POLICY.md` | The **deployable capability floor** — the dump→edit→check→load workflow, the `fak-policy/v1` manifest schema, the closed refusal vocabulary, and the honest scope of what the floor does and does not bound. The adopter's front door. |
| `PARTITION.md` | The `dos-plan-price` fleet partition: wave-0 gate, the 4 wave-1 leaves, the 3 wave-2 workers, the serial tail, and the **growth slots** for post-v0.1 ideas. |
| `WORKER-PACKET.md` | The per-worker dispatch packet the fleet consumes (goal · leased tree · `dos hook stop` witness · `dos verify` done-proof). |
| `LIVE-RESULTS.md` | The **live** turn-count A/B: a real model (Gemini OpenAI-compat + local Qwen2.5) drives the `fak agent` loop twice over one task; each run carries a transcript hash. The honest read of what fusion does and does not buy live. |
| `TICKETS.md` | Issues the live `fak agent` lane surfaced (FIXED / OPEN / NOTE). |
| `RECALL-RESULTS.md` | The **session-recall** lane: a quarantine that survives the session boundary (a finished session as a *core dump*; benign pages demand-paged byte-identical, sealed pages refused across the boundary + re-screened on page-in). Witness-grounded; 5/5 adversarially confirmed. |
| `CDB-RESULTS.md` | The **context-debugger** lane (`fak debug`): attach to a finished session as a core dump and answer a follow-up by *demand-paging only the working set* (Denning) — never replaying the address space. Ingests a REAL Claude Code transcript; measured on a 2.8 MB session, an 18 KB page table over a 1.2 MB swap device, follow-ups paging in ~1.8–6.2% of the resident image. |
| `MEMORY-DREAM-CLEANUP-RESULTS.md` | The **memory-dream cleanup** lane (`fak dream`): an offline sleep pass over a core image that re-screens resident pages, pre-seals refuted witnesses, repairs sealed descriptors from metadata only, surfaces duplicate aliases, and prunes unreferenced CAS bytes. |
| `IN-KERNEL-MODEL-RESULTS.md` | The **model fused into the kernel**: a pure-Go SmolLM2-135M forward pass (134.5M params / 272 tensors), every rung proven against HuggingFace (0/134.5M decode bit-mismatches, layer cos=1.000000, KV-decode + KV-quarantine-evict token-for-token identical). The kernel owns the KV cache; the design is in `IN-KERNEL-MODEL-DESIGN.md`. |
| `MODEL-BASELINE-RESULTS.md` | The fused forward pass **measured** against the next-best CPU baselines (HF transformers, llama.cpp): naive tax → parity lane (decodes *faster* than same-precision HF f32) → an int8/Q8 SIMD lane at near-parity with llama.cpp Q8_0 (~7.7 ms/tok, the in-flight Act 3). Every number recomputed from raw JSON, survived a 4-skeptic adversarial pass. |
| `MODEL-ARCH-SUPPORT-AUDIT-2026-06-18.md` | Current top-10 architecture support audit: what is witnessed today, what is only loader/shape-compatible, and which GitHub issues cover the remaining families. |
| `FAK-NATIVE-QWEN35-RESULTS.md` | The Qwen3.5/Qwen3.6 Gated-DeltaNet lane: 0.8B coherent f32 chat plus the 2026-06-19 pure-fak Qwen3.6-27B GGUF->Q8 smoke with first-token llama.cpp parity on the M3 Pro. |
| `KV-QUARANTINE-BRIDGE-RESULTS.md` | The deepest "model is secondary" rung: the byte-gate drives the **KV-gate** — a `Quarantine` verdict evicts the poison's K/V span, leaving the attention cache bit-identical (max\|Δ\|=0) to never-having-seen it. |
| `TURN-TAX-RESULTS.md` | The **turn-tax** benchmark (`fak turntax`): prices the extra error-code MODEL turn the 1-shot kernel deletes (forced vs elision, with a consistency guard and a happy-path=0 control), keeping the safety floor on its own axis. |
| `FLEET-SWEEP-RESULTS.md` | The 2-D **turns x agents** sweep: shared-cache fleet vs isolated agents, exact-zero no-share controls, and the scoped-invalidation eraser that fixes the write-rate crossover. |
| `FANOUT-BENCH-RESULTS.md` | The one-master-goal **fan-out** benchmark: N=1..1024 sub-agents, real cross-agent tool-result dedup, transparent prefix-cache economics, and the fold-bound latency knee. |
| `VISUALS-benchmarking-status-2026-06-18.md` | The refreshed benchmark visual/status dashboard tying the current plots, headline numbers, and caveats together. |
| `EXPLAINER-trust-floor-two-lenses-2026-06-17.md` | **fak explained twice** — once for *security researchers*, once for *agent-optimization*, with a Rosetta table mapping each primitive to both vocabularies. |

## The contract in one breath

`Kernel.Submit` adjudicates (folds the LSM-style `Adjudicator` chain by `FoldRank`)
and enqueues; `Reap` returns the typed completion; `Syscall` is the sync convenience
over the two. Payloads are addressable `Ref`s (zero-copy is a backend swap).
`Verdict` is a closed, trainable, discriminated union with an open registered range
(the syscall-tuned model's clean target). Speculation/transaction effects are
provisional until `Promote`/`Rollback`. Everything else — engines, vDSO tiers,
rungs, the MMU codec, stewards, KPIs, witnesses, and ideas nobody's had yet —
attaches through a registry.

## What shipped (witness-closed)

| Subsystem | Tree | What it does | Witness |
|---|---|---|---|
| in-process adjudicator | `internal/adjudicator` | DOS reference monitor: provable-deny / unprovable-defer, structured 12-reason refusal, bounded-disclosure SELF_MODIFY witness, redact-transform | `go test`, `BenchmarkDecide` |
| deployable policy | `internal/policy` | the capability floor as a declarative, version-tagged JSON manifest (`--policy FILE`); closed-vocab deny validation; fail-loud load; `--dump`↔`--check` round-trip; `fak policy` verb | `go test` (9, incl. `TestRoundTrip`) |
| compliance attestation | `cmd/fak/attest.go` | `fak attest --policy FILE` proves the floor from preflight: runs the real adjudication fold over a probe set (derived from the manifest, or `--probes FILE`) and emits a re-checkable attestation — every deny enforced with its cited reason, every allow admitted, default-deny holds; exit 0 PROVEN / 1 drift | `go test ./cmd/fak` (attest_test.go) |
| tool vDSO | `internal/vdso` | 3-tier local fast path (pure / content-cache / static), world-versioned, LRU, canonical keys | `go test` |
| engine | `internal/engine` | OpenAI-compatible client, base_url-swappable, cassette replay, usage extraction, mock | `go test` |
| pre-flight + grammar | `internal/preflight`,`internal/grammar` | rung ladder + JSON-schema; positional→named auto-repair; fail-open; grammar dedup; hard-negative harvest | `go test` |
| context-MMU | `internal/ctxmmu` | write-time quarantine (secret/injection/poison/repeat), page-out to <2KB pointer, witness-gated page-in | `go test` + `poison.json` |
| security substrate | `internal/ifc`,`provenance`,`plancfi`,`witness`,`canon`,`normgate`,`agentdojo`,`harvest` | the kernel stops believing the model: source-stamped taint + sink-gated flows (ifc), kernel-authored trust/provenance, plan-CFI (`RequireApproval`), an effect-verifying `dos_verify` witness gate, a normalize-and-rescan admission driver (normgate), and a dynamic ASR-gated attack battery (agentdojo) | `go test` (per pkg); `cmd/ctxbench -chain` |
| KV-quarantine bridge | `internal/kvmmu` | the same ctxmmu `Quarantine` verdict mechanically evicts the poison's K/V span, leaving the kernel-owned attention cache bit-identical (max\|Δ\|=0) to never-having-seen it | `go test` (5); `KV-QUARANTINE-BRIDGE-RESULTS.md` |
| session core-dump + debugger + dream cleanup | `internal/recall`,`internal/cdb` | persist a finished session as a page-table-over-CAS core image; `fak debug` attaches to it (incl. a REAL transcript) and demand-pages only the working set a question touches; agent/requester tombstones suppress unwanted memories from future context without deleting audit bytes; `fak dream` auto-cleans the sleeping image by re-screening, pre-sealing refuted witnesses, and pruning dead CAS bytes | `go test` + `recall-report.json`,`cdb-report.json`,`dream-report.json` |
 | in-kernel model | `internal/model` | a pure-Go forward pass the kernel owns (KV cache as a Go structure), with proven bit-for-bit correctness for SmolLM2-135M (Llama family) and first-token parity for Qwen3.6-27B (Gated-DeltaNet); an int8/Q8 SIMD lane at near-parity with llama.cpp Q8_0 is the active **in-flight** extension | `go test` (oracle argmax-exact); `MODEL-BASELINE-RESULTS.md`; `FAK-NATIVE-QWEN35-RESULTS.md` |
| gateway | `internal/gateway` | `fak serve`: OpenAI-compatible HTTP (`/v1/chat/completions` adjudication proxy, `/v1/fak/*`) + MCP over stdio/HTTP, so any-language agents route tool calls through the syscall boundary; mints a tainted agent-scoped `Ref` from raw bytes (IFC/secret/self-modify rungs stay armed) | `go test`; v0.2.1 adversarial-review hardening |
| model routing + account binding | `internal/modelroute` | `fak route`: per-aspect + ensemble model routing as a pure, deterministic policy — `Route(Subject)→Decision` (a tool call / sub-query / step routes to its own model or ensemble) + `Combine(reduction,votes)→Result` (`first`/`vote`/`best_of`/`all_reduce`/`concat`); version-tagged JSON manifest, `--dump`↔`--check`; `--accounts` / `--accounts-dump` / `--accounts-check` bind abstract route model ids to provider accounts, upstream model names, and residency-honest engine routes. The served gateway path dispatches single picks and ensembles with `--route-manifest`; standalone `fak agent` route-manifest wiring remains the labeled follow-on | `go test` (`internal/modelroute`, `cmd/fak` route tests, gateway route-manifest tests); `docs/model-routing.md`; `docs/model-accounts.md` |
| dispatch fusion | `internal/kernel` | one in-process chain; no `os/exec` on the hot path | `go test` (ABSENCE proof) |
| KPI + A/B bench | `internal/metrics`,`internal/bench` | vDSO ablation; the primary gate; provenance + identical-workload guard | `report.json`, `baseline.json` |
| turn-tax bench | `internal/turnbench` | `fak turntax`: prices the extra error-code MODEL turn (malformed/duplicate/poison) a SOTA loop fires vs the 1-shot kernel, per lever, safety floor on its own axis | `go test` (incl. happy-path=0 control); `TURN-TAX-RESULTS.md` |
| stewards + RSI gate | `internal/steward`,`internal/shipgate` | single-invariant stewards + meta-prune; keep-or-revert on a non-forgeable keep-bit, worktree isolation, escalation breaker | `go test` |
| version-everything | `internal/modver` | `fak version modules`: a per-module version report over the module tree — content-addressed rev + date (+ optional joined `-scores`), with `-only <prefix>`, `-sort name\|rev\|date`, `-top N`, and `-json` views; `-stamp` appends changed-module rows to the `fak-module-versions/1` ledger (`docs/nightrun/module-versions.jsonl`, seeded with 410 modules) so a module's version is git-witnessed, not asserted | `go test` (`internal/modver`, `cmd/fak/version_modules_test.go`); `docs/notes/VERSION-EVERYTHING-SPINE-2026-07-03.md` |

## What this is NOT (labeled, not hidden — see `CLAIMS.md`)

- **The in-kernel model is a reference forward pass; GPU throughput is real, but it is not
   yet a full production serving engine.** The kernel ships proven bit-for-bit correctness for
   SmolLM2-135M (Llama family) and first-token parity for Qwen3.6-27B (Gated-DeltaNet). The CUDA backend takes
  it onto the GPU and reaches **decode parity with llama.cpp Q8_0 (≈120 tok/s on an RTX 4070;
  `GPU.md` §3b)** — so GPU throughput is **not** out of scope. What *is* still future work is
  production *serving* beyond the native in-kernel lifecycle scheduler (paged attention,
  multi-tenant SLA scheduling); the
  live `fak agent` / `fak serve` lanes drive an external OpenAI-compatible engine for that
  today.
- **NOT** zero-copy KV co-residence with an external engine: that remains the
  addressable-`Ref` seam wired to a **copy** backend (a backend swap later, behind
  capability `zerocopy`). The in-kernel model owns *its own* KV cache; sharing one KV
  arena with a separate serving process is the unbuilt stub.
- **NOT** GPU-dependent for the shipped pure-Go binary: token-per-watt / metrics-service
  KV-residency are read-only **SIMULATED** telemetry (no watt source on the box).
- **NOT** a fine-tuned *syscall/adjudication* model: the typed `LabelRow`/`VerdictKind`
  training targets exist (and `internal/harvest` now folds the live verdict stream into a
  corpus of them), but the model that would emit Verdicts from them is not trained — the
  fused model is a stock reference, not a tuned adjudicator.
- The vDSO real-world hit-rate is low (~0.7% addressable on real tau2-airline) — the
  demo trace is deliberately cache-favorable, which is why the headline is the
  call-mix-independent adjudication gate, not the vDSO win.

Every claim in `CLAIMS.md` carries exactly one of `[SHIPPED]` / `[SIMULATED]` /
`[STUB]` (lint-enforced). See `../BUILD-72h-fused-agent-kernel.md` for the original
scope and `../PLAN-fak-mvp-100-units-2026-06-16.md` for the 100-unit plan.

## Build history (how wave 0 was landed)

The following work was completed as the initial wave-0 build:

1. Land + freeze wave 0 (this artifact): `go build ./... && go vet ./...`, commit the
    golden conformance test, author the operator fixtures, run the vDSO purity gate.
2. `dos-plan-price PARTITION.md` → confirm collision-free; `dos-arbitrate` the leases.
3. `dos-goal-fleet` the wave-1 packets; gate wave 2 on `dos-witness-claim`; fold; tag.

See [PARTITION.md](https://github.com/anthony-chaudhary/fak/blob/main/PARTITION.md) for the current partition manifest and wave plan.

