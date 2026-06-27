---
title: "fak CLI Reference — Verbs of the Agent Kernel"
description: "The fak CLI reference: every verb (serve, run, preflight, bench, turntax, agent, recall, dream, debug, policy, hook) for the in-process agent kernel."
---

# fak — the agent kernel

`fak` is the runnable implementation of the Fused Agent Kernel: one Go binary
that puts a policy and quarantine boundary between an AI agent and its tools — with
zero external dependencies, so the gateway, the capability gate, the quarantine, the
audit log, and the metrics all live in that single process (no Python/CUDA stack, no
sidecars). The practical use is gateway-first: keep your existing model or agent host,
front it with `fak serve`, and make tool calls cross a reviewable capability floor
before anything dangerous executes.

Under the hood, every tool call becomes a syscall-like request: adjudicated
in-process, served from a local **tool vDSO** when possible, screened by a
**pre-flight + grammar ladder** before it fires, and admitted through a
**context-MMU** before tool results enter model context.

## Use fak with your coding agent (Claude Code, Cursor, …)

If you drive a coding agent, fak fits in two ways:

- **In front of the model** — point your agent's `ANTHROPIC_BASE_URL` (or OpenAI
  base URL) at `fak serve`, and every tool call the model proposes is
  denied / repaired / quarantined by the kernel before your agent runs it. No
  agent-side changes; the Anthropic `/v1/messages` and OpenAI
  `/v1/chat/completions` wires are both adjudicated, and a dropped or repaired
  call comes back with an in-band `[fak]` note so the agent adapts instead of
  looping. Witnessed live on macOS + Windows with the real Claude Code CLI:
  [`DOGFOOD-CLAUDE.md`](../DOGFOOD-CLAUDE.md) (one command — `scripts/dogfood-claude.sh`,
  or `scripts/dogfood-claude.ps1` on Windows).
- **As an MCP server** — `fak serve --stdio` exposes the kernel's verbs
  (`fak_adjudicate`, `fak_syscall`, `fak_admit`, …) as MCP tools, so your agent
  can ask the kernel for a verdict before it runs a tool, or screen a result it
  executed itself through the exfil floor. Copy-paste config + the tool catalog:
  [`examples/mcp/`](../examples/mcp/) (drop [`examples/mcp/.mcp.json`](../examples/mcp/.mcp.json)
  in your project root for Claude Code).

Both put the **same reviewable capability floor** (`--policy floor.json`) on every
tool call. New here? The fastest "feel it" is the no-credential boundary below;
the fastest "use it for real" is the dogfood path above.

## Try the boundary first

Run the no-credential proof before reading the architecture:

```bash
go run ./cmd/fak policy --check examples/customer-support-readonly-policy.json
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
go run ./cmd/fak agent --offline
```

This shows the adoption posture in one screen: a dangerous support action is
denied, a benign search is allowed, and the offline injection A/B blocks the
poisoned instruction while the task still completes.

## Try the live gateway

The boundary above is offline. To put the gate **in front of a real model** over
OpenAI-compatible HTTP, run `fak serve`: it fronts any OpenAI-compatible upstream
(Ollama / vLLM / llama.cpp / a cloud provider), and on every `/v1/chat/completions`
it denies / repairs / quarantines the tool calls the model proposes before returning
the survivors.

```bash
ollama serve &                         # any OpenAI-compatible server on :11434
ollama pull qwen2.5:1.5b
go run ./cmd/fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
# from a second terminal:
curl -s http://127.0.0.1:8080/healthz                       # liveness
curl -s http://127.0.0.1:8080/metrics | head                # Prometheus scrape
```

Point any OpenAI client at `http://127.0.0.1:8080/v1`. `fak serve` also speaks the
Anthropic `/v1/messages` wire, exposes `/metrics` + `/debug/vars`, and takes
`--policy floor.json` + `--require-key-env VAR` to harden it. One caveat worth
knowing up front: `stream:true` SSE is **synthesized from the finished,
already-adjudicated turn** — well-formed, but not true token-by-token streaming. (The
client's `max_tokens`/`temperature`/`top_p`/`stop` are now forwarded per request, so
long completions are no longer truncated.) Full walkthrough (Tiers 0–2):
[`GETTING-STARTED.md`](../GETTING-STARTED.md).

> **Status: shipped & benchmarked** (current release `v0.30.0` — single source of truth is
> the root [`VERSION`](../VERSION) file). `go build`/`go vet`/`go test ./...` green across the
> internal packages, CI green, and the A/B benchmark gate passes. Confirmed not by self-report
> but by the DOS truth syscall (`dos_verify`): the shipped line runs from **`v0.1.0`** (the
> syscall skeleton, sha `c72ddf1`) through **`v0.2.0`** (model fusion + security substrate +
> gateway) and **`v0.2.1`** (gateway hardening + the turn-tax bench). Note: `dos verify` on
> the v0.2.0/v0.2.1 release-bump tags returns `shipped:false` because those commits touch only
> `VERSION` + release notes; the actual code ships across the commits each tag caps (see
> `../STATUS.md` §1 for the caveat). The truth syscall confirms the *code* ships, not the
> version bump — see `../docs/releases/`.
>
> **What v0.2 added on top of the v0.1 syscall skeleton** (each its own witnessed lane):
> a **real model fused into the kernel** (a pure-Go SmolLM2-135M forward pass, every rung
> proven bit-for-bit vs HuggingFace, then made parity-fast — `MODEL-BASELINE-RESULTS.md`);
> a **kernel-authored security substrate** (information-flow control, a trust/provenance
> classifier, plan-CFI, an effect-verifying witness gate, and a normalize-and-rescan
> admission driver — the kernel stops believing the model's self-reports); a **gateway**
> (`fak serve`) fronting the syscall boundary over OpenAI-compatible HTTP + MCP; a durable
> **session core-dump + context debugger + dream cleanup** (`recall` / `fak debug` /
> `fak dream` — a quarantine that survives the process boundary, plus an offline
> re-screen/prune pass, `RECALL-RESULTS.md` / `CDB-RESULTS.md` /
> `MEMORY-DREAM-CLEANUP-RESULTS.md`); and a dedicated
> **turn-tax benchmark** (`fak turntax`) pricing the extra error-code model turn the
> 1-shot kernel deletes. The `fak agent` verb still drives the **live** turn-count A/B
> (real Gemini + local Qwen2.5, transcript-hashed) — see `LIVE-RESULTS.md`.

## Syscall subsystem check — useful, not the headline KPI (call-mix-independent)

`fak bench --suite tau2-smoke` replays a frozen tool-call trace through the one
binary and times the **in-process tool-call adjudication boundary** against a
**spawned-hook baseline** measured on the same machine (the same `Fold` decide, two
transports — apples-to-apples):

```
in-process adjudication p50 : 2,427 ns
spawned-hook        p50     : 6.913 ms  (process-per-decide, this machine, n=100)
SUBSYSTEM CHECK (gate_primary): pass   (~2,849x fusion speedup)
```

This is a **subsystem regression sentinel**, deliberately **not** the headline
product KPI: it is independent of the tool mix — it times the adjudication fold,
not the workload, so a low vDSO hit-rate can never turn it red — but it does not
prove production readiness, model quality, serving throughput, or the 45x fleet
claim. The vDSO hit-rate and token savings are reported as **soft UPSIDE
secondaries**, never the gate. See `STATUS.md` §2 and `CLAIMS.md` (unit 82).

## Verbs

```
fak run       --trace testdata/tau2/tau2-smoke.json    # replay a trace through the kernel
fak preflight --tool create_user --args '{"_positional":["alice"]}'   # rung-only check
fak bench     --suite tau2-smoke --out report.json     # A/B vDSO ablation -> report.json (the ns gate)
fak ablate    --sweep vdso                             # N-arm self-ablation: one frozen trace, feature on/off, deltas off the kernel counters
fak turntax   --suite turntax-airline                  # price the extra error-code MODEL turn the 1-shot kernel deletes
fak agent     --offline | --base-url URL --model M --api-key-env VAR  # LIVE turn-count A/B (see LIVE-RESULTS.md)
fak session   ls | status <id> | stop|pause|resume|throttle <id> | budget <id> [--turns N] [--addr URL]   # operator control of a served session's live drive state, over /v1/fak/session(s)
fak task      sample [--json] [--done N --total N]     # process-local task-manager snapshot: hardware/runtime sample + task/step/concept progress and ETA
fak console agent --account claude-seat --dry-run -- -p "task"  # native launch-plan for real Claude Code through fak guard, using a selected Claude config home
fak snapshot  kinds | demo | info | dump-fleet | restore-fleet   # dump/restore any primitive (turn|tool|session|fleet|RSI loop) to a portable sha256-integrity bundle
fak serve     --addr :8080 [--require-key-env VAR]     # OpenAI-compatible HTTP + MCP gateway (any-language agents)
fak recall    --dir DIR                                # persist/inspect a finished session as a durable core image
fak dream     --dir DIR --out-dir DIR                   # offline cleanup pass over a sleeping core image
fak debug     --session DIR --cmd report|info|bt|x|ws|grep|tombstone|context-query|context-diff   # attach to a session core image; demand-page its working set
fak answer-shape --text - --max-repeat 0.5 [--max-chars N]   # degeneration/verbosity witness over a text; exit 1 when it loops/runs away
fak doctor    --text - [--max-repeat 0.5] [--max-chars N]   # run the answer-shape witness + the kernel admit cross-check, then recommend
fak codelint  PATH...                                  # lint agent-written code (Go/JSON in-process, Python/CUDA via toolchain); exit 1 on a hard parse/compile error
fak policy    --dump | --check FILE                        # author/validate the deployable capability floor
fak route     --aspect tool_call --tool refund_payment [--manifest FILE] [--simulate "a,b,b"]   # which model/ensemble routes this aspect; --dump/--check author the routing manifest
fak routebench [--corpus FILE] [--routed F] [--single F] [--json]            # offline routing benchmark: per-aspect+ensemble vs single-model on cost/latency/quality (no model in the loop)
fak vcache    status | prove | prove-telemetry           # virtual provider-cache status plus planned/observed token-savings proof/refutation
fak callavoid prove-memo | account [--in FILE] [--json] [--gate]   # avoided-call economics: break-even memo proof + per-window amplification scorecard (JSON in/out)
fak cadence   [--json] [--check] [--append-history] [--window N]   # consolidated regular-cadence report: folds scores + work-done + releases into one control-pane envelope, with a durable trend ledger (docs/cadence/history.jsonl)
fak leaseref  live [--dir DIR] | list [--json] [--dir DIR] | reap [--dir DIR]   # cross-machine lease visibility: read refs/fak/locks/* into the dos_arbitrate live_leases shape (#825)
fak attest    --policy FILE [--probes FILE] [--json]        # compliance attestation: prove the capability floor from preflight (exit 0 PROVEN / 1 drift / 2 usage)
fak stopfailure plan | reset-stale [--apply]                # inspect and settle stale .dos/stop-failures breaker markers
fak hook      < call.json                              # spawned-hook decide (the A/B baseline)
```

`answer-shape` is the consumer-facing, GRADED dual of the context-MMU's write-time
repeat-admit rung: the kernel quarantines only the most blatant byte-repeat
pollution, while `fak answer-shape` judges the SHAPE of any candidate answer —
word-n-gram repeat, repeated-line blocks, and short-period tiling, headlined as one
`repeat` fraction — against thresholds you pick, off the hot path. It reads stdin on
`-` (or with no source), exits `1` when degenerate, so it gates a pipeline. `fak
doctor` wraps that witness into operator recommendations and additionally reports the
real verdict the context-MMU would reach on the same bytes (`ctxmmu.ScreenBytes`) —
the fak analogue of `dos doctor`.

`fak codelint PATH...` is the write-/definition-time CODE check at the kernel
boundary — the code-content dual of `fak preflight`'s tool-registry check. It routes
each file (or every file under a directory) to the owning language pack: Go and JSON
parse in-process via the stdlib, Python and CUDA shell out to their toolchains
(degrading to no-opinion when the toolchain is absent). It reports only HARD
parse/compile errors (the zero-false-positive tier — semantic checks are out of
scope) and exits `1` so it gates a pipeline. Because the input is untrusted model
output, it honors no in-content ignore comment, and it runs off the hot path.

`fak callavoid` is the operator-facing surface over the avoided-call economics leaf —
no Go required. Both subcommands are JSON-first (read input from stdin or `--in FILE`,
emit JSON), and the arithmetic is `internal/callavoid`'s, verbatim and deterministic:

```bash
# is memoizing this exact pure call net-positive? (k accesses, validate/mutation/capture costs)
echo '{"accesses":20,"validate_cost":0.02,"mutation_rate":0.05,"capture_cost":0.1}' \
  | fak callavoid prove-memo            # -> {"status":"PROVEN","decision":"memoize",...}

# how much amplification did a window of work get? (a Tally of the kernel's counters)
echo '{"execute":4,"memo_hit":6}' \
  | fak callavoid account               # -> {"status":"amplifying","grade":"B","amplification":2.46,...}

# gate a pipeline: exit 1 when avoidance was a NET LOSS this window
echo '{"stale_miss":5}' | fak callavoid account --gate    # exit 1 (regressing)
```

It exits `0` on a valid decision, `2` on malformed input (an unknown field or non-JSON,
caught loudly — never a silent zero-value decision), and `1` only under `--gate` on a
regressing window. Field names are the snake_case struct tags shown above.

`fak leaseref` is the operator-facing READ side of the cross-machine lease
substrate (`#825`). `internal/leaseref` persists a lease record (tree globs,
holder, acquire time, TTL) under the `refs/fak/locks/<id>` ref namespace, so the
lease rides ordinary `git fetch` / `git push` between clones — the same mechanism
`grite` uses with `refs/grite/locks`. This verb projects that ref store into an
admission decision:

```bash
# make a peer's lease (held on another machine) visible locally, then feed an arbiter
git fetch origin 'refs/fak/locks/*:refs/fak/locks/*'
dos arbitrate --lane docs --tree 'docs/**' --leases "$(fak leaseref live)"

fak leaseref list           # every record under refs/fak/locks/*, marked LIVE / EXPIRED
fak leaseref reap           # delete the expired (reapable) records — a crashed holder is bounded
```

`live` emits the **non-expired** records as the `dos_arbitrate` `live_leases`
array `[{lane,lane_kind,tree}, …]` (each ref-stored lease is a tree-scoped
`cluster` lane), so an arbiter on machine B can *see* a lease machine A pushed.
The write side is `internal/safecommit` (opt-in `FAK_LEASEREF=1`), which publishes
its commit lock here alongside the same-host `flock`. The honest boundary, kept in
the code and these docs: this is **distribution / visibility, not atomic
acquisition** — it lets the arbiter *see* a cross-machine conflict, it does not
arbitrate a same-fetch-window race; a signature envelope over the record is
deferred follow-up. Exits `0` ok, `2` on a usage/parse error, `1` on a git/store
failure.

`run`, `preflight`, and `agent` take `--policy FILE` to load the capability floor
from a declarative JSON **manifest** instead of the compiled-in default — so WHICH
tools the agent may call is a reviewable file, not a Go edit. See `POLICY.md`.

`fak route` is the same idea applied to model selection: which MODEL — or which
ENSEMBLE of models + a reduction — serves a given **aspect** of a request. Where a
SOTA router picks one model for the whole request, fak routes at every level — a
single tool call, a sub-query, a reasoning step — each to a different model, with
first-class ensembles (`vote` / `best_of` / `all_reduce` / `concat`), all from one
reviewable JSON manifest (`--dump` → edit → `--check` → `--manifest`). The routing
**decision** + the ensemble **reduce** are shipped and pure (witnessed by
`go test`); executing a decision on live engines is the wiring tracked in the
model-routing epic. `--simulate` folds stand-in member outputs through the chosen
plan's reduction so the ensemble half runs end to end with no model in the loop. See
[`docs/model-routing.md`](model-routing.md).

`fak routebench` is the offline measuring instrument: it runs a corpus of recorded
cases through TWO manifests — a per-aspect + ensemble policy vs a single-model
baseline (the SOTA shape) — and prints the delta on **cost / latency / quality**.
Each case carries the stand-in OUTPUT every candidate model produces (like `fak
route --simulate`), so it reuses the pure `Route` + `Combine` halves and is
deterministic end to end — no key, no GPU. Default (no args): the built-in 8-case
demo corpus + `DefaultManifest` vs a one-frontier-model baseline; `--corpus` /
`--routed` / `--single` load your own, `--dump-corpus` emits the starter corpus to
edit. Every figure is a ROUGH lens, never a bill or a measured SLA. See
[`docs/model-routing.md`](model-routing.md#the-offline-routing-benchmark-fak-routebench).

`fak vcache status|prove|prove-telemetry` is the proof surface for the virtual
provider-cache work. `status` reports the honest current state: the M5 Governor is up
as a local, off-path policy engine, while provider calibration/warming/recall remain
open in #716-#718 and the Codex/OpenAI cached-token probe is proven by #727. `prove` runs
the deterministic star-anchor savings arithmetic without a provider or model; the
default Codex-like workload (4096-token anchor, 7 sibling requests, 10-token suffixes,
0.1 read / 1.25 write multipliers) proves 21,094.4 token-equivalents saved, 73.4%,
and exits 1 for refuted workloads such as an anchor below the provider minimum.
`prove-telemetry --file experiments/agent-live/vcache-claude-prefix-probe-2026-06-25.jsonl`
replays observed provider counters and proves 13,141.5 token-equivalents saved
(4.73%) on the four-turn Claude Code prefix probe, with the first positive request at
4; the same verifier refutes the first three turns because the cache reads have not
repaid the 1h cache-write cost yet. The same JSONL reader accepts raw OpenAI
Responses usage (`usage.input_tokens_details.cached_tokens`), Chat Completions
usage (`usage.prompt_tokens_details.cached_tokens`), Codex CLI `token_count` rows,
and `codex exec --json` `turn.completed` usage rows. The replayable Codex artifact
at `experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl` proves
9,147,340.8 token-equivalents saved (85.98%) over 68 token-count events. `status`
reports the verifier as ready, includes a cached-token sample proof and zero-cache
refutation, and keeps the raw OpenAI API probe as an optional no-credential skip path.
These are cost proofs only: correctness never depends on a provider cache hit.

`scripts/ci.ps1` (or `make ci`) runs build + vet + test + the CLAIMS lint as one gate.

> It is *designed for extension*: other ideas bake in as a new package + one
> registration, never a core edit (see `ARCHITECTURE.md`).

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
| model routing (decision) | `internal/modelroute` | `fak route`: per-aspect + ensemble model routing as a pure, deterministic policy — `Route(Subject)→Decision` (a tool call / sub-query / step routes to its own model or ensemble) + `Combine(reduction,votes)→Result` (`first`/`vote`/`best_of`/`all_reduce`/`concat`); version-tagged JSON manifest, `--dump`↔`--check`. Live dispatch is the wiring tracked in the model-routing epic | `go test` (`internal/modelroute`, `cmd/fak` route tests); `docs/model-routing.md` |
| dispatch fusion | `internal/kernel` | one in-process chain; no `os/exec` on the hot path | `go test` (ABSENCE proof) |
| KPI + A/B bench | `internal/metrics`,`internal/bench` | vDSO ablation; the primary gate; provenance + identical-workload guard | `report.json`, `baseline.json` |
| turn-tax bench | `internal/turnbench` | `fak turntax`: prices the extra error-code MODEL turn (malformed/duplicate/poison) a SOTA loop fires vs the 1-shot kernel, per lever, safety floor on its own axis | `go test` (incl. happy-path=0 control); `TURN-TAX-RESULTS.md` |
| stewards + RSI gate | `internal/steward`,`internal/shipgate` | single-invariant stewards + meta-prune; keep-or-revert on a non-forgeable keep-bit, worktree isolation, escalation breaker | `go test` |

## What this is NOT (labeled, not hidden — see `CLAIMS.md`)

- **The in-kernel model is a reference forward pass; GPU throughput is real, but it is not
   yet a full production serving engine.** The kernel ships proven bit-for-bit correctness for
   SmolLM2-135M (Llama family) and first-token parity for Qwen3.6-27B (Gated-DeltaNet). The CUDA backend takes
  it onto the GPU and reaches **decode parity with llama.cpp Q8_0 (≈120 tok/s on an RTX 4070;
  `GPU.md` §3b)** — so GPU throughput is **not** out of scope. What *is* still future work is
  production *serving* (continuous batching, paged attention, multi-tenant scheduling); the
  live `fak agent` / `fak serve` lanes drive an external OpenAI-compatible engine for that
  today.
- **NOT** zero-copy KV co-residence with an external engine: that remains the
  addressable-`Ref` seam wired to a **copy** backend (a backend swap later, behind
  capability `zerocopy`). The in-kernel model owns *its own* KV cache; sharing one KV
  arena with a separate serving process is the unbuilt stub.
- **NOT** GPU-dependent: continuous-batching / token-per-watt / metrics-service
  KV-residency are read-only **SIMULATED** telemetry (no GPU on the box).
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

See [PARTITION.md](../PARTITION.md) for the current partition manifest and wave plan.

License: Apache-2.0 (matches the Microsoft Agent Governance Toolkit dep).
