---
title: "fak learning path — L600"
description: "A staged part of the fak learning path, split out of LEARNING-PATH.md so each stage stays a bounded read."
---

# L600 — Mastery: benchmarks, honesty discipline, and extending the kernel

**Stage 6 of the path** · prev: [L500 — Serving, Integration, and the In-Kernel Model](serving-integration.md) · next: [The shipped-surface appendix](appendix-shipped-surface.md) · back to the [overview and L100–L200](../../LEARNING-PATH.md)

**Read:** [`docs/proofs/gateway.md`](../proofs/gateway.md)

**Lab:**
```bash
go test -run 'Verdict|Adjud|HTTPSyscall|DefaultDeny|DenyIsValue|FailsClosed' ./internal/gateway/ -count=1 -timeout 180s -v
```

**Checkpoint:** State the two gateway theorems and explain why buildCall minting its own tainted agent-scoped Ref (not accepting one off the wire) is what prevents a network bypass. Name the honest gap the proof discloses, and explain why this is the serving-side analogue of the security floor.

---

## L600 — Mastery: benchmarks, honesty discipline, and extending the kernel

**Theme.** Honest baselines and the benchmark authority, the fleet/web/parity results, the AgentDojo red-team, the claims ledger and status gates, the additive ABI + architest, the RSI ship-gate, the three-gate leaf pattern, and the dispatch loop.

**Who joins here.** A contributor or reviewer who has worked through the cores and serving. Join here if you want to read fak's numbers honestly, land an optimization that survives review, or operate the self-improvement and issue-dispatch loops.

**Assumes you can already pass:** **FAK 207**, **FAK 208**, **FAK 209**, **FAK 210**.

| Course | Hard prerequisites |
|---|---|
| **FAK 601** — The Claims Ledger: SHIPPED/SIMULATED/STUB and the 0/29-Novel Posture | **FAK 207** |
| **FAK 602** — STATUS, Subsystem Checks, and What a Passing Boundary Does NOT Prove | **FAK 601** |
| **FAK 603** — The Repro Packet: A No-Credential Offline Boundary Reproduction | **FAK 601**, **FAK 105** |
| **FAK 604** — The Fleet Benchmark Suite: Five Model-Agnostic Kernel Demos | **FAK 405**, **FAK 407** |
| **FAK 605** — Honest Baselines: Naive/Cold vs Tuned Warm-Cache, Measured vs Modeled | **FAK 604**, **FAK 403** |
| **FAK 606** — Benchmark-Authority: The Single Source of Truth Discipline | **FAK 605** |
| **FAK 607** — A/B Paired-Replay Isolation: Attributable Deltas | **FAK 604**, **FAK 407** |
| **FAK 608** — Metrics: Percentiles, KPIs, and the A/B Gate | **FAK 607** |
| **FAK 609** — WebVoyager Baselines and Baseline Stratification | **FAK 605** |
| **FAK 610** — fak vs vLLM / SGLang / llama.cpp / Provider KV Caching | **FAK 609**, **FAK 405** |
| **FAK 611** — The Hardware Matrix: Portability as a Correctness Claim | **FAK 606**, **FAK 530** |
| **FAK 612** — Local-vs-Frontier Parity: Three Axes, Never Blended | **FAK 303**, **FAK 607** |
| **FAK 613** — The AgentDojo Red-Team Threat Model and Two-Gate Defense | **FAK 303**, **FAK 315** |
| **FAK 614** — The RSI Ship-Gate: The Non-Forgeable Keep-Bit and the Self-Measured Loop | **FAK 207**, **FAK 210** |
| **FAK 615** — Extending fak: The Three-Gate Leaf Pattern | **FAK 209**, **FAK 210**, **FAK 614** |
| **FAK 616** — The Witness-Gated Issue-Dispatch Loop | **FAK 614**, **FAK 307** |
| **FAK 617** — Loops All the Way Down: The Durable Verified Loop, Loop Health, and Session Net-True | **FAK 614**, **FAK 616** |
| **FAK 618** — Navigating the Shipped Surface: Verb, Command, or Internal Leaf? | **FAK 209**, **FAK 617** |
| **FAK 619** — From Objective to Runtime Evidence and Retained Learning | **FAK 614**, **FAK 618** |

### FAK 601 — The Claims Ledger: SHIPPED/SIMULATED/STUB and the 0/29-Novel Posture

**Prerequisites:** **FAK 207**

**You'll be able to:**
- Assign exactly one tag (SHIPPED / SIMULATED / STUB) to a capability claim and justify it
- Explain what the 0/29-novel finding means for how fak frames its contribution (the assembly, not a novel primitive)
- Surface the honest ceilings (the ~100% evadable detector; baselines that are vs-naive not vs-tuned)

**Read:** [`CLAIMS.md`](../../CLAIMS.md), [`STATUS.md`](../../STATUS.md)

**Lab:**
```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\ci.ps1
```

**Checkpoint:** Given a capability described as 'GPU backend witnessed real' vs 'token-per-watt telemetry', assign the correct tag to each and justify it; explain what the 0/29-novel finding means for how fak frames its contribution.

### FAK 602 — STATUS, Subsystem Checks, and What a Passing Boundary Does NOT Prove

**Prerequisites:** **FAK 601**

**You'll be able to:**
- Read STATUS.md and SUBSYSTEM-CHECKS.md with each check's explicit 'what it does not prove' column
- State what the tau2-smoke boundary-tax check proves and three things it does not
- Name the two real product gates (Phase 0 clean-node, Phase 1 non-reference 7-9B GPU parity)

**Read:** [`STATUS.md`](../../STATUS.md), [`SUBSYSTEM-CHECKS.md`](../../SUBSYSTEM-CHECKS.md)

**Lab:**
```bash
python tools\subsystem_check_audit.py --profile smoke --out-json fak\experiments\subsystem-checks\latest-smoke.json --out-md fak\experiments\subsystem-checks\latest-smoke.md
```

**Checkpoint:** State what the tau2-smoke boundary-tax check proves and at least three things it explicitly does not, and name the two real product gates.

### FAK 603 — The Repro Packet: A No-Credential Offline Boundary Reproduction

**Prerequisites:** **FAK 601**, **FAK 105**

**You'll be able to:**
- Run the four packet commands and state what each of the four witnesses proves
- State what the packet's Non-Claims section deliberately does NOT prove (detector recall, production readiness, fleet-scale)
- Put the smallest honest artifact in front of a skeptic

**Read:** [`docs/repro-packet.md`](../repro-packet.md)

**Lab:**
```bash
go run ./cmd/fak policy --check examples/customer-support-readonly-policy.json && go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}" && go run ./cmd/fak agent --offline
```

**Checkpoint:** Run the four packet commands and state, from the output, what each of the four witnesses proves and what the packet's Non-Claims section says it deliberately does NOT prove.

### FAK 604 — The Fleet Benchmark Suite: Five Model-Agnostic Kernel Demos

**Prerequisites:** **FAK 405**, **FAK 407**

**You'll be able to:**
- Name the five demos (fan-out, turn-tax sweep, A/B + safety floor, RadixAttention hit rate, token accounting)
- For each demo, name the one kernel counter or ablation it reads
- Explain why none of them needs a GPU

**Read:** [`docs/explainers/fleet-benchmarks.md`](../explainers/fleet-benchmarks.md)

**Lab:**
```bash
go run ./cmd/fanbench -agent-max 1024 -grid log  # then: go run ./cmd/fleetbench -agents 50 -turns 50 -trials 24 -profile read-heavy -granularity resource
```

**Checkpoint:** Name the five demos and state, for each, the one kernel counter or ablation it reads. Explain why none of them needs a GPU.

### FAK 605 — Honest Baselines: Naive/Cold vs Tuned Warm-Cache, Measured vs Modeled

**Prerequisites:** **FAK 604**, **FAK 403**

**You'll be able to:**
- Report every multiple against BOTH a naive/cold reference and the best already-shipped warm baseline
- Never blend measured kernel events with modeled cost
- Explain which number survives contact with a tuned SGLang stack and why

**Read:** [`docs/explainers/fleet-benchmarks.md`](../explainers/fleet-benchmarks.md), [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md)

**Lab:**
```bash
go run ./cmd/ctxdemo -print  # read the same table's (refx)=35.5x cold column vs fak-win=1.1x warm column side by side
```

**Checkpoint:** Given the ctxdemo fleet-5x50 row (35.5x vs cold, 1.1x vs warm), explain which number survives contact with a tuned SGLang stack and why, and which half of a turntax result is measured vs modeled.

### FAK 606 — Benchmark-Authority: The Single Source of Truth Discipline

**Prerequisites:** **FAK 605**

**You'll be able to:**
- State the rule for adding/changing a benchmark number and the three pieces of evidence that must back it (source commit, JSON artifact, reproduce command)
- Trace a row to its cited artifact and confirm the field value
- Explain why a stale claim is tombstoned (e.g. 11.2x->5.3x), not removed, and what made the old number shrink

**Read:** [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md), [`docs/explainers/fleet-benchmarks.md`](../explainers/fleet-benchmarks.md)

**Lab:**
```bash
Pick any row in BENCHMARK-AUTHORITY.md (e.g. RadixAttention hit rate 86.7%) and trace it: open its cited JSON artifact and confirm the field value matches; run the row's reproduce command.
```

**Checkpoint:** State the rule for adding/changing a benchmark number and what three pieces of evidence must back it. Explain why the F1 tombstone (50x5 11.2x->5.3x) is kept, not removed, and what made the old number shrink.

### FAK 607 — A/B Paired-Replay Isolation: Attributable Deltas

**Prerequisites:** **FAK 604**, **FAK 407**

**You'll be able to:**
- State the two isolation invariants: only the toggled variable differs, and Net.TurnsSaved delta == VDSOHits exactly
- Explain why the happy-path control saving 0 matters
- Replay one frozen trace through a freshly-reset kernel twice toggling one lever

**Read:** [`docs/proofs/bench-ab-isolation.md`](../proofs/bench-ab-isolation.md)

**Lab:**
```bash
go test ./internal/turnbench/ -count=1 -run 'TestRun_VDSOAblationIsARealPathSwap|TestRun_HappyPathSavesNothing|TestStochastic_ZeroRateP50IsZero' -v
```

**Checkpoint:** Explain the two invariants the isolation proof discharges and why the happy-path control saving 0 matters.

### FAK 608 — Metrics: Percentiles, KPIs, and the A/B Gate

**Prerequisites:** **FAK 607**

**You'll be able to:**
- Show why pct(p)=sorted[int(p/100*(n-1))] is monotone non-decreasing in p (P50<=P99)
- Explain the identical-workload guard and the fail-closed gate at a zero baseline
- State the doc's two honest OPENs (one sample-set instance witnessed; KPI fold-equals-definition lives in bench.go)

**Read:** [`docs/proofs/metrics.md`](../proofs/metrics.md)

**Lab:**
```bash
go test ./internal/metrics/ -run 'TestHistPercentilesMonotonic|TestValidateWorkloadHash|TestComputeGate' -count=1 -timeout 120s -v
```

**Checkpoint:** Show why pct(p) is monotone non-decreasing in p. Then explain the doc's two honest OPENs.

### FAK 609 — WebVoyager Baselines and Baseline Stratification

**Prerequisites:** **FAK 605**

**You'll be able to:**
- Distinguish A/C (8.8-9.7x), B/C (1.0-1.10x), and A/B (8.8x worker-independent) on the 643-task WebVoyager set
- Identify which is the structural turn-tax and which is the marginal-vs-tuned win
- Explain why fak does not appear on the success-rate leaderboard (capability vs efficiency)

**Read:** [`docs/webbench-baselines.md`](../webbench-baselines.md)

**Lab:**
```bash
go run ./cmd/fak webbench describe --dataset testdata/webbench/sample-tasks.jsonl
```

**Checkpoint:** On WebVoyager, distinguish A/C, B/C, and A/B. Which is the structural turn-tax, which is the marginal-vs-tuned win, and why does fak not appear on the success-rate leaderboard?

### FAK 610 — fak vs vLLM / SGLang / llama.cpp / Provider KV Caching

**Prerequisites:** **FAK 609**, **FAK 405**

**You'll be able to:**
- Explain why a per-instance vLLM cache stores ~10x more tokens than fak for a 100-agent fleet
- Name the one capability (addressable/governance eviction) an opportunistic LRU radix cache structurally cannot offer
- Position fak honestly: matches SGLang's hit rate, does NOT win raw throughput, adds the cross-worker layer

**Read:** [`docs/fak-vs-alternatives-comparison.md`](../fak-vs-alternatives-comparison.md)

**Lab:**
```bash
go run ./cmd/radixbench -scale 1  # compare fak's hit rate against SGLang's published 50-99% band; note policy-eviction witness
```

**Checkpoint:** For a 100-agent / 100-issue fleet, explain why a per-instance vLLM cache stores ~10x more tokens than fak, and name the one capability that an opportunistic LRU radix cache structurally cannot offer.

### FAK 611 — The Hardware Matrix: Portability as a Correctness Claim

**Prerequisites:** **FAK 606**, **FAK 530**

**You'll be able to:**
- Explain why running the same correctness gates on four platforms (Metal, Vulkan, CUDA Ada+Ampere) is itself a result
- Distinguish which numbers may differ across boxes (live wall-clock) from those that must reproduce byte-for-byte (deterministic token-count/hit-rate)
- Inspect the machine-readable node catalog

**Read:** [`docs/HARDWARE-MATRIX.md`](../HARDWARE-MATRIX.md), [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md)

**Lab:**
```bash
python tools/bench_catalog.py show  # inspect the machine-readable node catalog (roles, runs, by-model indexes)
```

**Checkpoint:** Explain why running the SAME correctness gates on four hardware platforms is itself a result, and which class of numbers is allowed to differ across boxes and why.

### FAK 612 — Local-vs-Frontier Parity: Three Axes, Never Blended

**Prerequisites:** **FAK 303**, **FAK 607**

**You'll be able to:**
- Name the three never-blended axes (safety, cost, capability) and who delivers each
- Explain why a local model running fewer turns is not 'faster'
- Explain why the safety win (injection containment) is structural rather than alignment-probabilistic

**Read:** [`docs/explainers/local-vs-frontier-parity.md`](../explainers/local-vs-frontier-parity.md), [`SOTA-COMPARISON.md`](../../SOTA-COMPARISON.md)

**Lab:**
```bash
go -C fak run ./cmd/paritybench --local 'fak/experiments/parity/local-*.json' --reference-cards fak/experiments/parity/reference-frontier.json --reference claude-sonnet --out-md fak/experiments/parity/PARITY.md
```

**Checkpoint:** Name the three never-blended axes and who delivers each. Explain why a local model running FEWER turns is not 'faster', and why the safety win is structural rather than alignment-probabilistic.

### FAK 613 — The AgentDojo Red-Team Threat Model and Two-Gate Defense

**Prerequisites:** **FAK 303**, **FAK 315**

**You'll be able to:**
- Explain why detection-only shows ASR > 0 on paraphrased attacks while full-stack (capability floor + provenance IFC) holds at 0
- Identify which of the four compiled-loop arrows is intentionally NOT built (an RL generator) and why the generative expander is an honest stand-in
- Score Attack Success Rate against two independent gates under an adaptive attacker

**Read:** [`examples/agentdojo-redteam/README.md`](../../examples/agentdojo-redteam/README.md), [`docs/fak/security.md`](../fak/security.md)

**Lab:**
```bash
./examples/agentdojo-redteam/run.sh   # exit 0 iff full-stack ASR == 0 (every attack barred)
```

**Checkpoint:** Why does the detection-only defense show ASR > 0 on paraphrased attacks while full-stack holds at 0? Which of the four compiled-loop arrows is intentionally NOT built, and why is the generative expander an honest stand-in?

### FAK 614 — The RSI Ship-Gate: The Non-Forgeable Keep-Bit and the Self-Measured Loop

**Prerequisites:** **FAK 207**, **FAK 210**

**You'll be able to:**
- Explain why shipgate.Evaluate KEEPs only on strict metric gain AND green suite AND clean truth syscall
- Explain why the unexported keep-bit set only inside Evaluate makes 'no measurable win -> REVERT' forgery-proof
- Explain why the loop re-derives its baseline from latest main every run

**Read:** [`docs/rsi-loop.md`](../rsi-loop.md), [`docs/proofs/shipgate.md`](../proofs/shipgate.md)

**Lab:**
```bash
go run ./cmd/rsiloop -mode improve -repo . -baseline-ref main -candidates 6,8,8,10 -journal /tmp/rsi.jsonl
```

**Checkpoint:** Explain cycle 3 of the witnessed rsiloop run: why a candidate with a green suite AND a clean tree is still REVERTED, and why the loop re-derives its baseline from latest main every run.

### FAK 615 — Extending fak: The Three-Gate Leaf Pattern

**Prerequisites:** **FAK 209**, **FAK 210**, **FAK 614**

**You'll be able to:**
- Attach at a Register* seam, prove correctness with a deterministic witness, then prove a speed win via the non-forgeable keep-bit
- For a new quantization kernel, name the seam (internal/compute), the correctness class to declare, and the exact gate command that proves it earns its keep
- Explain why a contributor cannot land a plausible-but-wrong (gate 2) or correct-but-slower (gate 3) kernel

**Read:** [`EXTENDING.md`](../../EXTENDING.md), [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

**Lab:**
```bash
python tools/extend_preflight.py
```

**Checkpoint:** For a new quantization kernel, name which seam it uses, which correctness class it should declare, and which exact gate command proves it earns its keep (the Gate 3 keep-bit from FAK 614).

### FAK 616 — The Witness-Gated Issue-Dispatch Loop

**Prerequisites:** **FAK 614**, **FAK 307**

**You'll be able to:**
- Trace the loop: route -> spawn one worker -> require an #N-cited commit -> bind commit to issue via dos commit-audit -> close only when re-verified per-SHA
- Run the read-only issue-gardening pass, distinguish mechanical actions from review-only priority/area/ownership decisions, and name the current top backlog rot from the report
- Explain why a resolved issue whose commit omits #N can never be witnessed-closed
- Explain how the loop guarantees the live-worker population can never exceed its cap

**Read:** [`docs/dispatch-loop.md`](../dispatch-loop.md), [`.claude/skills/issue-triage/SKILL.md`](../../.claude/skills/issue-triage/SKILL.md), [`docs/SKILL-CONTEXT-MEMORY.md`](../SKILL-CONTEXT-MEMORY.md)

**Lab:**
```bash
python tools/issue_triage.py --markdown --out docs/_audits/issue-triage-YYYY-MM-DD.md
python tools/issue_triage.py --actions --out docs/_audits/issue-actions-YYYY-MM-DD.json
python tools/dispatch_status.py
```

**Checkpoint:** From the issue-triage report, name the largest current backlog gap
and the top three review-only P0/P1 rows. Then explain why a resolved issue whose
commit omits #N can never be witnessed-closed, how the loop guarantees the
live-worker population can never exceed its cap, and why an identical skill
invocation can be served as procedural-memory HIT rather than re-rendered.

### FAK 617 — Loops All the Way Down: The Durable Verified Loop, Loop Health, and Session Net-True

**Prerequisites:** **FAK 614**, **FAK 616**

**You'll be able to:**
- Place every fak mechanism on the five-ring loop ladder (tool-call → turn → session → fleet → RSI) and name the witness primitive each ring carries, plus the five orthogonal threads (trust, cost, memory, observability, governance)
- Distinguish the durable loop ledger (`fak loop run -- CMD`, which records a hash-chained `HeadBefore..HeadAfter` witness) and the verified driver (`fak loop drive`) from the hand-fed one-shot `rsicycle`, and say what a `dark-loop` state means in `fak loop health`
- Read a session's net-true verdict (HELPED / WASH / HURT) and explain why cost data alone (tokens, dollars) cannot grade whether a session *achieved* anything

**Read:** [`docs/explainers/engineering-is-building-loops.md`](../explainers/engineering-is-building-loops.md), [`docs/rsi-loop.md`](../rsi-loop.md), [`docs/fak/session-observability-rsi-loop.md`](../fak/session-observability-rsi-loop.md)

**Lab:**
```bash
go test ./internal/loopmgr/ ./internal/rsiloop/ ./internal/sessionobs/ -count=1 -timeout 120s
```

**Checkpoint:** Draw the five-ring ladder and name the witness primitive each ring carries (the adjudicator's provable refusal, ctxmmu's Clear+rescreen, recall's sealed page, the fleet's per-SHA `dos commit-audit`, the RSI keep-bit). Then explain why a `fak loop drive` turn that the model calls "done" still re-arms unless a dos witness agrees, and why a session that burned 200 turns and hit a STOP must grade HURT, not WASH, even though both spent tokens.

### FAK 618 — Navigating the Shipped Surface: Verb, Command, or Internal Leaf?

**Prerequisites:** **FAK 209**, **FAK 617**

**You'll be able to:**
- Start with a supported `fak <verb>` when operating the product, use a standalone
  `cmd/<name>` only for a bounded fixture, publisher, or compatibility lab, and open an
  `internal/<leaf>` when changing the pure contract behind either entry point
- Follow a request from the appendix's user-facing verb to its implementation leaf, then
  identify the witness surface that prevents the command's output from becoming a claim
- Distinguish read-only reports and dry-run plans from commands that launch, enroll,
  publish, reap, or otherwise mutate state before choosing a first probe

**Lab:**
```bash
go run ./cmd/fak help architecture
go run ./cmd/fak index query "architecture report" --json
go test ./internal/archreport -count=1
```

**Expected result:** Help supplies the operator contract, the index locates the owning
surface, and the package test exercises the contract without making an external change.
Use the three maps in the appendix to repeat that trace for an operations, model, or
curriculum task.

**Checkpoint:** For a new model-performance observation, explain why `fak model-observe`
is the normal operator door, `cmd/modelperfobs` is the bounded capture/report utility,
and `internal/modelperfobs` is the reusable measurement contract. Then name which layer
you would test after changing only the observation fold, and which command is safe to run
first when an entry offers both a report and a mutating action.

### FAK 619 — From Objective to Runtime Evidence and Retained Learning

**Prerequisites:** **FAK 614**, **FAK 618**

**You'll be able to:**

- Turn an objective into a bounded plan, preserve external-study provenance, and compile
  findings into explicit transfer candidates without confusing any of those steps with
  execution
- Inspect what the current binary can run before loading a model, then distinguish that
  preflight from actually starting the unified runtime
- Trace native-performance facts from bounded metrics and artifacts through SLO coverage
  into the scorecard that decides what the next improvement cycle must learn

The work-and-learning route has three distinct boundaries. `fak agentic` calls
`internal/agentic` to compile broad text into a deterministic, read-only, offline
expand → experiment → contract plan; its `fak ultracode` handoff is data, not a worker
launch. `fak study` stores and retrieves immutable content-addressed research receipts.
`fak learning-mesh compile --file LEDGER.json` then calls `internal/learningmesh` to
compare provenance-bearing mechanisms across declared hardware, framework, engine, and
baseline envelopes. A `COPY`, `ADAPT`, `BENCHMARK_ONLY`, `REJECT`, or `UNKNOWN` candidate
is still a candidate, never permission to change the product execution path.

Two discovery leaves support that route without becoming hidden authorities:
`internal/docsearch` loads the curated documentation map used by repository discovery,
while `internal/openviking` is an optional, bounded HTTP adapter for OpenViking search,
message, and commit operations. The local documentation map remains usable offline, and
an OpenViking result still needs the same study provenance and witness as any other
external observation.

The runtime route starts with `fak runtime-capabilities`, whose `internal/runtimecap`
fold separates “the binary runs,” “the governed control plane runs,” and “this requested
model backend is runnable.” Exact `--backend` requests fail closed; only an explicit
`--prefer-backend ... --fallback-policy local_cpu_degraded --cpu-envelope ...` posture
may select the portable CPU fallback, and remote placement must pass every declared gate
before payload load. `fak up` is the short product door for `fak serve`: it starts the
same gateway, flags, policy, metrics, and session lifecycle rather than a second server.
Behind optional runtime operations, `internal/dockerprocess` bounds Docker Compose calls
for the rich dashboard, and `internal/harnessserve` owns one adapter-provided loopback
model process with readiness, one-token probe, ownership, and bounded shutdown receipts.
Update automation stays separate: `internal/selfupdate` classifies `current`, `stale`,
`divergent`, or audit-only `attention` and supplies an explicit next command; it does not
silently replace a running runtime.

The native-performance route is another evidence pipeline, not one giant metric bag:

| Stage | Owning contract and relationship |
| --- | --- |
| Define and collect | `internal/nativeperfobscontract` freezes the Qwen3.8, `fak-native` signal set and its cardinality/freshness rules; `internal/nativeperfbackend` defines bounded per-backend Prometheus snapshots where unavailable values remain absent rather than zero. |
| Correlate and expose evidence | `internal/nativeperfcorrelation` replaces high-cardinality run, request, trace, and receipt IDs with a scrubbed bounded key; `internal/nativeperfartifact` maps that key to at most five public-safe, expiring receipt/profile/trace/report links. |
| Decide operational health | `internal/nativeperfslo` compares only matched `module@rev` + benchmark + Qwen3.8 model + backend envelopes and preserves `missing_evidence`; `internal/nativeperfcoverage` proves dashboards, contracts, PromQL, fixtures, and live receipts agree. |
| Select and learn | `internal/sweepcert` validates extrema, thresholds, censored edges, constraints, and point provenance across a declared sweep. `fak performance-rsi-scorecard` feeds versioned evidence to `internal/perfrsiscore`, which scores the complete improvement loop, names debt and the dominant bottleneck, and can compare a prior report; it does not claim raw model speed. |

**Lab:**

```bash
go run ./cmd/fak agentic --json --objective "turn one measured performance finding into a bounded learning cycle"
go run ./cmd/fak study search --limit 5 --store /tmp/fak-learning-study "native performance"
go run ./cmd/fak learning-mesh compile --file docs/_witnesses/issue-9839/mechanisms.json > /tmp/fak-learning-candidates.json
cmp /tmp/fak-learning-candidates.json docs/_witnesses/issue-9839/candidates.json
go run ./cmd/fak runtime-capabilities
go run ./cmd/fak up --help
go run ./cmd/fak performance-rsi-scorecard --input internal/perfrsiscore/testdata/complete.json --json > /tmp/fak-learning-performance-rsi.json
go test ./internal/agentic ./internal/dockerprocess ./internal/docsearch ./internal/harnessserve ./internal/learningmesh ./internal/nativeperfartifact ./internal/nativeperfbackend ./internal/nativeperfcorrelation ./internal/nativeperfcoverage ./internal/nativeperfobscontract ./internal/nativeperfslo ./internal/openviking ./internal/perfrsiscore ./internal/runtimecap ./internal/selfupdate ./internal/sweepcert -count=1
```

**Expected result:** The agentic plan reports `read_only=true` and `offline=true`; an
empty study store returns `[]`; the learning-mesh output matches its captured candidate
