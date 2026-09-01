---
title: "Simulation in performance benchmarking: deterministic feedback, calibrated estimates, and claim ceilings"
description: "A dated field-borrow pass over fak's simulated performance evidence and current simulator practice, defining where models help, where they mislead, and the smallest provenance-safe middle ground tracked by #9424."
---

# Simulation in performance benchmarking

> **Observed at:** `2026-08-27`. **Verdict:** fak is **PARTIAL**. It already has useful
> deterministic simulators, synthetic full-shape execution, projected rooflines, strict
> simulated-versus-measured prose fences, and a generic calibration standard. It does not yet
> bind those pieces in the shared benchmark artifact with a machine-readable validity envelope,
> calibration residual, and maximum admissible claim.
>
> **Durable study:**
> `study_30d16d6e80876b199a258e294c0fd99d0f708ae6378848ae2983e0d1b8e60f7d`.
> **Implementation tracker:** [#9424](https://github.com/anthony-chaudhary/fak/issues/9424).
> **Companions:** [benchmark governance](../../BENCHMARK-GOVERNANCE.md),
> [prediction calibration](../standards/prediction-calibration.md),
> [system-baseline evidence](CONCEPT-SYSTEM-BASELINE-CONTAMINATION-2026-08-26.md),
> [measured roofline evidence #9128](https://github.com/anthony-chaudhary/fak/issues/9128).

## Outcome

Simulation is valuable here when it answers a narrower question than a hardware benchmark:

- Does the mechanism preserve legality, invariants, work counts, or address/action counts?
- Which resource ceiling can bind under stated assumptions?
- Does one configuration rank ahead of another inside the same frozen workload and model?
- Which few candidates deserve an expensive lab run?
- After hardware calibration, what absolute range is plausible inside the validated envelope?

It hurts when a deterministic point estimate is treated as achieved performance. The safe
middle ground is a **multi-fidelity promotion ladder**, not a generic simulated score:

1. exact structural counts;
2. analytical bounds and bottleneck hypotheses;
3. trace or cycle simulation for matched local ranking;
4. independently calibrated absolute estimates with residual bands;
5. matched real-hardware measurement for achieved and competitive claims.

Hardware availability should improve a simulator rather than make it disappear. Run the model
and the real workload side by side, retain the residuals, and use the calibrated model to cover
nearby configurations cheaply. Hardware unavailability lowers the maximum permissible claim; it
does not make the feedback worthless.

## Self-query witness: PARTIAL

The field-borrow query returned no discoverable capability:

```text
fak capabilities "prediction calibration residual hardware performance simulator"
-> no matching capability

fak dev index claims --limit 12 simulation benchmark hardware
-> no matching claim
```

Raw inspection found the capability in pieces:

| Present piece | Exact seam | What it proves | Boundary |
|---|---|---|---|
| Simulated claim vocabulary | `CLAIMS.md:6-7` | `[SIMULATED]` means labeled stand-in data and illustrative numbers | Prose/status ledger, not a benchmark receipt contract |
| Theory-versus-measurement policy | `BENCHMARK-GOVERNANCE.md:104-129` | A formula, geometry model, simulation, or projection is `THEORETICAL`; it cannot be called measured or a win | Finer provenance is optional and non-normative |
| Deterministic projected roofline | `cmd/coalescebench/main.go:364-372` | Same seed/config gives the same route/coalescing projection; placeholders and “not served” are explicit | No real router trace, device execution, or achieved tok/s |
| Measured-current/estimated-ceiling join | `internal/roofline/roofline.go:381-386` | Estimated ceilings can guide a drive while missing measured currents stay pending | The estimate is not calibrated per target configuration |
| Full-shape synthetic execution | `internal/model/synthetic_perf_test.go:10-78`; `internal/model/synthetic.go:61-69` | Real fak code executes a deterministic model shape and exposes host-path costs without downloading a checkpoint | Random weights do not prove model quality, real-weight traffic, or target-device timing |
| Prediction calibration | `docs/standards/prediction-calibration.md:64-121` | Independent `WITNESSED|OBSERVED` reality, residual bands, direction-aware over-claim, and fail-closed insufficient evidence | Not joined to performance simulation artifacts |
| Shared benchmark envelope | `internal/benchcli/benchcli.go:104-178` | Code, harness, machine, model, config, results, invalidation, and witness identity | Missing evidence type, claim ceiling, simulator identity, calibration envelope/residual, and excluded effects |

This is why the verdict is **PARTIAL**, not ABSENT: useful deterministic feedback and honesty
fences already ship. The missing capability is the typed join that lets automated consumers use
simulation without promoting it beyond what it can prove.

## Evidence ladder and permitted claims

| Evidence type | Useful deterministic output | Maximum default claim | Promotion requirement | Invalid when |
|---|---|---|---|---|
| `structural_count` | bytes, operations, cache events, dependencies, page-ins, legal schedules | `correctness_only` | none for exact modeled invariants; keep excluded behavior explicit | the implementation changes the counted semantics |
| `analytical_bound` | roofline, capacity, lower/upper bound, bottleneck hypothesis, sensitivity | `bottleneck_only` | measured ceilings/counters to classify achieved behavior | assumptions, resource law, fit, precision, or workload shape changes |
| `trace_sim` | repeatable cache/branch/memory/scheduler comparison on one captured stream | `relative_rank` | rank stability across plausible configs; hardware finalists | the change can alter ISA, control flow, synchronization, allocation, library/kernel choice, or arrivals |
| `cycle_sim` | detailed cycle/counter prediction inside a modeled architecture | `relative_rank` | independent holdout calibration for `absolute_estimate` | simulator/config/ISA/toolchain/hardware/workload leaves the calibrated envelope |
| `learned_estimate` | cheap proposal ordering or pruning over a large search space | `relative_rank` | OOD detection plus real measurements that refresh the model | feature/compiler/target/workload distribution drifts or the model cannot abstain |
| `calibrated_sim` | absolute estimate plus error band in a named validity domain | `absolute_estimate` | independent held-out real-hardware anchors and residual gate | calibration is in-sample, stale, too thin, or outside the target envelope |
| `hardware_measurement` | achieved wall-clock, counters, energy, memory, quality, tails | `measured_absolute` | matched envelope, repetitions/noise/ambient gates, witness artifact | engine, quality, workload, machine, config, or baseline is incomparable |

No simulated rung may support fak-versus-llama.cpp throughput, latency, energy, cost, p99, or a
“native faster” headline. Those claims are statements about achieved systems, including effects
the simulator may omit.

## Dated primary-source ledger

| Source | State, event, immutable anchor | What changed in the design | License disposition | Refresh trigger |
|---|---|---|---|---|
| Accel-Sim 2.0 | released 2026-08-25; [`v2.0.0@6465301`](https://github.com/accel-sim/accel-sim-framework/releases/tag/v2.0.0); paper v1 submitted 2026-08-23, [`arXiv:2608.22602v1`](https://arxiv.org/abs/2608.22602v1) | Trace-driven H100/H200 cycle modeling is validated against real H100 silicon. The reported 99% Pearson correlation still carries 13.4% mean absolute cycle error over 34,000+ kernel instances. Correlation and determinism therefore travel with a residual and a scope, never replace them. The framework also separates tracer, simulator, hardware correlator, and tuner—the same separation fak needs between prediction and independent reality. | **ADAPT**, BSD-2-Clause at the tag; no source copied. Paper is **INSPIRE-ONLY**. | release, modeled GPU/ISA, tracer/profiler/driver change, or residual drift |
| Apache TVM MetaSchedule | v0.25.0 released 2026-06-19 at [`c7ba073`](https://github.com/apache/tvm/releases); [current MetaSchedule architecture](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html) observed 2026-08-27 | The XGBoost cost model reduces the number of real-device measurements; the builder/runner still measures candidates and feeds results back. Learned models are proposal/ranking filters, not hardware-free witnesses. | **ADAPT**, Apache-2.0; mechanism only, no code copied. | release, feature/cost-model/compiler/target change, rank inversion, OOD rate |
| SimGrid 4.1 | current released docs observed 2026-08-27; [calibration guide](https://simgrid.org/doc/latest/Calibrating_the_models.html) | The official guide says defaults may not match a particular system and prescribes real-platform experiments, parameter fitting, then simulation-versus-real comparison. This supplies the shadow-calibration pattern and the fail-closed response to an uncalibrated profile. | **INSPIRE-ONLY**, LGPL-2.1; no implementation copied. | release/model/fabric/runtime change or calibration residual drift |
| gem5 | v25.1.0.1 current stable release observed 2026-08-27; [gem5 resources paper](https://www.gem5.org/assets/files/papers/enabling2021ispass.pdf) | Atomic, timing-simple, and out-of-order modes explicitly trade speed for timing fidelity. The paper's GPU case finds a counterintuitive ranking caused by overly simple dependency tracking, showing why a plausible simulated total can still select the wrong lever. | **INSPIRE-ONLY**, BSD-3-Clause; no implementation copied. | release, CPU/GPU/memory model, OS/compiler/workload change |
| FireSim | shipped FireSim 1.12 docs observed 2026-08-27; [target abstraction](https://docs.fires.im/en/1.12.0/Golden-Gate/LI-BDN.html) | Cycle-exact FPGA-hosted models preserve target timing; abstract models deliberately trade fidelity for simulation performance and resource use. “Cycle accurate” is therefore a property of a declared boundary, not a blanket property of every component. | **INSPIRE-ONLY**; documentation principle only, exact code reuse not evaluated. | release, target graph/bridge/model-boundary change |
| NVIDIA Nsight Compute 13.3 | shipped vendor profiler docs observed 2026-08-27; [profiling and roofline guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Hardware counters supply the independent anchor. The profiler itself can perturb context: cache flushing improves replay determinism but can misrepresent cache-sensitive application context, and persistence state affects startup behavior. Calibration receipts must capture collection mode and excluded context rather than treating the profiler as transparent truth. | **INSPIRE-ONLY**, proprietary docs/tool. | profiler/CUDA/driver/device revision or metric-definition change |

Coverage checked: current releases, official docs, papers, license roots where code reuse was a
candidate, and fak's code/tests/open and closed issue backlog. Accel-Sim issues/PRs/discussions,
gem5's full validation corpus, and simulator implementation internals were not exhaustively mined
because this pass proposes no simulator port. Any optional adapter should get its own
`study-repo` pass before code reuse.

## Where simulation helps fak now

### High-value, low-risk uses

- **CI regression witnesses:** exact work/byte/action counts, cache admission, scheduler legality,
  deterministic eviction/coalescing behavior, and stable report rendering.
- **Bottleneck triage:** roofline and sensitivity sweeps can distinguish “this cannot be compute
  bound under these assumptions” from “this achieved compute throughput.”
- **Candidate pruning:** sweep cache sizes, batch sizes, route skew, placements, or mappings and
  send only robust finalists to scarce GPUs.
- **Counterfactuals:** change one modeled parameter while holding the trace and workload fixed to
  explain which mechanism could have caused a measured delta.
- **Coverage around anchors:** when hardware exists, calibrate on a designed subset and use the
  model to explore nearby configurations with an explicit error band.
- **Long or expensive spans:** measure representative live segments, simulate the expensive span,
  and validate the extrapolation on held-out spans—the pattern already used by some fak authority
  rows, provided the modeled half stays labeled.

### Cases that need detailed simulation or a learned residual

- Memory/cache/fabric policies whose relative ordering depends on contention timing.
- New accelerator configurations unavailable as silicon, where architecture exploration is the
  point and no achieved-performance claim is made.
- Large search spaces where analytical rules eliminate invalid candidates, a cost model ranks the
  survivors, and a small hardware budget closes the loop.

## Where it hurts or becomes misleading

- Absolute TTFT, inter-token latency, throughput, energy, cost, p95/p99, jitter, and SLA claims.
- Thermal throttling, DVFS, clock residency, power caps, NUMA, allocator/OS scheduling, driver
  behavior, launch overhead, closed-library kernel selection, and background contention.
- Feedback-dependent scheduling, admission, cache replacement, or networking when a frozen trace
  prevents the proposed change from changing future events.
- Cross-generation extrapolation, especially when ISA, memory hierarchy, synchronization, or
  library code generation changes.
- One fixed random seed used as if it were a tail distribution. A fixed seed proves replay;
  independent streams and replications support inference.
- A learned model that returns a number outside its training envelope instead of abstaining.
- A detailed simulation that costs more time and compute than the next lab run but is kept because
  it looks more rigorous. Simulator host wall time, CPU, memory, bytes, and operator cost belong in
  the receipt.

## The proposed middle-ground receipt

Add an optional `simulation_evidence` block beside the existing shared benchmark artifact rather
than inventing a parallel “simulated benchmark score.” At minimum it carries:

```text
evidence_type
claim_ceiling
engine {name, revision, mode, config_digest}
workload {trace_or_input_digest, provenance, capture_hardware, compiler, libraries}
randomness {seed, stream_assignment, repetitions}
validity_envelope
excluded_effects
calibration {hardware_profile, observed_at, independent_holdout, sample, residual, error_band, verdict}
simulator_cost {wall_time, cpu_time, peak_memory, artifact_bytes}
```

The validator owns the promotion rule. A `cycle_sim` requesting `measured_absolute` fails before
publication. A `calibrated_sim` requesting `absolute_estimate` passes only with an independent,
in-envelope holdout and a non-insufficient calibration verdict. Hardware artifacts can request
`measured_absolute`, but their existing engine, quality, repetition, ambient, and baseline gates
still apply.

Calibration should reuse the existing direction-aware prediction-calibration semantics rather
than create a second residual vocabulary. Calibrate counter-by-counter as well as end-to-end:
bytes, cache traffic, instructions, stalls, cycles, then aggregate time. A plausible total can
hide compensating errors.

## Candidate matrix and support budget

| Candidate | Disposition | Cohort and relationship to default | Incremental cost | Falsifier |
|---|---|---|---|---|
| Typed simulation evidence + claim ceiling in the shared artifact | **DEFAULT** | Every simulated/model-based performance emitter; strengthens current prose fences | additive schema, validator, lint, one migrated emitter | an invalid evidence/claim pair can reach a measured or competitive authority row |
| Multi-fidelity escalation from counts to hardware | **DEFAULT** | Common performance work; cheap rungs prune lab work but never replace achieved evidence | orchestration/policy plus simulator-cost accounting | it sends more candidates to hardware or costs more than direct measurement without expanding useful coverage |
| Shadow calibration whenever hardware exists | **DEFAULT when applicable** | Existing GPU/Metal/CPU campaigns; turns hardware anchors into reusable nearby coverage | paired runs, residual store, invalidation triggers | held-out residual/rank stability does not improve decision quality |
| Accel-Sim trace/cycle adapter | **OPTIONAL-MODULE / RECIPE** | Named CUDA/Hopper kernel studies needing counterfactual cycle detail; never the native default engine | external toolchain, trace capture, config/profile maintenance, license/attribution | setup/simulation cost exceeds lab measurement or target falls outside supported ISA/workload envelope |
| gem5/FireSim/SimGrid adapters | **RECIPE** | CPU, SoC, and distributed/fabric research with an explicit model boundary | environment-specific operations and calibration | no surviving cohort needs pre-silicon/counterfactual exploration |
| Learned performance cost model | **WATCH** | Very large candidate spaces after enough hardware data exists | training corpus, feature/version/OOD governance | rank inversions or OOD abstention failures exceed saved hardware work |
| Simulated competitive or SLA headline | **EXCLUDE** | None; achieved-system claims require matched execution | high trust cost with no valid evidence gain | reopen only if the claim is explicitly an estimate and independently calibrated in the exact envelope |

## Default and coverage frontiers

The **default frontier** is not a cycle simulator. It is current deterministic structural and
analytical evidence plus a typed claim ceiling, simulator-cost accounting, and hardware
promotion gate. This is cheap enough for CI and applies to the simulators fak already owns.

The **coverage frontier** is optional detailed adapters and learned selectors for named workloads:
CUDA trace/cycle studies, CPU/SoC simulation, fabric/queue experiments, and calibrated nearby
configuration prediction. Each adapter owns a validity envelope, calibration profile, and review
trigger. No adapter silently changes fak-native execution into another runtime.

## P1-P4 check

- **P1 — preserved:** receipts are compact and digest-bound; large traces and calibration corpora
  remain referenced artifacts.
- **P2 — advanced:** deterministic pruning can save lab runs, but simulator compute/storage and
  validation overhead count toward net value.
- **P3 — advanced:** validity envelopes, excluded effects, abstention, and claim ceilings keep
  adaptation bounded and provenance honest.
- **P4 — advanced:** CI runs cheap rungs, lab dispatch consumes finalists, and hardware receipts
  close the calibration loop through one artifact vocabulary.

## Durable trail

[#9424](https://github.com/anthony-chaudhary/fak/issues/9424) is the first checkable spine:
additive receipt metadata, a fail-closed evidence/claim validator, calibration reuse, authority
lint, and one migrated existing emitter. Simulator integrations and calibration profiles belong
in follow-ons after that join works end to end.
