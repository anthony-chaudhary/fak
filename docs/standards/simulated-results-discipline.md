---
title: "Simulated results discipline — why projected numbers are not facts and how the repo fences them"
description: "The repository's authoritative standard for handling simulated, projected, and modeled results: the cognitive and proxy risks in RSI loops, valid vs invalid uses of roofline projections, the over-explain at every stage protocol across all five touchpoints, the six mandatory unmodeled effects, and the dual-track RSI loop architecture separating cheap exploration from unforgiving physical verification."
---

# Simulated results discipline

> **TL;DR:** An analytical model or simulation is a hypothesis about physics rather than physical reality. Use rooflines to prune dead ends and bound search spaces. Never declare victory, close performance issues, or land trunk commits without physical silicon witnesses.

An analytical model calculates theoretical limits. An in-memory Go unit test mocks execution. Neither authors a single byte of physical execution on silicon. When handled carelessly, modeled figures corrupt engineering judgment and mislead operators. In autonomous loops, they drive optimization into hallucinated dead ends.

The repository invariant is simple: **a simulated result may guide where to look, but it may never declare what was found.**

```bash
# Validate claim tags and exposure state across the repository
go test -v -run TestCLAIMSLedger ./internal/claimcheck
```

This standard is the simulation-specific companion to the [net-true-value standard](net-true-value.md), the [observer-effect standard](observer-effect.md), and the [support-maturity honesty fence](support-maturity-honesty-fence.md).

---

## 1. Why we must be super careful with simulated results

Simulations and mathematical projections are cognitively seductive and computationally fragile. In an agentic repository driven by recursive self-improvement (RSI) loops, four distinct failure modes make un-fenced simulated numbers uniquely dangerous.

### 1.1 Cognitive seduction and anchor bias

Humans and AI agents latch onto concrete numbers. Once a figure like `240 tok/s (projected)` appears in a spec or terminal trace, it becomes an anchor:

- Unconscious baseline elevation: Evaluators treat the simulated number as an entitlement. When physical hardware later delivers `135 tok/s` over a `70 tok/s` baseline, people perceive that real win as a defect.
- Narrative lock-in: Once an agent quotes an optimistic speedup, subsequent reasoning rationalizes why the projection holds. The agent stops looking for reasons why silicon will fall short.
- Illusion of precision: Quoting decimals like `184.32 tok/s` conveys false rigor. The underlying model often omitted PCIe bus latency entirely.

### 1.2 Proxy gaming and Goodhart's law in autonomous loops

When an autonomous RSI loop optimizes against a simulation, Goodhart's Law bites quickly. The optimizer rapidly exploits every blind spot in the model:

- Optimizing unmodeled shortcuts: If the simulator omits memory fragmentation, the agent discovers architectures that exploit that omission.
- Degenerate solutions: Optimizers eliminate synchronization barriers because thread execution was modeled as sequential. They choose non-contiguous tensor layouts that minimize paper FLOPs while destroying hardware memory coalescing.
- Severe hardware penalty: The resulting code scores near 100% on the proxy. On real silicon, it deadlocks or crawls at 10% efficiency.

### 1.3 Compounding simulation error across recursive generations

In multi-generation recursive loops, simulation errors compound multiplicatively across generations:

```text
Error_total = Product(1 + epsilon_i) for i = 1 to N
```

- Generation 0: Assumes ideal memory bandwidth without DRAM refresh (+15% error).
- Generation 1: Assumes zero-latency speculative token verification (+25% further error).
- Generation 2: Optimizes batch scheduling on top of Gen 1, ignoring OS thread preemption (+30% further error).
- Generation 3+: By the fourth turn, the loop operates in an imaginary universe. Theoretical throughput looks amazing, but the code fails on physical silicon.

### 1.4 Premature victory declarations

The most common operational failure is confusing a green test in an in-memory harness with physical throughput:

- Passing a mock benchmark in `go test` in 3 milliseconds proves interface conformance. It proves zero hardware throughput.
- Declaring a performance issue resolved on the basis of a mock is an honesty violation. Unit tests prove contracts; physical hardware runs prove performance.

---

## 2. Valid vs. invalid uses of roofline projections

Roofline modeling is an essential engineering tool. Its legitimate utility is strictly bounded.

### 2.1 Legitimate and necessary uses of rooflines

Analytical models and rooflines are valid as negative constraints and sanity checks. They are never proofs of achievement:

1. Negative constraints: A roofline sets the upper bound of physics. If a kernel claims 4.2 TB/s on an NVIDIA L4 with a 300 GB/s physical limit, the design is broken.
2. Bounding search space: Before spending GPU hours on fleet nodes, an agent prunes algorithms that cannot beat the baseline even at 100% roofline.
3. Detecting measurement bugs: When measured wall-clock throughput exceeds 100% of theoretical roofline, you uncovered a benchmark bug. Look for timer rollover, cached repeats, or compiler dead-code elimination.
4. Directing RSI loops toward true bottlenecks: Rooflines classify kernels as compute-bound or memory-bandwidth-bound. This stops agents from tuning math instructions when memory bandwidth is the limiter.

### 2.2 Theoretical ceiling vs. operational reality

A roofline is an asymptotic ceiling rather than an expected delivery date. Real hardware never achieves theoretical peak under real workloads.

Every projection must state its operational context:

- Hardware SKU and stepping: Consumer cards, enterprise PCIe cards, and server modules differ in bus width. Cache hierarchy and ECC memory overhead also differ fundamentally.
- Clock speeds and DVFS: Datasheets quote peak boost clocks. Heavy sustained matrix compute triggers DVFS throttling, dropping clocks by 15–30%.
- Thermal budget and airflow: Workstation enclosures differ from 1U server chassis. Thermal buildup degrades clock stability over long runs.
- Batch size and dimensions: Roofline peaks assume infinite batch sizes. Real agent serving runs at small batches ($B=1$ to $B=4$), operating in latency-bound territory.
- Memory strides and coalescing: Calculations assume aligned 128-byte transactions. Non-coalesced access reduces effective memory bandwidth by 40–70%.
- Bus topology: Inter-socket NUMA hops, PCIe switches, and host-device links impose fixed latency penalties.
- OS and driver tax: Driver launch queues, CUDA streams, and interrupts add overhead per kernel launch.
- MoE routing skew: Mixture-of-Experts routing follows a power-law. Hot experts saturate while cold experts idle, destroying naive average throughput estimates.

### 2.3 The 50–75% realistic hardware efficiency envelope

On production silicon, tuned state-of-the-art kernels achieve **50% to 75% of theoretical roofline**. Reaching 80%+ requires heroic hand-tuning of assembly instructions for one specific GPU architecture.

Any projection assuming $>75\%$ efficiency without citing an existing physical witness on identical silicon is an uncalibrated fantasy. RSI loops must default to a 50% achievable factor when evaluating ideas.

---

## 3. The "Over-Explain at Every Stage" protocol

Any work touching modeled or simulated numbers must follow the Over-Explain protocol across all five touchpoints.

### Stage 1: Design / Plan / Spec

Every spec, RFC, or plan that includes projected numbers must:
1. Open with an explicit disclaimer block:
   > `[MODELED PROJECTION]: All figures in this document are analytical estimates. Zero bytes have run on physical silicon. These numbers represent theoretical ceilings, not measured commitments.`
2. Cite the current physically measured baseline on target hardware.
3. State the required physical witness command before any implementation lands.

### Stage 2: Code / CLI / Terminal output

When a tool or test prints simulated metrics:
1. Loud header banner:
   ```text
   ======================================================================
   [SIMULATED RUN] WARNING: NOT EXECUTED ON PHYSICAL SILICON
   Hardware: EMULATED / ANALYTICAL ROOFLINE
   Unmodeled effects: Bus overhead, thermal throttling, memory contention
   ======================================================================
   ```
2. Inline token prefixes: Every number emitted must carry an inline tag:
   - `[SIMULATED] Throughput: 142.5 tok/s (assumes 65% roofline)`
   - `[MODELED]   Latency:    7.02 ms (bus transfer omitted)`
3. No naked numbers: Printing an unadorned number to stdout or stderr is treated as a bug.

### Stage 3: Data schemas and receipts

Any JSON or YAML receipt must enforce typed provenance fields:

```json
{
  "$schema": "https://fak.dev/schemas/benchmark-receipt-v1.json",
  "provenance": "MODELED",
  "is_physical_silicon": false,
  "hardware_target": "NVIDIA-L4",
  "theoretical_ceiling": 185.0,
  "modeled_estimate": 120.25,
  "baseline_measured": 65.4,
  "assumptions": [
    "batch_size_4",
    "coalesced_128b_reads",
    "sustained_boost_clock_2040mhz"
  ],
  "unmodeled_effects": [
    "pcie_dma_handshake_latency",
    "host_cgo_context_switch",
    "thermal_clock_scaling",
    "bank_conflict_penalty"
  ],
  "missing_witness": "make bench-l4-native RUN=TestQwenPrefill"
}
```

Receipts with `is_physical_silicon: false` must label provenance as `MODELED` or `SIMULATED`. The `assumptions`, `unmodeled_effects`, and `missing_witness` fields are required.

### Stage 4: Documentation and benchmark tables

In documentation and reports, simulated figures must use the Three-Column Contrast format:

| Real Measured Baseline (`WITNESSED`) | Modeled Projection (`MODELED`) | Theoretical Physical Ceiling (`ROOFLINE`) |
|---|---|---|
| 65.4 tok/s (L4 PCIe, commit `7a1f8c`) | 120.3 tok/s (est. 65% roofline) | 185.0 tok/s (100% memory bus cap) |

Tables containing modeled figures must include an adjacent warning box and state the physical command needed to verify the projection.

### Stage 5: Agent governance and commit stamping

The repository commit and issue tools enforce strict rules:

1. Issue closure refusal: No performance issue may be closed using a simulated receipt. Closure requires a `WITNESSED` receipt on sanctioned fleet hardware.
2. Commit subject typing:
   - Commits containing analytical models or simulated data must use `docs(...)`, `spec(...)`, or `chore(...)` prefixes.
   - Commits using `perf(...)` or `feat(...)` that cite only simulated metrics are rejected. A `perf(...)` subject requires a physical witness.
3. Trailer discipline: The `(fak <leaf>)` trailer on a performance commit asserts verification on physical silicon.

---

## 4. The Six Mandatory Unmodeled Effects

Every simulation or roofline calculation must articulate these six physical effects:

```
                    ┌─────────────────────────────────────────────────────────┐
                    │            Theoretical Roofline (100%)                  │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
 1. Bus & Controller Overhead    ───────────────►├─ (PCIe framing, DMA setup, MMIO)
 2. Memory Contention & Strides  ───────────────►├─ (Bank conflicts, TLB, refresh)
 3. Thermal & Power Throttling   ───────────────►├─ (DVFS clock drop, junction temp)
 4. Workload & Routing Skew      ───────────────►├─ (MoE expert skew, token variance)
 5. OS & Host System Jitter      ───────────────►├─ (CFS, CGO tax, GC safepoints)
 6. Software Stack Abstraction   ───────────────►├─ (CUDA launch, stream sync, copies)
                                                 │
                    ┌────────────────────────────▼────────────────────────────┐
                    │      Real Silicon Achievable Envelope (50–75%)          │
                    └─────────────────────────────────────────────────────────┘
```

1. Bus Protocol and Controller Overhead: PCIe Transaction Layer Packets add 12–16% framing overhead. Direct Memory Access requires descriptor queues and interrupts. Memory-Mapped I/O read operations stall CPU pipelines.
2. Memory Contention and Non-Ideal Strides: Concurrent reads to the same memory bank stall controllers. Non-unit strides waste cache lines. TLB misses on 4KB or 2MB pages cause expensive multi-level page walks. DRAM periodic refresh cycles pause ranks.
3. Thermal and Power Budget Throttling: Sustained matrix compute generates heat. As junction temperature approaches limits ($80^\circ\text{C}$ to $83^\circ\text{C}$), DVFS throttles core clocks. Cards hit board power limits, reducing clocks by 15–30%.
4. Synthetic vs. Real Workload Skew: MoE token routing follows a power-law distribution. Hot experts saturate while cold experts idle. Variable prompt lengths in real agent traffic create batch padding waste.
5. Operating System and Host Jitter: Linux CFS thread preemption causes tail latency spikes. Go CGO transitions add 50–100 ns per invocation. Go runtime GC safepoints pause host dispatch threads.
6. Software Stack Abstraction Tax: Calling `cudaLaunchKernel` costs 3–10 µs of CPU overhead. Stream synchronization forces GPU pipeline bubbles. Unpinned memory forces host staging copies before DMA transfers.

---

## 5. Dual-Track RSI Loop Architecture

Autonomous optimization in the repository operates under a two-track architecture:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                 TRACK 1: ANALYTICAL EXPLORATION                         │
  │                 (Fast, Cheap, Unsafe, Local Workstation)               │
  │                                                                        │
  │  • Roofline math           • Static AST rewriting    • Local CPU runs  │
  │  • Candidate generation    • Search-space pruning    • Mock benchmarks │
  │                                                                        │
  │  OUTPUT: Candidate patches tagged [SIMULATED]                          │
  │  RESTRICTIONS: Cannot land on main, cannot close issues, cannot tag M6 │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼ Candidate passed Track 1 filters
  ┌────────────────────────────────────────────────────────────────────────┐
  │                 TRACK 2: PHYSICAL LANDING GATE                         │
  │                 (Strict, Unforgiving, Fleet Compute Nodes)             │
  │                                                                        │
  │  • Real GPU / NPU hardware • Sanctioned fleet nodes   • Real workloads │
  │  • Six Unmodeled Effects   • Wall-clock net-true test • NVML / metrics │
  │                                                                        │
  │  OUTPUT: Verified commits tagged [WITNESSED]                           │
  │  AUTHORITY: Lands on main, closes issues, updates BENCHMARK-AUTHORITY  │
  └────────────────────────────────────────────────────────────────────────┘
```

### Track 1: Analytical Exploration (Cheap, Fast, Unsafe)

Track 1 is an unconstrained search engine:
- Environment: Runs locally on workstations or CI runners. Requires zero accelerator hardware.
- Tools: Roofline calculations, AST rewriting, mock fixtures, and synthetic traces.
- Output: Candidate diffs and analytical hypotheses tagged `[SIMULATED]` or `[MODELED]`.
- Barriers: Cannot commit to trunk with `perf(...)`, cannot close issues, and cannot update `BENCHMARK-AUTHORITY.md`.

### Track 2: Physical Landing Gate (Strict, Expensive, Unforgiving)

Track 2 is the physical reality monitor:
- Environment: Runs exclusively on sanctioned hardware such as GCP L4, GPU server, or Apple Silicon Metal as cataloged in [fleet-compute-nodes.md](../fleet-compute-nodes.md).
- Tools: Native compiled kernels, end-to-end token timing, and hardware counter metrics like NVML and `powermetrics`.
- Scope: Measures net-true performance against tuned baselines under real operational context.
- Authority: The only track authorized to emit `WITNESSED` receipts, land performance code, and close performance issues.

Candidates failing Track 2 are rejected immediately. The failure is recorded in the loop journal, calibrating Track 1 models against future repeats.

---

## 6. Rule to stick enforcement matrix

Repository gates enforce each requirement mechanically:

| Rule | Enforcement stick | Gate action |
|---|---|---|
| No unadorned simulated numbers | `internal/claimcheck` & `fak claims-lint` | Refuses untagged claims in `CLAIMS.md`; requires `[SIMULATED]` or `[STUB]`. |
| No premature performance closure | DOS verification monitors (`dos verify`) | Refuses issue closure when commit diff lacks a physical test witness. |
| No fake `perf(...)` commits | Git commit-msg hook (`tools/githooks/commit-msg`) | Blocks `perf(...)` commits if the diff touches only mock or simulated files. |
| Receipt schema compliance | Struct validator & CI schema checks | Rejects receipts where `is_physical_silicon == false` but provenance is not `MODELED`/`SIMULATED`. |
| Provenance honesty | Conflation scorecard (`tools/conflation_scorecard.py`) | Flags conflation debt if modeled values are presented as witnessed facts. |
| No self-reported support promotion | Support maturity fence (`internal/shipgate`) | Holds support ladder rung at highest `WITNESSED` level; ignores `MODELED` targets. |
| No silent fallback to llama.cpp | Native inference gate (`docs/native-inference-goal.md`) | Rejects benchmark runs that quietly switch engine to bypass native hurdles. |
| Hardware dispatch compliance | Hardware gate scanner (`fak hwgate-lint`) | Prevents terminal stops claiming local hardware limits; forces fleet dispatch. |

---

## 7. Checklist for contributors and autonomous agents

Before submitting work touching simulated, modeled, or projected results, verify each item:

- Provenance labeled: Non-physical numbers carry the `MODELED` or `SIMULATED` tag.
- Honest titles: Titles avoid quoting simulated figures as facts.
- Warning banners: Commands and test binaries print the simulated warning banner.
- Three-Column Contrast: Tables show Real Baseline versus Modeled Projection versus Theoretical Ceiling.
- Six Unmodeled Effects: The six physical effects are listed and evaluated.
- Envelope respected: Models avoid assuming $>75\%$ roofline efficiency.
- Track 2 witness: Performance claims cite a physical run on sanctioned fleet hardware.
- Commit subject typed: Commits use `docs(...)`, `spec(...)`, or `chore(...)` unless physically witnessed.
