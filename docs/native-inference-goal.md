---
title: "Fak-native inference doctrine: own the engine and beat the reference"
description: "The canonical boundary for local inference work: fak-native is the product and performance path; llama.cpp is an explicit benchmark, diagnosis, interoperability, or borrowing reference, never a silent fallback."
---

# Fak-native inference is the product path

Audience: maintainers and agents implementing, optimizing, benchmarking, or reviewing local
inference in fak.

Authority: this is the canonical execution-engine doctrine. Architecture and performance docs
should link here. Benchmark indexes, issues, and runtime docs should do the same instead of
restating a different engine policy.

> **TL;DR:** Build and optimize local inference inside fak. Use llama.cpp deliberately as a
> reference, never quietly as the implementation.

## The invariant, in plain language

<!-- native-engine-doctrine:product-path -->
**Product path:** fak-native is the product and performance path for local inference.

<!-- native-engine-doctrine:matched-envelope -->
**Performance target:** In matched, quality-constrained envelopes, fak-native must beat llama.cpp.

<!-- native-engine-doctrine:explicit-reference-only -->
**Reference boundary:** llama.cpp is permitted only when explicitly selected for benchmarks, parity/reference diagnosis, migration/interoperability, or ego-free borrowing.

<!-- native-engine-doctrine:no-silent-fallback -->
**Fallback boundary:** fak never selects llama.cpp as a fallback for native or performance work.

<!-- native-engine-doctrine:owned-stack -->
**Ownership reason:** fak must understand and control the full engine stack so higher-order gains compose. That stack covers kernels and memory, scheduling and cache, plus adaptation and operations.

These are engineering invariants, not an unsupported claim that fak currently wins every
comparison. A current performance or efficiency claim still needs a scoped row in
[`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md), with the engine and matched envelope
named. When the evidence says fak-native is behind, say so and keep the gap as native product
work.

New native-performance work prefers Qwen3.8.
Qwen3.6 requires an explicit task-specific exception. Allowed exceptions include regression or compatibility work, historical comparison, and hardware/artifact constraints.
Preserve historical Qwen3.6 artifacts; do not rename or rewrite them as Qwen3.8 evidence.

## What fak-native means

The model executes **inside the fak-owned inference path**:

- fak loads or interprets the model artifact and owns the model architecture path.
- fak owns the memory layout and lifecycle that its runtime exposes.
- fak chooses and coordinates CPU, CUDA, Metal, or other compute-HAL work.
- fak owns scheduling, batching, and placement. It also owns cache identity, reuse,
  eviction, and recovery.
- fak emits the engine identity and the evidence needed to attribute a result.

Calling a device library such as a vendor BLAS from a fak backend does not surrender the engine.
The deciding question is whether fak owns the execution plan and model state. It must also own
the lifecycle, observability, and replacement boundary. A llama.cpp process or library executing
the model is an external runtime even when fak launched it, fronts it, or consumes its output.

| Term | Meaning |
|---|---|
| fak-native | Model loading and inference execute through fak's in-kernel model and compute path. This is the local product and performance path. |
| native backend | A compute-HAL implementation selected inside fak-native, such as CPU, CUDA, or Metal. Changing a native backend does not replace the inference engine. |
| explicit external reference runtime | llama.cpp or another independently implemented engine selected explicitly for a bounded benchmark, comparison, parity/reference diagnosis, migration/interoperability task, or ego-free study and borrowing. |
| gateway/provider upstream | A remote or separately served model behind the fak gateway. This can be a supported operating mode, but it is external inference and is never evidence for fak-native performance. |

## Why engine ownership matters

An isolated tokens-per-second number is not the whole product. fak's advantage has to compose
across the complete execution path:

| Owned surface | What fak can improve because it owns the surface |
|---|---|
| Kernels | Select precision, fuse work, change tiling, exploit device features, and attribute a bottleneck to the real operation. |
| Memory | Keep weights and state resident, choose layouts, budget capacity, move data deliberately, and avoid opaque copies. |
| Scheduling | Batch compatible work, coordinate many sessions, place work by capacity and locality, and make queueing visible. |
| Cache | Give KV and prefix state stable identity, prove reuse legality, evict exact spans, and join cache state to the agent session. |
| Adaptation | Apply bounded model or policy adaptations with explicit provenance, lifetime, rollback, and compatibility. |
| Operations | Expose one lifecycle for startup, health, evidence, failure, recovery, and resource accounting. |

If an external runtime silently replaces the native path, these gains stop composing. fak may
observe the outside engine. It cannot safely treat opaque scheduling and memory as kernel-owned
state. The same limit applies to cache and adaptation state. A local improvement can therefore
make one benchmark look better while erasing the higher-order product fak is trying to build.

Ownership does not mean refusing established ideas or libraries. It means fak keeps the
decision boundary and integrates borrowed mechanisms into a path it can inspect, test, replace,
and operate.

## The matched-envelope rule

“Beat llama.cpp” is meaningful only inside a declared comparison envelope. A matched result
names, or explicitly reconciles, every row below.

| Axis | Required comparison state |
|---|---|
| Model | Architecture, artifact, tokenizer, and prompt template. |
| Numeric | Quantization, precision, and the quality or correctness floor. |
| Hardware | Device placement, thread budget, and memory budget. |
| Request | Prompt length, context length, decode length, and output stopping rule. |
| Load | Concurrency, batching, scheduling, and request arrival shape. |
| State | Cold or warm state, weight residency, prefix/KV state, and cache policy. |
| Outcome | Time to first token, decode rate, aggregate throughput, memory, energy, cost, or end-to-end task completion. |

An unmatched comparison can diagnose a direction, but it cannot close a native performance
claim. Equal tokens per second with worse correctness does not win. A fast external engine plus
fak's gateway is not a fak-native result. A native result that is slower today remains valuable
evidence because it names the gap that native work must retire.

## The four explicit llama.cpp uses

| Explicit use | What is allowed | What the result means |
|---|---|---|
| Benchmark | Run a pinned llama.cpp build as the tuned baseline in a matched envelope. | A comparison bar. It proves only what the measured rows and envelope state. |
| Parity/reference diagnosis | Compare tokenization, logits, greedy tokens, tensor transforms, or intermediate values to localize a correctness difference. | Reference evidence. Passing or failing narrows the defect; it does not transfer engine ownership. |
| Migration/interoperability | Read or produce compatible artifacts, validate a migration, or explicitly front a llama.cpp service while moving a workload. | Compatibility evidence. The run remains external inference unless the model executes inside fak. |
| Ego-free borrowing | Study a useful algorithm, kernel pattern, format decision, or operational lesson; preserve attribution and license obligations; implement and prove the useful part in fak-native. | Prior art and implementation input. The borrowed idea becomes native only after fak owns and witnesses its path. |

Every external use is selected deliberately and labeled in the command, configuration, result,
or receipt. “Available on this host,” “faster in the last run,” or “native support is missing”
does not authorize an automatic substitution.

## Vendor accelerator runtimes

Closed vendor accelerator runtimes execute outside the fak-native engine.
This classification applies when FLM or OGA/Lemonade selects a closed runtime;
the frontend name alone does not identify the execution engine.

Select one of the same four explicit uses for each vendor-runtime run:

- Benchmark: measure an external baseline in a matched envelope.
- Parity/reference diagnosis: compare outputs to locate a correctness difference.
- Migration/interoperability: validate compatible artifacts or explicitly front an external service.
- Ego-free borrowing: study mechanisms with attribution and license compliance, then implement and witness them inside fak.

Vendor-runtime receipts name the actual engine, device, and explicit use.
NPU placement does not convert external execution into fak-native performance evidence.
fak never automatically substitutes a vendor accelerator runtime for native execution.
When native support is unavailable, return an explicit unsupported result; an operator
may separately select an external comparison or interoperability run.

## Default and failure behavior

For work classified as native inference or native performance:

| Situation | Required behavior |
|---|---|
| Whole-engine selection is omitted. | Resolve to fak-native. |
| A native CPU, CUDA, or Metal backend is selected. | Stay inside the fak-owned compute boundary. |
| The model, backend, device, or memory envelope is unsupported. | Return an explicit unsupported, unavailable, or not-yet result. |
| Native launch fails. | Keep the native failure and its evidence. |
| The operator selects llama.cpp. | Reclassify the run as benchmark, diagnosis, interoperability, or borrowing work. |

Do not add `auto`, recovery, convenience, or “best available” behavior that turns a native
request into llama.cpp execution without the operator making that change. A loud native gap is
actionable product evidence. A quiet substitution is a false success.

## Review gate

Before accepting a native implementation or performance claim, ask:

1. Did the model execute inside fak?
2. Does the command, result, or receipt name the engine?
3. If llama.cpp ran, which of the four explicit uses justified it?
4. Is the comparison envelope matched and quality-constrained?
5. If fak-native could not run, did the path report that gap instead of changing engines?
6. If an external idea was borrowed, where is the fak-owned implementation and its native
   witness?

The shortest review question is: **Did the model execute inside fak, and does the receipt name
that engine?** If not, classify the run as external comparison or interoperability evidence and
do not use it to close a fak-native performance claim.

## Deterministic docs guard

This docs-only guard pins the five required invariant sentences and the seven inbound links
that make the doctrine discoverable. Run it from the repository root:

```powershell
$canonical = 'docs/native-inference-goal.md'
$requiredPhrases = @(
  'fak-native is the product and performance path for local inference.'
  'In matched, quality-constrained envelopes, fak-native must beat llama.cpp.'
  'llama.cpp is permitted only when explicitly selected for benchmarks, parity/reference diagnosis, migration/interoperability, or ego-free borrowing.'
  'fak never selects llama.cpp as a fallback for native or performance work.'
  'fak must understand and control the full engine stack so higher-order gains compose. That stack covers kernels and memory, scheduling and cache, plus adaptation and operations.'
)
$requiredPhrases += @(
  'Closed vendor accelerator runtimes execute outside the fak-native engine.'
  'Vendor-runtime receipts name the actual engine, device, and explicit use.'
  'fak never automatically substitutes a vendor accelerator runtime for native execution.'
)
$requiredLinks = [ordered]@{
  'README.md'                  = 'docs/native-inference-goal.md'
  'AGENTS.md'                  = 'docs/native-inference-goal.md'
  'llms.txt'                   = 'docs/native-inference-goal.md'
  'docs/architecture.md'       = 'native-inference-goal.md'
  'docs/performance.md'        = 'native-inference-goal.md'
  'docs/index.md'              = 'native-inference-goal.md'
  'docs/benchmarks/README.md'  = '../native-inference-goal.md'
}
$failures = @()
$body = (Get-Content -Raw -LiteralPath $canonical) -replace '(?s)\x60{3}.*?\x60{3}', ''
foreach ($phrase in $requiredPhrases) {
  if (-not $body.Contains($phrase)) { $failures += "missing phrase: $phrase" }
}
foreach ($entry in $requiredLinks.GetEnumerator()) {
  if (-not (Select-String -Quiet -SimpleMatch -LiteralPath $entry.Key -Pattern $entry.Value)) {
    $failures += "missing link: $($entry.Key) -> $($entry.Value)"
  }
}
if ($failures.Count) {
  $failures | ForEach-Object { Write-Error $_ }
  exit 1
}
"PASS native-inference-doctrine phrases=$($requiredPhrases.Count) inbound_links=$($requiredLinks.Count)"
```

Expected output:

```text
PASS native-inference-doctrine phrases=8 inbound_links=7
```

Run the repository link and index gates after this guard; they prove the linked target resolves,
while this guard proves the required distinctions and entry-point projections remain present.

## Related authorities

- [External system architecture](architecture.md) places native and external execution inside
  the larger agent-kernel boundary.
- [Performance outcomes](performance.md) routes a claimed outcome to its current witness.
- [Benchmark methodology](benchmark-methodology.md) defines authority, generation, and
  reproduction rules.
- [Benchmark sheets](benchmarks/README.md) index the current native/reference result packets.
- [Adding a model](new-model-playbook.md) applies the native-engine boundary to model support.
