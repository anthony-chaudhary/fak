# Mixed-precision assignment contract

`mixedprecision/v1` is a neutral metadata contract for deterministic per-module
precision maps. It does not choose a quantization method, execute a kernel, or
claim that lower modeled weight bits preserve model quality.

## Research/model scope

The contract represents the common layerwise seam described by public mixed
precision work and runtime interfaces without adopting a fak-only artifact:

- HAWQ-V2 uses Hessian-aware layer sensitivity to allocate a mixed-precision
  bit budget: <https://arxiv.org/abs/1911.03852>.
- AWQ motivates activation-aware protection of salient weights and is recorded
  as a pinned recipe, not treated as a universal winner:
  <https://arxiv.org/abs/2306.00978>.
- vLLM documents quantization as a runtime-specific compatibility matrix, so a
  combination can be typed `delegate` rather than silently called supported:
  <https://docs.vllm.ai/en/latest/features/quantization/>.

A descriptor pins three independent inputs (`id`, exact `version`, SHA-256, and
optional source URI): the model artifact, assignment recipe, and runtime
implementation. The caller supplies an explicit support matrix for those exact
versions. Unknown versions and undeclared triples are refused.

Rules are exact names or terminal-`*` prefixes. Module names and precision IDs
are trimmed and case-normalized, assignments are sorted by canonical module
name, and overlapping rules are ambiguous by design. An unmatched module is
one of three visible policies: refuse, delegate, or assign a declared fallback.
The returned budget reports module and parameter coverage, fallback exposure,
weighted assigned bits, and average assigned bits. It excludes scales,
zero-points, padding, metadata, activation/KV state, and runtime workspace.

## Quality and hardware evidence

`modeled` and `observed` are disjoint evidence types:

- `modeled` requires its model basis and forbids a hardware envelope or sample
  count. The fixture's weight-bit total is modeled accounting, not a measured
  memory, speed, or quality result.
- `observed` requires accelerator, runtime, driver, OS, dataset digest, and a
  non-zero sample count. The package validates and preserves caller-supplied
  measurements; it never manufactures them.

No observed hardware or quality result ships with this leaf. Consequently this
contract makes no throughput, latency, memory-footprint, perplexity, accuracy,
or task-quality claim. A future lab run can attach observed evidence without
reclassifying a model estimate as measurement.

## Witness

`contract_test.go` independently reads three JSON fixtures and proves typed
supported, unsupported, and delegated outcomes. A deterministic 500-case
property witness proves canonicalization, parameter coverage, weighted-bit
accounting, and stable IDs; separate cases prove refusal/delegation for
ambiguous and unmatched names, explicit fallback accounting, exact-version
pins, strict unknown-field handling, and modeled/observed separation.
