# Heterogeneity-aware microscaling: bounded interoperability evaluation

Issue: [#6248](https://github.com/anthony-chaudhary/fak/issues/6248), child of #6221.  
Contract: `internal/microscaleeval` (`fak.microscale-eval/v1`).

## Verdict first

**Integrate the neutral descriptor/adjudicator; delegate AdaMX execution; do not claim a native AdaMX kernel.** The contract can identify an exact declared OCP/vendor profile as `supported`, route a known heterogeneous descriptor to an explicitly named adapter as `delegate`, and return `unsupported` for unknown schemas, families, malformed pins, or known formats outside the runtime envelope. It never silently treats AdaMX, OCP MXFP4, and NVIDIA FP4 block scaling as interchangeable.

No model weights or AdaMX implementation artifact accompanied the paper sources inspected for this issue, and this leaf does not implement a kernel. Consequently all research comparisons below are **modeled/paper-reported**, not locally observed. Hardware performance and quality remain unclaimed; a future observed claim must include a run digest and hardware fingerprint accepted by the Go contract.

## Pinned research inputs

| Input | Pin | Local read-back |
|---|---|---|
| AdaMX paper | arXiv `2608.03867v1`, published 2026-08-04, “Heterogeneity-Aware Microscaling for Efficient Low-Bit LLM Inference” | PDF SHA-256 `9ba23a489f5198f97c930da6845dca9deb7898f8864cbe2a35ffdd7ed0fedaf7`; arXiv Atom response SHA-256 `2b05030099371834ac52d30a9a34a7657b397e397d91139eedf1c19e5a1c1f0d` |
| OCP MX comparison basis | OCP Microscaling Formats (MX) v1.0 as cited by the paper and NVIDIA documentation | semantic pin `OCP-MX-v1.0`; no redistributed copy |
| Vendor runtime comparison | NVIDIA cuBLAS 13.3 documentation, section “16/32-Element 1D Block Scaling for FP8 and FP4 Data Types”, retrieved 2026-08-10 | HTML SHA-256 `40c41df0b9fab5bc7e3d3c6ce963794938d6f85845ebe10a3271f58f9c8b7fcb` |

The hashes identify the exact bytes evaluated, not an endorsement or executable artifact. Reproduction downloads and extracted text stay in the gitignored `_scratch/issue-6248/` directory.

## Schema diff: standard and vendor assumptions versus AdaMX

| Axis | OCP MX / MXFP4 baseline | NVIDIA cuBLAS 13.3 documented FP4 block scaling | AdaMX paper (`2608.03867v1`) | Contract consequence |
|---|---|---|---|---|
| Shared scale | MX uses a shared scale for a block; MXFP4 is E2M1 elements with E8M0 shared scale | FP4 mode uses 16-element blocks with `UE4M3` scale; docs describe 32-element `UE8M0` for FP8 and say mixed precisions are unsupported | Builds from MX-style shared scaling but adds block metadata and adaptive recovery/format choices | `ScaleFormat`, `BlockSize`, and `Family` are independent, exact-match fields |
| Element format | One declared element format per MX format (MXFP4 uses E2M1) | FP4 narrow values; runtime mode fixes the supported precision/scale combination | Per-block format choice; weights select among FP/INT behavior while activations use an operand-specific design | Weight and activation `Operand` descriptors are separate; `MixedOperands` must be explicit |
| Precision recovery | Baseline MXFP4 has no AdaMX per-block recovery selector | No AdaMX recovery selector is documented | Selects a precision-recovery scheme per block; paper discusses outlier and microexponent recovery | `Recovery` is independent of element format and `PerBlockMode` is explicit |
| Block size | MXFP4 comparison uses 32-value blocks | Documented FP4 scale mode is vector-16 `UE4M3` | Paper evaluates two operating points: 16-value high-accuracy and 32-value lower-EBW | No block-size inference or fallback |
| Runtime availability | Open interchange format, not proof that a particular runtime executes every combination | Concrete native profiles are constrained combinations; docs explicitly reject mixed precision in this mode | Paper describes a custom 22 nm FD-SOI accelerator prototype and simulations, not a generally available cuBLAS profile | Known AdaMX descriptors become `delegate` only when a delegate is named; otherwise `unsupported` |

### Bounded format identifiers

The leaf intentionally recognizes only:

- `ocp-mx-v1`
- `nvidia-nvfp4`
- `adamx-paper-v1`

These IDs are comparison identities, not a new serialization format. Unknown families and unknown schema versions refuse explicitly. A runtime profile is exact: empty fields are not wildcards, and family, scale, block size, both operands, and per-block behavior must all match.

## Full research evaluation (paper-reported, not observed here)

### Mechanism

The paper identifies two kinds of quantization heterogeneity:

1. **Across blocks:** the preferred element format and recovery scheme vary by block.
2. **Across operands:** weights and activations benefit from different encoding choices.

AdaMX therefore selects the precision-recovery scheme per block and the representation per operand. Its two block-size points trade metadata/equivalent-bit-width against accuracy: block size 16 is the high-accuracy point; block size 32 is the lower-EBW/storage point. The paper's hardware design adds a decoder, compute unit, and quantization logic around this representation.

### Quality envelope reported by the authors

The following numbers are copied as paper claims and remain `modeled` in this repository:

- Models evaluated span **3B to 70B** parameters.
- The abstract reports removal of **83% of MXFP4 accuracy loss on commonsense** and **82% on MMLU**.
- Against NVFP4 it reports removal of **43% of commonsense loss** and **27% of MMLU loss**.
- For multimodal Gemma-4 12B, it reports beating MXFP4 on four vision-language benchmarks and retaining **up to 96% of FP16 accuracy**.

Those figures are not converted into a fak gain claim: this issue did not independently reproduce datasets, checkpoints, calibration, evaluation harness, or kernels. “Up to” remains “up to”; model/benchmark averages are not generalized beyond the paper's envelope.

### Hardware/performance envelope reported by the authors

The paper reports a **22 nm FD-SOI accelerator prototype**. Against its otherwise-identical MXFP4 accelerator baseline with FP4-only multipliers, the abstract reports about **1% system-energy overhead** for AdaMX; at the lower-EBW point it reports reduced storage/memory footprint and energy while remaining more accurate than that baseline.

This is not an observed CUDA/GPU result. The relevant sanctioned lab nodes cannot turn a paper-only custom decoder/accelerator into a faithful hardware witness without the executable artifact, recipe, model pins, and reference vectors. Dispatching an unrelated MXFP4 GEMM would answer a different question and would falsely substitute vendor native support for AdaMX. The contract therefore requires:

- observed source/run identity;
- SHA-256 of the machine-readable run witness; and
- hardware/runtime fingerprint.

Without all three, `observed` is rejected rather than downgraded or invented.

## Provenance contract

Every adjudication preserves four separate immutable pins:

1. **Artifact:** encoded tensor/container or fixture bytes.
2. **Recipe:** quantizer/calibration/selector configuration.
3. **Runtime:** decoder/kernel/compiler implementation.
4. **Model:** source model/checkpoint.

Each pin requires `id`, `revision`, and lowercase SHA-256. This prevents a paper ID, model name, or CUDA version from standing in for the complete chain. Quality/performance evidence is independently typed:

- `modeled`: cited source only; it cannot carry a hardware witness.
- `observed`: source plus run SHA-256 plus hardware fingerprint.

## Named witness

`TestNamedWitnessProducesClosedOutcomes` reads three public in-code fixtures through the real adjudicator:

1. exact OCP MXFP4 profile → `supported/native_profile_match`;
2. AdaMX-like per-block/per-operand descriptor with a named adapter → `delegate/heterogeneous_runtime_required`;
3. unknown family → `unsupported/unsupported_format`.

Additional tests prove unknown schemas and incomplete pins refuse, incomplete observed evidence cannot masquerade as measurement, and NVIDIA-style FP4 does not silently match OCP MXFP4.

Reproduce from WSL:

```bash
cd /mnt/c/work/fak
go test ./internal/microscaleeval -run 'TestNamedWitnessProducesClosedOutcomes|TestUnknownSchemaAndIncompletePinsRefuse|TestObservedCannotBeClaimedWithoutRunAndHardwareWitness|TestVendorFP4DoesNotSilentlyMatchOCP' -count=1
```

Repository-scoped acceptance:

```powershell
fak validate --mine internal/microscaleeval/contract.go --mine internal/microscaleeval/contract_test.go --mine docs/research/quantization/heterogeneous-microscaling.md
fak buildcheck --vet ./internal/microscaleeval
```

## Integration decision

- **Integrate:** the typed descriptor, provenance chain, evidence typing, and exact capability adjudication.
- **Delegate:** AdaMX decode/execute only to a specifically pinned adapter/runtime that advertises the heterogeneous behavior.
- **Unsupported:** unknown schema/family, malformed metadata, incomplete provenance/evidence, or a known format with neither exact native support nor a declared delegate.

This is deliberately neutral: it preserves OCP and vendor identities and does not select a universal quantization winner.
