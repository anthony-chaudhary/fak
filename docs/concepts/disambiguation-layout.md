---
title: "layout names"
description: "Map of exact repository symbols that use the layout root, kept distinct from the broader family label and sibling operations."
---

# layout names

This map positions the current `layout` coverage backlog. Each entry names the exact repository symbol; the family label remains the broader domain and is not a substitute for the symbol.

## Follow-on discovered names

- **`daylayout`** — the exact `layout` symbol; distinct from the broader family label and sibling operations.

## LML units: layoutcontract vs layoutoracle vs the packing checks

The layout-mismatch ledger (LML) landed three units whose machinery sounds alike but
occupies different layers: one DECLARES the packed layout a kernel expects, one COMPUTES
byte geometry from first principles and refuses declared-vs-actual disagreement, and one
CHECKS that a study record's layout obligations match the loader packing path that will
feed the kernel. Catalog rows for every name below live in
`tools/concept_disambiguation_scorecard.data/rows-layout-lml-units.json`.

### layoutcontract (internal/compute/layoutcontract)

Declarative registry of tensor layout contracts for every quantized kernel family
present in-tree (Q4_K, Q8_0, NVFP4, MXFP4). Each `Contract` pins the closed fact set a
loader must satisfy - storage order, interleave width, scale layout, padding rule,
packing path - plus a source ref pinned at `path:line@sha`, so loader/kernel drift shows
up as a stale ref to re-pin instead of a silent runtime transpose or repack. Accessors:
`Get` / `Must` (fail closed with `ErrUnknownFamily`), `All`, `Families`.

*Not* `layoutoracle` (which recomputes byte geometry and refuses mismatches - the oracle
computes, the registry declares), *not* `Layout (tensor)` element ordering, *not* the
newmodel `WeightLayout` checkpoint-packing envelope, and *not* `fak pack` (the OCI image
bundler - "pack" there means archive an image, not pack weights).

### layoutcontract (fak-dev verb)

Read-only CLI surface over the registry: `families` lists registered families,
`show FAMILY` emits the full contract (receipt schema `fak.layoutcontract/1`),
`check FAMILY --observed FILE` compares an observed-layout file under the closed field
vocabulary `storage_order` / `scale_layout` / `interleave_width` / `padding_rule` /
`packing_stage` (receipt `fak.layoutcontract-check/1`; exit 0 match, 1 mismatch,
2 usage-or-unknown-family), and `orphans` reports registered families with no production
call site (receipt `fak.layoutcontract-orphans/1`). No subcommand mutates state.

*Not* the registry itself (it adds no facts), and *not* `fak-dev kernel-oracle` (kernel
performance ranking against ggml test-backend-ops, not layout).

### ErrUnknownFamily (layoutcontract)

Fail-closed sentinel returned by `Get`/`Must` for an unregistered family tag; callers
pin it with `errors.Is`, and the fak-dev verb maps it to exit code 2.

*Not* the studymonitor packing sentinels (`ErrLayoutMismatch` / `ErrLayoutUnknown`) and
*not* the layoutoracle sentinels - those refuse geometry/packing disagreements, while
this one refuses a lookup the registry does not register.

### CheckLoaderPackingAgainstInventory (studymonitor)

Compares the layout obligations a `kernel_inventory` evidence record fills
(`expected_input_layout`, `swizzle_pattern`, `scale_layout`, `packing_source`) against a
declared loader packing path (`PackingPathFacts`) and refuses typed instead of
accepting: nil when every filled obligation matches, `ErrLayoutMismatch` when both sides
declare differing values on an obligated axis, `ErrLayoutUnknown` when a filled value is
outside the closed starter vocabulary or the packing path leaves the axis undeclared.
Witnessed class: LML-07 (ssm_out value-head-tiled GGUF columns fed through the Q8_0
deinterleave).

*Not* the layoutcontract registry (declaration, not comparison) and *not* `layoutoracle`
(byte geometry) - this check only verifies record-vs-packing-path agreement on four
string axes and never inspects bytes.

### PackingPathFacts (studymonitor)

The struct declaring what a loader packing path produces for one kernel family's tensors
at load time, on the same four axes as the record it is checked against
(`ExpectedInputLayout`, `SwizzlePattern`, `ScaleLayout`, `PackingSource`). Every field is
optional: empty means the packing path does not declare that axis, which the check treats
as unverifiable whenever the record obligates it.

Declared packing-path FACTS - *not* the check that consumes them and *not* the registry
contract.

### ErrLayoutMismatch (studymonitor)

Closed sentinel for a record whose filled layout obligation differs from the value the
declared loader packing path produces on the same axis (both sides declare, they differ).
Callers pin it with `errors.Is` and never assert on the wrapped message prose.

Disagreement, *not* unverifiability (`ErrLayoutUnknown`), and *not*
`layoutoracle.ErrGeometryMismatch` (numeric descriptor geometry).

### ErrLayoutUnknown (studymonitor)

Closed sentinel for a filled layout obligation that cannot be verified: the record value
falls outside the closed starter vocabulary, or the packing path leaves the obligated
axis undeclared. Novel values and undeclared packing axes refuse here rather than being
coerced to match.

Unverifiability, *not* disagreement (`ErrLayoutMismatch`); *not*
`layoutoracle.ErrInvalidDescriptor` (a malformed descriptor).

### KnownExpectedInputLayouts (studymonitor)

Starter closed vocabulary for the `expected_input_layout` axis: row-major, tile-swizzled,
column-interleaved, value-head-tiled, grouped. The value-head-tiled/grouped pair names
the Qwen3.5-family ssm_out arrangement (GGUF input columns stored value-head-tiled while
the GDN core emits grouped order, commit 36d3e634c3, ledger row LML-07). Registry readers
stay open to newer values; only the packing check fails closed.

A starter VALUE LIST for one record axis - *not* `KnownScaleLayouts` (the scale_layout
axis) and *not* the layoutcontract `StorageOrder` Go enum.

### KnownScaleLayouts (studymonitor)

Starter closed vocabulary for the `scale_layout` axis: linear (separate row-major scale
array aligned to the weight grid) versus 128x4-tiled (the TensorRT-LLM NVFP4 block-scale
image), aligned with the scale layouts fak registers per kernel family.

Value list for the scale_layout RECORD AXIS - *not* `KnownExpectedInputLayouts` and *not*
`layoutcontract.ScaleLayout` (the Go enum naming the same arrangements as
`ScaleLinear`/`ScaleTiled128x4`/`ScalePackedInRecord`).

### layoutoracle (package)

Independent geometry/stride oracle for tensor descriptors: given a declared descriptor
(what the admission plan expects) and an actual descriptor (what the artifact's real
header carries), it computes byte geometry from first principles - min/max offset over
the strided box times dtype size, overflow-checked - and refuses mismatches before
dispatch. It exists because total-byte agreement does not imply layout agreement: LML-02
witnessed published native-safetensors inventory shapes applied unchanged to converted
GGUF where mHC [4H,24] flattened vs grouped share the same element count but different
axis partitionings. Per the TensorLayoutPlan.ParityCommand doctrine, parity evidence must
come from an independent oracle; a candidate-produced golden is not an independent oracle.

FIRST-PRINCIPLES computation and refusal - *not* the declarative layoutcontract registry,
*not* `Layout (tensor)` metadata, and *not* `fak-dev kernel-oracle` (ranking).

### Descriptor (layoutoracle)

One side of a geometry check - the declared expectation or the artifact's actual header
geometry: `Shape` (outermost-first logical extents; nil/empty is a rank-0 scalar),
`Strides` (ELEMENT strides, not bytes; nil means row-major contiguous, a zero stride
marks a padded/broadcast axis), `DTypeSize` (bytes per element, >= 1), `Groups`
(leading-axis partition: 1 = flat, G > 1 = G blocks of Shape[0]/G along axis 0).

A geometry DECLARATION - *not* a computed result (`ExpectedBytes` derives its span,
`CheckGeometry` compares two). Identical shape/strides/dtype with different Groups
describe different logical layouts (group-major vs flat storage) and refuse despite
matching total bytes.

### ExpectedBytes (layoutoracle)

Computes a descriptor's byte geometry from first principles: with non-negative strides
the minimum offset over the strided box is always 0, so the element span is maxOffset+1
where maxOffset sums (Shape[i]-1)*Strides[i]; zero-size tensors occupy 0 bytes, rank-0
returns DTypeSize, and arithmetic that would overflow int64 errors with `ErrOverflow`
instead of wrapping.

Pure byte-span COMPUTATION on one descriptor - *not* `CheckGeometry` (pairwise) and *not*
a layout fact registry.

### CheckGeometry (layoutoracle)

Refuses a declared-vs-actual `Descriptor` pair whose byte geometry or logical grouping
disagrees, BEFORE dispatch: both sides validate first (an invalid side yields
`*InvalidDescriptor` naming the side), then rank, per-axis shape, per-axis element
stride, dtype size, byte span, and leading-axis grouping are compared in order; the first
disagreement is reported as a `*Mismatch` naming the axis and both values, unwrapping to
`ErrGroupingMismatch` for grouping disagreements and `ErrGeometryMismatch` for all
others. Identical geometry (including nil-vs-explicit row-major strides) passes.

PAIRWISE refusal on numeric geometry - *not* the studymonitor packing check (string
axes) and *not* the fak-dev layoutcontract check subcommand (contract fields over an
observed file).

### ErrGeometryMismatch (layoutoracle)

Sentinel pinning a structural disagreement between two VALID descriptors: rank,
per-axis shape, per-axis element stride, dtype size, or computed byte span. The concrete
`*Mismatch` values carrying those kinds Unwrap to it; callers pin behavior with
`errors.Is`, never rendered text.

Byte-geometry disagreement - *not* `ErrGroupingMismatch` (identical bytes, different
partitioning) and *not* `studymonitor.ErrLayoutMismatch` (string axes).

### ErrGroupingMismatch (layoutoracle)

Sentinel pinning a leading-axis partitioning disagreement: the descriptors agree on every
byte-level property (rank, shape, strides, dtype, span) but declare different group
counts - flat vs grouped storage of the same element set, the LML-02 defect shape.
Compared only after all byte properties agree.

SAME BYTES, different logical layout - *not* `ErrGeometryMismatch` (some byte-level
property differs).

### ErrInvalidDescriptor (layoutoracle)

Sentinel pinning a descriptor that violates the closed policy: negative dim or stride,
non-positive dtype size, inconsistent group partitioning, or stride/rank length mismatch.
The concrete `*InvalidDescriptor` refusal names which side ("declared"/"actual"), which
field (shape/strides/dtype_size/groups), the offending axis (-1 for whole-field), and the
value.

SINGLE-SIDE validity refusal raised before any pairwise comparison - *not* the mismatch
sentinels and *not* `studymonitor.ErrLayoutUnknown` (an out-of-vocabulary record value).

### ErrOverflow (layoutoracle)

Sentinel pinning byte-geometry arithmetic that exceeds int64 (the mulChecked/addChecked
guards in the effectiveStrides/ExpectedBytes paths); huge dims error instead of wrapping.

Arithmetic-range refusal, orthogonal to correctness - *not* any mismatch sentinel (those
require finite, comparable geometry on both sides).
