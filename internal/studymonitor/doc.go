// Package studymonitor maintains the monitored-repositories registry and the
// study-inventory contract: which source classes a repository study must
// cover, how the generated inventory map proves them, and where a tree with
// GPU kernels owes a per-kernel inventory (kernel_inventory).
//
// Layout-obligation check policy (layout-guard ticket lg-kernel-inventory-fields,
// ledger row LML-07):
//
// A kernel_inventory evidence record may name the tensor layout its studied
// kernel family expects at load time (expected_input_layout, swizzle_pattern,
// scale_layout, packing_source). Registry readers stay open: Validate accepts
// values outside the starter vocabularies because newer studies introduce new
// patterns, and entries written before the fields existed read back
// zero-valued and serialize byte-identically.
//
// The packing check fails closed instead. CheckLoaderPackingAgainstInventory
// compares a record's filled layout obligations against a declared loader
// packing path (PackingPathFacts) and refuses with a typed sentinel rather
// than silently accepting:
//
//   - ErrLayoutMismatch: both sides declare a value on an obligated axis and
//     they differ — the LML-07 class, ssm_out value-head-tiled GGUF columns
//     fed through the Q8_0 deinterleave packing path;
//   - ErrLayoutUnknown: a filled obligation cannot be verified — the record
//     value is outside the closed vocabulary, or the packing path leaves the
//     obligated axis undeclared.
//
// An empty record carries no obligation and checks clean. When several axes
// fail, the first in field declaration order is reported. Callers pin the
// refusal class with errors.Is and never assert on the wrapped message prose.
package studymonitor
