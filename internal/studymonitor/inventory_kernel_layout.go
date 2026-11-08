package studymonitor

import (
	"slices"
	"strings"
)

// The layout obligation fields on the kernel_inventory evidence record name
// what tensor layout the studied quantized kernel family expects at load time:
// storage order, swizzle permutation, scale placement, and which side packed
// the payload. A compute kernel ported without its loader packing logic forces
// runtime transpose/repack and erases its speedup, so a study records these
// obligations where the loader packing path can later be checked against them.
//
// Every field is an open string enum: the empty string means unknown until a
// study fills it in, and readers must keep accepting values outside the
// starter lists below instead of refusing the record, because newer studies
// introduce new patterns. Entries written before the fields existed read back
// zero-valued and serialize byte-identically (every tag is omitempty), so the
// extension is backward-compatible for existing consumers.

var (
	// KnownExpectedInputLayouts is the starter vocabulary for
	// expected_input_layout, aligned with the storage orders fak registers per
	// kernel family: row-major payloads, tile-swizzled activations, and
	// column-interleaved quant blocks. The Qwen3.5-family ssm_out order pair is
	// named by internal/compute/qwen35_value_head_tiling.go at commit
	// 36d3e634c3: GGUF input columns are stored value-head-tiled while the GDN
	// core emits grouped order; layout ledger row LML-07 (lessons PHL-045 and
	// PHL-127) witnessed the cost of riding tiled columns through the Q8_0
	// deinterleave.
	KnownExpectedInputLayouts = []string{"row-major", "tile-swizzled", "column-interleaved", "value-head-tiled", "grouped"}
	// KnownSwizzlePatterns is the starter vocabulary for swizzle_pattern, named
	// after the transform pairs shipped in-tree: compute.SwizzleNVFP4Scales and
	// bench.RepackQ4KInterleaved.
	KnownSwizzlePatterns = []string{"none", "nvfp4-scale-128x4", "q4k-column-interleave"}
	// KnownScaleLayouts is the starter vocabulary for scale_layout: linear
	// row-major scales versus the 128x4 tiled block-scale layout.
	KnownScaleLayouts = []string{"linear", "128x4-tiled"}
	// KnownPackingSources is the starter vocabulary for packing_source: which
	// side produced the packed layout the kernel expects.
	KnownPackingSources = []string{"loader-native", "runtime-repack", "upstream-prepacked"}
)

// HasLayoutObligation reports whether the record carries at least one filled
// layout obligation field. Whitespace-only counts as unfilled.
func (evidence InventorySourceEvidence) HasLayoutObligation() bool {
	return strings.TrimSpace(evidence.ExpectedInputLayout) != "" ||
		strings.TrimSpace(evidence.SwizzlePattern) != "" ||
		strings.TrimSpace(evidence.ScaleLayout) != "" ||
		strings.TrimSpace(evidence.PackingSource) != ""
}

// UnknownLayoutValues returns the names of the filled layout obligation fields
// whose values fall outside the starter vocabularies, in field declaration
// order. It never refuses: an unfamiliar value is data a newer reader
// understands, not an error.
func (evidence InventorySourceEvidence) UnknownLayoutValues() []string {
	var unknown []string
	if value := strings.TrimSpace(evidence.ExpectedInputLayout); value != "" && !slices.Contains(KnownExpectedInputLayouts, value) {
		unknown = append(unknown, "expected_input_layout")
	}
	if value := strings.TrimSpace(evidence.SwizzlePattern); value != "" && !slices.Contains(KnownSwizzlePatterns, value) {
		unknown = append(unknown, "swizzle_pattern")
	}
	if value := strings.TrimSpace(evidence.ScaleLayout); value != "" && !slices.Contains(KnownScaleLayouts, value) {
		unknown = append(unknown, "scale_layout")
	}
	if value := strings.TrimSpace(evidence.PackingSource); value != "" && !slices.Contains(KnownPackingSources, value) {
		unknown = append(unknown, "packing_source")
	}
	return unknown
}
