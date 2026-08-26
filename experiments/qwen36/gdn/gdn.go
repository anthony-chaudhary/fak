// Package gdn holds the pieces the host-runnable Qwen3.6-27B Gated-DeltaNet (GDN)
// experiments under experiments/qwen36/ share: the real linear_attn layer shapes, the
// activation/norm primitives, the parallel f32 GEMM, the seeded layer-weight fixture,
// and the delta-rule recurrent scan itself.
//
// The math here is copied verbatim from the trunk kernels it reproduces —
// internal/model/metal_prefill_hybrid_core.go:202-246 (the prefill twin of
// qwen35.go:linearAttnStep) and qwen35.go's norm helpers — so the experiments measure
// the trunk's numerics rather than a paraphrase of them. Each sibling command supplies
// its own PERTURBATION (scan reduction order, f16 state storage, weight quantization)
// and its own reporting; only the kernel and its perturbation switch live here.
//
// Nothing in here loads a model artifact, touches a GPU, or shells out. It is pure,
// deterministic f32 arithmetic: every reduction runs in a fixed serial order inside a
// worker that owns a disjoint slice of the output, so results do not depend on
// GOMAXPROCS or on goroutine scheduling.
package gdn

import (
	"encoding/json"
	"io"
	"strings"
)

// Qwen3.6-27B Gated-DeltaNet layer dims (the 48 linear_attn layers), sourced from the
// in-tree fixture internal/model/quant_q4k_resident_test.go.
const (
	Hidden27B = 5120 // HiddenSize H of the real 27B; some experiments sweep smaller H
	NK        = 16   // LinearNumKeyHeads
	NV        = 48   // LinearNumValueHeads
	KHd       = 128  // LinearKeyHeadDim
	VHd       = 128  // LinearValueHeadDim
	K         = 4    // ssm.conv_kernel (Qwen3-Next)
)

// Derived per-layer widths. All three are independent of the hidden size H.
const (
	KeyDim  = NK * KHd          // 2048
	ValDim  = NV * VHd          // 6144
	ConvDim = 2*KeyDim + ValDim // 10240
)

// EmitJSON writes v to w as 2-space-indented JSON (json.Encoder appends the trailing
// newline). It is the machine-readable `-json` output every qwen36 experiment emits;
// callers deliberately discard the error, matching the pre-extraction `_ = enc.Encode(v)`.
func EmitJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

// Verdict joins complete verdict clauses with spaces. Experiment reports use this to keep
// long evidence statements readable without duplicating string-concatenation scaffolding.
func Verdict(clauses ...string) string {
	return strings.Join(clauses, " ")
}
