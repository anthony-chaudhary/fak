package radixkv

import (
	"fmt"
	"strings"
)

// regimefence.go fences prefix reuse by the DECODE REGIME the KV bytes were
// produced under (issue #5273). The bare radix Tree matches a prefix on token
// identity ALONE, and namespace.go / binding.go add a tenant salt and the #432
// materialization binding (model / tokenizer / serializer / RoPE / policy /
// admitter). None of those require the two axes this issue names as the live
// hazard the moment KV quantization (#2240) lands and models hot-swap in one
// process: the KV element dtype and the quant mode the span was written under.
//
// A KV span written as f32 is GARBAGE when read back under an int8 or nf4 quant
// mode, and a span from model X is GARBAGE for model Y even at the same dtype.
// Token-only (or salt-only) reuse identity cannot see that, so a request running
// under one regime could be served bytes produced under an incompatible one — a
// correctness and isolation hole, not a perf detail. lmdeploy leaves this
// invariant implicit (process-fixed config); fak ENFORCES it.
//
// This file is the pure-Go fence, no more: a DecodeRegime descriptor over the
// four axes the issue enumerates, a deterministic reuse key derived from it, and
// a fail-closed Match so a prefix is reused ONLY when EVERY axis is equal. It
// owns no bytes, reads no clock, and touches no GPU or network — it is a value
// type a reuse site consults BEFORE a walk, the same posture binding.go takes
// toward the cachemeta key.

// DecodeRegime is the numeric regime a KV prefix was produced under and MUST be
// reused within. Every axis is load-bearing for correctness: a cached prefix is
// structurally unreachable from a request that would decode it under a different
// regime.
type DecodeRegime struct {
	// ModelID is the model whose weights produced the KV. A prefix from one
	// model is never valid for another even on an identical token path.
	ModelID string
	// Dtype is the KV element storage type (e.g. "f32", "f16", "bf16"). KV
	// bytes written as one dtype are unreadable as another.
	Dtype string
	// QuantMode is the KV quantization mode the span was written under (e.g.
	// "none", "int8", "nf4"). Named to avoid coining a compound identifier;
	// this is the axis #2240 makes live.
	QuantMode string
	// RoPEScheme is the RoPE / position scheme (theta, scaling, alignment)
	// the KV positions were rotated under. A prefix rotated under one scheme
	// is misaligned under another.
	RoPEScheme string
}

// MismatchAxis names the FIRST regime axis that diverged (AxisNone on a match,
// AxisIncomplete when either side is unset). It is a typed, closed-vocabulary
// reason a caller can log or branch on rather than a free-text string.
type MismatchAxis string

const (
	// AxisNone means every axis matched: reuse is admitted.
	AxisNone MismatchAxis = ""
	// AxisIncomplete means an axis was unset on one or both regimes, so the
	// match is unprovable and reuse fails closed.
	AxisIncomplete MismatchAxis = "incomplete"
	// AxisModel means the ModelID diverged.
	AxisModel MismatchAxis = "model"
	// AxisDtype means the KV element Dtype diverged.
	AxisDtype MismatchAxis = "dtype"
	// AxisQuant means the KV QuantMode diverged.
	AxisQuant MismatchAxis = "quant"
	// AxisRoPE means the RoPEScheme diverged.
	AxisRoPE MismatchAxis = "rope"
)

// Complete reports whether every axis the issue enumerates is populated. An
// unset axis is an unprovable identity, so a regime with any empty field can
// neither key a stored prefix nor request one — callers fail closed on it.
func (r DecodeRegime) Complete() bool {
	return r.ModelID != "" && r.Dtype != "" && r.QuantMode != "" && r.RoPEScheme != ""
}

// ReuseKey derives a stable, deterministic reuse key from the regime. Equal
// regimes always render the exact same key regardless of construction order,
// and any single-axis change changes the key — so a key comparison is a total
// stand-in for a field-by-field match. The axes are emitted in a FIXED order
// with labeled, separator-delimited segments, so the mapping is one-to-one (no
// two distinct regimes collide on a key).
func (r DecodeRegime) ReuseKey() string {
	return strings.Join([]string{
		"model=" + r.ModelID,
		"dtype=" + r.Dtype,
		"quant=" + r.QuantMode,
		"rope=" + r.RoPEScheme,
	}, ";")
}

// Match reports whether a stored prefix keyed by r may be reused for a request
// keyed by want, returning the FIRST divergent axis as a typed reason. It fails
// CLOSED: if either regime is incomplete the match is unprovable and it refuses
// with AxisIncomplete; otherwise any single divergent axis refuses with that
// axis. Equal, complete regimes match with AxisNone.
func (r DecodeRegime) Match(want DecodeRegime) (bool, MismatchAxis) {
	if !r.Complete() || !want.Complete() {
		return false, AxisIncomplete
	}
	switch {
	case r.ModelID != want.ModelID:
		return false, AxisModel
	case r.Dtype != want.Dtype:
		return false, AxisDtype
	case r.QuantMode != want.QuantMode:
		return false, AxisQuant
	case r.RoPEScheme != want.RoPEScheme:
		return false, AxisRoPE
	}
	return true, AxisNone
}

// Reusable is the boolean face of Match: true only when want is compatible with
// r on every axis (both complete and equal). It is the check a reuse site runs
// BEFORE walking the tree, so a prefix produced under an incompatible regime is
// never even reached.
func (r DecodeRegime) Reusable(want DecodeRegime) bool {
	ok, _ := r.Match(want)
	return ok
}

// RACCompatibilityFence defines the multi-axis compatibility contract that a
// cached KV span must satisfy against the target decode context before reuse (issue #8463).
type RACCompatibilityFence struct {
	// ModelID identifies the model architecture and weights.
	ModelID string `json:"model_id"`
	// TokenizerID identifies the tokenizer vocabulary and encoding scheme.
	TokenizerID string `json:"tokenizer_id"`
	// AttentionRegime specifies the attention mechanism (e.g. "causal", "swa_window_4096", "softcap_50").
	AttentionRegime string `json:"attention_regime"`
	// Dtype is the KV element numerical data type (e.g. "f32", "f16", "bf16", "fp8", "fp8_e4m3", "int4").
	Dtype string `json:"dtype"`
	// QuantMode specifies the KV quantization scheme (e.g. "none", "int8", "int4", "fp8").
	QuantMode string `json:"quant_mode"`
	// RoPEScheme specifies rotary position embedding configuration and scaling (e.g. "default", "yarn_32k", "linear_8k").
	RoPEScheme string `json:"rope_scheme"`
	// MaxContextLen specifies the maximum supported context window length.
	MaxContextLen int `json:"max_context_len,omitempty"`
}

// IsSupported reports whether the fence specifies a supported decode regime.
func (f RACCompatibilityFence) IsSupported() bool {
	if f.AttentionRegime == "unsupported" || f.RoPEScheme == "unsupported" || f.QuantMode == "unsupported" || f.Dtype == "unsupported" {
		return false
	}
	return true
}

// Match verifies exact regime match against want, returning false and a mismatch reason on any divergence.
func (f RACCompatibilityFence) Match(want RACCompatibilityFence) (bool, string) {
	if !f.IsSupported() {
		return false, fmt.Sprintf("unsupported candidate regime: %s", f.AttentionRegime)
	}
	if !want.IsSupported() {
		return false, fmt.Sprintf("unsupported target regime: %s", want.AttentionRegime)
	}
	if f.ModelID != want.ModelID {
		return false, fmt.Sprintf("model mismatch: %q != %q", f.ModelID, want.ModelID)
	}
	if f.TokenizerID != want.TokenizerID {
		return false, fmt.Sprintf("tokenizer mismatch: %q != %q", f.TokenizerID, want.TokenizerID)
	}
	if f.AttentionRegime != want.AttentionRegime {
		return false, fmt.Sprintf("attention regime mismatch: %q != %q", f.AttentionRegime, want.AttentionRegime)
	}
	if f.Dtype != want.Dtype {
		return false, fmt.Sprintf("dtype mismatch: %q != %q", f.Dtype, want.Dtype)
	}
	if f.QuantMode != want.QuantMode {
		return false, fmt.Sprintf("quant mode mismatch: %q != %q", f.QuantMode, want.QuantMode)
	}
	if f.RoPEScheme != want.RoPEScheme {
		return false, fmt.Sprintf("rope scheme mismatch: %q != %q", f.RoPEScheme, want.RoPEScheme)
	}
	if f.MaxContextLen != 0 && want.MaxContextLen != 0 && f.MaxContextLen != want.MaxContextLen {
		return false, fmt.Sprintf("max context len mismatch: %d != %d", f.MaxContextLen, want.MaxContextLen)
	}
	return true, ""
}

// CompatibleWith is an alias for Match for backward compatibility.
func (f RACCompatibilityFence) CompatibleWith(want RACCompatibilityFence) (bool, string) {
	return f.Match(want)
}

// CompatibleWithShift reports whether rotary embeddings can be shifted/re-rotated across positional offsets.
func (f RACCompatibilityFence) CompatibleWithShift(want RACCompatibilityFence) bool {
	matched, _ := f.Match(want)
	if !matched {
		return false
	}
	// Sliding Window Attention (SWA) or local window attention regimes break receptive fields
	// when shifted across window boundaries.
	attn := strings.ToLower(f.AttentionRegime)
	if strings.Contains(attn, "swa") || strings.Contains(attn, "sliding_window") || strings.Contains(attn, "window") {
		return false
	}
	if strings.Contains(attn, "prefix_lm") || strings.Contains(attn, "bidirectional") {
		return false
	}

	// RoPE schemes must support relative rotation.
	rope := strings.ToLower(f.RoPEScheme)
	if rope == "none" || rope == "absolute" || rope == "alibi" || rope == "unsupported" {
		return false
	}
	return true
}
