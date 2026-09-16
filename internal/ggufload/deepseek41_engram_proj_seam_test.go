package ggufload

import (
	"fmt"
	"strings"
	"testing"
)

// TestDeepSeek41EngramProjectionSuffixesResolveToForwardNames is the bounded
// reproduction for a name-seam gap between the GGUF loader and the native V4.1
// forward.
//
// The reference Engram module (inference/model.py at
// deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277)
// carries a projection `self.wkv` plus two per-HC norm parameters `q_weight` and
// `k_weight`. The converted GGUF spells those suffixes `engram_wkv` / `engram_q`
// / `engram_k`.
//
// The reduced native forward (internal/model/v41_forward.go:295,
// v41_forward_engram.go:212) consumes those same projection-side tensors under
// ITS canonical spellings: engram_kv.weight, engram_q_norm.weight,
// engram_k_norm.weight. Today the loader maps the GGUF suffixes to
// model.engram.<L>.{engram_wkv,engram_q,engram_k}.weight, so after a shard load
// the forward's v41AdmitShape lookup for engram_kv.weight cannot find the tensor
// and the declared Engram layer refuses - a loader/forward name mismatch rather
// than a genuine unsupported-model refusal.
//
// This test pins the loader's canonical name for each projection-side suffix to
// the name the forward actually reads, so the two halves of the seam agree. It
// fails on the pre-fix map and passes once the loader resolves the forward
// spellings.
func TestDeepSeek41EngramProjectionSuffixesResolveToForwardNames(t *testing.T) {
	// forwardLeaf is the canonical per-layer suffix internal/model consumes for
	// each Engram projection-side tensor (layerName(l, forwardLeaf)).
	cases := []struct {
		gguf    string // the converted GGUF suffix (vcruz dialect spelling)
		forward string // the canonical name internal/model/v41_forward*.go reads
	}{
		{"engram_wkv.weight", "engram_kv.weight"},
		{"engram_q.weight", "engram_q_norm.weight"},
		{"engram_k.weight", "engram_k_norm.weight"},
	}

	for _, tc := range cases {
		for _, layer := range []int{0, 7} {
			raw := fmt.Sprintf("blk.%d.%s", layer, tc.gguf)
			got, ok := CanonicalTensorNameArch(raw, "deepseek41")
			if !ok {
				t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = ok=false; the projection-side Engram suffix the native forward consumes must resolve", raw)
				continue
			}
			want := fmt.Sprintf("model.engram.%d.%s", layer, tc.forward)
			if got != want {
				t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = %q, want forward-consumed %q", raw, got, want)
			}
			// An Engram projection must never fall through to a generic
			// attention/MLP leaf, or the forward would silently read the wrong
			// tensor instead of the one it names.
			if strings.Contains(got, "self_attn.") || strings.Contains(got, "mlp.") {
				t.Errorf("%q mapped into a generic attention/MLP namespace: %q", raw, got)
			}
		}
	}
}

// TestDeepSeek41EngramForwardNamesAreAdmitted proves the alias is bidirectional:
// a converted file whose projection-side suffixes already carry the forward's
// canonical spelling (engram_kv.weight, engram_q_norm.weight,
// engram_k_norm.weight) must resolve to the SAME canonical leaf, so neither
// dialect spelling silently hard-fails the shard load.
func TestDeepSeek41EngramForwardNamesAreAdmitted(t *testing.T) {
	names := []string{"engram_kv.weight", "engram_q_norm.weight", "engram_k_norm.weight"}
	for _, suffix := range names {
		for _, layer := range []int{0, 7} {
			raw := fmt.Sprintf("blk.%d.%s", layer, suffix)
			got, ok := CanonicalTensorNameArch(raw, "deepseek41")
			if !ok {
				t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = ok=false; the native-forward Engram spelling must resolve", raw)
				continue
			}
			want := fmt.Sprintf("model.engram.%d.%s", layer, suffix)
			if got != want {
				t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = %q, want %q", raw, got, want)
			}
		}
	}
}

// TestDeepSeek41EngramProjectionSuffixesDoNotCollide pins that the two dialect
// spellings of each projection-side tensor converge on ONE canonical leaf. A
// divergence would let a file carrying both spellings write two different targets
// (or the loader could pick one while the forward reads the other).
func TestDeepSeek41EngramProjectionSuffixesDoNotCollide(t *testing.T) {
	pairs := [][2]string{
		{"engram_wkv.weight", "engram_kv.weight"},
		{"engram_q.weight", "engram_q_norm.weight"},
		{"engram_k.weight", "engram_k_norm.weight"},
	}
	for _, pair := range pairs {
		for _, layer := range []int{0, 7} {
			a, okA := CanonicalTensorNameArch(fmt.Sprintf("blk.%d.%s", layer, pair[0]), "deepseek41")
			b, okB := CanonicalTensorNameArch(fmt.Sprintf("blk.%d.%s", layer, pair[1]), "deepseek41")
			if !okA || !okB {
				t.Errorf("layer %d: %s=%q(ok=%v) %s=%q(ok=%v); both dialect spellings must resolve", layer, pair[0], a, okA, pair[1], b, okB)
				continue
			}
			if a != b {
				t.Errorf("layer %d: dialect spellings %s and %s resolve to different canonical names %q vs %q", layer, pair[0], pair[1], a, b)
			}
		}
	}
}
