package ggufload

import (
	"fmt"
	"strings"
	"testing"
)

// deepSeek41MHCForwardLeaf names the per-layer mHC tensors the reduced native
// V4.1 forward both ADMITS (internal/model/v41_forward.go:659-665, v41StageMHC)
// and CONSUMES (internal/model/v41_forward.go:864-866). They are the loader's
// contract target: after a shard load the forward's v41AdmitShape lookup must
// find every one of them under these exact canonical spellings.
var deepSeek41MHCForwardLeaf = []string{
	"mhc.mixes.weight",
	"mhc.base",
	"mhc.scale",
}

// TestDeepSeek41GGUFV41MHCTapsResolveToForwardNames pins the loader-side seam for
// the V4.1 hyper-connection (mHC) tensors. Before the fix the "deepseek41"
// canonical map carried ONLY the GUESSED hc_attn_*/hc_ffn_* tap arms, which
// re-nest into a per-layer hc.<leaf>.weight namespace that the native forward
// never reads. The published converter instead emits the mHC coefficient block
// under the forward-consumed mhc.* names, so a declared V4.1 layer refused on a
// naming mismatch rather than a genuine unsupported-model refusal.
//
// RED before the fix: CanonicalTensorNameArch("blk.0.mhc_base.weight",
// "deepseek41") = ok=false; the forward's required leaf is never produced.
func TestDeepSeek41GGUFV41MHCTapsResolveToForwardNames(t *testing.T) {
	dialects := map[string]string{
		"mhc_mixes.weight": "mhc.mixes.weight",
		"mhc_base.weight":  "mhc.base",
		"mhc_scale.weight": "mhc.scale",
	}
	for _, layer := range []int{0, 7} {
		for suffix, wantLeaf := range dialects {
			ggufName := fmt.Sprintf("blk.%d.%s", layer, suffix)
			got, ok := CanonicalTensorNameArch(ggufName, "deepseek41")
			if !ok {
				t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = ok=false; the forward admits %q and the shard load must resolve it",
					ggufName, wantLeaf)
				continue
			}
			want := layerName(layer, wantLeaf)
			if got != want {
				t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = %q, want forward-consumed %q",
					ggufName, got, want)
			}
		}
	}
}

// TestDeepSeek41GGUFV41MHCForwardNamesAreAdmitted proves the forward-consumed
// spellings themselves also resolve, so a file already carrying the canonical
// mhc.* leaves (rather than the converter dialect) does not hard-fail the load.
func TestDeepSeek41GGUFV41MHCForwardNamesAreAdmitted(t *testing.T) {
	for _, layer := range []int{0, 7} {
		for _, leaf := range deepSeek41MHCForwardLeaf {
			ggufName := fmt.Sprintf("blk.%d.%s", layer, leaf)
			got, ok := CanonicalTensorNameArch(ggufName, "deepseek41")
			if !ok {
				t.Errorf("forward-consumed leaf %q did not resolve under deepseek41", ggufName)
				continue
			}
			if want := layerName(layer, leaf); got != want {
				t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = %q, want %q", ggufName, got, want)
			}
		}
	}
}

// TestDeepSeek41GGUFV41MHCTapsDoNotCollide proves the three mHC coefficient
// leaves stay distinct and never collapse onto the legacy hc_* tap namespace or
// a generic attention/MLP name (which would let an mHC coefficient block fall
// through to the wrong forward).
func TestDeepSeek41GGUFV41MHCTapsDoNotCollide(t *testing.T) {
	seen := map[string]string{}
	for _, suffix := range []string{"mhc_mixes.weight", "mhc_base.weight", "mhc_scale.weight"} {
		got, ok := CanonicalTensorNameArch("blk.0."+suffix, "deepseek41")
		if !ok {
			t.Errorf("mHC suffix %q did not resolve under deepseek41", suffix)
			continue
		}
		if prev, dup := seen[got]; dup {
			t.Errorf("mHC suffixes %q and %q collide on canonical name %q", prev, suffix, got)
		}
		seen[got] = suffix
		if strings.Contains(got, "self_attn.") || strings.Contains(got, "mlp.") {
			t.Errorf("mHC suffix %q mapped into a generic attention/MLP namespace: %q", suffix, got)
		}
		if strings.Contains(got, ".hc.") {
			t.Errorf("mHC suffix %q collapsed onto the legacy hc.* tap namespace: %q", suffix, got)
		}
	}
}

// TestDeepSeek41GGUFV41MHCSublayerDialectResolvesToForwardNames is the regression
// guard for the real per-sublayer dialect the staged vcruz305 Q2_K artifact emits
// (parsed from the GGUF header: blk.N.hc_attn_base.weight [24] F32,
// blk.N.hc_attn_fn.weight [HCMult*H, 24], blk.N.hc_attn_scale.weight [3] F32, and
// the hc_ffn_* trio). There is NO mhc_mixes tensor in that artifact. Before the
// fix deepseek41MHCSuffixName had no arm for these suffixes and the hc_attn_*
// arms in the canonical map re-nested them into the DEAD per-layer
// hc.<leaf>.weight namespace the native forward never reads, so the forward's
// required mhc.mixes.weight was never populated and admission refused by name
// (fak#13258). Assert each sublayer suffix resolves to the forward-consumed leaf,
// and that the attn and ffn trios stay distinct.
func TestDeepSeek41GGUFV41MHCSublayerDialectResolvesToForwardNames(t *testing.T) {
	dialects := map[string]string{
		"hc_attn_base.weight":  "mhc.base",
		"hc_attn_scale.weight": "mhc.scale",
		"hc_attn_fn.weight":    "mhc.mixes.weight",
		"hc_ffn_base.weight":   "mhc.ffn_base",
		"hc_ffn_scale.weight":  "mhc.ffn_scale",
		"hc_ffn_fn.weight":     "mhc.ffn_mixes.weight",
	}
	seen := map[string]string{}
	for _, layer := range []int{0, 7} {
		for suffix, wantLeaf := range dialects {
			ggufName := fmt.Sprintf("blk.%d.%s", layer, suffix)
			got, ok := CanonicalTensorNameArch(ggufName, "deepseek41")
			if !ok {
				t.Errorf("published V4.1 sublayer mHC tap %q did not resolve under deepseek41; the forward admits %q and the shard load must resolve it",
					ggufName, wantLeaf)
				continue
			}
			want := layerName(layer, wantLeaf)
			if got != want {
				t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = %q, want forward-consumed %q", ggufName, got, want)
			}
			if layer == 0 {
				if prev, dup := seen[got]; dup {
					t.Errorf("V4.1 sublayer mHC taps %q and %q collide on canonical name %q", prev, suffix, got)
				}
				seen[got] = suffix
			}
		}
	}
}

// TestDeepSeek41GGUFV41MHCSublayerDialectDoesNotLeakToSiblings proves the
// per-sublayer hc_attn_*/hc_ffn_* arms stay gated on archIsDeepSeek41: every
// non-deepseek41 arch must REFUSE them, so a V4.1-only tensor cannot mis-route
// into a sibling forward (fak#13258).
func TestDeepSeek41GGUFV41MHCSublayerDialectDoesNotLeakToSiblings(t *testing.T) {
	taps := []string{
		"hc_attn_base.weight", "hc_attn_scale.weight", "hc_attn_fn.weight",
		"hc_ffn_base.weight", "hc_ffn_scale.weight", "hc_ffn_fn.weight",
	}
	for _, sibling := range []string{"llama", "qwen2", "deepseek2", "glm_moe_dsa"} {
		for _, tap := range taps {
			ggufName := "blk.0." + tap
			if got, ok := CanonicalTensorNameArch(ggufName, sibling); ok {
				t.Errorf("V4.1 sublayer mHC arm leaked into sibling arch %q: CanonicalTensorNameArch(%q, %q) = %q, want ok=false",
					sibling, ggufName, sibling, got)
			}
		}
	}
}

// TestDeepSeek41GGUFV41MHCTapsDoNotLeakToSiblings proves the mHC arms stay gated
// on archIsDeepSeek41: every non-deepseek41 arch must REFUSE the converter
// spelling (the strongest form of no-leak). An mHC coefficient block resolving
// under llama/qwen2/deepseek2/glm would mis-route a V4.1-only tensor into a
// sibling forward.
func TestDeepSeek41GGUFV41MHCTapsDoNotLeakToSiblings(t *testing.T) {
	taps := []string{"mhc_mixes.weight", "mhc_base.weight", "mhc_scale.weight", "mhc.mixes.weight", "mhc.base", "mhc.scale"}
	for _, sibling := range []string{"llama", "qwen2", "deepseek2", "glm_moe_dsa"} {
		for _, tap := range taps {
			ggufName := "blk.0." + tap
			if got, ok := CanonicalTensorNameArch(ggufName, sibling); ok {
				t.Errorf("V4.1 mHC arm leaked into sibling arch %q: CanonicalTensorNameArch(%q, %q) = %q, want ok=false",
					sibling, ggufName, sibling, got)
			}
		}
	}
}
