package ggufload

import (
	"fmt"
	"testing"
)

// TestDeepSeek41GGUFV41RouterGateNameAndStorage is the #13266 regression guard.
//
// The published V4.1 artifact stores the MoE router gate as
// blk.<L>.ffn_gate_inp.weight with header dims [H, NumExperts] (the deepseek2
// spelling), while the native non-MLA V4.1 forward reads the canonical leaf
// model.layers.<L>.ffn.gate.weight (internal/model/v41_forward.go:685 admission,
// :1114 read, :1381 matRows(wGate, xn, cfg.NumExperts, H)).
//
// Before this fix the deepseek41 file fell through to the SHARED batched-MoE
// branch of CanonicalTensorNameArch (archUsesGGUFBatchedMoEExperts includes
// "deepseek41"), which rewrites ffn_gate_inp.weight to the deepseek2/GLM
// spelling model.layers.<L>.mlp.gate.weight. The native forward never reads
// that name, so admission refused the model with a named
// ErrV41ForwardStage for a tensor the file DOES carry. The gate below pins the
// loader-name resolution red on the parent commit and green after the fix.
func TestDeepSeek41GGUFV41RouterGateNameAndStorage(t *testing.T) {
	for _, layer := range []int{0, 7, 39} {
		ggufName := fmt.Sprintf("blk.%d.ffn_gate_inp.weight", layer)
		got, ok := CanonicalTensorNameArch(ggufName, "deepseek41")
		if !ok {
			t.Fatalf("CanonicalTensorNameArch(%q, deepseek41) = ok=false; the artifact's router gate must resolve", ggufName)
		}
		want := layerName(layer, "ffn.gate.weight")
		if got != want {
			t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = %q, want the native-forward leaf %q", ggufName, got, want)
		}
		if got == layerName(layer, "mlp.gate.weight") {
			t.Errorf("router gate fell through to the deepseek2/GLM spelling %q; the native V4.1 forward reads ffn.gate.weight", got)
		}
	}
}

// TestDeepSeek41GGUFV41RouterGateDoesNotLeakToSiblings is the negative arm: the
// ffn_gate_inp.weight suffix keeps resolving to the shared deepseek2/GLM
// mlp.gate.weight leaf for every OTHER arch that reaches that map. Only the
// deepseek41 branch may produce the native ffn.gate.weight leaf.
func TestDeepSeek41GGUFV41RouterGateDoesNotLeakToSiblings(t *testing.T) {
	for _, sibling := range []string{"llama", "qwen2", "deepseek2", "glm_moe_dsa"} {
		got, ok := CanonicalTensorNameArch("blk.0.ffn_gate_inp.weight", sibling)
		if !ok {
			continue // refused: the strongest form of no-leak
		}
		if got == layerName(0, "ffn.gate.weight") {
			t.Errorf("sibling %q resolved ffn_gate_inp.weight to the V4.1 NATIVE leaf %q; only the deepseek41 branch may produce it",
				sibling, got)
		}
	}
}

// TestDeepSeek41GGUFV41RouterBiasStillMaps confirms the score-correction bias
// companion is unchanged by the gate fix: the artifact's exp_probs_b.bias must
// keep resolving to the native ffn.gate.e_score_correction_bias leaf.
func TestDeepSeek41GGUFV41RouterBiasStillMaps(t *testing.T) {
	got, ok := CanonicalTensorNameArch("blk.0.exp_probs_b.bias", "deepseek41")
	if !ok {
		t.Fatal("CanonicalTensorNameArch(blk.0.exp_probs_b.bias, deepseek41) = ok=false; the router bias must resolve")
	}
	if want := layerName(0, "ffn.gate.e_score_correction_bias"); got != want {
		t.Errorf("CanonicalTensorNameArch(blk.0.exp_probs_b.bias, deepseek41) = %q, want %q", got, want)
	}
}
