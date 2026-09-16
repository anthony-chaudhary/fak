package ggufload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// expert_checkpoint_q3k_tier_test.go — the mixed-quant streamed-expert witness for fak#13144.
//
// The real vcruz305/DeepSeek-V4.1-Flash-GGUF Q2_K checkpoint stores its routed-expert slabs with
// TWO quant types in one layer: ffn_gate_exps / ffn_up_exps are type 10 (Q2_K), while
// ffn_down_exps is type 11 (Q3_K). Before this change the R5 streamed tier staged Q2_K but had no
// way to stage Q3_K, so FusedExpertTensors declined the down-projection slabs and the load either
// refused (the landed partial-decline guard) or eager-dequantized the bulk to f32 (the ~393 GiB
// physical OOM fak#13144 records).
//
// The reversal: Q3_K is now ADMITTED. fak#13149 added the compute.NewQ3K verbatim host tensor
// kind the fak#13122 settlement said was missing, so the tier stages a Q3_K expert exactly like
// Q2_K — bytes verbatim, no f32 expansion. This overturns the fak#13122 settlement because its
// premise (no Q3_K compute tensor exists) no longer holds.
//
// The RED->GREEN property: over the exact mixed Q2_K/Q3_K shape, WithStreamedExperts must produce a
// model with ALL THREE projections streamed (zero experts resident in host RAM) — not a refusal and
// not a partial stream that leaves the Q3_K down slabs in RAM.

// frozenMixedExpertGGUF writes a complete glm_moe_dsa GGUF whose routed-expert slabs are typed
// per-projection, mirroring the published V4.1 Q2_K artifact: gate/up as the majority quant and
// down as the odd one out. It returns the path and the expert count.
func frozenMixedExpertGGUF(t *testing.T, gateUp, down TensorType) (path string, experts int) {
	t.Helper()
	const (
		H, V                = 256, 8
		qLora, kvLora       = 32, 32
		qkNope, qkRope, vHd = 16, 16, 16
		nH                  = 2
		idxHeads, idxDim    = 2, 16
		E, I, sharedI       = 3, 256, 256
	)
	blob := glmMoeDsaFullGGUFWithTypes(H, V, qLora, kvLora, qkNope, qkRope, vHd, nH, idxHeads, idxDim, E, I, sharedI,
		func(name string) TensorType {
			switch {
			case strings.HasSuffix(name, "ffn_down_exps.weight"):
				return down
			case strings.Contains(name, "_exps.weight"):
				return gateUp
			default:
				return TensorF32
			}
		})
	path = filepath.Join(t.TempDir(), "glm_mixed_experts.gguf")
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	return path, E
}

// TestStreamedExpertsAdmitsMixedQ2KQ3KExpertQuant is the fak#13144 RED->GREEN regression. Over the
// exact published V4.1 Q2_K shape (Q2_K gate/up, Q3_K down), a WithStreamedExperts request must
// admit ALL THREE projections to the R5 tier: the Q3_K down slabs are staged through the bounded
// on-demand f32 arm rather than refused or eager-dequantized.
func TestStreamedExpertsAdmitsMixedQ2KQ3KExpertQuant(t *testing.T) {
	// Exactly the published V4.1 Q2_K shape: Q2_K gate/up, Q3_K down.
	path, E := frozenMixedExpertGGUF(t, TensorQ2_K, TensorQ3_K)

	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	// The tier describes every routed-expert slab now: one descriptor per projection covers all E
	// experts as a batched slab, so gate + up + down = three. The Q3_K down slab is no longer
	// declined, which is the admission this regression pins.
	shards, err := ws.FusedExpertTensors()
	if err != nil {
		t.Fatalf("FusedExpertTensors: %v", err)
	}
	if got := fusedCount(shards); got != 3 {
		t.Fatalf("described %d fused slabs over %d experts, want 3 (Q2_K gate/up + Q3_K down all admitted)", got, E)
	}

	// And there must be nothing unstageable: the partial-decline refusal must not fire.
	unstageable, err := ws.UnstageableRoutedExpertSlabs()
	if err != nil {
		t.Fatalf("UnstageableRoutedExpertSlabs: %v", err)
	}
	if len(unstageable) != 0 {
		t.Fatalf("Q3_K down slabs still reported unstageable: %v; the tier must admit them", unstageable)
	}

	// The load must succeed and leave ZERO routed experts in host RAM: every slab, including the
	// Q3_K down projection, is served through the bounded tier rather than materialized.
	m, err := ws.QuantModelQ4KProfileOptions(nil, WithStreamedExperts(0))
	if err != nil {
		t.Fatalf("the exact V4.1 Q2_K/Q3_K checkpoint was refused under WithStreamedExperts: %v", err)
	}
	if m == nil {
		t.Fatal("streamed load returned a nil model")
	}
	if got := m.KQuantCount(); got != 0 {
		t.Fatalf("a streamed load left %d experts resident, want 0; the Q3_K down slabs must not be eager-f32", got)
	}
	st := m.ExpertCheckpointStats()
	if !st.Enabled || st.Tensors != E*3 {
		t.Fatalf("streamed model reports %+v, want an enabled tier over %d experts (all three projections)", st, E*3)
	}
}

// TestStreamedExpertsAcceptsFullyStageableMixedKQuant is the verbatim control: a checkpoint whose
// every expert slab has a compute kind of its own (here Q2_K gate/up with Q4_K down - the mixed-but-
// all-verbatim UD shape) must stream through the unchanged verbatim arms.
func TestStreamedExpertsAcceptsFullyStageableMixedKQuant(t *testing.T) {
	path, E := frozenMixedExpertGGUF(t, TensorQ2_K, TensorQ4_K)

	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	m, err := ws.QuantModelQ4KProfileOptions(nil, WithStreamedExperts(0))
	if err != nil {
		t.Fatalf("a fully-stageable mixed k-quant checkpoint was refused: %v", err)
	}
	if got := m.KQuantCount(); got != 0 {
		t.Fatalf("a streamed load left %d experts resident, want 0", got)
	}
	st := m.ExpertCheckpointStats()
	if !st.Enabled || st.Tensors != E*3 {
		t.Fatalf("streamed model reports %+v, want an enabled tier over %d experts", st, E*3)
	}
}
