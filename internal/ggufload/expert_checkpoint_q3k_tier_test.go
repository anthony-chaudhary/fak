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
// ffn_down_exps is type 11 (Q3_K). The R5 streamed tier stages Q2_K but has no compute tensor for
// Q3_K, so FusedExpertTensors declines the down-projection slabs.
//
// The bug this pins: a PARTIAL decline used to be silent. The tier built fine over the Q2_K
// gate/up slabs, so the load did not refuse; the Q3_K down-projection slabs then dropped off the
// streamed set and were eager-dequantized to f32 by computeQ4KTensorWork, materializing the whole
// expert bulk the caller passed WithStreamedExperts to avoid. On the physical Halo that is the
// ~393 GiB runtime OOM recorded on fak#13144.
//
// The settlement (fak#13122): Q3_K has no compute tensor kind, so the tier cannot stage it and must
// not pretend to. The honest answer to a request for BOUNDED experts over a checkpoint whose expert
// bulk cannot be fully bounded is a refusal that names the unstageable slabs — not a quiet partial
// stream that leaves the bulk in RAM.

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

// TestStreamedExpertsRefusesPartiallyUnstageableExpertQuant is the fak#13144 regression. A
// WithStreamedExperts request over a checkpoint whose expert bulk is only PARTIALLY stageable must
// refuse, naming the slabs it cannot bound; it must never return a model that quietly eager-
// dequantizes the remainder to f32.
func TestStreamedExpertsRefusesPartiallyUnstageableExpertQuant(t *testing.T) {
	// Exactly the published V4.1 Q2_K shape: Q2_K gate/up, Q3_K down.
	path, E := frozenMixedExpertGGUF(t, TensorQ2_K, TensorQ3_K)

	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	// The tier describes only the stageable gate/up slabs; the Q3_K down slab is declined. That
	// asymmetry is what makes the request partial rather than empty.
	shards, err := ws.FusedExpertTensors()
	if err != nil {
		t.Fatalf("FusedExpertTensors: %v", err)
	}
	// One descriptor per projection covers all E experts as a batched slab, so the stageable Q2_K
	// gate/up pair yields two; the Q3_K down slab is the declined third.
	if got := fusedCount(shards); got != 2 {
		t.Fatalf("described %d fused slabs over %d experts, want 2 (the stageable Q2_K gate/up projections)", got, E)
	}

	// The load must refuse: the caller asked for bounded experts and the checkpoint cannot deliver
	// bounded experts. A nil error here is the silent f32 fall-through fak#13144 records.
	m, err := ws.QuantModelQ4KProfileOptions(nil, WithStreamedExperts(0))
	if err == nil {
		experts := -1
		if m != nil {
			experts = m.KQuantCount()
		}
		t.Fatalf("a partially-unstageable expert checkpoint loaded under WithStreamedExperts; "+
			"the Q3_K down-projection slabs would be eager-dequantized to f32 (the fak#13144 OOM). "+
			"model non-nil=%v experts=%d", m != nil, experts)
	}
	if m != nil {
		t.Fatal("a refused streamed load still returned a model")
	}
	// The refusal must name the unstageable projection so an operator can act on it, rather than
	// reporting a generic failure or the checkpoint merely having "no streamable slab".
	if !strings.Contains(err.Error(), "ffn_down_exps") {
		t.Fatalf("refusal %q does not name the unstageable slab; an operator cannot tell which "+
			"projection keeps the expert bulk in RAM", err)
	}
}

// TestStreamedExpertsAcceptsFullyStageableMixedKQuant is the control: the refusal above must key on
// UNstageability, not on quantization itself. A checkpoint whose every expert slab is stageable
// (here Q2_K gate/up with Q4_K down - the mixed-but-all-staged UD shape) must still stream.
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
