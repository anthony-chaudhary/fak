package model

import (
	"math"
	"testing"
)

// TestQwen35LinearStepParallelMatchesSerial pins the parallel-over-value-heads Gated-DeltaNet
// recurrent scan (linearAttnStep) BIT-FOR-BIT to the serial reference. The per-head update
// mutates only its own recurrent state and writes a disjoint slice of core, so spreading the
// nV heads across workers reorders WORK, never a single head's reduction — exactly the contract
// parMatRows holds (TestParallelMatchesSerial). The GDN scan runs on 3-of-4 layers of a Qwen3.6
// hybrid, so this parallelism is the per-token decode lever for the 75%-of-layers linear path.
//
// The config is sized so nV*kHd*vHd crosses parThreshold (65536), forcing the parFor branch of
// linearAttnStep at the package-default worker count; the serial reference pins numWorkers=1.
func TestQwen35LinearStepParallelMatchesSerial(t *testing.T) {
	cfg := Config{
		HiddenSize:            64,
		NumLayers:             4,
		NumHeads:              8,
		NumKVHeads:            4,
		HeadDim:               8,
		IntermediateSize:      128,
		VocabSize:             97,
		RMSNormEps:            1e-5,
		RopeTheta:             10000,
		TieWordEmbeddings:     true,
		EOSTokenID:            -1,
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim:   3,
		LinearKeyHeadDim:      64,
		LinearNumKeyHeads:     8,
		LinearValueHeadDim:    64,
		LinearNumValueHeads:   16, // nV*kHd*vHd = 16*64*64 = 65536 == parThreshold, so the parFor branch fires
		AttnOutputGate:        true,
		FullAttentionInterval: 4,
		NormGain1p:            true,
	}
	if cfg.LinearNumValueHeads*cfg.LinearKeyHeadDim*cfg.LinearValueHeadDim < parThreshold {
		t.Fatalf("config does not cross parThreshold; the parallel branch would not fire")
	}
	m := NewSynthetic(cfg)
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31}

	// Serial reference: force one worker so linearAttnStep takes the serial branch.
	saved := numWorkers
	numWorkers = 1
	serSess := m.NewSession()
	serSess.Prefill(prompt[:4])
	var serLogits []float32
	for _, id := range prompt[4:] {
		serLogits = serSess.Step(id)
	}

	// Parallel: restore the package default (>1 on the CI box) so the parFor branch runs.
	numWorkers = saved
	if numWorkers <= 1 {
		t.Skipf("numWorkers=%d: cannot exercise the parallel branch on this box", numWorkers)
	}
	parSess := m.NewSession()
	parSess.Prefill(prompt[:4])
	var parLogits []float32
	for _, id := range prompt[4:] {
		parLogits = parSess.Step(id)
	}

	if len(serLogits) != len(parLogits) {
		t.Fatalf("logit length mismatch: serial %d != parallel %d", len(serLogits), len(parLogits))
	}
	for i := range serLogits {
		if math.Float32bits(serLogits[i]) != math.Float32bits(parLogits[i]) {
			t.Fatalf("GDN parallel scan not bit-identical at logit %d: serial %v (0x%08x) != parallel %v (0x%08x)",
				i, serLogits[i], math.Float32bits(serLogits[i]), parLogits[i], math.Float32bits(parLogits[i]))
		}
	}
}
