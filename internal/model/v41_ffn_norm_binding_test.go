package model

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41OwnedFFNCollapseSingleConsumption(t *testing.T) {
	flat := make([]float32, 64)
	carrier := newV41OwnedFFNCollapse(nil, flat, 4, 16)
	copied := *carrier
	if &carrier.rows[0][0] != &flat[0] || &carrier.rows[3][0] != &flat[48] {
		t.Fatal("carrier copied/reconstructed storage")
	}
	for _, row := range carrier.rows {
		if len(row) != 16 || cap(row) != 16 {
			t.Fatal("uncapped row")
		}
	}
	if !carrier.state.loopClaimed.CompareAndSwap(false, true) || copied.state.loopClaimed.CompareAndSwap(false, true) {
		t.Fatal("copied carrier granted second consumption")
	}
	carrier.state.flat[16] = 7
	if copied.rows[1][0] != 7 || carrier.rows[0][0] != 0 || carrier.rows[2][0] != 0 {
		t.Fatal("chunk view not isolated")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41FFNNormReplacementStripsAuthority(t *testing.T) {
	var absent *v41FFNNormBinding
	if absent.normFunc() != nil {
		t.Fatal("nil binding callback")
	}
	called := false
	ordinary := func(int, []float32) ([]float32, error) { called = true; return nil, nil }
	replacement := &v41FFNNormBinding{ordinary: ordinary}
	if replacement.owner != nil || replacement.backend != nil || replacement.publication != nil {
		t.Fatal("replacement retained authority")
	}
	replacement.normFunc()(0, nil)
	if !called {
		t.Fatal("ordinary behavior lost")
	}
	if _, handled, err := replacement.panel(nil, nil, nil, nil, 0, nil, 1e-5); handled || err != nil {
		t.Fatal("ordinary callback selected panel")
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41FFNPreparedScalarGainOnce(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	b := &v41QueryNormTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
	s := v41DenseTestSession(t, m, b)
	const leaf = "ffn_norm.weight"
	b.queryWeight = func() compute.Tensor { return s.halW[layerName(0, leaf)] }
	input := make([]float32, m.Cfg.HiddenSize)
	for i := range input {
		input[i] = 1
	}
	var weight compute.Tensor
	attempts := 0
	miss := func(gain compute.Tensor) ([]float32, bool, error) {
		attempts++
		if gain.Buf() == nil {
			t.Fatal("gain prefix missing")
		}
		return nil, false, nil
	}
	first, err := s.v41RMSNormPrepared(layerName(0, leaf), input, len(input), 1, "v41-ffn-norm", 0, nil, &weight, miss)
	if err != nil || len(first) != len(input) || attempts != 1 || b.norms != 1 {
		t.Fatalf("scalar miss continuation: %v, attempts=%d norms=%d", err, attempts, b.norms)
	}
	original := weight.Buf()
	second, err := s.v41RMSNormPrepared(layerName(0, leaf), input, len(input), 1, "v41-ffn-norm", 0, nil, &weight, nil)
	if err != nil || len(second) != len(input) || weight.Buf() != original || b.norms != 2 {
		t.Fatalf("prepared gain lifetime: %v", err)
	}
}
