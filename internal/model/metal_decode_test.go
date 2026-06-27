//go:build darwin && cgo && fakmetal

package model

// metal_decode_test.go — the correctness gate for the GPU-resident Q8 decode forward
// (internal/metalgemm/decode.m + metal_decode.go, issue #67). It runs the WHOLE decode token on the
// GPU in one command buffer; this test holds it to the proven CPU Q8 decode on a hermetic dense
// synthetic model. The resident path uses an f16 activation × Q8 weight dequant-GEMV (more accurate
// than the CPU int8×int8 dot), so it agrees with the CPU path up to GPU float-accumulation order
// (logit cosine ~1.0, same argmax) — a wiring bug (wrong weight, KV append, RoPE position, attention
// stride) diverges O(1).

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestMetalDecodeResidentMatchesCPU prefills a prompt on the CPU, then steps the SAME token through
// both the CPU Q8 decode (s.Metal=false) and the GPU-resident decode (s.Metal=true) from identical
// caches, and compares the resulting logits. Then it runs a short greedy continuation both ways and
// checks the token sequences match — the end-to-end "the resident forward decodes the same tokens"
// gate, the dense twin of TestMetalQ4KDecodeMatchesCPU.
func TestMetalDecodeResidentMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer func() { metalgemm.ResetQ8(); metalgemm.DecodeReset() }()

	cfg := syntheticDecodeCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	if !m.metalDecodeConfig() {
		t.Fatal("metalDecodeConfig declined for the dense synthetic model — resident path would not engage")
	}
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43}

	// One step, both ways, from identical CPU-prefilled caches.
	cpu := m.NewSession()
	cpu.Quant = true
	lgC := cpu.Prefill(prompt)
	gpu := m.NewSession()
	gpu.Quant = true
	lgG := gpu.Prefill(prompt) // CPU prefill (s.Metal still false)
	gpu.Metal = true           // resident decode for the Step calls below

	if argmaxF(lgC) != argmaxF(lgG) {
		t.Fatalf("CPU prefill diverged between sessions: %d vs %d", argmaxF(lgC), argmaxF(lgG))
	}
	tok := argmaxF(lgC)
	cStep := cpu.Step(tok)
	gStep := gpu.Step(tok)
	cos, maxRel := cosineAndMaxRel(cStep, gStep)
	if argmaxF(cStep) != argmaxF(gStep) || cos < 0.999 {
		t.Errorf("resident decode step: cpu argmax=%d gpu argmax=%d cos=%.6f maxRel=%.4g (want same argmax, cos>=0.999)\n  cpu[:6]=%v\n  gpu[:6]=%v",
			argmaxF(cStep), argmaxF(gStep), cos, maxRel, head6(cStep), head6(gStep))
	} else {
		t.Logf("resident decode step: argmax=%d cos=%.6f maxRel=%.4g OK", argmaxF(gStep), cos, maxRel)
	}

	// Multi-token greedy continuation, both ways (fresh sessions so the one-step Step above does not
	// perturb them). The sequence should match; a divergence point localizes any compounding error.
	greedy := func(metal bool) []int {
		s := m.NewSession()
		s.Quant = true
		lg := s.Prefill(prompt)
		s.Metal = metal
		var seq []int
		for i := 0; i < 8; i++ {
			n := argmaxF(lg)
			seq = append(seq, n)
			lg = s.Step(n)
		}
		return seq
	}
	cpuSeq := greedy(false)
	gpuSeq := greedy(true)
	for i := range cpuSeq {
		if cpuSeq[i] != gpuSeq[i] {
			t.Errorf("greedy decode diverged at token %d: cpu=%d gpu=%d\n  cpu=%v\n  gpu=%v", i, cpuSeq[i], gpuSeq[i], cpuSeq, gpuSeq)
			return
		}
	}
	t.Logf("resident decode greedy sequence matches CPU = %v", gpuSeq)
}

func head6(v []float32) []float32 {
	if len(v) < 6 {
		return v
	}
	return v[:6]
}
