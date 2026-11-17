package model

// kimik31_latent_moe_test.go — parity + ordering tests for the K3.1 latent-MoE
// host oracle (ticket 1611-kimi-k31, port leaf sglang-kimi-k31-4 host side; goal
// Wave 1.2/1.3). The load-bearing property is the reduce-before-norm ORDERING:
// sum(norm(x_i)) != norm(sum(x_i)), so a port that normalizes per-rank and
// reduces after would diverge from the reference. These tests pin the front,
// norm and tail against an independent float64 transcription and PROVE the
// ordering matters (the wrong order differs from the right one on non-trivial
// inputs).
//
// fak-test:runtime fast est=5ms lane=default

import (
	"math"
	"math/rand"
	"testing"
)

func naiveRMS(x, w []float64, eps float64) []float64 {
	var ss float64
	for _, v := range x {
		ss += v * v
	}
	inv := 1.0 / math.Sqrt(ss/float64(len(x))+eps)
	out := make([]float64, len(x))
	for i := range x {
		out[i] = x[i] * inv * w[i]
	}
	return out
}

func naiveMatVec(w, x []float64, outDim, inDim int) []float64 {
	out := make([]float64, outDim)
	for o := 0; o < outDim; o++ {
		var s float64
		row := o * inDim
		for i := 0; i < inDim; i++ {
			s += w[row+i] * x[i]
		}
		out[o] = s
	}
	return out
}

func tinyLatentMoE(seed int64) (KimiK31LatentMoEConfig, KimiK31LatentMoEWeights) {
	cfg := KimiK31LatentMoEConfig{Hidden: 6, Latent: 4, ExpertIntermediate: 3, LatentUseNorm: true, RMSNormEps: 1e-5}
	rng := rand.New(rand.NewSource(seed))
	pick := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = rng.Float32()*2 - 1
		}
		return out
	}
	w := KimiK31LatentMoEWeights{
		DownProj:   pick(cfg.Latent * cfg.Hidden),
		UpProj:     pick(cfg.Hidden * cfg.Latent),
		LatentNorm: pick(cfg.Latent),
	}
	return cfg, w
}

// TestKimiK31LatentFrontNormTailMatchReference pins each stage against an
// independent float64 transcription.
func TestKimiK31LatentFrontNormTailMatchReference(t *testing.T) {
	cfg, w := tinyLatentMoE(9)
	if err := ValidateKimiK31LatentMoE(cfg, w); err != nil {
		t.Fatalf("validate: %v", err)
	}
	rng := rand.New(rand.NewSource(21))
	x := make([]float32, cfg.Hidden)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	// A stand-in "combined expert output" in latent space.
	latent := make([]float32, cfg.Latent)
	for i := range latent {
		latent[i] = rng.Float32()*2 - 1
	}

	front := KimiK31LatentRoutedInput(cfg, w, x)
	wantFront := naiveMatVec(f64s(w.DownProj), f64s(x), cfg.Latent, cfg.Hidden)
	for i := range front {
		if math.Abs(float64(front[i])-wantFront[i]) > 1e-5 {
			t.Errorf("front[%d] = %v, want %v", i, front[i], wantFront[i])
		}
	}

	normed := KimiK31LatentNorm(cfg, w, latent)
	wantNorm := naiveRMS(f64s(latent), f64s(w.LatentNorm), float64(cfg.RMSNormEps))
	for i := range normed {
		if math.Abs(float64(normed[i])-wantNorm[i]) > 1e-5 {
			t.Errorf("norm[%d] = %v, want %v", i, normed[i], wantNorm[i])
		}
	}

	tail := KimiK31LatentTail(cfg, w, normed)
	wantTail := naiveMatVec(f64s(w.UpProj), f64s(normed), cfg.Hidden, cfg.Latent)
	for i := range tail {
		if math.Abs(float64(tail[i])-wantTail[i]) > 1e-5 {
			t.Errorf("tail[%d] = %v, want %v", i, tail[i], wantTail[i])
		}
	}
}

// TestKimiK31LatentReduceBeforeNormOrdering is the load-bearing witness: with
// two tensor-parallel partial sums, norm(reduce(a,b)) differs from
// reduce(norm(a),norm(b)) — so the reference's reduce-before-norm order is
// observable, and the host oracle must take the already-reduced latent.
func TestKimiK31LatentReduceBeforeNormOrdering(t *testing.T) {
	cfg, w := tinyLatentMoE(3)
	rank0 := []float32{1, 2, 3, 4}
	rank1 := []float32{0.5, -1, 0.25, 2}

	// Correct: reduce first, then norm.
	reduced := make([]float32, cfg.Latent)
	for i := range reduced {
		reduced[i] = rank0[i] + rank1[i]
	}
	got := KimiK31LatentNorm(cfg, w, reduced)

	// Wrong: norm each rank, then reduce.
	n0 := KimiK31LatentNorm(cfg, w, append([]float32(nil), rank0...))
	n1 := KimiK31LatentNorm(cfg, w, append([]float32(nil), rank1...))
	for i := range got {
		wrong := n0[i] + n1[i]
		if math.Abs(float64(got[i]-wrong)) > 1e-5 {
			return // the orders differ, as the contract claims
		}
	}
	t.Fatal("norm(a+b) equalled norm(a)+norm(b): the reduce-before-norm ordering is not observable on this input")
}

// TestKimiK31LatentNormOptional pins the latent_moe_use_norm=false branch: the
// latent passes through unchanged.
func TestKimiK31LatentNormOptional(t *testing.T) {
	cfg, w := tinyLatentMoE(4)
	cfg.LatentUseNorm = false
	w.LatentNorm = nil
	if err := ValidateKimiK31LatentMoE(cfg, w); err != nil {
		t.Fatalf("validate without norm: %v", err)
	}
	latent := []float32{1, -2, 3, -4}
	got := KimiK31LatentNorm(cfg, w, latent)
	for i := range latent {
		if got[i] != latent[i] {
			t.Fatalf("no-norm latent[%d] = %v, want pass-through %v", i, got[i], latent[i])
		}
	}
}

// TestKimiK31LatentValidateRefuses proves the shape validator fails closed.
func TestKimiK31LatentValidateRefuses(t *testing.T) {
	cfg, w := tinyLatentMoE(5)
	if err := ValidateKimiK31LatentMoE(cfg, w); err != nil {
		t.Fatalf("well-formed config must validate: %v", err)
	}
	bad := w
	bad.DownProj = bad.DownProj[:len(bad.DownProj)-1]
	if err := ValidateKimiK31LatentMoE(cfg, bad); err == nil {
		t.Fatal("expected a refusal for a short DownProj")
	}
	badNorm := cfg
	badNorm.LatentUseNorm = true
	badW := w
	badW.LatentNorm = nil
	if err := ValidateKimiK31LatentMoE(badNorm, badW); err == nil {
		t.Fatal("expected a refusal for a missing latent norm weight when use_norm is set")
	}
}
