//go:build cuda

package model

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// End-to-end resident CUDA prefill of a synthetic one-period Qwen3.8-27B model: 3 GDN + 1
// full-attention layer at the 27B axes with Q8 weights, a reduced vocabulary and no q/k norm.
// It fits an 8 GB card where the 64-layer model does not, so it is the token-time arm for
// GDN prompt-kernel changes (#9359). Each rep is a fresh session; it logs prefill tok/s and
// a logits digest, which must match across reps and across bitwise-equal kernel arms.
// fak-test:runtime slow est=20s lane=optin
func TestCUDAQwen38PeriodPrefillTiming(t *testing.T) {
	if os.Getenv("FAK_GDN_E2E_BENCH") != "1" {
		t.Skip("set FAK_GDN_E2E_BENCH=1")
	}
	be := requiredCUDAGDNParityBackend(t)
	tokens := 2060
	if v, err := strconv.Atoi(os.Getenv("FAK_GDN_E2E_TOKENS")); err == nil && v > 0 {
		tokens = v
	}
	reps := 5
	if v, err := strconv.Atoi(os.Getenv("FAK_GDN_E2E_REPS")); err == nil && v > 0 {
		reps = v
	}
	cfg := qwen38OnePeriodSyntheticConfig()
	start := time.Now()
	m := NewSynthetic(cfg)
	m.Quantize()
	t.Logf("E2E synthetic build+quantize %.1fs", time.Since(start).Seconds())
	prompt := make([]int, tokens)
	for i := range prompt {
		prompt[i] = (i*2654435761 + 7) % cfg.VocabSize
	}
	var samples []float64
	var digest string
	for r := 0; r <= reps; r++ {
		s := m.NewBackendSession(be)
		s.Quant = true
		t0 := time.Now()
		logits := s.Prefill(prompt)
		elapsed := time.Since(t0)
		route := s.qwen35HAL.prefillRoute
		s.Close()
		be.Recycle()
		if route.EffectivePath != compute.Qwen35SequencePrefillPath {
			t.Fatalf("prefill route %q, want resident %q", route.EffectivePath, compute.Qwen35SequencePrefillPath)
		}
		h := sha256.New()
		var buf [4]byte
		finite := true
		for _, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				finite = false
			}
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			h.Write(buf[:])
		}
		d := hex.EncodeToString(h.Sum(nil))[:16]
		if !finite {
			t.Fatalf("rep %d logits not finite", r)
		}
		if digest == "" {
			digest = d
		} else if d != digest {
			t.Fatalf("rep %d logits digest %s != %s (non-deterministic)", r, d, digest)
		}
		tokS := float64(tokens) / elapsed.Seconds()
		t.Logf("E2E rep=%d tokens=%d prefill_ms=%.2f prefill_tok_s=%.3f digest=%s", r, tokens, float64(elapsed.Microseconds())/1000, tokS, d)
		if r > 0 {
			samples = append(samples, tokS)
		}
	}
	sort.Float64s(samples)
	t.Logf("E2E_SUMMARY tokens=%d reps=%d median_prefill_tok_s=%.3f digest=%s", tokens, reps, samples[len(samples)/2], digest)
}

// qwen38OnePeriodSyntheticConfig is one 3:1 period of the Qwen3.8-27B hybrid at the 27B axes with
// a reduced vocabulary and no q/k norm; the 64-layer model repeats it 16 times.
func qwen38OnePeriodSyntheticConfig() Config {
	return Config{
		Name:                  "Qwen3.8-27B one-period synthetic",
		ModelType:             "qwen3_5_text",
		HiddenSize:            5120,
		NumLayers:             4,
		NumHeads:              24,
		NumKVHeads:            4,
		HeadDim:               256,
		IntermediateSize:      17408,
		VocabSize:             4096,
		RMSNormEps:            1e-6,
		RopeTheta:             10000000,
		PartialRotaryFactor:   0.25,
		QKNorm:                false,
		TieWordEmbeddings:     true,
		EOSTokenID:            -1,
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim:   4,
		LinearKeyHeadDim:      128,
		LinearNumKeyHeads:     16,
		LinearValueHeadDim:    128,
		LinearNumValueHeads:   48,
		AttnOutputGate:        true,
		FullAttentionInterval: 4,
		NormGain1p:            true,
	}
}
