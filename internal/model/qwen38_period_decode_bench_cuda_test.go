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
)

// End-to-end resident CUDA decode of the synthetic one-period Qwen3.8-27B model (see
// qwen38OnePeriodSyntheticConfig): the token-time arm for GDN decode-kernel changes (#9359).
// Each rep is a fresh session that prefills the same prompt, then greedily decodes; it logs
// decode tok/s and a digest of every decode step's logits, which must match across reps and
// across bitwise-equal kernel arms. Every step must run all three GDN layers on the device.
// fak-test:runtime slow est=25s lane=optin
func TestCUDAQwen38PeriodDecodeTiming(t *testing.T) {
	if os.Getenv("FAK_GDN_E2E_DECODE_BENCH") != "1" {
		t.Skip("set FAK_GDN_E2E_DECODE_BENCH=1")
	}
	be := requiredCUDAGDNParityBackend(t)
	envInt := func(name string, def int) int {
		if v, err := strconv.Atoi(os.Getenv(name)); err == nil && v > 0 {
			return v
		}
		return def
	}
	promptLen := envInt("FAK_GDN_E2E_DECODE_PROMPT", 64)
	steps := envInt("FAK_GDN_E2E_DECODE_STEPS", 128)
	reps := envInt("FAK_GDN_E2E_REPS", 5)
	cfg := qwen38OnePeriodSyntheticConfig()
	start := time.Now()
	m := NewSynthetic(cfg)
	m.Quantize()
	t.Logf("E2E synthetic build+quantize %.1fs", time.Since(start).Seconds())
	gdnLayers := 0
	for _, kind := range cfg.LayerTypes {
		if kind == "linear_attention" {
			gdnLayers++
		}
	}
	prompt := make([]int, promptLen)
	for i := range prompt {
		prompt[i] = (i*2654435761 + 7) % cfg.VocabSize
	}
	stepLogits := make([][]float32, steps)
	for i := range stepLogits {
		stepLogits[i] = make([]float32, cfg.VocabSize)
	}
	var samples []float64
	var digest string
	for r := 0; r <= reps; r++ {
		s := m.NewBackendSession(be)
		s.Quant = true
		next := argmaxF32(s.Prefill(prompt))
		opsBefore := be.Qwen35GDNOperationCount()
		t0 := time.Now()
		for i := 0; i < steps; i++ {
			logits := s.Step(next)
			copy(stepLogits[i], logits)
			next = argmaxF32(logits)
		}
		elapsed := time.Since(t0)
		ops := be.Qwen35GDNOperationCount() - opsBefore
		s.Close()
		be.Recycle()
		if want := uint64(steps * gdnLayers); ops != want {
			t.Fatalf("rep %d ran %d device GDN decode operations, want %d (%d steps x %d GDN layers)", r, ops, want, steps, gdnLayers)
		}
		h := sha256.New()
		var buf [4]byte
		for i, logits := range stepLogits {
			for _, v := range logits {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("rep %d step %d logits not finite", r, i)
				}
				binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
				h.Write(buf[:])
			}
		}
		d := hex.EncodeToString(h.Sum(nil))[:16]
		if digest == "" {
			digest = d
		} else if d != digest {
			t.Fatalf("rep %d decode logits digest %s != %s (non-deterministic)", r, d, digest)
		}
		tokS := float64(steps) / elapsed.Seconds()
		t.Logf("E2E rep=%d prompt=%d steps=%d decode_ms=%.2f decode_tok_s=%.3f digest=%s", r, promptLen, steps, float64(elapsed.Microseconds())/1000, tokS, d)
		if r > 0 {
			samples = append(samples, tokS)
		}
	}
	sort.Float64s(samples)
	t.Logf("E2E_DECODE_SUMMARY prompt=%d steps=%d reps=%d median_decode_tok_s=%.3f digest=%s", promptLen, steps, reps, samples[len(samples)/2], digest)
}
