//go:build darwin && arm64 && cgo

// fak-test:runtime slow est=30s

package ggufload

import (
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func qwen35DevicePrefillLogits(t *testing.T, m *model.Model, prompt []int) ([]float32, model.Qwen35MetalForwardSequenceReceipt) {
	t.Helper()
	s := m.NewSession()
	defer s.Close()
	s.Q4K = true
	s.MetalQ4K = true
	logits := s.Prefill(prompt)
	return logits, s.Qwen35MetalForwardSequenceReceipt()
}

// A Qwen3.5-family hybrid prompt of 160 tokens runs a 128-token and a 32-token device prefill
// panel, whose Q8_0 projections take q8_mul_mm; the logits must match the scalar q8_gemm arm.
func TestQwen35SmallHybridQ8MulMMDevicePrefillParity(t *testing.T) {
	path := os.Getenv("FAK_TEST_QWEN35_HYBRID_GGUF")
	if path == "" {
		t.Skip("FAK_TEST_QWEN35_HYBRID_GGUF not set")
	}
	if !metalgemm.Available() || !metalgemm.Q8MulMMReady() {
		t.Skip("no Metal device with q8_mul_mm")
	}
	m, err := LoadModelQ4K(path)
	if err != nil {
		t.Fatal(err)
	}
	if !m.Cfg.IsQwen35Hybrid() || m.Cfg.NumLayers == 64 {
		t.Skipf("want a non-64-layer Qwen3.5-family hybrid, got layers=%d", m.Cfg.NumLayers)
	}
	prompt := make([]int, 160)
	for i := range prompt {
		prompt[i] = (i*7919 + 13) % 150000
	}
	defer metalgemm.SetQ8UseMulMM(true)
	metalgemm.SetQ8UseMulMM(false)
	want, _ := qwen35DevicePrefillLogits(t, m, prompt)
	metalgemm.SetQ8UseMulMM(true)
	got, receipt := qwen35DevicePrefillLogits(t, m, prompt)

	if receipt.ExecutedPanels == 0 || receipt.Tokens != len(prompt) {
		t.Fatalf("device prefill not taken: receipt=%+v", receipt)
	}
	var dot, ng, nw float64
	for i := range want {
		dot += float64(got[i]) * float64(want[i])
		ng += float64(got[i]) * float64(got[i])
		nw += float64(want[i]) * float64(want[i])
	}
	cos := dot / math.Sqrt(ng*nw)
	if cos < 0.9999 || argmaxF32(got) != argmaxF32(want) {
		t.Fatalf("q8_mul_mm prefill logits diverge: cos=%.7f argmax mul_mm=%d q8_gemm=%d", cos, argmaxF32(got), argmaxF32(want))
	}
	t.Logf("panels=%d logits cos=%.8f argmax=%d", receipt.ExecutedPanels, cos, argmaxF32(got))
}
