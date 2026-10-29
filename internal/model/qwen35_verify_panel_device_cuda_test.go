//go:build cuda && cgo

package model

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// requiredCUDASequenceBackend returns the CUDA backend when it advertises the
// whole-sequence and all-row logits contracts. FAK_CUDA_REQUIRE_DEVICE=1 turns
// a missing device into a failure on an acceptance node.
func requiredCUDASequenceBackend(t *testing.T) compute.Backend {
	t.Helper()
	be := compute.Pick("cuda")
	if be == nil || be.Name() != "cuda" {
		if os.Getenv("FAK_CUDA_REQUIRE_DEVICE") == "1" {
			t.Fatal("required physical CUDA backend is unavailable")
		}
		t.Skip("CUDA unavailable; FAK_CUDA_REQUIRE_DEVICE=1 prohibits this skip")
	}
	seq, ok := be.(Qwen35SequencePrefillBackend)
	if !ok || seq.Qwen35SequencePrefillPath() != compute.Qwen35SequencePrefillPath {
		t.Fatalf("CUDA backend %T lacks whole-sequence prefill", be)
	}
	all, ok := be.(compute.Qwen35SequenceAllLogitsBackend)
	if !ok || all.Qwen35SequenceAllLogitsPath() != compute.Qwen35SequenceAllLogitsPath {
		t.Fatalf("CUDA backend %T lacks the all-row logits contract", be)
	}
	return be
}

func compareCUDASequenceRow(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) || len(want) == 0 {
		t.Fatalf("%s lengths got=%d want=%d", label, len(got), len(want))
	}
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("%s[%d] nonfinite got=%g", label, i, g)
		}
		if math.Abs(g-w) > 2e-3+2e-3*math.Abs(w) {
			t.Fatalf("%s[%d] got=%g want=%g", label, i, g, w)
		}
	}
	if argmaxF32(got) != argmaxF32(want) {
		t.Fatalf("%s argmax got=%d want=%d", label, argmaxF32(got), argmaxF32(want))
	}
}

// cudaQwen35AllLogitsTestCfg narrows the hybrid fixture to the attention
// geometry the CUDA prompt-attention panel admits (4 KV heads x 256) while
// keeping every other axis tiny, so the test fits beside a resident service.
func cudaQwen35AllLogitsTestCfg() Config {
	cfg := qwen35HybridTestCfg()
	cfg.HiddenSize, cfg.IntermediateSize = 64, 128
	cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim = 24, 4, 256
	cfg.PartialRotaryFactor = 0.25
	return cfg
}

// TestCUDAQwen35SequenceAllLogitsRowsMatchCPU is the CUDA arm of the resident
// verification contract n-gram speculation relies on: one sequence call with
// NeedAllLogits returns a logits row per draft token equal to serial CPU decode,
// and the following ordinary step continues from the committed state.
//
// fak-test:runtime fast est=2s lane=default
func TestCUDAQwen35SequenceAllLogitsRowsMatchCPU(t *testing.T) {
	backend := requiredCUDASequenceBackend(t)
	m := NewSynthetic(cudaQwen35AllLogitsTestCfg())
	device, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	device.captureTargetHidden = false
	serial := m.NewSession()
	defer serial.Close()

	prefix := []int{3, 7, 11, 5, 17, 19}
	boundary := device.Prefill(prefix)
	serial.Prefill(prefix)
	for sample, draft := range [][]int{{23, 2, 29, 31}, {37, 41}, {53}} {
		rows, receipt, err := device.verifyQwen35DevicePanel(draft, boundary)
		if err != nil {
			t.Fatalf("sample %d: %v", sample, err)
		}
		if len(rows) != len(draft) || receipt.Path != targetVerificationQwen38DevicePanelPath || receipt.TargetVerificationOperations != 1 || receipt.TargetDecodeSteps != 0 {
			t.Fatalf("sample %d rows=%d receipt=%+v", sample, len(rows), receipt)
		}
		for row, token := range draft {
			compareCUDASequenceRow(t, fmt.Sprintf("sample_%d_row_%d", sample, row), rows[row], serial.Step(token))
		}
		continuation := (sample*11 + 2) % m.Cfg.VocabSize
		want := serial.Step(continuation)
		got := device.Step(continuation)
		compareCUDASequenceRow(t, fmt.Sprintf("sample_%d_continuation", sample), got, want)
		boundary = got
	}
	t.Logf("engine=fak-native backend=cuda device=%s per_row_logits=true continuation_parity=true", backend.Tier())
}
