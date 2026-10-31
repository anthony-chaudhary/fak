//go:build cuda && cgo && fak_cuda_v41_deferred_full_qualification && fak_cuda_v41_routed_mmid

package model

import (
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime integration est=3s lane=cuda
// Executes the actual two routed model consumers on CUDA with deterministic
// Q2_K/Q3_K fixture weights. This is pipeline integration, not full-model
// numerical/performance qualification; existing MMID parity tests remain required.
func TestV41RoutedFullQualificationCUDA(t *testing.T) {
	if os.Getenv("FAK_CUDA_REQUIRE_DEVICE") != "1" {
		t.Fatal("qualification requires FAK_CUDA_REQUIRE_DEVICE=1; no skip/fallback")
	}
	be, ok := compute.Lookup("cuda")
	if !ok || be.Name() != "cuda" || !be.Caps().DeviceMemory {
		t.Fatal("real CUDA backend required")
	}
	if _, ok := be.(compute.V41DeferredStatusBackend); ok {
		t.Fatal("ordinary CUDA unexpectedly implements the deferred interface")
	}
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41RoutedFusedEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "1")
	m := v41IncrementalExpertFixture(t, true, false)
	if m.Cfg.HiddenSize != 256 || m.Cfg.MoEIntermediateSize != 256 || m.Cfg.NumExpertsPerTok != 6 {
		t.Fatal("pinned fixture geometry changed")
	}
	m.Cfg.SwigluLimit = 10
	routed, admitted := compute.ResolveV41RoutedDeferredStatus(be)
	if !admitted || !routed.SupportsV41DeferredStatus(256) || !routed.SupportsV41DeferredStatus(512) {
		t.Fatal("full routed qualification archive/runtime admission unavailable")
	}
	if generic, present := compute.ResolveV41DeferredStatus(be); present && generic.SupportsV41DeferredStatus(256) {
		t.Fatal("full routed qualifier leaked producer support to generic consumers")
	}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	scope := s.beginBackendExecution()
	defer scope.finish()
	picks, x := v41RoutedLayerFaultPicks(m)
	fn := s.v41RoutedLayerFunc()
	if fn == nil || s.Backend != be {
		t.Fatal("routed model binding unavailable or tensor owner replaced")
	}
	out, outcome, err := fn(0, picks, x)
	if err != nil || outcome != v41RoutedExpertHandled || len(out) != m.Cfg.HiddenSize {
		t.Fatalf("ordinary deferred decode: outcome=%v extent=%d want=%d err=%v", outcome, len(out), m.Cfg.HiddenSize, err)
	}
	for _, value := range out {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatal("decode returned nonfinite output")
		}
	}
	second := make([]float32, len(x))
	for i, value := range x {
		second[i] = v41OracleBF16(value * .5)
	}
	panel, outcome, err := s.v41RoutedExpertPanel(0, "ffn.experts.1", [][]float32{x, second}, []float32{.25, .375}, nil)
	if err != nil || outcome != v41RoutedExpertHandled || len(panel) != 2 {
		t.Fatalf("routed panel: outcome=%v rows=%d err=%v", outcome, len(panel), err)
	}
	for _, row := range panel {
		if len(row) != m.Cfg.HiddenSize {
			t.Fatalf("routed panel width %d != hidden size %d", len(row), m.Cfg.HiddenSize)
		}
		for _, value := range row {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("panel returned nonfinite output")
			}
		}
	}
	t.Log("real model routed ordinary-deferred decode and two-row ID/BF16 panel completed; no full numerical/performance claim")
}
