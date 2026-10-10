package model

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=20ms lane=default
func TestValidateBackendForwardConfigRejectsSlidingWindow(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	be := compute.Pick("cpu-ref")
	if be == nil {
		t.Fatal("cpu-ref backend missing")
	}
	base := Config{HiddenSize: 16, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 4, IntermediateSize: 32, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1}
	checkRefusal := func(t *testing.T, err error) {
		t.Helper()
		var unsupported *UnsupportedBackendForwardError
		if !errors.As(err, &unsupported) || unsupported.Forward != forwardGenericHAL || unsupported.Backend != be.Name() || unsupported.IntendedPath != "compute HAL" {
			t.Fatalf("wrong typed window refusal: %T %v", err, err)
		}
		if !strings.Contains(unsupported.Reason, "sliding-window") || !strings.Contains(unsupported.Reason, "unwindowed") || strings.Contains(err.Error(), "issue #4714") {
			t.Fatalf("wrong generic window diagnostic: %v", err)
		}
	}
	for _, tc := range []struct {
		name   string
		window []int
		reject bool
	}{
		{name: "all windowed", window: []int{2, 2}, reject: true},
		{name: "later window", window: []int{-1, 2}, reject: true},
		{name: "short window", window: []int{1}, reject: true},
		{name: "nil window"},
		{name: "empty window", window: []int{}},
		{name: "full attention", window: []int{-1, -1}},
		{name: "short full attention", window: []int{-1}},
		{name: "unused trailing window", window: []int{-1, -1, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			cfg.Window = tc.window
			err := ValidateBackendForwardConfig(cfg, be)
			if tc.reject {
				checkRefusal(t, err)
			} else if err != nil {
				t.Fatalf("full-attention control refused: %v", err)
			}
			if err := ValidateBackendForwardConfig(cfg, nil); err != nil {
				t.Fatalf("legacy reference admission changed: %v", err)
			}
		})
	}

	// These controls preserve dedicated admission only. They do not qualify
	// execution of synthetic window configurations on those architectures.
	for _, modelType := range []string{"deepseek41", "deepseek2", "glm_moe_dsa", "minimax_m3", "gemma4"} {
		cfg := base
		cfg.ModelType, cfg.Window = modelType, []int{2, 2}
		if err := ValidateBackendForwardConfig(cfg, be); err != nil {
			t.Fatalf("dedicated route %q changed: %v", modelType, err)
		}
	}
	hybrid := qwen35HybridTestCfg()
	hybrid.Window = []int{2}
	accepted := &pathQwen35Backend{recordingQwen35Backend: &recordingQwen35Backend{}, path: Qwen35GDNCUDAPath}
	if err := ValidateBackendForwardConfig(hybrid, accepted); err != nil {
		t.Fatalf("hybrid structural admission changed: %v", err)
	}
	var unsupportedHybrid *UnsupportedBackendForwardError
	if err := ValidateBackendForwardConfig(hybrid, be); !errors.As(err, &unsupportedHybrid) || unsupportedHybrid.Forward != ForwardQwen35GDN {
		t.Fatalf("hybrid refusal changed: %v", err)
	}

	cfg := base
	cfg.Window = []int{2, 2}
	m := NewSynthetic(cfg)
	defer m.CloseWeights()
	spy := &layerNormAdmissionBackend{Backend: be}
	state := m.ensureWeightCloser()
	state.mu.Lock()
	before := state.sessions
	state.mu.Unlock()
	s, err := m.NewBackendSessionChecked(spy)
	if s != nil {
		s.Close()
		t.Fatal("unwindowed HAL returned a session")
	}
	checkRefusal(t, err)
	state.mu.Lock()
	after := state.sessions
	state.mu.Unlock()
	if before != after || spy.newKV != 0 || spy.resets != 0 || spy.uploads != 0 {
		t.Fatalf("refusal acquired resources: holds=%d->%d kv=%d reset=%d uploads=%d", before, after, spy.newKV, spy.resets, spy.uploads)
	}

	// NewBackendSessionChecked(nil) selects the default HAL; NewSession is the
	// explicitly selected legacy reference path and remains available.
	legacy := m.NewSession()
	defer legacy.Close()
	if legacy.Backend != nil {
		t.Fatal("reference session unexpectedly selected HAL")
	}
	for _, logits := range [][]float32{legacy.Prefill([]int{3, 7, 11}), legacy.Step(19)} {
		if len(logits) != cfg.VocabSize {
			t.Fatalf("reference logits=%d, want %d", len(logits), cfg.VocabSize)
		}
		for _, x := range logits {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatal("reference produced nonfinite logits")
			}
		}
	}
}
