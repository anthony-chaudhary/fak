package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestSessionTargetHiddenCapturesExactPreFinalNorm(t *testing.T) {
	cfg := cfgV(8, 2, 2, 1, 4, 16)
	m := NewSynthetic(cfg)
	ordinary := m.NewSession()
	ordinary.tokenHidden(7, 0)
	if _, err := ordinary.TargetHiddenAt(0); err == nil {
		t.Fatal("ordinary session captured target hidden without the opt-in MTP path")
	}
	ordinary.Close()

	gotSession := m.NewSession()
	gotSession.captureTargetHidden = true
	t.Cleanup(gotSession.Close)

	token := 7
	gotNormed := gotSession.tokenHidden(token, 0)
	gotHidden, err := gotSession.TargetHiddenAt(0)
	if err != nil {
		t.Fatalf("target hidden at position 0: %v", err)
	}

	ref := m.NewSession()
	t.Cleanup(ref.Close)
	x := append([]float32(nil), m.embedRows()[token*cfg.HiddenSize:(token+1)*cfg.HiddenSize]...)
	scaleEmbedInPlace(x, cfg)
	for layer := 0; layer < cfg.NumLayers; layer++ {
		cos, sin := ropeRowForLayer(cfg, layer, 0)
		x = ref.blockStep(layer, 0, x, cos, sin, f32Kernel{m})
	}
	ref.Cache.appendPosition(0, token)

	assertFloat32BitsEqual(t, "captured pre-final-norm hidden", x, gotHidden)
	assertFloat32BitsEqual(t, "tokenHidden final norm", m.finalNorm(x), gotNormed)
	if reflect.DeepEqual(gotHidden, gotNormed) {
		t.Fatal("captured hidden equals post-final-norm output; witness needs a genuine pre-final-norm vector")
	}
}

func TestSessionTargetHiddenHistoryIsPositionalAndDefensive(t *testing.T) {
	m := NewSynthetic(cfgV(8, 2, 2, 1, 4, 16))
	s := m.NewSession()
	s.captureTargetHidden = true
	t.Cleanup(s.Close)

	s.tokenHidden(3, 0)
	first, err := s.TargetHiddenAt(0)
	if err != nil {
		t.Fatalf("target hidden at position 0: %v", err)
	}
	first[0] = float32(math.Inf(1))
	again, err := s.TargetHiddenAt(0)
	if err != nil {
		t.Fatalf("second target hidden at position 0: %v", err)
	}
	if math.IsInf(float64(again[0]), 1) {
		t.Fatal("TargetHiddenAt returned session-owned storage instead of a defensive copy")
	}
	if _, err := s.TargetHiddenAt(1); err == nil {
		t.Fatal("position 1 hidden became visible before the target evaluated position 1")
	}

	s.tokenHidden(5, 1)
	second, err := s.TargetHiddenAt(1)
	if err != nil {
		t.Fatalf("target hidden at position 1: %v", err)
	}
	if reflect.DeepEqual(again, second) {
		t.Fatal("position 0 and position 1 hidden vectors unexpectedly alias or repeat")
	}

	if removed := s.evictKV(1, 1); removed != 1 {
		t.Fatalf("evicted positions = %d, want 1", removed)
	}
	if _, err := s.TargetHiddenAt(1); err == nil {
		t.Fatal("rolled-back position 1 hidden remained visible")
	}
	stillFirst, err := s.TargetHiddenAt(0)
	if err != nil {
		t.Fatalf("position 0 after rollback: %v", err)
	}
	assertFloat32BitsEqual(t, "position 0 survives rollback", again, stillFirst)

	s.tokenHidden(9, 1)
	replacement, err := s.TargetHiddenAt(1)
	if err != nil {
		t.Fatalf("replacement target hidden at position 1: %v", err)
	}
	if reflect.DeepEqual(second, replacement) {
		t.Fatal("re-evaluated position 1 retained the evicted token's hidden state")
	}

	s.tokenHidden(11, 2)
	if removed := s.evictKV(1, 1); removed != 1 {
		t.Fatalf("middle-evicted positions = %d, want 1", removed)
	}
	if _, err := s.TargetHiddenAt(1); err == nil {
		t.Fatal("compacted position 1 exposed hidden captured for the evicted token")
	}
}

func TestSessionTokenEmbeddingUsesTargetTableWithBoundsAndCopies(t *testing.T) {
	m := NewSynthetic(cfgV(8, 1, 2, 1, 4, 16))
	s := m.NewSession()
	t.Cleanup(s.Close)

	token := 11
	got, err := s.TokenEmbedding(token)
	if err != nil {
		t.Fatalf("token embedding %d: %v", token, err)
	}
	start := token * m.Cfg.HiddenSize
	want := append([]float32(nil), m.embedRows()[start:start+m.Cfg.HiddenSize]...)
	assertFloat32BitsEqual(t, "target embedding row", want, got)

	got[0] = float32(math.Inf(-1))
	again, err := s.TokenEmbedding(token)
	if err != nil {
		t.Fatalf("second token embedding %d: %v", token, err)
	}
	if math.IsInf(float64(again[0]), -1) {
		t.Fatal("TokenEmbedding returned the target model's mutable embedding storage")
	}
	for _, bad := range []int{-1, m.Cfg.VocabSize} {
		if _, err := s.TokenEmbedding(bad); err == nil {
			t.Fatalf("TokenEmbedding(%d) succeeded outside vocab [0,%d)", bad, m.Cfg.VocabSize)
		}
	}
}

func qwen35MTPEnabledSyntheticModel(t *testing.T) *Model {
	t.Helper()
	mtp := qwen35MTPTinyForwardModel(t)
	cfg := mtp.Cfg
	cfg.LayerTypes = []string{"full_attention"}
	cfg.RMSNormEps = 1e-5
	m := NewSynthetic(cfg)

	for _, name := range qwen35MTPRequiredTensors {
		meta := mtp.manifest[name]
		start := len(m.raw)
		m.raw = append(m.raw, mtp.raw[meta.Offset:meta.Offset+meta.Nbytes]...)
		meta.Offset = start
		m.manifest[name] = meta
	}
	mode, err := m.Qwen35MTPMode(false)
	if err != nil {
		t.Fatalf("MTP-enabled synthetic mode: %v", err)
	}
	if !mode.Enabled {
		t.Fatalf("MTP-enabled synthetic mode = %+v, want enabled", mode)
	}
	return m
}

func assertQwen35MTPUnsupported(t *testing.T, err error) *Qwen35MTPSpecDecodeUnsupportedError {
	t.Helper()
	var unsupported *Qwen35MTPSpecDecodeUnsupportedError
	if !errors.As(err, &unsupported) || !errors.Is(err, ErrQwen35MTPSpecDecodeUnsupported) {
		t.Fatalf("error = %v, want typed Qwen35 MTP speculative-decode unsupported verdict", err)
	}
	return unsupported
}
