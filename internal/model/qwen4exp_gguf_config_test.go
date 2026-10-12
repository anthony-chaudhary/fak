package model

import (
	"errors"
	"testing"
)

func hybridLayerTypes(n int) []string {
	lt := make([]string, n)
	for l := range lt {
		lt[l] = "linear_attention"
		if (l+1)%4 == 0 {
			lt[l] = "full_attention"
		}
	}
	return lt
}

// fak-test:runtime fast est=1ms
func TestHasPLEEngramKeysOnArchNotVocab(t *testing.T) {
	q27 := Config{ModelType: "qwen35", VocabSize: 248320, NumLayers: 64, LayerTypes: hybridLayerTypes(64)}
	if q27.HasPLEEngram() {
		t.Fatal("Qwen3.8-27B (qwen35, vocab 248320) reported a PLE table")
	}
	flash := Config{ModelType: "qwen4exp", VocabSize: 248320, NumLayers: 48, LayerTypes: hybridLayerTypes(48),
		Qwen4Exp: &Qwen4ExpHParams{PLE: &Qwen4ExpPLE{Layers: []int{1}}}}
	if !flash.HasPLEEngram() {
		t.Fatal("Flash-Next GGUF with ple.layers reported no PLE table")
	}
	flash.Qwen4Exp.PLE = nil
	if flash.HasPLEEngram() {
		t.Fatal("qwen4exp GGUF without ple.* keys reported a PLE table")
	}
}

// fak-test:runtime fast est=1ms
func TestQwen4ExpRefusedOnGenericForward(t *testing.T) {
	cfg := Config{ModelType: "qwen4exp", NumLayers: 48, LayerTypes: hybridLayerTypes(48)}
	_, err := ClassifyForwardPath(cfg, nil)
	var unsupported *UnsupportedQwen4ExpExecutionError
	if !errors.As(err, &unsupported) {
		t.Fatalf("ClassifyForwardPath(qwen4exp) = %v, want UnsupportedQwen4ExpExecutionError", err)
	}
	if _, err := ClassifyForwardPath(Config{ModelType: "qwen35", NumLayers: 4, LayerTypes: hybridLayerTypes(4)}, nil); err != nil {
		t.Fatalf("qwen35 hybrid now refused: %v", err)
	}
}
