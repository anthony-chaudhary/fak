package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen4exp"
)

// fak-test:runtime fast est=10ms
func TestRunQwen4ExpDiagnosticStubRequiresExactGeometry(t *testing.T) {
	cfg := Config{ModelType: "qwen4exp", NumLayers: qwen4exp.StubLayers, HiddenSize: qwen4exp.StubHiddenSize, VocabSize: qwen4exp.StubVocabSize, Qwen4Exp: &Qwen4ExpHParams{HCCount: qwen4exp.StubHCCount}}
	calls := 0
	seams := qwen4exp.StubSeams{
		Embed: func(int) ([]float32, string, error) {
			calls++
			return make([]float32, qwen4exp.StubHiddenSize), "sha256:embed", nil
		},
		ProjectSamples: func(_ []float32, indices []int) ([]float32, string, error) {
			calls++
			return make([]float32, len(indices)), "sha256:head", nil
		},
	}
	if _, err := RunQwen4ExpDiagnosticStub(cfg, []int{1}, seams); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("seam calls=%d want 2", calls)
	}
	bad := cfg
	bad.HiddenSize--
	calls = 0
	if _, err := RunQwen4ExpDiagnosticStub(bad, []int{1}, seams); !errors.Is(err, qwen4exp.ErrDiagnosticStub) {
		t.Fatalf("bad geometry error=%v", err)
	}
	if calls != 0 {
		t.Fatalf("invalid geometry reached weight seams %d times", calls)
	}
}

// fak-test:runtime fast est=1ms
func TestQwen4ExpDiagnosticStubDoesNotOpenNativeExecution(t *testing.T) {
	cfg := Config{ModelType: "qwen4exp", NumLayers: qwen4exp.StubLayers, HiddenSize: qwen4exp.StubHiddenSize, VocabSize: qwen4exp.StubVocabSize, Qwen4Exp: &Qwen4ExpHParams{HCCount: qwen4exp.StubHCCount}}
	if err := refuseQwen4ExpNative(cfg); err == nil {
		t.Fatal("generic qwen4exp execution is no longer refused")
	}
}
