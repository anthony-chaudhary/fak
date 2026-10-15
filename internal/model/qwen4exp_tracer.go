package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/internal/qwen4exp"
)

// Qwen4ExpDiagnosticReceipt composes primitive run metadata with the shared trace.
type Qwen4ExpDiagnosticReceipt struct {
	qwen4exp.StubReceipt
	Artifact computetrace.Artifact `json:"artifact"`
}

func RunQwen4ExpDiagnosticStub(cfg Config, tokenIDs []int, seams qwen4exp.StubSeams) (Qwen4ExpDiagnosticReceipt, error) {
	if !cfg.IsQwen4Exp() || cfg.Qwen4Exp == nil {
		return Qwen4ExpDiagnosticReceipt{}, fmt.Errorf("%w: require qwen4exp config", qwen4exp.ErrDiagnosticStub)
	}
	if cfg.NumLayers != qwen4exp.StubLayers || cfg.HiddenSize != qwen4exp.StubHiddenSize || cfg.VocabSize != qwen4exp.StubVocabSize || cfg.Qwen4Exp.HCCount != qwen4exp.StubHCCount {
		return Qwen4ExpDiagnosticReceipt{}, fmt.Errorf("%w: geometry layers=%d hidden=%d vocab=%d hc=%d, want %d/%d/%d/%d", qwen4exp.ErrDiagnosticStub, cfg.NumLayers, cfg.HiddenSize, cfg.VocabSize, cfg.Qwen4Exp.HCCount, qwen4exp.StubLayers, qwen4exp.StubHiddenSize, qwen4exp.StubVocabSize, qwen4exp.StubHCCount)
	}
	var recorder *computetrace.Recorder
	metadata, err := qwen4exp.RunDiagnosticStub(tokenIDs, seams, func(eventCount int) qwen4exp.StubTrace {
		recorder = computetrace.NewWithSampleBudget(eventCount, eventCount*qwen4exp.MaxStubSamplesPerEvent)
		return qwen4ExpStubTrace{recorder: recorder}
	})
	if err != nil {
		return Qwen4ExpDiagnosticReceipt{}, err
	}
	artifact := recorder.Artifact()
	artifact.InputTokens = append([]int(nil), tokenIDs...)
	return Qwen4ExpDiagnosticReceipt{StubReceipt: metadata, Artifact: artifact}, nil
}

type qwen4ExpStubTrace struct{ recorder *computetrace.Recorder }

func (qwen4ExpStubTrace) Digest(parts ...string) string { return computetrace.Digest(parts...) }

func (qwen4ExpStubTrace) SampleIndices(width, limit int) []int {
	return computetrace.ActivationSampleIndices(width, limit)
}

func (trace qwen4ExpStubTrace) Record(layer, token int, phase string, shape []int, indices []int, values []float32, inputDigest, weightDigest, candidate string) {
	trace.recorder.Record(computetrace.Event{
		Operation:        "qwen4exp_diagnostic_stub",
		Phase:            phase,
		Backend:          "diagnostic-stub",
		Device:           "host",
		Candidate:        candidate,
		Route:            "qwen4exp/diagnostic-stub/identity-only",
		InputDType:       "f32",
		WeightDType:      "source",
		OutputDType:      "f32",
		Shapes:           [][]int{shape},
		Status:           "stub_missing_native_layer_math",
		Layer:            layer,
		Token:            token,
		InputDigest:      inputDigest,
		WeightDigest:     weightDigest,
		SampleIndices:    indices,
		SampleValues:     values,
		ProvenanceDigest: computetrace.Digest("qwen4exp", "diagnostic-stub", candidate),
	})
}
