package qwen4exp

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

const (
	StubLayers     = 48
	StubHiddenSize = 2560
	StubHCCount    = 4
	StubVocabSize  = 248320
	MaxStubTokens  = 8
)

var ErrDiagnosticStub = errors.New("qwen4exp diagnostic stub")

type StubSeams struct {
	InputDigest    string
	Embed          func(tokenID int) (row []float32, weightDigest string, err error)
	ProjectSamples func(hidden []float32, sampleIndices []int) (values []float32, weightDigest string, err error)
}

type StubReceipt struct {
	Mode          string                `json:"mode"`
	Layers        int                   `json:"layers"`
	Tokens        int                   `json:"tokens"`
	MissingStages []string              `json:"missing_stages"`
	Artifact      computetrace.Artifact `json:"artifact"`
}

func RunDiagnosticStub(tokenIDs []int, seams StubSeams) (StubReceipt, error) {
	if len(tokenIDs) == 0 || len(tokenIDs) > MaxStubTokens {
		return StubReceipt{}, fmt.Errorf("%w: token count must be in [1,%d]", ErrDiagnosticStub, MaxStubTokens)
	}
	if seams.Embed == nil || seams.ProjectSamples == nil {
		return StubReceipt{}, fmt.Errorf("%w: embedding and projection seams are required", ErrDiagnosticStub)
	}
	if seams.InputDigest == "" {
		parts := make([]string, len(tokenIDs))
		for i, id := range tokenIDs {
			parts[i] = strconv.Itoa(id)
		}
		seams.InputDigest = computetrace.Digest(parts...)
	}
	for _, tokenID := range tokenIDs {
		if tokenID < 0 || tokenID >= StubVocabSize {
			return StubReceipt{}, fmt.Errorf("%w: token id %d outside [0,%d)", ErrDiagnosticStub, tokenID, StubVocabSize)
		}
	}

	eventCount := len(tokenIDs) * (StubLayers + 2)
	recorder := computetrace.NewWithSampleBudget(eventCount, eventCount*computetrace.MaxActivationSamplesPerEvent)
	residualIndices := computetrace.ActivationSampleIndices(StubHiddenSize*StubHCCount, computetrace.MaxActivationSamplesPerEvent)
	logitIndices := computetrace.ActivationSampleIndices(StubVocabSize, computetrace.MaxActivationSamplesPerEvent)

	for token, tokenID := range tokenIDs {
		hidden, embedDigest, err := seams.Embed(tokenID)
		if err != nil {
			return StubReceipt{}, fmt.Errorf("%w: embed token %d: %v", ErrDiagnosticStub, tokenID, err)
		}
		if len(hidden) != StubHiddenSize {
			return StubReceipt{}, fmt.Errorf("%w: embedding width=%d want %d", ErrDiagnosticStub, len(hidden), StubHiddenSize)
		}
		if index := firstNonFinite(hidden); index >= 0 {
			return StubReceipt{}, fmt.Errorf("%w: embedding token %d has non-finite value at %d", ErrDiagnosticStub, tokenID, index)
		}
		embedIndices := computetrace.ActivationSampleIndices(StubHiddenSize, computetrace.MaxActivationSamplesPerEvent)
		recordStubEvent(recorder, -1, token, "embedding", []int{StubHiddenSize}, embedIndices, gather(hidden, embedIndices), seams.InputDigest, embedDigest, "real-embedding")

		residualSamples := make([]float32, len(residualIndices))
		for i, index := range residualIndices {
			residualSamples[i] = hidden[index%StubHiddenSize]
		}
		for layer := 0; layer < StubLayers; layer++ {
			recordStubEvent(recorder, layer, token, "residual_streams", []int{StubHCCount, StubHiddenSize}, residualIndices, residualSamples, seams.InputDigest, "stub:no-layer-weights", "identity-replicated-hc4-missing-real-layer")
		}

		logits, headDigest, err := seams.ProjectSamples(hidden, append([]int(nil), logitIndices...))
		if err != nil {
			return StubReceipt{}, fmt.Errorf("%w: project token %d: %v", ErrDiagnosticStub, tokenID, err)
		}
		if len(logits) != len(logitIndices) {
			return StubReceipt{}, fmt.Errorf("%w: projection samples=%d want %d", ErrDiagnosticStub, len(logits), len(logitIndices))
		}
		if index := firstNonFinite(logits); index >= 0 {
			return StubReceipt{}, fmt.Errorf("%w: projection token %d has non-finite sample at %d", ErrDiagnosticStub, tokenID, index)
		}
		recordStubEvent(recorder, StubLayers, token, "logits", []int{StubVocabSize}, logitIndices, logits, seams.InputDigest, headDigest, "real-sampled-projection-of-stub-hidden")
	}

	artifact := recorder.Artifact()
	artifact.InputTokens = append([]int(nil), tokenIDs...)
	return StubReceipt{
		Mode:          "diagnostic_stub",
		Layers:        StubLayers,
		Tokens:        len(tokenIDs),
		MissingStages: []string{"ple", "gdn_or_qsa_attention", "hyper_connection_mix", "moe", "output_hc_mix_and_group_norm"},
		Artifact:      artifact,
	}, nil
}

func firstNonFinite(values []float32) int {
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return i
		}
	}
	return -1
}

func recordStubEvent(recorder *computetrace.Recorder, layer, token int, phase string, shape []int, indices []int, values []float32, inputDigest, weightDigest, candidate string) {
	recorder.Record(computetrace.Event{
		Operation:        "qwen4exp_diagnostic_stub",
		Phase:            phase,
		Backend:          "diagnostic-stub",
		Device:           "host",
		Candidate:        candidate,
		Route:            "qwen4exp/diagnostic-stub/identity-only",
		InputDType:       "f32",
		WeightDType:      "source",
		OutputDType:      "f32",
		Shapes:           [][]int{append([]int(nil), shape...)},
		Status:           "stub_missing_native_layer_math",
		Layer:            layer,
		Token:            token,
		InputDigest:      inputDigest,
		WeightDigest:     weightDigest,
		SampleIndices:    append([]int(nil), indices...),
		SampleValues:     append([]float32(nil), values...),
		ProvenanceDigest: computetrace.Digest("qwen4exp", "diagnostic-stub", candidate),
	})
}

func gather(row []float32, indices []int) []float32 {
	values := make([]float32, len(indices))
	for i, index := range indices {
		values[i] = row[index]
	}
	return values
}
