package qwen4exp

import (
	"errors"
	"fmt"
	"math"
	"strconv"
)

const (
	StubLayers             = 48
	StubHiddenSize         = 2560
	StubHCCount            = 4
	StubVocabSize          = 248320
	MaxStubTokens          = 8
	MaxStubSamplesPerEvent = 32
)

var ErrDiagnosticStub = errors.New("qwen4exp diagnostic stub")

type StubSeams struct {
	InputDigest    string
	Embed          func(tokenID int) (row []float32, weightDigest string, err error)
	ProjectSamples func(hidden []float32, sampleIndices []int) (values []float32, weightDigest string, err error)
}

type StubReceipt struct {
	Mode          string   `json:"mode"`
	Layers        int      `json:"layers"`
	Tokens        int      `json:"tokens"`
	MissingStages []string `json:"missing_stages"`
}

// StubTrace is supplied by the composing layer; qwen4exp owns no artifact schema.
// Record receives owned copies of its slice arguments.
type StubTrace interface {
	Digest(parts ...string) string
	SampleIndices(width, limit int) []int
	Record(layer, token int, phase string, shape, indices []int, values []float32, inputDigest, weightDigest, candidate string)
}

func RunDiagnosticStub(tokenIDs []int, seams StubSeams, newTrace func(eventCount int) StubTrace) (StubReceipt, error) {
	if len(tokenIDs) == 0 || len(tokenIDs) > MaxStubTokens {
		return StubReceipt{}, fmt.Errorf("%w: token count must be in [1,%d]", ErrDiagnosticStub, MaxStubTokens)
	}
	if seams.Embed == nil || seams.ProjectSamples == nil {
		return StubReceipt{}, fmt.Errorf("%w: embedding and projection seams are required", ErrDiagnosticStub)
	}
	for _, tokenID := range tokenIDs {
		if tokenID < 0 || tokenID >= StubVocabSize {
			return StubReceipt{}, fmt.Errorf("%w: token id %d outside [0,%d)", ErrDiagnosticStub, tokenID, StubVocabSize)
		}
	}

	if newTrace == nil {
		return StubReceipt{}, fmt.Errorf("%w: trace factory is required", ErrDiagnosticStub)
	}
	trace := newTrace(len(tokenIDs) * (StubLayers + 2))
	if trace == nil {
		return StubReceipt{}, fmt.Errorf("%w: trace sink is required", ErrDiagnosticStub)
	}
	if seams.InputDigest == "" {
		parts := make([]string, len(tokenIDs))
		for i, id := range tokenIDs {
			parts[i] = strconv.Itoa(id)
		}
		seams.InputDigest = trace.Digest(parts...)
	}
	residualIndices, err := stubSampleIndices(trace, StubHiddenSize*StubHCCount)
	if err != nil {
		return StubReceipt{}, err
	}
	logitIndices, err := stubSampleIndices(trace, StubVocabSize)
	if err != nil {
		return StubReceipt{}, err
	}
	embedIndices, err := stubSampleIndices(trace, StubHiddenSize)
	if err != nil {
		return StubReceipt{}, err
	}

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
		recordStubEvent(trace, -1, token, "embedding", []int{StubHiddenSize}, embedIndices, gather(hidden, embedIndices), seams.InputDigest, embedDigest, "real-embedding")

		residualSamples := make([]float32, len(residualIndices))
		for i, index := range residualIndices {
			residualSamples[i] = hidden[index%StubHiddenSize]
		}
		for layer := 0; layer < StubLayers; layer++ {
			recordStubEvent(trace, layer, token, "residual_streams", []int{StubHCCount, StubHiddenSize}, residualIndices, residualSamples, seams.InputDigest, "stub:no-layer-weights", "identity-replicated-hc4-missing-real-layer")
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
		recordStubEvent(trace, StubLayers, token, "logits", []int{StubVocabSize}, logitIndices, logits, seams.InputDigest, headDigest, "real-sampled-projection-of-stub-hidden")
	}

	return StubReceipt{
		Mode:          "diagnostic_stub",
		Layers:        StubLayers,
		Tokens:        len(tokenIDs),
		MissingStages: []string{"ple", "gdn_or_qsa_attention", "hyper_connection_mix", "moe", "output_hc_mix_and_group_norm"},
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

func stubSampleIndices(trace StubTrace, width int) ([]int, error) {
	indices := trace.SampleIndices(width, MaxStubSamplesPerEvent)
	if len(indices) != min(width, MaxStubSamplesPerEvent) {
		return nil, fmt.Errorf("%w: trace sample count=%d for width=%d", ErrDiagnosticStub, len(indices), width)
	}
	for i, index := range indices {
		if index < 0 || index >= width || (i > 0 && index <= indices[i-1]) {
			return nil, fmt.Errorf("%w: invalid trace sample index %d for width=%d", ErrDiagnosticStub, index, width)
		}
	}
	return append([]int(nil), indices...), nil
}

func recordStubEvent(trace StubTrace, layer, token int, phase string, shape []int, indices []int, values []float32, inputDigest, weightDigest, candidate string) {
	trace.Record(layer, token, phase, append([]int(nil), shape...), append([]int(nil), indices...), append([]float32(nil), values...), inputDigest, weightDigest, candidate)
}

func gather(row []float32, indices []int) []float32 {
	values := make([]float32, len(indices))
	for i, index := range indices {
		values[i] = row[index]
	}
	return values
}
