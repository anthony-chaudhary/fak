package qwen4exp

import (
	"errors"
	"testing"
)

type stubTraceProbe struct {
	t       *testing.T
	bad     string
	records int
	samples [][]int
}

func (*stubTraceProbe) Digest(...string) string { return "input-digest" }

func (p *stubTraceProbe) SampleIndices(width, limit int) []int {
	indices := make([]int, min(width, limit))
	for i := range indices {
		indices[i] = i * (width / len(indices))
	}
	switch p.bad {
	case "empty":
		indices = nil
	case "short":
		indices = indices[:len(indices)-1]
	case "long":
		indices = append(indices, width-1)
	case "negative":
		indices[0] = -1
	case "outside":
		indices[len(indices)-1] = width
	case "duplicate":
		indices[1] = indices[0]
	}
	p.samples = append(p.samples, indices)
	return indices
}

func (p *stubTraceProbe) Record(layer, token int, phase string, shape, indices []int, values []float32, inputDigest, weightDigest, candidate string) {
	p.records++
	if len(indices) != MaxStubSamplesPerEvent || indices[0] != 0 || len(values) != len(indices) || values[0] != 7 || inputDigest != "input-digest" {
		p.t.Fatalf("borrowed trace data changed at %s layer %d: indices=%v values=%v digest=%q", phase, layer, indices, values, inputDigest)
	}
	// Hostile retention/mutation cannot affect the next residual event or projection.
	shape[0], indices[0], values[0] = -1, -1, -1
}

// fak-test:runtime fast est=10ms
func TestDiagnosticStubTraceSinkBoundary(t *testing.T) {
	calls := 0
	seams := StubSeams{
		Embed: func(int) ([]float32, string, error) {
			calls++
			row := make([]float32, StubHiddenSize)
			for i := range row {
				row[i] = 7
			}
			return row, "embed-digest", nil
		},
		ProjectSamples: func(hidden []float32, indices []int) ([]float32, string, error) {
			calls++
			if hidden[0] != 7 || indices[0] != 0 || len(indices) != MaxStubSamplesPerEvent {
				t.Fatal("trace sink changed projection input")
			}
			indices[0] = -1 // The emitted logit indices must still be isolated.
			values := make([]float32, len(indices))
			for i := range values {
				values[i] = 7
			}
			return values, "head-digest", nil
		},
	}
	sink := &stubTraceProbe{t: t}
	factories := 0
	receipt, err := RunDiagnosticStub([]int{7}, seams, func(count int) StubTrace {
		factories++
		if count != StubLayers+2 {
			t.Fatalf("event budget=%d", count)
		}
		return sink
	})
	if err != nil || receipt.Mode != "diagnostic_stub" || calls != 2 || factories != 1 || sink.records != StubLayers+2 {
		t.Fatalf("receipt=%+v error=%v seam calls=%d factories=%d records=%d", receipt, err, calls, factories, sink.records)
	}
	for _, indices := range sink.samples {
		if indices[0] != 0 {
			t.Fatal("sink-owned sample indices were mutated")
		}
	}
	for _, bad := range []string{"empty", "short", "long", "negative", "outside", "duplicate"} {
		t.Run(bad, func(t *testing.T) {
			calls = 0
			sink := &stubTraceProbe{t: t, bad: bad}
			_, err := RunDiagnosticStub([]int{7}, seams, func(int) StubTrace { return sink })
			if !errors.Is(err, ErrDiagnosticStub) || calls != 0 || sink.records != 0 {
				t.Fatalf("error=%v seam calls=%d records=%d", err, calls, sink.records)
			}
		})
	}
}
