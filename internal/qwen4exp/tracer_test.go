package qwen4exp

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:runtime fast est=10ms
func TestDiagnosticStubRecordsBoundedTruthfulSpine(t *testing.T) {
	embeds, projects := 0, 0
	receipt, err := RunDiagnosticStub([]int{7, 11}, StubSeams{
		Embed: func(tokenID int) ([]float32, string, error) {
			embeds++
			row := make([]float32, StubHiddenSize)
			for i := range row {
				row[i] = float32(tokenID*StubHiddenSize + i)
			}
			return row, "sha256:embed", nil
		},
		ProjectSamples: func(hidden []float32, indices []int) ([]float32, string, error) {
			projects++
			values := make([]float32, len(indices))
			for i, index := range indices {
				values[i] = hidden[index%len(hidden)]
			}
			return values, "sha256:head", nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if embeds != 2 || projects != 2 {
		t.Fatalf("seam calls embed/project = %d/%d, want 2/2", embeds, projects)
	}
	if receipt.Mode != "diagnostic_stub" || receipt.Layers != StubLayers || receipt.Tokens != 2 {
		t.Fatalf("receipt metadata = %#v", receipt)
	}
	if got, want := len(receipt.Artifact.Events), 2*(StubLayers+2); got != want {
		t.Fatalf("events=%d want %d", got, want)
	}
	if receipt.Artifact.Dropped != 0 || receipt.Artifact.DroppedSampleValues != 0 {
		t.Fatalf("bounded trace dropped data: %#v", receipt.Artifact)
	}
	for _, event := range receipt.Artifact.Events {
		if event.Backend != "diagnostic-stub" || event.Status != "stub_missing_native_layer_math" {
			t.Fatalf("event is not explicitly stub-labeled: %#v", event)
		}
		if event.Phase == "residual_streams" {
			if event.Candidate != "identity-replicated-hc4-missing-real-layer" || len(event.Shapes) != 1 || event.Shapes[0][0] != StubHCCount || event.Shapes[0][1] != StubHiddenSize {
				t.Fatalf("identity residual is not truthful HC4 placeholder: %#v", event)
			}
		}
	}
}

// fak-test:runtime fast est=10ms
func TestDiagnosticStubComparatorPlumbingFindsLayerZero(t *testing.T) {
	receipt, err := RunDiagnosticStub([]int{3}, deterministicSeams())
	if err != nil {
		t.Fatal(err)
	}
	reference := receipt.Artifact
	reference.Events = append([]computetrace.Event(nil), receipt.Artifact.Events...)
	for i := range reference.Events {
		if reference.Events[i].Layer == 0 && reference.Events[i].Phase == "residual_streams" {
			reference.Events[i].SampleValues = append([]float32(nil), reference.Events[i].SampleValues...)
			reference.Events[i].SampleValues[0]++
			break
		}
	}
	report := computetrace.FirstDivergence(reference, receipt.Artifact, computetrace.DivergenceTolerance{RelL2: 0}, computetrace.ActivationTolerance{})
	if report.FirstDivergent == nil || report.FirstDivergent.Layer != 0 || report.FirstDivergent.Stage != "residual_streams" {
		t.Fatalf("altered diagnostic reference first divergence = %#v", report.FirstDivergent)
	}
}

// fak-test:runtime fast est=1ms
func TestDiagnosticStubRejectsInvalidSeamsAndWidth(t *testing.T) {
	if _, err := RunDiagnosticStub(nil, StubSeams{}); !errors.Is(err, ErrDiagnosticStub) {
		t.Fatalf("empty input error=%v", err)
	}
	seams := deterministicSeams()
	seams.Embed = func(int) ([]float32, string, error) { return []float32{1}, "digest", nil }
	if _, err := RunDiagnosticStub([]int{1}, seams); !errors.Is(err, ErrDiagnosticStub) {
		t.Fatalf("bad width error=%v", err)
	}
	if _, err := RunDiagnosticStub([]int{-1}, deterministicSeams()); !errors.Is(err, ErrDiagnosticStub) {
		t.Fatalf("negative token error=%v", err)
	}
	seams = deterministicSeams()
	seams.ProjectSamples = func(_ []float32, indices []int) ([]float32, string, error) {
		values := make([]float32, len(indices))
		values[0] = float32(math.NaN())
		return values, "digest", nil
	}
	if _, err := RunDiagnosticStub([]int{1}, seams); !errors.Is(err, ErrDiagnosticStub) {
		t.Fatalf("non-finite projection error=%v", err)
	}
}

func deterministicSeams() StubSeams {
	return StubSeams{
		InputDigest: "sha256:input",
		Embed: func(tokenID int) ([]float32, string, error) {
			row := make([]float32, StubHiddenSize)
			for i := range row {
				row[i] = float32(tokenID+i) / StubHiddenSize
			}
			return row, "sha256:embed", nil
		},
		ProjectSamples: func(hidden []float32, indices []int) ([]float32, string, error) {
			values := make([]float32, len(indices))
			for i := range values {
				values[i] = hidden[i%len(hidden)]
			}
			return values, "sha256:head", nil
		},
	}
}
