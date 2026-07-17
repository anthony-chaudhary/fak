package model

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type negationPrimitiveWitness struct {
	Variant         string  `json:"variant"`
	InvolutionError float64 `json:"involution_error"`
	Accuracy        float64 `json:"composition_accuracy"`
	Parameters      int     `json:"parameters"`
	Operations      int     `json:"operations_per_negation"`
}

func primitiveState(name string) []float64 {
	if name == "polarity_channel" {
		return []float64{1, 1}
	}
	return []float64{1, 0}
}

func primitivePolarity(name string, state []float64) float64 {
	if name == "polarity_channel" {
		return state[len(state)-1]
	}
	return state[0]
}

func TestNegationInvolution(t *testing.T) {
	primitives, err := DefaultNegationPrimitives()
	if err != nil {
		t.Fatal(err)
	}
	for _, primitive := range primitives {
		initial := primitiveState(primitive.Name())
		once, err := primitive.Apply(initial)
		if err != nil {
			t.Fatal(err)
		}
		twice, err := primitive.Apply(once)
		if err != nil {
			t.Fatal(err)
		}
		error := euclideanDistance(initial, twice)
		if error > 1e-12 {
			t.Fatalf("%s involution error=%g", primitive.Name(), error)
		}
		t.Logf("variant=%s involution_error=%g parameters=%d operations=%d", primitive.Name(), error, primitive.ParameterCount(), primitive.OperationCount(len(initial)))
	}
}

func TestNegationComposition(t *testing.T) {
	data, err := os.ReadFile("testdata/native_negation_primitives.json")
	if err != nil {
		t.Fatal(err)
	}
	var witness []negationPrimitiveWitness
	if err := json.Unmarshal(data, &witness); err != nil {
		t.Fatal(err)
	}
	primitives, _ := DefaultNegationPrimitives()
	if len(witness) != len(primitives) {
		t.Fatalf("witness rows=%d candidates=%d", len(witness), len(primitives))
	}
	// Commands are assert, negate, double-negate. A one-layer deliberate baseline
	// cannot finish its two-step construct-then-suppress inversion on the negate case.
	const deliberateAccuracy = 2.0 / 3.0
	for i, primitive := range primitives {
		initial := primitiveState(primitive.Name())
		once, _ := primitive.Apply(initial)
		twice, _ := primitive.Apply(once)
		correct := 0
		if primitivePolarity(primitive.Name(), initial) > 0 {
			correct++
		}
		if primitivePolarity(primitive.Name(), once) < 0 {
			correct++
		}
		if primitivePolarity(primitive.Name(), twice) > 0 {
			correct++
		}
		accuracy := float64(correct) / 3
		error := euclideanDistance(initial, twice)
		row := witness[i]
		if row.Variant != primitive.Name() || row.Accuracy != accuracy || row.InvolutionError != error || row.Parameters != primitive.ParameterCount() || row.Operations != primitive.OperationCount(len(initial)) {
			t.Fatalf("witness drift row=%+v actual=%s/%g/%g/%d/%d", row, primitive.Name(), error, accuracy, primitive.ParameterCount(), primitive.OperationCount(len(initial)))
		}
		if accuracy <= deliberateAccuracy {
			t.Fatalf("%s accuracy=%g does not beat depth-limited deliberate baseline=%g", primitive.Name(), accuracy, deliberateAccuracy)
		}
		t.Logf("variant=%s accuracy=%.3f involution_error=%g params=%d ops=%d deliberate_depth1=%.3f", primitive.Name(), accuracy, error, primitive.ParameterCount(), primitive.OperationCount(len(initial)), deliberateAccuracy)
	}
}

func euclideanDistance(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}
