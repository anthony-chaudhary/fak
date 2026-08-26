package qwen4exp

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestNativeFourLayerOracleFixture(t *testing.T) {
	oracle := New()
	trace, err := oracle.Run([]int{17, 23})
	if err != nil {
		t.Fatal(err)
	}
	if trace.Engine != Engine {
		t.Fatalf("engine = %q, want %q", trace.Engine, Engine)
	}
	for token, got := range trace.Tokens {
		if len(got.Layers) != len(Cadence) {
			t.Fatalf("token %d layer count = %d", token, len(got.Layers))
		}
		for layer, lt := range got.Layers {
			if lt.Kind != Cadence[layer] {
				t.Fatalf("token %d layer %d kind = %q", token, layer, lt.Kind)
			}
			if len(lt.Route.ExpertIDs) != ExpertsPerToken || len(lt.Route.ExpertWeights) != ExpertsPerToken {
				t.Fatalf("token %d layer %d route is not exact top-%d", token, layer, ExpertsPerToken)
			}
			if !lt.Route.SharedExpert {
				t.Fatalf("token %d layer %d omitted shared expert", token, layer)
			}
			seen := make(map[int]bool, ExpertsPerToken)
			weightSum := float32(0)
			for i, id := range lt.Route.ExpertIDs {
				if id < 0 || id >= NumRoutedExperts || seen[id] {
					t.Fatalf("token %d layer %d invalid expert ID %d", token, layer, id)
				}
				seen[id] = true
				weightSum += lt.Route.ExpertWeights[i]
			}
			if weightSum < 0.99999 || weightSum > 1.00001 {
				t.Fatalf("token %d layer %d route weights sum to %g", token, layer, weightSum)
			}
			if layer < 3 && len(lt.Recurrent) != HiddenSize*HiddenSize {
				t.Fatalf("token %d layer %d recurrent values = %d", token, layer, len(lt.Recurrent))
			}
			if layer == 3 && len(lt.SelectedToken) != token+1 {
				t.Fatalf("token %d sparse IDs = %v", token, lt.SelectedToken)
			}
		}
	}

	got, err := json.MarshalIndent(trace, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	path := "testdata/tiny_prompt_trace.json"
	if os.Getenv("QWEN4EXP_UPDATE_FIXTURE") == "1" {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("native trace drifted from pinned fixture; regenerate only against the selected parity oracle\ngot:\n%s", got)
	}
}

func TestStateRestoreIsBitExact(t *testing.T) {
	first := New()
	if _, err := first.Run([]int{17}); err != nil {
		t.Fatal(err)
	}
	encoded := first.State().MarshalBinary()
	var restored State
	if err := restored.UnmarshalBinary(encoded); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded, restored.MarshalBinary()) {
		t.Fatal("FP32 recurrent state changed bits during restore")
	}

	resumed := New()
	resumed.Restore(restored)
	got, err := resumed.Run([]int{23})
	if err != nil {
		t.Fatal(err)
	}
	want, err := first.Run([]int{23})
	if err != nil {
		t.Fatal(err)
	}
	// The fixture checks sparse history. This check isolates recurrent restoration:
	// the first three layers must resume byte-for-byte from the FP32 state.
	for layer := 0; layer < 3; layer++ {
		if !reflect.DeepEqual(got.Tokens[0].Layers[layer], want.Tokens[0].Layers[layer]) {
			t.Fatalf("restored layer %d diverged", layer)
		}
	}
}

func TestExactSparseTop2048AndStableTies(t *testing.T) {
	history := make([][HiddenSize]float32, SparseSelectionCapacity+3)
	for i := range history {
		history[i][0] = float32(i%29) / 29
	}
	ids := selectSparseTokens(history)
	if len(ids) != SparseSelectionCapacity {
		t.Fatalf("selected %d tokens, want %d", len(ids), SparseSelectionCapacity)
	}
	seen := make(map[int]bool, len(ids))
	for _, id := range ids {
		if id < 0 || id >= len(history) || seen[id] {
			t.Fatalf("invalid selected token ID %d", id)
		}
		seen[id] = true
	}
	// Equal learned scores use the lower token ID, making fixture generation stable.
	var tied [][HiddenSize]float32
	tied = append(tied, [HiddenSize]float32{}, [HiddenSize]float32{})
	if got := selectSparseTokens(tied); !reflect.DeepEqual(got, []int{0, 1}) {
		t.Fatalf("stable tie selection = %v", got)
	}
}

func TestTensorSchemaRejectsIndexDrift(t *testing.T) {
	schema := requiredTensorSchema()
	if err := ValidateTensorLayout(schema); err != nil {
		t.Fatal(err)
	}
	schema[0].Shape = []int{HiddenSize * HiddenSize}
	if err := ValidateTensorLayout(schema); err == nil {
		t.Fatal("accepted flattened checkpoint tensor")
	}
}

func TestStateRejectsWrongShape(t *testing.T) {
	var state State
	if err := state.UnmarshalBinary(make([]byte, 3*HiddenSize*HiddenSize*4-1)); err == nil {
		t.Fatal("accepted truncated state")
	}
}
