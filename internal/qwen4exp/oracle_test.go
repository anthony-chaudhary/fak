package qwen4exp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
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

func TestGatedDeltaChunkBoundaryEquivalence(t *testing.T) {
	if HiddenSize != 4 {
		t.Fatalf("test contract requires d=4, got %d", HiddenSize)
	}

	type testCase struct {
		name        string
		steps       int
		initial     [HiddenSize * HiddenSize]float32
		residual    [HiddenSize]float32
		q           [HiddenSize]float32
		k           [HiddenSize]float32
		v           [HiddenSize]float32
		z           [HiddenSize]float32
		a           [HiddenSize]float32
		b           [HiddenSize]float32
		wantOutputs [][HiddenSize]float32
		wantStates  [][HiddenSize * HiddenSize]float32
	}

	entry := func(row, col int, value float32) (out [HiddenSize * HiddenSize]float32) {
		out[row*HiddenSize+col] = value
		return out
	}
	row0 := func(a, b, c, d float32) (out [HiddenSize * HiddenSize]float32) {
		out[0], out[1], out[2], out[3] = a, b, c, d
		return out
	}
	vec := func(a, b, c, d float32) [HiddenSize]float32 {
		return [HiddenSize]float32{a, b, c, d}
	}

	e0 := vec(1, 0, 0, 0)
	e1 := vec(0, 1, 0, 0)
	e2 := vec(0, 0, 1, 0)
	zeroVector := vec(0, 0, 0, 0)
	zeroState := [HiddenSize * HiddenSize]float32{}
	cases := []testCase{
		{
			name:        "zero",
			steps:       3,
			wantOutputs: [][HiddenSize]float32{zeroVector, zeroVector, zeroVector},
			wantStates:  [][HiddenSize * HiddenSize]float32{zeroState, zeroState, zeroState},
		},
		{
			name:        "rank-one write",
			steps:       3,
			q:           e0,
			k:           e0,
			v:           e0,
			wantOutputs: [][HiddenSize]float32{vec(0.25, 0, 0, 0), vec(0.25, 0, 0, 0), vec(0.25, 0, 0, 0)},
			wantStates:  [][HiddenSize * HiddenSize]float32{entry(0, 0, 0.5), entry(0, 0, 0.5), entry(0, 0, 0.5)},
		},
		{
			name:        "off-diagonal orientation",
			steps:       3,
			q:           e1,
			k:           e1,
			v:           e2,
			wantOutputs: [][HiddenSize]float32{vec(0, 0, 0.25, 0), vec(0, 0, 0.25, 0), vec(0, 0, 0.25, 0)},
			wantStates:  [][HiddenSize * HiddenSize]float32{entry(2, 1, 0.5), entry(2, 1, 0.5), entry(2, 1, 0.5)},
		},
		{
			name:        "state-dependent decay",
			steps:       3,
			initial:     entry(0, 0, 1),
			q:           e0,
			wantOutputs: [][HiddenSize]float32{vec(0.25, 0, 0, 0), vec(0.125, 0, 0, 0), vec(0.0625, 0, 0, 0)},
			wantStates:  [][HiddenSize * HiddenSize]float32{entry(0, 0, 0.5), entry(0, 0, 0.25), entry(0, 0, 0.125)},
		},
		{
			name:        "prediction-before-decay",
			steps:       2,
			initial:     entry(0, 0, 1),
			q:           e0,
			k:           e0,
			wantOutputs: [][HiddenSize]float32{zeroVector, zeroVector},
			wantStates:  [][HiddenSize * HiddenSize]float32{zeroState, zeroState},
		},
		{
			name:        "accumulation order",
			steps:       3,
			initial:     row0(1<<24, 1, -(1 << 24), 1),
			q:           vec(1, 1, 1, 1),
			wantOutputs: [][HiddenSize]float32{vec(0.25, 0, 0, 0), vec(0.125, 0, 0, 0), vec(0.0625, 0, 0, 0)},
			wantStates: [][HiddenSize * HiddenSize]float32{
				row0(1<<23, 0.5, -(1 << 23), 0.5),
				row0(1<<22, 0.25, -(1 << 22), 0.25),
				row0(1<<21, 0.125, -(1 << 21), 0.125),
			},
		},
	}

	assertVectorBits := func(t *testing.T, label string, got, want [HiddenSize]float32) {
		t.Helper()
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d] bits = %08x, want %08x (%g != %g)", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]), got[i], want[i])
			}
		}
	}
	assertStateBits := func(t *testing.T, label string, got, want State) {
		t.Helper()
		for layer := range want.Recurrent {
			for i := range want.Recurrent[layer] {
				if math.Float32bits(got.Recurrent[layer][i]) != math.Float32bits(want.Recurrent[layer][i]) {
					t.Fatalf("%s layer=%d index=%d bits = %08x, want %08x (%g != %g)", label, layer, i, math.Float32bits(got.Recurrent[layer][i]), math.Float32bits(want.Recurrent[layer][i]), got.Recurrent[layer][i], want.Recurrent[layer][i])
				}
			}
		}
	}
	runStep := func(tc testCase, state *State) [HiddenSize]float32 {
		return gatedDelta(tc.residual, tc.q, tc.k, tc.v, tc.z, tc.a, tc.b, &state.Recurrent[0])
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fullState := State{}
			fullState.Recurrent[0] = tc.initial
			fullOutputs := make([][HiddenSize]float32, tc.steps)
			for token := 0; token < tc.steps; token++ {
				fullOutputs[token] = runStep(tc, &fullState)
				assertVectorBits(t, fmt.Sprintf("token %d output", token+1), fullOutputs[token], tc.wantOutputs[token])
				wantState := State{}
				wantState.Recurrent[0] = tc.wantStates[token]
				assertStateBits(t, fmt.Sprintf("token %d state", token+1), fullState, wantState)
			}

			for split := 0; split <= tc.steps; split++ {
				splitState := State{}
				splitState.Recurrent[0] = tc.initial
				for token := 0; token < split; token++ {
					runStep(tc, &splitState)
				}

				var restored State
				if err := restored.UnmarshalBinary(splitState.MarshalBinary()); err != nil {
					t.Fatalf("split %d state round-trip: %v", split, err)
				}
				assertStateBits(t, fmt.Sprintf("split %d restored prefix state", split), restored, splitState)
				for token := split; token < tc.steps; token++ {
					got := runStep(tc, &restored)
					assertVectorBits(t, fmt.Sprintf("split %d suffix token %d output", split, token+1), got, fullOutputs[token])
				}
				assertStateBits(t, fmt.Sprintf("split %d final state", split), restored, fullState)
			}
		})
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
