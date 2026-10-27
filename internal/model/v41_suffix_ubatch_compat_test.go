package model

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41SuffixUbatchRoutedIndependentRows(t *testing.T) {
	for _, mode := range []string{"success", "later_row_error"} {
		t.Run(mode, func(t *testing.T) {
			m := &Model{Cfg: Config{HiddenSize: 4, MoEIntermediateSize: 3, HeadDim: v41KVLoraRank}}
			// Continuations own separate rows, not a shared packed panel.
			inputs := [][]float32{{1, 2, 3, 4}, {-2, -4, 6, 8}}
			original := [][]float32{{1, 2, 3, 4}, {-2, -4, 6, 8}}
			works := []*v41LayerStepWork{
				{layer: 0, input: inputs[0], full: true, picks: []routePick{{expert: 1, weight: .5}, {expert: 0, weight: .25}}},
				{layer: 0, input: inputs[1], full: true, picks: []routePick{{expert: 0, weight: .125}, {expert: 1, weight: .75}}},
			}
			fault := errors.New("later suffix row failed")
			calls := 0
			st := &v41ForwardState{routedExpert: func(_ int, _ string, input []float32, weight float32) ([]float32, v41RoutedExpertOutcome, error) {
				calls++
				if mode == "later_row_error" && input[0] < 0 {
					return nil, v41RoutedExpertError, fault
				}
				out := make([]float32, len(input))
				for i, value := range input {
					out[i] = value * weight
				}
				return out, v41RoutedExpertHandled, nil
			}}
			got, err := m.v41SuffixUbatchRouted(0, works, st, &v41ProjScratch{})
			if !reflect.DeepEqual(inputs, original) {
				t.Fatal("suffix contraction mutated continuation inputs")
			}
			if mode == "later_row_error" {
				if !errors.Is(err, fault) || got != nil || calls != 2 {
					t.Fatalf("selected error: result=%v err=%v calls=%d", got, err, calls)
				}
				return
			}
			want := [][]float32{{.75, 1.5, 2.25, 3}, {-1.75, -3.5, 5.25, 7}}
			if err != nil || calls != 4 || !reflect.DeepEqual(got, want) {
				t.Fatalf("independent rows: result=%v want=%v err=%v calls=%d", got, want, err, calls)
			}
			got[0][0] = 99
			if got[1][0] != want[1][0] || !reflect.DeepEqual(inputs, original) {
				t.Fatal("routed output aliases another row or continuation input")
			}
		})
	}
}
