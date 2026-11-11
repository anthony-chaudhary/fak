package compute

import (
	"errors"
	"reflect"
	"testing"
)

// weightBackend is a test helper that addresses a BackendID for the
// WeightLocalityNode.WeightBackend pointer field.
func weightBackend(b BackendID) *BackendID { return &b }

// TestAssignBackendByWeightLocality pins the borrowed pass-1 contract: a
// weight pins its op to the backend that owns the weight buffer, a weightless
// op inherits from its first assigned input, and an op with no assigned input
// stays unassigned.
func TestAssignBackendByWeightLocality(t *testing.T) {
	t.Run("chain", func(t *testing.T) {
		// 5-node chain: weights on two backends, one weightless op
		// inheriting from its input.
		nodes := []WeightLocalityNode{
			{ID: "n0", WeightBackend: weightBackend(0)},
			{ID: "n1", Inputs: []string{"n0"}, WeightBackend: weightBackend(1)},
			{ID: "n2", Inputs: []string{"n1"}},
			{ID: "n3", Inputs: []string{"n2"}, WeightBackend: weightBackend(0)},
			{ID: "n4", Inputs: []string{"n3"}},
		}
		got, err := AssignByWeightLocality(nodes)
		if err != nil {
			t.Fatalf("AssignByWeightLocality: %v", err)
		}
		want := map[string]BackendID{
			"n0": 0,
			"n1": 1,
			"n2": 1,
			"n3": 0,
			"n4": 0,
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("AssignByWeightLocality = %v, want %v", got, want)
		}
	})

	t.Run("unresolvedStaysUnassigned", func(t *testing.T) {
		// A weightless source op and a weightless op whose only input is
		// itself unresolved both stay unassigned.
		nodes := []WeightLocalityNode{
			{ID: "src"},
			{ID: "mid", Inputs: []string{"src"}},
			{ID: "sink", Inputs: []string{"mid"}},
		}
		got, err := AssignByWeightLocality(nodes)
		if err != nil {
			t.Fatalf("AssignByWeightLocality: %v", err)
		}
		want := map[string]BackendID{
			"src":  BackendUnassigned,
			"mid":  BackendUnassigned,
			"sink": BackendUnassigned,
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("AssignByWeightLocality = %v, want %v", got, want)
		}
	})

	t.Run("inheritanceSkipsUnassignedInputs", func(t *testing.T) {
		// A weightless op with an unassigned first input inherits from the
		// first assigned input instead.
		nodes := []WeightLocalityNode{
			{ID: "src"},
			{ID: "w", WeightBackend: weightBackend(2)},
			{ID: "op", Inputs: []string{"src", "w"}},
		}
		got, err := AssignByWeightLocality(nodes)
		if err != nil {
			t.Fatalf("AssignByWeightLocality: %v", err)
		}
		if got["op"] != 2 {
			t.Fatalf("op = %v, want 2", got["op"])
		}
	})

	t.Run("emptyList", func(t *testing.T) {
		got, err := AssignByWeightLocality(nil)
		if err != nil {
			t.Fatalf("AssignByWeightLocality(nil): %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("AssignByWeightLocality(nil) = %v, want empty", got)
		}
	})

	t.Run("sentinels", func(t *testing.T) {
		cases := []struct {
			name  string
			nodes []WeightLocalityNode
			want  error
		}{
			{
				name:  "emptyID",
				nodes: []WeightLocalityNode{{ID: ""}},
				want:  ErrWeightLocalityEmptyID,
			},
			{
				name: "duplicateID",
				nodes: []WeightLocalityNode{
					{ID: "a"},
					{ID: "a"},
				},
				want: ErrWeightLocalityDuplicateID,
			},
			{
				name: "unknownInput",
				nodes: []WeightLocalityNode{
					{ID: "a", Inputs: []string{"ghost"}},
				},
				want: ErrWeightLocalityUnknownInput,
			},
			{
				name: "forwardInput",
				nodes: []WeightLocalityNode{
					{ID: "a", Inputs: []string{"b"}},
					{ID: "b"},
				},
				want: ErrWeightLocalityForwardInput,
			},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				_, err := AssignByWeightLocality(tc.nodes)
				if !errors.Is(err, tc.want) {
					t.Fatalf("AssignByWeightLocality error = %v, want errors.Is %v", err, tc.want)
				}
			})
		}
	})
}
