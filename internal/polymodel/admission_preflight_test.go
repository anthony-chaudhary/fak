package polymodel

import (
	"errors"
	"reflect"
	"testing"
)

type admissionPoolState struct {
	budget int64
	used   int64
	clock  uint64
	models map[ModelID]entry
}

func snapshotAdmissionPool(p *Pool) admissionPoolState {
	s := admissionPoolState{budget: p.budget, used: p.used, clock: p.clock, models: map[ModelID]entry{}}
	for id, e := range p.models {
		s.models[id] = *e
	}
	return s
}

// CheckAdmission is a snapshot refusal query, not a reservation or a Touch.
// The MaxInt64 row is refusal-only; feasible near-ceiling eviction arithmetic
// remains outside this extraction's contract.
// fak-test:runtime fast est=10ms lane=default
func TestPoolCheckAdmissionSnapshot(t *testing.T) {
	const maxBytes = int64(^uint64(0) >> 1)
	for _, tc := range []struct {
		name    string
		budget  int64
		seed    []Model
		request Model
		want    error
		victims []ModelID
	}{
		{
			name: "pinned-full", budget: 8,
			seed:    []Model{{ID: "pin", WeightBytes: 8, Pinned: true}},
			request: Model{ID: "next", WeightBytes: 4}, want: ErrPinnedNoRoom,
		},
		{
			name: "mixed-insufficient", budget: 8,
			seed:    []Model{{ID: "pin", WeightBytes: 4, Pinned: true}, {ID: "cold", WeightBytes: 4}},
			request: Model{ID: "next", WeightBytes: 8}, want: ErrPinnedNoRoom,
		},
		{
			name: "exact-available-fit", budget: 8,
			seed:    []Model{{ID: "pin", WeightBytes: 4, Pinned: true}, {ID: "cold", WeightBytes: 4}},
			request: Model{ID: "next", WeightBytes: 4}, victims: []ModelID{"cold"},
		},
		{
			name: "existing-id-precedes-replacement-descriptor", budget: 8,
			seed:    []Model{{ID: "pin", WeightBytes: 4, Pinned: true}, {ID: "cold", WeightBytes: 4}},
			request: Model{ID: "pin", WeightBytes: maxBytes, Family: "replacement"},
		},
		{
			name: "empty-id-precedes-oversize", budget: 8,
			seed:    []Model{{ID: "pin", WeightBytes: 8, Pinned: true}},
			request: Model{WeightBytes: maxBytes}, want: ErrEmptyID,
		},
		{name: "oversize", budget: 8, request: Model{ID: "next", WeightBytes: 9}, want: ErrTooLarge},
		{name: "zero-size", request: Model{ID: "zero"}},
		{name: "negative-size-normalizes", request: Model{ID: "negative", WeightBytes: -1}},
		{
			name: "max-int-pinned-refusal", budget: maxBytes,
			seed:    []Model{{ID: "pin", WeightBytes: maxBytes, Pinned: true}},
			request: Model{ID: "next", WeightBytes: 1}, want: ErrPinnedNoRoom,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := NewPool(tc.budget)
			for _, m := range tc.seed {
				mustAdmit(t, p, m)
			}
			before := snapshotAdmissionPool(p)
			if err := p.CheckAdmission(tc.request); !errors.Is(err, tc.want) {
				t.Fatalf("CheckAdmission = %v, want %v", err, tc.want)
			}
			if !reflect.DeepEqual(before, snapshotAdmissionPool(p)) {
				t.Fatal("preflight changed descriptors, residency, budget, clock or LRU stamps")
			}
			evicted, err := p.Admit(tc.request)
			if !errors.Is(err, tc.want) {
				t.Fatalf("Admit on the same snapshot = %v, want %v", err, tc.want)
			}
			if !reflect.DeepEqual(evicted, tc.victims) {
				t.Fatalf("evicted %v, want %v", evicted, tc.victims)
			}
			if err != nil {
				if !reflect.DeepEqual(before, snapshotAdmissionPool(p)) {
					t.Fatal("refused Admit changed the pool")
				}
				return
			}
			want := tc.request
			if old, exists := before.models[want.ID]; exists {
				want = old.m
				if p.clock <= before.clock || p.models[want.ID].used <= before.models[want.ID].used {
					t.Fatal("successful existing-ID Admit no longer touches recency")
				}
			} else if want.WeightBytes < 0 {
				want.WeightBytes = 0
			}
			if got, ok := p.Get(want.ID); !ok || got != want {
				t.Fatalf("admitted descriptor = %+v/%v, want %+v", got, ok, want)
			}
			if p.Used() < 0 || p.Used() > p.Budget() {
				t.Fatalf("used %d outside budget %d in bounded control", p.Used(), p.Budget())
			}
		})
	}
}
