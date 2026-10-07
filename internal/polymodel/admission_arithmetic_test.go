package polymodel

import (
	"errors"
	"reflect"
	"testing"
)

// Sizes are metadata only: Pool owns no weights and these cases allocate only
// a handful of descriptors, including when the byte budget is MaxInt64.
// fak-test:runtime fast est=10ms lane=default
func TestPoolAdmissionArithmeticBoundaries(t *testing.T) {
	const maxBytes = int64(^uint64(0) >> 1)
	for _, tc := range []struct {
		name    string
		budget  int64
		seed    []Model
		request Model
		want    error
		victims []ModelID
		used    int64
	}{
		{
			name: "max-budget-one-byte-over", budget: maxBytes,
			seed:    []Model{{ID: "cold", WeightBytes: maxBytes}},
			request: Model{ID: "next", WeightBytes: 1},
			victims: []ModelID{"cold"}, used: 1,
		},
		{
			name: "below-max-budget-overflow", budget: maxBytes - 1,
			seed:    []Model{{ID: "cold", WeightBytes: maxBytes - 1}},
			request: Model{ID: "next", WeightBytes: 2},
			victims: []ModelID{"cold"}, used: 2,
		},
		{
			name: "max-sized-request", budget: maxBytes,
			seed:    []Model{{ID: "cold", WeightBytes: 1}},
			request: Model{ID: "next", WeightBytes: maxBytes},
			victims: []ModelID{"cold"}, used: maxBytes,
		},
		{
			name: "max-budget-zero-and-multiple-victims", budget: maxBytes,
			seed: []Model{
				{ID: "zero"}, {ID: "cold", WeightBytes: 1},
				{ID: "pin", WeightBytes: maxBytes - 3, Pinned: true},
				{ID: "hot", WeightBytes: 2},
			},
			request: Model{ID: "next", WeightBytes: 3},
			victims: []ModelID{"zero", "cold", "hot"}, used: maxBytes,
		},
		{
			name: "small-budget-same-victim-order", budget: 8,
			seed: []Model{
				{ID: "zero"}, {ID: "cold", WeightBytes: 1},
				{ID: "pin", WeightBytes: 5, Pinned: true},
				{ID: "hot", WeightBytes: 2},
			},
			request: Model{ID: "next", WeightBytes: 3},
			victims: []ModelID{"zero", "cold", "hot"}, used: 8,
		},
		{
			name: "exact-headroom-no-eviction", budget: maxBytes,
			seed:    []Model{{ID: "pin", WeightBytes: maxBytes - 3, Pinned: true}},
			request: Model{ID: "next", WeightBytes: 3}, used: maxBytes,
		},
		{
			name: "pinned-short-by-one", budget: maxBytes,
			seed: []Model{
				{ID: "pin", WeightBytes: maxBytes - 1, Pinned: true},
				{ID: "cold", WeightBytes: 1},
			},
			request: Model{ID: "next", WeightBytes: 2}, want: ErrPinnedNoRoom,
		},
		{
			name: "existing-id-keeps-descriptor", budget: maxBytes - 1,
			seed:    []Model{{ID: "pin", WeightBytes: maxBytes - 1, Family: "old", Pinned: true}},
			request: Model{ID: "pin", WeightBytes: maxBytes, Family: "replacement"}, used: maxBytes - 1,
		},
		{
			name: "negative-size-normalizes-at-full-budget", budget: maxBytes,
			seed:    []Model{{ID: "pin", WeightBytes: maxBytes, Pinned: true}},
			request: Model{ID: "next", WeightBytes: -maxBytes - 1}, used: maxBytes,
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
				t.Fatal("CheckAdmission changed pool state")
			}
			victims, err := p.Admit(tc.request)
			if !errors.Is(err, tc.want) || !reflect.DeepEqual(victims, tc.victims) {
				t.Fatalf("Admit = (%v, %v), want (%v, %v)", victims, err, tc.victims, tc.want)
			}
			if err != nil {
				if !reflect.DeepEqual(before, snapshotAdmissionPool(p)) {
					t.Fatal("refused Admit changed pool state")
				}
				return
			}
			if p.Budget() != before.budget || p.Used() != tc.used || p.clock != before.clock+1 {
				t.Fatalf("budget/used/clock = %d/%d/%d, want %d/%d/%d",
					p.Budget(), p.Used(), p.clock, before.budget, tc.used, before.clock+1)
			}
			want := tc.request
			wantLen := len(before.models) - len(tc.victims)
			if old, exists := before.models[want.ID]; exists {
				want = old.m
			} else {
				wantLen++
				if want.WeightBytes < 0 {
					want.WeightBytes = 0
				}
			}
			if got, ok := p.Get(want.ID); !ok || got != want || p.Len() != wantLen {
				t.Fatalf("admitted descriptor/length = %+v/%v/%d, want %+v/true/%d", got, ok, p.Len(), want, wantLen)
			}
			if p.models[want.ID].used != p.clock {
				t.Fatal("successful Admit did not touch the admitted ID")
			}
			for _, id := range tc.victims {
				if p.Has(id) {
					t.Fatalf("victim %q is still resident", id)
				}
				delete(before.models, id)
			}
			for id, old := range before.models {
				if id != want.ID {
					if got, ok := p.models[id]; !ok || *got != old {
						t.Fatalf("surviving descriptor or recency changed for %q", id)
					}
				}
			}
			var sum int64
			for _, e := range p.models {
				if e.m.WeightBytes < 0 || e.m.WeightBytes > p.Budget()-sum {
					t.Fatal("resident byte sum exceeds budget")
				}
				sum += e.m.WeightBytes
			}
			if sum != p.Used() {
				t.Fatalf("resident byte sum %d differs from Used %d", sum, p.Used())
			}
		})
	}
}
