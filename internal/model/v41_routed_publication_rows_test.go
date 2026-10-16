package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41RoutedLayerPublicationRows(t *testing.T) {
	legacy := func(picks int, useID, useFused bool) []compute.Tensor {
		return make([]compute.Tensor, picks)
	}
	sentinel := compute.NewF32(compute.Default(), []int{1}, []float32{1})
	for _, mode := range []struct {
		name            string
		useID, useFused bool
	}{
		{name: "per_pick"},
		{name: "id", useID: true},
		{name: "fused", useFused: true},
		{name: "fused_with_id", useID: true, useFused: true},
	} {
		t.Run(mode.name, func(t *testing.T) {
			skipped := mode.useID || mode.useFused
			for _, picks := range []int{0, 1, 6, compute.V41RoutedIDMaxPicks, compute.V41RoutedIDMaxPicks + 1} {
				got := v41RoutedLayerPublicationRows(picks, mode.useID, mode.useFused)
				if skipped {
					if got != nil {
						t.Fatalf("picks=%d: unused publication table allocated", picks)
					}
					continue
				}
				want := legacy(picks, mode.useID, mode.useFused)
				if got == nil || len(got) != picks || cap(got) != picks || !reflect.DeepEqual(got, want) {
					t.Fatalf("picks=%d: per-pick table lost owned, zeroed, full-capacity make semantics", picks)
				}
				next := v41RoutedLayerPublicationRows(picks, mode.useID, mode.useFused)
				appended := append(got, sentinel)
				if len(appended) != picks+1 || appended[picks].Buf() != sentinel.Buf() {
					t.Fatalf("picks=%d: appended publication row changed", picks)
				}
				if picks > 0 {
					appended[0] = sentinel
					if !reflect.DeepEqual(got, want) {
						t.Fatal("append reused the full-capacity table's storage")
					}
					got[0] = sentinel
					if !reflect.DeepEqual(next, want) {
						t.Fatal("retained publication table aliases the next call")
					}
				}
			}

			const allocationPicks = 6
			var retained []compute.Tensor
			measure := func(allocate func(int, bool, bool) []compute.Tensor) float64 {
				return testing.AllocsPerRun(100, func() {
					retained = allocate(allocationPicks, mode.useID, mode.useFused)
				})
			}
			old, current := measure(legacy), measure(v41RoutedLayerPublicationRows)
			wantAllocs := float64(1)
			if skipped {
				wantAllocs = 0
				if retained != nil {
					t.Fatal("allocation witness retained an unused table")
				}
			} else if len(retained) != allocationPicks || cap(retained) != allocationPicks || !reflect.DeepEqual(retained, legacy(allocationPicks, false, false)) {
				t.Fatal("allocation witness did not retain six zeroed publication rows")
			}
			t.Logf("top-6 publication table allocs/op: unconditional control=%g conditional=%g", old, current)
			if old != 1 || current != wantAllocs {
				t.Fatalf("publication-table allocation mismatch: control=%g candidate=%g, want 1 and %g", old, current, wantAllocs)
			}
		})
	}
}
