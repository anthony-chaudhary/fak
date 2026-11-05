package fitplan

import (
	"reflect"
	"strconv"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestContextLadderStopsBeforeIntOverflow(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name                  string
		anchor, modelMax, pin int
		want                  []int
	}{
		{"unknown maximum", 8192, 0, 4096, []int{4096, 8192}},
		{"below first power", 0, 4095, 0, []int{4095}},
		{"first power deduplicated", 4096, 4096, 4096, []int{4096}},
		{"extras above maximum", 7000, 5000, 5000, []int{4096, 5000, 7000}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := contextLadder(tc.anchor, tc.modelMax, tc.pin); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("contextLadder = %v, want %v", got, tc.want)
			}
		})
	}

	maxInt := int(^uint(0) >> 1)
	highestPower := int(uint(1) << (strconv.IntSize - 2))
	for _, maximum := range []int{highestPower - 1, highestPower, maxInt} {
		t.Run(strconv.Itoa(maximum), func(t *testing.T) {
			// Enumerate by a fixed, host-width-bounded bit index, not by
			// repeatedly doubling a signed counter as the implementation does.
			want := []int{3000}
			for bit := 12; bit < strconv.IntSize-1; bit++ {
				value := int(uint(1) << bit)
				if value <= maximum {
					want = append(want, value)
				}
			}
			if want[len(want)-1] != maximum {
				want = append(want, maximum)
			}
			if got := contextLadder(3000, maximum, 4096); !reflect.DeepEqual(got, want) {
				t.Fatalf("contextLadder = %v, want %v", got, want)
			}
		})
	}

	t.Run("bounded pinned search", func(t *testing.T) {
		in := searchInput(16 * gib)
		in.Model.MaxContextTokens = maxInt
		in.Workload.Concurrency = 1
		in.Pins = SearchPins{ContextTokens: 4096, Slots: 1}
		rep := Search(in)
		if rep.Status != StatusFits || rep.Chosen == nil || rep.Chosen.ContextTokens != 4096 || rep.Chosen.Slots != 1 {
			t.Fatalf("Search = %+v; want a fitting pinned 4096-token, one-slot row", rep)
		}
	})
}
