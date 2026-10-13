package deepseekv4moe

import (
	"container/list"
	"errors"
	"reflect"
	"testing"
)

func slruFixture(ids ...int) []ExpertGroup {
	groups := make([]ExpertGroup, len(ids))
	for i, id := range ids {
		groups[i] = ExpertGroup{Expert: id}
	}
	return groups
}

func TestAdaptiveSLRUScanResistance(t *testing.T) {
	stream := CachePolicyStream{Name: "scan", Groups: slruFixture(0, 1, 0, 2, 3, 4, 5, 0)}
	receipt, err := CachePolicyReplayWith(stream, 3, []CachePolicy{ReplayArmAdaptiveSLRU, CachePolicyLRU, CachePolicyOPT})
	if err != nil {
		t.Fatal(err)
	}
	slru, lru, opt := receipt.Rows[0], receipt.Rows[1], receipt.Rows[2]
	if slru.Hits != 2 || lru.Hits != 1 || slru.Hits > opt.Hits || receipt.Measurement != CachePolicyMeasurementSimulated {
		t.Fatalf("unexpected replay: %+v", receipt)
	}
	for _, row := range receipt.Rows {
		if row.Hits+row.Misses != len(stream.Groups) {
			t.Fatalf("access reconciliation: %+v", row)
		}
	}
}

func TestAdaptiveSLRUWarmupTransition(t *testing.T) {
	for _, tc := range []struct{ capacity, requests, want int }{
		{10, 40, 1}, {10, 41, 8}, {3, 12, 1}, {3, 13, 2}, {1, 4, 1}, {1, 5, 1},
	} {
		if got := adaptiveSLRUProtectedLimit(tc.capacity, tc.requests); got != tc.want {
			t.Fatalf("capacity=%d requests=%d: got %d want %d", tc.capacity, tc.requests, got, tc.want)
		}
	}
	state := adaptiveSLRUState{capacity: 3, resident: make(map[ExpertGroup]*list.Element)}
	for i := 0; i < 12; i++ {
		state.access(ExpertGroup{Expert: i % 2})
	}
	if state.protected.Len() != 1 {
		t.Fatalf("warmup protected=%d", state.protected.Len())
	}
	demoted := state.probation.Back().Value.(adaptiveSLRUEntry).group
	state.access(demoted)
	if state.requests != 13 || state.protected.Len() != 2 {
		t.Fatalf("transition requests=%d protected=%d", state.requests, state.protected.Len())
	}
}

func TestAdaptiveSLRUBoundsAndDeterminism(t *testing.T) {
	groups := slruFixture(0, 1, 0, 2, 0, 3, 1, 2, 0)
	for capacity := 1; capacity <= 3; capacity++ {
		state := adaptiveSLRUState{capacity: capacity, resident: make(map[ExpertGroup]*list.Element)}
		for _, group := range groups {
			state.access(group)
			if len(state.resident) > capacity || len(state.resident) != state.probation.Len()+state.protected.Len() || state.protected.Len() > adaptiveSLRUProtectedLimit(capacity, state.requests) {
				t.Fatalf("capacity=%d resident=%d probation=%d protected=%d", capacity, len(state.resident), state.probation.Len(), state.protected.Len())
			}
		}
		stream := CachePolicyStream{Groups: groups}
		one, err := CachePolicyReplayWith(stream, capacity, []CachePolicy{ReplayArmAdaptiveSLRU})
		if err != nil {
			t.Fatal(err)
		}
		two, err := CachePolicyReplayWith(stream, capacity, []CachePolicy{ReplayArmAdaptiveSLRU})
		if err != nil || !reflect.DeepEqual(one, two) {
			t.Fatalf("nondeterministic: %v", err)
		}
		if capacity == 1 && one.Rows[0].Hits != 0 {
			t.Fatalf("capacity1 hits=%d", one.Rows[0].Hits)
		}
	}
	for _, capacity := range []int{0, -1} {
		_, err := CachePolicyReplayWith(CachePolicyStream{}, capacity, []CachePolicy{ReplayArmAdaptiveSLRU})
		if !errors.Is(err, ErrInvalidTraceCapacity) {
			t.Fatalf("capacity=%d error=%v", capacity, err)
		}
	}
}

func TestAdaptiveSLRULargeCapacityTinyTrace(t *testing.T) {
	capacity := int(^uint(0) >> 1)
	if adaptiveSLRUProtectedLimit(capacity, capacity) != capacity/10 {
		t.Fatal("large capacity warmup overflow")
	}
	state := adaptiveSLRUState{capacity: capacity, resident: make(map[ExpertGroup]*list.Element)}
	for _, group := range slruFixture(0, 1, 0) {
		state.access(group)
	}
	if len(state.resident) != 2 || state.protected.Len() != 1 {
		t.Fatalf("tiny resident=%d protected=%d", len(state.resident), state.protected.Len())
	}
	receipt, err := CachePolicyReplayWith(CachePolicyStream{Groups: slruFixture(0, 1, 0)}, capacity, []CachePolicy{ReplayArmAdaptiveSLRU})
	if err != nil {
		t.Fatal(err)
	}
	if row := receipt.Rows[0]; row.Hits != 1 || row.Misses != 2 || row.Evictions != 0 {
		t.Fatalf("tiny trace: %+v", row)
	}
}

func TestAdaptiveSLRUExplicitSelectionPreservesInventories(t *testing.T) {
	want := []CachePolicy{CachePolicyLRU, CachePolicyLFU, CachePolicyHeat, CachePolicyRandom, CachePolicyOPT}
	if !reflect.DeepEqual(CachePolicyArms(), want) || !reflect.DeepEqual(CachePolicyScoredArms(), []CachePolicy{CachePolicyLFRU, CachePolicyScoreEMA}) {
		t.Fatal("legacy inventories changed")
	}
	if arm, ok := ParseReplayArm("slru"); !ok || arm != ReplayArmAdaptiveSLRU || arm.String() != "slru" {
		t.Fatalf("selector: %v %v", arm, ok)
	}
	stream := CachePolicyStream{Groups: slruFixture(0, 1, 0)}
	legacy, err := CachePolicyReplay(stream, 2)
	if err != nil {
		t.Fatal(err)
	}
	explicit, err := CachePolicyReplayWith(stream, 2, want)
	if err != nil || !reflect.DeepEqual(legacy, explicit) {
		t.Fatalf("legacy mismatch: %v", err)
	}
}
