package model

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41SingleIndexRowAllocationAndOwnership(t *testing.T) {
	m := &Model{Cfg: Config{DeepSeekV41: &DeepSeekV41Config{
		CompressRatios: []int{4}, IndexSourceLayerIDs: []int{0},
	}}}
	plan := V41AttentionPlan{Layer: 0, Ratio: 4, TopKWidth: 512}
	local := []int32{2, 0, -1}
	var got []int32
	allocs := testing.AllocsPerRun(100, func() {
		var err error
		got, err = m.v41AttentionIndexList(plan, nil, local, 8, 3, 1)
		if err != nil {
			panic(err)
		}
	})
	if len(got) != plan.TopKWidth {
		t.Fatalf("row width %d, want %d", len(got), plan.TopKWidth)
	}
	for i, id := range got {
		want := int32(-1)
		if i < len(local) {
			want = local[i]
		}
		if id != want {
			t.Fatalf("slot %d = %d, want %d", i, id, want)
		}
	}
	local[0] = 1
	if got[0] != 2 {
		t.Fatal("normalized row aliases caller input")
	}
	got[1] = 2
	if local[1] != 0 {
		t.Fatal("caller input aliases normalized row")
	}
	for _, invalid := range [][]int32{{3}, {-2}, make([]int32, plan.TopKWidth+1)} {
		if out, err := m.v41AttentionIndexList(plan, nil, invalid, 8, 3, 1); out != nil || !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("invalid selection returned %v, %v", out, err)
		}
	}
	t.Logf("single source row allocations=%g width=%d", allocs, plan.TopKWidth)
	if allocs > 1 {
		t.Fatalf("single source row allocated %g objects, want only its owned normalized buffer", allocs)
	}
}
