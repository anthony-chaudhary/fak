package model

import (
	"errors"
	"slices"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41StepVisibleKeysAllocationAndOwnership(t *testing.T) {
	window512 := make([]int32, 512)
	for i := range window512 {
		window512[i] = int32(513 + i)
	}
	for _, tc := range []struct {
		name              string
		pos, window, ring int
		want              []int32
	}{
		{"first", 0, 3, 3, []int32{0}},
		{"global", 3, -1, 4, []int32{0, 1, 2, 3}},
		{"wrapped-local", 6, 3, 2, []int32{4, 5, 6}},
		{"window-512", 1024, 512, 512, window512},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step := &v41PlainLayerStep{
				m: &Model{Cfg: Config{Window: []int{tc.window}}}, pos: tc.pos,
				layerState: &V41AttentionState{nextWindowPos: tc.pos, windowSize: tc.ring},
			}
			prior, err := step.visibleKeys()
			if err != nil || !slices.Equal(prior, tc.want) {
				t.Fatalf("visible keys = %v, %v; want %v", prior, err, tc.want)
			}
			prior[0] = -99
			var got []int32
			allocs := testing.AllocsPerRun(100, func() {
				got, err = step.visibleKeys()
				if err != nil || !slices.Equal(got, tc.want) || prior[0] != -99 {
					panic("visible keys changed or alias an earlier call")
				}
			})
			t.Logf("visible keys: %g allocations", allocs)
			if allocs > 1 {
				t.Errorf("visible keys allocated %g objects, want one owned key buffer", allocs)
			}
			if tc.pos > 0 {
				step.layerState.windowSize = 1
				if keys, err := step.visibleKeys(); keys != nil || !errors.Is(err, ErrV41ForwardStage) {
					t.Fatalf("missing retained key was not refused: keys=%v error=%v", keys, err)
				}
			}
		})
	}
}
