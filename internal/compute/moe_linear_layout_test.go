package compute

// CPU-oracle witnesses for the linearized MoE expert weight layout
// (moe_linear_layout.go, ADAPT of ggml-org/llama.cpp PR #31047 at pin
// 69ee2ecc0150ba7d1204a7f7393dfd651569db8c, MIT). The fixtures are fixed and
// hand-checkable: six experts of four three-byte rows each. The witnesses pin
// (1) offset-table addressing returns byte-identical tiles to the strided
// reference, and (2) the launch planner issues no gather step for a fully
// resident layout — and counts gathers when residency is partial. Software
// evidence only: no hardware tuning, no speed claim.

import (
	"bytes"
	"errors"
	"testing"
)

const (
	moeLinearTestExperts = 6
	moeLinearTestRows    = 4
	moeLinearTestRowB    = 3
)

// moeLinearTestPool fills the strided pool with a deterministic per-byte
// pattern so any wrong offset or stride shows up as a byte mismatch.
func moeLinearTestPool() []byte {
	pool := make([]byte, moeLinearTestExperts*moeLinearTestRows*moeLinearTestRowB)
	for i := range pool {
		pool[i] = byte(i % 251)
	}
	return pool
}

// moeLinearTestLayout builds the fixture layout: experts 4, 1, 5 resident in
// that launch order out of six strided experts.
func moeLinearTestLayout(t *testing.T, pool []byte) *MoeLinearLayout {
	t.Helper()
	layout, err := NewMoeLinearLayout(pool, moeLinearTestExperts, moeLinearTestRows, moeLinearTestRowB, []int{4, 1, 5})
	if err != nil {
		t.Fatalf("NewMoeLinearLayout: %v", err)
	}
	return layout
}

func TestMoeLinearLayoutTilesMatchStridedReference(t *testing.T) {
	pool := moeLinearTestPool()
	preserved := append([]byte(nil), pool...)
	layout := moeLinearTestLayout(t, pool)

	const block = moeLinearTestRows * moeLinearTestRowB // 12 bytes per expert
	if got, want := len(layout.Pool), 3*block; got != want {
		t.Fatalf("compacted pool = %d bytes, want %d", got, want)
	}
	wantOffsets := []uint64{0, uint64(block), 2 * uint64(block), 3 * uint64(block)}
	if len(layout.Offsets) != len(wantOffsets) {
		t.Fatalf("offset table length = %d, want %d", len(layout.Offsets), len(wantOffsets))
	}
	for i, want := range wantOffsets {
		if layout.Offsets[i] != want {
			t.Fatalf("Offsets[%d] = %d, want %d", i, layout.Offsets[i], want)
		}
	}
	if !bytes.Equal(pool, preserved) {
		t.Fatal("constructor mutated the input strided pool")
	}

	ranges := [][2]int{{0, moeLinearTestRows}, {1, 3}, {0, 1}, {3, 4}}
	for _, expert := range []int{4, 1, 5} {
		for _, r := range ranges {
			got, err := layout.Tile(expert, r[0], r[1])
			if err != nil {
				t.Fatalf("Tile(%d, %d, %d): %v", expert, r[0], r[1], err)
			}
			want, err := StridedExpertTile(pool, expert, r[0], r[1], moeLinearTestRows, moeLinearTestRowB)
			if err != nil {
				t.Fatalf("StridedExpertTile(%d, %d, %d): %v", expert, r[0], r[1], err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("Tile(%d, [%d,%d)) = %x, strided reference %x", expert, r[0], r[1], got, want)
			}
		}
	}
}

func TestMoeLinearLayoutLaunchPlanIssuesNoGatherSteps(t *testing.T) {
	pool := moeLinearTestPool()
	layout := moeLinearTestLayout(t, pool)

	requests := []MoeLinearRequest{
		{ExpertID: 4, RowStart: 0, RowEnd: 4}, // full block of first active expert
		{ExpertID: 1, RowStart: 1, RowEnd: 3}, // interior tile of second
		{ExpertID: 5, RowStart: 0, RowEnd: 1}, // single row of third
		{ExpertID: 4, RowStart: 2, RowEnd: 4}, // repeat expert, tail tile
	}
	plan, err := layout.PlanLaunches(requests)
	if err != nil {
		t.Fatalf("PlanLaunches: %v", err)
	}
	if plan.GatherSteps != 0 {
		t.Fatalf("GatherSteps = %d, want 0 for a fully resident layout", plan.GatherSteps)
	}
	if len(plan.Launches) != len(requests) {
		t.Fatalf("launches = %d, want %d (one contiguous span per request)", len(plan.Launches), len(requests))
	}
	const block = moeLinearTestRows * moeLinearTestRowB
	want := []struct {
		expert int
		offset uint64
		length uint64
	}{
		{4, 0, uint64(block)},
		{1, uint64(block) + moeLinearTestRowB, 2 * uint64(moeLinearTestRowB)},
		{5, 2 * uint64(block), uint64(moeLinearTestRowB)},
		{4, 2 * uint64(moeLinearTestRowB), 2 * uint64(moeLinearTestRowB)},
	}
	for i, l := range plan.Launches {
		if l.ExpertID != want[i].expert || l.OffsetBytes != want[i].offset || l.LengthBytes != want[i].length {
			t.Errorf("launch[%d] = {%d @%d len %d}, want {%d @%d len %d}",
				i, l.ExpertID, l.OffsetBytes, l.LengthBytes, want[i].expert, want[i].offset, want[i].length)
		}
		if l.OffsetBytes+l.LengthBytes > uint64(len(layout.Pool)) {
			t.Errorf("launch[%d] span [%d,%d) leaves the compacted pool of %d bytes",
				i, l.OffsetBytes, l.OffsetBytes+l.LengthBytes, len(layout.Pool))
		}
		// The planned span must hold exactly the tile the strided reference names.
		req := requests[i]
		ref, err := StridedExpertTile(pool, req.ExpertID, req.RowStart, req.RowEnd, moeLinearTestRows, moeLinearTestRowB)
		if err != nil {
			t.Fatalf("StridedExpertTile(launch %d): %v", i, err)
		}
		if !bytes.Equal(layout.Pool[l.OffsetBytes:l.OffsetBytes+l.LengthBytes], ref) {
			t.Errorf("launch[%d] span bytes differ from strided reference", i)
		}
	}
}

// moeLinearStubResidency serves a fixed set of full blocks and nothing else,
// standing in for a paged pool where most experts are not on device.
type moeLinearStubResidency struct {
	spans map[int][2]uint64
}

func (s moeLinearStubResidency) ResidentSpan(expertID int) (uint64, uint64, bool) {
	sp, ok := s.spans[expertID]
	return sp[0], sp[1], ok
}

func TestMoeLinearLayoutPartialResidencyCountsGatherSteps(t *testing.T) {
	const block = moeLinearTestRows * moeLinearTestRowB
	res := moeLinearStubResidency{spans: map[int][2]uint64{
		4: {0, block},
	}}
	requests := []MoeLinearRequest{
		{ExpertID: 4, RowStart: 0, RowEnd: 4},
		{ExpertID: 1, RowStart: 1, RowEnd: 3},
		{ExpertID: 5, RowStart: 0, RowEnd: 1},
		{ExpertID: 4, RowStart: 2, RowEnd: 4},
	}
	plan, err := PlanMoeLinearLaunches(res, moeLinearTestRows, moeLinearTestRowB, requests)
	if err != nil {
		t.Fatalf("PlanMoeLinearLaunches: %v", err)
	}
	if plan.GatherSteps != 2 {
		t.Fatalf("GatherSteps = %d, want 2 (experts 1 and 5 not resident)", plan.GatherSteps)
	}
	if len(plan.Launches) != 2 {
		t.Fatalf("launches = %d, want 2 (only the resident expert's tiles)", len(plan.Launches))
	}
	for i, l := range plan.Launches {
		if l.ExpertID != 4 {
			t.Errorf("launch[%d] expert = %d, want 4", i, l.ExpertID)
		}
	}
}

func TestMoeLinearLayoutRefusals(t *testing.T) {
	pool := moeLinearTestPool()

	t.Run("geometry", func(t *testing.T) {
		if _, err := NewMoeLinearLayout(pool[:71], moeLinearTestExperts, moeLinearTestRows, moeLinearTestRowB, nil); !errors.Is(err, ErrMoeLinearGeometry) {
			t.Errorf("short pool: err = %v, want ErrMoeLinearGeometry", err)
		}
		if _, err := NewMoeLinearLayout(pool, moeLinearTestExperts, 0, moeLinearTestRowB, nil); !errors.Is(err, ErrMoeLinearGeometry) {
			t.Errorf("zero rows: err = %v, want ErrMoeLinearGeometry", err)
		}
	})

	t.Run("selection", func(t *testing.T) {
		if _, err := NewMoeLinearLayout(pool, moeLinearTestExperts, moeLinearTestRows, moeLinearTestRowB, []int{4, 9}); !errors.Is(err, ErrMoeLinearUnknownExpert) {
			t.Errorf("out-of-range expert: err = %v, want ErrMoeLinearUnknownExpert", err)
		}
		if _, err := NewMoeLinearLayout(pool, moeLinearTestExperts, moeLinearTestRows, moeLinearTestRowB, []int{1, 1}); !errors.Is(err, ErrMoeLinearDuplicateExpert) {
			t.Errorf("repeated expert: err = %v, want ErrMoeLinearDuplicateExpert", err)
		}
	})

	layout := moeLinearTestLayout(t, pool)

	t.Run("tile addressing", func(t *testing.T) {
		if _, err := layout.Tile(0, 0, 4); !errors.Is(err, ErrMoeLinearNotResident) {
			t.Errorf("valid-but-unselected expert: err = %v, want ErrMoeLinearNotResident", err)
		}
		if _, err := layout.Tile(9, 0, 1); !errors.Is(err, ErrMoeLinearUnknownExpert) {
			t.Errorf("unknown expert: err = %v, want ErrMoeLinearUnknownExpert", err)
		}
		for _, tc := range [][2]int{{-1, 2}, {2, 2}, {0, 5}} {
			if _, err := layout.Tile(4, tc[0], tc[1]); !errors.Is(err, ErrMoeLinearTileRange) {
				t.Errorf("rows [%d,%d): err = %v, want ErrMoeLinearTileRange", tc[0], tc[1], err)
			}
		}
	})

	t.Run("planner shapes", func(t *testing.T) {
		if _, err := layout.PlanLaunches([]MoeLinearRequest{{ExpertID: 4, RowStart: 0, RowEnd: 5}}); !errors.Is(err, ErrMoeLinearTileRange) {
			t.Errorf("overlong tile: err = %v, want ErrMoeLinearTileRange", err)
		}
		if _, err := layout.PlanLaunches([]MoeLinearRequest{{ExpertID: -1, RowStart: 0, RowEnd: 1}}); !errors.Is(err, ErrMoeLinearUnknownExpert) {
			t.Errorf("negative expert: err = %v, want ErrMoeLinearUnknownExpert", err)
		}
	})

	t.Run("strided oracle shapes", func(t *testing.T) {
		if _, err := StridedExpertTile(pool, 9, 0, 1, moeLinearTestRows, moeLinearTestRowB); !errors.Is(err, ErrMoeLinearUnknownExpert) {
			t.Errorf("unknown expert: err = %v, want ErrMoeLinearUnknownExpert", err)
		}
		if _, err := StridedExpertTile(pool, 0, 0, 5, moeLinearTestRows, moeLinearTestRowB); !errors.Is(err, ErrMoeLinearTileRange) {
			t.Errorf("overlong tile: err = %v, want ErrMoeLinearTileRange", err)
		}
	})
}
