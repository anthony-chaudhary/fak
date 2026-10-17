package compute

import (
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
// The deferred panel readback must not regress into extra per-row work: one
// slice header plus exactly one full-cap row allocation per logical row. This
// ceiling is the allocation half of the #13874 witness; the decode loop itself
// is a bulk copy on little-endian hosts.
func TestV41DeferredReadAllocCeiling(t *testing.T) {
	layout := []int{8, 4, 2, 1}
	width, total := 5, 0
	sizes := []int{32, 16}
	for _, n := range layout {
		sizes = append(sizes, n*width*4)
		total += n
	}
	image := newV41HostReadbackImage(sizes, total, v41NativeLittleEndian())
	if image.floats != nil {
		t.Fatal("packed read retained a typed arena")
	}
	// One slice header for the row table, then one row allocation per logical row.
	const wantAllocs = 1 + 15
	if got := testing.AllocsPerRun(50, func() {
		rows := image.materializePanelRows(sizes, 2, layout, width, total)
		if len(rows) != total {
			t.Fatalf("logical count %d, want %d", len(rows), total)
		}
	}); got != wantAllocs {
		t.Fatalf("materializePanelRows allocs/run = %v, want %d", got, wantAllocs)
	}

	// The single-row typed arm stays allocation-free for the row itself.
	single := newV41HostReadbackImage([]int{16, 64}, 1, v41NativeLittleEndian())
	if got := testing.AllocsPerRun(50, func() {
		if rows := single.materializeRows([]int{16, 64}, 1); len(rows) != 1 {
			t.Fatal("single row count")
		}
	}); got != 1 {
		t.Fatalf("typed single-row materializeRows allocs/run = %v, want 1", got)
	}
}

// fak-test:runtime fast est=5ms lane=default
// Dispatch flags are process constants: after the first read the cached value is
// returned without touching the environment again, so a later mutation is not
// observed until the cache is explicitly reset.
func TestDispatchFlagEnvCachedOnce(t *testing.T) {
	t.Setenv("FAK_V41_Q3K_TOKEN_PANEL", "0")
	ResetComputeEnvCache()
	if v41Q3KTokenPanelEnabled() {
		t.Fatal("explicit 0 must disable the Q3 panel")
	}
	// Mutating the environment without a reset must not change the cached answer.
	t.Setenv("FAK_V41_Q3K_TOKEN_PANEL", "1")
	if v41Q3KTokenPanelEnabled() {
		t.Fatal("cached flag was re-read from the environment")
	}
	ResetComputeEnvCache()
	if !v41Q3KTokenPanelEnabled() {
		t.Fatal("reset did not observe the current environment")
	}
}
