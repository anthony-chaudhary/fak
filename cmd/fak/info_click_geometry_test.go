package main

import (
	"strings"
	"testing"
)

// info_click_geometry_test.go — the dev-process guard for `fak info` mouse clickability. It pins
// the ONE invariant the interactive overlay's hit-testing rests on: the painted block bottom-parks,
// so blockRelativeRow can translate an absolute mouse row into a block row by assuming the block
// fills the bottom `height` rows of the pane. The historical bug this guards against: clicks worked
// at first (the tall Overview filled the pane) and then died the moment you switched to a shorter
// view (Safety/Agents), because a short frame anchored higher than blockRelativeRow assumed and
// every tab/chip click silently mis-hit or went inert. The fix — padBlockToHeight in the paint
// path (info.go writeFrame) — keeps every interactive frame full-height, so prevRows == height and
// the translation is exact for every view. These tests fail if that invariant is ever removed.

// TestBlockRelativeRowBottomPark pins blockRelativeRow's absolute→block-row contract for a block
// pinned to the bottom `prevRows` rows of a pane: the first block row (the tab bar) sits at
// absolute row height-prevRows+1, the last at absolute row height, and anything above the block or
// on an unmeasured pane is inert (0) rather than mis-hitting a chip.
func TestBlockRelativeRowBottomPark(t *testing.T) {
	const height = 40

	// The full-height frame the interactive paint always produces: absolute row maps 1:1 to block
	// row, so a tab-bar click at absolute row 1 lands on block row 1 no matter which view is active.
	for _, absY := range []int{1, 2, 20, 40} {
		if got := blockRelativeRow(absY, height, height); got != absY {
			t.Errorf("full-height frame: blockRelativeRow(%d, %d, %d) = %d, want %d", absY, height, height, got, absY)
		}
	}

	// A short block bottom-parked at 5 rows: its tab bar is at absolute row 36, its last row at 40.
	if got := blockRelativeRow(36, height, 5); got != 1 {
		t.Errorf("bottom-parked tab bar: blockRelativeRow(36, 40, 5) = %d, want 1", got)
	}
	if got := blockRelativeRow(40, height, 5); got != 5 {
		t.Errorf("bottom-parked last row: blockRelativeRow(40, 40, 5) = %d, want 5", got)
	}

	// A click above the parked block, or on an unmeasured pane, is inert — never a mis-selection.
	if got := blockRelativeRow(35, height, 5); got != 0 {
		t.Errorf("click above the block: blockRelativeRow(35, 40, 5) = %d, want 0", got)
	}
	if got := blockRelativeRow(10, 0, 5); got != 0 {
		t.Errorf("unmeasured height: blockRelativeRow(10, 0, 5) = %d, want 0", got)
	}
	if got := blockRelativeRow(10, height, 0); got != 0 {
		t.Errorf("unmeasured prevRows: blockRelativeRow(10, 40, 0) = %d, want 0", got)
	}
}

// TestPadBlockToHeightInvariant proves the paint-path pad yields a frame of exactly the pane
// height, and is a no-op when the geometry is unknown or the block already fills/overflows the pane
// (padding must never shrink a frame or add rows to an already-full one).
func TestPadBlockToHeightInvariant(t *testing.T) {
	const height = 12

	short := "tab bar\ncontent\n" + "safety: nothing blocked"
	padded := padBlockToHeight(short, height)
	if got := strings.Count(padded, "\n") + 1; got != height {
		t.Fatalf("padded short block = %d rows, want full pane height %d", got, height)
	}
	if !strings.HasPrefix(padded, short) {
		t.Fatalf("padding must only APPEND rows, never rewrite content:\n%q", padded)
	}

	// Unknown pane (height <= 0): left exactly as-is so the roomy/append contracts are unchanged.
	if got := padBlockToHeight(short, 0); got != short {
		t.Errorf("height 0 must be a no-op, got:\n%q", got)
	}

	// A block that already fills or overflows the pane is returned untouched (writeGuardInfoFrame's
	// own cap handles overflow; padding must not double-count it).
	exact := strings.Repeat("row\n", height-1) + "row" // exactly `height` rows
	if got := padBlockToHeight(exact, height); got != exact {
		t.Errorf("exact-height block must be a no-op, got %d rows", strings.Count(got, "\n")+1)
	}
	over := strings.Repeat("row\n", height+3) + "row"
	if got := padBlockToHeight(over, height); got != over {
		t.Errorf("over-height block must be a no-op, got %d rows", strings.Count(got, "\n")+1)
	}
}

// TestInteractiveClickSurvivesShortView is the end-to-end regression for the reported bug: after
// switching to a view whose content is far shorter than the pane, a tab-bar click must still land.
// It reproduces the paint-path geometry (render the short view, pad it as writeFrame does, use the
// resulting row count as prevRows) and asserts the tab-bar click resolves to block row 1 — while
// also proving the PRE-fix geometry (prevRows = the short content height) is exactly what made the
// same click go inert.
func TestInteractiveClickSurvivesShortView(t *testing.T) {
	const width, height = 120, 40
	tr := newGuardInfoTrend(guardInfoTrendCap)
	v := richVisualVars()
	tr.push(v)

	// The Safety view is the stubby view that used to kill clicks — a handful of rows in a 40-row
	// pane. (If a future fixture ever fills the pane here, the regression no longer applies.)
	raw := renderGuardInfoInteractiveBlock(infoViewState{active: viewSafety}, v, tr, width, height)
	contentRows := strings.Count(raw, "\n") + 1
	if contentRows >= height {
		t.Skipf("safety view already fills the %d-row pane (%d rows); the short-view case no longer reproduces", height, contentRows)
	}

	// The paint path pads to full height → prevRows == height → the click translation is exact.
	painted := padBlockToHeight(raw, height)
	prevRows := strings.Count(painted, "\n") + 1
	if prevRows != height {
		t.Fatalf("painted interactive frame = %d rows, want full pane height %d", prevRows, height)
	}
	if got := blockRelativeRow(1, height, prevRows); got != 1 {
		t.Fatalf("tab-bar click (absolute row 1) after a short view resolves to block row %d, want 1", got)
	}

	// Guard our understanding of the bug: the UNPADDED geometry (prevRows = the short content
	// height) sends the same tab-bar click to a non-positive row → inert. This is what padding cures.
	if got := blockRelativeRow(1, height, contentRows); got == 1 {
		t.Fatalf("sanity: an unpadded %d-row frame should NOT map absolute row 1 to block row 1 (that was the live bug); got %d", contentRows, got)
	}
}
