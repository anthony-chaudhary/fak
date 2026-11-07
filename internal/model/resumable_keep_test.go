package model

import (
	"reflect"
	"testing"
)

// TestResumableKeepBoundaries pins the resumable-boundary and keep-trim policy
// adapted from modular jenga_block_manager: boundary-set divergence, longest-hit
// validation, the lowest-keep window rule, and recurrent publish suppression.
func TestResumableKeepBoundaries(t *testing.T) {
	const page = 4

	t.Run("boundary set diverges exactly when a page-aligned prompt ends mid-step", func(t *testing.T) {
		// Case A: page-aligned prompt (8 tokens) ending inside this step's
		// commit range (-1, 12]. The replay boundary (8/4-1=1) diverges from
		// the committed boundary (12/4=3).
		p := NewResumableKeepPolicy(page, true, true)
		full := CacheGroup{Shape: ShapeFull, Published: []bool{true, true, true, true}}
		req := NewResumableState()
		keep := p.Step(&req, 12, 8, full)
		if want := []int{1, 3}; !reflect.DeepEqual(keep, want) {
			t.Fatalf("page-aligned prompt mid-step: keep = %v, want %v", keep, want)
		}
		if req.CommittedIdx != 12 {
			t.Fatalf("committed idx = %d, want 12", req.CommittedIdx)
		}

		// Case B: same step, prompt of 10 tokens does not end on a page
		// boundary, so no distinct replay boundary exists: only the committed
		// boundary survives.
		reqB := NewResumableState()
		keepB := p.Step(&reqB, 12, 10, full)
		if want := []int{3}; !reflect.DeepEqual(keepB, want) {
			t.Fatalf("off-page prompt: keep = %v, want %v", keepB, want)
		}

		// Case C: page-aligned prompt that already ended in an earlier step:
		// the replay candidate is out of range (-1..8 vs 8..12), so the second
		// step records only its committed boundary.
		reqC := NewResumableState()
		if keepC := p.Step(&reqC, 8, 8, full); !reflect.DeepEqual(keepC, []int{1, 2}) {
			t.Fatalf("first step: keep = %v, want [1 2]", keepC)
		}
		if keepC2 := p.Step(&reqC, 12, 8, full); !reflect.DeepEqual(keepC2, []int{3}) {
			t.Fatalf("second step: keep = %v, want [3]", keepC2)
		}

		// Fast path: nothing new committed and at most one recorded boundary
		// reuses the previous boundaries unchanged.
		reqD := NewResumableState()
		p.Step(&reqD, 4, 8, full) // keep = [1]
		if keepD := p.Step(&reqD, 4, 8, full); !reflect.DeepEqual(keepD, []int{1}) {
			t.Fatalf("fast path: keep = %v, want [1]", keepD)
		}
	})

	t.Run("candidate rejected when depth differs from longest published hit", func(t *testing.T) {
		p := NewResumableKeepPolicy(page, true, true)

		// Full group: leading published run is 2, so depth 3 cannot resume.
		gap := CacheGroup{Shape: ShapeFull, Published: []bool{true, true, false, true, true}}
		if p.ResumableAt(3, 5, gap) {
			t.Fatal("full group: depth 3 accepted though longest published hit is 2")
		}
		if !p.ResumableAt(2, 5, gap) {
			t.Fatal("full group: depth 2 rejected though longest published hit is 2")
		}

		// Sliding-window group (window 2): the deepest complete trailing
		// window over [T T F T] ends at block 2, so depth 4 cannot resume.
		win := CacheGroup{Shape: ShapeSlidingWindow, BlocksInWindow: 2, Published: []bool{true, true, false, true}}
		if p.ResumableAt(4, 4, win) {
			t.Fatal("sliding window: depth 4 accepted though longest published hit is 2")
		}
		if !p.ResumableAt(2, 4, win) {
			t.Fatal("sliding window: depth 2 rejected though longest published hit is 2")
		}
		// A complete trailing window at the tail validates the full depth.
		winTail := CacheGroup{Shape: ShapeSlidingWindow, BlocksInWindow: 2, Published: []bool{false, false, true, true}}
		if !p.ResumableAt(4, 4, winTail) {
			t.Fatal("sliding window: depth 4 rejected though the trailing window is complete")
		}

		// Range guards: zero and beyond-committed depths never resume.
		allPub := CacheGroup{Shape: ShapeFull, Published: []bool{true, true, true, true}}
		if p.ResumableAt(0, 4, allPub) {
			t.Fatal("depth 0 accepted")
		}
		if p.ResumableAt(5, 4, allPub) {
			t.Fatal("depth past committed blocks accepted")
		}
	})

	t.Run("lowest-keep window rule", func(t *testing.T) {
		// With keep boundaries {k1=2 < k2=6}, the sliding-window group keeps
		// the window ending at k1; it covers every higher keep boundary.
		firstNeeded, end := SlidingWindowTrim(2, 6, 2, 6)
		if end != 2 || firstNeeded != 0 {
			t.Fatalf("trim(keep={2,6}) = [%d,%d), want [0,2)", firstNeeded, end)
		}
		// A higher lowest-boundary trims to a nonzero floor.
		if fn, e := SlidingWindowTrim(2, 6, 4, 6); e != 4 || fn != 2 {
			t.Fatalf("trim(keep={4,6}) = [%d,%d), want [2,4)", fn, e)
		}
		// With nothing in keep, the window ends at the committed depth.
		if fn, e := SlidingWindowTrim(2, 6); e != 6 || fn != 4 {
			t.Fatalf("trim(no keep) = [%d,%d), want [4,6)", fn, e)
		}
		// A full-history group keeps every block regardless of keep.
		if fn, e := FullGroupTrim(6, 2, 6); e != 6 || fn != 0 {
			t.Fatalf("full trim = [%d,%d), want [0,6)", fn, e)
		}
	})

	t.Run("recurrent publish suppressed when replay already resumable", func(t *testing.T) {
		p := NewResumableKeepPolicy(page, true, true)
		full := CacheGroup{Shape: ShapeFull, Published: []bool{true, true, true, true}}

		// Step one commits 0->12 for a 16-token page-aligned prompt: the
		// replay boundary 16/4-1=3 is validated now, before the prompt ends.
		req := NewResumableState()
		if keep := p.Step(&req, 12, 16, full); !reflect.DeepEqual(keep, []int{3}) {
			t.Fatalf("setup: keep = %v, want [3]", keep)
		}

		// The forward that just ran ends the prompt (processedLength 16) and
		// its replay can already resume from boundary 3: recurrent publish is
		// suppressed.
		if p.PublishesStates(&req, 12, 16, 16) {
			t.Fatal("publish not suppressed: replay boundary 3 already resumable")
		}

		// Controls: none of these suppress.
		if !p.PublishesStates(&req, 12, 15, 16) {
			t.Fatal("suppressed although the prompt has not ended in this forward")
		}
		if !p.PublishesStates(&req, 12, 16, 15) {
			t.Fatal("suppressed although the prompt is not page-aligned")
		}
		noEvents := NewResumableKeepPolicy(page, false, false)
		if !noEvents.PublishesStates(&req, 12, 16, 16) {
			t.Fatal("suppressed although the checkpointer handles no end-of-prompt event")
		}
		empty := NewResumableState()
		if !p.PublishesStates(&empty, 12, 16, 16) {
			t.Fatal("suppressed although the replay boundary is not resumable")
		}
	})
}
