package radixkv

import (
	"strings"
	"testing"
)

// prefix_hit_test.go — Upstream test vectors for the joint-prefix matcher ported
// from modular/modular rust_kv/kv-core/src/prefix_hit.rs at revision
// 8d559749621418f39f31df7da998bcce884fa5b1 (vendored near-verbatim under
// third_party/modular/rust_kv; per-file byte receipts in PINNED-SOURCE.json).
// The upstream Rust test module is the oracle: every subtest below ports one
// #[test] from that module by name, with exact integer equality.

// held adapts a positional residency mask to the predicate form the rules take.
func held(mask []bool) func(int) bool {
	return func(idx int) bool { return mask[idx] }
}

// fullRule is one group's callback under the full-attention rule.
func fullRule(mask []bool) func(int) int {
	return func(candidate int) int { return LongestFullAttentionHit(candidate, held(mask)) }
}

// windowRule is one group's callback under the sliding-window rule.
func windowRule(mask []bool, blocksInWindow int) func(int) int {
	return func(candidate int) int {
		return LongestSlidingWindowHit(candidate, blocksInWindow, held(mask))
	}
}

// jointHit is the ported form of the joint rule: one callable per group, as the
// upstream test helper spells it.
func jointHit(numHashes int, rules ...func(int) int) int {
	return LongestJointPrefixHit(numHashes, len(rules), func(gi, candidate int) int {
		return rules[gi](candidate)
	})
}

// view builds a GroupResidency that asked about probed blocks and can move
// loadableBlocks of them from slotOffset.
func view(slotOffset, localPrefixPages, blocksInWindow, probed, loadableBlocks int, resident []bool) *GroupResidency {
	return &GroupResidency{
		SlotOffset:       slotOffset,
		LocalPrefixPages: localPrefixPages,
		BlocksInWindow:   blocksInWindow,
		Probed:           probed,
		LoadableBlocks:   loadableBlocks,
		Resident:         resident,
	}
}

// TestPrefixHitUpstreamVectors ports every #[test] vector from the upstream
// prefix_hit.rs test module. Each subtest names its upstream source test.
//
// fak-test:runtime fast est=10ms lane=default
func TestPrefixHitUpstreamVectors(t *testing.T) {
	t.Run("full_attention_stops_at_the_first_hole", func(t *testing.T) {
		mask := []bool{true, true, true, false, true}
		if got := LongestFullAttentionHit(5, held(mask)); got != 3 {
			t.Fatalf("full attention hit = %d, want 3", got)
		}
	})

	t.Run("sliding_window_takes_the_deepest_complete_window", func(t *testing.T) {
		// MAX's mask for this case, and deliberately one with TWO complete
		// windows -- maximal runs at 2..3 and 5..6, both long enough for a
		// window of 2 -- so returning the shallower one is a visible failure.
		// The window also sits past a gap here: slots 0 and 1 are missing and
		// are never read at this stopping point, so they cost nothing.
		mask := []bool{false, false, true, true, false, true, true}
		if got := LongestSlidingWindowHit(7, 2, held(mask)); got != 7 {
			t.Fatalf("sliding-window hit = %d, want 7", got)
		}
	})

	t.Run("sliding_window_falls_back_to_a_root_anchored_run", func(t *testing.T) {
		// No run reaches the window, so all that is left is the run that
		// survives to the root, which needs no block under it.
		mask := []bool{true, true, false, true}
		if got := LongestSlidingWindowHit(4, 3, held(mask)); got != 2 {
			t.Fatalf("sliding-window hit = %d, want 2", got)
		}
	})

	t.Run("two_full_groups_agree_on_the_shorter_run", func(t *testing.T) {
		a := []bool{true, true, true, false}
		b := []bool{true, true, false, false}
		if got := jointHit(4, fullRule(a), fullRule(b)); got != 2 {
			t.Fatalf("joint = %d, want 2", got)
		}
	})

	t.Run("full_and_windowed_agree_below_both_their_answers", func(t *testing.T) {
		// The case no minimum can reach. Full holds A..F; the windowed group
		// holds [C D] and [F G] with window 2, and is missing E. E rules out
		// BOTH of the first two candidates, so the answer is not reachable by
		// shortening either group's own answer once.
		full := []bool{true, true, true, true, true, true, false}
		window := []bool{false, false, true, true, false, true, true}
		if got := jointHit(7, fullRule(full), windowRule(window, 2)); got != 4 {
			t.Fatalf("joint = %d, want 4", got)
		}
	})

	t.Run("a_window_the_full_group_cannot_reach_is_no_hit", func(t *testing.T) {
		// The windowed group holds a COMPLETE window, [F G], but it ends past
		// where the full group reaches. Capping the candidate at F leaves the
		// windowed group holding only F: a run of 1 against a window of 2, and
		// nothing shallower has two in a row either.
		full := []bool{true, true, true, true, true, true, false}
		window := []bool{false, false, false, false, false, true, true}
		if got := jointHit(7, fullRule(full), windowRule(window, 2)); got != 0 {
			t.Fatalf("joint = %d, want 0", got)
		}
	})

	t.Run("narrowing_reopens_a_group_already_asked", func(t *testing.T) {
		// Why the joint rule ITERATES rather than asking each group once. The
		// windowed group answers 2 (B alone is a complete window of 1), which
		// narrows the candidate to 2; the full group then answers 1, because
		// its run from the root is just A. Re-asking the windowed group at 1
		// is what catches that B is gone from that candidate and A is not
		// held: a single round-robin pass stops at 1 and claims a hit the KV
		// cannot back.
		window := []bool{false, true}
		full := []bool{true, false}
		if got := jointHit(2, windowRule(window, 1), fullRule(full)); got != 0 {
			t.Fatalf("joint = %d, want 0", got)
		}
	})

	t.Run("two_windowed_groups_converge_below_the_minimum_of_their_answers", func(t *testing.T) {
		// Two WINDOWED groups of different widths, which is the shape the
		// run-and-min rule this replaced could not express at all: neither
		// answer is a depth the other can be compared against, so a minimum is
		// not merely imprecise here, it is unsound. Narrow (window 2) holds
		// [0 1 2] and [5 6]; wide (window 3) holds [0 1 2 3]. Their own
		// answers are 7 and 4, so a minimum claims 4. At 4 the narrow group's
		// window is [2 3] and it does not hold 3, so 4 is not servable. Only
		// re-asking converges, on 3.
		narrow := []bool{true, true, true, false, false, true, true}
		wide := []bool{true, true, true, true, false, false, false}

		if got := LongestSlidingWindowHit(7, 2, held(narrow)); got != 7 {
			t.Fatalf("narrow alone = %d, want 7", got)
		}
		if got := LongestSlidingWindowHit(7, 3, held(wide)); got != 4 {
			t.Fatalf("wide alone = %d, want 4", got)
		}

		joint := jointHit(7, windowRule(narrow, 2), windowRule(wide, 3))
		if joint != 3 {
			t.Fatalf("joint = %d, want 3 (the minimum of the two answers, 4, is not "+
				"servable: the narrow group's window at 4 needs block 3, which it does not hold)", joint)
		}
		// Both groups really do hold their window at the agreed depth.
		if got := LongestSlidingWindowHit(joint, 2, held(narrow)); got != joint {
			t.Fatalf("narrow at agreed depth = %d, want %d", got, joint)
		}
		if got := LongestSlidingWindowHit(joint, 3, held(wide)); got != joint {
			t.Fatalf("wide at agreed depth = %d, want %d", got, joint)
		}
	})

	t.Run("a_lone_windowed_group_needs_nothing_to_bound_it", func(t *testing.T) {
		window := []bool{false, false, true, true}
		if got := jointHit(4, windowRule(window, 2)); got != 4 {
			t.Fatalf("joint = %d, want 4", got)
		}
	})

	t.Run("no_groups_is_no_hit", func(t *testing.T) {
		if got := jointHit(5); got != 0 {
			t.Fatalf("joint = %d, want 0", got)
		}
	})

	t.Run("blocks_in_window_excludes_the_query_token", func(t *testing.T) {
		// MAX's own table for the same arithmetic
		// (test_blocks_in_window_excludes_the_query_token). The -1 sentinel is
		// the Go spelling of upstream None: a full-attention group has no
		// window at all, where MAX carries a -1 sentinel it translates at its
		// own boundary.
		for _, tc := range []struct {
			window, pageSize, want int
		}{
			{2, 128, 1}, // two is the shallowest window config validation admits
			{128, 128, 1},
			{129, 128, 1},
			{130, 128, 2},
			{1024, 128, 8}, // Gemma 4's geometry, the one this PR is measured on
			{-1, 128, -1},  // full attention: no window at all
		} {
			if got := BlocksInWindowOf(tc.window, tc.pageSize); got != tc.want {
				t.Fatalf("BlocksInWindowOf(%d, %d) = %d, want %d", tc.window, tc.pageSize, got, tc.want)
			}
		}
	})

	t.Run("a_joint_never_exceeds_what_a_group_can_move", func(t *testing.T) {
		// The cap is this module's invariant, so this module proves it rather
		// than leaving the proof one crate away. A windowed group holding a
		// complete window above a gap is a true statement about residency and
		// an unfillable one about transfer; the joint must not exceed any
		// group's LoadableDepth.
		full := make([]bool, 19)
		for i := range full {
			full[i] = true
		}
		window := make([]bool, 19)
		for slot := 11; slot < 19; slot++ {
			window[slot] = true
		}
		// The rule does see the window; the leading run reaches nothing.
		if got := LongestSlidingWindowHit(19, 8, held(window)); got != 19 {
			t.Fatalf("window rule sees the band: hit = %d, want 19", got)
		}

		groups := []*GroupResidency{
			view(0, 0, -1, 19, 19, full),
			view(0, 0, 8, 19, 0, window),
		}
		if got := LongestJointResidencyHit(groups); got != 0 {
			t.Fatalf("joint = %d, want 0 (the joint must not exceed any group's LoadableDepth)", got)
		}
	})

	t.Run("the_cap_bounds_the_candidate_not_the_answer", func(t *testing.T) {
		// Capping the CANDIDATE is not the same as capping the ANSWER, and
		// this is the case that separates them. The windowed group can move
		// one block from slot 3 and holds no complete window of 2 at or below
		// that depth, so the only sound joint is 0. An answer-cap would run
		// the rules unbounded and then clamp, keeping a number the windowed
		// group never endorsed at the depth it was clamped to.
		full := make([]bool, 12)
		for i := range full {
			full[i] = true
		}
		window := make([]bool, 6)
		for i := range window {
			window[i] = true
		}
		groups := []*GroupResidency{
			view(0, 0, -1, 12, 12, full),
			view(3, 0, 2, 6, 1, window),
		}
		if got := LongestJointResidencyHit(groups); got != 0 {
			t.Fatalf("joint = %d, want 0 (the only sound joint)", got)
		}

		answerCapped := LongestJointPrefixHit(12, len(groups), func(gi, candidate int) int {
			g := groups[gi]
			return LongestGroupHit(candidate, g.BlocksInWindow, g.IsCached)
		})
		fillable := groups[0].LoadableDepth()
		for _, g := range groups[1:] {
			if d := g.LoadableDepth(); d < fillable {
				fillable = d
			}
		}
		if answerCapped > fillable {
			answerCapped = fillable
		}
		if answerCapped == 0 {
			t.Fatal("answer-capped variant also gave 0; if these agreed, this test would not be pinning the choice")
		}
	})

	t.Run("a_null_head_reads_absent_below_its_offset", func(t *testing.T) {
		// The below-offset branch decides whether a null head reads as present,
		// and only a band whose held run is SHORTER than its window reaches it:
		// the walk exhausts the band, resets, and steps down through the
		// offset. A band whose run already fills its window returns before it
		// gets there, which is why the partly-filled-band case does not cover
		// this.
		full := make([]bool, 6)
		for i := range full {
			full[i] = true
		}
		// Two held blocks at slots 4..6 against a window of four.
		shortBand := []bool{true, true}
		nullHeaded := []*GroupResidency{
			view(0, 0, -1, 6, 6, full),
			view(4, 0, 4, 2, 2, shortBand),
		}
		if got := LongestJointResidencyHit(nullHeaded); got != 0 {
			t.Fatalf("null head = %d, want 0 (a null head must not read as resident: nothing fills the window)", got)
		}

		// The same band whose head IS resident locally does serve it, which is
		// the other direction of the same branch.
		localHead := []*GroupResidency{
			view(0, 0, -1, 6, 6, full),
			view(4, 4, 4, 2, 2, shortBand),
		}
		if got := LongestJointResidencyHit(localHead); got != 6 {
			t.Fatalf("local head = %d, want 6", got)
		}
	})

	t.Run("blocks_held_of_hit", func(t *testing.T) {
		for _, tc := range []struct {
			numHitBlocks, blocksInWindow, want int
		}{
			{4, -1, 4}, // a full group holds the whole hit
			{4, 2, 2},  // a windowed group holds only its window
			{2, 8, 2},  // a window wider than the hit holds the whole hit
		} {
			if got := BlocksHeldOfHit(tc.numHitBlocks, tc.blocksInWindow); got != tc.want {
				t.Fatalf("BlocksHeldOfHit(%d, %d) = %d, want %d", tc.numHitBlocks, tc.blocksInWindow, got, tc.want)
			}
		}
	})

	t.Run("a_recurrent_leaf_takes_its_deepest_state", func(t *testing.T) {
		mask := []bool{true, false, true, false}
		if got := LongestRecurrentHit(4, held(mask)); got != 3 {
			t.Fatalf("recurrent hit = %d, want 3", got)
		}
		// Narrowed below that state, it answers from the next one down.
		if got := LongestRecurrentHit(2, held(mask)); got != 1 {
			t.Fatalf("recurrent hit narrowed = %d, want 1", got)
		}
		if got := LongestRecurrentHit(4, held([]bool{false, false, false, false})); got != 0 {
			t.Fatalf("recurrent hit absent = %d, want 0", got)
		}
	})

	t.Run("each_shape_selects_its_rule", func(t *testing.T) {
		mask := []bool{true, false, true, true}
		at := func(shape LeafShape) int { return shape.LongestHit(len(mask), held(mask)) }
		if got := at(LeafShape{Kind: LeafFull}); got != 1 {
			t.Fatalf("Full = %d, want 1", got)
		}
		if got := at(LeafShape{Kind: LeafSlidingWindow, BlocksInWindow: 2}); got != 4 {
			t.Fatalf("SlidingWindow{2} = %d, want 4", got)
		}
		if got := at(LeafShape{Kind: LeafRecurrent}); got != 4 {
			t.Fatalf("Recurrent = %d, want 4", got)
		}
		if got := at(LeafShape{Kind: LeafScratch}); got != 4 {
			t.Fatalf("Scratch = %d, want 4", got)
		}
		scratch := LeafShape{Kind: LeafScratch}
		if got := scratch.LongestHit(3, func(int) bool { return false }); got != 3 {
			t.Fatalf("Scratch over an empty cache = %d, want 3", got)
		}
	})

	t.Run("a_scratch_leaf_never_narrows_the_joint", func(t *testing.T) {
		full := []bool{true, true, false}
		shapes := []LeafShape{{Kind: LeafScratch}, {Kind: LeafFull}}
		joint := LongestJointPrefixHit(3, len(shapes), func(gi, candidate int) int {
			return shapes[gi].LongestHit(candidate, held(full))
		})
		if joint != 2 {
			t.Fatalf("joint = %d, want 2", joint)
		}
	})

	t.Run("a_recurrent_leaf_settles_with_a_window", func(t *testing.T) {
		// The state at 4 is narrowed to 3 by the window, which drops the state
		// to the one at 2; the window holds [0, 1] complete, so both agree there.
		state := []bool{false, true, false, true}
		window := []bool{true, true, true, false}
		rules := []func(int) int{
			func(candidate int) int { return LongestRecurrentHit(candidate, held(state)) },
			func(candidate int) int { return LongestSlidingWindowHit(candidate, 2, held(window)) },
		}
		if got := jointHit(4, rules...); got != 2 {
			t.Fatalf("joint = %d, want 2", got)
		}
	})
}

// TestPrefixHitRejectsZeroWindow pins the capability rejection: a zero window
// is window_size == 1, a query token attending to no history, which would make
// every page a hit that attention never reads back. Upstream rejects that
// window at config validation (SERVOPT-1627) and debug-asserts in the rule;
// the port panics, because this sits on the per-request load path and a zero
// has no answer to return, only a caller to fix.
//
// fak-test:runtime fast est=10ms lane=default
func TestPrefixHitRejectsZeroWindow(t *testing.T) {
	defer func() {
		rec := recover()
		if rec == nil {
			t.Fatal("LongestSlidingWindowHit(5, 0, ...) did not panic; a zero window must be rejected")
		}
		msg, ok := rec.(string)
		if !ok || !strings.Contains(msg, "spans at least one block") {
			t.Fatalf("panic value %v does not name the zero-window rejection", rec)
		}
	}()
	LongestSlidingWindowHit(5, 0, func(int) bool { return false })
}
