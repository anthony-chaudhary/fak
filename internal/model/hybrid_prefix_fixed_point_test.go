// Package model_test keeps this fixture in the external test package: the
// production joint matcher lives in internal/radixkv, which imports
// internal/model, so an internal (package model) test file could not import it
// back without an import cycle. pagedkv_radix_test.go sets the same precedent.
package model_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// hybrid_prefix_fixed_point_test.go — Candidate-D fixture for the 2026-11-03
// OSS-study follow-up: joint hybrid-prefix validity is a fixed point.
//
// Contract under test (paraphrased behavioral contract, clean-room): with hybrid
// attention (sliding-window + full/dense groups), a prefix hit valid for one group is
// NOT downward-closed — shortening the shared prefix to satisfy one group can
// invalidate another group's earlier answer. Joint reuse must iterate: repeatedly
// narrow until ALL groups agree on the reused length (a fixed point), clamping
// per-group callback answers, rather than taking a single simple minimum.
//
// Study verdict this fixture pins: NO active cache path in fak combines two group
// matchers into one reuse decision. The two leaf matchers named by the study are
// test-only:
//   - radixkv.PagedRadixKVPool.BindPrefix (uniform page validity, paged_binding.go)
//   - model.AlignRadixTree.MatchLongestPrefix (block-aligned validity, gdn_align_cache.go:367)
// Live hybrid paths fail closed instead of combining: gemma4 refuses KV prefix reuse
// outright (gemma4_session.go KVPrefixReuseSupported), recurrent hybrids restore only
// whole-prefix snapshots at block boundaries (agent/inkernel_host_snapshot_reuse.go),
// and the sliding-window trim has no production caller (swa.go:14). Per-group radix
// payloads are the declared R2 follow-on (kvgroups.go header).
//
// This fixture is therefore a CONTRACT-LEVEL test of the narrowing algorithm. Group B's
// validity is the REAL AlignRadixTree.MatchLongestPrefix (the existing block-aligned
// matcher) and group A's window-band validity models the TrimToWindow /
// KVLayerResidentPositions fact; both feed the PRODUCTION joint matcher
// (radixkv.LongestJointPrefixHit, the rust_kv/kv-core prefix_hit.rs port) through the
// group-clamp interface below. fixedPointNarrow remains as a tiny local reference
// implementation, clearly labelled TEST-ONLY REFERENCE, used only to cross-check that the
// production matcher settles on the same jointly valid length.

// groupClamp is one attention group's callback: given a candidate reuse length, it
// returns the longest length <= candidate that this group can actually serve. A
// well-formed clamp satisfies g(0) == 0, g(c) <= c, and monotone non-decreasing.
type groupClamp func(candidate int) int

// assertClampWellFormed checks the three well-formedness properties over 0..max.
func assertClampWellFormed(t *testing.T, name string, g groupClamp, max int) {
	t.Helper()
	if got := g(0); got != 0 {
		t.Fatalf("%s: clamp(0) = %d, want 0", name, got)
	}
	prev := 0
	for c := 1; c <= max; c++ {
		got := g(c)
		if got > c {
			t.Fatalf("%s: clamp(%d) = %d > candidate", name, c, got)
		}
		if got < prev {
			t.Fatalf("%s: clamp not monotone: clamp(%d)=%d < clamp(%d)=%d", name, c, got, c-1, prev)
		}
		prev = got
	}
}

// jointlyValid reports whether length L is served by EVERY group (well-formed clamps:
// g(L) == L iff L is in g's valid set).
func jointlyValid(groups []groupClamp, L int) bool {
	for _, g := range groups {
		if g(L) != L {
			return false
		}
	}
	return true
}

// naiveSinglePassMin is the policy the contract warns against: ONE round of per-group
// clamping from the structural candidate, no re-check. Sound only when every group's
// valid set is downward-closed.
func naiveSinglePassMin(groups []groupClamp, candidate int) int {
	best := candidate
	for _, g := range groups {
		if c := g(candidate); c < best {
			best = c
		}
	}
	return best
}

// fixedPointNarrow is the CONTRACT policy: iterate per-group clamping until every
// group's answer agrees on the current length. Monotone clamps guarantee termination;
// the result is the greatest jointly valid length (0 = total disagreement, full miss).
// Returns the settled length and the number of NARROWING ROUNDS — passes in which the
// candidate strictly decreased (the final confirming pass, where no group moves, is not
// counted). A downward-closed-only fleet settles in exactly one narrowing round.
func fixedPointNarrow(groups []groupClamp, candidate int) (int, int) {
	cur := candidate
	narrowings := 0
	for {
		next := cur
		for _, g := range groups {
			if c := g(cur); c < next {
				next = c
			}
		}
		if next == cur {
			return cur, narrowings
		}
		cur = next
		narrowings++
	}
}

// productionGroupHit adapts the fixture's group clamps to the PRODUCTION matcher's
// indexed callback (radixkv.LongestJointPrefixHit) and counts NARROWING TURNS — turns in
// which a group's answer falls strictly below the candidate it was asked about. The
// production matcher clamps over-answers itself; pre-clamping here keeps the counter's
// semantics identical either way.
func productionGroupHit(groups []groupClamp, narrowingTurns *int) func(group, candidate int) int {
	return func(gi, candidate int) int {
		ans := groups[gi](candidate)
		if ans > candidate {
			ans = candidate
		}
		if ans < candidate {
			*narrowingTurns++
		}
		return ans
	}
}

// windowBandGroup models a sliding-window attention group whose per-group KV pool was
// trimmed to rows [lo, hi] — the correctness fact Session.TrimToWindow (swa.go) and
// Config.KVLayerResidentPositions (kvgroups.go) both exploit: positions older than the
// window are provably un-attended and may be dropped. A reuse length L is servable iff
// its attend span [L-w, L) lies inside [lo, hi], so the valid set is the BAND
// [lo+w, hi+1]: NOT downward-closed — the group accepts a longer prefix while rejecting
// every shorter one below the band floor.
func windowBandGroup(lo, hi, w int) groupClamp {
	top := hi + 1   // largest L with span end L-1 <= hi
	floor := lo + w // smallest L with span start L-w >= lo
	return func(candidate int) int {
		if candidate <= 0 {
			return 0
		}
		if candidate > top {
			candidate = top
		}
		if candidate < floor {
			return 0
		}
		return candidate
	}
}

// alignBlockGroup adapts the REAL AlignRadixTree.MatchLongestPrefix
// (gdn_align_cache.go:367) to the group-clamp interface. The matcher walks whole blocks
// only, so its answer for a candidate is the block-aligned floor of the stored prefix —
// a lattice validity, also not downward-closed.
func alignBlockGroup(tree *model.AlignRadixTree, toks []int) groupClamp {
	return func(candidate int) int {
		if candidate <= 0 {
			return 0
		}
		if candidate > len(toks) {
			candidate = len(toks)
		}
		_, matched, _ := tree.MatchLongestPrefix(toks[:candidate])
		return matched
	}
}

// TestHybridPrefixFixedPointJointValidity distinguishes a naive single-pass minimum
// from iterative narrowing to a fixed point on a hybrid (window + dense-block) pair,
// running the contract through the PRODUCTION joint matcher
// (radixkv.LongestJointPrefixHit) with the local reference kept as a cross-check oracle.
// The name carries the HybridPrefixFixedPoint token so the qualification witness
// (-run 'PrefixHit|HybridPrefixFixedPoint') selects it.
func TestHybridPrefixFixedPointJointValidity(t *testing.T) {
	const (
		l0      = 40 // structural candidate: the shared structure sees 40 tokens
		blockSz = 4  // dense group's restoration lattice (whole blocks only)
		wLo     = 17 // window group's trimmed row span [17, 38]
		wHi     = 38
		winW    = 16 // window width
	)

	// 40 distinct token ids, stored block-aligned in the REAL align tree.
	toks := make([]int, l0)
	for i := range toks {
		toks[i] = 1000 + i*7
	}
	slabs := make([]*model.RecurrentStateSlab, l0/blockSz)
	for b := range slabs {
		slabs[b] = model.NewRecurrentStateSlab(b, blockSz, toks[b*blockSz:(b+1)*blockSz], nil)
	}
	tree := model.NewAlignRadixTree(blockSz)
	if err := tree.InsertSequence(toks, slabs); err != nil {
		t.Fatalf("InsertSequence: %v", err)
	}

	// Group A: sliding-window band [33, 39] (accepts 39, rejects 32 and below).
	grpA := windowBandGroup(wLo, wHi, winW)
	// Group B: dense group restorable only at multiples of 4 (REAL matcher).
	grpB := alignBlockGroup(tree, toks)
	groups := []groupClamp{grpA, grpB}

	assertClampWellFormed(t, "window-band", grpA, l0)
	assertClampWellFormed(t, "align-block", grpB, l0)

	// The scenario the contract names: B's round-1 answer (40) covers position 32,
	// which A's window band (floor 33) invalidates.
	if got := grpB(32); got != 32 {
		t.Fatalf("setup: dense group must accept 32, clamp gave %d", got)
	}
	if got := grpA(32); got != 0 {
		t.Fatalf("setup: window band must invalidate 32, clamp gave %d", got)
	}

	t.Run("naive-single-pass-min-is-not-jointly-valid", func(t *testing.T) {
		naive := naiveSinglePassMin(groups, l0)
		// Round 1: A clamps 40 -> 39 (its trimmed pool ends at row 38), B holds 40.
		// One pass of min stops at 39 — which B cannot restore (not block-aligned).
		if naive != 39 {
			t.Fatalf("naive single-pass min = %d, want 39 (A's clamp wins round 1)", naive)
		}
		if jointlyValid(groups, naive) {
			t.Fatal("fixture broken: naive answer unexpectedly jointly valid")
		}
		if _, m, _ := tree.MatchLongestPrefix(toks[:naive]); m != naive {
			t.Logf("characterized gap: naive answer %d is not jointly valid — the dense "+
				"group's real matcher restores only %d of it", naive, m)
		}
	})

	t.Run("production-matcher-reaches-the-fixed-point", func(t *testing.T) {
		turns := 0
		fp := radixkv.LongestJointPrefixHit(l0, len(groups), productionGroupHit(groups, &turns))
		// 40 --A--> 39 --B--> 36 --agree--> 36. A second narrowing turn is REQUIRED:
		// after A's window narrows 40 to 39, B's lattice rejects 39 and settles 36.
		if fp != 36 {
			t.Fatalf("production fixed point = %d, want 36", fp)
		}
		if turns < 2 {
			t.Fatalf("production narrowing settled in %d turn(s); the fixture needs >= 2 to distinguish the policies", turns)
		}
		if !jointlyValid(groups, fp) {
			t.Fatalf("fixed point %d is not jointly valid", fp)
		}
		// Joint validity at the fixed point, checked through the REAL matcher.
		if _, m, _ := tree.MatchLongestPrefix(toks[:fp]); m != fp {
			t.Fatalf("fixed point %d not served by dense group: matcher restores %d", fp, m)
		}
		// Maximality: nothing above the fixed point is jointly valid.
		for L := fp + 1; L <= l0; L++ {
			if jointlyValid(groups, L) {
				t.Fatalf("maximality broken: %d > fixed point %d is jointly valid", L, fp)
			}
		}
	})

	t.Run("reference-oracle-agrees-with-production", func(t *testing.T) {
		// TEST-ONLY REFERENCE cross-check: the local simultaneous-min iteration must
		// settle on the same jointly valid length as the production round-robin
		// matcher; divergence means either the port or the fixture drifted.
		fpRef, rounds := fixedPointNarrow(groups, l0)
		turns := 0
		fpProd := radixkv.LongestJointPrefixHit(l0, len(groups), productionGroupHit(groups, &turns))
		if fpRef != fpProd {
			t.Fatalf("reference %d diverged from production %d", fpRef, fpProd)
		}
		if rounds < 2 || turns < 2 {
			t.Fatalf("settled too fast: ref=%d round(s), prod=%d turn(s); the fixture needs >= 2 each to distinguish the policies", rounds, turns)
		}
	})

	t.Run("downward-closed-groups-agree-in-one-round", func(t *testing.T) {
		// Control: when every valid set is downward-closed, the naive min IS the fixed
		// point and the policies must coincide (the fixture is not vacuously red).
		capAt := func(n int) groupClamp {
			return func(c int) int {
				if c < n {
					return c
				}
				return n
			}
		}
		gs := []groupClamp{capAt(30), capAt(24)}
		naive := naiveSinglePassMin(gs, l0)
		fp, rounds := fixedPointNarrow(gs, l0)
		turns := 0
		fpProd := radixkv.LongestJointPrefixHit(l0, len(gs), productionGroupHit(gs, &turns))
		if naive != 24 || fp != 24 || fpProd != 24 {
			t.Fatalf("control: naive=%d ref=%d prod=%d, want 24/24/24", naive, fp, fpProd)
		}
		if rounds != 1 {
			t.Fatalf("control: downward-closed groups settled in %d reference rounds, want 1", rounds)
		}
	})

	t.Run("total-disagreement-settles-at-zero", func(t *testing.T) {
		// A second REAL align tree with block size 16: lattice {0, 16, 32}. Against the
		// same window band [33, 39] no positive length is jointly valid, so the
		// contract degrades to a full miss (0) rather than serving an invalid length.
		tree16 := model.NewAlignRadixTree(16)
		slabs16 := make([]*model.RecurrentStateSlab, 40/16)
		for b := range slabs16 {
			slabs16[b] = model.NewRecurrentStateSlab(b, 16, toks[b*16:(b+1)*16], nil)
		}
		if err := tree16.InsertSequence(toks, slabs16); err != nil {
			t.Fatalf("InsertSequence(16): %v", err)
		}
		gs := []groupClamp{grpA, alignBlockGroup(tree16, toks)}
		assertClampWellFormed(t, "align-block-16", gs[1], l0)

		naive := naiveSinglePassMin(gs, l0)
		if naive != 32 {
			t.Fatalf("naive single-pass min = %d, want 32 (B's clamp wins round 1)", naive)
		}
		if jointlyValid(gs, naive) {
			t.Fatal("fixture broken: naive answer unexpectedly jointly valid")
		}
		turns := 0
		fpProd := radixkv.LongestJointPrefixHit(l0, len(gs), productionGroupHit(gs, &turns))
		fp, rounds := fixedPointNarrow(gs, l0)
		if fpProd != 0 || fp != 0 {
			t.Fatalf("fixed points: prod=%d ref=%d, want 0/0 (no positive jointly valid length)", fpProd, fp)
		}
		if turns < 2 || rounds < 2 {
			t.Fatalf("settled too fast: prod=%d turn(s), ref=%d round(s), want >= 2 each", turns, rounds)
		}
	})
}
