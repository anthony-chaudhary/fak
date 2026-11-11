package radixkv

// prefix_hit.go — which prefix of a hash chain a cache tree can reuse.
//
// Ported from modular/modular rust_kv/kv-core/src/prefix_hit.rs at revision
// 8d559749621418f39f31df7da998bcce884fa5b1 (Apache-2.0 WITH LLVM-exception;
// vendored near-verbatim under third_party/modular/rust_kv with per-file byte
// receipts in PINNED-SOURCE.json; see THIRD_PARTY_NOTICES.md "Modular
// rust_kv/kv-core vendored tree and joint-prefix matcher port").
//
// The rules here are about attention shape alone. They take residency as a
// predicate, so the same code answers for any tier: the device pools, or a KV
// connector reporting what an external store holds. A connector therefore
// needs no notion of full attention, sliding window or SSM — it reports
// presence, and this decides what that presence is worth.
//
// Two shapes, and they behave differently under narrowing:
//
//   - a full-attention group reads its whole history, so its hit is the run
//     from the root and shortening an answer can never invalidate it;
//   - a sliding-window group reads only the last blocksInWindow blocks before
//     wherever the sequence stops, so its hit is a SUFFIX run whose validity
//     moves with the stopping point. Shortening can invalidate it.
//
// That second property is why LongestJointPrefixHit iterates instead of taking
// a minimum: narrowing for one group can invalidate another's window, which
// narrows it again.
//
// Changed for Fak: Go closure signatures replace Rust `impl Fn`/`impl FnMut`
// parameters; LeafShape becomes a fak-native Go type; debug_assert! becomes
// a panic in LongestSlidingWindowHit (the zero-window case has no answer to
// return, only a caller to fix). The fixed-point iteration, narrowing rules
// and per-shape hit rules are ported verbatim.

import "fmt"

// LongestFullAttentionHit returns the run of cached blocks from the root,
// which is what a full group reads.
//
// numHashes is how many blocks the request wants and isCached whether the
// block at an index is held. The result is downward-closed: a group serving n
// also serves anything shorter.
func LongestFullAttentionHit(numHashes int, isCached func(int) bool) int {
	run := 0
	for idx := 0; idx < numHashes; idx++ {
		if !isCached(idx) {
			break
		}
		run++
	}
	return run
}

// LongestSlidingWindowHit returns the deepest stopping point whose window this
// group holds complete.
//
// Prefix-cache hits for a sliding window differ greatly from full attention.
// The window size includes the query token: with the query token at idx=42
// and a window size of 10, the query attends to tokens idx=32 through idx=41.
// Per token, for example:
//
//	[X]: Token is in Prefix Cache
//	 . : Token is not in Prefix Cache
//	 ^ : Eligible Prefix Cache hit
//
//	  Tokens [A]  [B]   .   [D]  [E]  [F]   .    .   [I]  [J]  [K]  [L]  [M]
//	w_size=2  ^    ^         ^    ^    ^              ^    ^    ^    ^    ^
//	w_size=3  ^    ^              ^    ^                   ^    ^    ^    ^
//	w_size=4  ^    ^                   ^                        ^    ^    ^
//	w_size=5  ^    ^                                                 ^    ^
//	w_size=6  ^    ^                                                      ^
//	w_size=7  ^    ^
//
// As the window grows, fewer stopping points are eligible. Eligibility is a
// consecutive run of window_size - 1 tokens: [DEF] is a run of 3 tokens, so F
// is a valid hit for w_size=4 and below. A partial window is also a hit when
// the run starts at the root: [A] and [AB] are hits for any window size. Here
// the same rule is applied in whole blocks, with blocksInWindow the window
// less the query token, rounded up to blocks.
//
// Walks backwards counting a consecutive run, because the window sits at the
// END of the candidate: the first place the run reaches blocksInWindow is the
// deepest stopping point that works. A shallower one may not — its window
// covers different blocks — so this cannot be found by shortening a
// full-attention answer.
//
// A run that survives to index 0 is a hit of just that run, with no window
// check: nothing sits below the root to be missing.
//
// blocksInWindow is at least 1 for every group that gets here. Zero is
// window_size == 1, a query token attending to no history, which would make
// every page a hit that attention never reads back; KVCacheConfig::validate
// rejects that window and so does MAX's KVCacheGroupId (SERVOPT-1627).
// Panicked rather than reported: this sits on the per-request load path and a
// zero has no answer to return, only a caller to fix.
func LongestSlidingWindowHit(numHashes, blocksInWindow int, isCached func(int) bool) int {
	if blocksInWindow <= 0 {
		panic("radixkv: a sliding-window group spans at least one block")
	}
	run := 0
	for idx := numHashes - 1; idx >= 0; idx-- {
		if !isCached(idx) {
			run = 0
			continue
		}
		run++
		if run >= blocksInWindow {
			return idx + run
		}
	}
	return run
}

// LongestJointPrefixHit returns the longest prefix EVERY group can serve at
// once.
//
// Each group answers under the run the others have already allowed, so the
// run is settled once every group has accepted it in turn. A group asked
// again about a prefix it just returned has to return that same length, or
// the loop would walk the run to nothing.
//
// Iterating is required rather than tidy. A minimum over the groups' answers
// is wrong twice over: a windowed group's answer is not a depth that can be
// compared with a full group's, and narrowing for one windowed group can
// invalidate another's window, which narrows it again.
//
// groupHit answers for group k how much of a candidate length it can serve.
// Indexed rather than taking one closure per group, because every caller
// holds its groups in a slice already and building a closure each would
// allocate per request on the load path. Returns the agreed prefix length, or
// 0 when the groups cannot agree on any.
//
// An answer is clamped to the candidate rather than asserted equal to it:
// this is exported, so the closure is not always one of this package's own
// rules, and a larger answer would raise candidate while resetting accepted
// and need not terminate. A debug assertion alone compiles out of the build
// that matters.
//
// Each non-accepting turn drops the candidate by at least one and each turn
// costs O(candidate), so the bound is O(numHashes^2) predicate evaluations on
// the per-request load path. Real residency settles in a couple of turns; the
// ceiling is only reachable with several windowed groups disagreeing.
func LongestJointPrefixHit(numHashes, numGroups int, groupHit func(group, candidate int) int) int {
	if numGroups == 0 {
		return 0
	}
	candidate := numHashes
	accepted := 0
	turn := 0
	for candidate != 0 && accepted < numGroups {
		numHitBlocks := groupHit(turn, candidate)
		if numHitBlocks > candidate {
			numHitBlocks = candidate
		}
		if numHitBlocks == candidate {
			accepted++
		} else {
			accepted = 1
		}
		candidate = numHitBlocks
		turn = (turn + 1) % numGroups
	}
	return candidate
}

// LongestGroupHit returns how much of candidate a group of this shape serves:
// the rule above that its blocksInWindow selects. LeafShape.LongestHit also
// covers recurrent and scratch leaves.
func LongestGroupHit(candidate int, blocksInWindow int, isCached func(int) bool) int {
	if blocksInWindow < 0 {
		return LongestFullAttentionHit(candidate, isCached)
	}
	return LongestSlidingWindowHit(candidate, blocksInWindow, isCached)
}

// LongestRecurrentHit returns the deepest boundary a recurrent leaf holds a
// state for.
//
// A recurrent state is published at one boundary rather than as a run of
// blocks, so any resident boundary is a complete hit on its own and the
// deepest one wins. A shallower candidate is answered from below it, which
// keeps the joint rule's narrowing sound.
func LongestRecurrentHit(numHashes int, isCached func(int) bool) int {
	for idx := numHashes - 1; idx >= 0; idx-- {
		if isCached(idx) {
			return idx + 1
		}
	}
	return 0
}

// LeafShape is a leaf's attention shape, which selects the rule that reads
// its residency.
type LeafShape struct {
	// Kind selects the rule: LeafFull reads its whole history
	// (LongestFullAttentionHit); LeafSlidingWindow reads only the last
	// blocksInWindow blocks before the stopping point
	// (LongestSlidingWindowHit, window at least one block); LeafRecurrent
	// holds one state per published boundary (LongestRecurrentHit);
	// LeafScratch is never published, so it has no opinion on a hit and must
	// not shorten what the leaves that do cache agree on.
	Kind LeafKind
	// BlocksInWindow is the window in whole blocks, excluding the query
	// token. Meaningful only for LeafSlidingWindow.
	BlocksInWindow int
}

// LeafKind enumerates the four upstream leaf shapes.
type LeafKind int

const (
	LeafFull LeafKind = iota
	LeafSlidingWindow
	LeafRecurrent
	LeafScratch
)

// LongestHit returns how much of candidate a leaf of this shape serves.
func (s LeafShape) LongestHit(candidate int, isCached func(int) bool) int {
	switch s.Kind {
	case LeafFull:
		// Reads its whole history, so the hit is the run from the root.
		return LongestFullAttentionHit(candidate, isCached)
	case LeafSlidingWindow:
		// The hit is a SUFFIX run, so it cannot be found by shortening a
		// full-attention answer: a shallower stopping point's window covers
		// different blocks.
		return LongestSlidingWindowHit(candidate, s.BlocksInWindow, isCached)
	case LeafRecurrent:
		// A state is a single published boundary rather than a run, so the
		// deepest one that stands is the answer.
		return LongestRecurrentHit(candidate, isCached)
	default:
		// Never published, so it has no opinion and must not shorten what the
		// leaves that do cache agree on.
		return candidate
	}
}

// BlocksHeldOfHit returns how many blocks a group actually holds of a hit
// that deep.
//
// The companion to the rules above, and not the same number: a windowed
// group's hit can cover a whole prefix while the group holds only the window
// at the end of it, because the slots below the window are never read. A full
// group holds all of it.
//
// Every group's blocks END at numHitBlocks, so a group holding n of them
// covers hashes[numHitBlocks-n : numHitBlocks] and the numHitBlocks-n slots
// below are the null block. blocksInWindow is the group's window in whole
// blocks, or -1 for a group that attends over its whole history.
func BlocksHeldOfHit(numHitBlocks, blocksInWindow int) int {
	if blocksInWindow < 0 {
		return numHitBlocks
	}
	return min(numHitBlocks, blocksInWindow)
}

// GroupResidency is one group's residency for a request, in ABSOLUTE page
// slots.
//
// The adaptation between what a store answers (positions in what this group
// asked for) and what the rules above take (absolute slots), so the two
// engines' rules can be applied to a round without restating the slot-offset
// convention at each call site.
type GroupResidency struct {
	// SlotOffset is the absolute slot resident[0] describes.
	SlotOffset int
	// LocalPrefixPages is the root-anchored local prefix already resident
	// below SlotOffset. Equal to SlotOffset for an ordinary claim, and 0 for
	// a null-headed window whose lower slots are genuinely absent. Values
	// strictly between the two are not a shape any producer emits and are
	// rejected in debug.
	LocalPrefixPages int
	// BlocksInWindow is this group's window in whole blocks, or -1 for full
	// attention.
	BlocksInWindow int
	// Probed is how many blocks this group was asked about, which bounds the
	// slots it can speak for. Carried separately from len(Resident) so a
	// group that answered nothing at all still constrains the joint over its
	// whole ask rather than dropping out of it.
	Probed int
	// LoadableBlocks is the blocks from SlotOffset that the TRANSFER can
	// move: the store's leading run. Residency past a gap is visible to the
	// rules but unreachable by the load path, whose device ids pair
	// positionally with a leading run, so this is what bounds a credit that
	// a request will be staked on.
	LoadableBlocks int
	// Resident is what the store answered, positionally over what this group
	// asked for. May be shorter than Probed; the missing tail reads absent.
	Resident []bool
}

// IsCached reports residency at an ABSOLUTE slot. Below SlotOffset the answer
// is the local prefix rather than the store's: those slots were never asked
// about, and a root-anchored claim already holds them.
func (g *GroupResidency) IsCached(slot int) bool {
	if slot < g.SlotOffset {
		// The branch is load-bearing on LocalPrefixPages being one of the two
		// shapes: a value in between would report every below-offset slot
		// absent rather than only those past the prefix. The guard lives
		// here now.
		if g.LocalPrefixPages != 0 && g.LocalPrefixPages != g.SlotOffset {
			panic(fmt.Sprintf("radixkv: local_prefix_pages %d is neither 0 nor slot_offset %d", g.LocalPrefixPages, g.SlotOffset))
		}
		return g.LocalPrefixPages == g.SlotOffset
	}
	if slot-g.SlotOffset >= len(g.Resident) {
		return false
	}
	return g.Resident[slot-g.SlotOffset]
}

// Extent is one past the deepest slot this group can speak about.
func (g *GroupResidency) Extent() int {
	return g.SlotOffset + g.Probed
}

// LoadableDepth is the deepest absolute prefix the transfer path can actually
// fill for this group.
func (g *GroupResidency) LoadableDepth() int {
	return g.SlotOffset + g.LoadableBlocks
}

// LongestJointResidencyHit returns the joint prefix these groups serve for
// one request, in ABSOLUTE page slots.
//
// The whole reconciliation, in one place so a connector and a test fixture
// cannot drift: each group answers under the run the others have allowed,
// through the rule its own shape selects.
//
// The candidate starts at the deepest slot any group was asked about, capped
// by the shallowest depth any of them can actually FILL (LoadableDepth).
// Without that cap the rules will happily credit a complete window sitting
// above a gap, which is a true statement about residency and an unfillable
// one about transfer: the caller stakes the request on the credit, the load
// path moves the leading run only, and the round fails and rolls back every
// member. Capping the candidate rather than the answer keeps each group's
// rule evaluated only at depths it could be asked to fill.
//
// Ragged extents are admitted on purpose, where MAX's connector rejects them:
// groups carry per-group SlotOffsets and ask different ranges, so there is no
// single width to check against. A slot above a group's own extent reads
// absent, which narrows the joint rather than widening it.
func LongestJointResidencyHit(groups []*GroupResidency) int {
	deepestAsked := 0
	for _, g := range groups {
		if e := g.Extent(); e > deepestAsked {
			deepestAsked = e
		}
	}
	fillable := 0
	if len(groups) > 0 {
		fillable = groups[0].LoadableDepth()
		for _, g := range groups[1:] {
			if d := g.LoadableDepth(); d < fillable {
				fillable = d
			}
		}
	}
	return LongestJointPrefixHit(
		min(deepestAsked, fillable),
		len(groups),
		func(gi, candidate int) int {
			g := groups[gi]
			return LongestGroupHit(candidate, g.BlocksInWindow, g.IsCached)
		},
	)
}

// BlocksInWindowOf returns a group's window in whole blocks, or -1 for a
// group attending over its whole history.
//
// ceildiv(window-1, pageSize), which is the arithmetic MAX's
// KVCacheGroupId.blocks_in_window runs for a windowed leaf (MAX returns a -1
// sentinel for a full one where this returns -1). The formula is kept
// identical rather than simplified because these rules are shared with MAX,
// and an off-by-one here moves the depth the two engines agree on.
func BlocksInWindowOf(window int, pageSize int) int {
	if window < 0 {
		return -1
	}
	return (window - 1 + pageSize - 1) / pageSize
}
