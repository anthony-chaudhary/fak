package abi

import (
	"sort"
	"testing"
)

// namedBlock is a reserved Range paired with its identifier, so a partition
// failure names the two colliding blocks instead of just their integers.
type namedBlock struct {
	name string
	r    Range
}

// reservedFamilies groups the reserved-range blocks by the NUMBER SPACE they draw
// from. The disjointness contract (registry.go, "RESERVED RANGES") is a
// within-space promise: two blocks in the SAME space (two OpCode blocks, two
// EventKind blocks) must not overlap, or two subsystems could draw the same
// integer. Blocks in DIFFERENT spaces (an OpCode block vs an ExtKey block) share
// integers freely — the *Vendor blocks all reuse [1<<16,1<<17) across spaces by
// design — so the partition is checked per family, never across families.
func reservedFamilies() map[string][]namedBlock {
	return map[string][]namedBlock{
		"OpCode": {
			{"OpsCore", OpsCore}, {"OpsSpec", OpsSpec},
			{"OpsAsync", OpsAsync}, {"OpsVendor", OpsVendor},
		},
		"ExtKey": {
			{"ExtSpec", ExtSpec}, {"ExtAsync", ExtAsync}, {"ExtLabel", ExtLabel},
			{"ExtTrust", ExtTrust}, {"ExtVendor", ExtVendor},
		},
		"VerdictKind": {
			{"VerdictsVendor", VerdictsVendor},
		},
		"EventKind": {
			{"EventsCore", EventsCore}, {"EventsKPI", EventsKPI},
			{"EventsLabel", EventsLabel}, {"EventsVendor", EventsVendor},
		},
	}
}

// TestReservedRangesPartitionDisjoint verifies the promise registry.go makes in
// its "RESERVED RANGES" header — that the per-subsystem blocks form a disjoint
// partition of each number space — which Register* documents but does NOT
// enforce: RegisterOp (registry.go RegisterOp) and RegisterVerdictKind panic only
// on an exact-integer duplicate, never on a block that straddles two ranges.
// Without this test an additive block like Range{80,112} over OpsSpec{64,96} +
// OpsAsync{96,128} would pass every existing check and stay silent until two
// subsystems happened to draw the same integer — the exact cross-subsystem
// collision the contract claims to prevent, on an additive-only ABI authored by
// many leaves. See issue #4270.
func TestReservedRangesPartitionDisjoint(t *testing.T) {
	for family, blocks := range reservedFamilies() {
		// Every block must be a well-formed half-open [Lo,Hi) with Lo < Hi; an
		// empty or inverted block is a declaration bug on its own.
		for _, b := range blocks {
			if b.r.Lo >= b.r.Hi {
				t.Errorf("%s: block %s is empty or inverted: [%d,%d)",
					family, b.name, b.r.Lo, b.r.Hi)
			}
		}
		// Sort by Lo (tie-break Hi); then any block whose Lo lands before the
		// previous block's Hi overlaps it. The blocks are half-open, so an exact
		// Lo == prevHi is legal abutment (OpsCore[0,64) then OpsSpec[64,96)), NOT
		// an overlap.
		sorted := append([]namedBlock(nil), blocks...)
		sort.Slice(sorted, func(i, j int) bool {
			if sorted[i].r.Lo != sorted[j].r.Lo {
				return sorted[i].r.Lo < sorted[j].r.Lo
			}
			return sorted[i].r.Hi < sorted[j].r.Hi
		})
		for i := 1; i < len(sorted); i++ {
			prev, cur := sorted[i-1], sorted[i]
			if cur.r.Lo < prev.r.Hi {
				t.Errorf("%s: blocks %s [%d,%d) and %s [%d,%d) overlap — "+
					"the reserved-range partition is not disjoint",
					family, prev.name, prev.r.Lo, prev.r.Hi,
					cur.name, cur.r.Lo, cur.r.Hi)
			}
		}
	}
}

// TestInRangeMembershipHalfOpen pins the half-open [Lo,Hi) semantics that the
// partition check and every "draw a number from this block" claim rely on: Lo is
// a member, Hi is not. It also keeps inRange — the membership predicate the
// "RESERVED RANGES" contract is written in terms of — exercised in production
// code rather than dead outside pkg/abi's own private copy.
func TestInRangeMembershipHalfOpen(t *testing.T) {
	r := Range{64, 96} // OpsSpec-shaped
	cases := []struct {
		n    uint32
		want bool
	}{
		{63, false}, // just below Lo
		{64, true},  // Lo is inclusive
		{80, true},  // interior
		{95, true},  // last member
		{96, false}, // Hi is exclusive
		{97, false}, // just above Hi
	}
	for _, c := range cases {
		if got := inRange(c.n, r); got != c.want {
			t.Errorf("inRange(%d, [%d,%d)) = %v, want %v",
				c.n, r.Lo, r.Hi, got, c.want)
		}
	}
}
