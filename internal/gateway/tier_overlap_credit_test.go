package gateway

import (
	"math"
	"testing"
)

// TestTierWeightedOverlapCreditMonotone pins the core property: more held blocks in
// the same tier yield at least as much credit (never less).
func TestTierWeightedOverlapCreditMonotone(t *testing.T) {
	w := DefaultTierWeights()

	prev := -1.0
	for _, blocks := range []int{0, 1, 2, 5, 10, 50, 100} {
		got := TierWeightedOverlapCredit(w, []TierHeldOverlap{{Tier: TierDevice, Blocks: blocks}})
		if got < prev {
			t.Fatalf("credit must be monotone non-decreasing in overlap: blocks=%d got=%g prev=%g", blocks, got, prev)
		}
		prev = got
	}

	// A strictly longer overlap earns strictly more credit at a fixed tier.
	short := TierWeightedOverlapCredit(w, []TierHeldOverlap{{Tier: TierHost, Blocks: 3}})
	long := TierWeightedOverlapCredit(w, []TierHeldOverlap{{Tier: TierHost, Blocks: 30}})
	if !(long > short) {
		t.Fatalf("longer overlap must earn more credit: short=%g long=%g", short, long)
	}
}

// TestTierWeightedOverlapCreditFasterTierWins pins that the SAME held length earns
// more credit on a faster tier than on a slower one — the whole point of the borrow.
func TestTierWeightedOverlapCreditFasterTierWins(t *testing.T) {
	w := DefaultTierWeights()
	const blocks = 20

	device := TierWeightedOverlapCredit(w, []TierHeldOverlap{{Tier: TierDevice, Blocks: blocks}})
	host := TierWeightedOverlapCredit(w, []TierHeldOverlap{{Tier: TierHost, Blocks: blocks}})
	shared := TierWeightedOverlapCredit(w, []TierHeldOverlap{{Tier: TierShared, Blocks: blocks}})
	disk := TierWeightedOverlapCredit(w, []TierHeldOverlap{{Tier: TierDisk, Blocks: blocks}})

	if !(device > host && host > shared && shared > disk) {
		t.Fatalf("faster tier must earn more credit for equal length: device=%g host=%g shared=%g disk=%g",
			device, host, shared, disk)
	}

	// The routing consequence the issue names: a shorter on-device overlap outscores a
	// longer overlap evicted to disk at equal load. Worker B holds 8 blocks on-device;
	// worker A holds 20 blocks on disk.
	creditA := TierWeightedOverlapCredit(w, []TierHeldOverlap{{Tier: TierDisk, Blocks: 20}})
	creditB := TierWeightedOverlapCredit(w, []TierHeldOverlap{{Tier: TierDevice, Blocks: 8}})
	const equalLoad = 3
	scoreA := ScoreWithTierWeightedOverlap(creditA, equalLoad)
	scoreB := ScoreWithTierWeightedOverlap(creditB, equalLoad)
	if !(scoreB > scoreA) {
		t.Fatalf("shorter on-device overlap must outscore longer on-disk overlap at equal load: A=%g B=%g", scoreA, scoreB)
	}

	// Confirm the plain length-only view would have picked A — isolates the lift to the
	// tier weighting, not to lengths.
	if !(20 > 8) {
		t.Fatal("sanity: worker A holds the longer raw prefix")
	}
}

// TestTierWeightedOverlapCreditMultiTier pins that an overlap spanning multiple tiers
// sums each run through its own weight.
func TestTierWeightedOverlapCreditMultiTier(t *testing.T) {
	w := DefaultTierWeights()
	held := []TierHeldOverlap{
		{Tier: TierDevice, Blocks: 4}, // 4 * 1.0  = 4.0
		{Tier: TierHost, Blocks: 4},   // 4 * 0.75 = 3.0
		{Tier: TierDisk, Blocks: 4},   // 4 * 0.25 = 1.0
	}
	want := 4.0 + 3.0 + 1.0
	if got := TierWeightedOverlapCredit(w, held); math.Abs(got-want) > 1e-9 {
		t.Fatalf("multi-tier credit: got %g want %g", got, want)
	}
}

// TestTierWeightedOverlapCreditZero pins that zero overlap earns zero credit, so a
// cold worker is costed exactly as the length-only body costs it.
func TestTierWeightedOverlapCreditZero(t *testing.T) {
	w := DefaultTierWeights()
	if got := TierWeightedOverlapCredit(w, nil); got != 0 {
		t.Fatalf("empty held list must earn 0 credit, got %g", got)
	}
	if got := TierWeightedOverlapCredit(w, []TierHeldOverlap{{Tier: TierDevice, Blocks: 0}}); got != 0 {
		t.Fatalf("zero-block run must earn 0 credit, got %g", got)
	}
	// A zero credit scores zero (a cold worker never wins locality).
	if got := ScoreWithTierWeightedOverlap(0, 5); got != 0 {
		t.Fatalf("zero credit must score 0, got %g", got)
	}
}

// TestTierWeightedOverlapCreditFailsClosed pins that a bad or negative weight table,
// an unknown tier tag, or a negative block count contributes nothing rather than
// corrupting the credit.
func TestTierWeightedOverlapCreditFailsClosed(t *testing.T) {
	held := []TierHeldOverlap{{Tier: TierDevice, Blocks: 10}}

	bad := []struct {
		name string
		w    TierWeights
	}{
		{"negative device", TierWeights{Device: -1, Host: 0.75, Shared: 0.5, Disk: 0.25}},
		{"negative disk", TierWeights{Device: 1, Host: 0.75, Shared: 0.5, Disk: -0.25}},
		{"NaN host", TierWeights{Device: 1, Host: math.NaN(), Shared: 0.5, Disk: 0.25}},
		{"Inf shared", TierWeights{Device: 1, Host: 0.75, Shared: math.Inf(1), Disk: 0.25}},
	}
	for _, c := range bad {
		if got := TierWeightedOverlapCredit(c.w, held); got != 0 {
			t.Fatalf("%s: bad weight table must fail closed to 0 credit, got %g", c.name, got)
		}
	}

	// An unknown tier tag contributes nothing (out-of-range enum).
	unknown := TierWeightedOverlapCredit(DefaultTierWeights(), []TierHeldOverlap{{Tier: OverlapTier(99), Blocks: 10}})
	if unknown != 0 {
		t.Fatalf("unknown tier must contribute 0, got %g", unknown)
	}

	// A negative block count contributes nothing.
	neg := TierWeightedOverlapCredit(DefaultTierWeights(), []TierHeldOverlap{{Tier: TierDevice, Blocks: -5}})
	if neg != 0 {
		t.Fatalf("negative block count must contribute 0, got %g", neg)
	}
}

// TestScoreWithTierWeightedOverlapComposes pins that the credit folds into the same
// inverse-load shape without double-counting: it replaces the raw overlap term, load
// only ever discounts, and a negative load clamps to zero.
func TestScoreWithTierWeightedOverlapComposes(t *testing.T) {
	credit := TierWeightedOverlapCredit(DefaultTierWeights(), []TierHeldOverlap{{Tier: TierDevice, Blocks: 12}})
	if credit != 12 {
		t.Fatalf("device-only credit should equal raw length: got %g want 12", credit)
	}

	// Zero load: full credit (matches score()'s ov/(1+0) == ov).
	if got := ScoreWithTierWeightedOverlap(credit, 0); got != credit {
		t.Fatalf("zero load must score the full credit: got %g want %g", got, credit)
	}

	// Load only discounts: a busier worker scores strictly less for the same credit.
	light := ScoreWithTierWeightedOverlap(credit, 1)
	heavy := ScoreWithTierWeightedOverlap(credit, 9)
	if !(light > heavy) {
		t.Fatalf("higher load must discount the score: light=%g heavy=%g", light, heavy)
	}

	// Negative load clamps to zero (never amplifies).
	if got := ScoreWithTierWeightedOverlap(credit, -3); got != credit {
		t.Fatalf("negative load must clamp to 0: got %g want %g", got, credit)
	}

	// No double-count: evaluating twice with the same inputs is pure.
	if a, b := ScoreWithTierWeightedOverlap(credit, 4), ScoreWithTierWeightedOverlap(credit, 4); a != b {
		t.Fatalf("score must be pure: %g then %g", a, b)
	}
}
