package ctxresidency_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/ctxresidency"
)

// pressure_test.go — witnesses for the #4037 reclaimable carve. The headline
// property is the one the issue is about: a near-full window whose residency is
// mostly RECLAIMABLE must not read as pressure, while the same total that is
// mostly PINNED must. A single lumped number cannot tell those apart, so the
// first test drives both from the SAME resident total.

// gateway's live pressure scale (internal/gateway ctxStepBoundedPercent /
// ctxStepCheckpointPercent). These are the caller's constants, passed in — this
// package declares no thresholds of its own, and these locals exist only so the
// tests grade against the real scale rather than an invented one.
const (
	boundedPct    = 50
	checkpointPct = 80
)

// TestPressureRedOnlyOnTrueNonReclaimablePressure is the acceptance witness: at
// an IDENTICAL 90%-full window, the verdict must be driven by the pinned band
// alone. Mostly-reclaimable stays "any"; mostly-pinned goes "checkpoint". Before
// the carve both cases were one 9000-token resident figure and graded the same.
func TestPressureRedOnlyOnTrueNonReclaimablePressure(t *testing.T) {
	const budget = 10000

	// 90% full, but only 500 tokens are pinned — a shed buys back 8500.
	slack := ctxresidency.PressureBands(500, 8500, budget, boundedPct, checkpointPct)
	// 90% full, and 8500 tokens are pinned by live dependents — a shed buys
	// back 500 and the window stays crowded.
	stuck := ctxresidency.PressureBands(8500, 500, budget, boundedPct, checkpointPct)

	if slack.LumpedPct != stuck.LumpedPct {
		t.Fatalf("fixture broken: the two cases must present the SAME lumped fullness, got %.1f%% vs %.1f%%",
			slack.LumpedPct, stuck.LumpedPct)
	}
	if slack.Class != ctxresidency.PressureAny {
		t.Errorf("mostly-reclaimable 90%% window graded %q, want %q — a reclaimable band must not read as pressure (true=%.1f%% lumped=%.1f%%)",
			slack.Class, ctxresidency.PressureAny, slack.TruePressurePct, slack.LumpedPct)
	}
	if stuck.Class != ctxresidency.PressureCheckpoint {
		t.Errorf("mostly-pinned 90%% window graded %q, want %q — pinned tokens ARE pressure (true=%.1f%%)",
			stuck.Class, ctxresidency.PressureCheckpoint, stuck.TruePressurePct)
	}
	if !slack.Reclaimable() {
		t.Error("the mostly-reclaimable window must report Reclaimable() so a full-looking bar can justify staying green")
	}
	// The gap between the two percents is exactly what a one-band meter hid.
	if slack.TruePressurePct >= slack.LumpedPct {
		t.Errorf("TruePressurePct=%.1f must sit BELOW LumpedPct=%.1f when slack exists", slack.TruePressurePct, slack.LumpedPct)
	}
}

// TestPressureRungsMatchTheInjectedScale pins each rung boundary against the
// caller's 50/80 scale, including the exact-boundary cases (a rung fires at >=,
// matching gateway's adviseCtxStep so the two surfaces cannot disagree).
func TestPressureRungsMatchTheInjectedScale(t *testing.T) {
	const budget = 1000
	for _, tc := range []struct {
		name      string
		committed int
		want      ctxresidency.PressureClass
	}{
		{"empty", 0, ctxresidency.PressureAny},
		{"below bounded", 499, ctxresidency.PressureAny},
		{"exactly bounded", 500, ctxresidency.PressureBounded},
		{"below checkpoint", 799, ctxresidency.PressureBounded},
		{"exactly checkpoint", 800, ctxresidency.PressureCheckpoint},
		{"over budget", 1200, ctxresidency.PressureCheckpoint},
	} {
		got := ctxresidency.PressureBands(tc.committed, 0, budget, boundedPct, checkpointPct)
		if got.Class != tc.want {
			t.Errorf("%s: committed=%d of %d graded %q, want %q", tc.name, tc.committed, budget, got.Class, tc.want)
		}
	}
}

// TestPressureBandsPartitionTheBudget proves the three bands are a partition, so
// a renderer can lay them end to end without a gap or an overdraw — and that an
// overrun clamps free to 0 instead of reporting negative space.
func TestPressureBandsPartitionTheBudget(t *testing.T) {
	b := ctxresidency.PressureBands(300, 200, 1000, boundedPct, checkpointPct)
	if sum := b.CommittedTokens + b.ReclaimableTokens + b.FreeTokens; sum != b.BudgetTokens {
		t.Errorf("bands sum to %d, want the budget %d (committed=%d reclaimable=%d free=%d)",
			sum, b.BudgetTokens, b.CommittedTokens, b.ReclaimableTokens, b.FreeTokens)
	}
	over := ctxresidency.PressureBands(900, 400, 1000, boundedPct, checkpointPct)
	if over.FreeTokens != 0 {
		t.Errorf("an over-budget window reported FreeTokens=%d, want 0 (never negative free space)", over.FreeTokens)
	}
	neg := ctxresidency.PressureBands(-5, -7, 1000, boundedPct, checkpointPct)
	if neg.CommittedTokens != 0 || neg.ReclaimableTokens != 0 || neg.FreeTokens != 1000 {
		t.Errorf("negative inputs must clamp to zero, got %+v", neg)
	}
}

// TestPressureFailsClosedWithoutBudgetOrScale proves the fold never fabricates a
// tier. Grading against a zero budget or a zero rung would read every window as
// checkpoint; it says unknown instead — while still reporting the bands, because
// the split is a measurement that stays true even when the verdict cannot be.
func TestPressureFailsClosedWithoutBudgetOrScale(t *testing.T) {
	noBudget := ctxresidency.PressureBands(500, 200, 0, boundedPct, checkpointPct)
	if noBudget.Class != ctxresidency.PressureUnknown {
		t.Errorf("no budget graded %q, want %q", noBudget.Class, ctxresidency.PressureUnknown)
	}
	if noBudget.TruePressurePct != 0 || noBudget.LumpedPct != 0 {
		t.Errorf("no budget must not fabricate percentages, got true=%.1f lumped=%.1f", noBudget.TruePressurePct, noBudget.LumpedPct)
	}
	if noBudget.CommittedTokens != 500 || noBudget.ReclaimableTokens != 200 {
		t.Errorf("the measured bands must survive an ungradeable verdict, got %+v", noBudget)
	}

	noScale := ctxresidency.PressureBands(950, 0, 1000, 0, 0)
	if noScale.Class != ctxresidency.PressureUnknown {
		t.Errorf("no threshold scale graded %q, want %q — a zero rung would read every window as checkpoint", noScale.Class, ctxresidency.PressureUnknown)
	}
	if noScale.TruePressurePct != 95 {
		t.Errorf("an ungradeable verdict must still measure: TruePressurePct=%.1f, want 95", noScale.TruePressurePct)
	}
}

// TestQueryCarvesResidentIntoCommittedAndReclaimable is the end-to-end witness
// over a real kvmmu context: the carve is driven by the SAME dependent count
// that decides each span's State, and the two bands partition ResidentTokens
// exactly — so the totals can never disagree with the per-span rows or with the
// kernel's own CacheLen.
func TestQueryCarvesResidentIntoCommittedAndReclaimable(t *testing.T) {
	c, _ := newCtx(t)
	c.Append("a", "t1", []int{1, 2, 3}) // no dependents -> reclaimable
	c.Append("b", "t2", []int{4, 5})    // given a dependent below -> committed

	var bKV cachemeta.EntryID
	for _, seg := range c.Segments() {
		if seg.ID == "b" {
			bKV = seg.KV
		}
	}
	if !bKV.Valid() {
		t.Fatal("segment b did not expose a cachemeta KV identity")
	}
	c.TrackEntry(cachemeta.FromAttentionIndex(cachemeta.AttentionIndex{
		Tokens: []int{4, 5}, ModelID: "llama", TokenizerID: "tok", IndexerID: "idx:v1",
		LayerGroup: "0-1", Layers: []int{0, 1}, DecisionDigest: cachemeta.DigestBytes([]byte("b-topk")),
		ParentKV: bKV, Owner: "test", Causal: true, CausalityWitness: "unit:carve",
	}))

	snap := ctxresidency.Query(c, nil)
	if snap.CommittedTokens+snap.ReclaimableTokens != snap.ResidentTokens {
		t.Errorf("committed(%d)+reclaimable(%d) != ResidentTokens(%d): the carve must partition the resident total",
			snap.CommittedTokens, snap.ReclaimableTokens, snap.ResidentTokens)
	}
	if snap.ResidentTokens != c.CacheLen() {
		t.Errorf("ResidentTokens=%d != kvmmu.CacheLen=%d (the carve must not disturb the reconciliation)", snap.ResidentTokens, c.CacheLen())
	}
	if snap.CommittedTokens != 2 {
		t.Errorf("CommittedTokens=%d, want 2 (span b is pinned by a live attention_index dependent)", snap.CommittedTokens)
	}
	if snap.ReclaimableTokens != 3 {
		t.Errorf("ReclaimableTokens=%d, want 3 (span a has no dependents — a clean eviction candidate)", snap.ReclaimableTokens)
	}

	// The bands a meter would render off this live snapshot: at a budget of 10
	// the window is half full, but only 2 tokens are truly pinned.
	b := snap.Pressure(10, boundedPct, checkpointPct)
	if b.Class != ctxresidency.PressureAny {
		t.Errorf("live snapshot graded %q at 20%% true pressure, want %q", b.Class, ctxresidency.PressureAny)
	}
	if b.CommittedTokens != 2 || b.ReclaimableTokens != 3 || b.FreeTokens != 5 {
		t.Errorf("bands = %d/%d/%d, want 2/3/5 (committed/reclaimable/free)", b.CommittedTokens, b.ReclaimableTokens, b.FreeTokens)
	}
}
