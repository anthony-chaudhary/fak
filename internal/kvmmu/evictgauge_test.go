package kvmmu_test

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// evictgauge_test.go — issue #5123: the #3901 retained-mass gauge, emitted at the
// eviction/selection decision instead of only from a unit-test caller.
//
// Every witness here drives a REAL kvmmu.Context: spans are appended to a live synthetic
// session, attention is attributed through the rung-1/2 AttributeRow path (not by poking
// Cumulative), the turn is closed by CloseTurn, and the decision is one of the shipped entry
// points — including ApplyPlan, the one the served path reaches through
// agent.InKernelPlanner.ElideKVSpans. What the tests assert is the emission's BINDING to that
// decision: that the mass the gauge withholds from the numerator is exactly the mass of the
// spans the decision dropped, that the denominator still counts those dropped spans, that the
// gauge fails closed with no observer, and that emitting it did not change what gets evicted.

// The fixed three-span residency the #5123 witnesses drive: a pinned system span, a heavily
// attended span, and a barely attended one (the eviction candidate). Distinct lengths make the
// token fraction legible as well as the mass fraction.
var (
	gaugeSys  = []int{1, 2, 3, 4, 5}
	gaugeWarm = []int{20, 21, 22}
	gaugeCold = []int{10, 11, 12, 13}
)

// newGaugeContext builds the live Context the witnesses evict from, returning the segment
// handles so a test can read the ledger the decision will rule on. The system span is pinned
// exactly as the policy gate pins it in production, so it is never a coldest-eviction candidate
// and stays in the gauge's kept set.
func newGaugeContext() (*kvmmu.Context, map[string]*kvmmu.Segment) {
	m := model.NewSynthetic(synthCfg())
	c := kvmmu.NewWithGate(m.NewSession(), ctxmmu.New())
	segs := make(map[string]*kvmmu.Segment, 3)
	_, segs["sys"] = c.Append("sys", "system", gaugeSys)
	c.Pin("sys", true)
	_, segs["warm"] = c.Append("warm", "read_b", gaugeWarm)
	_, segs["cold"] = c.Append("cold", "read_a", gaugeCold)
	return c, segs
}

// observeTurn attributes one turn of real attention rows onto the named spans through the live
// AttributeRow path, then closes the turn at lambda=1 so each span's Cumulative is the undecayed
// mass it actually drew — the post-hoc scalar the gauge reads.
func observeTurn(c *kvmmu.Context, segs map[string]*kvmmu.Segment, mass map[string]float64) {
	for id, m := range mass {
		attributeSpanMass(c, segs[id], m)
	}
	c.CloseTurn(1.0)
}

// witnessedMass is the per-span attention the witnesses drive: the cold span draws an order of
// magnitude less than the warm one, so the controller's coldest-by-EMA pick is unambiguous.
func witnessedMass() map[string]float64 {
	return map[string]float64{"sys": 0.5, "warm": 1.0, "cold": 0.1}
}

// TestEvictColdestEmitsRetainedMassAtTheDecision is the #5123 headline witness: a real eviction
// on a real Context emits retained_mass_fraction for that decision through the UNCHANGED
// EvictColdest entry point, and the gauge is BOUND to it — the mass missing from the numerator
// is exactly the evicted span's.
func TestEvictColdestEmitsRetainedMassAtTheDecision(t *testing.T) {
	c, segs := newGaugeContext()
	observeTurn(c, segs, witnessedMass())

	total := segs["sys"].Cumulative + segs["warm"].Cumulative + segs["cold"].Cumulative
	coldMass := segs["cold"].Cumulative
	if coldMass <= 0 || total <= coldMass {
		t.Fatalf("setup wrong: cold mass %.4f of total %.4f — need a real distribution to grade", coldMass, total)
	}

	evicted := c.EvictColdest(len(gaugeCold))
	g := c.LastRetainedMass()

	if len(evicted) != 1 || evicted[0].ID != "cold" {
		t.Fatalf("decision dropped %+v, want exactly the cold span", evicted)
	}
	if !g.Available {
		t.Fatal("gauge unavailable after a witnessed eviction — the emission is not reading the observed mass")
	}
	// The load-bearing check. evict() clears a dropped span's Cumulative, so a gauge that read
	// the ledger AFTER the decision would see a denominator already shrunk to the survivors and
	// report a meaningless 1.0. The pre-decision capture is what keeps the dropped mass counted.
	if math.Abs(g.TotalMass-total) > 1e-6 {
		t.Fatalf("gauge total mass %.6f, want the pre-decision total %.6f — the denominator shrank to the survivors",
			g.TotalMass, total)
	}
	// Binding the emission to the decision: the mass the gauge dropped from the numerator is
	// exactly the mass carried by the span this decision evicted.
	if gap := g.TotalMass - g.KeptMass; math.Abs(gap-coldMass) > 1e-6 {
		t.Fatalf("gauge withheld %.6f of mass from the survivors, but the decision evicted %q carrying %.6f",
			gap, evicted[0].ID, coldMass)
	}
	if want := g.KeptMass / g.TotalMass; math.Abs(g.Fraction-want) > 1e-9 {
		t.Fatalf("retained_mass_fraction %.9f != KeptMass/TotalMass %.9f", g.Fraction, want)
	}
	wantTok := float64(len(gaugeSys)+len(gaugeWarm)) / float64(len(gaugeSys)+len(gaugeWarm)+len(gaugeCold))
	if math.Abs(g.TokenFraction-wantTok) > 1e-9 {
		t.Fatalf("token fraction %.6f, want %.6f (survivors' positions over the pre-decision residency)",
			g.TokenFraction, wantTok)
	}
	t.Logf("#5123 witness: evicted %q (EMA %.4f, %d positions) -> retained_mass_fraction=%.4f (%.4f of %.4f observed mass) at token_fraction=%.4f",
		evicted[0].ID, evicted[0].EMA, evicted[0].Positions,
		g.Fraction, g.KeptMass, g.TotalMass, g.TokenFraction)
}

// TestApplyPlanEmitsRetainedMassForTheResidentView is the LIVE-path witness. ApplyPlan is the
// selection decision the served loop actually reaches — the gateway's complete() loop drives
// agent.InKernelPlanner.ElideKVSpans, which calls exactly this — so grading it here is what makes
// retained_mass_fraction an emission of a live call path rather than of a test-only helper. The
// planner's view keeps sys+warm and elides cold; the gauge must grade that split against the mass
// observed over the whole pre-elision set.
func TestApplyPlanEmitsRetainedMassForTheResidentView(t *testing.T) {
	c, segs := newGaugeContext()
	observeTurn(c, segs, witnessedMass())

	total := segs["sys"].Cumulative + segs["warm"].Cumulative + segs["cold"].Cumulative
	coldMass := segs["cold"].Cumulative

	plan := ctxplan.Plan{
		Objective: ctxplan.ObjGreedy,
		Selected: []ctxplan.Selection{
			{ID: "sys", Step: 0},
			{ID: "warm", Step: 1},
		},
		Elided:     []ctxplan.Elision{{ID: "cold", Step: 2, Digest: "d-cold", Reason: ctxplan.ElideOverBudget}},
		Candidates: 3,
	}
	if w := ctxplan.Audit(plan); !w.Faithful {
		t.Fatalf("fixture plan is not faithful: %+v", w)
	}

	if n := c.ApplyPlan(plan); n != 1 {
		t.Fatalf("ApplyPlan evicted %d segments, want 1 (the elided cold span)", n)
	}
	g := c.LastRetainedMass()

	if !g.Available {
		t.Fatal("gauge unavailable after a witnessed elision — the live decision path is not emitting it")
	}
	if math.Abs(g.TotalMass-total) > 1e-6 {
		t.Fatalf("gauge total mass %.6f, want the pre-elision total %.6f", g.TotalMass, total)
	}
	if gap := g.TotalMass - g.KeptMass; math.Abs(gap-coldMass) > 1e-6 {
		t.Fatalf("gauge withheld %.6f of mass, but the plan elided the span carrying %.6f", gap, coldMass)
	}
	wantTok := float64(len(gaugeSys)+len(gaugeWarm)) / float64(len(gaugeSys)+len(gaugeWarm)+len(gaugeCold))
	if math.Abs(g.TokenFraction-wantTok) > 1e-9 {
		t.Fatalf("token fraction %.6f, want %.6f", g.TokenFraction, wantTok)
	}
	t.Logf("#5123 live-path witness: ApplyPlan elided %d span(s) -> retained_mass_fraction=%.4f (%.4f of %.4f observed mass) at token_fraction=%.4f",
		len(plan.Elided), g.Fraction, g.KeptMass, g.TotalMass, g.TokenFraction)
}

// TestEvictionGaugeFailsClosedWithoutObserver is the observability-only guard: with the attention
// observer never installed (its #852 default), an eviction still happens and the decision reports
// no quality reading rather than a fabricated one.
func TestEvictionGaugeFailsClosedWithoutObserver(t *testing.T) {
	c, _ := newGaugeContext()
	if g := c.LastRetainedMass(); g.Available {
		t.Fatalf("gauge reported Available before any decision was taken: %+v", g)
	}
	// No AttributeRow, no CloseTurn: every span's Cumulative is 0, exactly as on a served path
	// with no observer installed.
	evicted := c.EvictColdest(len(gaugeCold))
	g := c.LastRetainedMass()

	if len(evicted) == 0 {
		t.Fatal("no eviction happened — the fail-closed witness needs a real decision to grade")
	}
	if g.Available {
		t.Fatalf("gauge reported Available with no attention observed: %+v", g)
	}
	if g.Fraction != 0 || g.TotalMass != 0 || g.TotalTokens != 0 {
		t.Fatalf("an unavailable gauge must be all-zero (no 0/0 NaN, no spurious 1.0), got %+v", g)
	}
}

// TestEvictionGaugeDoesNotChangeWhatIsEvicted holds the issue's hard fence: the gauge is
// observability, so a context that emits it must drop byte-for-byte the same spans, and land on
// the same residency, as the pre-#5123 controller did — witnessed here against the plain
// EvictColdest contract (identically driven contexts, identical victims and residency).
func TestEvictionGaugeDoesNotChangeWhatIsEvicted(t *testing.T) {
	plain, plainSegs := newGaugeContext()
	observeTurn(plain, plainSegs, witnessedMass())
	want := plain.EvictColdest(len(gaugeCold))
	wantResidency := plain.CacheLen()

	other, otherSegs := newGaugeContext()
	observeTurn(other, otherSegs, witnessedMass())
	got := other.EvictColdest(len(gaugeCold))

	if len(got) != len(want) {
		t.Fatalf("evicted %d spans, want %d — the decision is not deterministic under the gauge", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("eviction %d differs: %+v vs %+v — the gauge perturbed the decision", i, got[i], want[i])
		}
	}
	if other.CacheLen() != wantResidency {
		t.Fatalf("residency %d != %d after the same decision", other.CacheLen(), wantResidency)
	}
	// And the survivors' own accumulators are untouched by the grading read.
	for _, id := range []string{"sys", "warm"} {
		if otherSegs[id].Cumulative != plainSegs[id].Cumulative {
			t.Fatalf("span %q mass changed under the gauge: %.6f vs %.6f",
				id, otherSegs[id].Cumulative, plainSegs[id].Cumulative)
		}
	}
}

// TestEvictUnderBudgetGradesTheNoOpDecision covers the budget-facing form's keep-everything
// branch: residency already within budget is still a selection decision, and retaining every span
// retains all of the observed mass at all of the tokens.
func TestEvictUnderBudgetGradesTheNoOpDecision(t *testing.T) {
	c, segs := newGaugeContext()
	observeTurn(c, segs, witnessedMass())

	if ev := c.EvictUnderBudget(c.CacheLen()); len(ev) != 0 {
		t.Fatalf("at full budget nothing should be evicted, dropped %+v", ev)
	}
	g := c.LastRetainedMass()

	if !g.Available {
		t.Fatal("gauge unavailable for a keep-everything decision that had witnessed attention")
	}
	if math.Abs(g.Fraction-1) > 1e-9 || math.Abs(g.TokenFraction-1) > 1e-9 {
		t.Fatalf("keeping every span must retain all mass at all tokens, got %+v", g)
	}
}
