package ctxplan

import (
	"strings"
	"testing"
)

// turnSpans builds a small, deterministic store the planner can score against a
// forecast: a system pin, a relevant span the forecast wants, and an irrelevant one.
func turnSpans() []Span {
	return []Span{
		{ID: "sys", Step: 0, Role: "system", Descriptor: "system framing", Bytes: 40, Durability: DurabilityDurable},
		{ID: "refund", Step: 1, Role: "user", Descriptor: "the refund fee dispute", Bytes: 40, Durability: DurabilitySession},
		{ID: "weather", Step: 2, Role: "tool", Descriptor: "the weather forecast today", Bytes: 40, Durability: DurabilityTurn},
	}
}

func refundForecast() Forecast {
	return Forecast{Intents: []string{"refund fee"}, Horizon: 1, Pins: []string{"sys"}}
}

// countingCost wraps TokenCost and counts how many spans it priced, so a test can
// PROVE a HIT did no re-scoring (a cache hit must not call the cost model at all).
type countingCost struct{ calls int }

func (c *countingCost) cost(s Span) int { c.calls++; return TokenCost(s) }

func TestPlanCacheStableForecastHits(t *testing.T) {
	spans := turnSpans()
	f := refundForecast()
	budget := Budget{Tokens: 64}

	fresh := PlanCells(spans, f, budget, nil)

	cc := &countingCost{}
	var cache PlanCache

	// Turn 1: cold MISS, plan computed, cache primed. The cost model is called.
	p1, hit1 := cache.PlanWithCache(spans, f, budget, cc.cost)
	if hit1 {
		t.Fatalf("turn 1 must MISS (cold cache), got HIT")
	}
	if cc.calls == 0 {
		t.Fatalf("turn 1 MISS must score the spans (cost model never called)")
	}
	if p1.Explain() != fresh.Explain() {
		t.Fatalf("turn 1 plan must equal a fresh PlanCells plan")
	}

	// Turn 2: SAME forecast, SAME store -> HIT, byte-identical plan, NO re-scoring.
	before := cc.calls
	p2, hit2 := cache.PlanWithCache(spans, f, budget, cc.cost)
	if !hit2 {
		t.Fatalf("turn 2 must HIT (stable forecast over an unchanged store)")
	}
	if cc.calls != before {
		t.Fatalf("turn 2 HIT must NOT re-score: cost model called %d extra time(s)", cc.calls-before)
	}
	if p2.Explain() != fresh.Explain() {
		t.Fatalf("reused plan must be byte-identical to a fresh re-plan (issue #561 witness)")
	}
}

func TestPlanIDDeterministicAndMovesOnMaterialChange(t *testing.T) {
	spans := turnSpans()
	spans[1].Digest = "sha256:refund-v1"
	spans[2].Digest = "sha256:weather-v1"
	f := refundForecast()
	budget := Budget{Tokens: 64}

	a := PlanCells(spans, f, budget, nil)
	b := PlanCells(append([]Span(nil), spans...), f, budget, nil)
	if a.ID == "" {
		t.Fatal("managed-context plans must carry a deterministic plan_id")
	}
	if a.ID != b.ID {
		t.Fatalf("identical managed-context inputs must produce the same plan_id: %q != %q", a.ID, b.ID)
	}
	if want := PlanID(a, ForecastFingerprint(f)); a.ID != want {
		t.Fatalf("plan_id must be the canonical PlanID digest: got %q want %q", a.ID, want)
	}
	if !strings.Contains(a.Explain(), "plan_id="+a.ID) {
		t.Fatalf("Explain must emit the plan_id, got:\n%s", a.Explain())
	}

	budgetChanged := PlanCells(spans, f, Budget{Tokens: budget.Tokens - 1}, nil)
	if budgetChanged.ID == a.ID {
		t.Fatalf("a budget change must move the plan_id")
	}

	forecastChanged := f
	forecastChanged.Intents = []string{"weather forecast"}
	if got := PlanCells(spans, forecastChanged, budget, nil); got.ID == a.ID {
		t.Fatalf("a forecast change must move the plan_id")
	}

	selectedChanged := append([]Span(nil), spans...)
	selectedChanged[1].Descriptor = "the refund fee dispute escalated"
	if got := PlanCells(selectedChanged, f, budget, nil); got.ID == a.ID {
		t.Fatalf("a selected span material change must move the plan_id")
	}
}

func TestPlanIDBindsElidedHandlesAndQueryView(t *testing.T) {
	spans := turnSpans()
	spans[1].Digest = "sha256:refund-v1"
	spans[2].Digest = "sha256:weather-v1"
	f := refundForecast()
	budget := Budget{Tokens: 10} // fits the pinned system span, leaving the others recoverable.

	a := PlanCells(spans, f, budget, nil)
	if len(a.Elided) == 0 {
		t.Fatal("test setup must leave at least one span elided")
	}

	handleChanged := append([]Span(nil), spans...)
	handleChanged[1].Digest = "sha256:refund-v2"
	if got := PlanCells(handleChanged, f, budget, nil); got.ID == a.ID {
		t.Fatalf("an elided recovery-handle change must move the plan_id")
	}

	query := PlanQuery{Intents: f.Intents, Budget: &budget, Horizon: f.Horizon, Pins: f.Pins}
	view := query.Plan(spans, nil)
	if view.PlanID == "" || view.PlanID != view.Plan.ID {
		t.Fatalf("PlanView must emit the same plan_id as the embedded plan: view=%q plan=%q", view.PlanID, view.Plan.ID)
	}
}

func TestPlanCacheChangedForecastMisses(t *testing.T) {
	spans := turnSpans()
	budget := Budget{Tokens: 64}
	var cache PlanCache

	if _, hit := cache.PlanWithCache(spans, refundForecast(), budget, nil); hit {
		t.Fatalf("turn 1 must MISS (cold)")
	}
	// A different forecast intent must MISS and recompute.
	changed := Forecast{Intents: []string{"auth token rotation"}, Horizon: 1, Pins: []string{"sys"}}
	v := cache.Lookup(spans, changed, budget)
	if v.Hit {
		t.Fatalf("a changed forecast must MISS, got HIT")
	}
	if v.Reason != PlanCacheMissForecast {
		t.Fatalf("changed forecast must MISS with reason %q, got %q", PlanCacheMissForecast, v.Reason)
	}
}

func TestPlanCacheNonAppendStoreGrowthMisses(t *testing.T) {
	spans := turnSpans()
	f := refundForecast()
	budget := Budget{Tokens: 64}
	var cache PlanCache
	cache.PlanWithCache(spans, f, budget, nil) // prime

	// (a) Append of a LIVE, possibly-higher-benefit span -> MISS (must fold it in).
	appendedLive := append(turnSpans(), Span{
		ID: "refund2", Step: 3, Role: "user", Descriptor: "another refund fee question", Bytes: 40, Durability: DurabilitySession,
	})
	if v := cache.Lookup(appendedLive, f, budget); v.Hit {
		t.Fatalf("appending a live candidate must MISS (a higher-benefit span must force a recompute)")
	} else if v.Reason != PlanCacheMissStore {
		t.Fatalf("appended live span must MISS with reason %q, got %q", PlanCacheMissStore, v.Reason)
	}

	// (b) NON-APPEND: a pre-existing span's descriptor edited in place -> MISS.
	edited := turnSpans()
	edited[1].Descriptor = "the refund fee dispute, escalated" // selection-relevant change
	if v := cache.Lookup(edited, f, budget); v.Hit {
		t.Fatalf("editing a pre-existing span must MISS (non-append growth)")
	} else if v.Reason != PlanCacheMissStore {
		t.Fatalf("edited span must MISS with reason %q, got %q", PlanCacheMissStore, v.Reason)
	}

	// (c) NON-APPEND: a pre-existing span removed -> MISS.
	removed := turnSpans()[:2]
	if v := cache.Lookup(removed, f, budget); v.Hit {
		t.Fatalf("removing a span must MISS (non-append growth)")
	}
}

func TestPlanCacheInertAppendHits(t *testing.T) {
	spans := turnSpans()
	f := refundForecast()
	budget := Budget{Tokens: 64}
	var cache PlanCache
	cache.PlanWithCache(spans, f, budget, nil) // prime

	// Append a SEALED span: it can never be resident, so the selection is unchanged and
	// the cached plan stays optimal -> HIT.
	withSealed := append(turnSpans(), Span{
		ID: "poison", Step: 3, Role: "tool", Descriptor: "quarantined", Bytes: 40, Sealed: true,
	})
	v := cache.Lookup(withSealed, f, budget)
	if !v.Hit {
		t.Fatalf("appending an INERT (sealed) span must HIT (it can never be resident)")
	}
	// And the reused plan equals a fresh plan over the larger store (the sealed span is
	// elided either way) — observable equivalence holds.
	fresh := PlanCells(withSealed, f, budget, nil)
	// The cached plan was built over the SMALLER store, so it does not list the sealed
	// span; what matters for the witness is the RESIDENT selection, which is identical.
	if residentExplain(v.Plan) != residentExplain(fresh) {
		t.Fatalf("inert-append HIT must keep the same resident selection as a fresh plan")
	}
}

func TestForecastFingerprintIsDeterministicAndCanonical(t *testing.T) {
	a := Forecast{Intents: []string{"refund fee", "auth token"}, Horizon: 2, Pins: []string{"sys", "goal"}}
	// Same inputs -> same fingerprint (determinism).
	if ForecastFingerprint(a) != ForecastFingerprint(a) {
		t.Fatalf("fingerprint must be deterministic")
	}
	// Intent ORDER and pin ORDER (and a duplicate pin) must not change the fingerprint:
	// they cannot change selection.
	b := Forecast{Intents: []string{"auth token", "refund fee"}, Horizon: 2, Pins: []string{"goal", "sys", "sys"}}
	if ForecastFingerprint(a) != ForecastFingerprint(b) {
		t.Fatalf("intent/pin order and duplicate pins must NOT change the fingerprint")
	}
	// A zero Weights literal must fingerprint the same as an explicit DefaultWeights
	// (orDefault makes them plan identically).
	z := Forecast{Intents: []string{"refund fee"}, Horizon: 1}
	d := Forecast{Intents: []string{"refund fee"}, Horizon: 1, Weights: DefaultWeights()}
	if ForecastFingerprint(z) != ForecastFingerprint(d) {
		t.Fatalf("a zero-weight forecast must fingerprint as DefaultWeights")
	}
	// A real scoring change MUST change the fingerprint.
	if ForecastFingerprint(a) == ForecastFingerprint(Forecast{Intents: []string{"refund fee"}, Horizon: 2, Pins: []string{"sys", "goal"}}) {
		t.Fatalf("dropping an intent must change the fingerprint")
	}
	if ForecastFingerprint(a) == ForecastFingerprint(Forecast{Intents: a.Intents, Horizon: 3, Pins: a.Pins}) {
		t.Fatalf("a different horizon must change the fingerprint")
	}
	tuned := a
	tuned.Weights = Weights{Relevance: 2.0}
	if ForecastFingerprint(a) == ForecastFingerprint(tuned) {
		t.Fatalf("retuned weights must change the fingerprint")
	}
}

func TestStoreVersionIsScanOrderIndependentButContentSensitive(t *testing.T) {
	spans := turnSpans()
	reordered := []Span{spans[2], spans[0], spans[1]}
	if StoreVersion(spans) != StoreVersion(reordered) {
		t.Fatalf("StoreVersion must be independent of scan order")
	}
	// Flipping a seal bit IS a selection change -> the version must move.
	sealed := turnSpans()
	sealed[1].Sealed = true
	if StoreVersion(spans) == StoreVersion(sealed) {
		t.Fatalf("flipping a span's seal bit must change StoreVersion")
	}
	// A pure append changes the version too (appendedInert is the finer check).
	grown := append(turnSpans(), Span{ID: "x", Step: 9, Descriptor: "new", Bytes: 4})
	if StoreVersion(spans) == StoreVersion(grown) {
		t.Fatalf("appending a span must change StoreVersion")
	}
}

// residentExplain renders only the resident (Selected) ids+benefits of a plan, so two
// plans built over stores that differ only in an inert (always-elided) span can be
// compared on the part the cache promises is identical: the resident selection.
func residentExplain(p Plan) string {
	out := ""
	for _, s := range p.Selected {
		out += s.ID + ":" + s.Role + ";"
	}
	return out
}
