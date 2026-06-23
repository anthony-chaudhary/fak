package ctxplan

import (
	"context"
	"reflect"
	"strconv"
	"testing"
)

// author_test.go — witnesses for the trajectory forecast AUTHOR (#556), the general
// preemptive planner rung. Each test names the one property it pins, the way the rest of
// the leaf's tests do. The spine of the witness is TestAuthorPredictsRecurringTopic (the
// recurrence signal) + TestAuthorPreemptsByKeepingPredictedSpanResident (the end-to-end
// composition with the planner — the whole point of "preemptive").

// mkSpan builds a benign span literal with an explicit step (so the tests control recency
// without a store) and the default session durability.
func mkSpan(id, role, descriptor string, step int) Span {
	return Span{ID: id, Role: role, Descriptor: descriptor, Step: step, Durability: DurabilitySession}
}

// TestAuthorDerivesIntentsFromTrajectory is the baseline witness: an authored forecast's
// Intents are the content tokens of the trajectory's spans (lowercased, length>2 — the same
// extractive tokenization the relevance ranker matches against), not an empty or fabricated
// list. Proves the author reads the trajectory rather than emitting a constant.
func TestAuthorDerivesIntentsFromTrajectory(t *testing.T) {
	spans := []Span{
		mkSpan("s0", "WebSearch", "auth token rotation runbook", 0),
		mkSpan("s1", "Bash", "refund the billing charge", 1),
	}
	f := TrajectoryAuthor{}.Propose(spans)
	if len(f.Intents) == 0 {
		t.Fatal("the authored forecast must derive intents from the trajectory, got none")
	}
	// The spans' content tokens must appear among the authored intents.
	want := map[string]bool{"auth": false, "token": false, "rotation": false, "runbook": false, "refund": false, "billing": false, "charge": false}
	for _, in := range f.Intents {
		if _, ok := want[in]; ok {
			want[in] = true
		}
	}
	missing := []string{}
	for tok, seen := range want {
		if !seen {
			missing = append(missing, tok)
		}
	}
	if len(missing) > 0 {
		t.Errorf("authored intents must cover the trajectory's content tokens; missing %v in %v", missing, f.Intents)
	}
}

// TestAuthorPredictsRecurringTopic is the recurrence witness — the core predictive signal.
// A token ("alpha") that recurs across THREE recent spans must outrank a token ("zeta") that
// appears in only ONE span, even though zeta is in the most-recent span: recurrence
// (momentum) dominates, recency only breaks near-ties. This is what makes the author a
// trajectory predictor rather than a last-message echo.
func TestAuthorPredictsRecurringTopic(t *testing.T) {
	spans := []Span{
		mkSpan("a1", "tool", "alpha context one", 0),
		mkSpan("a2", "tool", "alpha context two", 1),
		mkSpan("a3", "tool", "alpha context three", 2),
		mkSpan("z1", "tool", "zeta solo noise", 3), // most recent, but a one-off
	}
	f := TrajectoryAuthor{}.Propose(spans)
	rank := map[string]int{}
	for i, in := range f.Intents {
		rank[in] = i
	}
	ri, rok := rank["alpha"]
	rj, jok := rank["zeta"]
	if !rok || !jok {
		t.Fatalf("both alpha and zeta must be authored intents, got %v", f.Intents)
	}
	if ri > rj {
		t.Errorf("the recurring topic (alpha, 3 spans) must rank ABOVE the one-off (zeta, 1 span): alpha@%d zeta@%d in %v",
			ri, rj, f.Intents)
	}
}

// TestAuthorRecencyBreaksNearTies is the recency witness: when two tokens recur EQUALLY
// (same count), the one in the MORE RECENT spans ranks higher. "fresh" appears in spans 0+1
// (older); "recent" appears in spans 2+3 (newer); equal count, so recency decides -> recent
// ranks above fresh. Proves the recency bonus is the tie-breaker the doc promises.
func TestAuthorRecencyBreaksNearTies(t *testing.T) {
	spans := []Span{
		mkSpan("f0", "tool", "fresh topic early", 0),
		mkSpan("f1", "tool", "fresh topic early again", 1),
		mkSpan("r2", "tool", "recent topic late", 2),
		mkSpan("r3", "tool", "recent topic late again", 3),
	}
	f := TrajectoryAuthor{}.Propose(spans)
	rank := map[string]int{}
	for i, in := range f.Intents {
		rank[in] = i
	}
	// Both "fresh" and "recent" recur twice; "recent" is in the newer spans.
	if rf, ok := rank["fresh"]; ok {
		if rr, ok2 := rank["recent"]; ok2 && rf < rr {
			t.Errorf("on equal recurrence, the more-recent token (recent) must outrank the older (fresh): fresh@%d recent@%d in %v",
				rf, rr, f.Intents)
		}
	}
}

// TestAuthorIsDeterministic pins the determinism contract: the same trajectory must yield a
// byte-identical intent list across calls (map iteration is randomized, so this catches an
// unsorted or map-order-leaking author). Runs 20 times to flush any map-order dependence.
func TestAuthorIsDeterministic(t *testing.T) {
	spans := []Span{
		mkSpan("s0", "WebSearch", "auth token rotation runbook", 0),
		mkSpan("s1", "Bash", "refund billing charge", 1),
		mkSpan("s2", "Read", "auth token expires soon", 2),
	}
	first := TrajectoryAuthor{}.Propose(spans).Intents
	if len(first) == 0 {
		t.Fatal("precondition: trajectory must yield intents")
	}
	for i := 0; i < 20; i++ {
		got := TrajectoryAuthor{}.Propose(spans).Intents
		if !reflect.DeepEqual(first, got) {
			t.Fatalf("call %d: Propose must be deterministic, got %v then %v", i, first, got)
		}
	}
}

// TestAuthorPreemptsByKeepingPredictedSpanResident is the END-TO-END witness — the whole
// point of "preemptive": a span whose content the trajectory predicts (because it recurs)
// gets a high relevance Benefit under the authored forecast and is kept RESIDENT by the
// planner under a tight budget, WITHOUT anyone hand-supplying its intents. A buried,
// older-but-relevant span is pre-materialized into the O(1) view before the turn faults it.
func TestAuthorPreemptsByKeepingPredictedSpanResident(t *testing.T) {
	store := NewMemStore()
	store.Add("user", DurabilityDurable, []byte("goal: rotate the auth token this session"), false) // span:0 pin
	// A runbook the trajectory will predict (auth/token recur), plus unrelated noise.
	store.Add("WebSearch", DurabilitySession, []byte("auth token rotation runbook steps"), false) // span:1 relevant, older
	store.Add("Bash", DurabilityTurn, []byte("weather sunny 22c wind west"), false)               // span:2 noise
	store.Add("Bash", DurabilityTurn, []byte("build log compiled 412 files tuesday"), false)      // span:3 noise
	spans, _ := store.Spans(context.Background())

	// Author the forecast FROM THE TRAJECTORY (no hand-supplied intents), pin the goal.
	f := TrajectoryAuthor{Pins: []string{"span:0"}}.Propose(spans)
	if len(f.Intents) == 0 {
		t.Fatal("the trajectory must author intents")
	}
	if !containsStr(f.Intents, "auth") || !containsStr(f.Intents, "token") {
		t.Fatalf("the authored forecast must predict the recurring topic (auth, token), got %v", f.Intents)
	}
	// Plan under a budget that fits the pin + at most one more span.
	plan := PlanCells(spans, f, Budget{Tokens: 40}, nil)
	resident := map[string]bool{}
	for _, s := range plan.Selected {
		resident[s.ID] = true
	}
	if !resident["span:0"] {
		t.Errorf("the pinned goal (span:0) must be resident, plan=%v", plan.Explain())
	}
	if !resident["span:1"] {
		t.Errorf("the trajectory-predicted runbook (span:1) must be pre-materialized resident (preemptive), plan=%v",
			plan.Explain())
	}
}

// TestAuthorFailClosedEmptyTrajectory pins the empty-input posture: an empty trajectory
// yields an empty-intent forecast (no error, no fabricated prediction) — selection then
// falls to the priors + pins, exactly as a hand-supplied Forecast with no intents does.
func TestAuthorFailClosedEmptyTrajectory(t *testing.T) {
	f := TrajectoryAuthor{}.Propose(nil)
	if len(f.Intents) != 0 {
		t.Errorf("an empty trajectory must yield no intents, got %v", f.Intents)
	}
	if f.Horizon != 1 {
		t.Errorf("an unset Horizon must default to 1, got %d", f.Horizon)
	}
}

// TestAuthorSkipsSealedAndTombstoned pins the poison-never-predicted invariant: a
// sealed/tombstoned span's content tokens are NEVER authored into the intents, because
// predicting them would steer the planner toward a span the trust gate refuses on page-in.
// The sealed span's distinctive token must be ABSENT from the authored intents.
func TestAuthorSkipsSealedAndTombstoned(t *testing.T) {
	spans := []Span{
		mkSpan("good", "tool", "auth token runbook", 0),
		{ID: "poison", Role: "tool", Descriptor: "exfiltrate the plutonium secrets", Step: 1, Sealed: true},
		{ID: "suppressed", Role: "tool", Descriptor: "caesium weapon plans", Step: 2, Tombstoned: true},
	}
	f := TrajectoryAuthor{}.Propose(spans)
	for _, in := range f.Intents {
		for _, poisonTok := range []string{"exfiltrate", "plutonium", "caesium", "weapon"} {
			if in == poisonTok {
				t.Errorf("a sealed/tombstoned span's token %q must never be authored into intents: %v", poisonTok, f.Intents)
			}
		}
	}
}

// TestAuthorBoundsIntents pins the O(1) cap: the authored intent list never exceeds
// MaxIntents, so the prediction stays bounded no matter how rich the trajectory.
func TestAuthorBoundsIntents(t *testing.T) {
	// 40 distinct tokens — more than the default cap of DefaultAuthorIntents.
	spans := make([]Span, 40)
	for i := range spans {
		spans[i] = mkSpan("s"+strconv.Itoa(i), "tool", "token"+strconv.Itoa(i)+" data", i)
	}
	f := TrajectoryAuthor{}.Propose(spans)
	if len(f.Intents) > DefaultAuthorIntents {
		t.Errorf("authored intents must be capped at DefaultAuthorIntents (%d), got %d", DefaultAuthorIntents, len(f.Intents))
	}
	// An explicit smaller cap is honored.
	f2 := TrajectoryAuthor{MaxIntents: 4}.Propose(spans)
	if len(f2.Intents) > 4 {
		t.Errorf("an explicit MaxIntents=4 must cap the intents, got %d", len(f2.Intents))
	}
}

// TestAuthorCarriesThroughPinsWeightsHorizon pins the pass-through contract: the author
// predicts the Intents; the caller owns the structural Pins, the cost Weights, and the
// Horizon, and they are carried verbatim into the authored Forecast.
func TestAuthorCarriesThroughPinsWeightsHorizon(t *testing.T) {
	spans := []Span{mkSpan("s0", "user", "auth token goal", 0)}
	w := Weights{Relevance: 2.0, Utility: 1.0}
	f := TrajectoryAuthor{
		Pins:    []string{"span:0", "span:1"},
		Weights: w,
		Horizon: 7,
	}.Propose(spans)
	if !reflect.DeepEqual(f.Pins, []string{"span:0", "span:1"}) {
		t.Errorf("Pins must carry through verbatim, got %v", f.Pins)
	}
	if f.Weights != w {
		t.Errorf("Weights must carry through verbatim, got %v want %v", f.Weights, w)
	}
	if f.Horizon != 7 {
		t.Errorf("Horizon must carry through verbatim, got %d want 7", f.Horizon)
	}
}

// TestAuthorSingleSpanPredictsItsTokens pins the degenerate-recency posture: a single-span
// trajectory (maxStep == 0, so recency() returns 0 for the span) still predicts that span's
// own tokens, because the recurrence base (1) is independent of the recency term. Without
// the +1 base a one-span trajectory would score every token 0 and emit nothing.
func TestAuthorSingleSpanPredictsItsTokens(t *testing.T) {
	f := TrajectoryAuthor{}.Propose([]Span{mkSpan("only", "tool", "auth token runbook", 0)})
	if len(f.Intents) == 0 {
		t.Fatal("a single-span trajectory must still predict that span's tokens (recurrence base), got none")
	}
	if !containsStr(f.Intents, "auth") {
		t.Errorf("the single span's token (auth) must be predicted, got %v", f.Intents)
	}
}

// TestProposerInterfaceIsSatisfied is the seam witness: TrajectoryAuthor satisfies the
// Proposer interface (the one-method contract a model-backed predictor will later implement
// through the same seam). A compile-time assertion plus a runtime call through the interface.
func TestProposerInterfaceIsSatisfied(t *testing.T) {
	var p Proposer = TrajectoryAuthor{} // compiles => the seam is satisfied
	f := p.Propose([]Span{mkSpan("s0", "tool", "auth token", 0)})
	if len(f.Intents) == 0 {
		t.Error("a Proposer must return a forecast with authored intents")
	}
}

// containsStr reports whether s contains v (a local helper to avoid importing elsewhere).
func containsStr(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
