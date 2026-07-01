package vcacheobserve

import "testing"

const sec = int64(1000)

// planningSpikeTurns builds one family whose cache_creation spikes right after a
// reset event — the spike should be attributed to context planning.
func planningSpikeTurns() []Turn {
	return []Turn{
		{Family: "alpha", UnixMillis: 0, InputTokens: 100, CacheCreation: 40000},
		{Family: "alpha", UnixMillis: 10 * sec, InputTokens: 50, CacheRead: 40000, CacheCreation: 500},
		{Family: "alpha", UnixMillis: 20 * sec, InputTokens: 50, CacheRead: 40000, CacheCreation: 500},
		// a context reset lands at 25s; the very next turn re-warms the whole prefix
		// (a real cache_create spike, ~80x the ~500 baseline).
		{Family: "alpha", UnixMillis: 30 * sec, InputTokens: 100, CacheCreation: 42000},
	}
}

// providerSpikeTurns builds one family whose cache_creation spikes with NO lifecycle
// event anywhere nearby — a natural TTL expiry/cold miss the join should attribute to
// provider cache behavior.
func providerSpikeTurns() []Turn {
	return []Turn{
		{Family: "beta", UnixMillis: 0, InputTokens: 100, CacheCreation: 30000},
		{Family: "beta", UnixMillis: 10 * sec, InputTokens: 50, CacheRead: 30000, CacheCreation: 400},
		{Family: "beta", UnixMillis: 20 * sec, InputTokens: 50, CacheRead: 30000, CacheCreation: 400},
		// no reset/compaction/page-fault/prefix-mutation event was ever logged for
		// "beta" — this spike is an unexplained provider-side miss (e.g. TTL expiry).
		{Family: "beta", UnixMillis: 1000 * sec, InputTokens: 100, CacheCreation: 31000},
	}
}

func TestVCacheContextJoinAttributesContextPlanning(t *testing.T) {
	turns := planningSpikeTurns()
	events := []LifecycleEvent{
		ResetEvent("alpha", 25*sec, "reset", "hidden restart re-entered cold"),
	}
	rep := JoinContext(JoinInput{Turns: turns, Events: events})

	if rep.Schema != JoinSchema {
		t.Fatalf("schema: got %q want %q", rep.Schema, JoinSchema)
	}
	if rep.Turns != len(turns) || rep.Events != len(events) {
		t.Fatalf("counts: got turns=%d events=%d want turns=%d events=%d", rep.Turns, rep.Events, len(turns), len(events))
	}

	var spike *AttributedChange
	for i := range rep.Changes {
		if rep.Changes[i].Change == ChangeCacheCreateSpike {
			spike = &rep.Changes[i]
		}
	}
	if spike == nil {
		t.Fatalf("expected a cache_create_spike change, got changes=%+v", rep.Changes)
	}
	if spike.Cause != CausePlanning {
		t.Fatalf("cause: got %s want %s (detail=%s)", spike.Cause, CausePlanning, spike.Detail)
	}
	if spike.MatchedEvent == nil {
		t.Fatalf("expected a matched event for a planning-attributed change")
	}
	if spike.MatchedEvent.Kind != EventContextReset {
		t.Fatalf("matched event kind: got %s want %s", spike.MatchedEvent.Kind, EventContextReset)
	}
	if spike.Family != "alpha" {
		t.Fatalf("family: got %q want alpha", spike.Family)
	}
	// Both detectors legitimately co-fire on a full re-warm transition (it is both a
	// cache_create_spike AND a hit_rate_drop) — every change on this all-planning
	// fixture must be planning-attributed, none provider-attributed.
	if rep.Summary.PlanningAttributed == 0 || rep.Summary.PlanningAttributed != rep.Summary.TotalChanges {
		t.Fatalf("summary.planning_attributed: got %d want == total %d (summary=%+v)", rep.Summary.PlanningAttributed, rep.Summary.TotalChanges, rep.Summary)
	}
	if rep.Summary.ProviderAttributed != 0 {
		t.Fatalf("summary.provider_attributed: got %d want 0", rep.Summary.ProviderAttributed)
	}
}

func TestVCacheContextJoinAttributesProviderCacheBehavior(t *testing.T) {
	turns := providerSpikeTurns()
	// No lifecycle events at all — nothing should ever match.
	rep := JoinContext(JoinInput{Turns: turns})

	var spike *AttributedChange
	for i := range rep.Changes {
		if rep.Changes[i].Change == ChangeCacheCreateSpike {
			spike = &rep.Changes[i]
		}
	}
	if spike == nil {
		t.Fatalf("expected a cache_create_spike change, got changes=%+v", rep.Changes)
	}
	if spike.Cause != CauseProviderBehavior {
		t.Fatalf("cause: got %s want %s (detail=%s)", spike.Cause, CauseProviderBehavior, spike.Detail)
	}
	if spike.MatchedEvent != nil {
		t.Fatalf("expected no matched event for a provider-attributed change, got %+v", spike.MatchedEvent)
	}
	// Both detectors legitimately co-fire here too; on this all-provider fixture every
	// change must be provider-attributed, none planning-attributed.
	if rep.Summary.ProviderAttributed == 0 || rep.Summary.ProviderAttributed != rep.Summary.TotalChanges {
		t.Fatalf("summary.provider_attributed: got %d want == total %d (summary=%+v)", rep.Summary.ProviderAttributed, rep.Summary.TotalChanges, rep.Summary)
	}
	if rep.Summary.PlanningAttributed != 0 {
		t.Fatalf("summary.planning_attributed: got %d want 0", rep.Summary.PlanningAttributed)
	}
}

// TestVCacheContextJoinMixedFamiliesIndependentAttribution runs both fixtures
// together in one JoinInput and confirms the join keys strictly by (family, time):
// alpha's reset must never explain beta's unrelated spike, and vice versa.
func TestVCacheContextJoinMixedFamiliesIndependentAttribution(t *testing.T) {
	var turns []Turn
	turns = append(turns, planningSpikeTurns()...)
	turns = append(turns, providerSpikeTurns()...)
	events := []LifecycleEvent{
		ResetEvent("alpha", 25*sec, "reset", "hidden restart re-entered cold"),
	}
	rep := JoinContext(JoinInput{Turns: turns, Events: events})

	if rep.Summary.PlanningAttributed == 0 {
		t.Fatalf("planning_attributed: got 0, want > 0 (changes=%+v)", rep.Changes)
	}
	if rep.Summary.ProviderAttributed == 0 {
		t.Fatalf("provider_attributed: got 0, want > 0 (changes=%+v)", rep.Changes)
	}
	for _, c := range rep.Changes {
		if c.Change != ChangeCacheCreateSpike {
			continue
		}
		switch c.Family {
		case "alpha":
			if c.Cause != CausePlanning {
				t.Fatalf("alpha spike must be planning-attributed, got %s", c.Cause)
			}
		case "beta":
			if c.Cause != CauseProviderBehavior {
				t.Fatalf("beta spike must be provider-attributed (no cross-family leakage), got %s", c.Cause)
			}
		}
	}
}

// TestVCacheContextJoinHitRateDropDetection covers the second detector: a hit-rate
// collapse with a nearby page-fault event should be planning-attributed.
func TestVCacheContextJoinHitRateDropDetection(t *testing.T) {
	turns := []Turn{
		{Family: "gamma", UnixMillis: 0, InputTokens: 50, CacheRead: 9500, CacheCreation: 0},
		{Family: "gamma", UnixMillis: 10 * sec, InputTokens: 50, CacheRead: 9500, CacheCreation: 0},
		// hit rate collapses to 0 right after a page-fault "deny" decision.
		{Family: "gamma", UnixMillis: 20 * sec, InputTokens: 9000, CacheRead: 0, CacheCreation: 0},
	}
	events := []LifecycleEvent{
		PageFaultEventFrom("gamma", 19*sec, "deny", "required span denied page-in"),
	}
	rep := JoinContext(JoinInput{Turns: turns, Events: events})

	var drop *AttributedChange
	for i := range rep.Changes {
		if rep.Changes[i].Change == ChangeHitRateDrop {
			drop = &rep.Changes[i]
		}
	}
	if drop == nil {
		t.Fatalf("expected a hit_rate_drop change, got changes=%+v", rep.Changes)
	}
	if drop.Cause != CausePlanning {
		t.Fatalf("cause: got %s want %s", drop.Cause, CausePlanning)
	}
	if drop.MatchedEvent == nil || drop.MatchedEvent.Kind != EventPageFault {
		t.Fatalf("expected a matched page_fault event, got %+v", drop.MatchedEvent)
	}
}

func TestVCacheContextJoinEmptyInputsAreSafe(t *testing.T) {
	rep := JoinContext(JoinInput{})
	if rep.Schema != JoinSchema {
		t.Fatalf("schema: got %q want %q", rep.Schema, JoinSchema)
	}
	if len(rep.Changes) != 0 {
		t.Fatalf("expected no changes on empty input, got %+v", rep.Changes)
	}
}

func TestVCacheContextJoinValidLifecycleEventKind(t *testing.T) {
	for _, k := range []LifecycleEventKind{EventContextReset, EventCompaction, EventPageFault, EventPrefixMutation} {
		if !ValidLifecycleEventKind(k) {
			t.Fatalf("expected %s to be a valid lifecycle event kind", k)
		}
	}
	if ValidLifecycleEventKind("bogus_kind") {
		t.Fatalf("expected an unknown kind to be rejected")
	}
}

func TestVCacheContextJoinConstructorsEchoVocabulary(t *testing.T) {
	r := ResetEvent("alpha", 1000, "reset", "d")
	if r.Kind != EventContextReset || r.Outcome != "reset" {
		t.Fatalf("ResetEvent: got %+v", r)
	}
	c := CompactionEvent("alpha", 1000, "d")
	if c.Kind != EventCompaction || c.Outcome != "compaction" {
		t.Fatalf("CompactionEvent: got %+v", c)
	}
	p := PageFaultEventFrom("alpha", 1000, "page_in", "d")
	if p.Kind != EventPageFault || p.Outcome != "page_in" {
		t.Fatalf("PageFaultEventFrom: got %+v", p)
	}
	m := PrefixMutationEvent("alpha", 1000, 500, "d")
	if m.Kind != EventPrefixMutation || m.Outcome != "diverged" {
		t.Fatalf("PrefixMutationEvent: got %+v", m)
	}
}
