package compute

import (
	"reflect"
	"testing"
)

// TestDecidePrewarmAdmissionFailsClosedOnTrigger pins the load-bearing fence-1: a prefix
// that is not proven byte-known is NEVER warmed, regardless of how inviting the window looks.
// This is what makes the prewarm a pure prefetch and not a speculation (issue #810).
func TestDecidePrewarmAdmissionFailsClosedOnTrigger(t *testing.T) {
	// An otherwise-perfect candidate (free pool, long window, fast warm) but the prefix is
	// not byte-known: must skip.
	c := PrewarmCandidate{
		PrefixByteKnown:   false,
		WarmPoolFree:      true,
		ToolLatencyMillis: 5000,
		WarmMillis:        100,
		ResidencyMillis:   10000,
		PrefixTokens:      4096,
	}
	got := DecidePrewarmAdmission(c)
	if got.Verdict != WarmSkip || got.Reason != ReasonPrefixNotKnown {
		t.Fatalf("unknown-prefix candidate: got %v/%q, want warm_skip/%q", got.Verdict, got.Reason, ReasonPrefixNotKnown)
	}
	// The zero value (default-constructed) must skip.
	if z := DecidePrewarmAdmission(PrewarmCandidate{}); z.Verdict != WarmSkip {
		t.Fatalf("zero candidate: got %v, want warm_skip", z.Verdict)
	}
}

// TestDecidePrewarmAdmissionPollutionGate pins fence-2: even a byte-known prefix with a fine
// window is refused when the lowest-priority warm pool has no free capacity — a warm must
// never evict demand-driven residency to place an opportunistic bet.
func TestDecidePrewarmAdmissionPollutionGate(t *testing.T) {
	c := PrewarmCandidate{
		PrefixByteKnown:   true,
		WarmPoolFree:      false,
		ToolLatencyMillis: 5000,
		WarmMillis:        100,
		ResidencyMillis:   10000,
	}
	got := DecidePrewarmAdmission(c)
	if got.Verdict != WarmSkip || got.Reason != ReasonPoolPressure {
		t.Fatalf("pool-pressure candidate: got %v/%q, want warm_skip/%q", got.Verdict, got.Reason, ReasonPoolPressure)
	}
}

// TestDecidePrewarmAdmissionTimeliness exercises the fence-3 window logic: no window, too
// short, lands hot, and the boundary case where the warm completes exactly at arrival.
func TestDecidePrewarmAdmissionTimeliness(t *testing.T) {
	base := PrewarmCandidate{PrefixByteKnown: true, WarmPoolFree: true}

	cases := []struct {
		name       string
		mut        func(c *PrewarmCandidate)
		wantV      PrewarmVerdict
		wantReason PrewarmReason
	}{
		{
			name:       "no latency window",
			mut:        func(c *PrewarmCandidate) { c.ToolLatencyMillis = 0; c.WarmMillis = 50; c.ResidencyMillis = 1000 },
			wantV:      WarmSkip,
			wantReason: ReasonNoLatencyWindow,
		},
		{
			name:       "window too short (warm slower than tool)",
			mut:        func(c *PrewarmCandidate) { c.ToolLatencyMillis = 100; c.WarmMillis = 400; c.ResidencyMillis = 1000 },
			wantV:      WarmSkip,
			wantReason: ReasonWindowTooShort,
		},
		{
			name:       "lands hot (completes in time, survives until arrival)",
			mut:        func(c *PrewarmCandidate) { c.ToolLatencyMillis = 1000; c.WarmMillis = 200; c.ResidencyMillis = 5000 },
			wantV:      WarmNow,
			wantReason: ReasonLandsHot,
		},
		{
			name:       "boundary: warm completes exactly at arrival",
			mut:        func(c *PrewarmCandidate) { c.ToolLatencyMillis = 300; c.WarmMillis = 300; c.ResidencyMillis = 0 },
			wantV:      WarmNow,
			wantReason: ReasonLandsHot,
		},
	}
	for _, tc := range cases {
		c := base
		tc.mut(&c)
		got := DecidePrewarmAdmission(c)
		if got.Verdict != tc.wantV || got.Reason != tc.wantReason {
			t.Fatalf("%s: got %v/%q, want %v/%q", tc.name, got.Verdict, got.Reason, tc.wantV, tc.wantReason)
		}
	}
}

// TestDecidePrewarmAdmissionDeferClosedForm pins the prefetch-distance knob: when warming now
// would land too early to survive eviction, the decision defers by exactly (slack - R), and
// re-deciding after that defer yields WarmNow. This is the closed-form correctness the
// timeliness fence promises.
func TestDecidePrewarmAdmissionDeferClosedForm(t *testing.T) {
	// T=10000 tool, W=200 warm, R=2000 residency -> slack=9800 > 2000 -> defer 7800.
	c := PrewarmCandidate{
		PrefixByteKnown:   true,
		WarmPoolFree:      true,
		ToolLatencyMillis: 10000,
		WarmMillis:        200,
		ResidencyMillis:   2000,
		PrefixTokens:      8192,
	}
	d := DecidePrewarmAdmission(c)
	if d.Verdict != WarmDefer || d.Reason != ReasonDeferTooEarly {
		t.Fatalf("too-early candidate: got %v/%q, want warm_defer/%q", d.Verdict, d.Reason, ReasonDeferTooEarly)
	}
	wantDefer := (10000 - 200) - 2000 // slack - R = 7800
	if d.DeferMillis != wantDefer {
		t.Fatalf("defer delay: got %d, want %d", d.DeferMillis, wantDefer)
	}
	// Re-decide after the defer: the remaining latency window is T - DeferMillis.
	c2 := c
	c2.ToolLatencyMillis = c.ToolLatencyMillis - d.DeferMillis // 10000 - 7800 = 2200
	d2 := DecidePrewarmAdmission(c2)
	if d2.Verdict != WarmNow || d2.Reason != ReasonLandsHot {
		t.Fatalf("re-decide after defer: got %v/%q, want warm_now/%q", d2.Verdict, d2.Reason, ReasonLandsHot)
	}
	// And the now-warm slack must be within the residency budget (it lands hot).
	if slack := c2.ToolLatencyMillis - c2.WarmMillis; slack > c2.ResidencyMillis {
		t.Fatalf("post-defer slack %d exceeds residency %d — would not land hot", slack, c2.ResidencyMillis)
	}
}

// TestDecidePrewarmAdmissionZeroResidencyForcesLastMoment pins the R=0 edge: with no
// residency budget a lowest-priority prefix is reclaimed instantly, so the only hot warm is
// one that completes exactly at arrival — the decision must defer to the last moment.
func TestDecidePrewarmAdmissionZeroResidencyForcesLastMoment(t *testing.T) {
	c := PrewarmCandidate{
		PrefixByteKnown:   true,
		WarmPoolFree:      true,
		ToolLatencyMillis: 1000,
		WarmMillis:        150,
		ResidencyMillis:   0,
	}
	d := DecidePrewarmAdmission(c)
	if d.Verdict != WarmDefer {
		t.Fatalf("zero-residency: got %v, want warm_defer", d.Verdict)
	}
	if d.DeferMillis != 1000-150 { // slack - 0 = 850: start so warm completes at arrival
		t.Fatalf("zero-residency defer: got %d, want %d", d.DeferMillis, 1000-150)
	}
}

// TestDecidePrewarmAdmissionDeterministic pins the house invariant: the decision is a pure
// function of its integer inputs, so the same candidate yields a byte-identical decision
// every time (no hardware/clock drift).
func TestDecidePrewarmAdmissionDeterministic(t *testing.T) {
	c := PrewarmCandidate{PrefixByteKnown: true, WarmPoolFree: true, ToolLatencyMillis: 4321, WarmMillis: 123, ResidencyMillis: 777}
	first := DecidePrewarmAdmission(c)
	for i := 0; i < 100; i++ {
		if got := DecidePrewarmAdmission(c); !reflect.DeepEqual(got, first) {
			t.Fatalf("non-deterministic at iter %d: got %+v, want %+v", i, got, first)
		}
	}
}

// TestPlanPrewarmAdmission pins the fold: a mixed batch produces index-aligned decisions and
// correct aggregate stats, and the empty input is the documented nil/zero.
func TestPlanPrewarmAdmission(t *testing.T) {
	if items, stats := PlanPrewarmAdmission(nil); items != nil || stats != (PrewarmStats{}) {
		t.Fatalf("nil input: got %v/%+v, want nil/zero", items, stats)
	}

	cands := []PrewarmCandidate{
		// 0: lands hot, 4096 tokens warmed
		{PrefixByteKnown: true, WarmPoolFree: true, ToolLatencyMillis: 1000, WarmMillis: 100, ResidencyMillis: 5000, PrefixTokens: 4096},
		// 1: unknown prefix -> skip
		{PrefixByteKnown: false, WarmPoolFree: true, ToolLatencyMillis: 1000, WarmMillis: 100, ResidencyMillis: 5000, PrefixTokens: 999},
		// 2: too early -> defer
		{PrefixByteKnown: true, WarmPoolFree: true, ToolLatencyMillis: 10000, WarmMillis: 200, ResidencyMillis: 2000, PrefixTokens: 1234},
		// 3: lands hot, 2048 tokens warmed
		{PrefixByteKnown: true, WarmPoolFree: true, ToolLatencyMillis: 800, WarmMillis: 300, ResidencyMillis: 3000, PrefixTokens: 2048},
		// 4: pool pressure -> skip
		{PrefixByteKnown: true, WarmPoolFree: false, ToolLatencyMillis: 1000, WarmMillis: 100, ResidencyMillis: 5000, PrefixTokens: 555},
	}
	items, stats := PlanPrewarmAdmission(cands)
	if len(items) != len(cands) {
		t.Fatalf("items len = %d, want %d", len(items), len(cands))
	}
	for i, it := range items {
		if it.Index != i {
			t.Fatalf("item %d index = %d, want %d", i, it.Index, i)
		}
	}
	want := PrewarmStats{Candidates: 5, Warmed: 2, Deferred: 1, Skipped: 2, TokensWarmed: 4096 + 2048}
	if stats != want {
		t.Fatalf("stats = %+v, want %+v", stats, want)
	}
	// Spot-check that only WarmNow candidates contributed tokens (defer's 1234 must not count).
	if stats.TokensWarmed != 6144 {
		t.Fatalf("TokensWarmed = %d, want 6144 (defer/skip tokens excluded)", stats.TokensWarmed)
	}
}
