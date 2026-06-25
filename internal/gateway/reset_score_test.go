package gateway

import "testing"

// TestResetScoreCases covers the five acceptance cases the issue (#792) names -- healthy
// cache, prefix-staleness, decay, cooldown, and unknown-provider -- plus the cut-by-default
// posture: every case asserts both the ShouldReset bit and the closed reason.
func TestResetScoreCases(t *testing.T) {
	p := DefaultResetPolicy()
	big := DefaultResetCooldownTurns + 5 // well past the cooldown

	cases := []struct {
		name       string
		state      CacheHealthState
		wantReset  bool
		wantReason ResetReason
	}{
		{
			// Healthy: the prefix is still landing (read ratio high) -> keep cutting.
			name:       "healthy_cache_keeps_cutting",
			state:      CacheHealthState{ObservedTurns: 10, RecentReadRatio: 0.92, HasProviderSignal: true, TurnsSinceReset: big},
			wantReset:  false,
			wantReason: ResetReasonHealthy,
		},
		{
			// Stale prefix: read ratio cratered below the stale floor, cooldown elapsed -> reset.
			name:       "stale_prefix_recommends_reset",
			state:      CacheHealthState{ObservedTurns: 10, RecentReadRatio: 0.02, HasProviderSignal: true, TurnsSinceReset: big},
			wantReset:  true,
			wantReason: ResetReasonStalePrefix,
		},
		{
			// Decay: below healthy but above the hard stale floor, cooldown elapsed -> reset,
			// reason decay (the softer, trending form).
			name:       "decay_below_floor_recommends_reset",
			state:      CacheHealthState{ObservedTurns: 10, RecentReadRatio: 0.15, HasProviderSignal: true, TurnsSinceReset: big},
			wantReset:  true,
			wantReason: ResetReasonDecay,
		},
		{
			// Cooldown: a reset is warranted (stale), but the session reset too recently -> hold.
			name:       "cooldown_holds_a_warranted_reset",
			state:      CacheHealthState{ObservedTurns: 10, RecentReadRatio: 0.02, HasProviderSignal: true, TurnsSinceReset: 1},
			wantReset:  false,
			wantReason: ResetReasonCooldown,
		},
		{
			// Unknown provider: no cache counters at all -> cut-by-default, never reset.
			name:       "unknown_provider_cuts_by_default",
			state:      CacheHealthState{ObservedTurns: 10, RecentReadRatio: 0, HasProviderSignal: false, TurnsSinceReset: big},
			wantReset:  false,
			wantReason: ResetReasonUnknown,
		},
		{
			// Too few turns: even with a low ratio, not enough signal -> unknown.
			name:       "too_few_turns_is_unknown",
			state:      CacheHealthState{ObservedTurns: 2, RecentReadRatio: 0.01, HasProviderSignal: true, TurnsSinceReset: big},
			wantReset:  false,
			wantReason: ResetReasonUnknown,
		},
		{
			// Idle hint strengthens a mid-band ratio into a stale verdict even above the hard floor.
			name:       "idle_hint_forces_stale",
			state:      CacheHealthState{ObservedTurns: 10, RecentReadRatio: 0.20, HasProviderSignal: true, ProviderIdleHint: true, TurnsSinceReset: big},
			wantReset:  true,
			wantReason: ResetReasonStalePrefix,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := p.ResetScore(tc.state)
			if d.ShouldReset != tc.wantReset {
				t.Errorf("ShouldReset = %v, want %v (reason %s, score %.2f)", d.ShouldReset, tc.wantReset, d.Reason, d.Score)
			}
			if d.Reason != tc.wantReason {
				t.Errorf("Reason = %q, want %q", d.Reason, tc.wantReason)
			}
			if d.Score < 0 || d.Score > 1 {
				t.Errorf("Score = %.3f, want within [0,1]", d.Score)
			}
		})
	}
}

// TestResetScoreScoreMonotonic proves the score rises as the read ratio falls -- a shadow log
// can watch the reset pressure build before the bit ever flips.
func TestResetScoreScoreMonotonic(t *testing.T) {
	p := DefaultResetPolicy()
	base := CacheHealthState{ObservedTurns: 10, HasProviderSignal: true, TurnsSinceReset: 100}

	ratios := []float64{0.30, 0.20, 0.10, 0.05, 0.0}
	prev := -1.0
	for _, r := range ratios {
		st := base
		st.RecentReadRatio = r
		got := p.ResetScore(st).Score
		if got < prev {
			t.Fatalf("score not monotonic: ratio %.2f gave %.3f, below previous %.3f", r, got, prev)
		}
		prev = got
	}
	// The healthy end is 0; the fully-stale end is 1.
	if s := p.ResetScore(withRatio(base, 0.30)).Score; s != 0 {
		t.Errorf("healthy-floor score = %.3f, want 0", s)
	}
	if s := p.ResetScore(withRatio(base, 0.0)).Score; s != 1 {
		t.Errorf("stale-floor score = %.3f, want 1", s)
	}
}

// TestResetScoreCooldownReportsPressure proves the cooldown HOLDS the bit but still reports the
// score, so the held pressure is visible in a shadow log (not silently zeroed).
func TestResetScoreCooldownReportsPressure(t *testing.T) {
	p := DefaultResetPolicy()
	st := CacheHealthState{ObservedTurns: 10, RecentReadRatio: 0.02, HasProviderSignal: true, TurnsSinceReset: 0}
	d := p.ResetScore(st)
	if d.ShouldReset {
		t.Fatal("cooldown must hold the reset bit")
	}
	if d.Reason != ResetReasonCooldown {
		t.Fatalf("reason = %q, want cooldown", d.Reason)
	}
	if d.Score <= 0 {
		t.Fatalf("cooldown score = %.3f, want > 0 (the held pressure must be visible)", d.Score)
	}
}

// TestResetScoreCutByDefaultPosture is the safety assertion: across a sweep of states, the only
// states that recommend a reset are ones with a provider signal, enough turns, a sub-healthy
// ratio, AND an elapsed cooldown. Nothing resets on a guess.
func TestResetScoreCutByDefaultPosture(t *testing.T) {
	p := DefaultResetPolicy()
	for turns := 0; turns <= 12; turns += 2 {
		for _, ratio := range []float64{0.0, 0.05, 0.2, 0.5, 0.95} {
			for _, sig := range []bool{false, true} {
				for _, since := range []int{0, 4, 100} {
					st := CacheHealthState{ObservedTurns: turns, RecentReadRatio: ratio, HasProviderSignal: sig, TurnsSinceReset: since}
					d := p.ResetScore(st)
					if !d.ShouldReset {
						continue
					}
					// A reset was recommended -- assert every precondition held.
					if !sig || turns < p.MinObservedTurns || ratio >= p.HealthyReadRatio || since < p.CooldownTurns {
						t.Errorf("reset recommended on a guess: turns=%d ratio=%.2f sig=%v since=%d (reason %s)",
							turns, ratio, sig, since, d.Reason)
					}
				}
			}
		}
	}
}

func withRatio(s CacheHealthState, r float64) CacheHealthState {
	s.RecentReadRatio = r
	return s
}
