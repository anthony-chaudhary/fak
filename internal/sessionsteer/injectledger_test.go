package sessionsteer

import (
	"reflect"
	"testing"
)

// TestInjectionLedgerPermittedTurnIndices is the headline table (#5922): drive N synthetic
// completed turns through one rule in each mode, fire whenever the ledger allows, and
// assert the EXACT turn indices at which injection was permitted — including that stream
// chunks inside a turn never advance the counter.
func TestInjectionLedgerPermittedTurnIndices(t *testing.T) {
	const turns = 25
	cases := []struct {
		name      string
		mode      RepeatMode
		gap       int
		chunks    int   // stream chunks simulated inside every turn
		wantFires []int // completed-turn counts at which the rule fires
	}{
		{
			// once: exactly one fire, at the first opportunity, regardless of chunks.
			name: "once fires exactly once", mode: RepeatOnce, gap: 0, chunks: 7,
			wantFires: []int{0},
		},
		{
			// after-gap, default gap 10: fires at 0, then every 10 completed turns.
			name: "after-gap default gap", mode: RepeatAfterGap, gap: 0, chunks: 3,
			wantFires: []int{0, 10, 20},
		},
		{
			// after-gap, explicit small gap: the cadence follows the configured gap.
			name: "after-gap configurable gap 4", mode: RepeatAfterGap, gap: 4, chunks: 0,
			wantFires: []int{0, 4, 8, 12, 16, 20, 24},
		},
		{
			// Unknown mode normalizes fail-closed to once — never unbounded repetition.
			name: "unknown mode fails closed to once", mode: RepeatMode("every-chunk"), gap: 1, chunks: 9,
			wantFires: []int{0},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			l := NewInjectionLedger()
			var fires []int
			for i := 0; i < turns; i++ {
				if l.Allow("rule", tc.mode, tc.gap) {
					fires = append(fires, l.CompletedTurns())
					l.RecordInjection("rule")
				}
				// Chunks stream within the turn; none of them may advance the counter
				// or flip a verdict that the turn boundary hasn't earned.
				for c := 0; c < tc.chunks; c++ {
					l.StreamChunk()
					if got := l.CompletedTurns(); got != i {
						t.Fatalf("chunk advanced completed turns: got %d, want %d", got, i)
					}
				}
				l.TurnEnd()
			}
			if !reflect.DeepEqual(fires, tc.wantFires) {
				t.Fatalf("permitted turn indices = %v, want %v", fires, tc.wantFires)
			}
		})
	}
}

// TestInjectionLedgerChunksNeverAdvance pins the transport contract in isolation: any
// number of stream deltas inside one turn leaves both the counter and every after-gap
// verdict unchanged, and TurnEnd resets the within-turn chunk count.
func TestInjectionLedgerChunksNeverAdvance(t *testing.T) {
	l := NewInjectionLedger()
	l.RecordInjection("r") // injected at turn 0
	for i := 0; i < 100; i++ {
		l.StreamChunk()
	}
	if got := l.CompletedTurns(); got != 0 {
		t.Fatalf("100 chunks advanced completed turns to %d, want 0", got)
	}
	if got := l.ChunksThisTurn(); got != 100 {
		t.Fatalf("ChunksThisTurn = %d, want 100", got)
	}
	if l.Allow("r", RepeatAfterGap, 1) {
		t.Fatal("after-gap(1) rule became eligible from chunks alone; the gap must be counted in completed turns")
	}
	l.TurnEnd()
	if got := l.ChunksThisTurn(); got != 0 {
		t.Fatalf("ChunksThisTurn after TurnEnd = %d, want 0 (reset at the boundary)", got)
	}
	if !l.Allow("r", RepeatAfterGap, 1) {
		t.Fatal("after-gap(1) rule not eligible after one completed turn")
	}
}

// TestInjectionLedgerOnceSuppressesForLifetime pins the once mode across a long horizon:
// after its single record the rule is refused at every later turn of the session.
func TestInjectionLedgerOnceSuppressesForLifetime(t *testing.T) {
	l := NewInjectionLedger()
	if !l.Allow("r", RepeatOnce, 0) {
		t.Fatal("fresh once rule must be allowed its first fire")
	}
	l.RecordInjection("r")
	for i := 0; i < 1000; i++ {
		l.TurnEnd()
		if l.Allow("r", RepeatOnce, 0) {
			t.Fatalf("once rule re-allowed at completed turn %d", l.CompletedTurns())
		}
	}
}

// TestInjectionLedgerAfterGapBoundary pins the boundary in BOTH directions: a re-fire is
// refused at gap-1 completed turns since injection and permitted at exactly gap — for the
// default and for an explicit gap.
func TestInjectionLedgerAfterGapBoundary(t *testing.T) {
	cases := []struct {
		name string
		gap  int // as passed to Allow (0 selects the default)
		eff  int // the effective gap the boundary sits at
	}{
		{name: "default gap 10", gap: 0, eff: DefaultRepeatGap},
		{name: "explicit gap 3", gap: 3, eff: 3},
		{name: "negative gap selects default", gap: -5, eff: DefaultRepeatGap},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			l := NewInjectionLedger()
			l.RecordInjection("r") // injected at turn 0
			for i := 0; i < tc.eff-1; i++ {
				l.TurnEnd()
				if l.Allow("r", RepeatAfterGap, tc.gap) {
					t.Fatalf("re-fire permitted at %d completed turns, before the gap of %d", l.CompletedTurns(), tc.eff)
				}
			}
			l.TurnEnd() // completed turns == eff
			if !l.Allow("r", RepeatAfterGap, tc.gap) {
				t.Fatalf("re-fire refused at exactly %d completed turns; the boundary is inclusive", tc.eff)
			}
		})
	}
}

// TestInjectionLedgerPersistRoundTrip pins persistence and the restored-age decision:
// names round-trip sorted, a restored once rule stays suppressed, and a restored after-gap
// rule waits a FULL fresh gap from the resume point (eligible at exactly gap, not before)
// even if it had nearly earned a re-fire before the session died.
func TestInjectionLedgerPersistRoundTrip(t *testing.T) {
	live := NewInjectionLedger()
	live.RecordInjection("zeta")
	live.RecordInjection("alpha")
	for i := 0; i < DefaultRepeatGap-1; i++ { // one turn short of re-fire eligibility
		live.TurnEnd()
	}
	names := live.InjectedRules()
	if want := []string{"alpha", "zeta"}; !reflect.DeepEqual(names, want) {
		t.Fatalf("InjectedRules = %v, want %v (sorted names only)", names, want)
	}

	resumed := NewInjectionLedger()
	resumed.Restore(names)
	if resumed.Allow("alpha", RepeatOnce, 0) {
		t.Fatal("restored once rule became eligible on resume; it must stay suppressed")
	}
	// Restored age is ZERO: the near-earned 9 turns of the live session do not carry over.
	for i := 0; i < DefaultRepeatGap-1; i++ {
		resumed.TurnEnd()
		if resumed.Allow("zeta", RepeatAfterGap, 0) {
			t.Fatalf("restored after-gap rule eligible at %d fresh turns; it must wait the full gap of %d", resumed.CompletedTurns(), DefaultRepeatGap)
		}
	}
	resumed.TurnEnd()
	if !resumed.Allow("zeta", RepeatAfterGap, 0) {
		t.Fatalf("restored after-gap rule still refused at %d fresh turns", resumed.CompletedTurns())
	}
	// A rule never persisted is untouched by the restore.
	if !resumed.Allow("other", RepeatOnce, 0) {
		t.Fatal("restore suppressed a rule that was never persisted")
	}
}

// TestInjectionLedgerRollback pins the abort/error decision: when a turn dies before
// delivery, Rollback erases the in-memory record so the live session agrees with what a
// resume would rebuild — the rule is eligible again, and it no longer appears in the
// persistence snapshot. Rolling back an unrecorded rule is a no-op.
func TestInjectionLedgerRollback(t *testing.T) {
	l := NewInjectionLedger()
	l.RecordInjection("r")
	if l.Allow("r", RepeatOnce, 0) {
		t.Fatal("recorded rule must be suppressed before rollback")
	}
	l.Rollback("r")
	if !l.Allow("r", RepeatOnce, 0) {
		t.Fatal("rolled-back rule must be eligible again (turn died before delivery)")
	}
	if got := l.InjectedRules(); len(got) != 0 {
		t.Fatalf("rolled-back rule still in persistence snapshot: %v", got)
	}
	l.Rollback("never-recorded") // no-op, must not panic
	// Zero-value ledger: every method is usable without NewInjectionLedger.
	var zero InjectionLedger
	zero.Rollback("x")
	zero.StreamChunk()
	zero.TurnEnd()
	if !zero.Allow("x", RepeatAfterGap, 0) {
		t.Fatal("zero-value ledger refused a rule with no record")
	}
}

// TestInjectionLedgerDeterministic pins purity: the ledger reads no clock and no
// filesystem, so replaying the same event sequence yields the identical verdict sequence.
func TestInjectionLedgerDeterministic(t *testing.T) {
	run := func() []bool {
		l := NewInjectionLedger()
		var verdicts []bool
		for i := 0; i < 30; i++ {
			ok := l.Allow("r", RepeatAfterGap, 7)
			verdicts = append(verdicts, ok)
			if ok {
				l.RecordInjection("r")
			}
			l.StreamChunk()
			l.TurnEnd()
		}
		return verdicts
	}
	if a, b := run(), run(); !reflect.DeepEqual(a, b) {
		t.Fatalf("same event sequence produced different verdicts:\n%v\n%v", a, b)
	}
}

// TestNormalizeRepeatMode pins the fail-closed mapping: only the exact after-gap token
// selects repetition; everything else — including case variants — collapses to once.
func TestNormalizeRepeatMode(t *testing.T) {
	cases := map[string]RepeatMode{
		"once":      RepeatOnce,
		"after-gap": RepeatAfterGap,
		"After-Gap": RepeatOnce,
		"always":    RepeatOnce,
		"":          RepeatOnce,
	}
	for in, want := range cases {
		if got := NormalizeRepeatMode(in); got != want {
			t.Errorf("NormalizeRepeatMode(%q) = %q, want %q", in, got, want)
		}
	}
}
