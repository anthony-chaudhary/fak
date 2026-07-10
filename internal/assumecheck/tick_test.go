package assumecheck

import "testing"

// tickVerdict builds a fresh verdict for one assumption, the shape the loop
// shell hands Tick after gather+adjudicate.
func tickVerdict(id string, o Outcome, reason string) Verdict {
	return Verdict{AssumptionID: id, Level: LevelInfra, Witness: WitnessConfigFlag, Outcome: o, Reason: reason}
}

// TestTickNoTransitionEmitsNoEvents pins the quiet half of the event rule: a
// tick with no HOLDS->VIOLATED edge emits ZERO events, whatever else the
// verdicts did — steady holds, a still-violated premise (the edge already
// fired), recovery back to holds, and a witnessing gap (UNVERIFIABLE/STALE) are
// all row-only.
func TestTickNoTransitionEmitsNoEvents(t *testing.T) {
	cases := []struct {
		name string
		prev map[string]Outcome
		now  []Verdict
	}{
		{
			name: "steady holds",
			prev: map[string]Outcome{"a": OutcomeHolds},
			now:  []Verdict{tickVerdict("a", OutcomeHolds, "still confirmed")},
		},
		{
			name: "still violated does not re-emit",
			prev: map[string]Outcome{"a": OutcomeViolated},
			now:  []Verdict{tickVerdict("a", OutcomeViolated, "still refuted")},
		},
		{
			name: "recovery back to holds",
			prev: map[string]Outcome{"a": OutcomeViolated},
			now:  []Verdict{tickVerdict("a", OutcomeHolds, "recovered")},
		},
		{
			name: "holds to unverifiable is a gap, not an event",
			prev: map[string]Outcome{"a": OutcomeHolds},
			now:  []Verdict{tickVerdict("a", OutcomeUnverifiable, "witness could not run")},
		},
		{
			name: "holds to stale is a gap, not an event",
			prev: map[string]Outcome{"a": OutcomeHolds},
			now:  []Verdict{tickVerdict("a", OutcomeStale, "evidence aged out")},
		},
		{
			name: "empty tick",
			prev: map[string]Outcome{"a": OutcomeHolds},
			now:  nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res := Tick(tc.prev, tc.now)
			if len(res.Events) != 0 {
				t.Fatalf("Tick emitted %d event(s), want 0: %+v", len(res.Events), res.Events)
			}
			if len(res.Rows) != len(tc.now) {
				t.Fatalf("Tick recorded %d row(s), want one per verdict (%d)", len(res.Rows), len(tc.now))
			}
			for _, r := range res.Rows {
				if r.Transition {
					t.Fatalf("row %q marked Transition without an event", r.Verdict.AssumptionID)
				}
			}
		})
	}
}

// TestTickHoldsToViolatedEmitsExactlyOneEvent pins the loud half: a seeded
// HOLDS->VIOLATED transition emits EXACTLY ONE event, carrying the assumption
// id, the closed outcome-class refusal token ASSUMPTION_VIOLATED (#3822 C4), a
// non-empty soft re-anchor payload, and the reversible marker — while the
// untouched sibling stays row-only.
func TestTickHoldsToViolatedEmitsExactlyOneEvent(t *testing.T) {
	prev := map[string]Outcome{
		"seat-config-dir-present": OutcomeHolds,
		"kernel-loop-alive":       OutcomeHolds,
	}
	now := []Verdict{
		tickVerdict("seat-config-dir-present", OutcomeViolated, "seat dir vanished"),
		tickVerdict("kernel-loop-alive", OutcomeHolds, "loop admitting"),
	}
	res := Tick(prev, now)

	if len(res.Rows) != len(now) {
		t.Fatalf("Tick recorded %d row(s), want one per verdict (%d)", len(res.Rows), len(now))
	}
	if len(res.Events) != 1 {
		t.Fatalf("Tick emitted %d event(s), want exactly 1: %+v", len(res.Events), res.Events)
	}
	ev := res.Events[0]
	if ev.AssumptionID != "seat-config-dir-present" {
		t.Fatalf("event names %q, want the transitioned assumption", ev.AssumptionID)
	}
	if ev.RefusalReason != "ASSUMPTION_VIOLATED" {
		t.Fatalf("event refusal reason = %q, want the closed outcome-class token ASSUMPTION_VIOLATED", ev.RefusalReason)
	}
	if ev.Reanchor == "" {
		t.Fatal("event carries no re-anchor payload")
	}
	if !ev.Reversible {
		t.Fatal("event not marked reversible — the ladder has no destructive rung")
	}
	for _, r := range res.Rows {
		want := r.Verdict.AssumptionID == "seat-config-dir-present"
		if r.Transition != want {
			t.Fatalf("row %q Transition=%v, want %v", r.Verdict.AssumptionID, r.Transition, want)
		}
	}
}

// TestTickUnknownPriorSeedsHolds pins the first-observation seed: an assumption
// absent from prev is judged as if it carried HOLDS, so the first witnessed
// violation of a newly watched premise still emits (the `--once` path), and the
// SECOND tick — judged against the Next map the first returned — does not
// re-emit.
func TestTickUnknownPriorSeedsHolds(t *testing.T) {
	now := []Verdict{tickVerdict("seat-config-dir-present", OutcomeViolated, "seat dir vanished")}

	first := Tick(nil, now)
	if len(first.Events) != 1 {
		t.Fatalf("first tick emitted %d event(s), want 1 (unknown prior seeds HOLDS)", len(first.Events))
	}
	if got := first.Next["seat-config-dir-present"]; got != OutcomeViolated {
		t.Fatalf("Next carries %q, want the fresh outcome %q", got, OutcomeViolated)
	}

	second := Tick(first.Next, now)
	if len(second.Events) != 0 {
		t.Fatalf("second tick re-emitted %d event(s) for the same standing violation, want 0", len(second.Events))
	}
}

// TestTickCarriesForwardUnwitnessedPrev pins the Next map contract: an
// assumption prev knows but this tick did not re-witness keeps its recorded
// outcome, so a skipped row never silently resets to the seeded default.
func TestTickCarriesForwardUnwitnessedPrev(t *testing.T) {
	prev := map[string]Outcome{"a": OutcomeViolated, "b": OutcomeHolds}
	res := Tick(prev, []Verdict{tickVerdict("b", OutcomeHolds, "still confirmed")})
	if got := res.Next["a"]; got != OutcomeViolated {
		t.Fatalf("unwitnessed prev entry reset to %q, want carried-forward %q", got, OutcomeViolated)
	}
}
