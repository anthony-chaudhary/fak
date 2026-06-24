package rsiloop

import "testing"

// observedSteps is a fixed candidate script: KEEP, KEEP, REVERT (no gain over the
// running baseline), KEEP — enough to exercise the observer across both verdicts.
func observedSteps() []fakeStep {
	return []fakeStep{
		{label: "c1", metric: 1, green: true, clean: true},
		{label: "c2", metric: 2, green: true, clean: true},
		{label: "c3", metric: 0, green: true, clean: true},
		{label: "c4", metric: 3, green: true, clean: true},
	}
}

// TestRunObserved_ObservesEveryVerdictWithoutChangingIt is the #588 guarantee: the
// telemetry observer is invoked once per produced row and sees exactly the journaled
// row, AND the loop's keep/revert outcome is identical with and without the observer —
// observing a verdict can never re-gate it.
func TestRunObserved_ObservesEveryVerdictWithoutChangingIt(t *testing.T) {
	h := fakeHarness("fake_kpi", false, 0, "fake@0", observedSteps())

	plain, err := Run(h, nil, 0, 0) // no observer — the authority
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	var seen []Row
	observed, err := RunObserved(h, nil, 0, 0, func(r Row) { seen = append(seen, r) })
	if err != nil {
		t.Fatalf("RunObserved: %v", err)
	}

	if len(seen) != len(observed.Rows) {
		t.Fatalf("observer saw %d rows, loop produced %d", len(seen), len(observed.Rows))
	}
	if plain.Kept != observed.Kept || plain.Cycles != observed.Cycles || plain.Final != observed.Final {
		t.Fatalf("observer changed the outcome: plain{kept=%d cycles=%d final=%s} observed{kept=%d cycles=%d final=%s}",
			plain.Kept, plain.Cycles, plain.Final, observed.Kept, observed.Cycles, observed.Final)
	}
	for i := range plain.Rows {
		if plain.Rows[i].Decision != observed.Rows[i].Decision || plain.Rows[i].Kept != observed.Rows[i].Kept {
			t.Errorf("row %d verdict differs with vs without observer: %s/%v vs %s/%v", i,
				plain.Rows[i].Decision, plain.Rows[i].Kept, observed.Rows[i].Decision, observed.Rows[i].Kept)
		}
		if seen[i].Cycle != observed.Rows[i].Cycle || seen[i].Decision != observed.Rows[i].Decision {
			t.Errorf("row %d: observer saw %s/cycle%d, journal got %s/cycle%d", i,
				seen[i].Decision, seen[i].Cycle, observed.Rows[i].Decision, observed.Rows[i].Cycle)
		}
	}
}

// TestRun_NilObserverIsRunObserved confirms the delegation: Run is exactly
// RunObserved with no observer, so existing callers keep their behavior.
func TestRun_NilObserverIsRunObserved(t *testing.T) {
	h := fakeHarness("fake_kpi", false, 0, "fake@0", observedSteps())
	a, err := Run(h, nil, 0, 0)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	b, err := RunObserved(h, nil, 0, 0, nil)
	if err != nil {
		t.Fatalf("RunObserved: %v", err)
	}
	if a.Kept != b.Kept || len(a.Rows) != len(b.Rows) || a.Final != b.Final {
		t.Fatalf("Run and RunObserved(nil) diverged: %+v vs %+v", a, b)
	}
}
