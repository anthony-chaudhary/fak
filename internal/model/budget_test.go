package model

import (
	"runtime"
	"strconv"
	"testing"
)

func TestResolveBudgetWorkers(t *testing.T) {
	cases := []struct {
		name        string
		envWorkers  string
		envBudget   string
		cores       int
		wantWorkers int
		wantSource  string
	}{
		// FAK_WORKERS — absolute override, unchanged historical behavior.
		{"workers absolute", "8", "", 32, 8, "FAK_WORKERS=8"},
		{"workers wins over budget", "4", "0.5", 32, 4, "FAK_WORKERS=4"},
		{"workers serial reference", "1", "", 32, 1, "FAK_WORKERS=1"},
		{"workers garbage falls through", "abc", "", 32, 32, "default(GOMAXPROCS)"},
		{"workers zero falls through", "0", "", 32, 32, "default(GOMAXPROCS)"},

		// FAK_BUDGET — fractional, portable across machine widths.
		{"half of 32", "", "0.5", 32, 16, "FAK_BUDGET=0.5"},
		{"three-quarters of 32", "", "0.75", 32, 24, "FAK_BUDGET=0.75"},
		{"full of 32", "", "1.0", 32, 32, "FAK_BUDGET=1.0"},
		{"half of 8 — same fraction, smaller box", "", "0.5", 8, 4, "FAK_BUDGET=0.5"},
		{"quarter of 8 — the portable '<=8 of 32'", "", "0.25", 8, 2, "FAK_BUDGET=0.25"},

		// Percent forms.
		{"percent 75 no sign", "", "75", 32, 24, "FAK_BUDGET=75"},
		{"percent 75 with sign", "", "75%", 32, 24, "FAK_BUDGET=75%"},
		{"percent 100", "", "100", 8, 8, "FAK_BUDGET=100"},

		// Rounding (half-up) and the floor of 1.
		{"round half up", "", "0.5", 33, 17, "FAK_BUDGET=0.5"}, // 16.5 -> 17
		{"tiny budget floors at 1", "", "0.01", 32, 1, "FAK_BUDGET=0.01"},
		{"tiny percent floors at 1", "", "1%", 32, 1, "FAK_BUDGET=1%"},

		// A bare value >1 is a PERCENT by the documented rule (75 -> 75%), so 1.5 -> 1.5%
		// -> floors to 1, NOT rejected. This is the consistent reading of "75 means 75%".
		{"bare 1.5 is 1.5 percent", "", "1.5", 32, 1, "FAK_BUDGET=1.5"},

		// Out-of-range / garbage budget falls through to default.
		{"budget zero", "", "0", 32, 32, "default(GOMAXPROCS)"},
		{"budget negative", "", "-1", 32, 32, "default(GOMAXPROCS)"},
		{"budget over 100 percent", "", "150", 32, 32, "default(GOMAXPROCS)"},
		{"budget over 100pct with sign", "", "150%", 32, 32, "default(GOMAXPROCS)"},
		{"budget non-numeric", "", "lots", 32, 32, "default(GOMAXPROCS)"},

		// Nothing set — default.
		{"empty both", "", "", 32, 32, "default(GOMAXPROCS)"},

		// Degenerate machine width is clamped to 1.
		{"zero cores clamps", "", "0.5", 0, 1, "FAK_BUDGET=0.5"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotW, gotSrc := resolveBudgetWorkers(tc.envWorkers, tc.envBudget, tc.cores, nil)
			if gotW != tc.wantWorkers {
				t.Errorf("workers = %d, want %d", gotW, tc.wantWorkers)
			}
			if gotSrc != tc.wantSource {
				t.Errorf("source = %q, want %q", gotSrc, tc.wantSource)
			}
		})
	}
}

// TestResolveBudgetNeverExceedsCores guards the invariant the whole feature rests on:
// a budget can only ever REDUCE the core footprint, never let a bench grab more cores
// than the machine has (which would oversubscribe a box that is also running agents).
func TestResolveBudgetNeverExceedsCores(t *testing.T) {
	for _, cores := range []int{1, 2, 8, 16, 32} {
		for _, b := range []string{"0.1", "0.5", "0.75", "1.0", "50", "100%", "999"} {
			w, _ := resolveBudgetWorkers("", b, cores, nil)
			if w < 1 || w > cores {
				t.Errorf("cores=%d budget=%s -> workers=%d (must be in [1,%d])", cores, b, w, cores)
			}
		}
	}
}

// TestResolveBudgetNotifiesOnGarbage confirms a malformed FAK_BUDGET surfaces a note
// (so a typo doesn't silently run at full width) while a valid one stays quiet.
func TestResolveBudgetNotifiesOnGarbage(t *testing.T) {
	var notes []string
	notify := func(s string) { notes = append(notes, s) }

	resolveBudgetWorkers("", "0.5", 32, notify)
	if len(notes) != 0 {
		t.Errorf("valid budget should not notify, got %v", notes)
	}

	resolveBudgetWorkers("", "bogus", 32, notify)
	if len(notes) != 1 {
		t.Fatalf("garbage budget should notify exactly once, got %d notes: %v", len(notes), notes)
	}
}

// TestSetWorkerBudget covers the post-init flag path: it must re-resolve numWorkers
// against the live core count, reject out-of-range budgets without mutating, and stamp a
// "-budget" source. It mutates package vars, so it saves and restores them.
func TestSetWorkerBudget(t *testing.T) {
	savedW, savedSrc := numWorkers, workerBudgetSource
	t.Cleanup(func() { numWorkers, workerBudgetSource = savedW, savedSrc })

	cores := runtime.GOMAXPROCS(0)

	if err := SetWorkerBudget(0.5); err != nil {
		t.Fatalf("SetWorkerBudget(0.5) errored: %v", err)
	}
	wantHalf, _ := budgetToWorkers(0.5, cores)
	if NumWorkers() != wantHalf {
		t.Errorf("after 0.5: NumWorkers()=%d, want %d (cores=%d)", NumWorkers(), wantHalf, cores)
	}
	if WorkerBudget() != "-budget=0.5" {
		t.Errorf("source=%q, want -budget=0.5", WorkerBudget())
	}

	// Percent form via the flag path (a bench may pass 75 meaning 75%).
	if err := SetWorkerBudget(75); err != nil {
		t.Fatalf("SetWorkerBudget(75) errored: %v", err)
	}
	want75, _ := budgetToWorkers(75, cores)
	if NumWorkers() != want75 {
		t.Errorf("after 75: NumWorkers()=%d, want %d", NumWorkers(), want75)
	}

	// Out-of-range leaves the count untouched and returns an error.
	before := NumWorkers()
	for _, bad := range []float64{0, -1, 150} {
		if err := SetWorkerBudget(bad); err == nil {
			t.Errorf("SetWorkerBudget(%v) should error", bad)
		}
		if NumWorkers() != before {
			t.Errorf("SetWorkerBudget(%v) mutated count to %d despite error", bad, NumWorkers())
		}
	}
}

// TestBudgetEnvAndFlagAgree pins that the env path and the flag path resolve the SAME
// worker count for the same budget — the single-source-of-truth budgetToWorkers floor.
func TestBudgetEnvAndFlagAgree(t *testing.T) {
	for _, cores := range []int{1, 4, 8, 32} {
		for _, frac := range []float64{0.25, 0.5, 0.75, 1.0} {
			envW, _ := resolveBudgetWorkers("", strconv.FormatFloat(frac, 'g', -1, 64), cores, nil)
			flagW, ok := budgetToWorkers(frac, cores)
			if !ok || envW != flagW {
				t.Errorf("cores=%d frac=%v: env=%d flag=%d (ok=%v)", cores, frac, envW, flagW, ok)
			}
		}
	}
}
