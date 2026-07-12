package resume

import "testing"

// TestPhaseClassifierSharedVocabulary is the shared table test #3801's DoD asked for and
// #4333 lands: phaseIsLaunchToken is the ONE typed helper both Attempt.IsLaunch (accounting)
// and the watchdog-status fold (MTTR) consult, so the two readers return identical
// launch/non-launch verdicts for every NON-EMPTY phase token. The fold's launch arm calls
// phaseIsLaunchToken directly (watchdog_status.go), and IsLaunch delegates to it, so any
// future edit that made a named token drift between the two readers reddens here.
func TestPhaseClassifierSharedVocabulary(t *testing.T) {
	// The full phase vocabulary any writer appends to a resume ledger a launch scanner might
	// read, plus a token no writer emits (schema drift), each with its single launch verdict.
	// Only the explicit launch tokens are launches; everything else is bookkeeping.
	cases := []struct {
		phase  string
		launch bool
	}{
		{"launched", true},
		{"resumed", true},
		{"LAUNCHED", true}, // classification is case-insensitive
		{"Resumed", true},
		{"queued", false},
		{"detected", false},
		{"auto_resume", false}, // a cycle-start, not a fired spawn
		{"deferred", false},
		{"considered", false},
		{"skipped", false},
		{"gate_fail_open", false},
		{"broker_denied", false}, // the broker refused; nothing launched (TestBrokerDeniedIsNotALaunch)
		{"status", false},
		{"tick", false},
		{"snapshot", false},
		{"progress", false},
		{"settled", false},
		{"operator_settled", false},
		{"consolidated", false},
		{"rearm", false},
		{"trajectory_decision", false},
		{"phase_unknown", false},
		{"some_novel_bookkeeping_token", false}, // an unrecognized token is NOT a launch
	}
	for _, c := range cases {
		if got := phaseIsLaunchToken(c.phase); got != c.launch {
			t.Errorf("phaseIsLaunchToken(%q) = %v, want %v", c.phase, got, c.launch)
		}
		// IsLaunch delegates to the shared classifier for every non-empty token. (A settled
		// Action row is a separate override path, covered by TestCountAttemptsAndLastLaunch.)
		if got := (Attempt{Phase: c.phase}).IsLaunch(); got != c.launch {
			t.Errorf("Attempt{Phase:%q}.IsLaunch() = %v, want %v (must match the shared classifier)", c.phase, got, c.launch)
		}
		// The fold consults the SAME classifier: a non-empty token is a fold launch iff it is
		// a launch token, because normalizeWatchdogPhase is identity (lowercased) on non-empty
		// input. This is the exact expression the fold's launch arm evaluates.
		if got := phaseIsLaunchToken(normalizeWatchdogPhase(c.phase)); got != c.launch {
			t.Errorf("fold launch verdict for %q = %v, want %v (must match IsLaunch)", c.phase, got, c.launch)
		}
	}

	// The ONE deliberate divergence: a phase-less row. Accounting counts it as a launch (the
	// pre-phase legacy spawn — the 114 production rows that must keep burning attempt budget),
	// while the fold folds it to phase_unknown, a non-launch, so it never mints a phantom
	// launched_unproven MTTR row (#3801). The two readers answer different questions; forcing
	// identical empty-phase verdicts would regress the give-up cap / LAUNCH_SPACING_FLOOR.
	if !(Attempt{Phase: ""}).IsLaunch() {
		t.Error("phase-less Attempt.IsLaunch() = false, want true (a legacy spawn still burns attempt budget)")
	}
	if normalizeWatchdogPhase("") != watchdogPhaseUnknown {
		t.Errorf("normalizeWatchdogPhase(%q) = %q, want %q", "", normalizeWatchdogPhase(""), watchdogPhaseUnknown)
	}
	if phaseIsLaunchToken(normalizeWatchdogPhase("")) {
		t.Error("fold launch verdict for a phase-less row = true, want false (phase_unknown is not a launch)")
	}
}
