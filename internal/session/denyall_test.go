package session

import (
	"strings"
	"testing"
)

// denyall_test.go — the replay fixture for issue #2593: a bounded deny-all loop
// breaker for guarded Codex/fakc `/goal` sessions that spin on the same refused
// host tool. Each test is a REPLAY: it feeds the breaker the per-turn
// observations an audited loop produced and asserts the decision the loop driver
// would have taken. The witness the issue names — "a replay fixture proving
// update_plan DEFAULT_DENY produces a bounded stop rather than unbounded /goal
// continuation" — is TestDenyAllBreakerBoundsRepeatedUpdatePlanDefaultDeny below.

// defaultDenyPlan is the exact per-turn shape the audited fakc loop produced:
// the only proposed tool call was update_plan, refused DEFAULT_DENY (not on the
// floor), no surviving call, no progress. update_plan is host plumbing (its
// schema is plan-state only), so its disposition is DenyAllHostPlumbing.
func defaultDenyPlan() DenyAllObservation {
	return DenyAllObservation{
		Tool:        "update_plan",
		Reason:      "DEFAULT_DENY",
		Progress:    false,
		Disposition: DenyAllHostPlumbing,
	}
}

// TestDenyAllBreakerBoundsRepeatedUpdatePlanDefaultDeny is THE #2593 witness. It
// replays the audited spin — repeated update_plan DEFAULT_DENY, no progress — and
// proves the breaker returns Continue for the first (threshold-1) turns and a
// bounded STOP on the threshold turn, NOT an unbounded sequence of continues.
func TestDenyAllBreakerBoundsRepeatedUpdatePlanDefaultDeny(t *testing.T) {
	var b DenyAllBreaker // zero value: Threshold falls back to DenyAllDefaultThreshold (3)
	threshold := DenyAllDefaultThreshold

	for turn := 1; turn <= threshold-1; turn++ {
		v := b.Observe(defaultDenyPlan())
		if !v.Continue || v.Stopped {
			t.Fatalf("turn %d: verdict = {Continue:%v Stopped:%v}, want continue (under threshold %d)", turn, v.Continue, v.Stopped, threshold)
		}
		if v.Consecutive != turn {
			t.Fatalf("turn %d: Consecutive = %d, want %d", turn, v.Consecutive, turn)
		}
	}

	// The threshold turn MUST stop — this is the bounded stop that ends the spin.
	stop := b.Observe(defaultDenyPlan())
	if stop.Continue || !stop.Stopped {
		t.Fatalf("threshold turn: verdict = {Continue:%v Stopped:%v}, want bounded STOP", stop.Continue, stop.Stopped)
	}
	if stop.Reason != ReasonDenyAllLoop {
		t.Fatalf("stop Reason = %q, want %q", stop.Reason, ReasonDenyAllLoop)
	}
	if stop.Consecutive != threshold {
		t.Fatalf("stop Consecutive = %d, want %d", stop.Consecutive, threshold)
	}
}

// TestDenyAllBreakerDiagnosticExplainsHarnessDialectCoverage proves acceptance
// criterion 2: the bounded-stop diagnostic explains that a host-plumbing refusal
// is a harness-dialect COVERAGE problem (not a dangerous action) and names the
// refused tool, the reason, and the floor source.
func TestDenyAllBreakerDiagnosticExplainsHarnessDialectCoverage(t *testing.T) {
	var b DenyAllBreaker
	// Drive to the bounded stop.
	for i := 0; i < DenyAllDefaultThreshold; i++ {
		b.Observe(defaultDenyPlan())
	}
	// The last Observe returned the stop; re-observe one more on the same stuck run
	// to capture a stop verdict with the diagnostic (the breaker stays stopped-past
	// threshold: once at threshold it stops; further identical turns keep depth >= threshold).
	v := b.Observe(defaultDenyPlan())
	if !v.Stopped {
		t.Fatalf("expected a bounded stop past threshold, got Continue=%v", v.Continue)
	}
	d := v.Diagnostic
	if d == "" {
		t.Fatal("bounded stop emitted an empty diagnostic")
	}
	for _, want := range []string{
		"update_plan",                       // refused tool name
		"DEFAULT_DENY",                      // refusal reason
		"host-plumbing",                     // disposition
		"cmd/fak/guard-default-policy.json", // floor source
		"COVERAGE problem",                  // the harness-dialect explanation (acceptance #2)
		"auto-continue",                     // names what it ended
	} {
		if !strings.Contains(d, want) {
			t.Errorf("diagnostic missing %q\ngot:\n%s", want, d)
		}
	}
	// The plumbing diagnostic must NOT tell the operator to disable the guard or to
	// allow-list without care — it must name the coverage fix.
	if strings.Contains(d, "disable fak guard") && !strings.Contains(d, "Do NOT disable fak guard") {
		t.Errorf("plumbing diagnostic appears to recommend disabling the guard:\n%s", d)
	}
}

// TestDenyAllBreakerDangerousDenialFailsClosed proves acceptance criterion 3: a
// genuinely DANGEROUS repeated denial still fails closed and the breaker NEVER
// auto-allows. The bounded stop fires (fail-closed — the loop stops), and the
// diagnostic explicitly warns against allow-listing an effectful tool.
func TestDenyAllBreakerDangerousDenialFailsClosed(t *testing.T) {
	var b DenyAllBreaker
	dangerous := DenyAllObservation{
		Tool:        "shell_command",
		Reason:      "DEFAULT_DENY",
		Progress:    false,
		Disposition: DenyAllEffectful,
	}
	for i := 0; i < DenyAllDefaultThreshold-1; i++ {
		if v := b.Observe(dangerous); !v.Continue {
			t.Fatalf("turn %d: expected continue under threshold, got Stopped=%v", i+1, v.Stopped)
		}
	}
	stop := b.Observe(dangerous)
	if !stop.Stopped {
		t.Fatal("a dangerous repeated denial must still hit the bounded stop (fail-closed), not auto-allow")
	}
	if stop.Reason != ReasonDenyAllLoop {
		t.Fatalf("dangerous stop Reason = %q, want %q", stop.Reason, ReasonDenyAllLoop)
	}
	// The effectful diagnostic must warn against allow-listing and name the arg rules.
	for _, want := range []string{"shell_command", "effectful", "do NOT allow-list", "dangerous-command"} {
		if !strings.Contains(stop.Diagnostic, want) {
			t.Errorf("dangerous diagnostic missing %q\n got:\n%s", want, stop.Diagnostic)
		}
	}
}

// TestDenyAllBreakerPosixPolicyBlockDiagnosticNamesTheSpelling covers the surfaces
// that carry `rm -rf`. Bash used to fall through to the generic branch, so the one
// surface where the scratchpad route is hardest to reach was the one told the least
// about it — it named no delete alternative and no scratchpad root at all.
//
// It also pins the SPELLING, which is the difference between a remedy and a loop:
// the carve-out resolves on a forward-slash path and not on a Windows-spelled one,
// because the POSIX tokenizer reads `\` as an escape (pinned from the deciding side
// by adjudicator's TestScratchCarveOutSpellingAsymmetry). An agent that re-targets
// into its scratchpad using backslashes is refused a second time, and a remedy that
// silently fails on the retry is indistinguishable from no remedy.
func TestDenyAllBreakerPosixPolicyBlockDiagnosticNamesTheSpelling(t *testing.T) {
	for _, tool := range []string{"Bash", "shell_command", "functions.shell_command"} {
		t.Run(tool, func(t *testing.T) {
			var b DenyAllBreaker
			obs := DenyAllObservation{
				Tool:        tool,
				Reason:      "POLICY_BLOCK",
				Progress:    false,
				Disposition: DenyAllEffectful,
			}
			var stop DenyAllVerdict
			for i := 0; i < DenyAllDefaultThreshold; i++ {
				stop = b.Observe(obs)
			}
			if !stop.Stopped {
				t.Fatalf("repeated %s POLICY_BLOCK must hit the bounded stop", tool)
			}
			d := stop.Diagnostic
			for _, want := range []string{
				"strictly INSIDE a declared scratchpad root",
				"not the root itself",
				"outside such a root recursive/forced deletes stay operator-only",
				// The spelling, without which the advertised retry fails silently.
				"FORWARD slashes",
				`C:\scratch\sess\build`,
				"C:scratchsessbuild",
			} {
				if !strings.Contains(d, want) {
					t.Errorf("%s POLICY_BLOCK diagnostic missing %q\n got:\n%s", tool, want, d)
				}
			}
			// The generic branch is what Bash used to get: no delete route at all.
			if !strings.Contains(d, "rm <file>") {
				t.Errorf("%s POLICY_BLOCK diagnostic names no exact-delete alternative:\n%s", tool, d)
			}
		})
	}
}

// The PowerShell surface has no escaping trap, so it must NOT carry the POSIX
// spelling note — advice that does not apply is noise an agent has to rule out.
func TestDenyAllBreakerPowerShellDiagnosticOmitsPosixSpelling(t *testing.T) {
	var b DenyAllBreaker
	obs := DenyAllObservation{
		Tool:        "PowerShell",
		Reason:      "POLICY_BLOCK",
		Progress:    false,
		Disposition: DenyAllEffectful,
	}
	var stop DenyAllVerdict
	for i := 0; i < DenyAllDefaultThreshold; i++ {
		stop = b.Observe(obs)
	}
	if strings.Contains(stop.Diagnostic, "FORWARD slashes") {
		t.Errorf("PowerShell diagnostic carries a POSIX-only spelling note:\n%s", stop.Diagnostic)
	}
}

// TestDenyAllBreakerPowerShellPolicyBlockDiagnosticIsActionable replays the
// Windows destructive-command churn shape: PowerShell repeatedly proposes the
// same Remove-Item -Recurse/-Force style call, the policy floor refuses it with a
// POLICY_BLOCK and an explicit sanctioned alternative, and auto-continuation must
// end with a diagnostic that preserves that alternative instead of mislabeling the
// reason as DEFAULT_DENY / missing allow-list coverage.
func TestDenyAllBreakerPowerShellPolicyBlockDiagnosticIsActionable(t *testing.T) {
	var b DenyAllBreaker
	obs := DenyAllObservation{
		Tool:        "PowerShell",
		Reason:      "POLICY_BLOCK",
		Progress:    false,
		Disposition: DenyAllEffectful,
	}
	var stop DenyAllVerdict
	for i := 0; i < DenyAllDefaultThreshold; i++ {
		stop = b.Observe(obs)
	}
	if !stop.Stopped {
		t.Fatal("repeated PowerShell POLICY_BLOCK must hit the bounded stop")
	}
	d := stop.Diagnostic
	for _, want := range []string{
		"PowerShell",
		"POLICY_BLOCK",
		"explicit policy rule blocked this call",
		"sanctioned alternative",
		"do NOT re-propose",
		"Remove-Item <file>",
		// The operator-only claim survives, but only where it is TRUE: outside a
		// declared scratchpad root.
		"outside such a root recursive/forced deletes stay operator-only",
		// ...and the route the floor actually grants must be named, or the last
		// line the agent reads before giving up denies a remedy that exists.
		"strictly INSIDE a declared scratchpad root",
		"not the root itself",
	} {
		if !strings.Contains(d, want) {
			t.Errorf("PowerShell POLICY_BLOCK diagnostic missing %q\n got:\n%s", want, d)
		}
	}
	// The breaker must never read as a blanket ban on the carve-out the adjudicator
	// grants. This is the exact sentence that shipped, and it is the reason an agent
	// cleaning its own scratch escalated to an operator who had nothing to approve.
	if strings.Contains(d, "aside; recursive/forced deletes stay operator-only") {
		t.Errorf("diagnostic still denies the scratchpad route the floor admits:\n%s", d)
	}
	if strings.Contains(d, "DEFAULT_DENY = not on the capability floor") {
		t.Fatalf("POLICY_BLOCK diagnostic must not carry DEFAULT_DENY semantics:\n%s", d)
	}
}

// TestDenyAllBreakerProgressResetsTheRun proves a stuck streak resets the instant
// the loop makes useful progress (a surviving call or meaningful work) — the
// breaker never false-stops a loop that is actually moving, only one spinning on
// the same refused call.
func TestDenyAllBreakerProgressResetsTheRun(t *testing.T) {
	var b DenyAllBreaker
	b.Observe(defaultDenyPlan())
	b.Observe(defaultDenyPlan()) // count == 2, one short of the default threshold
	// A turn that makes progress resets the streak to 0.
	v := b.Observe(DenyAllObservation{
		Tool:        "update_plan",
		Reason:      "DEFAULT_DENY",
		Progress:    true, // a surviving call elsewhere — the loop is not stuck
		Disposition: DenyAllHostPlumbing,
	})
	if !v.Continue || v.Consecutive != 0 {
		t.Fatalf("progress turn: verdict = {Continue:%v Consecutive:%d}, want continue/0 (reset)", v.Continue, v.Consecutive)
	}
	// The next stuck turn seeds a fresh run at 1, not 3.
	v = b.Observe(defaultDenyPlan())
	if !v.Continue || v.Consecutive != 1 {
		t.Fatalf("post-reset stuck turn: Consecutive = %d, want 1 (re-seeded, not continued)", v.Consecutive)
	}

	// Also verify todowrite progress resets the streak.
	b.Observe(defaultDenyPlan())
	v = b.Observe(DenyAllObservation{
		Tool:        "todowrite",
		Reason:      "DEFAULT_DENY",
		Progress:    true,
		Disposition: DenyAllHostPlumbing,
	})
	if !v.Continue || v.Consecutive != 0 {
		t.Fatalf("todowrite progress turn: verdict = {Continue:%v Consecutive:%d}, want continue/0 (reset)", v.Continue, v.Consecutive)
	}
}

// TestDenyAllBreakerCleanTurnResetsTheRun proves a pure-text / no-refused-call
// turn also resets the streak (the Claude Stop-hook give-up contract relies on a
// pure-text turn clearing the gauge; this breaker shares that semantics).
func TestDenyAllBreakerCleanTurnResetsTheRun(t *testing.T) {
	var b DenyAllBreaker
	b.Observe(defaultDenyPlan())
	b.Observe(defaultDenyPlan())
	v := b.Observe(DenyAllObservation{Tool: "", Reason: "", Progress: false}) // clean turn: no refused call
	if !v.Continue || v.Consecutive != 0 {
		t.Fatalf("clean turn: Consecutive = %d, want 0 (reset)", v.Consecutive)
	}
}

// TestDenyAllBreakerFlippedToolOrReasonReseeds proves the "unchanged" half of the
// stuck test: a run only counts while BOTH the tool and the reason are identical.
// A flap between two different refused tools (or two reasons) is NOT the same spin
// and re-seeds at 1, so two interleaved refusals never add up to a stop.
func TestDenyAllBreakerFlippedToolOrReasonReseeds(t *testing.T) {
	t.Run("different tool re-seeds", func(t *testing.T) {
		var b DenyAllBreaker
		b.Observe(defaultDenyPlan())       // update_plan, count 1
		v := b.Observe(DenyAllObservation{ // tool_search — different tool
			Tool: "tool_search", Reason: "DEFAULT_DENY",
			Progress: false, Disposition: DenyAllHostPlumbing,
		})
		if v.Consecutive != 1 {
			t.Fatalf("different tool: Consecutive = %d, want 1 (re-seeded)", v.Consecutive)
		}
	})
	t.Run("different reason re-seeds", func(t *testing.T) {
		var b DenyAllBreaker
		b.Observe(defaultDenyPlan())       // DEFAULT_DENY, count 1
		v := b.Observe(DenyAllObservation{ // POLICY_BLOCK — different reason
			Tool: "update_plan", Reason: "POLICY_BLOCK",
			Progress: false, Disposition: DenyAllHostPlumbing,
		})
		if v.Consecutive != 1 {
			t.Fatalf("different reason: Consecutive = %d, want 1 (re-seeded)", v.Consecutive)
		}
	})
}

// TestDenyAllBreakerResetClearsStaleRun proves Reset drops an in-progress streak at
// an objective boundary, so a breaker carried across /goal objectives cannot
// false-stop a fresh goal on a run the previous goal accrued.
func TestDenyAllBreakerResetClearsStaleRun(t *testing.T) {
	var b DenyAllBreaker
	b.Observe(defaultDenyPlan())
	b.Observe(defaultDenyPlan()) // count == 2, one short of the default threshold
	b.Reset()
	// After Reset the next stuck turn seeds a fresh run at 1, not 3.
	v := b.Observe(defaultDenyPlan())
	if !v.Continue || v.Consecutive != 1 {
		t.Fatalf("post-reset stuck turn: Consecutive = %d, want 1 (fresh run, not %d)", v.Consecutive, v.Consecutive)
	}
	// Reset is idempotent and a no-op on an already-clear breaker.
	b.Reset()
	b.Reset()
	if v := b.Observe(defaultDenyPlan()); v.Consecutive != 1 {
		t.Fatalf("idempotent reset: Consecutive = %d, want 1", v.Consecutive)
	}
}

// TestDenyAllBreakerCustomThreshold proves the bound is configurable (a tighter
// threshold stops sooner) and that the verdict carries the effective threshold so
// a renderer never has to re-derive it.
func TestDenyAllBreakerCustomThreshold(t *testing.T) {
	b := DenyAllBreaker{Threshold: 2}
	if v := b.Observe(defaultDenyPlan()); !v.Continue || v.Consecutive != 1 || v.Threshold != 2 {
		t.Fatalf("turn 1 @threshold2: verdict = %+v, want Continue/1/threshold2", v)
	}
	v := b.Observe(defaultDenyPlan())
	if v.Continue || !v.Stopped || v.Consecutive != 2 {
		t.Fatalf("turn 2 @threshold2: verdict = %+v, want bounded STOP at 2", v)
	}
}

// TestDenyAllBreakerZeroValueBondsAtDefault proves an unconfigured (zero-value)
// breaker still bounds at DenyAllDefaultThreshold rather than never firing — the
// safe default for a loop driver that forgets to set Threshold.
func TestDenyAllBreakerZeroValueBondsAtDefault(t *testing.T) {
	var b DenyAllBreaker
	for i := 0; i < DenyAllDefaultThreshold-1; i++ {
		if v := b.Observe(defaultDenyPlan()); !v.Continue {
			t.Fatalf("zero-value breaker stopped early at turn %d", i+1)
		}
	}
	if v := b.Observe(defaultDenyPlan()); !v.Stopped {
		t.Fatal("zero-value breaker did NOT bound at the default threshold — a forgotten config must still bound")
	}
}

// TestDenyAllBreakerNeverAutoAllows is the structural fail-closed guard: across
// every observation shape — under threshold, at threshold, past threshold, host
// plumbing, effectful — the breaker NEVER returns a verdict that lifts a refusal.
// (There is no "Allow" field and no path that would; this test pins that absence so
// a future edit cannot quietly add one.)
func TestDenyAllBreakerNeverAutoAllows(t *testing.T) {
	cases := []struct {
		name string
		obs  DenyAllObservation
	}{
		{"plumbing under threshold", defaultDenyPlan()},
		{"effectful under threshold", DenyAllObservation{Tool: "shell_command", Reason: "DEFAULT_DENY", Disposition: DenyAllEffectful}},
		{"clean turn", DenyAllObservation{Tool: "", Reason: ""}},
		{"progress turn", DenyAllObservation{Tool: "update_plan", Reason: "DEFAULT_DENY", Progress: true}},
		{"progress turn todowrite", DenyAllObservation{Tool: "todowrite", Reason: "DEFAULT_DENY", Progress: true, Disposition: DenyAllHostPlumbing}},
	}
	for _, tc := range cases {
		var b DenyAllBreaker
		// Drive well past any threshold so a stop, if it were going to auto-allow,
		// would have done so.
		var v DenyAllVerdict
		for i := 0; i < DenyAllDefaultThreshold+3; i++ {
			v = b.Observe(tc.obs)
		}
		// The breaker only chooses between Continue and Stopped. There is no third
		// "allowed" outcome, and a stop never carries an auto-allow.
		if v.Continue == v.Stopped {
			t.Errorf("%s: verdict Continue==Stopped (%v/%v) — must be exactly one", tc.name, v.Continue, v.Stopped)
		}
	}
}
