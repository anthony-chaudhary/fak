package watchdoghealth

import (
	"strings"
	"testing"
)

// auditgate_test.go — the regression witness for #6509: 47 RED findings in the retained
// meta-watchdog audit log while Task Scheduler recorded LastTaskResult=0 every 15 minutes.
// The load-bearing test is TestRedAuditCannotYieldSchedulerResultZero; the rest pin the
// dedupe/age/owner behaviour that turns a forever-appended warning into an aging decision.

// redAuditLog is a trimmed stand-in for the real retained log: a human (non-JSON) audit run
// whose reason list carries hard RED faults alongside AMBER ones. The wrapper that produced
// #6509 captured exactly this text and then exited 0.
const redAuditLog = `==============================================================
 watchdog-watchdog audit (n2/n3)   2026-08-11T19:15:04Z
==============================================================
live registry     : C:\Users\USER\AppData\Local\Fleet\registry
newest ledger     : 412.6 min ago  (STALL threshold 15 min)
supervision tower : 14 tasks | 3 DOWN | 6 latent-Interactive

VERDICT: RED
  - [RED] STALL: newest ledger write 413 min ago (> 15 min) -- watchdog is not ticking
  - [RED] FleetResumeWatchdog DOWN 0x800710E0 (LogonType=Interactive) -- migrate to S4U
  - [RED] FleetSupervisorWatchdog DOWN 0x800710E0 (LogonType=Interactive) -- migrate to S4U
  - [RED] FleetDosDispatchWatchdog DOWN 0x1
  - [AMBER] 6 tower task(s) still LogonType=Interactive (latent 0x800710E0 under RDP): FleetOwnerSeatResume
  - [AMBER] 4 resume(s) launched_unproven (no real model turn after launch)
  - [AMBER] backlog 31 deep and draining only ~4/tick -- needs a live ticking watchdog
  - [AMBER] n3: the auditor's own task is LogonType=Interactive -- it will die the SAME way it is meant to detect.
ACTION : Migrate down tower tasks to S4U (elevated): tools\migrate_fleet_tasks_to_s4u.ps1 -Apply -VerifyRun
`

// TestRedAuditCannotYieldSchedulerResultZero is the #6509 witness. Whatever shape the audit's
// output arrives in — the -Json object or the retained human log — a run carrying a RED
// finding must produce a typed nonzero status, and a scheduler result of 0 over it must be
// reported as a swallowed verdict rather than health.
func TestRedAuditCannotYieldSchedulerResultZero(t *testing.T) {
	cases := map[string][]byte{
		"human log":            []byte(redAuditLog),
		"json report":          []byte(`{"verdict":"RED","reasons":["[RED] FleetResumeWatchdog DOWN 0x800710E0 (LogonType=Interactive) -- migrate to S4U"]}`),
		"json with green head": []byte(`{"verdict":"GREEN","reasons":["[RED] FleetResumeWatchdog DOWN 0x1"]}`),
		"json with no head":    []byte(`{"reasons":["[RED] STALL: newest ledger write 413 min ago (> 15 min) -- watchdog is not ticking"]}`),
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			verdict, findings := ParseAuditReport(raw)
			g := AuditGate(verdict, findings, nil, 1_700_000_000)
			if g.Verdict != AuditRed {
				t.Fatalf("verdict = %s, want RED (findings %+v)", g.Verdict, findings)
			}
			if g.ExitCode != AuditExitRed {
				t.Fatalf("exit code = %d, want %d (RED)", g.ExitCode, AuditExitRed)
			}
			if g.ExitCode == AuditExitGreen {
				t.Fatalf("a RED audit must never produce scheduler result 0")
			}
			if !AuditResultSwallowed(g, 0) {
				t.Fatalf("scheduler result 0 over a RED gate must be reported as swallowed")
			}
			if AuditResultSwallowed(g, g.ExitCode) {
				t.Fatalf("propagating the gate's own exit code must not read as swallowed")
			}
		})
	}
}

// TestAuditExitCodesAreTypedAndDistinct pins the exit contract: only GREEN is 0, RED and AMBER
// are distinct nonzero statuses, and an unknown verdict token falls to a nonzero status rather
// than passing.
func TestAuditExitCodesAreTypedAndDistinct(t *testing.T) {
	if got := AuditExitCode(AuditGreen); got != 0 {
		t.Fatalf("GREEN exit = %d, want 0", got)
	}
	for _, v := range []AuditVerdict{AuditAmber, AuditRed, AuditUnreadable, "", "sort-of-fine", "GREENISH"} {
		if got := AuditExitCode(v); got == 0 {
			t.Fatalf("verdict %q exit = 0, want nonzero", v)
		}
	}
	if AuditExitCode(AuditAmber) == AuditExitCode(AuditRed) {
		t.Fatalf("AMBER and RED must carry distinct statuses")
	}
	if got := ParseAuditVerdict("greenish"); got != AuditUnreadable {
		t.Fatalf("an unrecognized verdict token must be UNREADABLE, got %s", got)
	}
}

// TestSwallowedOutputIsUnreadableNotGreen: a wrapper that captured nothing (or captured only
// the banner) must not be scored as a pass — the failure mode is silence.
func TestSwallowedOutputIsUnreadableNotGreen(t *testing.T) {
	for name, raw := range map[string][]byte{
		"empty":    nil,
		"blank":    []byte("   \n\n"),
		"no marks": []byte("=====\n watchdog-watchdog audit (n2/n3)\n=====\n"),
	} {
		t.Run(name, func(t *testing.T) {
			v, findings := ParseAuditReport(raw)
			if v != AuditUnreadable || len(findings) != 0 {
				t.Fatalf("ParseAuditReport = %s / %d findings, want UNREADABLE / 0", v, len(findings))
			}
			if got := AuditGate(v, findings, nil, 10).ExitCode; got == AuditExitGreen {
				t.Fatalf("an unreadable audit must not exit 0")
			}
		})
	}
}

// TestAuditGateDeduplicatesPersistentFindings is the "stop appending the same warning" half of
// the Done condition: the same standing fault re-observed with a different minute count is ONE
// aging record with a recurrence count, not a second finding.
func TestAuditGateDeduplicatesPersistentFindings(t *testing.T) {
	stall := func(min string) []AuditFinding {
		return ParseAuditFindings([]string{
			"[RED] STALL: newest ledger write " + min + " min ago (> 15 min) -- watchdog is not ticking",
		})
	}
	const t0, t1, t2 = 1_000, 1_900, 2_800

	first := AuditGate(AuditRed, stall("37"), nil, t0)
	if len(first.Ledger) != 1 || len(first.New) != 1 || len(first.Recurring) != 0 {
		t.Fatalf("first sighting must be one NEW record, got %+v", first)
	}
	second := AuditGate(AuditRed, stall("52"), first.Ledger, t1)
	if len(second.Ledger) != 1 {
		t.Fatalf("a re-observed stall must dedupe into one record, got %d: %+v", len(second.Ledger), second.Ledger)
	}
	r := second.Ledger[0]
	if r.Occurrences != 2 || !r.Recurring() {
		t.Fatalf("occurrences = %d, want 2 (recurring)", r.Occurrences)
	}
	if got := r.AgeSeconds(t1); got != t1-t0 {
		t.Fatalf("age = %ds, want %ds", got, t1-t0)
	}
	if !strings.Contains(r.Text, "52 min") {
		t.Fatalf("the record must carry the latest phrasing, got %q", r.Text)
	}
	if len(second.Recurring) != 1 || len(second.New) != 0 {
		t.Fatalf("the second sighting must be recurring, not new: %+v", second)
	}

	// Cleared: the record survives, is reported resolved once, and the gate goes green.
	third := AuditGate(AuditGreen, nil, second.Ledger, t2)
	if third.ExitCode != AuditExitGreen {
		t.Fatalf("a clean run must exit 0, got %d", third.ExitCode)
	}
	if len(third.Resolved) != 1 || !third.Ledger[0].Resolved {
		t.Fatalf("the cleared fault must be reported resolved, got %+v", third)
	}
	if got := third.Ledger[0].AgeSeconds(t2 + 10_000); got != t2-t0 {
		t.Fatalf("a resolved record's age must freeze at resolution, got %d", got)
	}
	// A second clean run does not re-report the same resolution.
	fourth := AuditGate(AuditGreen, nil, third.Ledger, t2+900)
	if len(fourth.Resolved) != 0 {
		t.Fatalf("an already-resolved record must not be re-reported, got %+v", fourth.Resolved)
	}
	// Returning counts a regression instead of filing a brand-new finding.
	fifth := AuditGate(AuditRed, stall("61"), fourth.Ledger, t2+1_800)
	if len(fifth.Ledger) != 1 {
		t.Fatalf("a returning fault must reuse its record, got %d", len(fifth.Ledger))
	}
	if got := fifth.Ledger[0]; got.Regressions != 1 || got.Resolved || got.Occurrences != 3 {
		t.Fatalf("regression bookkeeping wrong: %+v", got)
	}
}

// TestAuditFindingsRepeatedWithinOneRunCountOnce: the retained log's 47 REDs are mostly the
// same few faults repeated by re-runs; one audit run listing a finding twice must still be one
// occurrence.
func TestAuditFindingsRepeatedWithinOneRunCountOnce(t *testing.T) {
	f := ParseAuditFindings([]string{
		"[RED] FleetResumeWatchdog DOWN 0x800710E0 (LogonType=Interactive) -- migrate to S4U",
		"[RED] FleetResumeWatchdog DOWN 0x800710E0 (LogonType=Interactive) -- migrate to S4U",
	})
	g := AuditGate(AuditRed, f, nil, 100)
	if len(g.Ledger) != 1 || g.Ledger[0].Occurrences != 1 {
		t.Fatalf("a finding repeated within one run must count once, got %+v", g.Ledger)
	}
}

// TestAuditFindingKeyCollapsesOnlyNumbers pins the fingerprint: the counters that move every
// tick fall out of the key, while task names and HRESULTs — the tokens that say WHICH fault
// this is — survive, so two different DOWN tasks never collapse into one record.
func TestAuditFindingKeyCollapsesOnlyNumbers(t *testing.T) {
	a := AuditFindingKey("STALL: newest ledger write 37 min ago (> 15 min) -- watchdog is not ticking")
	b := AuditFindingKey("STALL: newest ledger write 412.6 min ago (> 15 min) -- watchdog is not ticking")
	if a != b {
		t.Fatalf("minute counts must not split a standing finding:\n %q\n %q", a, b)
	}
	if strings.Contains(a, "37") || !strings.Contains(a, "#") {
		t.Fatalf("numeric tokens must collapse to #, got %q", a)
	}
	distinct := []string{
		"FleetResumeWatchdog DOWN 0x800710E0",
		"FleetSupervisorWatchdog DOWN 0x800710E0",
		"FleetResumeWatchdog DOWN 0x1",
		"UserSeatDrain-1010 DOWN 0x41306",
	}
	seen := map[string]string{}
	for _, d := range distinct {
		k := AuditFindingKey(d)
		if prev, dup := seen[k]; dup {
			t.Fatalf("%q and %q must not share key %q", prev, d, k)
		}
		seen[k] = d
	}
	// The marker is not part of the identity: the same text with and without it is one key.
	if AuditFindingKey("[RED] FleetResumeWatchdog DOWN 0x1") != AuditFindingKey("FleetResumeWatchdog DOWN 0x1") {
		t.Fatalf("the severity marker must not change the recurrence key")
	}
}

// TestAuditFindingOwnership pins the actionable-owner routing: elevation-shaped faults go to
// the operator, faults a running actor already clears go to the fleet, and an unrecognized
// finding is UNROUTED and still surfaced to a person.
func TestAuditFindingOwnership(t *testing.T) {
	cases := []struct {
		text  string
		owner AuditOwner
	}{
		{"FleetResumeWatchdog DOWN 0x800710E0 (LogonType=Interactive) -- migrate to S4U", AuditOwnerOperator},
		{"6 tower task(s) still LogonType=Interactive (latent 0x800710E0 under RDP): A, B", AuditOwnerOperator},
		{"FleetDosDispatchWatchdog DOWN 0x1", AuditOwnerOperator},
		{"n3 GAP: this audit is not itself scheduled/looped", AuditOwnerOperator},
		{"fak not on PATH -- cannot read the launched-vs-proven witness", AuditOwnerOperator},
		{"STALL: newest ledger write 413 min ago (> 15 min) -- watchdog is not ticking", AuditOwnerFleet},
		{"4 resume(s) launched_unproven (no real model turn after launch)", AuditOwnerFleet},
		{"backlog 31 deep and draining only ~4/tick -- needs a live ticking watchdog", AuditOwnerFleet},
		{"a brand new fault nobody has taught the table about", AuditOwnerUnrouted},
	}
	for _, c := range cases {
		got := NewAuditFinding(AuditRed, c.text)
		if got.Owner != c.owner {
			t.Errorf("owner(%q) = %s, want %s", c.text, got.Owner, c.owner)
		}
		if c.owner != AuditOwnerFleet && !got.Owner.NeedsHuman() {
			t.Errorf("%s must wait on a person", got.Owner)
		}
		if c.owner != AuditOwnerUnrouted && got.Action == "" {
			t.Errorf("a routed finding must name a next move: %q", c.text)
		}
	}
	if AuditOwnerFleet.NeedsHuman() {
		t.Fatalf("a fleet-owned finding must not page")
	}
	// Only the fleet's open findings stay off the needs-human list.
	g := AuditGate(AuditRed, ParseAuditFindings(strings.Split(strings.TrimSpace(`
[RED] FleetResumeWatchdog DOWN 0x800710E0 (LogonType=Interactive) -- migrate to S4U
[RED] STALL: newest ledger write 413 min ago (> 15 min) -- watchdog is not ticking`), "\n")), nil, 500)
	if len(g.NeedsHuman) != 1 {
		t.Fatalf("needs-human = %v, want just the operator-owned DOWN task", g.NeedsHuman)
	}
}

// TestAuditorIndependence pins Done condition 3 as a checkable predicate: an auditor scheduled
// under its own Interactive logon shares the failure mode it detects and is RED, an unscheduled
// auditor is AMBER, and an S4U auditor is no finding at all.
func TestAuditorIndependence(t *testing.T) {
	f, ok := AuditorIndependenceFinding(true, "Interactive")
	if !ok || f.Severity != AuditRed {
		t.Fatalf("an Interactive-logon auditor must be RED, got %+v (ok=%v)", f, ok)
	}
	if AuditExitCode(AuditGate(AuditGreen, []AuditFinding{f}, nil, 1).Verdict) != AuditExitRed {
		t.Fatalf("a self-dependent auditor must redden the gate even under a GREEN header")
	}
	if f2, ok := AuditorIndependenceFinding(true, "InteractiveToken"); !ok || f2.Severity != AuditRed {
		t.Fatalf("InteractiveToken is the same principal class, got %+v (ok=%v)", f2, ok)
	}
	if f3, ok := AuditorIndependenceFinding(false, ""); !ok || f3.Severity != AuditAmber {
		t.Fatalf("an unscheduled auditor must be AMBER, got %+v (ok=%v)", f3, ok)
	}
	for _, lt := range []string{"S4U", "Password", "ServiceAccount"} {
		if f4, ok := AuditorIndependenceFinding(true, lt); ok {
			t.Fatalf("LogonType=%s is orthogonal to the failure mode, got finding %+v", lt, f4)
		}
	}
}

// TestAuditReportLines proves the report carries the four facts the appended log never did:
// age, recurrence, resolution, and the owner.
func TestAuditReportLines(t *testing.T) {
	down := ParseAuditFindings([]string{"[RED] FleetResumeWatchdog DOWN 0x800710E0 (LogonType=Interactive) -- migrate to S4U"})
	g := AuditGate(AuditRed, down, AuditGate(AuditRed, down, nil, 1_000).Ledger, 1_900)
	lines := AuditReportLines(g, 1_900)
	if len(lines) != 1 {
		t.Fatalf("want one report line, got %v", lines)
	}
	for _, want := range []string{"RED", "x2", "age=15m0s", "owner=operator", "FleetResumeWatchdog", "-> elevated:"} {
		if !strings.Contains(lines[0], want) {
			t.Errorf("report line %q missing %q", lines[0], want)
		}
	}
	resolved := AuditReportLines(AuditGate(AuditGreen, nil, g.Ledger, 2_800), 2_800)
	if len(resolved) != 1 || !strings.HasPrefix(resolved[0], "RESOLVED") {
		t.Fatalf("a cleared fault must report a RESOLVED line, got %v", resolved)
	}
}

// liveAuditCapture is a verbatim (whitespace-trimmed) `watchdog_watchdog_audit.ps1 -Json`
// capture taken from the fleet host on 2026-08-12 while reproducing #6509 — the shape
// ConvertTo-Json actually emits, escapes and all. It is here so the gate is proven against the
// real serializer rather than a hand-written idealisation of it.
const liveAuditCapture = `{
    "registry":  "C:\\Users\\USER\\AppData\\Local\\Fleet\\registry",
    "newest_ledger_age_min":  197,
    "tower_total":  22,
    "tower_down":  3,
    "n3_auditor_scheduled":  true,
    "n3_auditor_logontype":  "Interactive",
    "verdict":  "RED",
    "reasons":  [
                    "[RED] STALL: newest ledger write 197 min ago (\u003e 15 min) -- watchdog is not ticking",
                    "[RED] FakGLMGuardGateway18080 DOWN 0x41306",
                    "[RED] FleetIssueDispatch DOWN 0x41306",
                    "[RED] FleetIssueDispatchCodex DOWN 0x41306",
                    "[AMBER] 7 tower task(s) still LogonType=Interactive (latent 0x800710E0 under RDP): FakGuardedCodexWave100, FleetProcResourceGuard",
                    "[AMBER] 3 resume(s) launched_unproven (no real model turn after launch)",
                    "[RED] n3: the auditor\u0027s own task is LogonType=Interactive -- it will be refused (0x800710E0) the SAME way the tasks it audits were, so its silence cannot be read as health."
                ],
    "action":  "Migrate down tower tasks to S4U (elevated)."
}`

// TestGateOnLiveAuditCapture folds the real capture and pins what the scheduler must record:
// exit 3, seven distinct findings deduplicated into seven records, and the three DOWN tasks
// kept apart rather than collapsed by the shared "DOWN 0x41306" phrasing.
func TestGateOnLiveAuditCapture(t *testing.T) {
	verdict, findings := ParseAuditReport([]byte(liveAuditCapture))
	if verdict != AuditRed {
		t.Fatalf("verdict = %s, want RED", verdict)
	}
	g := AuditGate(verdict, findings, nil, 1_754_000_000)
	if g.ExitCode != AuditExitRed {
		t.Fatalf("exit = %d, want %d", g.ExitCode, AuditExitRed)
	}
	// The relation, not a frozen total: every marked reason in the capture becomes exactly one
	// ledger record, and a first run files all of them as new.
	if len(g.Ledger) != len(findings) {
		t.Fatalf("ledger = %d records for %d findings, want one record each: %+v", len(g.Ledger), len(findings), g.Ledger)
	}
	if len(g.New) != len(g.Ledger) || len(g.Recurring) != 0 {
		t.Fatalf("a first run must file every finding as new, got %d new / %d recurring", len(g.New), len(g.Recurring))
	}
	// The DOWN tasks share everything but their name; each must stay its own decision, so the
	// ledger carries as many of them as the capture named.
	wantDown, gotDown := 0, 0
	for _, f := range findings {
		if strings.Contains(f.Text, "DOWN 0x41306") {
			wantDown++
		}
	}
	for _, r := range g.Ledger {
		if strings.Contains(r.Text, "DOWN 0x41306") {
			gotDown++
		}
	}
	if wantDown < 2 {
		t.Fatalf("the capture must carry several same-shaped DOWN findings to be a dedupe test, got %d", wantDown)
	}
	if gotDown != wantDown {
		t.Fatalf("every distinct DOWN task must survive dedupe: %d records for %d findings", gotDown, wantDown)
	}
	// The n3 self-dependence finding is RED and operator-owned — the auditor is not
	// independent of the failure mode it checks.
	var sawN3 bool
	for _, r := range g.Ledger {
		if strings.HasPrefix(r.Text, "n3:") {
			sawN3 = true
			if r.Severity != AuditRed || r.Owner != AuditOwnerOperator {
				t.Fatalf("n3 self-dependence must be a RED operator finding, got %+v", r)
			}
		}
	}
	if !sawN3 {
		t.Fatalf("the n3 self-dependence finding must survive parsing: %+v", g.Ledger)
	}
	// Re-running the identical audit 15 minutes later must not grow the ledger.
	again := AuditGate(verdict, findings, g.Ledger, 1_754_000_900)
	if len(again.Ledger) != len(g.Ledger) || len(again.New) != 0 || len(again.Recurring) != len(g.Ledger) {
		t.Fatalf("a repeated audit must age the same %d records, got %d ledger / %d new / %d recurring",
			len(g.Ledger), len(again.Ledger), len(again.New), len(again.Recurring))
	}
}

// TestAuditGateSelfcheck runs the package's own no-I/O proof, the same one a CLI selfcheck
// verb surfaces.
func TestAuditGateSelfcheck(t *testing.T) {
	if err := AuditGateSelfcheck(); err != nil {
		t.Fatalf("AuditGateSelfcheck: %v", err)
	}
}
