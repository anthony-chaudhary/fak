package issuesmallness

import "testing"

const singleDeliverableBody = `## Goal
Add a filing-time lint for unresolved template artifacts.

## Scope for one GPT-5.5 session
- Keep this to one focused edit, report row, fixture, or doc update.

## Done condition
The issue filer rejects bodies containing the known corruption pattern.

## Witness
A regression test fails on a malformed-body fixture and passes on a normal body.
`

const twoDeliverableBody = `## Goal
- Add a retry counter to the worker heartbeat.
- Add a unit test asserting the counter increments on retry.

## Done condition
Both the counter and its test exist and pass.

## Witness
A focused unit test covers the retry counter.
`

const threeUnrelatedTasksBody = `## Goal
- Fix the login redirect bug on the accounts page.
- Add a new weekly throughput dashboard for the operator.
- Rewrite the onboarding documentation from scratch.

## Scope for one GPT-5.5 session
- Keep this to one focused edit, report row, fixture, or doc update.

## Done condition
All three items above are complete and merged.

## Witness
Manual review of the three deliverables.
`

const proseThreeTasksBody = `## Goal
Fix the login redirect bug on the accounts page; add a new weekly throughput
dashboard for the operator; and then rewrite the onboarding documentation.

## Done condition
All work above lands.

## Witness
A failing fixture proves bundled work is refused.
`

func TestLintBodySingleDeliverablePasses(t *testing.T) {
	got := LintBody(singleDeliverableBody)
	if got.Verdict != Pass || got.Count != 1 || got.WitnessCount != 1 {
		t.Fatalf("LintBody single = %#v, want pass with one deliverable and one witness", got)
	}
}

func TestLintBodyTwoDeliverablesWarns(t *testing.T) {
	got := LintBody(twoDeliverableBody)
	if got.Verdict != Warn || got.Count != 2 || got.WitnessCount != 1 {
		t.Fatalf("LintBody two = %#v, want warn with one witness", got)
	}
}

func TestLintBodyThreeUnrelatedTasksFails(t *testing.T) {
	got := LintBody(threeUnrelatedTasksBody)
	if got.Verdict != Fail || got.Count != 3 {
		t.Fatalf("LintBody three = %#v, want fail with three deliverables", got)
	}
	if joined := got.Items[0] + " " + got.Items[1] + " " + got.Items[2]; joined == "" {
		t.Fatalf("items unexpectedly empty: %#v", got)
	}
}

func TestLintBodyThreeProseTasksFails(t *testing.T) {
	got := LintBody(proseThreeTasksBody)
	if got.Verdict != Fail || got.Count != 3 {
		t.Fatalf("LintBody prose = %#v, want fail with three deliverables", got)
	}
}

func TestLintBodyMissingWitnessFails(t *testing.T) {
	body := "## Goal\nAdd a retry counter.\n"
	got := LintBody(body)
	if got.Verdict != Fail || got.WitnessCount != 0 {
		t.Fatalf("LintBody missing witness = %#v, want fail with zero witnesses", got)
	}
}

func TestLintBodyMultipleWitnessesFail(t *testing.T) {
	body := "## Goal\nAdd a retry counter.\n## Witness\n- Unit test.\n- Dry-run report.\n"
	got := LintBody(body)
	if got.Verdict != Fail || got.WitnessCount != 2 {
		t.Fatalf("LintBody multiple witnesses = %#v, want fail with two witnesses", got)
	}
}

func TestDoneConditionIsNotFoldedIntoGoalCount(t *testing.T) {
	body := "## Goal\nAdd a filing-time lint for template artifacts.\n## Done condition\nThe lint rejects bodies with the known pattern.\n## Witness\nA fixture fails.\n"
	got := LintBody(body)
	if got.Verdict != Pass || got.Count != 1 || got.SectionSource != "goal" {
		t.Fatalf("LintBody done folded = %#v, want pass from goal only", got)
	}
}

func TestReportOpenFlagsNonPassingRows(t *testing.T) {
	report := ReportOpen([]Issue{
		{Number: 1, Title: "clean one", Body: singleDeliverableBody},
		{Number: 2, Title: "bundled one", Body: threeUnrelatedTasksBody},
	})
	if report.Schema != Schema || report.Mode != "open" || report.Scanned != 2 {
		t.Fatalf("report header = %#v", report)
	}
	if report.Counts[Pass] != 1 || report.Counts[Fail] != 1 || !HasFailReport(report) {
		t.Fatalf("report counts = %#v, want pass=1 fail=1", report.Counts)
	}
	if len(report.Flagged) != 1 || report.Flagged[0].Number != 2 {
		t.Fatalf("flagged = %#v, want issue 2", report.Flagged)
	}
}
