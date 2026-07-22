package main

// Hermetic witness for `fak dispatch progress --weekly --file-issues` (#3338): the
// persistent-lane-wedge lens over the weekly retro. Every test drives the REAL path —
// a retro fixture ledger on disk -> buildDispatchWeeklyReport -> the wedge fold — so a
// change to either the retro's wedge computation or the filing bar is caught here.

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
)

func wedgeFixtureWindow() (time.Time, time.Time) {
	return time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC),
		time.Date(2026, 7, 1, 1, 0, 0, 0, time.UTC)
}

// wedgeReportFromRows writes a retro fixture ledger and folds it through the REAL
// weekly-report builder, so the findings under test come from the same computation the
// shipped `--weekly` surface renders.
func wedgeReportFromRows(t *testing.T, rows []map[string]any) (dispatchWeeklyReport, string) {
	t.Helper()
	runsDir := filepath.Join(t.TempDir(), dispatchProgressRunsDir)
	writeDispatchProgressRows(t, runsDir, rows)
	since, until := wedgeFixtureWindow()
	report, err := buildDispatchWeeklyReport(runsDir, since, until)
	if err != nil {
		t.Fatal(err)
	}
	return report, runsDir
}

// repeatedWedgeRows is the issue's "fixture with a repeated wedge": the cmd lane is
// blocked three times by the SAME class and witnesses no closes, while the docs lane
// ships cleanly. Exactly one lane should earn a ticket.
func repeatedWedgeRows() []map[string]any {
	return []map[string]any{
		{"utc": "2026-07-01T00:05:00Z", "lane": "cmd", "ok": false, "closed_now": 0, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:10:00Z", "lane": "cmd", "ok": false, "closed_now": 0, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:15:00Z", "lane": "cmd", "ok": false, "closed_now": 0, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:20:00Z", "lane": "docs", "ok": true, "closed_now": 1},
		{"utc": "2026-07-01T00:25:00Z", "lane": "docs", "ok": true, "closed_now": 1},
	}
}

// The headline done-when: a repeated wedge yields ONE candidate, fingerprinted and
// attributed to the wedged lane — not one per blocked attempt.
func TestDispatchWeeklyWedgeYieldsOneCandidateForRepeatedWedge(t *testing.T) {
	report, _ := wedgeReportFromRows(t, repeatedWedgeRows())

	got := foldDispatchWeeklyWedgeFindings(report, defaultDispatchWedgeFilingThresholds())
	if len(got) != 1 {
		t.Fatalf("findings = %+v, want exactly one candidate for the repeated cmd wedge", got)
	}
	f := got[0]
	if f.Title != "dispatch wedge: cmd lane repeatedly blocked by AUDIT_UNAVAILABLE" {
		t.Fatalf("title = %q", f.Title)
	}
	if f.CodeSite != "lane-wedge/cmd/AUDIT_UNAVAILABLE" {
		t.Fatalf("code site = %q", f.CodeSite)
	}
	if f.Fingerprint != dispatchWeeklyWedgeFingerprint("lane-wedge/cmd/AUDIT_UNAVAILABLE") {
		t.Fatalf("fingerprint = %q, not the code-site hash", f.Fingerprint)
	}
	// The ticket has to carry the evidence a human needs to act, not just a title.
	for _, want := range []string{"cmd lane took 3 dispatch attempts", "AUDIT_UNAVAILABLE", "clear AUDIT_UNAVAILABLE"} {
		if !strings.Contains(f.Detail, want) {
			t.Fatalf("detail missing %q:\n%s", want, f.Detail)
		}
	}
}

// The other half of the done-when: a healthy week files nothing.
func TestDispatchWeeklyWedgeHealthyWeekFilesNothing(t *testing.T) {
	report, runsDir := wedgeReportFromRows(t, []map[string]any{
		{"utc": "2026-07-01T00:05:00Z", "lane": "cmd", "ok": true, "closed_now": 2},
		{"utc": "2026-07-01T00:10:00Z", "lane": "cmd", "ok": true, "closed_now": 1},
		{"utc": "2026-07-01T00:15:00Z", "lane": "cmd", "ok": true, "closed_now": 3},
		{"utc": "2026-07-01T00:20:00Z", "lane": "docs", "ok": true, "closed_now": 1},
		{"utc": "2026-07-01T00:25:00Z", "lane": "docs", "ok": true, "closed_now": 1},
	})

	if got := foldDispatchWeeklyWedgeFindings(report, defaultDispatchWedgeFilingThresholds()); len(got) != 0 {
		t.Fatalf("healthy week produced findings %+v, want none", got)
	}

	var stdout, stderr bytes.Buffer
	if rc := runDispatchWeeklyWedgeAudit(&stdout, &stderr, runsDir, report, false, 0); rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	if !strings.Contains(stdout.String(), "no persistent lane wedge detected.") {
		t.Fatalf("stdout = %q, want the healthy-week line", stdout.String())
	}
}

// The FILING bar must be strictly stronger than the DISPLAY bar: a lane the retro
// renders as wedged is not automatically worth a durable ticket. Two attempts clears
// dispatchLaneIsWedged but is an anecdote, not a persistent wedge.
func TestDispatchWeeklyWedgeFilingBarIsStricterThanDisplayBar(t *testing.T) {
	report, _ := wedgeReportFromRows(t, []map[string]any{
		{"utc": "2026-07-01T00:05:00Z", "lane": "cmd", "ok": false, "closed_now": 0, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:10:00Z", "lane": "cmd", "ok": false, "closed_now": 0, "audit_error": "commit audit unavailable"},
	})
	if len(report.LaneWedges) != 1 {
		t.Fatalf("precondition: retro should RENDER the 2-attempt wedge, got %+v", report.LaneWedges)
	}
	if got := foldDispatchWeeklyWedgeFindings(report, defaultDispatchWedgeFilingThresholds()); len(got) != 0 {
		t.Fatalf("a rendered-but-thin wedge filed %+v, want nothing below MinAttempts", got)
	}
}

// A lane that keeps hitting a blocker but still SHIPS is noisy, not wedged — filing a
// ticket for it would train operators to ignore the feed.
func TestDispatchWeeklyWedgeDoesNotFileWhenLaneStillCloses(t *testing.T) {
	report, _ := wedgeReportFromRows(t, []map[string]any{
		{"utc": "2026-07-01T00:05:00Z", "lane": "cmd", "ok": false, "closed_now": 3, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:10:00Z", "lane": "cmd", "ok": false, "closed_now": 3, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:15:00Z", "lane": "cmd", "ok": false, "closed_now": 3, "audit_error": "commit audit unavailable"},
	})
	wedge := report.LaneWedges[0]
	if wedge.CloseRate < 0.5 {
		t.Fatalf("precondition: fixture should have a healthy close rate, got %+v", wedge)
	}
	if got := foldDispatchWeeklyWedgeFindings(report, defaultDispatchWedgeFilingThresholds()); len(got) != 0 {
		t.Fatalf("filed %+v for a lane that is still closing", got)
	}
}

// The synthesized LOW_WITNESSED_CLOSE_RATE fallback names no actionable cause (dominant
// count 0), so it must not become a ticket with nothing in it.
func TestDispatchWeeklyWedgeSkipsCauselessFallbackWedge(t *testing.T) {
	report, _ := wedgeReportFromRows(t, []map[string]any{
		{"utc": "2026-07-01T00:05:00Z", "lane": "cmd", "ok": true, "closed_now": 0},
		{"utc": "2026-07-01T00:10:00Z", "lane": "cmd", "ok": true, "closed_now": 0},
		{"utc": "2026-07-01T00:15:00Z", "lane": "cmd", "ok": true, "closed_now": 0},
	})
	if len(report.LaneWedges) != 1 || report.LaneWedges[0].DominantFailureClass != "LOW_WITNESSED_CLOSE_RATE" {
		t.Fatalf("precondition: want the causeless fallback wedge, got %+v", report.LaneWedges)
	}
	if got := foldDispatchWeeklyWedgeFindings(report, defaultDispatchWedgeFilingThresholds()); len(got) != 0 {
		t.Fatalf("filed %+v for a wedge with no named dominant failure", got)
	}
}

// Dedup durability: the fingerprint must be stable across retro windows. A wedge whose
// attempt counts and window stamps moved is the SAME wedge — folding those into the
// hash would re-file it every cadence, which is exactly what the issue asks to prevent.
func TestDispatchWeeklyWedgeFingerprintStableAcrossWindows(t *testing.T) {
	first, _ := wedgeReportFromRows(t, repeatedWedgeRows())
	rows := append(repeatedWedgeRows(), map[string]any{
		"utc": "2026-07-01T00:35:00Z", "lane": "cmd", "ok": false, "closed_now": 0, "audit_error": "commit audit unavailable",
	})
	second, _ := wedgeReportFromRows(t, rows)

	a := foldDispatchWeeklyWedgeFindings(first, defaultDispatchWedgeFilingThresholds())
	b := foldDispatchWeeklyWedgeFindings(second, defaultDispatchWedgeFilingThresholds())
	if len(a) != 1 || len(b) != 1 {
		t.Fatalf("want one finding per window, got %d and %d", len(a), len(b))
	}
	// Precondition: the two windows really did fold DIFFERENT evidence, so a stable
	// fingerprint below is a real invariant and not two identical inputs agreeing.
	if a[0].Detail == b[0].Detail {
		t.Fatal("precondition: the two windows folded identical detail; the test proves nothing")
	}
	if a[0].Fingerprint != b[0].Fingerprint {
		t.Fatalf("fingerprint drifted across windows: %q vs %q", a[0].Fingerprint, b[0].Fingerprint)
	}
	if a[0].Title != b[0].Title {
		t.Fatalf("title drifted across windows (breaks open-title dedup): %q vs %q", a[0].Title, b[0].Title)
	}
}

// The dry run is the DEFAULT, and it must be side-effect free: it prints candidates and
// writes no marker, so a routine retro can never storm the tracker. Once a marker
// exists, the same wedge is deduped away.
func TestDispatchWeeklyWedgeDryRunIsSideEffectFreeThenDedupsOnMarker(t *testing.T) {
	report, runsDir := wedgeReportFromRows(t, repeatedWedgeRows())
	fp := dispatchWeeklyWedgeFingerprint("lane-wedge/cmd/AUDIT_UNAVAILABLE")

	var stdout, stderr bytes.Buffer
	if rc := runDispatchWeeklyWedgeAudit(&stdout, &stderr, runsDir, report, false, 0); rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	out := stdout.String()
	if !strings.Contains(out, "candidate improvement tickets (dry-run") ||
		!strings.Contains(out, fp) {
		t.Fatalf("dry run did not print the candidate:\n%s", out)
	}
	if dispatchaudit.AlreadyFiled(runsDir, fp) {
		t.Fatal("dry run wrote a filed-marker; it must touch nothing")
	}

	// Now mark it filed, as a real --confirm run would, and re-run: deduped away.
	if err := dispatchaudit.MarkFiled(runsDir, fp); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	if rc := runDispatchWeeklyWedgeAudit(&stdout, &stderr, runsDir, report, false, 0); rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	if !strings.Contains(stdout.String(), "all wedges already filed") {
		t.Fatalf("second run did not dedup against the marker:\n%s", stdout.String())
	}
}

// --file-issues has no input outside the weekly retro, so the combination is refused up
// front rather than silently doing nothing.
func TestDispatchProgressFileIssuesRequiresWeekly(t *testing.T) {
	var stderr bytes.Buffer
	_, code := parseDispatchProgressFlags(&stderr, []string{"--file-issues"})
	if code != 2 {
		t.Fatalf("exit code = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "--file-issues requires --weekly") {
		t.Fatalf("stderr = %q", stderr.String())
	}
	stderr.Reset()
	opts, code := parseDispatchProgressFlags(&stderr, []string{"--weekly", "--file-issues", "--max-issues", "3"})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (%s)", code, stderr.String())
	}
	if !opts.FileIssues || opts.Confirm || opts.MaxIssues != 3 {
		t.Fatalf("opts = %+v, want file-issues on, dry-run default, cap 3", opts)
	}
}
