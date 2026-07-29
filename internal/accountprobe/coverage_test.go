package accountprobe

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

// coverage_test.go — #5391's witnesses. The defect is that absence of evidence reads as
// evidence of health: a seat whose newest probe row is eight days old produces no fresh
// verdict, so every read-side fold falls through and publishes it exactly as it publishes a
// seat that was probed a minute ago and found healthy. Each test below pins one half of the
// separation, and each names the mutant it reds under.

// probeRow builds one ledger line stamped at `when`.
func probeRow(account, status string, when time.Time) string {
	return fmt.Sprintf(`{"ts":%q,"account":%q,"status":%q}`,
		when.UTC().Format("2006-01-02T15:04:05Z"), account, status)
}

// coverageNow is a fixed clock; every age below is measured from it.
var coverageNow = time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)

// TestFreshlyProbedSeatIsMeasured pins the case that must NOT move: a seat probed minutes
// ago still grades measured, still reports the same age, and the pre-existing read surface
// (RecentProbeAgeMin) answers exactly what it answered before this fold existed.
//
// Mutant: grading a fresh row SeatHealthStale (or dropping the `> SeatCoverageMaxAgeMin`
// test so every row falls to stale) reds this.
func TestFreshlyProbedSeatIsMeasured(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd, probeRow(".claude-a", "OK", coverageNow.Add(-5*time.Minute)))

	cov := GradeSeat(".claude-a", rd, coverageNow)
	if cov.Health != SeatHealthFresh || !cov.Health.Measured() {
		t.Fatalf("health = %q measured=%v, want probe-fresh/true", cov.Health, cov.Health.Measured())
	}
	if !cov.HasAge || math.Abs(cov.AgeMin-5) > 1e-6 {
		t.Fatalf("age = %v (has=%v), want 5", cov.AgeMin, cov.HasAge)
	}
	if cov.Status != "OK" {
		t.Fatalf("status = %q, want OK", cov.Status)
	}
	// The existing read surface is untouched: a genuinely probed seat folds as it always did.
	age := RecentProbeAgeMin(".claude-a", rd, coverageNow)
	if age == nil || math.Abs(*age-5) > 1e-6 {
		t.Fatalf("RecentProbeAgeMin = %v, want 5 — the new grade must not perturb the old read", age)
	}
}

// TestEightDayOldRowIsStaleNotFresh is the observed defect, at the observed magnitude: the
// host in #5391 carried claude rows 8-9 days old in a ledger whose opencode-* rows were
// current to the minute. The stale seat must not read as measured, and must not read as
// blocked either — the issue is explicit that treating "no data" as "blocked" strands seats.
//
// Mutant: raising SeatCoverageMaxAgeMin past eight days, or deleting the age comparison,
// reds this. (So does grading an over-budget row SeatHealthNever, via the Status check: the
// row exists and its verdict is carried.)
func TestEightDayOldRowIsStaleNotFresh(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd,
		probeRow("opencode-glm", "OK", coverageNow.Add(-2*time.Minute)),
		probeRow(".claude-a", "OK", coverageNow.Add(-8*24*time.Hour)),
	)

	cov := GradeSeat(".claude-a", rd, coverageNow)
	if cov.Health != SeatHealthStale {
		t.Fatalf("8-day-old row health = %q, want probe-stale", cov.Health)
	}
	if cov.Health.Measured() {
		t.Fatal("an 8-day-old probe row must not license reading the seat as measured")
	}
	if !cov.HasAge || math.Abs(cov.AgeMin-8*24*60) > 1e-6 {
		t.Fatalf("age = %v (has=%v), want 11520", cov.AgeMin, cov.HasAge)
	}
	if cov.Status != "OK" {
		t.Fatalf("status = %q — a stale row's verdict is still carried, for reporting", cov.Status)
	}
	// The same ledger, same instant: the busy account is measured. That asymmetry inside one
	// file is the whole finding — the ledger is alive, it is simply not covering this seat.
	if got := GradeSeat("opencode-glm", rd, coverageNow); got.Health != SeatHealthFresh {
		t.Fatalf("opencode-glm health = %q, want probe-fresh", got.Health)
	}
}

// TestNeverProbedIsDistinctFromProbedOK is the acceptance line stated in the issue: "never
// probed" and "probed OK" must be distinguishable rather than both collapsing to available.
//
// Mutant: returning SeatHealthFresh (or any Measured() grade) for a seat missing from the
// ledger reds this — that mutant IS the bug being fixed.
func TestNeverProbedIsDistinctFromProbedOK(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd, probeRow(".claude-a", "OK", coverageNow.Add(-1*time.Minute)))

	probed := GradeSeat(".claude-a", rd, coverageNow)
	never := GradeSeat(".claude-b", rd, coverageNow)
	if never.Health != SeatHealthNever || never.Health.Measured() {
		t.Fatalf("never-probed health = %q measured=%v, want probe-never/false",
			never.Health, never.Health.Measured())
	}
	if never.Health == probed.Health {
		t.Fatalf("never-probed and probed-OK both graded %q — the collapse #5391 names", never.Health)
	}
	if never.HasAge || never.AgeMin != 0 {
		t.Fatalf("never-probed age = %v (has=%v); a zero age must not be fabricated",
			never.AgeMin, never.HasAge)
	}
	if never.Status != "" {
		t.Fatalf("never-probed status = %q, want empty", never.Status)
	}
}

// TestUndatableRowIsNotMeasured: a row whose timestamp will not parse cannot be shown to be
// recent, so it cannot license a health read. It is stale (the prober did touch this seat),
// not never, and carries no age.
//
// Mutant: falling through to SeatHealthFresh when parseLedgerTime returns nil reds this.
func TestUndatableRowIsNotMeasured(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd, `{"ts":"not-a-time","account":".claude-a","status":"OK"}`)

	cov := GradeSeat(".claude-a", rd, coverageNow)
	if cov.Health != SeatHealthStale || cov.Health.Measured() {
		t.Fatalf("undatable row health = %q measured=%v, want probe-stale/false",
			cov.Health, cov.Health.Measured())
	}
	if cov.HasAge {
		t.Fatalf("undatable row reported an age (%v); no age was established", cov.AgeMin)
	}
}

// TestCoverageBudgetBoundary pins the comparison direction at the declared budget: a row
// exactly at SeatCoverageMaxAgeMin is still measured, one minute past it is not. It also
// pins that the chosen constant separates the two populations #5391 observed — minutes-old
// rows on one side, days-old rows on the other.
//
// Mutant: `>=` instead of `>` in the budget test reds the at-budget case; any constant below
// ~5 minutes or above ~8 days reds one of the two population cases.
func TestCoverageBudgetBoundary(t *testing.T) {
	for _, tc := range []struct {
		name string
		age  time.Duration
		want SeatHealth
	}{
		{"minutes-old (the covered population)", 5 * time.Minute, SeatHealthFresh},
		{"exactly at budget", time.Duration(SeatCoverageMaxAgeMin) * time.Minute, SeatHealthFresh},
		{"one minute past budget", time.Duration(SeatCoverageMaxAgeMin+1) * time.Minute, SeatHealthStale},
		{"days-old (the uncovered population)", 8 * 24 * time.Hour, SeatHealthStale},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rd := t.TempDir()
			writeLedger(t, rd, probeRow(".claude-a", "OK", coverageNow.Add(-tc.age)))
			if got := GradeSeat(".claude-a", rd, coverageNow).Health; got != tc.want {
				t.Fatalf("age %v graded %q, want %q", tc.age, got, tc.want)
			}
		})
	}
}

// TestGradeSeatsCountsAndNote folds a mixed roster and checks the counts, the fraction, and
// the operator line. The note is the "make the gap visible" half of the issue.
//
// Mutant: counting a stale or never seat as Fresh reds the counts and the fraction; returning
// "" from Note when seats are unmeasured reds the note assertions.
func TestGradeSeatsCountsAndNote(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd,
		probeRow("opencode-glm", "OK", coverageNow.Add(-2*time.Minute)),
		probeRow(".claude-a", "OK", coverageNow.Add(-8*24*time.Hour)),
	)

	rep := GradeSeats([]string{"opencode-glm", ".claude-a", ".claude-b"}, rd, coverageNow)
	if rep.Fresh != 1 || rep.Stale != 1 || rep.Never != 1 {
		t.Fatalf("counts fresh/stale/never = %d/%d/%d, want 1/1/1", rep.Fresh, rep.Stale, rep.Never)
	}
	if rep.Unmeasured() != 2 {
		t.Fatalf("Unmeasured = %d, want 2", rep.Unmeasured())
	}
	if !rep.Sufficient || rep.LedgerAccounts != 2 {
		t.Fatalf("sufficient=%v ledgerAccounts=%d, want true/2", rep.Sufficient, rep.LedgerAccounts)
	}
	frac, ok := rep.MeasuredFraction()
	if !ok || math.Abs(frac-1.0/3.0) > 1e-9 {
		t.Fatalf("MeasuredFraction = %v,%v want 1/3,true", frac, ok)
	}
	note := rep.Note()
	if !strings.Contains(note, "INCOMPLETE") || !strings.Contains(note, "2/3") {
		t.Fatalf("note = %q, want an INCOMPLETE 2/3 line", note)
	}
	for _, want := range []string{".claude-a", "8.0d", ".claude-b", string(SeatHealthNever)} {
		if !strings.Contains(note, want) {
			t.Fatalf("note = %q, missing %q", note, want)
		}
	}
	if strings.Contains(note, "opencode-glm") {
		t.Fatalf("note = %q names a measured seat; the line is the gap, not the roster", note)
	}
}

// TestEmptyReadIsInsufficientNotZeroHealthy is the "do not fabricate a zero" witness. A
// ledger that yields nothing has not established that 0% of seats are healthy; it has
// established nothing. The report must say so, and MeasuredFraction must refuse to hand back
// a number that would be printed as a damning 0%.
//
// Mutant: returning (0, true) from MeasuredFraction on an unreadable ledger, or setting
// Sufficient unconditionally true, reds this.
func TestEmptyReadIsInsufficientNotZeroHealthy(t *testing.T) {
	rd := t.TempDir() // no probe_ledger.jsonl at all
	rep := GradeSeats([]string{".claude-a", ".claude-b"}, rd, coverageNow)
	if rep.Sufficient {
		t.Fatal("an absent ledger must not report a sufficient read")
	}
	if frac, ok := rep.MeasuredFraction(); ok {
		t.Fatalf("MeasuredFraction = %v,true — an empty read must report insufficient, not 0%%", frac)
	}
	if rep.Never != 2 {
		t.Fatalf("Never = %d, want 2 — every seat is still graded unmeasured", rep.Never)
	}
	note := rep.Note()
	if !strings.Contains(note, "UNKNOWN") || !strings.Contains(note, "2 seat(s) unmeasured") {
		t.Fatalf("note = %q, want an UNKNOWN line naming 2 unmeasured seats", note)
	}
}

// TestLiveLedgerWithNoRowsForTheseSeatsIsAnHonestZero is the other side of the same rule:
// when the ledger IS being written and simply says nothing about the enrolled seats, zero
// measured is a genuine finding and must be reported as a number, not withheld. That is the
// exact configuration #5391 was filed on — a 4579-line ledger, current to the minute, with
// nothing recent for the claude seats.
//
// Mutant: gating Sufficient on the enrolled seats' rows rather than on the ledger having any
// rows reds this (it would suppress the finding the issue is about).
func TestLiveLedgerWithNoRowsForTheseSeatsIsAnHonestZero(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd, probeRow("opencode-glm", "OK", coverageNow.Add(-2*time.Minute)))

	rep := GradeSeats([]string{".claude-a", ".claude-b"}, rd, coverageNow)
	frac, ok := rep.MeasuredFraction()
	if !ok || frac != 0 {
		t.Fatalf("MeasuredFraction = %v,%v want 0,true — a live ledger's silence IS the finding", frac, ok)
	}
	if !strings.Contains(rep.Note(), "INCOMPLETE") {
		t.Fatalf("note = %q, want INCOMPLETE", rep.Note())
	}
}

// TestFullyCoveredRosterIsSilent keeps the note from becoming background noise: when every
// enrolled seat is measured there is nothing to report, so the line is empty and the fraction
// is 1.
//
// Mutant: emitting a note unconditionally reds this.
func TestFullyCoveredRosterIsSilent(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd,
		probeRow("opencode-glm", "OK", coverageNow.Add(-2*time.Minute)),
		probeRow(".claude-a", "LIMIT", coverageNow.Add(-30*time.Minute)),
	)

	rep := GradeSeats([]string{"opencode-glm", ".claude-a"}, rd, coverageNow)
	if rep.Unmeasured() != 0 {
		t.Fatalf("Unmeasured = %d, want 0", rep.Unmeasured())
	}
	if note := rep.Note(); note != "" {
		t.Fatalf("note = %q, want empty on a fully covered roster", note)
	}
	frac, ok := rep.MeasuredFraction()
	if !ok || frac != 1 {
		t.Fatalf("MeasuredFraction = %v,%v want 1,true", frac, ok)
	}
	// A blocked-but-freshly-probed seat is measured: coverage is about evidence, not verdict.
	if got := GradeSeat(".claude-a", rd, coverageNow); !got.Health.Measured() || got.Status != "LIMIT" {
		t.Fatalf("fresh LIMIT row = %+v, want measured with its verdict carried", got)
	}
}
