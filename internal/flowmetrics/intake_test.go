package flowmetrics

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"
)

func intakeFixture(now time.Time) Input {
	var issues []Issue
	var commits []Commit

	// Create 20 days of activity (well over the 14-day minimum).
	// Days -20 to -1:
	for day := 20; day >= 1; day-- {
		dayStart := now.Add(-time.Duration(day) * 24 * time.Hour)

		// 2 issues opened each day
		issA := Issue{
			Number:    day*10 + 1,
			Title:     fmt.Sprintf("feat(task): work %d-A", day),
			CreatedAt: dayStart.Add(2 * time.Hour),
			// Closed 6 hours later
			ClosedAt: ptr(dayStart.Add(8 * time.Hour)),
		}
		issB := Issue{
			Number:    day*10 + 2,
			Title:     fmt.Sprintf("fix(task): work %d-B", day),
			CreatedAt: dayStart.Add(4 * time.Hour),
			// Remains unstarted -> backlog
		}
		issues = append(issues, issA, issB)

		// Commit starting and completing issA
		commits = append(commits, Commit{
			SHA:     fmt.Sprintf("sha-%d", day),
			When:    dayStart.Add(3 * time.Hour),
			Subject: fmt.Sprintf("feat(task): land %d-A (#%d) (fak task)", day, issA.Number),
			Leaf:    "task",
			Issues:  []int{issA.Number},
		})
	}

	return Input{
		Issues:     issues,
		Commits:    commits,
		Now:        now,
		WindowDays: 30,
		Tree: TreeWIP{
			Measured:  true,
			Buildable: true,
		},
	}
}

// TestRenderArrivalServiceColumnsDistinctlyLabelled verifies DoD criteria 1, 5, 6:
// The per-day series renders arrivals (opened), closes (closed), and backlog and in_flight
// as separate, distinctly labelled columns over at least 14 days.
func TestRenderArrivalServiceColumnsDistinctlyLabelled(t *testing.T) {
	in := intakeFixture(base)
	rep := Build(in)

	var buf bytes.Buffer
	RenderArrivalServiceReadout(&buf, rep, base)
	out := buf.String()

	// Check table header
	if !strings.Contains(out, "arrival vs service (daily series") {
		t.Fatalf("readout missing daily series header:\n%s", out)
	}

	// Verify both backlog and in_flight are distinctly labelled
	if !strings.Contains(out, "backlog") {
		t.Fatalf("readout missing 'backlog' column header:\n%s", out)
	}
	if !strings.Contains(out, "in_flight") && !strings.Contains(out, "in-flight") {
		t.Fatalf("readout missing 'in_flight' / 'in-flight' column header:\n%s", out)
	}

	// Verify arrivals and closes are labelled
	if !strings.Contains(out, "arrivals") && !strings.Contains(out, "opened") {
		t.Fatalf("readout missing arrivals/opened column header:\n%s", out)
	}
	if !strings.Contains(out, "closes") && !strings.Contains(out, "closed") {
		t.Fatalf("readout missing closes/closed column header:\n%s", out)
	}

	// Verify at least 14 daily rows are rendered
	lines := strings.Split(out, "\n")
	dailyRows := 0
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		// Check for date pattern YYYY-MM-DD
		if len(trimmed) >= 10 && trimmed[4] == '-' && trimmed[7] == '-' {
			dailyRows++
		}
	}
	if dailyRows < 14 {
		t.Fatalf("rendered %d daily rows, want at least 14:\n%s", dailyRows, out)
	}
}

// TestRenderArrivalServiceThreeWindows verifies DoD criteria 2, 5, 6:
// Trailing arrival rate and close rate per day are shown side by side over three windows (7d, 30d, 60d)
// with the net for each, so the ratio is re-derivable from the printed numbers.
func TestRenderArrivalServiceThreeWindows(t *testing.T) {
	in := intakeFixture(base)
	rep := Build(in)

	var buf bytes.Buffer
	RenderArrivalServiceReadout(&buf, rep, base)
	out := buf.String()

	if !strings.Contains(out, "rates over trailing windows") {
		t.Fatalf("readout missing trailing windows section:\n%s", out)
	}

	for _, win := range []string{"7d", "30d", "60d"} {
		if !strings.Contains(out, win) {
			t.Fatalf("readout missing window %q:\n%s", win, out)
		}
	}

	// Check that arrival rate, close rate, and net headers exist
	for _, header := range []string{"arrival rate/day", "close rate/day", "net"} {
		if !strings.Contains(out, header) {
			t.Fatalf("readout missing header %q:\n%s", header, out)
		}
	}

	// Check that arrivals and closes are present so ratio is re-derivable
	if !strings.Contains(out, "arrivals") || !strings.Contains(out, "closes") {
		t.Fatalf("readout missing arrivals or closes count in windows table:\n%s", out)
	}
	if !strings.Contains(out, "ratio") {
		t.Fatalf("readout missing ratio in windows table:\n%s", out)
	}
}

// TestRenderArrivalServiceDeclaredIntakeCapAndOvershoot verifies DoD criteria 3, 5, 6:
// Print the measured service rate as a declared intake cap (arrivals per day equal to trailing close rate)
// plus the current overshoot against it.
func TestRenderArrivalServiceDeclaredIntakeCapAndOvershoot(t *testing.T) {
	// 2 arrivals/day, 1 close/day over 30d window
	in := intakeFixture(base)
	rep := Build(in)

	var buf bytes.Buffer
	RenderArrivalServiceReadout(&buf, rep, base)
	out := buf.String()

	if !strings.Contains(out, "declared intake cap") {
		t.Fatalf("readout missing 'declared intake cap':\n%s", out)
	}
	if !strings.Contains(out, "arrivals/day") && !strings.Contains(out, "arrivals-per-day") {
		t.Fatalf("readout missing 'arrivals/day' or 'arrivals-per-day':\n%s", out)
	}
	if !strings.Contains(out, "current overshoot") && !strings.Contains(out, "overshoot") {
		t.Fatalf("readout missing 'current overshoot':\n%s", out)
	}
	if !strings.Contains(out, "close_rate") && !strings.Contains(out, "close rate") {
		t.Fatalf("readout missing 'close_rate' or 'close rate':\n%s", out)
	}

	// Check exact numbers for 30d:
	// In fixture: 20 days of 1 close/day = 20 closes / 30d = 0.7 closes/day.
	// 2 arrivals/day * 20 days = 40 arrivals / 30d = 1.3 arrivals/day.
	// Declared cap is 0.7 arrivals/day, overshoot is +0.7 arrivals/day.
	if !strings.Contains(out, "0.7 arrivals/day") {
		t.Fatalf("expected 0.7 arrivals/day for declared intake cap, got:\n%s", out)
	}
	if !strings.Contains(out, "+0.7 arrivals/day") {
		t.Fatalf("expected +0.7 arrivals/day for overshoot, got:\n%s", out)
	}
}

// TestRenderArrivalServiceIntakeWithinCap verifies negative overshoot when closes exceed arrivals.
func TestRenderArrivalServiceIntakeWithinCap(t *testing.T) {
	now := base
	var issues []Issue
	var commits []Commit

	// 10 issues created 40 days ago, all closed in the last 10 days (high service rate).
	// Only 2 issues opened in the last 30 days (low arrival rate).
	for i := 1; i <= 10; i++ {
		created := now.Add(-40 * 24 * time.Hour)
		closed := now.Add(-time.Duration(i) * 24 * time.Hour)
		issues = append(issues, Issue{
			Number:    i,
			CreatedAt: created,
			ClosedAt:  ptr(closed),
		})
		commits = append(commits, Commit{
			SHA:     fmt.Sprintf("sha-%d", i),
			When:    closed.Add(-time.Hour),
			Subject: fmt.Sprintf("fix(#%d) (fak task)", i),
			Leaf:    "task",
			Issues:  []int{i},
		})
	}
	// 2 new arrivals
	issues = append(issues,
		Issue{Number: 101, CreatedAt: now.Add(-5 * 24 * time.Hour)},
		Issue{Number: 102, CreatedAt: now.Add(-2 * 24 * time.Hour)},
	)

	rep := Build(Input{
		Issues:     issues,
		Commits:    commits,
		Now:        now,
		WindowDays: 30,
		Tree:       TreeWIP{Measured: true, Buildable: true},
	})

	var buf bytes.Buffer
	RenderArrivalServiceReadout(&buf, rep, now)
	out := buf.String()

	if !strings.Contains(out, "declared intake cap: 0.3 arrivals/day") {
		t.Fatalf("expected declared intake cap 0.3 arrivals/day (10 closes / 30d), got:\n%s", out)
	}
	// Arrival rate is 2 arrivals / 30d = 0.1 arrivals/day. Overshoot = 0.1 - 0.3 = -0.3 arrivals/day.
	if !strings.Contains(out, "-0.3 arrivals/day") {
		t.Fatalf("expected negative overshoot (-0.3 arrivals/day), got:\n%s", out)
	}
}

// TestRenderArrivalServiceFallbackFromCurve verifies that RenderArrivalServiceReadout
// computes windows and intake cap even if rep.ArrivalWindows is nil/empty.
func TestRenderArrivalServiceFallbackFromCurve(t *testing.T) {
	curve := []DayWIP{
		{Date: "2026-08-01", Opened: 2, Started: 1, Closed: 1, Backlog: 3, InFlight: 2},
		{Date: "2026-08-02", Opened: 1, Started: 1, Closed: 1, Backlog: 3, InFlight: 2},
		{Date: "2026-08-03", Opened: 3, Started: 2, Closed: 2, Backlog: 4, InFlight: 2},
		{Date: "2026-08-04", Opened: 0, Started: 1, Closed: 1, Backlog: 3, InFlight: 2},
		{Date: "2026-08-05", Opened: 2, Started: 2, Closed: 1, Backlog: 4, InFlight: 3},
		{Date: "2026-08-06", Opened: 1, Started: 1, Closed: 1, Backlog: 4, InFlight: 3},
		{Date: "2026-08-07", Opened: 1, Started: 1, Closed: 1, Backlog: 4, InFlight: 3},
		{Date: "2026-08-08", Opened: 2, Started: 1, Closed: 0, Backlog: 5, InFlight: 4},
		{Date: "2026-08-09", Opened: 1, Started: 1, Closed: 1, Backlog: 5, InFlight: 4},
		{Date: "2026-08-10", Opened: 2, Started: 2, Closed: 2, Backlog: 5, InFlight: 4},
		{Date: "2026-08-11", Opened: 1, Started: 1, Closed: 1, Backlog: 5, InFlight: 4},
		{Date: "2026-08-12", Opened: 0, Started: 0, Closed: 1, Backlog: 5, InFlight: 3},
		{Date: "2026-08-13", Opened: 2, Started: 1, Closed: 1, Backlog: 6, InFlight: 3},
		{Date: "2026-08-14", Opened: 1, Started: 1, Closed: 1, Backlog: 6, InFlight: 3},
	}
	rep := Report{Curve: curve}

	var buf bytes.Buffer
	RenderArrivalServiceReadout(&buf, rep, base)
	out := buf.String()

	for _, win := range []string{"7d", "30d", "60d"} {
		if !strings.Contains(out, win) {
			t.Fatalf("missing window %s in fallback:\n%s", win, out)
		}
	}
	if !strings.Contains(out, "declared intake cap") {
		t.Fatalf("missing declared intake cap in fallback:\n%s", out)
	}
	if !strings.Contains(out, "current overshoot") {
		t.Fatalf("missing current overshoot in fallback:\n%s", out)
	}
}

// TestRenderIntakeReportAlias verifies RenderIntakeReport behaves identically to RenderArrivalServiceReadout.
func TestRenderIntakeReportAlias(t *testing.T) {
	in := intakeFixture(base)
	rep := Build(in)

	var buf1, buf2 bytes.Buffer
	RenderArrivalServiceReadout(&buf1, rep, base)
	RenderIntakeReport(&buf2, rep, base)

	if buf1.String() != buf2.String() {
		t.Fatalf("RenderIntakeReport output != RenderArrivalServiceReadout output")
	}
}

// TestRenderArrivalServiceEmpty verifies that an empty report does not panic.
func TestRenderArrivalServiceEmpty(t *testing.T) {
	var buf bytes.Buffer
	RenderArrivalServiceReadout(&buf, Report{}, base)
	out := buf.String()
	if !strings.Contains(out, "no data") {
		t.Fatalf("expected 'no data' on empty report, got:\n%s", out)
	}
}

// TestCeilingConstantsUntouched verifies DoD criterion 4:
// Both ceiling constants (ArrivalServiceRatioCeiling = 1.10, UnstartedShareCeiling = 0.60)
// in internal/flowmetrics/report.go must remain untouched.
func TestCeilingConstantsUntouched(t *testing.T) {
	if ArrivalServiceRatioCeiling != 1.10 {
		t.Fatalf("ArrivalServiceRatioCeiling = %v, want 1.10", ArrivalServiceRatioCeiling)
	}
	if UnstartedShareCeiling != 0.60 {
		t.Fatalf("UnstartedShareCeiling = %v, want 0.60", UnstartedShareCeiling)
	}
}
