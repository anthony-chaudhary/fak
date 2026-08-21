package flowmetrics

import (
	"math"
	"testing"
	"time"
)

// base is a fixed anchor so every duration in these tests is exact and no test
// reads the wall clock.
var base = time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC)

func at(hoursFromBase float64) time.Time {
	return base.Add(time.Duration(hoursFromBase * float64(time.Hour)))
}

func ptr(t time.Time) *time.Time { return &t }

func TestParseIssueRefsFencesForeignAndKeepsSingleDigits(t *testing.T) {
	// #3 and #26 are real fak issues, so a digit-count fence would drop them.
	got := ParseIssueRefs("fix(x): thing (#3), (#26), #5420, and Closes #8417 (fak x)")
	want := []int{3, 26, 5420, 8417}
	if len(got) != len(want) {
		t.Fatalf("refs = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("refs = %v, want %v", got, want)
		}
	}
	// Foreign-repo citations must not be minted as fak issues.
	if got := ParseIssueRefs("perf: port llama.cpp #14762 and vLLM #35021"); len(got) != 0 {
		t.Fatalf("foreign refs leaked: %v", got)
	}
	// A sha-like token is not a reference.
	if got := ParseIssueRefs("chore: bump abc#123"); len(got) != 0 {
		t.Fatalf("in-word ref matched: %v", got)
	}
}

func TestParseCommitRefsTakesTrailersButNotProse(t *testing.T) {
	subject := "fix(gateway): treat same-tick ready as positive (fak gateway)"
	body := "The #5822 shape is unrelated prose.\n\nFixes #6018\n"
	got := ParseCommitRefs(subject, body)
	if len(got) != 1 || got[0] != 6018 {
		t.Fatalf("refs = %v, want [6018] — prose ref must not attribute, trailer must", got)
	}
}

func TestParseLeaf(t *testing.T) {
	if got := ParseLeaf("fix(devindex): prevent popup (#6020) (fak devindex)"); got != "devindex" {
		t.Fatalf("leaf = %q, want devindex", got)
	}
	if got := ParseLeaf("chore: no trailer here"); got != "" {
		t.Fatalf("leaf = %q, want empty", got)
	}
}

func TestBuildSpansDecomposesQueueAndActive(t *testing.T) {
	issues := []Issue{
		// Opened at 0, first commit at 10h, closed at 12h:
		// queue 10, active 2, lead 12, efficiency 2/12.
		{Number: 1, CreatedAt: at(0), ClosedAt: ptr(at(12))},
		// Never committed, still open: unstarted.
		{Number: 2, CreatedAt: at(1)},
		// Committed but never closed: in flight.
		{Number: 3, CreatedAt: at(2)},
	}
	commits := []Commit{
		{SHA: "a", When: at(10), Subject: "one (#1) (fak x)", Leaf: "x", Issues: []int{1}},
		{SHA: "b", When: at(11), Subject: "two (#1) (fak x)", Leaf: "x", Issues: []int{1}},
		{SHA: "c", When: at(5), Subject: "three (#3) (fak y)", Leaf: "y", Issues: []int{3}},
	}
	spans := BuildSpans(issues, commits)
	if len(spans) != 3 {
		t.Fatalf("spans = %d, want 3", len(spans))
	}
	one := spans[0]
	if one.QueueHours != 10 || one.ActiveHours != 2 || one.LeadHours != 12 {
		t.Fatalf("span1 queue/active/lead = %v/%v/%v, want 10/2/12",
			one.QueueHours, one.ActiveHours, one.LeadHours)
	}
	if math.Abs(one.FlowEfficiency-2.0/12.0) > 1e-9 {
		t.Fatalf("span1 efficiency = %v, want %v", one.FlowEfficiency, 2.0/12.0)
	}
	if one.Commits != 2 || one.Atomic() {
		t.Fatalf("span1 should be 2 commits and non-atomic, got %d/%v", one.Commits, one.Atomic())
	}
	// An unstarted span must report efficiency as UNKNOWN (-1), never 0.
	two := spans[1]
	if two.Started() || two.FlowEfficiency != -1 {
		t.Fatalf("span2 started=%v efficiency=%v, want false/-1", two.Started(), two.FlowEfficiency)
	}
	three := spans[2]
	if !three.Started() || three.Closed() {
		t.Fatalf("span3 should be started and open")
	}
	if !three.Atomic() {
		t.Fatalf("span3 is one commit in one lane and should be atomic")
	}
	// Age of in-flight work is measured from the START, not the open.
	if got := three.AgeHours(at(20)); got != 15 {
		t.Fatalf("span3 age = %v, want 15 (from first commit at 5h)", got)
	}
}

func TestBuildSpansFloorsNegativeDurations(t *testing.T) {
	// A rebased or skewed commit older than the issue must not produce a
	// negative queue that silently corrupts percentiles.
	issues := []Issue{{Number: 1, CreatedAt: at(10), ClosedAt: ptr(at(20))}}
	commits := []Commit{{SHA: "a", When: at(1), Issues: []int{1}, Leaf: "x"}}
	s := BuildSpans(issues, commits)[0]
	if s.QueueHours != 0 {
		t.Fatalf("queue = %v, want 0 floor", s.QueueHours)
	}
	if s.ActiveHours != 19 {
		t.Fatalf("active = %v, want 19", s.ActiveHours)
	}
}

func TestBuildSpansIgnoresCommitsForUnknownIssues(t *testing.T) {
	// Without a createdAt there is no queue term, so a phantom span would
	// pollute every percentile.
	spans := BuildSpans(
		[]Issue{{Number: 1, CreatedAt: at(0)}},
		[]Commit{{SHA: "a", When: at(1), Issues: []int{999}}},
	)
	if len(spans) != 1 || spans[0].Started() {
		t.Fatalf("unknown-issue commit must not create or start a span: %+v", spans)
	}
}

func TestPercentileIsInclusiveNearestRank(t *testing.T) {
	xs := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	cases := []struct{ p, want float64 }{
		{50, 5}, // ceil(0.5*10)-1 = 4 -> 5
		{90, 9},
		{100, 10},
		{0, 1},
	}
	for _, c := range cases {
		if got := Percentile(xs, c.p); got != c.want {
			t.Fatalf("p%v = %v, want %v", c.p, got, c.want)
		}
	}
	if got := Percentile(nil, 50); got != 0 {
		t.Fatalf("empty percentile = %v, want 0", got)
	}
	// The caller's slice must not be reordered.
	in := []float64{3, 1, 2}
	_ = Percentile(in, 50)
	if in[0] != 3 {
		t.Fatalf("input slice was mutated: %v", in)
	}
}

func TestWIPCurveCountsStartedOnlyAsInFlight(t *testing.T) {
	day := func(d int) time.Time { return time.Date(2026, 8, d, 0, 0, 0, 0, time.UTC) }
	issues := []Issue{
		// started day 1, closed day 3 -> in flight on days 1,2,3
		{Number: 1, CreatedAt: day(1), ClosedAt: ptr(day(3).Add(2 * time.Hour))},
		// open, never committed -> backlog, never in flight
		{Number: 2, CreatedAt: day(1)},
	}
	commits := []Commit{{SHA: "a", When: day(1).Add(time.Hour), Issues: []int{1}, Leaf: "x"}}
	curve := WIPCurve(BuildSpans(issues, commits), day(1), day(4))
	if len(curve) != 4 {
		t.Fatalf("curve len = %d, want 4", len(curve))
	}
	wantInFlight := []int{1, 1, 1, 0}
	for i, w := range wantInFlight {
		if curve[i].InFlight != w {
			t.Fatalf("day %d in_flight = %d, want %d (backlog must not count)",
				i+1, curve[i].InFlight, w)
		}
	}
	if curve[0].Opened != 2 || curve[0].Started != 1 {
		t.Fatalf("day1 opened/started = %d/%d, want 2/1", curve[0].Opened, curve[0].Started)
	}
	if curve[2].Closed != 1 {
		t.Fatalf("day3 closed = %d, want 1", curve[2].Closed)
	}
}

func TestWIPCurveCountsSameDayWorkAsInFlight(t *testing.T) {
	// The load-bearing case for this repo: 53.9% of issues open and close
	// within 24h. An end-of-day snapshot would score this day at zero WIP and
	// hide the majority of all work; overlap must show it.
	day := func(d int) time.Time { return time.Date(2026, 8, d, 0, 0, 0, 0, time.UTC) }
	issues := []Issue{{
		Number:    1,
		CreatedAt: day(2).Add(time.Hour),
		ClosedAt:  ptr(day(2).Add(5 * time.Hour)),
	}}
	commits := []Commit{{SHA: "a", When: day(2).Add(2 * time.Hour), Issues: []int{1}, Leaf: "x"}}
	curve := WIPCurve(BuildSpans(issues, commits), day(2), day(2))
	if len(curve) != 1 {
		t.Fatalf("curve len = %d, want 1", len(curve))
	}
	if curve[0].InFlight != 1 {
		t.Fatalf("same-day in_flight = %d, want 1 — a snapshot reading hides half this repo's work",
			curve[0].InFlight)
	}
	if curve[0].Opened != 1 || curve[0].Started != 1 || curve[0].Closed != 1 {
		t.Fatalf("same-day row = %+v, want opened/started/closed all 1", curve[0])
	}
}

func TestAgingWIPOrdersOldestFirstAndExcludesBacklog(t *testing.T) {
	issues := []Issue{
		{Number: 1, CreatedAt: at(0)},
		{Number: 2, CreatedAt: at(0)},
		{Number: 3, CreatedAt: at(0)},                       // unstarted
		{Number: 4, CreatedAt: at(0), ClosedAt: ptr(at(9))}, // closed
	}
	commits := []Commit{
		{SHA: "a", When: at(1), Issues: []int{1}},
		{SHA: "b", When: at(5), Issues: []int{2}},
		{SHA: "c", When: at(2), Issues: []int{4}},
	}
	got := AgingWIP(BuildSpans(issues, commits), at(20), 0)
	if len(got) != 2 {
		t.Fatalf("aging = %d rows, want 2 (unstarted and closed excluded)", len(got))
	}
	if got[0].Issue != 1 || got[1].Issue != 2 {
		t.Fatalf("aging order = %d,%d, want 1,2 (oldest start first)", got[0].Issue, got[1].Issue)
	}
	if len(AgingWIP(BuildSpans(issues, commits), at(20), 1)) != 1 {
		t.Fatalf("limit was not applied")
	}
}
