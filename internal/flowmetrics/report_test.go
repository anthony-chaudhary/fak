package flowmetrics

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// findKPI returns the named KPI, failing the test when it is absent — a KPI
// silently dropped from the fold would otherwise read as "no defect".
func findKPI(t *testing.T, rep Report, name string) KPI {
	t.Helper()
	for _, k := range rep.KPIs {
		if k.KPI == name {
			return k
		}
	}
	t.Fatalf("kpi %q missing from report; got %d kpis", name, len(rep.KPIs))
	return KPI{}
}

// healthyInput is a small world that trips nothing, so each test below can
// perturb exactly one axis and attribute the resulting defect to it.
func healthyInput(now time.Time) Input {
	var issues []Issue
	var commits []Commit
	for i := 1; i <= 10; i++ {
		open := now.Add(-time.Duration(20-i) * time.Hour)
		closed := open.Add(2 * time.Hour)
		issues = append(issues, Issue{Number: i, Title: "leaf", CreatedAt: open, ClosedAt: ptr(closed)})
		// Start 6 minutes after open: queue is small, so flow efficiency is high.
		commits = append(commits, Commit{
			SHA:     string(rune('a' + i)),
			When:    open.Add(6 * time.Minute),
			Subject: "fix(x): thing (fak x)",
			Leaf:    "x",
			Issues:  []int{i},
		})
	}
	return Input{
		Issues:  issues,
		Commits: commits,
		Now:     now,
		Tree: TreeWIP{
			Measured:    true,
			Rev:         "deadbeef",
			UntrackedGo: 2,
			BuildProbed: true,
			Buildable:   true,
		},
		WindowDays: 30,
	}
}

func TestBuildOnHealthyInputIsClean(t *testing.T) {
	rep := Build(healthyInput(base))
	if !rep.OK || rep.Verdict != "OK" {
		t.Fatalf("healthy input graded %s/%v: %s\ndefects: %s",
			rep.Verdict, rep.OK, rep.Reason, allDefects(rep))
	}
	if rep.Corpus["flow_debt"] != 0 {
		t.Fatalf("flow_debt = %v, want 0", rep.Corpus["flow_debt"])
	}
	if rep.Corpus["grade"] != "A" {
		t.Fatalf("grade = %v, want A", rep.Corpus["grade"])
	}
	if rep.Schema != Schema {
		t.Fatalf("schema = %q, want %q", rep.Schema, Schema)
	}
	if len(rep.KPIs) != 8 {
		t.Fatalf("kpis = %d, want 8", len(rep.KPIs))
	}
}

func allDefects(rep Report) string {
	var out []string
	for _, k := range rep.KPIs {
		out = append(out, k.Defects...)
	}
	return strings.Join(out, "\n")
}

func TestUnmeasuredTreeIsNotReportedAsClean(t *testing.T) {
	in := healthyInput(base)
	in.Tree = TreeWIP{} // zero value: gather skipped
	k := findKPI(t, Build(in), "local_wip")
	if k.Value != -1 || k.Score != 0 {
		t.Fatalf("unmeasured tree value/score = %v/%d, want -1/0", k.Value, k.Score)
	}
	if len(k.Soft) == 0 {
		t.Fatalf("unmeasured tree must emit a soft note, got none")
	}
	if len(k.Defects) != 0 {
		t.Fatalf("unmeasured is not a defect, it is unknown; got %v", k.Defects)
	}
}

func TestUnbuildableTreeIsADefectAndZeroScore(t *testing.T) {
	in := healthyInput(base)
	in.Tree.Buildable = false
	in.Tree.BuildError = "internal/gateway/x.go:252:9: not enough arguments\nmore lines"
	k := findKPI(t, Build(in), "local_wip")
	if k.Score != 0 {
		t.Fatalf("unbuildable score = %d, want 0", k.Score)
	}
	if len(k.Defects) == 0 {
		t.Fatalf("unbuildable tree must be a defect")
	}
	joined := strings.Join(k.Defects, " ")
	if !strings.Contains(joined, "wip_") {
		t.Fatalf("defect must cite the //go:build wip_<feature> remedy: %q", joined)
	}
	if strings.Contains(joined, "more lines") {
		t.Fatalf("build error must be trimmed to its first line: %q", joined)
	}
}

func TestUnstartedBacklogTripsItsCeiling(t *testing.T) {
	in := healthyInput(base)
	// 20 open issues, none referenced by a commit: 100% unstarted.
	for i := 100; i < 120; i++ {
		in.Issues = append(in.Issues, Issue{Number: i, CreatedAt: base.Add(-48 * time.Hour)})
	}
	k := findKPI(t, Build(in), "unstarted_backlog")
	if k.Value != 1 {
		t.Fatalf("unstarted share = %v, want 1", k.Value)
	}
	if len(k.Defects) != 1 {
		t.Fatalf("want exactly one unstarted_backlog defect, got %v", k.Defects)
	}
}

func TestArrivalVsServiceFlagsAWriteOnlyQueue(t *testing.T) {
	in := Input{Now: base, WindowDays: 7, Tree: TreeWIP{Measured: true}}
	for i := 1; i <= 5; i++ {
		in.Issues = append(in.Issues, Issue{Number: i, CreatedAt: base.Add(-24 * time.Hour)})
	}
	k := findKPI(t, Build(in), "arrival_vs_service")
	if k.Value != -1 {
		t.Fatalf("value = %v, want -1 (ratio undefined with zero closes)", k.Value)
	}
	if len(k.Defects) != 1 || !strings.Contains(k.Defects[0], "write-only") {
		t.Fatalf("want a write-only-queue defect, got %v", k.Defects)
	}
}

func TestAtomicityFloorTripsOnMultiCommitLandings(t *testing.T) {
	in := healthyInput(base)
	// Give every closed issue a second commit in a second lane.
	for i := 1; i <= 10; i++ {
		in.Commits = append(in.Commits, Commit{
			SHA:    "z" + string(rune('a'+i)),
			When:   in.Issues[i-1].CreatedAt.Add(30 * time.Minute),
			Leaf:   "y",
			Issues: []int{i},
		})
	}
	k := findKPI(t, Build(in), "atomicity")
	if k.Value != 0 {
		t.Fatalf("atomic share = %v, want 0", k.Value)
	}
	if len(k.Defects) != 1 {
		t.Fatalf("want one atomicity defect, got %v", k.Defects)
	}
	if !strings.Contains(k.Detail, "spanned multiple lanes") {
		t.Fatalf("detail should report the multi-lane count: %q", k.Detail)
	}
}

func TestAgingWIPCapTrips(t *testing.T) {
	in := healthyInput(base)
	// 15 issues started 30 days ago and never closed, over the cap of 12.
	for i := 200; i < 215; i++ {
		in.Issues = append(in.Issues, Issue{Number: i, CreatedAt: base.Add(-31 * 24 * time.Hour)})
		in.Commits = append(in.Commits, Commit{
			SHA:    "s" + string(rune('a'+i-200)),
			When:   base.Add(-30 * 24 * time.Hour),
			Leaf:   "x",
			Issues: []int{i},
		})
	}
	rep := Build(in)
	k := findKPI(t, rep, "aging_wip")
	if k.Value != 15 {
		t.Fatalf("stalled = %v, want 15", k.Value)
	}
	if len(k.Defects) != 1 {
		t.Fatalf("want one aging_wip defect, got %v", k.Defects)
	}
	if len(rep.Aging) == 0 || rep.Aging[0].Issue < 200 {
		t.Fatalf("aging list should surface the stalled items oldest-first, got %+v", rep.Aging)
	}
}

func TestFlowEfficiencyFloorTripsWhenWorkSits(t *testing.T) {
	// Opened, sat 100h, then started and closed in 1h: efficiency ~1%.
	in := Input{Now: base, WindowDays: 30, Tree: TreeWIP{Measured: true}}
	for i := 1; i <= 5; i++ {
		open := base.Add(-200 * time.Hour)
		in.Issues = append(in.Issues, Issue{Number: i, CreatedAt: open, ClosedAt: ptr(open.Add(101 * time.Hour))})
		in.Commits = append(in.Commits, Commit{
			SHA: "c" + string(rune('a'+i)), When: open.Add(100 * time.Hour), Leaf: "x", Issues: []int{i},
		})
	}
	k := findKPI(t, Build(in), "flow_efficiency")
	if k.Value > 0.05 {
		t.Fatalf("efficiency = %v, want ~0.01", k.Value)
	}
	if len(k.Defects) != 1 {
		t.Fatalf("want one flow_efficiency defect, got %v", k.Defects)
	}
	// queue_time must NOT also raise a hard defect for the same reality.
	q := findKPI(t, Build(in), "queue_time")
	if len(q.Defects) != 0 {
		t.Fatalf("queue_time must not double-charge the same reality: %v", q.Defects)
	}
	if len(q.Soft) == 0 {
		t.Fatalf("queue_time should note the wait as soft")
	}
}

func TestEmptyCohortsAreUnmeasuredNotZero(t *testing.T) {
	// No issues at all: every time-in-state axis is unknown, and unknown must
	// never be graded as a passing zero.
	rep := Build(Input{Now: base, WindowDays: 30, Tree: TreeWIP{Measured: true}})
	for _, name := range []string{"flow_efficiency", "queue_time", "atomicity"} {
		k := findKPI(t, rep, name)
		if k.Value != -1 {
			t.Fatalf("%s value = %v on an empty cohort, want -1", name, k.Value)
		}
		if len(k.Soft) == 0 {
			t.Fatalf("%s must emit a soft unmeasured note", name)
		}
	}
}

func TestGradeLetterBands(t *testing.T) {
	cases := map[int]string{0: "A", 1: "B", 2: "B", 3: "C", 4: "C", 5: "D", 6: "D", 7: "F", 99: "F"}
	for debt, want := range cases {
		if got := GradeLetter(debt); got != want {
			t.Fatalf("GradeLetter(%d) = %q, want %q", debt, got, want)
		}
	}
}

func TestPayloadMarshalsEmptySlicesNotNull(t *testing.T) {
	// The control pane readers index into defects/soft; a JSON null there is a
	// runtime error in the Python reader, so the empty slice is load-bearing.
	raw, err := json.Marshal(Build(healthyInput(base)))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(raw)
	if strings.Contains(s, `"defects":null`) || strings.Contains(s, `"soft":null`) {
		t.Fatalf("payload emitted a null slice: %s", s)
	}
	var round Report
	if err := json.Unmarshal(raw, &round); err != nil {
		t.Fatalf("round-trip: %v", err)
	}
	if round.Schema != Schema || len(round.KPIs) != 8 {
		t.Fatalf("round-trip lost fields: %+v", round)
	}
}
