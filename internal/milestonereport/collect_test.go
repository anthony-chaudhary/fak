package milestonereport

import (
	"strings"
	"testing"
)

func TestCountTaskList(t *testing.T) {
	body := "intro\n- [ ] one\n  - [x] two (indented)\n- [X] three\nnot a task\n- [] malformed\n"
	total, checked := countTaskList(body)
	if total != 3 || checked != 2 {
		t.Fatalf("task-list count = %d/%d, want 3 total / 2 checked", checked, total)
	}
}

// fakeRunner routes a `gh` argv to a canned (stdout, ok) by matching a substring of
// the joined args, so a test drives Collect without a real gh.
type fakeRunner map[string]struct {
	out string
	ok  bool
}

func (f fakeRunner) run(args []string) (string, string, bool) {
	joined := strings.Join(args, " ")
	for key, resp := range f {
		if strings.Contains(joined, key) {
			return resp.out, "", resp.ok
		}
	}
	return "", "no canned response", false
}

func TestCollectWithInjectedRunner(t *testing.T) {
	// One epic resolves via label, one via checklist, one fails entirely. The maturity
	// dimension is pure; the roadmap is driven by the fake runner.
	specs := []EpicSpec{
		{Number: 100, Title: "by-label", Label: "track-x"},
		{Number: 200, Title: "by-checklist"},
		{Number: 300, Title: "unreadable"},
	}
	saved := TrackedEpics
	TrackedEpics = specs
	defer func() { TrackedEpics = saved }()

	fake := fakeRunner{
		// label query for #100: 1 closed + 1 open child (plus the epic itself, excluded)
		"--label track-x": {out: `[{"number":100,"state":"OPEN"},{"number":101,"state":"CLOSED"},{"number":102,"state":"OPEN"}]`, ok: true},
		// checklist query for #200
		"view 200": {out: `{"body":"- [x] a\n- [ ] b"}`, ok: true},
		// #300: both label (none set) and checklist fail to read
		"view 300": {out: "", ok: false},
	}

	maturity, epics := Collect("", fake.run)
	if maturity.Err != "" {
		t.Fatalf("maturity must measure from the live grid, got err=%q", maturity.Err)
	}
	if epics.Tracked != 3 {
		t.Fatalf("tracked = %d, want 3", epics.Tracked)
	}
	if epics.Measured != 2 {
		t.Fatalf("measured = %d, want 2 (#100 label + #200 checklist)", epics.Measured)
	}
	if epics.PartialNote == "" {
		t.Fatalf("one unreadable epic must record a partial note, got %+v", epics)
	}
	if epics.Err != "" {
		t.Fatalf("a partial failure must NOT error the dimension, got %q", epics.Err)
	}

	byNum := map[int]EpicRow{}
	for _, row := range epics.Rows {
		byNum[row.Number] = row
	}
	if r := byNum[100]; r.Source != "label" || r.Closed != 1 || r.Total != 2 {
		t.Fatalf("#100 should resolve via label 1/2, got %+v", r)
	}
	if r := byNum[200]; r.Source != "checklist" || r.Closed != 1 || r.Total != 2 {
		t.Fatalf("#200 should resolve via checklist 1/2, got %+v", r)
	}
	if r := byNum[300]; r.Err == "" {
		t.Fatalf("#300 should be an errored row, got %+v", r)
	}
}

func TestCollectAllFailErrorsDimension(t *testing.T) {
	specs := []EpicSpec{{Number: 1, Title: "x"}}
	saved := TrackedEpics
	TrackedEpics = specs
	defer func() { TrackedEpics = saved }()

	allFail := fakeRunner{} // every query returns ok=false
	_, epics := Collect("", allFail.run)
	if epics.Err == "" || epics.OK {
		t.Fatalf("all epics failing must error the roadmap dimension, got %+v", epics)
	}
}
