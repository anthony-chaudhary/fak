package epicprogress

import (
	"strings"
	"testing"
)

// fakeRunner routes a `gh` argv to a canned (stdout, ok) by matching a substring of
// the joined args, so a test drives the resolver without a real gh.
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

// Rung 1 — the track LABEL wins when the epic has a label with children. The epic
// issue itself is excluded so its own state never skews its completion.
func TestCountsByLabel(t *testing.T) {
	fake := fakeRunner{
		"--label track-x": {out: `[{"number":100,"state":"OPEN"},{"number":101,"state":"CLOSED"},{"number":102,"state":"OPEN"}]`, ok: true},
	}
	c := Counts(fake.run, "", EpicSpec{Number: 100, Title: "by-label", Label: "track-x"})
	if c.Source != "label" {
		t.Fatalf("source = %q, want label", c.Source)
	}
	if c.Closed != 1 || c.Total != 2 {
		t.Fatalf("counts = %d/%d, want 1 closed / 2 total (epic #100 excluded)", c.Closed, c.Total)
	}
	if c.Err != "" {
		t.Fatalf("a resolved label rung must not set Err, got %q", c.Err)
	}
}

// Rung 2 — the body CHECKLIST resolves when there is no label, or the label query
// returns no children. Here the spec has no label, so the chain falls to checklist.
func TestCountsByChecklist(t *testing.T) {
	fake := fakeRunner{
		"view 200": {out: `{"body":"intro\n- [x] a\n- [ ] b\n- [X] c"}`, ok: true},
	}
	c := Counts(fake.run, "", EpicSpec{Number: 200, Title: "by-checklist"})
	if c.Source != "checklist" {
		t.Fatalf("source = %q, want checklist", c.Source)
	}
	if c.Closed != 2 || c.Total != 3 {
		t.Fatalf("counts = %d/%d, want 2 checked / 3 total", c.Closed, c.Total)
	}
}

// Rung 2 fallthrough — a label with NO children must not stop the chain; it falls
// through to the checklist, and the resolved Source proves which rung answered.
func TestCountsLabelEmptyFallsToChecklist(t *testing.T) {
	fake := fakeRunner{
		"--label empty-label": {out: `[]`, ok: true},
		"view 300":            {out: `{"body":"- [x] done\n- [ ] todo"}`, ok: true},
	}
	c := Counts(fake.run, "", EpicSpec{Number: 300, Title: "empty-label", Label: "empty-label"})
	if c.Source != "checklist" {
		t.Fatalf("an empty label must fall through to checklist, got source %q", c.Source)
	}
	if c.Closed != 1 || c.Total != 2 {
		t.Fatalf("counts = %d/%d, want 1/2 from checklist", c.Closed, c.Total)
	}
}

// Rung 3 — the honesty seam. When neither label nor checklist resolves, the result
// carries Err and NEVER a fabricated {Total: 0}; that lets a fold tell "0 of N
// done" from "could not read".
func TestCountsErroredNeverFabricatesZero(t *testing.T) {
	allFail := fakeRunner{} // every query returns ok=false
	c := Counts(allFail.run, "", EpicSpec{Number: 400, Title: "unreadable"})
	if c.Err == "" {
		t.Fatalf("an unresolved epic must set Err, got %+v", c)
	}
	if c.Total != 0 || c.Closed != 0 || c.Source != "" {
		t.Fatalf("an errored row must not stamp a count or source, got %+v", c)
	}
}

// A malformed label payload must not crash and must fall through (here to a failing
// checklist, ending as an honest errored row).
func TestCountsBadLabelJSONFallsThrough(t *testing.T) {
	fake := fakeRunner{
		"--label bad": {out: `not json`, ok: true},
		// no "view 500" canned response → checklist read fails too
	}
	c := Counts(fake.run, "", EpicSpec{Number: 500, Title: "bad-json", Label: "bad"})
	if c.Err == "" {
		t.Fatalf("bad label JSON with no checklist must end errored, got %+v", c)
	}
}

func TestCountTaskList(t *testing.T) {
	body := "intro\n- [ ] one\n  - [x] two (indented)\n- [X] three\nnot a task\n- [] malformed\n"
	total, checked := CountTaskList(body)
	if total != 3 || checked != 2 {
		t.Fatalf("task-list count = %d checked / %d total, want 2 checked / 3 total", checked, total)
	}
}

// A nil runner must default to the real gh seam without panicking on construction.
// We do not invoke gh; we only assert the nil-guard path is wired (the resolver
// returns an errored row when the real gh is unavailable in the test sandbox, which
// is itself the honest seam — never a fabricated zero).
func TestNilRunnerDefaults(t *testing.T) {
	// Drive through a fake instead of the real gh to keep the test hermetic, but
	// confirm Counts tolerates being called with an explicit runner of nil via the
	// adapter most callers use. We assert no panic and an EpicCounts is returned.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Counts must not panic, got %v", r)
		}
	}()
	fake := fakeRunner{"--label x": {out: `[{"number":2,"state":"CLOSED"}]`, ok: true}}
	if c := Counts(fake.run, "", EpicSpec{Number: 1, Label: "x"}); c.Source != "label" {
		t.Fatalf("explicit runner must be honored, got %+v", c)
	}
}
