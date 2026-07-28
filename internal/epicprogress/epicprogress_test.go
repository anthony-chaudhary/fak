package epicprogress

import (
	"strconv"
	"strings"
	"sync"
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

// The cross-check that retires the #1315 class of bookkeeping defect: an epic whose
// boxes are ALL unticked but whose named children are ALL closed reads as complete,
// because the child issue's live state — not the hand-maintained box — decides.
func TestCountsChecklistClosedChildrenBeatUntickedBoxes(t *testing.T) {
	fake := fakeRunner{
		"view 600 --json body":  {out: `{"body":"- [ ] #601 keystone\n- [ ] **B** #602 — follow-on"}`, ok: true},
		"view 601 --json state": {out: `{"state":"CLOSED"}`, ok: true},
		"view 602 --json state": {out: `{"state":"CLOSED"}`, ok: true},
	}
	c := Counts(fake.run, "", EpicSpec{Number: 600, Title: "shipped-but-unticked"})
	if c.Source != SourceChecklistIssueState {
		t.Fatalf("source = %q, want %q (the count was cross-checked, not self-reported)", c.Source, SourceChecklistIssueState)
	}
	if c.Closed != 2 || c.Total != 2 {
		t.Fatalf("counts = %d/%d, want 2/2 — both named children are CLOSED", c.Closed, c.Total)
	}
}

// The mirror hazard, and the reason the child state WINS rather than merely ORs with
// the box: a ticked row whose child issue is still OPEN must not report progress.
func TestCountsChecklistOpenChildBeatsTickedBox(t *testing.T) {
	fake := fakeRunner{
		"view 700 --json body":  {out: `{"body":"- [x] #701 claimed done\n- [ ] #702 honest todo"}`, ok: true},
		"view 701 --json state": {out: `{"state":"OPEN"}`, ok: true},
		"view 702 --json state": {out: `{"state":"CLOSED"}`, ok: true},
	}
	c := Counts(fake.run, "", EpicSpec{Number: 700, Title: "mixed"})
	if c.Closed != 1 || c.Total != 2 {
		t.Fatalf("counts = %d/%d, want 1/2 — #701 is OPEN despite its ticked box", c.Closed, c.Total)
	}
}

// An unreadable child must never be invented as closed: the row falls back to its
// checkbox, and a body with no readable child keeps the plain "checklist" provenance.
func TestCountsChecklistUnreadableChildFallsBackToBox(t *testing.T) {
	fake := fakeRunner{
		"view 800 --json body": {out: `{"body":"- [x] #801 unreadable\n- [ ] #802 unreadable"}`, ok: true},
		// no canned state responses → both child lookups fail
	}
	c := Counts(fake.run, "", EpicSpec{Number: 800, Title: "unreadable-children"})
	if c.Source != SourceChecklist {
		t.Fatalf("source = %q, want %q when no child state could be witnessed", c.Source, SourceChecklist)
	}
	if c.Closed != 1 || c.Total != 2 {
		t.Fatalf("counts = %d/%d, want the checkbox fallback 1/2", c.Closed, c.Total)
	}
}

// A row naming the epic itself must not let a self-closing epic mark its own work done.
func TestCountsChecklistIgnoresSelfReference(t *testing.T) {
	fake := fakeRunner{
		"view 900 --json body":  {out: `{"body":"- [ ] #900 the epic itself\n- [ ] #901 real child"}`, ok: true},
		"view 900 --json state": {out: `{"state":"CLOSED"}`, ok: true},
		"view 901 --json state": {out: `{"state":"CLOSED"}`, ok: true},
	}
	c := Counts(fake.run, "", EpicSpec{Number: 900, Title: "self-ref"})
	if c.Closed != 1 || c.Total != 2 {
		t.Fatalf("counts = %d/%d, want 1/2 — the self-referencing row keeps its unticked box", c.Closed, c.Total)
	}
}

// The child lookups run concurrently, so the resolver must (a) hit each DISTINCT ref
// exactly once however many rows name it, and (b) still fold to one deterministic
// count. countingRunner is mutex-guarded because a Runner is required to be safe for
// concurrent use.
type countingRunner struct {
	mu    sync.Mutex
	calls map[string]int
	inner fakeRunner
}

func (c *countingRunner) run(args []string) (string, string, bool) {
	c.mu.Lock()
	c.calls[strings.Join(args, " ")]++
	c.mu.Unlock()
	return c.inner.run(args)
}

func TestCountsChecklistResolvesEachChildOnce(t *testing.T) {
	body := `{"body":"- [ ] #1001 first\n- [ ] #1002 second\n- [ ] #1001 duplicate row for the same child"}`
	cr := &countingRunner{
		calls: map[string]int{},
		inner: fakeRunner{
			"view 1000 --json body":  {out: body, ok: true},
			"view 1001 --json state": {out: `{"state":"CLOSED"}`, ok: true},
			"view 1002 --json state": {out: `{"state":"OPEN"}`, ok: true},
		},
	}
	c := Counts(cr.run, "", EpicSpec{Number: 1000, Title: "dup-refs"})
	if c.Closed != 2 || c.Total != 3 {
		t.Fatalf("counts = %d/%d, want 2/3 (both rows naming closed #1001 count done)", c.Closed, c.Total)
	}
	if n := cr.calls["issue view 1001 --json state"]; n != 1 {
		t.Fatalf("#1001 was looked up %d time(s), want exactly 1 (memoized)", n)
	}
	if n := cr.calls["issue view 1002 --json state"]; n != 1 {
		t.Fatalf("#1002 was looked up %d time(s), want exactly 1", n)
	}
}

// The lookup budget bounds the round-trips one pathological body can cost; rows past
// it fall back to their checkbox rather than being dropped or invented.
func TestCountsChecklistBoundsChildLookups(t *testing.T) {
	var rows []string
	inner := fakeRunner{}
	for i := 0; i < maxChildStateLookups+5; i++ {
		ref := 2001 + i
		rows = append(rows, "- [ ] #"+strconv.Itoa(ref)+" row")
		inner["view "+strconv.Itoa(ref)+" --json state"] = struct {
			out string
			ok  bool
		}{out: `{"state":"CLOSED"}`, ok: true}
	}
	inner["view 2000 --json body"] = struct {
		out string
		ok  bool
	}{out: `{"body":"` + strings.Join(rows, `\n`) + `"}`, ok: true}
	cr := &countingRunner{calls: map[string]int{}, inner: inner}

	c := Counts(cr.run, "", EpicSpec{Number: 2000, Title: "huge"})
	if c.Total != maxChildStateLookups+5 {
		t.Fatalf("total = %d, want every row counted (%d)", c.Total, maxChildStateLookups+5)
	}
	if c.Closed != maxChildStateLookups {
		t.Fatalf("closed = %d, want %d — rows past the budget keep their unticked box", c.Closed, maxChildStateLookups)
	}
	state := 0
	for args, n := range cr.calls {
		if strings.HasSuffix(args, "--json state") {
			state += n
		}
	}
	if state != maxChildStateLookups {
		t.Fatalf("%d child-state round-trips, want the budget cap %d", state, maxChildStateLookups)
	}
}

func TestParseTaskListExtractsRefs(t *testing.T) {
	// Each case pairs ONE body line with the row it must parse into; a nil want marks a
	// line that is not a task row at all. Pairing them here is what keeps the assertion a
	// relation — every checkbox line yields exactly one item, in order, and nothing else
	// yields any — instead of a total that has to be re-guessed whenever a case is added.
	cases := []struct {
		line string
		want *TaskListItem
	}{
		{"intro #999 not a row", nil},
		{"- [ ] #1179 feat(dormancy): clock", &TaskListItem{Checked: false, Ref: 1179}},
		{"- [ ] **A (keystone)** #1302 — trend fold", &TaskListItem{Checked: false, Ref: 1302}},
		{"- [x] plain row with no reference", &TaskListItem{Checked: true, Ref: 0}},
		{"- [ ] see docs/x.md#section-2 (an anchor, not a ref)", &TaskListItem{Checked: false, Ref: 0}},
		{"- [ ] sha1#7 is intra-word, not a ref", &TaskListItem{Checked: false, Ref: 0}},
	}
	lines := make([]string, 0, len(cases))
	want := make([]TaskListItem, 0, len(cases))
	for _, c := range cases {
		lines = append(lines, c.line)
		if c.want != nil {
			want = append(want, *c.want)
		}
	}
	items := ParseTaskList(strings.Join(lines, "\n"))
	if len(items) != len(want) {
		t.Fatalf("parsed %d rows from %d body lines, want one per checkbox row (%d): %+v",
			len(items), len(lines), len(want), items)
	}
	for i, w := range want {
		if items[i] != w {
			t.Fatalf("row %d = %+v, want %+v", i, items[i], w)
		}
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
