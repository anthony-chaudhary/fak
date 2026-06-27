package fleet

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ---- roster -----------------------------------------------------------------

func TestRosterValidate(t *testing.T) {
	cases := []struct {
		name string
		ro   Roster
		want string // substring expected in the problem list; "" means valid
	}{
		{"good", Roster{Schema: RosterSchema, Boxes: []Box{{ID: "a"}, {ID: "b"}}}, ""},
		{"no-schema-ok", Roster{Boxes: []Box{{ID: "a"}}}, ""},
		{"wrong-schema", Roster{Schema: "other/v9", Boxes: []Box{{ID: "a"}}}, "is not"},
		{"empty", Roster{Schema: RosterSchema}, "no boxes"},
		{"empty-id", Roster{Boxes: []Box{{ID: ""}}}, "empty id"},
		{"space-id", Roster{Boxes: []Box{{ID: "a b"}}}, "whitespace or a path separator"},
		{"slash-id", Roster{Boxes: []Box{{ID: "a/b"}}}, "whitespace or a path separator"},
		{"dup-id", Roster{Boxes: []Box{{ID: "a"}, {ID: "a"}}}, "duplicates"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			probs := tc.ro.Validate()
			joined := strings.Join(probs, " | ")
			if tc.want == "" {
				if len(probs) != 0 {
					t.Fatalf("expected valid, got problems: %s", joined)
				}
				return
			}
			if !strings.Contains(joined, tc.want) {
				t.Fatalf("expected a problem containing %q, got: %s", tc.want, joined)
			}
		})
	}
}

func TestValidateReportsAllProblemsAtOnce(t *testing.T) {
	ro := Roster{Boxes: []Box{{ID: ""}, {ID: "x"}, {ID: "x"}}}
	probs := ro.Validate()
	if len(probs) < 2 {
		t.Fatalf("expected the empty-id AND duplicate problems together, got %d: %v", len(probs), probs)
	}
}

func TestTemplateRoundTrip(t *testing.T) {
	ro := Template(100, "a100x8", "lab-1", "box")
	if len(ro.Boxes) != 100 {
		t.Fatalf("want 100 boxes, got %d", len(ro.Boxes))
	}
	if ro.Boxes[0].ID != "box-001" || ro.Boxes[99].ID != "box-100" {
		t.Fatalf("ids not zero-padded in order: first=%q last=%q", ro.Boxes[0].ID, ro.Boxes[99].ID)
	}
	// Marshal -> reload -> validate: the scaffold is a valid roster as written.
	data, err := json.Marshal(ro)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := LoadRoster(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if probs := got.Validate(); len(probs) != 0 {
		t.Fatalf("template roster did not validate: %v", probs)
	}
	if got.Boxes[0].Class != "a100x8" || got.Boxes[0].Group != "lab-1" {
		t.Fatalf("class/group not carried: %+v", got.Boxes[0])
	}
}

func TestLoadRosterRejectsUnknownField(t *testing.T) {
	_, err := LoadRoster(strings.NewReader(`{"boxes":[{"id":"a","bogus":1}]}`))
	if err == nil {
		t.Fatal("expected an error on an unknown field, got nil")
	}
}

// ---- fold + score -----------------------------------------------------------

// TestFoldScoreFormula pins the documented score blend on a known fleet: 5 boxes,
// 2 live + 1 idle + 1 down all reporting, 1 silent. The arithmetic:
//
//	usable=3/5  reach=4/5  versionCoverage(modal 0.31.0)=2/5
//	100*(0.6*0.6 + 0.2*0.8 + 0.2*0.4) = 60
func TestFoldScoreFormula(t *testing.T) {
	ro := Template(5, "h100x8", "lab-1", "box")
	reps := []Report{
		{State: StateLive, Version: "0.31.0", AgeSec: 3},
		{State: StateLive, Version: "0.31.0"},
		{State: StateIdle, Version: "0.30.0"},
		{State: StateDown},
		{State: StateUnknown, Err: "no report"},
	}
	snap := Fold(ro, reps, FoldOpts{})

	if snap.Score != 60 {
		t.Fatalf("score = %d, want 60", snap.Score)
	}
	if snap.Reachable != 4 {
		t.Fatalf("reachable = %d, want 4", snap.Reachable)
	}
	if snap.ModalVersion != "0.31.0" {
		t.Fatalf("modal version = %q, want 0.31.0", snap.ModalVersion)
	}
	for st, want := range map[State]int{StateLive: 2, StateIdle: 1, StateDown: 1, StateUnknown: 1} {
		if snap.ByState[st] != want {
			t.Fatalf("by_state[%s] = %d, want %d", st, snap.ByState[st], want)
		}
	}
	if snap.Attention[0].Level != "crit" || !strings.Contains(snap.Attention[0].Title, "down or unreachable") {
		t.Fatalf("first attention item should be the down/unreachable crit, got %+v", snap.Attention[0])
	}
	if !hasWarn(snap.Attention, "off the fleet version") {
		t.Fatalf("expected a version-skew warn, got %+v", snap.Attention)
	}
}

func TestScoreEdges(t *testing.T) {
	t.Run("empty-roster", func(t *testing.T) {
		if s := Fold(Roster{}, nil, FoldOpts{}).Score; s != 0 {
			t.Fatalf("empty roster score = %d, want 0", s)
		}
	})
	t.Run("all-live-one-version", func(t *testing.T) {
		ro := Template(10, "cpu", "", "box")
		reps := make([]Report, 10)
		for i := range reps {
			reps[i] = Report{State: StateLive, Version: "1.0.0"}
		}
		snap := Fold(ro, reps, FoldOpts{})
		if snap.Score != 100 {
			t.Fatalf("all-live one-version score = %d, want 100", snap.Score)
		}
		if snap.Attention[0].Level != "ok" {
			t.Fatalf("a perfect fleet should report ok, got %+v", snap.Attention[0])
		}
	})
	t.Run("all-down-but-visible", func(t *testing.T) {
		ro := Template(10, "cpu", "", "box")
		reps := make([]Report, 10)
		for i := range reps {
			reps[i] = Report{State: StateDown}
		}
		if s := Fold(ro, reps, FoldOpts{}).Score; s != 20 {
			t.Fatalf("all-down score = %d, want 20 (reach credit only)", s)
		}
	})
	t.Run("all-silent", func(t *testing.T) {
		ro := Template(10, "cpu", "", "box")
		// nil reports -> the fold pads every box with an unknown.
		if s := Fold(ro, nil, FoldOpts{}).Score; s != 0 {
			t.Fatalf("all-silent score = %d, want 0", s)
		}
	})
}

func TestStaleAttention(t *testing.T) {
	ro := Template(2, "cpu", "", "box")
	reps := []Report{
		{State: StateLive, Version: "1.0.0", AgeSec: 10},
		{State: StateLive, Version: "1.0.0", AgeSec: 4000}, // > 15m default
	}
	snap := Fold(ro, reps, FoldOpts{})
	if !hasWarn(snap.Attention, "silent >") {
		t.Fatalf("expected a stale warn for the 4000s box, got %+v", snap.Attention)
	}
}

func TestModeOfDeterministicTieBreak(t *testing.T) {
	// "a" and "b" tie at 1; the lexically smaller key wins, every run.
	for i := 0; i < 20; i++ {
		k, n := modeOf(map[string]int{"b": 1, "a": 1})
		if k != "a" || n != 1 {
			t.Fatalf("modeOf tie = (%q,%d), want (a,1)", k, n)
		}
	}
}

// ---- file transport ---------------------------------------------------------

func TestReadReportsFileTransport(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("box-001.json", `{"state":"live","version":"1.0.0","age_sec":2}`)
	write("box-003.json", `{"state":"weird"}`) // unknown state -> coerced + flagged
	write("box-004.json", `{`)                 // malformed json
	// box-002 deliberately absent -> unreachable.

	ro := Roster{Boxes: []Box{{ID: "box-001"}, {ID: "box-002"}, {ID: "box-003"}, {ID: "box-004"}}}
	reps := ReadReports(dir, ro)
	if len(reps) != 4 {
		t.Fatalf("want 4 reports, got %d", len(reps))
	}
	if !reps[0].Reachable() || reps[0].Version != "1.0.0" {
		t.Fatalf("box-001 should be reachable live 1.0.0, got %+v", reps[0])
	}
	if reps[1].State != StateUnknown || reps[1].Err == "" {
		t.Fatalf("missing box-002 should be unknown+err, got %+v", reps[1])
	}
	if reps[2].State != StateUnknown || !strings.Contains(reps[2].Err, "unknown state") {
		t.Fatalf("box-003 weird state should be coerced to unknown+err, got %+v", reps[2])
	}
	if reps[3].State != StateUnknown || !strings.Contains(reps[3].Err, "bad report json") {
		t.Fatalf("box-004 malformed json should be unknown+err, got %+v", reps[3])
	}
	// The roster is the identity authority: every report carries its box id.
	for i, want := range []string{"box-001", "box-002", "box-003", "box-004"} {
		if reps[i].ID != want {
			t.Fatalf("reps[%d].ID = %q, want %q", i, reps[i].ID, want)
		}
	}
}

// TestWriteReportRoundTrip is the producer witness: WriteReport writes a report the
// reader accepts as reachable, with the schema forced and age re-stamped, and refuses
// an unsafe id (path traversal) and an unknown state.
func TestWriteReportRoundTrip(t *testing.T) {
	dir := t.TempDir()
	if err := WriteReport(dir, "da-cpu", Report{State: StateLive, Version: "0.31.0", Note: "self-report"}); err != nil {
		t.Fatalf("WriteReport: %v", err)
	}
	reps := ReadReports(dir, Roster{Boxes: []Box{{ID: "da-cpu"}}})
	if !reps[0].Reachable() || reps[0].State != StateLive || reps[0].Version != "0.31.0" {
		t.Fatalf("written report should read back as reachable live 0.31.0, got %+v", reps[0])
	}
	// On-disk schema is forced to the current major even if the caller omitted it.
	data, err := os.ReadFile(filepath.Join(dir, "da-cpu.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), ReportSchema) {
		t.Fatalf("written report should carry the schema tag, got: %s", data)
	}
	// An escaping id is refused — a self-report can never write outside the dir.
	if err := WriteReport(dir, "../evil", Report{State: StateLive}); err == nil {
		t.Fatal("WriteReport must refuse an unsafe id")
	}
	// An unknown state is refused at write time (fail-closed at the producer).
	if err := WriteReport(dir, "x", Report{State: State("bogus")}); err == nil {
		t.Fatal("WriteReport must refuse an unknown state")
	}
}

// ---- render: the 100-box scale witness --------------------------------------

// TestRenderScalesTo100 is the headline scale guarantee: a 100-box fleet renders to
// a BOUNDED summary frame (no per-box line) unless all is asked for, in which case
// every one of the 100 boxes appears. This is what keeps the operator view usable as
// the fleet grows toward the 100-box target.
func TestRenderScalesTo100(t *testing.T) {
	ro := Template(100, "a100x8", "lab-1", "box")
	reps := make([]Report, 100)
	for i := range reps {
		reps[i] = Report{State: StateLive, Version: "0.31.0"}
	}
	snap := Fold(ro, reps, FoldOpts{})
	if snap.Score != 100 {
		t.Fatalf("100 live boxes one version should score 100, got %d", snap.Score)
	}

	summary := Render(snap, false /*all*/, 72)
	if lines := strings.Count(summary, "\n") + 1; lines > 20 {
		t.Fatalf("summary frame for 100 boxes is %d lines, want a bounded view (<=20)", lines)
	}
	if !strings.Contains(summary, "readiness 100/100") || !strings.Contains(summary, "REACHABLE  100/100") {
		t.Fatalf("summary missing the headline numbers:\n%s", summary)
	}
	if boxLines := countBoxRows(summary); boxLines != 0 {
		t.Fatalf("summary view should print no per-box rows, found %d", boxLines)
	}

	full := Render(snap, true /*all*/, 72)
	if boxLines := countBoxRows(full); boxLines != 100 {
		t.Fatalf("--all view should print all 100 box rows, found %d", boxLines)
	}
}

func TestRenderShowsAttentionAndIsDeterministic(t *testing.T) {
	ro := Template(3, "cpu", "", "box")
	reps := []Report{
		{State: StateLive, Version: "1.0.0"},
		{State: StateDown},
		{State: StateLive, Version: "0.9.0"},
	}
	snap := Fold(ro, reps, FoldOpts{})
	a := Render(snap, false, 72)
	b := Render(snap, false, 72)
	if a != b {
		t.Fatal("render is not deterministic for identical input")
	}
	if !strings.Contains(a, "[CRIT]") || !strings.Contains(a, "[WARN]") {
		t.Fatalf("expected both a crit and a warn in the frame:\n%s", a)
	}
}

// ---- review-driven hardening tests ------------------------------------------

// TestEndpointFileSafety is the path-traversal guard: the file transport must refuse
// an endpoint that is not a clean single segment, so an escaping "../secret" reads as
// an error rather than an out-of-dir file.
func TestEndpointFileSafety(t *testing.T) {
	root := t.TempDir()
	// A report sitting OUTSIDE the reports dir that an escaping endpoint would reach.
	if err := os.WriteFile(filepath.Join(root, "secret.json"), []byte(`{"state":"live","version":"9.9.9"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	reportsDir := filepath.Join(root, "reports")
	if err := os.Mkdir(reportsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	ro := Roster{Boxes: []Box{
		{ID: "escape", Endpoint: "../secret"},
		{ID: "sep", Endpoint: "a/b"},
		{ID: "dots", Endpoint: ".."},
		{ID: "space", Endpoint: "a b"},
	}}
	reps := ReadReports(reportsDir, ro)
	for i, b := range ro.Boxes {
		r := reps[i]
		if r.Reachable() || r.Err == "" {
			t.Fatalf("unsafe endpoint %q (%s) must be refused, got %+v", b.Endpoint, b.ID, r)
		}
		if r.Version == "9.9.9" {
			t.Fatalf("escaping endpoint %q READ the out-of-dir file — path traversal not blocked", b.Endpoint)
		}
	}
}

// TestRefUniquenessValidate: two distinct ids resolving to the same report key (a
// shared endpoint) is a silent collision Validate must catch.
func TestRefUniquenessValidate(t *testing.T) {
	ro := Roster{Boxes: []Box{{ID: "a", Endpoint: "shared"}, {ID: "b", Endpoint: "shared"}}}
	if probs := ro.Validate(); !strings.Contains(strings.Join(probs, " | "), "same report key") {
		t.Fatalf("a shared endpoint should be flagged, got %v", probs)
	}
}

// TestReportSchemaGate: a wrong report-schema major is refused (mirroring the roster
// guard); a missing schema is accepted for back-compat.
func TestReportSchemaGate(t *testing.T) {
	dir := t.TempDir()
	must := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	must("v1.json", `{"schema":"fak.fleet.report/v1","state":"live","version":"1.0.0"}`)
	must("v2.json", `{"schema":"fak.fleet.report/v2","state":"live"}`)
	must("none.json", `{"state":"idle"}`)
	reps := ReadReports(dir, Roster{Boxes: []Box{{ID: "v1"}, {ID: "v2"}, {ID: "none"}}})
	if !reps[0].Reachable() {
		t.Fatalf("v1 report should be accepted: %+v", reps[0])
	}
	if reps[1].Reachable() || !strings.Contains(reps[1].Err, "unsupported report schema") {
		t.Fatalf("v2 report should be refused: %+v", reps[1])
	}
	if !reps[2].Reachable() {
		t.Fatalf("a schema-less report should be accepted (back-compat): %+v", reps[2])
	}
}

// TestWireCannotInjectErrOrID: a report file's "err"/"id" keys are reader-owned and
// must not flip a live box or override the roster identity.
func TestWireCannotInjectErrOrID(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "x.json"),
		[]byte(`{"state":"live","version":"1.0.0","err":"injected","id":"evil"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	reps := ReadReports(dir, Roster{Boxes: []Box{{ID: "x"}}})
	if !reps[0].Reachable() {
		t.Fatalf("a wire 'err' must not flip a live box to unreachable: %+v", reps[0])
	}
	if reps[0].ID != "x" {
		t.Fatalf("id must come from the roster, not the wire, got %q", reps[0].ID)
	}
}

// TestStaleFromFileMtime is the freshness backstop: a frozen report file (a dead
// bridge that stopped re-stamping age_sec) must age out via the file's own mtime and
// trip the stale warn, not read green forever.
func TestStaleFromFileMtime(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "frozen.json")
	if err := os.WriteFile(p, []byte(`{"state":"live","version":"1.0.0","age_sec":5}`), 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(p, old, old); err != nil {
		t.Fatal(err)
	}
	ro := Roster{Boxes: []Box{{ID: "frozen"}}}
	reps := ReadReports(dir, ro)
	if reps[0].AgeSec < 3600 {
		t.Fatalf("a 2h-old report file should floor age >= 1h, got age_sec=%v", reps[0].AgeSec)
	}
	if !hasWarn(Fold(ro, reps, FoldOpts{}).Attention, "silent >") {
		t.Fatalf("a frozen (dead-bridge) report should trip the stale warn")
	}
}

// TestRenderWorstCase100 is the scale witness for the HARD inputs the homogeneous test
// can't stress: 50 down boxes (a long crit list), version skew, and 20 distinct
// classes. It asserts the frame stays bounded in height AND that the long lists are
// truncated with a "(+k more)" marker rather than dumping every id/class.
func TestRenderWorstCase100(t *testing.T) {
	boxes := make([]Box, 100)
	reps := make([]Report, 100)
	for i := range boxes {
		boxes[i] = Box{ID: fmt.Sprintf("box-%03d", i+1), Class: fmt.Sprintf("class-%d", i%20)}
		switch {
		case i%2 == 0:
			reps[i] = Report{State: StateDown}
		case i%3 == 0:
			reps[i] = Report{State: StateLive, Version: "0.30.0"}
		default:
			reps[i] = Report{State: StateLive, Version: "0.31.0"}
		}
	}
	snap := Fold(Roster{Boxes: boxes}, reps, FoldOpts{})
	out := Render(snap, false, 72)
	if lines := strings.Count(out, "\n") + 1; lines > 24 {
		t.Fatalf("worst-case summary is %d lines, want bounded (<=24):\n%s", lines, out)
	}
	if !strings.Contains(out, "(+") {
		t.Fatalf("the 50-box down list must be truncated with a '(+k more)' marker:\n%s", out)
	}
	if c := strings.Count(out, "class-"); c > 8 {
		t.Fatalf("CLASS line must cap distinct classes (found %d), not list all 20:\n%s", c, out)
	}
}

// ---- helpers ----------------------------------------------------------------

func hasWarn(items []Item, substr string) bool {
	for _, it := range items {
		if it.Level == "warn" && strings.Contains(it.Title, substr) {
			return true
		}
	}
	return false
}

// countBoxRows counts rendered per-box rows: lines that begin with exactly two
// spaces then a box id (the BOXES table), distinct from the 8-space-indented
// attention detail lines.
func countBoxRows(out string) int {
	n := 0
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "  box-") {
			n++
		}
	}
	return n
}
