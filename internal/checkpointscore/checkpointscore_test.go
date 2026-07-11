package checkpointscore

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeGo(t *testing.T, root, dir, content string) {
	t.Helper()
	full := filepath.Join(root, filepath.FromSlash(dir))
	if err := os.MkdirAll(full, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", full, err)
	}
	if err := os.WriteFile(filepath.Join(full, "impl.go"), []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", full, err)
	}
}

// Scan is a pure tree read: an absent subsystem is not present and satisfies no axis; a present
// one satisfies an axis only when a real (non-test) source token is found.
func TestScanProbesTokensInSource(t *testing.T) {
	root := t.TempDir()
	// resume: only a recovery token present -> recovery yes, status no.
	writeGo(t, root, "internal/resume", "package resume\nconst store = \"resume_drivestate.jsonl\"\n")
	// loopmgr: both a recovery token and a status token.
	writeGo(t, root, "internal/loopmgr", "package loopmgr\n// loops.jsonl with prev_hash\nfunc Summarize() {}\n")
	// sessionjournal has real (token-free) source, and its recovery token lives ONLY in a
	// _test.go file -- which must NOT count toward the axis.
	writeGo(t, root, "internal/sessionjournal", "package sessionjournal\nfunc Fold() {}\n")
	sjDir := filepath.Join(root, "internal", "sessionjournal")
	if err := os.WriteFile(filepath.Join(sjDir, "x_test.go"), []byte("package sessionjournal\n// session-journal BootID\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	got := map[string]Scanned{}
	for _, s := range Scan(root) {
		got[s.Name] = s
	}

	if r := got["resume"]; !r.Present || !r.HasRecovery || r.HasStatus {
		t.Errorf("resume = %+v, want present, recovery, no status", r)
	}
	if l := got["loopmgr"]; !l.HasRecovery || !l.HasStatus {
		t.Errorf("loopmgr = %+v, want recovery and status", l)
	}
	if s := got["sessionjournal"]; !s.Present {
		t.Errorf("sessionjournal dir exists but Present=false")
	} else if s.HasRecovery {
		t.Errorf("sessionjournal recovery token only in _test.go must not count: %+v", s)
	}
	if w := got["watchdoghealth"]; w.Present || w.HasRecovery {
		t.Errorf("absent watchdoghealth reported present/recovered: %+v", w)
	}
}

// Build folds to checkpoint_debt = Σ KPI defects, deterministically.
func TestBuildDebtEqualsDefectsAndIsDeterministic(t *testing.T) {
	root := t.TempDir() // empty: every rostered subsystem absent, planned gap open

	a := Build(root)
	b := Build(root)
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if string(ja) != string(jb) {
		t.Fatalf("Build not deterministic:\n%s\n%s", ja, jb)
	}

	if a.Schema != Schema {
		t.Errorf("schema = %q, want %q", a.Schema, Schema)
	}
	debt, ok := a.Corpus[DebtKey]
	if !ok {
		t.Fatalf("corpus missing %q: %+v", DebtKey, a.Corpus)
	}

	sumDefects := 0
	for _, k := range a.KPIs {
		sumDefects += len(k.Defects)
	}
	// checkpoint_debt is folded into the corpus as an int (scorecard.Fold's headline integer).
	if debt != sumDefects {
		t.Errorf("checkpoint_debt %v != Σ defects %d", debt, sumDefects)
	}
	// Empty tree: 8 subsystems * 2 axes + 1 planned = 17.
	if sumDefects != len(roster)*2+len(planned) {
		t.Errorf("empty-tree defects = %d, want %d", sumDefects, len(roster)*2+len(planned))
	}
	if a.OK {
		t.Errorf("empty tree must not be OK")
	}
}

// A fully-satisfied tree scores clean.
func TestBuildCleanWhenEveryAxisSatisfied(t *testing.T) {
	root := t.TempDir()
	for _, s := range roster {
		// Emit a source file carrying the first recovery and first status token.
		src := "package x\n// " + s.RecoveryTokens[0] + " " + s.StatusTokens[0] + "\n"
		writeGo(t, root, s.Dir, src)
	}
	// Graduate the planned gap with a real durable-store token.
	for _, p := range planned {
		writeGo(t, root, p.Dir, "package x\n// Checkpoint via jsonl journal\nfunc Restore() {}\n")
	}
	payload := Build(root)
	if !payload.OK {
		t.Fatalf("satisfied tree not OK: %s", payload.Reason)
	}
	if payload.Corpus[DebtKey] != 0 {
		t.Errorf("clean tree debt = %v, want 0", payload.Corpus[DebtKey])
	}
}

// Gaps produce content-stable keys (no timestamp) and correctly-shaped ActionItems.
func TestGapsAndActionItemsAreStable(t *testing.T) {
	root := t.TempDir()
	// Give resume a recovery token so it only reds on status; leave the rest absent.
	writeGo(t, root, "internal/resume", "package resume\nconst s = \"resume_plan.json\"\n")

	g1 := Gaps(root)
	g2 := Gaps(root)
	if len(g1) != len(g2) {
		t.Fatalf("Gaps unstable length %d vs %d", len(g1), len(g2))
	}
	for i := range g1 {
		if g1[i].Key() != g2[i].Key() {
			t.Fatalf("gap key unstable at %d: %q vs %q", i, g1[i].Key(), g2[i].Key())
		}
	}

	// resume must NOT appear as a crash_recovery gap (it has a recovery token) but MUST appear
	// as a status gap.
	var resumeRecovery, resumeStatus bool
	for _, g := range g1 {
		if g.Subsystem == "resume" && g.Axis == "crash_recovery" {
			resumeRecovery = true
		}
		if g.Subsystem == "resume" && g.Axis == "status" {
			resumeStatus = true
		}
	}
	if resumeRecovery {
		t.Errorf("resume wrongly flagged for crash_recovery despite a recovery token")
	}
	if !resumeStatus {
		t.Errorf("resume should be flagged for missing status")
	}

	// The planned gap is always present under an empty-ish tree.
	items := ActionItems(g1, "fak checkpoint-scorecard --json")
	if len(items) != len(g1) {
		t.Fatalf("ActionItems dropped rows: %d vs %d", len(items), len(g1))
	}
	var planned bool
	for _, it := range items {
		if !strings.HasPrefix(it.Key, "checkpoint-debt/") {
			t.Errorf("issue key %q lacks stable prefix", it.Key)
		}
		if len(it.Labels) == 0 || it.Labels[0] != "checkpoint-debt" {
			t.Errorf("issue %q missing checkpoint-debt label: %v", it.Key, it.Labels)
		}
		if !strings.Contains(it.DoneCondition, it.Key) {
			t.Errorf("issue %q DoneCondition should cite its own key: %q", it.Key, it.DoneCondition)
		}
		if it.Key == "checkpoint-debt/unified-worker-checkpoint-planned" {
			planned = true
			if it.Grade != "F" {
				t.Errorf("planned gap grade = %q, want F", it.Grade)
			}
		}
	}
	if !planned {
		t.Errorf("expected the unified-worker-checkpoint planned gap in the action items")
	}
}

func TestSlugStableForm(t *testing.T) {
	cases := map[string]string{
		"resume-status":             "resume-status",
		"Unified Worker/Checkpoint": "unified-worker-checkpoint",
		"a__b--c":                   "a-b-c",
	}
	for in, want := range cases {
		if got := slug(in); got != want {
			t.Errorf("slug(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPlannedOpenGraduatesOnDurableStore(t *testing.T) {
	root := t.TempDir()
	p := planned[0]
	if !plannedOpen(root, p) {
		t.Errorf("absent planned dir should be open")
	}
	writeGo(t, root, p.Dir, "package wipcheckpoint\n// no durable store here yet\nfunc Todo() {}\n")
	if !plannedOpen(root, p) {
		t.Errorf("stub without a durable-store token should still be open")
	}
	writeGo(t, root, p.Dir, "package wipcheckpoint\nfunc Checkpoint() {} // writes a jsonl journal\n")
	if plannedOpen(root, p) {
		t.Errorf("a real durable-store impl should graduate the planned gap")
	}
}
