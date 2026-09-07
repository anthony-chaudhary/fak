package dogfoodcoverage

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func findTestRepoRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	dir := wd
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("cannot find repo root from %s", wd)
		}
		dir = parent
	}
}

func TestEvaluateOnRealRepoHasSchemaAndHardKPIs(t *testing.T) {
	root := findTestRepoRoot(t)
	payload := Evaluate(root, map[string]string{})

	if payload.Schema != Schema {
		t.Errorf("schema = %q, want %q", payload.Schema, Schema)
	}
	if payload.Coverage < 0.0 || payload.Coverage > 100.0 {
		t.Errorf("coverage = %f, want [0.0, 100.0]", payload.Coverage)
	}
	switch payload.Grade {
	case "A", "B", "C", "D", "F":
	default:
		t.Errorf("unexpected grade %q", payload.Grade)
	}

	keys := make(map[string]bool)
	for _, k := range payload.KPIs {
		keys[k.Key] = true
	}

	expectedHard := []string{
		"fleet_leaf_guarded",
		"fak_bin_resolvable",
		"guard_default_on",
		"issue_dispatch_wired",
		"guard_verb_present",
	}
	for _, exp := range expectedHard {
		if !keys[exp] {
			t.Errorf("missing hard KPI key %q", exp)
		}
	}
}

func TestGuardDisabledEnvLowersCoverageAndRaisesDebt(t *testing.T) {
	root := findTestRepoRoot(t)
	on := Evaluate(root, map[string]string{})
	off := Evaluate(root, map[string]string{"FLEET_DOGFOOD_GUARD": "0"})

	if off.DogfoodDebt <= on.DogfoodDebt {
		t.Errorf("off.DogfoodDebt (%d) should be > on.DogfoodDebt (%d)", off.DogfoodDebt, on.DogfoodDebt)
	}
	if off.Coverage >= on.Coverage {
		t.Errorf("off.Coverage (%f) should be < on.Coverage (%f)", off.Coverage, on.Coverage)
	}

	offKeys := make(map[string]bool)
	for _, k := range off.KPIs {
		offKeys[k.Key] = k.OK
	}
	if offKeys["guard_default_on"] {
		t.Errorf("guard_default_on should be false when FLEET_DOGFOOD_GUARD=0")
	}
}

func TestEvaluateDegradesWhenDispatchWorkerMissing(t *testing.T) {
	td := t.TempDir()
	if err := os.Mkdir(filepath.Join(td, "tools"), 0755); err != nil {
		t.Fatalf("mkdir tools: %v", err)
	}
	payload := Evaluate(td, map[string]string{})

	kpis := make(map[string]KPI)
	for _, k := range payload.KPIs {
		kpis[k.Key] = k
	}

	for _, key := range []string{"fleet_leaf_guarded", "fak_bin_resolvable", "guard_default_on"} {
		k, ok := kpis[key]
		if !ok {
			t.Fatalf("missing KPI %q", key)
		}
		if k.OK {
			t.Errorf("%s: OK should be false", key)
		}
		if !k.Hard {
			t.Errorf("%s: Hard should be true", key)
		}
		if !strings.Contains(k.Evidence, "dispatch surface moved") {
			t.Errorf("%s: Evidence %q should contain 'dispatch surface moved'", key, k.Evidence)
		}
	}
}

func TestEvaluateDegradesWhenDispatchWorkerSymbolRenamed(t *testing.T) {
	td := t.TempDir()
	toolsDir := filepath.Join(td, "tools")
	if err := os.Mkdir(toolsDir, 0755); err != nil {
		t.Fatalf("mkdir tools: %v", err)
	}
	dwPath := filepath.Join(toolsDir, "dispatch_worker.py")
	if err := os.WriteFile(dwPath, []byte("def build_command(*a, **k):\n    return ['claude']\n"), 0644); err != nil {
		t.Fatalf("write dispatch_worker.py: %v", err)
	}

	payload := Evaluate(td, map[string]string{})

	kpis := make(map[string]KPI)
	for _, k := range payload.KPIs {
		kpis[k.Key] = k
	}

	for _, key := range []string{"fleet_leaf_guarded", "fak_bin_resolvable", "guard_default_on"} {
		k, ok := kpis[key]
		if !ok {
			t.Fatalf("missing KPI %q", key)
		}
		if k.OK {
			t.Errorf("%s: OK should be false", key)
		}
		if !k.Hard {
			t.Errorf("%s: Hard should be true", key)
		}
		if !strings.Contains(k.Evidence, "dispatch surface moved") {
			t.Errorf("%s: Evidence %q should contain 'dispatch surface moved'", key, k.Evidence)
		}
	}
}

func TestCountAuditRowsCountsNonblankJSONLLines(t *testing.T) {
	td := t.TempDir()
	cfg := t.TempDir()
	jdir := filepath.Join(td, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(jdir, 0755); err != nil {
		t.Fatalf("mkdir guard-audit: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jdir, "gateway-claude.jsonl"), []byte("{\"seq\":1}\n{\"seq\":2}\n\n{\"seq\":3}\n"), 0644); err != nil {
		t.Fatalf("write gateway-claude.jsonl: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jdir, "docs-claude.jsonl"), []byte("{\"seq\":1}\n"), 0644); err != nil {
		t.Fatalf("write docs-claude.jsonl: %v", err)
	}

	rows, journals := CountAuditRows(td, map[string]string{"XDG_CONFIG_HOME": cfg})
	if rows != 4 {
		t.Errorf("rows = %d, want 4", rows)
	}
	if journals != 2 {
		t.Errorf("journals = %d, want 2", journals)
	}
}

func TestDiagnoseAuditGapDistinguishesTheThreeBlanks(t *testing.T) {
	td := t.TempDir()

	// (1) no journal dir at all
	msgNoDir := DiagnoseAuditGap(td)
	if !strings.Contains(msgNoDir, "no guarded worker") {
		t.Errorf("msgNoDir = %q, want 'no guarded worker'", msgNoDir)
	}

	// (2) dir but no files
	jdir := filepath.Join(td, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(jdir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	msgEmptyDir := DiagnoseAuditGap(td)
	if !strings.Contains(msgEmptyDir, "never exercised") {
		t.Errorf("msgEmptyDir = %q, want 'never exercised'", msgEmptyDir)
	}

	// (3) dir + only-blank files
	if err := os.WriteFile(filepath.Join(jdir, "docs-claude.jsonl"), []byte("\n\n"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	msgBlankFiles := DiagnoseAuditGap(td)
	if !strings.Contains(msgBlankFiles, "all blank") {
		t.Errorf("msgBlankFiles = %q, want 'all blank'", msgBlankFiles)
	}
	if !strings.Contains(msgBlankFiles, "auth/login") {
		t.Errorf("msgBlankFiles = %q, want 'auth/login'", msgBlankFiles)
	}

	// Distinct messages
	set := map[string]bool{
		msgNoDir:      true,
		msgEmptyDir:   true,
		msgBlankFiles: true,
	}
	if len(set) != 3 {
		t.Errorf("expected 3 distinct diagnostic messages, got %d", len(set))
	}

	// When rows exist, there is no gap
	if err := os.WriteFile(filepath.Join(jdir, "g.jsonl"), []byte("{\"seq\":1}\n"), 0644); err != nil {
		t.Fatalf("write g.jsonl: %v", err)
	}
	rows, _ := CountAuditRows(td, nil)
	if rows != 1 {
		t.Errorf("rows = %d, want 1", rows)
	}
	msgWithRows := DiagnoseAuditGap(td)
	if msgWithRows != "" {
		t.Errorf("DiagnoseAuditGap when rows exist = %q, want empty", msgWithRows)
	}
}

func TestGradeLadder(t *testing.T) {
	cases := []struct {
		coverage float64
		debt     int
		want     string
	}{
		{100.0, 0, "A"},
		{95.0, 0, "A"},
		{70.0, 0, "B"},
		{70.0, 1, "C"},
		{50.0, 2, "D"},
		{10.0, 5, "F"},
	}
	for _, tc := range cases {
		got := Grade(tc.coverage, tc.debt)
		if got != tc.want {
			t.Errorf("Grade(%.1f, %d) = %q, want %q", tc.coverage, tc.debt, got, tc.want)
		}
	}
}

func TestFormatReport(t *testing.T) {
	root := findTestRepoRoot(t)
	payload := Evaluate(root, map[string]string{})
	out := FormatReport(payload)
	if !strings.Contains(out, "dogfood-coverage") {
		t.Errorf("output missing 'dogfood-coverage': %s", out)
	}
	if !strings.Contains(out, "grade") {
		t.Errorf("output missing 'grade': %s", out)
	}
}

func TestCountAuditRowsWithUserJournal(t *testing.T) {
	td := t.TempDir()
	cfg := t.TempDir()
	// Must have .git present to activate user journal
	if err := os.WriteFile(filepath.Join(td, ".git"), []byte("gitdir: dummy"), 0644); err != nil {
		t.Fatalf("write .git: %v", err)
	}
	userDir := filepath.Join(cfg, "fak")
	if err := os.MkdirAll(userDir, 0755); err != nil {
		t.Fatalf("mkdir userDir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(userDir, "guard-audit.jsonl"), []byte("{\"u\":1}\n{\"u\":2}\n"), 0644); err != nil {
		t.Fatalf("write user journal: %v", err)
	}

	rows, journals := CountAuditRows(td, map[string]string{"XDG_CONFIG_HOME": cfg})
	if rows != 2 {
		t.Errorf("rows = %d, want 2", rows)
	}
	if journals != 1 {
		t.Errorf("journals = %d, want 1", journals)
	}
}

func TestCheckGateBehavior(t *testing.T) {
	root := findTestRepoRoot(t)
	// Guard on => debt 0 => check passes
	repOn := Evaluate(root, map[string]string{})
	if repOn.DogfoodDebt > 0 {
		t.Fatalf("expected debt 0 when guard is on, got %d", repOn.DogfoodDebt)
	}
	// Guard off => debt > 0 => check fails
	repOff := Evaluate(root, map[string]string{"FLEET_DOGFOOD_GUARD": "0"})
	if repOff.DogfoodDebt == 0 {
		t.Fatalf("expected debt > 0 when FLEET_DOGFOOD_GUARD=0, got 0")
	}
}
