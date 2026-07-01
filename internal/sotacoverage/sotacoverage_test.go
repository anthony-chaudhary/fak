package sotacoverage

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sotamatrix"
)

const cleanSource = `package sotamatrix
// Provenance: docs/notes/RESEARCH-backend-sota-matrix-2026-06-26.md
`

func cleanOps() []sotamatrix.Op {
	return []sotamatrix.Op{
		{
			Slug:        "alpha",
			FileGlobs:   []string{"internal/compute/cpuref.go"},
			FakPath:     "internal/compute/cpuref.go (Reference) + internal/model/x.go",
			PrimaryLink: "https://example.com/alpha",
			Oracle:      "cpuref bit-identity",
		},
		{
			Slug:        "beta",
			FileGlobs:   []string{"internal/model/moe_beta*.go"},
			FakPath:     "internal/model/moe_beta.go:42 (loader)",
			PrimaryLink: "https://example.com/beta",
			Oracle:      "HF reference",
		},
	}
}

func TestGradeBands(t *testing.T) {
	for _, tc := range []struct {
		debt int
		want string
	}{{0, "A"}, {1, "B"}, {2, "B"}, {4, "C"}, {9, "D"}, {40, "F"}} {
		if got := GradeLetter(tc.debt); got != tc.want {
			t.Fatalf("GradeLetter(%d)=%q, want %q", tc.debt, got, tc.want)
		}
	}
}

func TestRowsFromOpsPullsFields(t *testing.T) {
	rows := RowsFromOps(cleanOps())
	if len(rows) != 2 || rows[0].Slug != "alpha" || rows[1].Slug != "beta" {
		t.Fatalf("rows = %+v", rows)
	}
	if rows[0].FakPathFile != "internal/compute/cpuref.go" {
		t.Fatalf("alpha path = %q", rows[0].FakPathFile)
	}
	if rows[1].FakPathFile != "internal/model/moe_beta.go" {
		t.Fatalf("beta path = %q", rows[1].FakPathFile)
	}
}

func TestFirstFakPathHandlesDirectoryPointer(t *testing.T) {
	if got := FirstFakPathFile("internal/metalgemm/; internal/model/q.go"); got != "internal/metalgemm" {
		t.Fatalf("dir path = %q", got)
	}
	if got := FirstFakPathFile("internal/compute/cuda.go:1101 (AWQMatMul)"); got != "internal/compute/cuda.go" {
		t.Fatalf("line path = %q", got)
	}
}

func TestGlobMatchNormalizesSeparators(t *testing.T) {
	if !CoveredByMatrix("internal\\model\\beta.go", []string{"internal/model/beta*.go"}) {
		t.Fatal("expected beta glob to match")
	}
	if CoveredByMatrix("internal/model/gamma.go", []string{"internal/model/beta*.go"}) {
		t.Fatal("gamma should not match beta glob")
	}
}

func TestCleanMatrixHasNoHardDebt(t *testing.T) {
	root := makeRepo(t, cleanOps())
	payload := CollectWithOps(root, cleanOps(), cleanSource, "2026-06-30")
	if payload.Error != "" {
		t.Fatalf("error = %q", payload.Error)
	}
	if payload.Corpus.HardDebt != 0 || payload.Corpus.Grade != "A" || !payload.OK {
		t.Fatalf("payload = %+v", payload)
	}
}

func TestMissingFakPathRaisesDebt(t *testing.T) {
	ops := cleanOps()
	ops[0].FakPath = "internal/compute/DELETED.go (Reference)"
	root := makeRepo(t, cleanOps())
	payload := CollectWithOps(root, ops, cleanSource, "2026-06-30")
	by := kpisByName(payload.KPIs)
	if by["fak_path_exists"].Passed || by["fak_path_exists"].Debt != 1 || payload.OK {
		t.Fatalf("payload = %+v", payload)
	}
}

func TestMissingLinkAndOracleRaiseDebt(t *testing.T) {
	ops := cleanOps()
	ops[1].PrimaryLink = ""
	ops[1].Oracle = ""
	root := makeRepo(t, cleanOps())
	payload := CollectWithOps(root, ops, cleanSource, "2026-06-30")
	by := kpisByName(payload.KPIs)
	if by["has_primary_link"].Passed || by["has_primary_link"].Debt != 1 {
		t.Fatalf("link kpi = %+v", by["has_primary_link"])
	}
	if by["has_oracle"].Passed || by["has_oracle"].Debt != 1 {
		t.Fatalf("oracle kpi = %+v", by["has_oracle"])
	}
}

func TestUncoveredKernelFileRaisesTreeCoverageDebt(t *testing.T) {
	root := makeRepo(t, cleanOps())
	writeFile(t, root, "internal/model/moe.go", "package model\n")
	git(t, root, "add", "-A")
	payload := CollectWithOps(root, cleanOps(), cleanSource, "2026-06-30")
	tc := kpisByName(payload.KPIs)["tree_coverage"]
	if tc.Passed || tc.Debt < 1 || !containsItem(tc.Items, "internal/model/moe.go") {
		t.Fatalf("tree coverage = %+v", tc)
	}
}

func TestFreshnessIsSoftAndNeedsToday(t *testing.T) {
	root := makeRepo(t, cleanOps())
	payload := CollectWithOps(root, cleanOps(), cleanSource, "")
	fresh := kpisByName(payload.KPIs)["freshness"]
	if !fresh.Passed || fresh.Hard {
		t.Fatalf("freshness without today = %+v", fresh)
	}
	stale := CollectWithOps(root, cleanOps(), cleanSource, "2027-01-01")
	fresh = kpisByName(stale.KPIs)["freshness"]
	if fresh.Passed || fresh.Debt != 1 || fresh.Hard || stale.Corpus.HardDebt != 0 || !stale.OK {
		t.Fatalf("stale = %+v", stale)
	}
}

func TestLiveMatrixParsesAndCompleteGroupIsClean(t *testing.T) {
	payload := Collect(repoRoot(t), "2026-06-30")
	if payload.Error != "" {
		t.Fatalf("error = %q", payload.Error)
	}
	if payload.Corpus.MatrixRows < 5 {
		t.Fatalf("rows = %d", payload.Corpus.MatrixRows)
	}
	by := kpisByName(payload.KPIs)
	for _, name := range []string{"fak_path_exists", "has_primary_link", "has_oracle"} {
		if !by[name].Passed {
			t.Fatalf("%s failed: %+v", name, by[name])
		}
	}
	if payload.Corpus.DebtByGroup["complete"] != 0 {
		t.Fatalf("complete debt = %+v", payload.Corpus)
	}
}

func TestLiveDebtAndGradeConsistentAndJSONRoundTrips(t *testing.T) {
	payload := Collect(repoRoot(t), "2026-06-30")
	if payload.Corpus.Grade != GradeLetter(payload.Corpus.SOTADebt) {
		t.Fatalf("grade mismatch: %+v", payload.Corpus)
	}
	if payload.OK != (payload.Corpus.HardDebt == 0) {
		t.Fatalf("ok mismatch: %+v", payload.Corpus)
	}
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Payload
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Schema != Schema {
		t.Fatalf("schema = %q", decoded.Schema)
	}
}

func makeRepo(t *testing.T, ops []sotamatrix.Op) string {
	t.Helper()
	root := t.TempDir()
	writeFile(t, root, "go.mod", "module x\n")
	if err := os.Mkdir(filepath.Join(root, "cmd"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, root, "internal/sotamatrix/sotamatrix.go", "package sotamatrix\n")
	writeFile(t, root, "internal/compute/cpuref.go", "package compute\n")
	writeFile(t, root, "internal/model/x.go", "package model\n")
	writeFile(t, root, "internal/model/moe_beta.go", "package model\n")
	git(t, root, "init", "-q")
	git(t, root, "add", "-A")
	return root
}

func writeFile(t *testing.T, root, rel, body string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func git(t *testing.T, root string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func kpisByName(kpis []KPI) map[string]KPI {
	out := map[string]KPI{}
	for _, kpi := range kpis {
		out[kpi.Name] = kpi
	}
	return out
}

func containsItem(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod above %s", dir)
		}
		dir = parent
	}
}
