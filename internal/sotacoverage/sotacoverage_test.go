package sotacoverage

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sotamatrix"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
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
	if got := scorecard.StringValue(payload.Corpus["error"]); got != "" {
		t.Fatalf("error = %q", got)
	}
	if debt := scorecard.IntValue(payload.Corpus[DebtKey]); debt != 0 || !payload.OK {
		t.Fatalf("debt = %d, ok = %v (corpus %+v)", debt, payload.OK, payload.Corpus)
	}
	if payload.Corpus["grade"] != "A" || payload.Verdict != "OK" {
		t.Fatalf("grade/verdict = %v/%q", payload.Corpus["grade"], payload.Verdict)
	}
}

// TestPayloadRidesTheSharedKernel pins the migration itself: the card's payload is the
// shared pkg/scorecard envelope (kernel-written value/grade/pressure keys, the kernel's one
// grade table, [] not null KPI lists), not a hand-rolled look-alike.
func TestPayloadRidesTheSharedKernel(t *testing.T) {
	root := makeRepo(t, cleanOps())
	payload := CollectWithOps(root, cleanOps(), cleanSource, "2026-06-30")
	if payload.Schema != Schema {
		t.Fatalf("schema = %q", payload.Schema)
	}
	for _, key := range []string{"value", "value_unit", "score", "legacy_score", "grade", "pressure", "slack", DebtKey} {
		if _, ok := payload.Corpus[key]; !ok {
			t.Fatalf("corpus is missing the kernel key %q: %+v", key, payload.Corpus)
		}
	}
	composite, _ := payload.Corpus["score"].(float64)
	if want := scorecard.GradeStd(composite); payload.Corpus["grade"] != want {
		t.Fatalf("grade %v is not the kernel table's %q for score %v", payload.Corpus["grade"], want, composite)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), `"defects":null`) || strings.Contains(string(raw), `"soft":null`) {
		t.Fatalf("kernel KPI lists must marshal as [], got %s", raw)
	}
}

// TestCorpusKeysSurviveTheKernelMigration pins the control-pane contract the card emitted
// before it rode the kernel: every corpus key a downstream reader keys on is still there.
func TestCorpusKeysSurviveTheKernelMigration(t *testing.T) {
	root := makeRepo(t, cleanOps())
	payload := CollectWithOps(root, cleanOps(), cleanSource, "2026-06-30")
	if got := scorecard.IntValue(payload.Corpus["matrix_rows"]); got != 2 {
		t.Fatalf("matrix_rows = %d", got)
	}
	if got := scorecard.IntValue(payload.Corpus["hard_debt"]); got != scorecard.IntValue(payload.Corpus[DebtKey]) {
		t.Fatalf("hard_debt %d must alias %s %d", got, DebtKey, scorecard.IntValue(payload.Corpus[DebtKey]))
	}
	byGroup, ok := payload.Corpus["debt_by_group"].(map[string]int)
	if !ok {
		t.Fatalf("debt_by_group = %T", payload.Corpus["debt_by_group"])
	}
	for _, group := range []string{"complete", "honest", "fresh"} {
		if _, ok := byGroup[group]; !ok {
			t.Fatalf("debt_by_group is missing %q: %+v", group, byGroup)
		}
	}
}

func TestMissingFakPathRaisesDebt(t *testing.T) {
	ops := cleanOps()
	ops[0].FakPath = "internal/compute/DELETED.go (Reference)"
	root := makeRepo(t, cleanOps())
	payload := CollectWithOps(root, ops, cleanSource, "2026-06-30")
	by := kpisByKey(payload.KPIs)
	if len(by["fak_path_exists"].Defects) != 1 || payload.OK {
		t.Fatalf("payload = %+v", payload)
	}
	if !containsSubstring(by["fak_path_exists"].Defects, "internal/compute/DELETED.go") {
		t.Fatalf("defect does not name the missing path: %+v", by["fak_path_exists"].Defects)
	}
	if scorecard.IntValue(payload.Corpus[DebtKey]) != 1 {
		t.Fatalf("%s = %v", DebtKey, payload.Corpus[DebtKey])
	}
}

func TestMissingLinkAndOracleRaiseDebt(t *testing.T) {
	ops := cleanOps()
	ops[1].PrimaryLink = ""
	ops[1].Oracle = ""
	root := makeRepo(t, cleanOps())
	payload := CollectWithOps(root, ops, cleanSource, "2026-06-30")
	by := kpisByKey(payload.KPIs)
	if len(by["has_primary_link"].Defects) != 1 {
		t.Fatalf("link kpi = %+v", by["has_primary_link"])
	}
	if len(by["has_oracle"].Defects) != 1 {
		t.Fatalf("oracle kpi = %+v", by["has_oracle"])
	}
	if got := scorecard.IntValue(payload.Corpus["debt_by_group"].(map[string]int)["complete"]); got != 2 {
		t.Fatalf("complete group debt = %d", got)
	}
}

func TestUncoveredKernelFileRaisesTreeCoverageDebt(t *testing.T) {
	root := makeRepo(t, cleanOps())
	writeFile(t, root, "internal/model/moe.go", "package model\n")
	git(t, root, "add", "-A")
	payload := CollectWithOps(root, cleanOps(), cleanSource, "2026-06-30")
	tc := kpisByKey(payload.KPIs)["tree_coverage"]
	if len(tc.Defects) < 1 || !containsSubstring(tc.Defects, "internal/model/moe.go") {
		t.Fatalf("tree coverage = %+v", tc)
	}
	if tc.Score >= 100 {
		t.Fatalf("an uncovered file must drop the coverage score, got %v", tc.Score)
	}
}

// TestFreshnessIsSoftAndNeedsToday pins the one deliberate semantic move of the kernel
// migration: staleness is a SOFT nudge, so it lands in KPI.Soft and can never move the
// sota_debt integer or red the gate. Its count is still visible as corpus.soft_debt.
func TestFreshnessIsSoftAndNeedsToday(t *testing.T) {
	root := makeRepo(t, cleanOps())
	payload := CollectWithOps(root, cleanOps(), cleanSource, "")
	fresh := kpisByKey(payload.KPIs)["freshness"]
	if len(fresh.Soft) != 0 || len(fresh.Defects) != 0 {
		t.Fatalf("freshness without today = %+v", fresh)
	}

	stale := CollectWithOps(root, cleanOps(), cleanSource, "2027-01-01")
	fresh = kpisByKey(stale.KPIs)["freshness"]
	if len(fresh.Soft) != 1 || len(fresh.Defects) != 0 {
		t.Fatalf("stale freshness = %+v", fresh)
	}
	if scorecard.IntValue(stale.Corpus[DebtKey]) != 0 || !stale.OK {
		t.Fatalf("a soft signal must not become debt: %+v", stale.Corpus)
	}
	if got := scorecard.IntValue(stale.Corpus["soft_debt"]); got != 1 {
		t.Fatalf("soft_debt = %d, want 1", got)
	}
}

func TestUnreadableWorkspaceFoldsAsARetirableDefect(t *testing.T) {
	payload := Collect(filepath.Join(t.TempDir(), "not-a-repo"), "")
	if payload.OK {
		t.Fatal("a workspace with no go.mod must not fold OK")
	}
	if got := scorecard.StringValue(payload.Corpus["error"]); !strings.Contains(got, "no go.mod") {
		t.Fatalf("corpus.error = %q", got)
	}
	by := kpisByKey(payload.KPIs)
	if len(by["matrix_source"].Defects) != 1 {
		t.Fatalf("matrix_source kpi = %+v", by["matrix_source"])
	}
}

func TestMarkdownAndCompareRideTheKernelRenderers(t *testing.T) {
	root := makeRepo(t, cleanOps())
	payload := CollectWithOps(root, cleanOps(), cleanSource, "2026-06-30")

	md := scorecard.Markdown(payload, MarkdownDoc(payload))
	for _, want := range []string{"---\ntitle: ", "# SOTA-coverage scorecard", "| KPI | value | legacy score | detail |", DebtKey} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
	if scorecard.Markdown(payload, MarkdownDoc(payload)) != md {
		t.Fatal("markdown render is not deterministic")
	}

	// The regression gate must read a prior --json payload and prove the direction of travel.
	prior := map[string]any{"corpus": map[string]any{DebtKey: 3, "pressure": 12.0}}
	line := scorecard.Compare(payload, prior, DebtKey)
	if !strings.Contains(line, "compare: "+DebtKey+" 3 -> 0 (improved by 3)") {
		t.Fatalf("compare line missing the debt drop:\n%s", line)
	}
}

func TestLiveMatrixParsesAndCompleteGroupIsClean(t *testing.T) {
	payload := Collect(repoRoot(t), "2026-06-30")
	if got := scorecard.StringValue(payload.Corpus["error"]); got != "" {
		t.Fatalf("error = %q", got)
	}
	if got := scorecard.IntValue(payload.Corpus["matrix_rows"]); got < 5 {
		t.Fatalf("rows = %d", got)
	}
	by := kpisByKey(payload.KPIs)
	for _, key := range []string{"fak_path_exists", "has_primary_link", "has_oracle"} {
		if len(by[key].Defects) != 0 {
			t.Fatalf("%s failed: %+v", key, by[key])
		}
	}
	if got := payload.Corpus["debt_by_group"].(map[string]int)["complete"]; got != 0 {
		t.Fatalf("complete debt = %d", got)
	}
}

func TestLiveDebtAndGradeConsistentAndJSONRoundTrips(t *testing.T) {
	payload := Collect(repoRoot(t), "2026-06-30")
	debt := 0
	for _, k := range payload.KPIs {
		debt += len(k.Defects)
	}
	if got := scorecard.IntValue(payload.Corpus[DebtKey]); got != debt {
		t.Fatalf("%s %d is not the defect count %d", DebtKey, got, debt)
	}
	if payload.OK != (debt == 0) {
		t.Fatalf("ok mismatch: ok=%v debt=%d", payload.OK, debt)
	}
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	var decoded scorecard.Payload
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

func kpisByKey(kpis []scorecard.KPI) map[string]scorecard.KPI {
	out := map[string]scorecard.KPI{}
	for _, kpi := range kpis {
		out[kpi.Key] = kpi
	}
	return out
}

func containsSubstring(items []string, want string) bool {
	for _, item := range items {
		if strings.Contains(item, want) {
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
