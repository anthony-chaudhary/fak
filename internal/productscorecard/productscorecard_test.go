package productscorecard

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testRow(over map[string]any) Row {
	r := Row{
		"id": "r1", "concept": "C", "category": "security", "surface": "product",
		"what_you_get": "a thing", "audience": "developer", "maturity": "shipped",
		"claims_section": "Adjudication", "claims_tag": "SHIPPED",
		"first_command":      "go run ./cmd/fak preflight --tool t --args \"{}\"",
		"first_command_verb": "preflight", "needs_gpu": false, "needs_key": false,
		"witness_path": "internal/adjudicator", "witness": "TestX", "entry_doc": "docs/cli-reference.md",
		"verdict": "durable-product", "gaps": []string{}, "durability_note": "n",
	}
	for k, v := range over {
		r[k] = v
	}
	return r
}

func testTree() Tree {
	present := map[string]bool{"internal/adjudicator": true, "internal/foo": true, "docs/cli-reference.md": true, "GETTING-STARTED.md": true}
	return Tree{
		SectionTags: map[string]map[string]bool{
			"adjudication":                     {"SHIPPED": true},
			"pre-flight ladder + grammar rung": {"SHIPPED": true, "STUB": true},
		},
		Catalog:  []Section{{Section: "Adjudication", Norm: "adjudication"}, {Section: "Tool vDSO", Norm: "tool vdso"}},
		CmdDirs:  map[string]bool{"fak": true, "fanbench": true},
		DocVerbs: map[string]bool{"preflight": true, "serve": true, "agent": true, "bench": true},
		Exists:   func(p string) bool { return present[p] },
	}
}

func testCategories() map[string]bool {
	return map[string]bool{"security": true, "performance": true, "memory": true, "model": true, "tooling": true, "platform": true}
}

func testData(rows []Row) *Data {
	cats := []map[string]any{}
	for c := range testCategories() {
		cats = append(cats, map[string]any{"id": c, "name": c})
	}
	return &Data{Meta: map[string]any{"as_of": "2026-06-24", "fak_version": "t"}, Categories: cats, Rows: rows}
}

func TestProductScorecardPureHelpers(t *testing.T) {
	if GradeLetter(100) != "A" || GradeLetter(85) != "B" || GradeLetter(59) != "F" {
		t.Fatalf("grade ladder mismatch")
	}
	if got := NormSection("## Gateway (`fak serve`)"); got != "gateway" {
		t.Fatalf("norm section = %q", got)
	}
	if got := NormSection("Answer-shape: the witness"); got != "answer-shape" {
		t.Fatalf("colon cut = %q", got)
	}
	if got := NormSection("S7 write-time durability gate (x)"); got != "s7 write-time durability gate" {
		t.Fatalf("plain hyphen should survive: %q", got)
	}
	if !SectionMatch("Adjudication", "adjudication") || SectionMatch("a", "abc") || SectionMatch("Gateway", "tool vdso") {
		t.Fatalf("section match behavior changed")
	}
	if dir, verb := ParseCommand("go run ./cmd/fak preflight --tool t"); dir != "fak" || verb != "preflight" {
		t.Fatalf("parse fak command = %q %q", dir, verb)
	}
	if dir, verb := ParseCommand("go run ./cmd/fanbench -profile research"); dir != "fanbench" || verb != "" {
		t.Fatalf("parse non-fak command = %q %q", dir, verb)
	}
	if dir, verb := ParseCommand("go test ./internal/model"); dir != "" || verb != "" {
		t.Fatalf("parse non-cmd command = %q %q", dir, verb)
	}
}

func TestProductScorecardExpectedVerdictSurfaceGate(t *testing.T) {
	cases := []struct {
		name string
		row  Row
		want string
	}{
		{"durable", testRow(nil), "durable-product"},
		{"key", testRow(map[string]any{"needs_key": true}), "usable-today"},
		{"benchmark", testRow(map[string]any{"surface": "benchmark"}), "usable-today"},
		{"subsystem", testRow(map[string]any{"surface": "subsystem"}), "real-not-easy"},
		{"seam", testRow(map[string]any{"surface": "seam"}), "real-not-easy"},
		{"no command", testRow(map[string]any{"first_command": ""}), "real-not-easy"},
		{"stub", testRow(map[string]any{"maturity": "stub"}), "honest-stub"},
		{"simulated", testRow(map[string]any{"maturity": "simulated"}), "honest-stub"},
		{"concept", testRow(map[string]any{"maturity": "concept"}), "concept-only"},
	}
	for _, tc := range cases {
		got, _ := ExpectedVerdict(tc.row)
		if got != tc.want {
			t.Fatalf("%s: got %q want %q", tc.name, got, tc.want)
		}
	}
}

func TestProductScorecardKPIDefectTriggers(t *testing.T) {
	k := KPIWellFormed([]Row{testRow(nil), testRow(nil)}, testCategories())
	if !contains(k.Defects, "duplicate id") {
		t.Fatalf("expected duplicate id defect: %#v", k.Defects)
	}
	k = KPIWellFormed([]Row{testRow(map[string]any{"surface": "nonsense"})}, testCategories())
	if !contains(k.Defects, "surface") {
		t.Fatalf("expected surface defect: %#v", k.Defects)
	}
	k = KPIWellFormed([]Row{{"id": "x"}}, testCategories())
	if len(k.Defects) <= 5 {
		t.Fatalf("expected many missing-field defects: %#v", k.Defects)
	}

	st := map[string]map[string]bool{"adjudication": {"SHIPPED": true}}
	k = KPIClaimHonest([]Row{testRow(map[string]any{"maturity": "stub", "verdict": "honest-stub", "claims_tag": "SHIPPED"})}, st)
	if !contains(k.Defects, "disagrees") {
		t.Fatalf("expected maturity/tag disagreement: %#v", k.Defects)
	}
	k = KPIClaimHonest([]Row{testRow(map[string]any{"maturity": "stub", "verdict": "honest-stub", "claims_tag": "STUB"})}, st)
	if !contains(k.Defects, "carries only") {
		t.Fatalf("expected ledger overclaim: %#v", k.Defects)
	}
	k = KPIClaimHonest([]Row{testRow(map[string]any{"claims_section": "Pre-flight ladder + grammar rung"})}, testTree().SectionTags)
	if len(k.Defects) != 0 {
		t.Fatalf("membership in mixed section should be honest: %#v", k.Defects)
	}

	tree := testTree()
	k = KPICommandResolves([]Row{testRow(map[string]any{"first_command": "go test ./internal/x", "first_command_verb": "test"})}, tree.CmdDirs, tree.DocVerbs)
	if !contains(k.Defects, "no `./cmd") {
		t.Fatalf("expected no cmd-dir defect: %#v", k.Defects)
	}
	k = KPICommandResolves([]Row{testRow(map[string]any{"first_command": "go run ./cmd/ghost x"})}, tree.CmdDirs, tree.DocVerbs)
	if !contains(k.Defects, "does not exist") {
		t.Fatalf("expected missing cmd dir defect: %#v", k.Defects)
	}
	k = KPICommandResolves([]Row{testRow(map[string]any{"first_command": "go run ./cmd/fak frobnicate"})}, tree.CmdDirs, tree.DocVerbs)
	if !contains(k.Defects, "not documented") {
		t.Fatalf("expected undocumented verb defect: %#v", k.Defects)
	}

	exists := tree.Exists
	if len(KPIWitnessed([]Row{testRow(nil)}, exists).Defects) != 0 {
		t.Fatalf("witness happy path failed")
	}
	if len(KPIWitnessed([]Row{testRow(map[string]any{"witness_path": "internal/ghost"})}, exists).Defects) == 0 {
		t.Fatalf("missing witness path should defect")
	}
	if len(KPIDiscoverable([]Row{testRow(map[string]any{"entry_doc": "docs/ghost.md"})}, exists).Defects) == 0 {
		t.Fatalf("missing entry doc should defect")
	}
	if len(KPIDiscoverable([]Row{testRow(map[string]any{"maturity": "concept", "verdict": "concept-only", "entry_doc": ""})}, exists).Defects) != 0 {
		t.Fatalf("concept should be discoverability-exempt")
	}
	k = KPIVerdictConsistency([]Row{testRow(map[string]any{"surface": "subsystem", "verdict": "durable-product"})})
	if !contains(k.Defects, "implies 'real-not-easy'") {
		t.Fatalf("expected verdict overclaim: %#v", k.Defects)
	}
}

func TestProductScorecardCoverageAndClaimsParsing(t *testing.T) {
	cov := CoverageReport([]Section{{Section: "Adjudication", Norm: "adjudication"}, {Section: "Tool vDSO", Norm: "tool vdso"}}, []Row{testRow(nil)})
	if intValue(cov["covered"]) != 1 || intValue(cov["coverage_debt"]) != 1 {
		t.Fatalf("coverage mismatch: %#v", cov)
	}
	unc := cov["uncovered"].([]Section)
	if unc[0].Norm != "tool vdso" {
		t.Fatalf("uncovered = %#v", unc)
	}
	text := "## The product\n- [SHIPPED] a\n- [STUB] b\n## What fak is NOT\n- [STUB] c\n## Prior-art posture\n- [SHIPPED] d\n"
	catalog, tags := ParseClaimsCatalog(text)
	if len(catalog) != 1 || catalog[0].Norm != "the product" {
		t.Fatalf("catalog = %#v", catalog)
	}
	if !tags["the product"]["SHIPPED"] || !tags["the product"]["STUB"] {
		t.Fatalf("tags = %#v", tags)
	}
}

func TestProductScorecardLoadDataDirMergesModularFiles(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, filepath.Join(dir, "_meta.json"), map[string]any{
		"meta":       map[string]any{"as_of": "2026-06-24", "fak_version": "t"},
		"categories": []map[string]any{{"id": "security", "name": "Sec"}},
	})
	writeJSON(t, filepath.Join(dir, "rows-security.json"), map[string]any{"rows": []Row{testRow(nil)}})
	data, errText := LoadDataDir(dir)
	if errText != "" || data == nil {
		t.Fatalf("load data dir: %s", errText)
	}
	if len(data.Rows) != 1 || stringValue(data.Rows[0], "_source_file") != "rows-security.json" {
		t.Fatalf("loaded rows = %#v", data.Rows)
	}
}

func TestProductScorecardBuildPayloadScenarios(t *testing.T) {
	tree := testTree()
	tree.Catalog = []Section{{Section: "Adjudication", Norm: "adjudication"}}
	p := BuildPayload(".", testData([]Row{testRow(nil)}), tree, "")
	if !p.OK || p.Verdict != "OK" || intValue(p.Corpus["product_debt"]) != 0 || stringAny(p.Corpus["grade"]) != "A" {
		t.Fatalf("zero-debt payload = %#v", p)
	}
	if floatValue(mapValue(p.Corpus["coverage"])["coverage_pct"]) != 100.0 {
		t.Fatalf("coverage = %#v", p.Corpus["coverage"])
	}
	p = BuildPayload(".", testData([]Row{testRow(nil)}), testTree(), "")
	if p.OK || p.Finding != "coverage_debt" || intValue(p.Corpus["coverage_debt"]) != 1 || intValue(p.Corpus["honesty_defects"]) != 0 {
		t.Fatalf("coverage-debt payload = %#v", p)
	}
	p = BuildPayload(".", testData([]Row{testRow(map[string]any{"surface": "subsystem", "verdict": "durable-product"})}), tree, "")
	if p.OK || p.Finding != "product_debt" || intValue(p.Corpus["honesty_defects"]) < 1 {
		t.Fatalf("honesty-defect payload = %#v", p)
	}
	p = BuildPayload(".", nil, tree, "missing data")
	if p.OK || p.Verdict != "AUDIT_ERROR" {
		t.Fatalf("audit error payload = %#v", p)
	}
}

func TestProductScorecardManagedContextSLODebt(t *testing.T) {
	tree := testTree()
	tree.Catalog = []Section{{Section: "Adjudication", Norm: "adjudication"}}
	data := testData([]Row{testRow(nil)})
	for _, req := range requiredManagedContextSLOs {
		slo := req
		slo.Status = "pass"
		slo.Source = "unit fixture"
		slo.Detail = "fixture passes"
		if req.ID == "context_visibility" {
			slo.Status = "fail"
			slo.Detail = "debug report does not yet show known/unknown/assumed buckets"
			slo.NextAction = "add the visible managed-context report fixture"
		}
		data.ManagedContextSLOs = append(data.ManagedContextSLOs, slo)
	}

	p := BuildPayload(".", data, tree, "")
	if p.OK || p.Finding != "managed_context_debt" || intValue(p.Corpus["managed_context_debt"]) != 1 || intValue(p.Corpus["product_debt"]) != 1 {
		t.Fatalf("managed-context payload = %#v, want one hard SLO debt", p)
	}
	mc := mapValue(p.Corpus["managed_context"])
	if intValue(mc["debt"]) != 1 || intValue(mc["total"]) != len(requiredManagedContextSLOs) || intValue(mc["passed"]) != len(requiredManagedContextSLOs)-1 {
		t.Fatalf("managed-context report = %#v, want one failing SLO", mc)
	}
	out := Render(p)
	for _, want := range []string{"managed-context SLOs:", "managed-context SLO work-list:", "context_visibility [fail]"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}

func TestProductScorecardRenderersAndDocFolder(t *testing.T) {
	tree := testTree()
	tree.Catalog = []Section{{Section: "Adjudication", Norm: "adjudication"}}
	p := BuildPayload(".", testData([]Row{testRow(nil)}), tree, "")
	for name, text := range map[string]string{
		"render":   Render(p),
		"critical": RenderCritical(p),
		"gaps":     RenderGaps(p),
		"chart":    RenderChart(p),
		"compare":  RenderCompare(payloadMap(p), p),
	} {
		if strings.TrimSpace(text) == "" {
			t.Fatalf("%s rendered empty", name)
		}
	}
	chart := RenderChart(p)
	for _, want := range []string{"verdict ladder", "verdict mix by category", "can a person run it today?", "coverage", "legend:"} {
		if !strings.Contains(chart, want) {
			t.Fatalf("chart missing %q:\n%s", want, chart)
		}
	}
	files := RenderDocFolder(p, "2026-06-24")
	if !strings.Contains(files["README.md"], "Product scorecard") || !strings.Contains(files["README.md"], "Standing at a glance") {
		t.Fatalf("doc folder readme missing expected content")
	}
}

func TestProductScorecardLiveRealDataSmoke(t *testing.T) {
	root := repoRootFromTest(t)
	if _, err := os.Stat(filepath.Join(root, DataDirRel)); err != nil {
		t.Skip("product scorecard data not present")
	}
	p := Collect(root, "")
	if p.Schema != Schema {
		t.Fatalf("schema = %q", p.Schema)
	}
	if p.Verdict == "AUDIT_ERROR" {
		t.Fatalf("live data audit error: %s", p.Reason)
	}
	if !p.OK {
		t.Skipf("current shared tree product scorecard is not green: %s", p.Reason)
	}
	if intValue(p.Corpus["product_debt"]) != 0 || intValue(mapValue(p.Corpus["coverage"])["coverage_debt"]) != 0 {
		t.Fatalf("live data not complete/honest: %s", p.Reason)
	}
}

func writeJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func contains(items []string, sub string) bool {
	for _, it := range items {
		if strings.Contains(it, sub) {
			return true
		}
	}
	return false
}

func payloadMap(p Payload) map[string]any {
	var out map[string]any
	b, _ := json.Marshal(p)
	_ = json.Unmarshal(b, &out)
	return out
}

func repoRootFromTest(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(wd, "go.mod")); err == nil {
			return wd
		}
		parent := filepath.Dir(wd)
		if parent == wd {
			t.Fatal("repo root not found")
		}
		wd = parent
	}
}
