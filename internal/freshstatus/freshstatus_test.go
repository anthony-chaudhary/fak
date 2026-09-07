package freshstatus

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var testNow = time.Date(2026, 6, 25, 12, 0, 0, 0, time.UTC)

func testCatalog(runs []map[string]any, machines int) map[string]any {
	machMap := make(map[string]any)
	for i := 0; i < machines; i++ {
		machMap[fmt.Sprintf("m%d", i)] = map[string]any{}
	}
	var rawRuns []any
	for _, r := range runs {
		rawRuns = append(rawRuns, r)
	}
	return map[string]any{
		"runs":     rawRuns,
		"machines": machMap,
	}
}

func testPanes(gitV, benchV, workV, indV string, benchStale bool) []map[string]any {
	return []map[string]any{
		{
			"key":                "git",
			"label":              "git",
			"verdict":            gitV,
			"ok":                 gitV == "OK",
			"reason":             "abc (main), clean",
			"sha":                "abc",
			"branch":             "main",
			"dirty":              0,
			"ahead":              0,
			"behind":             0,
			"push_lag_seconds":   nil,
			"oldest_unpushed_ts": nil,
			"dirty_lag_seconds":  nil,
			"oldest_dirty_path":  nil,
			"oldest_dirty_mtime": nil,
			"push_lag_stale":     false,
			"dirty_lag_stale":    false,
		},
		{
			"key":     "benchmarks",
			"label":   "benchmarks",
			"verdict": benchV,
			"ok":      benchV == "OK",
			"reason":  "55 runs",
			"stale":   benchStale,
			"provenance": map[string]int{
				"measured":   1,
				"modeled":    1,
				"functional": 1,
				"unknown":    1,
			},
			"runs":     55,
			"machines": 5,
			"newest":   "x",
			"age_days": 1.0,
		},
		{
			"key":         "work",
			"label":       "work",
			"verdict":     workV,
			"ok":          workV != "ACTION",
			"reason":      "0 plans",
			"total_plans": 0,
			"shipped":     nil,
			"remaining":   nil,
		},
		{
			"key":         "industry",
			"label":       "industry",
			"verdict":     indV,
			"ok":          indV == "OK",
			"reason":      "parity 0",
			"parity_debt": 0,
			"grade":       "A",
		},
	}
}

func TestParseRunTSCompactAndISO(t *testing.T) {
	want := time.Date(2026, 6, 25, 5, 0, 17, 0, time.UTC)
	if got, ok := ParseRunTS("20260625T050017Z"); !ok || !got.Equal(want) {
		t.Errorf("compact parse got %v (%v), want %v", got, ok, want)
	}
	if got, ok := ParseRunTS("2026-06-25T05:00:17Z"); !ok || !got.Equal(want) {
		t.Errorf("ISO parse got %v (%v), want %v", got, ok, want)
	}
	if _, ok := ParseRunTS("garbage"); ok {
		t.Errorf("expected garbage to fail parse")
	}
	if _, ok := ParseRunTS(""); ok {
		t.Errorf("expected empty string to fail parse")
	}
}

func TestFoldBenchmarksCountsAndProvenance(t *testing.T) {
	cat := testCatalog([]map[string]any{
		{"timestamp": "20260625T050017Z", "model": "qwen", "tags": []any{"radix-benchmark"}, "run_id": "radix-x"},
		{"timestamp": "20260624T050017Z", "model": "webvoyager", "tags": []any{"turn-tax"}, "run_id": "turn-tax-x"},
		{"timestamp": "20260623T050017Z", "model": "experiment", "tags": []any{"parity"}, "run_id": "parity-x"},
		{"timestamp": "20260622T050017Z", "model": "experiment", "tags": []any{"experiment"}, "run_id": "bare-x"},
	}, 2)

	pane := FoldBenchmarks(cat, testNow, StaleDays)
	if pane["runs"] != 4 || pane["machines"] != 2 {
		t.Errorf("runs/machines = %v/%v, want 4/2", pane["runs"], pane["machines"])
	}

	prov, ok := pane["provenance"].(map[string]int)
	if !ok {
		t.Fatalf("provenance not map[string]int: %v", pane["provenance"])
	}
	wantProv := map[string]int{"measured": 1, "modeled": 1, "functional": 1, "unknown": 1}
	for k, w := range wantProv {
		if prov[k] != w {
			t.Errorf("provenance[%s] = %d, want %d", k, prov[k], w)
		}
	}

	if pane["verdict"] != "OK" || pane["ok"] != true {
		t.Errorf("verdict/ok = %v/%v, want OK/true", pane["verdict"], pane["ok"])
	}
	if pane["newest"] != "20260625T050017Z" {
		t.Errorf("newest = %v, want 20260625T050017Z", pane["newest"])
	}
	ageDays, ok := toFloat(pane["age_days"])
	if !ok || ageDays >= 1.0 {
		t.Errorf("age_days = %v, want < 1.0", pane["age_days"])
	}
}

func TestFoldBenchmarksFlagsStale(t *testing.T) {
	cat := testCatalog([]map[string]any{
		{"timestamp": "20260501T000000Z", "model": "x", "tags": []any{}},
	}, 2)

	pane := FoldBenchmarks(cat, testNow, 30)
	if pane["stale"] != true || pane["verdict"] != "ACTION" || pane["ok"] != false {
		t.Errorf("stale/verdict/ok = %v/%v/%v, want true/ACTION/false", pane["stale"], pane["verdict"], pane["ok"])
	}
	reason, _ := pane["reason"].(string)
	if !strings.Contains(reason, "STALE") {
		t.Errorf("expected STALE in reason: %s", reason)
	}
}

func TestFoldBenchmarksMissingCatalogIsHardError(t *testing.T) {
	pane := FoldBenchmarks(nil, testNow, StaleDays)
	if pane["verdict"] != "ERROR" || pane["ok"] != false {
		t.Errorf("verdict/ok = %v/%v, want ERROR/false", pane["verdict"], pane["ok"])
	}
}

func TestFoldBenchmarksEmptyRunsIsAction(t *testing.T) {
	cat := testCatalog(nil, 1)
	pane := FoldBenchmarks(cat, testNow, StaleDays)
	if pane["runs"] != 0 || pane["verdict"] != "ACTION" {
		t.Errorf("runs/verdict = %v/%v, want 0/ACTION", pane["runs"], pane["verdict"])
	}
	prov, ok := pane["provenance"].(map[string]int)
	if !ok {
		t.Fatalf("provenance not map[string]int: %v", pane["provenance"])
	}
	for _, k := range Tags {
		if _, exists := prov[k]; !exists {
			t.Errorf("provenance missing key %q", k)
		}
	}
}

func TestFoldBenchmarksFunctionalSeparatesFromThroughput(t *testing.T) {
	cat := testCatalog([]map[string]any{
		{"timestamp": "20260625T050017Z", "tags": []any{"agent-live"}, "run_id": "a"},
		{"timestamp": "20260625T050017Z", "tags": []any{"safetensors-load-rss"}, "run_id": "b"},
		{"timestamp": "20260625T050017Z", "tags": []any{"parity"}, "run_id": "c"},
	}, 1)
	pane := FoldBenchmarks(cat, testNow, StaleDays)
	prov := pane["provenance"].(map[string]int)
	if prov["functional"] != 3 || prov["measured"] != 0 || prov["unknown"] != 0 {
		t.Errorf("provenance = %v, want functional=3, measured=0, unknown=0", prov)
	}
}

func TestEnrichCatalogEnginesStampsFromLocalArtifacts(t *testing.T) {
	td := t.TempDir()
	runDir := filepath.Join(td, "experiments", "benchmark", "runs", "by-machine", "m", "r")
	if err := os.MkdirAll(runDir, 0755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	artifactPath := filepath.Join(runDir, "01-decode.json")
	if err := os.WriteFile(artifactPath, []byte(`{"engine": "fak-in-kernel Q8_0 ... decode"}`), 0644); err != nil {
		t.Fatalf("write artifact failed: %v", err)
	}

	catalog := map[string]any{
		"runs": []any{
			map[string]any{
				"run_id": "r",
				"tags":   []any{"agent-live"},
				"path":   "experiments/benchmark/runs/by-machine/m/r",
			},
		},
		"machines": map[string]any{"m": map[string]any{}},
	}

	enriched := EnrichCatalogEngines(td, catalog)
	enrichedRuns := enriched["runs"].([]map[string]any)
	run := enrichedRuns[0]
	engines, ok := run["artifact_engines"].([]string)
	if !ok || len(engines) != 1 || engines[0] != "fak-in-kernel Q8_0 ... decode" {
		t.Errorf("artifact_engines = %v, want ['fak-in-kernel Q8_0 ... decode']", run["artifact_engines"])
	}

	// original catalog not mutated
	origRuns := catalog["runs"].([]any)
	origRun := origRuns[0].(map[string]any)
	if _, exists := origRun["artifact_engines"]; exists {
		t.Errorf("original catalog was mutated")
	}

	// classifier reads engine -> measured, not agent-live tag
	if got := Classify(run); got != "measured" {
		t.Errorf("Classify(run) = %q, want 'measured'", got)
	}
}

func TestFoldWorkZeroPlansIsSkipNotError(t *testing.T) {
	pane := FoldWork(map[string]any{"counts": map[string]any{"total_plans": 0}}, "")
	if pane["verdict"] != "SKIP" || pane["ok"] != true {
		t.Errorf("verdict/ok = %v/%v, want SKIP/true", pane["verdict"], pane["ok"])
	}
	if pane["total_plans"] != 0 {
		t.Errorf("total_plans = %v, want 0", pane["total_plans"])
	}
}

func TestFoldWorkAbsentIsSkip(t *testing.T) {
	pane := FoldWork(nil, "plan_audit unavailable")
	if pane["verdict"] != "SKIP" || pane["ok"] != true {
		t.Errorf("verdict/ok = %v/%v, want SKIP/true", pane["verdict"], pane["ok"])
	}
}

func TestFoldWorkWithPlansReportsOK(t *testing.T) {
	pane := FoldWork(map[string]any{
		"counts": map[string]any{
			"total_plans": 4,
			"shipped":     3,
			"remaining":   1,
		},
	}, "")
	if pane["verdict"] != "OK" || pane["total_plans"] != 4 {
		t.Errorf("verdict/total_plans = %v/%v, want OK/4", pane["verdict"], pane["total_plans"])
	}
	reason, _ := pane["reason"].(string)
	if !strings.Contains(reason, "3 shipped") || !strings.Contains(reason, "1 remaining") {
		t.Errorf("reason missing shipped/remaining: %s", reason)
	}
}

func TestFoldIndustryReportsParityDebt(t *testing.T) {
	pane := FoldIndustry(map[string]any{
		"corpus": map[string]any{
			"parity_debt": 7,
			"grade":       "B",
		},
	}, "")
	pd, _ := toInt(pane["parity_debt"])
	grade, _ := pane["grade"].(*string)
	var gradeStr string
	if grade != nil {
		gradeStr = *grade
	}
	if pane["verdict"] != "OK" || pd != 7 || gradeStr != "B" {
		t.Errorf("verdict/parity_debt/grade = %v/%v/%v, want OK/7/B", pane["verdict"], pd, gradeStr)
	}
}

func TestFoldIndustryAbsentIsSkip(t *testing.T) {
	pane := FoldIndustry(nil, "timed out")
	if pane["verdict"] != "SKIP" || pane["ok"] != true {
		t.Errorf("verdict/ok = %v/%v, want SKIP/true", pane["verdict"], pane["ok"])
	}
}

func TestFoldIndustryNoDebtKeyIsSkip(t *testing.T) {
	pane := FoldIndustry(map[string]any{
		"corpus": map[string]any{
			"grade": "A",
		},
	}, "")
	if pane["verdict"] != "SKIP" {
		t.Errorf("verdict = %v, want SKIP", pane["verdict"])
	}
	grade, _ := pane["grade"].(*string)
	if grade == nil || *grade != "A" {
		t.Errorf("grade = %v, want A", pane["grade"])
	}
}

func TestGitPaneSmokeOnRealRepo(t *testing.T) {
	root := RepoRoot("")
	pane := GitPane(root, time.Now().UTC(), DefaultPushLagActionSeconds, DefaultDirtyLagActionSeconds)
	if pane["key"] != "git" {
		t.Fatalf("key = %v, want git", pane["key"])
	}
	v, _ := pane["verdict"].(string)
	if v != "OK" && v != "ACTION" && v != "ERROR" {
		t.Fatalf("unexpected verdict: %s", v)
	}
	if v == "OK" {
		if pane["sha"] == nil || pane["sha"] == "" {
			t.Errorf("expected sha in OK git pane")
		}
		if _, ok := pane["dirty"].(int); !ok {
			t.Errorf("dirty not int: %v", pane["dirty"])
		}
		if _, ok := pane["push_lag_seconds"]; !ok {
			t.Errorf("missing push_lag_seconds key")
		}
		if _, ok := pane["dirty_lag_seconds"]; !ok {
			t.Errorf("missing dirty_lag_seconds key")
		}
	}
}

func TestGitPanePushLagTripsAction(t *testing.T) {
	now := time.Date(2026, 6, 25, 12, 0, 0, 0, time.UTC)
	commitTime := now.Add(-2 * time.Hour).Unix()

	mockRunner := func(root string, args ...string) (string, error) {
		cmdStr := strings.Join(args, " ")
		switch cmdStr {
		case "rev-parse --short HEAD":
			return "abc1234", nil
		case "rev-parse --abbrev-ref HEAD":
			return "main", nil
		case "status --porcelain":
			return "", nil
		case "rev-list --left-right --count @{upstream}...HEAD":
			return "0 1", nil
		case "log --format=%ct @{upstream}..HEAD":
			return fmt.Sprintf("%d", commitTime), nil
		default:
			return "", nil
		}
	}

	pane := GitPaneWithRunner(".", mockRunner, now, 45*60, 45*60)
	ahead, _ := toInt(pane["ahead"])
	behind, _ := toInt(pane["behind"])
	if ahead != 1 || behind != 0 {
		t.Errorf("ahead/behind = %v/%v, want 1/0", ahead, behind)
	}
	lag, ok := toInt(pane["push_lag_seconds"])
	if !ok || lag <= 45*60 {
		t.Errorf("push_lag_seconds = %v, want > %d", pane["push_lag_seconds"], 45*60)
	}
	if pane["verdict"] != "ACTION" || pane["ok"] != false {
		t.Errorf("verdict/ok = %v/%v, want ACTION/false", pane["verdict"], pane["ok"])
	}
	reason, _ := pane["reason"].(string)
	if !strings.Contains(reason, "push to origin") {
		t.Errorf("expected 'push to origin' in reason: %s", reason)
	}

	// generous threshold -> OK
	okPane := GitPaneWithRunner(".", mockRunner, now, 1e9, 45*60)
	if okPane["verdict"] != "OK" {
		t.Errorf("verdict with high threshold = %v, want OK", okPane["verdict"])
	}
}

func TestGitPaneDirtyLagTripsAction(t *testing.T) {
	td := t.TempDir()
	dirtyPath := filepath.Join(td, "b.txt")
	if err := os.WriteFile(dirtyPath, []byte("content"), 0644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	oldTime := testNow.Add(-2 * time.Hour)
	if err := os.Chtimes(dirtyPath, oldTime, oldTime); err != nil {
		t.Fatalf("chtimes: %v", err)
	}

	mockRunner := func(root string, args ...string) (string, error) {
		cmdStr := strings.Join(args, " ")
		switch cmdStr {
		case "rev-parse --short HEAD":
			return "abc1234", nil
		case "rev-parse --abbrev-ref HEAD":
			return "main", nil
		case "status --porcelain":
			return " M b.txt", nil
		case "rev-list --left-right --count @{upstream}...HEAD":
			return "0 0", nil
		default:
			return "", nil
		}
	}

	pane := GitPaneWithRunner(td, mockRunner, testNow, 45*60, 45*60)
	dirty, _ := toInt(pane["dirty"])
	if dirty != 1 {
		t.Errorf("dirty = %v, want 1", dirty)
	}
	lag, ok := toInt(pane["dirty_lag_seconds"])
	if !ok || lag <= 45*60 {
		t.Errorf("dirty_lag_seconds = %v, want > %d", pane["dirty_lag_seconds"], 45*60)
	}
	oldestPath, _ := pane["oldest_dirty_path"].(*string)
	if oldestPath == nil || *oldestPath != "b.txt" {
		t.Errorf("oldest_dirty_path = %v, want 'b.txt'", pane["oldest_dirty_path"])
	}
	if pane["dirty_lag_stale"] != true || pane["verdict"] != "ACTION" || pane["ok"] != false {
		t.Errorf("stale/verdict/ok = %v/%v/%v, want true/ACTION/false", pane["dirty_lag_stale"], pane["verdict"], pane["ok"])
	}
	reason, _ := pane["reason"].(string)
	if !strings.Contains(reason, "fak sweep --json") {
		t.Errorf("expected 'fak sweep --json' in reason: %s", reason)
	}

	// generous threshold -> OK
	okPane := GitPaneWithRunner(td, mockRunner, testNow, 45*60, 1e9)
	if okPane["verdict"] != "OK" || okPane["dirty_lag_stale"] != false {
		t.Errorf("verdict/stale with high threshold = %v/%v, want OK/false", okPane["verdict"], okPane["dirty_lag_stale"])
	}
}

func TestFoldAllGreenWhenNoAction(t *testing.T) {
	panes := testPanes("OK", "OK", "SKIP", "OK", false)
	out := Fold(panes, ".", "c0", "2026-06-25T12:00:00Z")
	if !out.OK || out.Verdict != "OK" || out.Finding != "all_green" {
		t.Errorf("ok/verdict/finding = %v/%v/%v, want true/OK/all_green", out.OK, out.Verdict, out.Finding)
	}
	if out.Schema != Schema {
		t.Errorf("schema = %s, want %s", out.Schema, Schema)
	}
	for _, wantKey := range []string{"git", "benchmarks", "work", "industry"} {
		if _, exists := out.Panes[wantKey]; !exists {
			t.Errorf("missing pane %s", wantKey)
		}
	}
}

func TestFoldSoftSkipNeverTrips(t *testing.T) {
	panes := testPanes("OK", "OK", "SKIP", "SKIP", false)
	out := Fold(panes, ".", "c0", "2026-06-25T12:00:00Z")
	if out.Verdict != "OK" {
		t.Errorf("verdict = %s, want OK", out.Verdict)
	}
	if !strings.Contains(out.Reason, "skipped") {
		t.Errorf("expected 'skipped' in reason: %s", out.Reason)
	}
}

func TestFoldHardGitErrorTripsToAction(t *testing.T) {
	panes := testPanes("ERROR", "OK", "SKIP", "OK", false)
	out := Fold(panes, ".", "c0", "2026-06-25T12:00:00Z")
	if out.OK != false || out.Verdict != "ACTION" || out.Finding != "needs_attention" {
		t.Errorf("ok/verdict/finding = %v/%v/%v, want false/ACTION/needs_attention", out.OK, out.Verdict, out.Finding)
	}
	if !strings.Contains(out.NextAction, "git") {
		t.Errorf("expected 'git' in next_action: %s", out.NextAction)
	}
}

func TestFoldStaleBenchmarksTripsAndPointsToCatalog(t *testing.T) {
	panes := testPanes("OK", "ACTION", "SKIP", "OK", true)
	out := Fold(panes, ".", "c0", "2026-06-25T12:00:00Z")
	if out.Verdict != "ACTION" {
		t.Errorf("verdict = %s, want ACTION", out.Verdict)
	}
	if !strings.Contains(out.NextAction, "bench_catalog.py build") {
		t.Errorf("expected bench_catalog.py build in next_action: %s", out.NextAction)
	}
}

func TestFoldGitPushLagPointsToPushNotRepoFix(t *testing.T) {
	panes := testPanes("OK", "OK", "SKIP", "OK", false)
	panes[0]["verdict"] = "ACTION"
	panes[0]["ok"] = false
	panes[0]["ahead"] = 3
	panes[0]["push_lag_seconds"] = 50 * 60
	panes[0]["push_lag_stale"] = true
	panes[0]["reason"] = "abc (main), clean tree, +3/-0 vs upstream, 3 unpushed, oldest 50m old"

	out := Fold(panes, ".", "c0", "2026-06-25T12:00:00Z")
	if out.Verdict != "ACTION" {
		t.Errorf("verdict = %s, want ACTION", out.Verdict)
	}
	if !strings.Contains(out.NextAction, "push to origin") || !strings.Contains(out.NextAction, "50m") {
		t.Errorf("next_action missing 'push to origin' or '50m': %s", out.NextAction)
	}
	if strings.Contains(out.NextAction, "not a repo") {
		t.Errorf("unexpected 'not a repo' in next_action: %s", out.NextAction)
	}
}

func TestFoldGitDirtyLagPointsToSweep(t *testing.T) {
	panes := testPanes("OK", "OK", "SKIP", "OK", false)
	dirtyPath := "internal/x/y.go"
	panes[0]["verdict"] = "ACTION"
	panes[0]["ok"] = false
	panes[0]["dirty"] = 7
	panes[0]["dirty_lag_seconds"] = 80 * 60
	panes[0]["oldest_dirty_path"] = &dirtyPath
	panes[0]["dirty_lag_stale"] = true
	panes[0]["reason"] = "abc (main), 7 dirty, oldest dirty 80m old at internal/x/y.go"

	out := Fold(panes, ".", "c0", "2026-06-25T12:00:00Z")
	if out.Verdict != "ACTION" {
		t.Errorf("verdict = %s, want ACTION", out.Verdict)
	}
	if !strings.Contains(out.NextAction, "fak sweep --json") {
		t.Errorf("expected 'fak sweep --json' in next_action: %s", out.NextAction)
	}
	if !strings.Contains(out.NextAction, "internal/x/y.go") {
		t.Errorf("expected path in next_action: %s", out.NextAction)
	}
	if strings.Contains(out.NextAction, "push to origin") {
		t.Errorf("unexpected 'push to origin' in next_action: %s", out.NextAction)
	}
}

func TestRenderListsEveryPane(t *testing.T) {
	out := Fold(testPanes("OK", "OK", "SKIP", "OK", false), ".", "c0", "2026-06-25T12:00:00Z")
	text := Render(out)
	for _, label := range []string{"git", "benchmarks", "work", "industry"} {
		if !strings.Contains(text, label) {
			t.Errorf("missing label %s in render: %s", label, text)
		}
	}
	if !strings.Contains(text, "fresh status") {
		t.Errorf("missing 'fresh status' in render: %s", text)
	}
}

func TestRenderDocHasProvenanceAndPanes(t *testing.T) {
	out := Fold(testPanes("OK", "OK", "SKIP", "OK", false), ".", "c0", "2026-06-25T12:00:00Z")
	doc := RenderDoc(out, "2026-06-25")
	if !strings.Contains(doc, "Fresh status snapshot (2026-06-25)") {
		t.Errorf("doc missing title: %s", doc)
	}
	for _, term := range []string{"measured", "modeled", "unknown", "| Pane | Verdict | Detail |"} {
		if !strings.Contains(doc, term) {
			t.Errorf("doc missing %q", term)
		}
	}
}

func TestLiveCollectAndFold(t *testing.T) {
	root := RepoRoot("")
	panes := Collect(root, CollectOptions{
		Timeout: 30 * time.Second,
	})
	if len(panes) != 4 {
		t.Fatalf("expected 4 panes, got %d", len(panes))
	}
	out := Fold(panes, root, HeadCommit(root, nil), "now")

	data, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("json marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("json unmarshal: %v", err)
	}
	for _, field := range []string{"schema", "ok", "verdict", "finding", "reason", "next_action", "panes"} {
		if _, ok := m[field]; !ok {
			t.Errorf("missing %s in payload", field)
		}
	}
	if _, ok := out.Panes["git"]; !ok {
		t.Errorf("missing git pane")
	}
	if _, ok := out.Panes["benchmarks"]; !ok {
		t.Errorf("missing benchmarks pane")
	}
}

func TestMainCheckReturnsZeroWhenGreen(t *testing.T) {
	orig := CollectFn
	defer func() { CollectFn = orig }()

	CollectFn = func(root string, opts CollectOptions) []map[string]any {
		return testPanes("OK", "OK", "SKIP", "OK", false)
	}

	code := Run(io.Discard, io.Discard, []string{"--check"})
	if code != 0 {
		t.Errorf("expected exit 0 on green check, got %d", code)
	}

	CollectFn = func(root string, opts CollectOptions) []map[string]any {
		return testPanes("ERROR", "OK", "SKIP", "OK", false)
	}

	code = Run(io.Discard, io.Discard, []string{"--check"})
	if code != 1 {
		t.Errorf("expected exit 1 on error check, got %d", code)
	}
}

func TestMainJSONEmitsEnvelope(t *testing.T) {
	orig := CollectFn
	defer func() { CollectFn = orig }()

	CollectFn = func(root string, opts CollectOptions) []map[string]any {
		return testPanes("OK", "OK", "SKIP", "OK", false)
	}

	var stdout, stderr bytes.Buffer
	code := Run(&stdout, &stderr, []string{"--json"})
	if code != 0 {
		t.Fatalf("expected exit 0 on json, got %d (stderr: %s)", code, stderr.String())
	}

	var out map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, stdout.String())
	}

	if out["schema"] != Schema {
		t.Errorf("schema = %v, want %s", out["schema"], Schema)
	}

	panes, ok := out["panes"].(map[string]any)
	if !ok {
		t.Fatalf("panes not map[string]any")
	}
	for _, k := range []string{"git", "benchmarks", "work", "industry"} {
		if _, exists := panes[k]; !exists {
			t.Errorf("missing pane %q in json panes", k)
		}
	}
}

func TestMainWriteDoc(t *testing.T) {
	orig := CollectFn
	defer func() { CollectFn = orig }()

	CollectFn = func(root string, opts CollectOptions) []map[string]any {
		return testPanes("OK", "OK", "SKIP", "OK", false)
	}

	td := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := Run(&stdout, &stderr, []string{"--write-doc", "--date", "2026-06-25", "--workspace", td})
	if code != 0 {
		t.Fatalf("write-doc failed with code %d: %s", code, stderr.String())
	}

	expectedDoc := filepath.Join(td, "docs", "notes", "FRESH-STATUS-2026-06-25.md")
	content, err := os.ReadFile(expectedDoc)
	if err != nil {
		t.Fatalf("could not read generated doc: %v", err)
	}
	if !strings.Contains(string(content), "Fresh status snapshot (2026-06-25)") {
		t.Errorf("generated doc content missing title: %s", string(content))
	}
	if !strings.Contains(stdout.String(), "wrote snapshot -> docs/notes/FRESH-STATUS-2026-06-25.md") {
		t.Errorf("stdout missing confirmation: %s", stdout.String())
	}
}

func TestMainInvalidArgs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := Run(&stdout, &stderr, []string{"extra_arg"})
	if code != 2 {
		t.Errorf("expected exit 2 on extra arg, got %d", code)
	}

	code = Run(&stdout, &stderr, []string{"--unknown-flag"})
	if code != 2 {
		t.Errorf("expected exit 2 on unknown flag, got %d", code)
	}
}
