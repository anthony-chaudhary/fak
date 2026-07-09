// Hermetic tests for the L9 prefill-sweep driver (Go port of the retired
// tools/glm52_prefill_sweep.py + its test). GPU-free and network-free by construction:
// every test exercises the pure planner or the pure ledger-land helper. The purity test
// overrides the execGit + httpDo seams to FAIL, then proves --dry-run reaches neither —
// so "produces no prefill number, only enables the measurement" is enforced, not just
// documented.
package glm52prefillsweep

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPlanCoversAllFiveLengthsInOrder(t *testing.T) {
	plan := BuildPlan("zai-org/GLM-5.2", "land/root", nil, DefaultMaxTokens, true, FragileMinLen)
	if len(plan) != 5 {
		t.Fatalf("want 5 steps, got %d", len(plan))
	}
	got := make([]int, len(plan))
	for i, s := range plan {
		got[i] = s.PromptLen
	}
	want := []int{128, 512, 2048, 4096, 8192}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("lengths = %v, want %v", got, want)
		}
	}
}

func TestRequestBodiesArePrefillDominantAndWellFormed(t *testing.T) {
	plan := BuildPlan("my-model", "land/root", nil, 1, true, FragileMinLen)
	for _, step := range plan {
		body := step.RequestBody
		if body.Model != "my-model" {
			t.Errorf("model = %q, want my-model", body.Model)
		}
		if body.Temperature != 0 {
			t.Errorf("temperature = %d, want 0", body.Temperature)
		}
		if body.MaxTokens > 4 {
			t.Errorf("max_tokens = %d, want <= 4 (prefill-dominant)", body.MaxTokens)
		}
		if !body.Stream {
			t.Errorf("stream = false, want true")
		}
		if body.StreamOptions == nil || !body.StreamOptions.IncludeUsage {
			t.Errorf("stream_options = %+v, want {include_usage:true}", body.StreamOptions)
		}
		if len(body.Messages) == 0 || body.Messages[0].Role != "user" {
			t.Fatalf("messages[0].role != user: %+v", body.Messages)
		}
		content := body.Messages[0].Content
		if content == "" {
			t.Fatalf("empty prompt content")
		}
		// The synthetic prompt must scale with the target length.
		if n := len(strings.Fields(content)); n != step.PromptLen {
			t.Errorf("content words = %d, want %d", n, step.PromptLen)
		}
	}
}

func TestLandPathsAreTemplatedPerLength(t *testing.T) {
	root := "experiments/benchmark/runs/by-machine/nodeA/STAMP-glm52-prefill-sweep"
	plan := BuildPlan("m", root, nil, DefaultMaxTokens, true, FragileMinLen)
	got := map[int]string{}
	for _, s := range plan {
		got[s.PromptLen] = s.LandSubdir
	}
	if got[512] != root+"/p512" {
		t.Errorf("p512 land = %q", got[512])
	}
	if got[8192] != root+"/p8192" {
		t.Errorf("p8192 land = %q", got[8192])
	}
}

func TestTwoLargestLengthsFlaggedFragile(t *testing.T) {
	plan := BuildPlan("m", "root", nil, DefaultMaxTokens, true, FragileMinLen)
	fragile := map[int]bool{}
	for _, s := range plan {
		if s.FragileOnSM80 {
			fragile[s.PromptLen] = true
		}
	}
	if len(fragile) != 2 || !fragile[4096] || !fragile[8192] {
		t.Errorf("fragile set = %v, want {4096,8192}", fragile)
	}
}

func TestBlockingModeOmitsStreamOptions(t *testing.T) {
	plan := BuildPlan("m", "root", nil, DefaultMaxTokens, false, FragileMinLen)
	for _, step := range plan {
		if step.RequestBody.Stream {
			t.Errorf("stream = true in blocking mode")
		}
		if step.RequestBody.StreamOptions != nil {
			t.Errorf("stream_options present in blocking mode: %+v", step.RequestBody.StreamOptions)
		}
	}
	// And confirm the marshaled JSON has no stream_options key.
	b, _ := json.Marshal(plan[0].RequestBody)
	if strings.Contains(string(b), "stream_options") {
		t.Errorf("blocking body JSON contains stream_options: %s", b)
	}
}

// withFailingIO overrides the git + HTTP seams to fail, restoring them on cleanup. Any
// call from the code under test is a purity-contract violation.
func withFailingIO(t *testing.T) {
	t.Helper()
	origGit, origHTTP := execGit, httpDo
	execGit = func(root string, args ...string) (string, int) {
		t.Fatalf("dry-run must not shell out to git (args=%v)", args)
		return "", 1
	}
	httpDo = func(req *http.Request, timeout time.Duration) (*http.Response, error) {
		t.Fatalf("dry-run must not perform network I/O (url=%s)", req.URL)
		return nil, nil
	}
	t.Cleanup(func() { execGit, httpDo = origGit, origHTTP })
}

func TestDryRunWritesPlanAndTouchesNoNetworkOrGPU(t *testing.T) {
	withFailingIO(t)
	out := filepath.Join(t.TempDir(), "plan.json")
	code := Run(io_Discard{}, io_Discard{}, []string{"--dry-run", "--out", out})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	var report map[string]any
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if report["dry_run"] != true {
		t.Errorf("dry_run = %v, want true", report["dry_run"])
	}
	if report["mode"] != "plan" {
		t.Errorf("mode = %v, want plan", report["mode"])
	}
	if plan, ok := report["plan"].([]any); !ok || len(plan) != 5 {
		t.Errorf("plan len = %v, want 5", report["plan"])
	}
	notes, _ := report["notes"].([]any)
	joined := ""
	for _, n := range notes {
		joined += " " + strings.ToLower(n.(string))
	}
	if !strings.Contains(joined, "no prefill number is produced") {
		t.Errorf("honesty fence missing from notes: %q", joined)
	}
}

func TestNoEndpointFallsBackToPlan(t *testing.T) {
	withFailingIO(t)
	out := filepath.Join(t.TempDir(), "plan.json")
	code := Run(io_Discard{}, io_Discard{}, []string{"--out", out}) // no --endpoint, no --dry-run
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	raw, _ := os.ReadFile(out)
	var report map[string]any
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if report["mode"] != "plan" {
		t.Errorf("mode = %v, want plan", report["mode"])
	}
}

func fakeOKRecord(t *testing.T) Record {
	t.Helper()
	m := measurement{
		ok: true, httpStatus: 200, ttftSeconds: fptr(0.25), totalSeconds: fptr(0.26),
		promptTokens: iptr(512), completionTokens: iptr(1), source: "stream-ttft",
	}
	return RecordForLength(m, "zai-org/GLM-5.2", "http://n:8000/v1", 512, 1, true)
}

func TestManifestCarriesLineageAndArtifactResultStaysRaw(t *testing.T) {
	lineage := Lineage{
		LineageSchema: LineageSchema, AppVersion: "test", UTC: "2026-07-08T00:00:00Z",
		GitCommit: "deadbeefcafe1234", GoVersion: "python-driver", Node: "nodeA",
	}
	land := filepath.Join(t.TempDir(), "p512")
	manifestPath, err := WriteLedgerArtifact(land, fakeOKRecord(t), lineage)
	if err != nil {
		t.Fatal(err)
	}
	var manifest map[string]any
	raw, _ := os.ReadFile(manifestPath)
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	lin, _ := manifest["lineage"].(map[string]any)
	if lin["git_commit"] != "deadbeefcafe1234" {
		t.Errorf("lineage.git_commit = %v", lin["git_commit"])
	}
	if lin["utc"] == "" || lin["utc"] == nil {
		t.Errorf("lineage.utc empty")
	}
	art, _ := manifest["benchmark_artifact"].(map[string]any)
	if rid, _ := art["run_id"].(string); rid == "" {
		t.Errorf("benchmark_artifact.run_id empty")
	}
	if scope, _ := manifest["scope"].(string); !strings.Contains(scope, "NOT the 753B") {
		t.Errorf("scope lost at manifest top level: %q", scope)
	}
	if got, want := manifest["prefill_tok_s"].(float64), 512.0/0.25; got != want {
		t.Errorf("prefill_tok_s = %v, want %v", got, want)
	}
	// result.json is the RAW record with NO lineage/envelope.
	var result map[string]any
	rraw, _ := os.ReadFile(filepath.Join(land, "result.json"))
	if err := json.Unmarshal(rraw, &result); err != nil {
		t.Fatal(err)
	}
	if _, ok := result["lineage"]; ok {
		t.Errorf("result.json leaked lineage")
	}
	if _, ok := result["benchmark_artifact"]; ok {
		t.Errorf("result.json leaked benchmark_artifact")
	}
	if _, err := os.Stat(filepath.Join(land, "RESULTS.md")); err != nil {
		t.Errorf("RESULTS.md missing: %v", err)
	}
}

func TestFailedLengthRecordsFailWithoutNumber(t *testing.T) {
	m := measurement{ok: false, httpStatus: 500, errMsg: "CUDA illegal memory access"}
	rec := RecordForLength(m, "m", "http://n/v1", 8192, 1, true)
	if rec.Status != "FAIL" {
		t.Errorf("status = %q, want FAIL", rec.Status)
	}
	if rec.PrefillTokS != nil {
		t.Errorf("prefill_tok_s = %v, want nil", *rec.PrefillTokS)
	}
	if !strings.Contains(rec.Error, "illegal memory access") {
		t.Errorf("error = %q", rec.Error)
	}
}

func TestGLMLandDirEmptyDisablesLand(t *testing.T) {
	withFailingIO(t)
	t.Setenv("GLM_LAND_DIR", "")
	out := filepath.Join(t.TempDir(), "plan.json")
	Run(io_Discard{}, io_Discard{}, []string{"--dry-run", "--out", out})
	raw, _ := os.ReadFile(out)
	var report map[string]any
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if report["land_enabled"] != false {
		t.Errorf("land_enabled = %v, want false", report["land_enabled"])
	}
	plan, _ := report["plan"].([]any)
	step0, _ := plan[0].(map[string]any)
	if step0["land_subdir"] != "" {
		t.Errorf("land_subdir = %v, want empty", step0["land_subdir"])
	}
}

// io_Discard is a trivial io.Writer sink so Run's stdout/stderr are dropped in tests.
type io_Discard struct{}

func (io_Discard) Write(p []byte) (int, error) { return len(p), nil }
