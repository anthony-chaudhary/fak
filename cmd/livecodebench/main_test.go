package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunFixtureSmokeJSONClaimDisallowed(t *testing.T) {
	fixture := filepath.Join("..", "..", "internal", "livecodebench", "testdata", "fixture.json")
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	code := run([]string{"--fixture", fixture, "--check", "--json"})
	_ = w.Close()
	os.Stdout = oldStdout
	var out bytes.Buffer
	_, _ = out.ReadFrom(r)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stdout=%s", code, out.String())
	}
	if !strings.Contains(out.String(), `"result_claim_allowed": false`) {
		t.Fatalf("json did not pin result_claim_allowed=false:\n%s", out.String())
	}
}

// TestRunExportCustomEvaluatorWritesGradeableInput pins #2102: `livecodebench
// export --format custom-evaluator` must emit the exact
// [{question_id, code_list}] shape lcb_runner.runner.custom_evaluator
// consumes, ordered to match the fixture's benchmark problems.
func TestRunExportCustomEvaluatorWritesGradeableInput(t *testing.T) {
	fixture := filepath.Join("..", "..", "internal", "livecodebench", "testdata", "fixture.json")
	dir := t.TempDir()
	out := filepath.Join(dir, "custom_evaluator_input.json")

	code := run([]string{"export", "--format", "custom-evaluator", "--fixture", fixture, "--out", out})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}

	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read export output: %v", err)
	}
	var items []struct {
		QuestionID string   `json:"question_id"`
		CodeList   []string `json:"code_list"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		t.Fatalf("export output not valid JSON: %v\n%s", err, raw)
	}
	if len(items) != 4 {
		t.Fatalf("item count = %d, want 4", len(items))
	}
	if items[0].QuestionID != "fixture-codegeneration-1" {
		t.Fatalf("item 0 question_id = %q, want fixture-codegeneration-1 (order drift)", items[0].QuestionID)
	}
	if len(items[0].CodeList) == 0 {
		t.Fatal("item 0 code_list is empty")
	}
}

// TestRunExportRejectsUnsupportedFormat guards the --format flag against a
// silent no-op for a format the exporter cannot produce.
func TestRunExportRejectsUnsupportedFormat(t *testing.T) {
	fixture := filepath.Join("..", "..", "internal", "livecodebench", "testdata", "fixture.json")
	code := run([]string{"export", "--format", "nonsense", "--fixture", fixture})
	if code == 0 {
		t.Fatal("exit = 0, want nonzero for an unsupported --format")
	}
}

// TestRunFetchFromWritesNormalizedSuite pins #2090: `livecodebench fetch --from`
// replays a committed/offline upstream rows file into a normalized, sourced
// Suite JSON with a provenance header, with no network. The written suite must
// re-load and validate (proving the provenance/count invariant holds).
func TestRunFetchFromWritesNormalizedSuite(t *testing.T) {
	from := filepath.Join("..", "..", "internal", "livecodebench", "testdata", "upstream_sample.json")
	dir := t.TempDir()
	out := filepath.Join(dir, "suite.json")

	code := run([]string{"fetch", "--from", from, "--release-version", "release_v2", "--revision", "release_v2", "--out", out})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read fetch output: %v", err)
	}
	var suite struct {
		Schema         string `json:"schema"`
		ReleaseVersion string `json:"release_version"`
		Provenance     struct {
			DatasetID    string `json:"dataset_id"`
			Revision     string `json:"revision"`
			ProblemCount int    `json:"problem_count"`
		} `json:"provenance"`
		Problems []struct {
			QuestionID string `json:"question_id"`
		} `json:"problems"`
	}
	if err := json.Unmarshal(raw, &suite); err != nil {
		t.Fatalf("fetch output not valid JSON: %v\n%s", err, raw)
	}
	if suite.Schema != "fak.livecodebench-suite.v1" {
		t.Fatalf("schema = %q", suite.Schema)
	}
	if suite.ReleaseVersion != "release_v2" {
		t.Fatalf("release = %q, want release_v2", suite.ReleaseVersion)
	}
	if suite.Provenance.DatasetID == "" || suite.Provenance.ProblemCount != len(suite.Problems) {
		t.Fatalf("provenance %+v does not match %d problems", suite.Provenance, len(suite.Problems))
	}
	if len(suite.Problems) != 3 {
		t.Fatalf("problem count = %d, want 3", len(suite.Problems))
	}
}

// TestRunFetchRequiresExactlyOneSource guards the source flags: with neither
// --from nor --fetch (or both), fetch must refuse rather than silently do
// nothing or attempt an unintended network call.
func TestRunFetchRequiresExactlyOneSource(t *testing.T) {
	if code := run([]string{"fetch", "--release-version", "release_v2"}); code == 0 {
		t.Fatal("exit = 0 with no source, want nonzero")
	}
	from := filepath.Join("..", "..", "internal", "livecodebench", "testdata", "upstream_sample.json")
	if code := run([]string{"fetch", "--release-version", "release_v2", "--from", from, "--fetch"}); code == 0 {
		t.Fatal("exit = 0 with both sources, want nonzero")
	}
}

// TestRunPreflightNeverProbesNetworkByDefault pins #2111: --preflight must
// never emit a benchmark number, and without --probe-dataset/--probe-gateway
// it must not attempt network I/O -- the dataset and gateway gates report
// "not probed" instead of blocking forever on an unreachable host.
func TestRunPreflightNeverProbesNetworkByDefault(t *testing.T) {
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	code := run([]string{"--preflight", "--json"})
	_ = w.Close()
	os.Stdout = oldStdout
	var out bytes.Buffer
	_, _ = out.ReadFrom(r)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stdout=%s", code, out.String())
	}
	var report struct {
		Schema             string `json:"schema"`
		Status             string `json:"status"`
		ResultClaimAllowed bool   `json:"result_claim_allowed"`
		Gates              []struct {
			Name   string `json:"name"`
			OK     bool   `json:"ok"`
			Detail string `json:"detail"`
		} `json:"gates"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("preflight output not valid JSON: %v\n%s", err, out.String())
	}
	if report.Schema != "fak.livecodebench-preflight.v1" {
		t.Fatalf("schema = %q", report.Schema)
	}
	if report.ResultClaimAllowed {
		t.Fatal("preflight must never allow a result claim")
	}
	if report.Status != "BLOCKED_PREFLIGHT" && report.Status != "READY" {
		t.Fatalf("status = %q, want BLOCKED_PREFLIGHT or READY", report.Status)
	}
	byName := map[string]bool{}
	for _, g := range report.Gates {
		byName[g.Name] = true
		if g.Name == "hf_dataset_reachable" || g.Name == "fak_gateway_reachable" {
			if g.OK {
				t.Fatalf("gate %q should not be OK without --probe-dataset/--probe-gateway", g.Name)
			}
			if !strings.Contains(g.Detail, "not probed") {
				t.Fatalf("gate %q detail = %q, want it to say not probed", g.Name, g.Detail)
			}
		}
	}
	for _, want := range []string{"uv_present", "python311_present", "hf_dataset_reachable", "fak_gateway_reachable", "sandbox_available"} {
		if !byName[want] {
			t.Fatalf("missing gate %q", want)
		}
	}
}
