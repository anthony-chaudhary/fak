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

// TestRunParityFlagsDocumentedAndWriteReport pins #2109: the default runner
// exposes the lcb_runner.runner.main-parity flags, each `--help` string names
// its upstream analog, and a default invocation runs the committed fixture
// end-to-end and writes a result-claim-gated report.
func TestRunParityFlagsDocumentedAndWriteReport(t *testing.T) {
	wantUpstream := map[string]string{
		"model":           "--model",
		"scenario":        "--scenario",
		"evaluate":        "--evaluate",
		"release-version": "--release_version",
		"n":               "-n",
		"temperature":     "--temperature",
		"use-cache":       "--use_cache",
	}
	seen := map[string]bool{}
	for _, f := range lcbParityFlags {
		if !strings.Contains(f.usage(), "lcb_runner.runner.main") {
			t.Fatalf("flag %q usage does not name the upstream module: %q", f.name, f.usage())
		}
		up, ok := wantUpstream[f.name]
		if !ok {
			t.Fatalf("unexpected parity flag %q", f.name)
		}
		if !strings.Contains(f.usage(), up) {
			t.Fatalf("flag %q usage missing upstream analog %q: %q", f.name, up, f.usage())
		}
		seen[f.name] = true
	}
	for name := range wantUpstream {
		if !seen[name] {
			t.Fatalf("missing lcb_runner-parity flag %q", name)
		}
	}

	fixture := filepath.Join("..", "..", "internal", "livecodebench", "testdata", "fixture.json")
	dir := t.TempDir()
	out := filepath.Join(dir, "report.json")
	code := run([]string{
		"--fixture", fixture,
		"--model", "glm-5.2",
		"--scenario", "codegeneration",
		"-n", "1",
		"--temperature", "0.2",
		"--release-version", "release_v6",
		"--use-cache",
		"--evaluate",
		"--out", out,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read report output: %v", err)
	}
	var report struct {
		Schema             string `json:"schema"`
		ResultClaimAllowed bool   `json:"result_claim_allowed"`
		Questions          int    `json:"questions"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatalf("report output not valid JSON: %v\n%s", err, raw)
	}
	if report.ResultClaimAllowed {
		t.Fatal("result_claim_allowed must be false even with --evaluate (honesty fence)")
	}
	if report.Questions == 0 {
		t.Fatal("scenario-filtered run wrote a report with no questions")
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

// TestRunContractWritesGatedOfficialRunContract pins #2110: `livecodebench
// contract` emits a JSON (+MD) official-run contract that performs no run and
// asserts result_claim_allowed=false.
func TestRunContractWritesGatedOfficialRunContract(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "contract.json")
	md := filepath.Join(dir, "contract.md")

	code := run([]string{"contract",
		"--release-version", "release_v6",
		"--scenario", "codegeneration",
		"--start-date", "2025-01-01",
		"--end-date", "2025-06-30",
		"--model", "glm-5.2",
		"--serving-backend", "SGLang W4AFP8",
		"--out", out,
		"--md", md,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}

	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read contract output: %v", err)
	}
	var contract struct {
		Schema             string `json:"schema"`
		ResultClaimAllowed bool   `json:"result_claim_allowed"`
		Grading            struct {
			CustomEvaluatorCommand string `json:"custom_evaluator_command"`
		} `json:"grading"`
	}
	if err := json.Unmarshal(raw, &contract); err != nil {
		t.Fatalf("contract output not valid JSON: %v\n%s", err, raw)
	}
	if contract.Schema != "fak.livecodebench-official-run-contract.v1" {
		t.Fatalf("schema = %q", contract.Schema)
	}
	if contract.ResultClaimAllowed {
		t.Fatal("result_claim_allowed must be false")
	}
	if !strings.Contains(contract.Grading.CustomEvaluatorCommand, "lcb_runner.runner.custom_evaluator") {
		t.Fatalf("grading command missing official evaluator: %q", contract.Grading.CustomEvaluatorCommand)
	}
	if _, err := os.Stat(md); err != nil {
		t.Fatalf("markdown not written: %v", err)
	}
}

// TestRunReportWritesJSONAndMarkdown pins #2112: `livecodebench report` renders
// a normalized suite as machine JSON (--out) and human markdown (--md); the
// markdown carries the evidence-class / claim-boundary banner and per-problem
// rows linking each question_id to its verdict and evidence id.
func TestRunReportWritesJSONAndMarkdown(t *testing.T) {
	suite := filepath.Join("..", "..", "internal", "livecodebench", "testdata", "suite_release_v2_sample.json")
	dir := t.TempDir()
	out := filepath.Join(dir, "report.json")
	md := filepath.Join(dir, "report.md")

	code := run([]string{"report",
		"--suite", suite,
		"--generated-at", "2026-07-09T00:00:00Z",
		"--out", out,
		"--md", md,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}

	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("read report output: %v", err)
	}
	var report struct {
		Schema             string `json:"schema"`
		EvidenceClass      string `json:"evidence_class"`
		ResultClaimAllowed bool   `json:"result_claim_allowed"`
		Problems           []struct {
			QuestionID string `json:"question_id"`
			Verdict    string `json:"verdict"`
			EvidenceID string `json:"evidence_id"`
		} `json:"problems"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatalf("report output not valid JSON: %v\n%s", err, raw)
	}
	if report.Schema != "fak.livecodebench-report.v1" {
		t.Fatalf("schema = %q", report.Schema)
	}
	if report.ResultClaimAllowed {
		t.Fatal("result_claim_allowed must be false for an ungraded report")
	}
	if len(report.Problems) == 0 {
		t.Fatal("report carries no per-problem verdict rows")
	}
	for _, p := range report.Problems {
		if p.EvidenceID == "" {
			t.Fatalf("problem %q verdict %q has no evidence_id", p.QuestionID, p.Verdict)
		}
	}

	rawMD, err := os.ReadFile(md)
	if err != nil {
		t.Fatalf("read markdown output: %v", err)
	}
	for _, want := range []string{
		"- Evidence class: `local-ungraded`",
		"- Result claim allowed: `false`",
		"- Claim boundary:",
		"| question_id | scenario | arm | verdict | evidence_id |",
		"| `lcb-sample-001` | codegeneration | - | ungraded | `local-ungraded:lcb-sample-001` |",
	} {
		if !strings.Contains(string(rawMD), want) {
			t.Fatalf("markdown missing %q\n---\n%s", want, rawMD)
		}
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

// A 404 from /rows is the ONLY thing a caller sees when the dataset viewer refuses a
// script-based dataset, and it invites the wrong fix (re-probe every release). Assert the
// message carries the real reason and names the offline path instead.
func TestRowsFetchFailureExplainsViewerRefusal(t *testing.T) {
	const ds = "livecodebench/code_generation_lite"
	const why = "The dataset viewer doesn't support this dataset because it runs arbitrary Python code."

	got := rowsFetchFailure(404, ds, "release_v6", why)
	for _, want := range []string{"HTTP 404", ds, "release_v6", "arbitrary Python code", "--from", "hf download"} {
		if !strings.Contains(got, want) {
			t.Fatalf("404 message missing %q:\n%s", want, got)
		}
	}
	if !strings.Contains(got, "ANY --release-version") {
		t.Fatalf("404 message should say the release pin is not the problem:\n%s", got)
	}

	// No reason to add, or a status that is not a 404: stay terse, never invent a cause.
	if got := rowsFetchFailure(404, ds, "release_v6", "   "); !strings.HasSuffix(got, "config=release_v6") {
		t.Fatalf("empty refusal should leave the bare message, got %q", got)
	}
	if got := rowsFetchFailure(500, ds, "release_v6", why); strings.Contains(got, "--from") {
		t.Fatalf("non-404 should not claim the viewer refused it, got %q", got)
	}
}
