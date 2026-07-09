package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/livecodebench"
)

func writeTestSuite(t *testing.T, dir string) string {
	t.Helper()
	suite := livecodebench.Suite{
		Schema:         "fak.livecodebench-suite.v1",
		Benchmark:      "livecodebench",
		ReleaseVersion: "release_v6",
		Provenance:     livecodebench.Provenance{DatasetID: "livecodebench/code_generation_lite", Revision: "release_v6", ProblemCount: 2},
		Problems: []livecodebench.Problem{
			{QuestionID: "q0", Scenario: livecodebench.ScenarioCodeGeneration, Prompt: "solve a"},
			{QuestionID: "q1", Scenario: livecodebench.ScenarioCodeGeneration, Prompt: "solve b"},
		},
	}
	path := filepath.Join(dir, "suite.json")
	buf, _ := json.Marshal(suite)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestRunFakAndABEndToEnd witnesses #2105's acceptance end-to-end: the fak arm
// runs the same problems/model/n/temperature/release as the raw arm through an
// adjudicating stand-in gateway (its responses carry the `fak` extension), and
// `livecodebench ab` asserts SameProblemIDs / SamePromptHash true, emits
// per-arm summaries + deltas, and pins the pass-rate delta to the ungraded
// sentinel with result_claim_allowed=false.
func TestRunFakAndABEndToEnd(t *testing.T) {
	// Raw stand-in gateway: plain OpenAI-compatible response, no fak extension.
	rawSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"print(1)"}}],` +
			`"usage":{"prompt_tokens":50,"completion_tokens":5}}`))
	}))
	defer rawSrv.Close()
	// Fak stand-in gateway: same wire plus the `fak` adjudication extension.
	fakSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"print(1)"}}],` +
			`"usage":{"prompt_tokens":55,"completion_tokens":5},` +
			`"fak":{"adjudications":[{"tool":"x","admitted":false},{"tool":"y","admitted":true,"repaired_arguments":{"a":1}}],` +
			`"result_admissions":[{"tool":"x"}]}}`))
	}))
	defer fakSrv.Close()

	dir := t.TempDir()
	suitePath := writeTestSuite(t, dir)
	rawOut := filepath.Join(dir, "raw.json")
	fakOut := filepath.Join(dir, "fak.json")
	abOut := filepath.Join(dir, "ab.json")

	shared := []string{"--suite", suitePath, "--model", "glm-4.6", "--n", "2", "--temperature", "0.7"}
	if code := runRaw(append(append([]string{}, shared...), "--endpoint", rawSrv.URL+"/v1", "--out", rawOut)); code != 0 {
		t.Fatalf("runRaw exit %d", code)
	}
	if code := runFak(append(append([]string{}, shared...), "--endpoint", fakSrv.URL+"/v1", "--out", fakOut)); code != 0 {
		t.Fatalf("runFak exit %d", code)
	}

	var fakRep livecodebench.FakArmReport
	rb, err := os.ReadFile(fakOut)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(rb, &fakRep); err != nil {
		t.Fatal(err)
	}
	if fakRep.Arm != "fak" || fakRep.Release != "release_v6" || fakRep.Model != "glm-4.6" || fakRep.N != 2 {
		t.Fatalf("fak run identity not recorded: %+v", fakRep)
	}
	// 2 problems × n=2 = 4 samples; each response carried 2 adjudications
	// (1 denied, 1 safe-resolved) and 1 result admission.
	want := livecodebench.FakArmAdjudication{AdjudicatedSamples: 4, Adjudications: 8, Denied: 4, SafeResolves: 4, ResultAdmissions: 4}
	if fakRep.Adjudication != want {
		t.Fatalf("adjudication evidence = %+v, want %+v", fakRep.Adjudication, want)
	}

	if code := runAB([]string{"--raw", rawOut, "--fak", fakOut, "--out", abOut, "--check"}); code != 0 {
		t.Fatalf("runAB exit %d", code)
	}
	var c livecodebench.ArmComparison
	cb, err := os.ReadFile(abOut)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(cb, &c); err != nil {
		t.Fatal(err)
	}
	if !c.SameProblemIDs || !c.SamePromptHash || !c.SameModel || !c.SameN || !c.SameTemperature || !c.SameRelease {
		t.Fatalf("parity assertions must hold: %+v (mismatches %v)", c, c.Mismatches)
	}
	if c.ResultClaimAllowed || c.PassRateDelta != livecodebench.PassRateDeltaUngraded {
		t.Fatalf("comparison must never claim a pass-rate delta: %+v", c)
	}
	// fak minus raw: 4 samples × (55-50) prompt tokens.
	if c.UsageDelta.PromptTokens != 20 || c.UsageDelta.Samples != 0 {
		t.Fatalf("usage delta wrong: %+v", c.UsageDelta)
	}
}

// TestRunABCheckFailsOnForeignReports guards the assertion path: a fak report
// generated over different problems must fail `ab --check`.
func TestRunABCheckFailsOnForeignReports(t *testing.T) {
	dir := t.TempDir()
	raw := livecodebench.RawArmReport{Arm: "raw", Model: "m", N: 1, Release: "release_v6",
		Problems: []livecodebench.RawArmProblem{{QuestionID: "q0", PromptSHA256: "aa", Completions: []string{"x"}}}}
	fak := livecodebench.FakArmReport{Arm: "fak", Model: "m", N: 1, Release: "release_v6",
		Problems: []livecodebench.RawArmProblem{{QuestionID: "OTHER", PromptSHA256: "bb", Completions: []string{"x"}}}}
	rawPath, fakPath := filepath.Join(dir, "raw.json"), filepath.Join(dir, "fak.json")
	for path, v := range map[string]any{rawPath: raw, fakPath: fak} {
		buf, _ := json.Marshal(v)
		if err := os.WriteFile(path, buf, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if code := runAB([]string{"--raw", rawPath, "--fak", fakPath, "--check"}); code == 0 {
		t.Fatalf("ab --check must fail on foreign reports")
	}
}
