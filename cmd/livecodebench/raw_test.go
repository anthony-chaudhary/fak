package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/livecodebench"
)

// TestRunRawEndToEndViaGateway witnesses the acceptance: n completions per
// problem produced via a stand-in OpenAI-compatible gateway, the run identity
// (model/endpoint/n/temperature) recorded, and provider-cache usage folded
// through Usage.CachedPromptTokens().
func TestRunRawEndToEndViaGateway(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		atomic.AddInt32(&calls, 1)
		w.Header().Set("Content-Type", "application/json")
		// Report the cache hit via the OpenAI-compatible prompt_tokens_details
		// shape so the fold must go through the normalized accessor.
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"print(1)"}}],` +
			`"usage":{"prompt_tokens":50,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":20}}}`))
	}))
	defer srv.Close()

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
	dir := t.TempDir()
	path := filepath.Join(dir, "suite.json")
	buf, _ := json.Marshal(suite)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	outPath := filepath.Join(dir, "raw.json")

	code := runRaw([]string{
		"--suite", path,
		"--model", "glm-4.6",
		"--endpoint", srv.URL + "/v1",
		"--n", "3",
		"--temperature", "0.7",
		"--concurrency", "2",
		"--out", outPath,
	})
	if code != 0 {
		t.Fatalf("runRaw exit %d", code)
	}
	if got := atomic.LoadInt32(&calls); got != 6 {
		t.Fatalf("want 6 gateway calls (2 problems × n=3), got %d", got)
	}

	var rep livecodebench.RawArmReport
	rb, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(rb, &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Model != "glm-4.6" || rep.Endpoint != srv.URL+"/v1" || rep.N != 3 || rep.Temperature != 0.7 {
		t.Fatalf("run identity not recorded: %+v", rep)
	}
	if len(rep.Problems) != 2 || len(rep.Problems[0].Completions) != 3 {
		t.Fatalf("want 2 problems × 3 completions, got %+v", rep.Problems)
	}
	if rep.Usage.Samples != 6 || rep.Usage.CachedPromptTokens != 6*20 {
		t.Fatalf("usage fold wrong: %+v", rep.Usage)
	}
}

// TestGatewayBearerPrecedence pins the env order the raw arm reads its
// credential from, and that an unauthenticated endpoint gets no header at all
// (an empty return, not a "Bearer " with nothing after it).
func TestGatewayBearerPrecedence(t *testing.T) {
	for _, env := range []string{"LCB_API_KEY", "FAK_GATEWAY_KEY", "OPENAI_API_KEY"} {
		t.Setenv(env, "")
	}
	if got := bearerFromEnv(); got != "" {
		t.Fatalf("no credential set: want %q, got %q", "", got)
	}
	t.Setenv("OPENAI_API_KEY", "sk-openai")
	if got := bearerFromEnv(); got != "sk-openai" {
		t.Fatalf("OPENAI_API_KEY only: got %q", got)
	}
	t.Setenv("FAK_GATEWAY_KEY", "sk-fak")
	if got := bearerFromEnv(); got != "sk-fak" {
		t.Fatalf("FAK_GATEWAY_KEY outranks OPENAI_API_KEY: got %q", got)
	}
	t.Setenv("LCB_API_KEY", "  sk-lcb  ")
	if got := bearerFromEnv(); got != "sk-lcb" {
		t.Fatalf("LCB_API_KEY wins and is trimmed: got %q", got)
	}
}

func TestGatewaySamplerRetainsFinishReasonAndReasoningOnly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","reasoning_content":"unfinished reasoning"},"finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":8}}`))
	}))
	defer srv.Close()
	sample := gatewaySampler(srv.Client(), livecodebench.RawArmConfig{Endpoint: srv.URL, Model: "m"}, 8)
	content, usage, err := sample(context.Background(), livecodebench.Problem{QuestionID: "q", Prompt: "p"}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if content != "unfinished reasoning" || usage.FinishReason != "length" || !usage.ReasoningOnly {
		t.Fatalf("content=%q usage=%+v", content, usage)
	}
}

func TestRawArmSummarySurfacesTerminationCounts(t *testing.T) {
	report := livecodebench.RawArmReport{Model: "m", Endpoint: "e", N: 2, Problems: make([]livecodebench.RawArmProblem, 3), Usage: livecodebench.RawArmUsage{CachedPromptTokens: 7, Truncated: 2, ReasoningOnly: 1}}
	got := rawArmSummary(report)
	for _, want := range []string{"2 truncated", "1 reasoning-only"} {
		if !strings.Contains(got, want) {
			t.Fatalf("summary %q does not contain %q", got, want)
		}
	}
}
