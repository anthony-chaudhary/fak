package startupbisectwitness_test

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

const (
	artifactSHA   = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	acceptedSHA   = "8145dc0beac8396db08d75dee1a969faf0e80bf9"
	allegedBadSHA = "2a7cbe0c5d1a909df8f1353e6477bb7344734856"
)

type result struct {
	Revision       string `json:"revision"`
	Result         string `json:"result"`
	ElapsedSeconds int    `json:"elapsed_seconds"`
}

type witness struct {
	Schema   string `json:"schema"`
	Issue    int    `json:"issue"`
	Verdict  string `json:"verdict"`
	Artifact struct {
		Bytes  int64  `json:"bytes"`
		SHA256 string `json:"sha256"`
	} `json:"artifact"`
	Readiness struct {
		Endpoint        string `json:"endpoint"`
		DeadlineSeconds int    `json:"deadline_seconds"`
	} `json:"readiness"`
	DocumentedRecipe struct {
		Environment       []string `json:"environment"`
		CacheDisplacement int64    `json:"cache_displacement_bytes"`
		Accepted          result   `json:"accepted"`
		AllegedBad        result   `json:"alleged_bad"`
	} `json:"documented_recipe"`
	ReportedRecipeControl struct {
		MissingRequiredEnvironment string   `json:"missing_required_environment"`
		Accepted                   result   `json:"accepted"`
		IssueEraHeadTrials         []result `json:"issue_era_head_trials"`
	} `json:"reported_recipe_control"`
	Execution struct {
		Engine         string `json:"engine"`
		Backend        string `json:"backend"`
		FallbackActive bool   `json:"fallback_active"`
		LlamaCPPUsed   bool   `json:"llama_cpp_used"`
	} `json:"execution"`
	Boundary struct {
		LastGood *string `json:"last_good"`
		FirstBad *string `json:"first_bad"`
		Reason   string  `json:"reason"`
	} `json:"boundary"`
	Invariant struct {
		Path               string   `json:"path"`
		Name               string   `json:"name"`
		Statement          string   `json:"statement"`
		ChangedSourcePaths []string `json:"changed_source_paths"`
	} `json:"bounded_invariant"`
}

func TestStartupBisectReadback(t *testing.T) {
	body, err := os.ReadFile("bisect.json")
	if err != nil {
		t.Fatal(err)
	}
	var got witness
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak.qwen38-metal-startup-bisect/1" || got.Issue != 8964 {
		t.Fatalf("identity = %q issue=%d", got.Schema, got.Issue)
	}
	if got.Verdict != "NO_BISECT_INVALID_PREDICATE" {
		t.Fatalf("verdict = %q", got.Verdict)
	}
	if got.Artifact.SHA256 != artifactSHA || got.Artifact.Bytes != 17106775008 {
		t.Fatalf("artifact = %#v", got.Artifact)
	}
	if got.Readiness.DeadlineSeconds != 180 || !strings.HasSuffix(got.Readiness.Endpoint, "/v1/models") {
		t.Fatalf("readiness = %#v", got.Readiness)
	}
	assertEndpointReady(t, got.DocumentedRecipe.Accepted, acceptedSHA)
	assertEndpointReady(t, got.DocumentedRecipe.AllegedBad, allegedBadSHA)
	if got.DocumentedRecipe.CacheDisplacement <= 36<<30 {
		t.Fatalf("cache displacement = %d, want > physical memory", got.DocumentedRecipe.CacheDisplacement)
	}
	if !contains(got.DocumentedRecipe.Environment, "FAK_Q4K=1") {
		t.Fatalf("documented environment = %v", got.DocumentedRecipe.Environment)
	}
	if got.ReportedRecipeControl.MissingRequiredEnvironment != "FAK_Q4K=1" {
		t.Fatalf("missing environment = %q", got.ReportedRecipeControl.MissingRequiredEnvironment)
	}
	if got.ReportedRecipeControl.Accepted.Revision != acceptedSHA || got.ReportedRecipeControl.Accepted.Result == "READY" {
		t.Fatalf("reported-recipe accepted control = %#v", got.ReportedRecipeControl.Accepted)
	}
	if len(got.ReportedRecipeControl.IssueEraHeadTrials) != 2 {
		t.Fatalf("issue-era trials = %d", len(got.ReportedRecipeControl.IssueEraHeadTrials))
	}
	for _, trial := range got.ReportedRecipeControl.IssueEraHeadTrials {
		if trial.Revision != allegedBadSHA || trial.Result == "READY" {
			t.Fatalf("issue-era trial = %#v", trial)
		}
	}
	if got.Boundary.LastGood != nil || got.Boundary.FirstBad != nil || got.Boundary.Reason == "" {
		t.Fatalf("invalid boundary = %#v", got.Boundary)
	}
	if got.Invariant.Path != "cmd/fak/serve_load_helpers.go" || !strings.Contains(got.Invariant.Statement, "FAK_Q4K") || len(got.Invariant.ChangedSourcePaths) != 0 {
		t.Fatalf("bounded invariant = %#v", got.Invariant)
	}
	if got.Execution.Engine != "inkernel" || got.Execution.Backend != "metal" || got.Execution.FallbackActive || got.Execution.LlamaCPPUsed {
		t.Fatalf("execution = %#v", got.Execution)
	}

	for _, name := range []string{"accepted.log", "current-main-run-1.log", "current-main-run-2.log"} {
		assertScrubbedLog(t, name)
	}
}

func assertEndpointReady(t *testing.T, got result, revision string) {
	t.Helper()
	if got.Revision != revision || got.Result != "READY" || got.ElapsedSeconds <= 0 || got.ElapsedSeconds >= 180 {
		t.Fatalf("endpoint = %#v", got)
	}
}

func assertScrubbedLog(t *testing.T, name string) {
	t.Helper()
	body, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	text := string(body)
	for _, required := range []string{"artifact_sha256=" + artifactSHA, "revision=", "readiness_endpoint=", "result="} {
		if !strings.Contains(text, required) {
			t.Fatalf("%s missing %q", name, required)
		}
	}
	for _, forbidden := range []string{"/Users/", "anthony", "Serial Number", "Hardware UUID", "credential", "token="} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("%s contains private marker %q", name, forbidden)
		}
	}
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
