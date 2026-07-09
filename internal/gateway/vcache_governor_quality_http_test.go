package gateway

// vcache_governor_quality_http_test.go -- the END-TO-END HTTP guard for the M5 governor
// verdict-quality block (#1492).
//
// The unit tests in vcache_governor_quality_test.go prove the chain verifier, the keep-bit,
// and the fail-closed scoring in process. They stop short of the seam the operator actually
// reads: that a real served turn makes /debug/vars emit `vcache_governor_quality` with a
// verified chain. That omitempty wiring (debug.go) is the one uncovered rung, and a nil
// return there would silently drop the block without failing a single unit test. This
// closes it: a chat turn flows through the full live chain (HTTP -> handler ->
// logInferenceTurn -> observeVCacheTurn -> observeVCacheGovernorDecision -> journal) and
// the audited score is then read back off the wire.

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestDebugVarsEndpointExposesVCacheGovernorQualityOverHTTP proves the governor's verdicts
// are scorable and non-forgeable at the boundary an operator scrapes: a served turn with
// provider cache activity journals a verdict, and /debug/vars publishes the audited score
// with its hash chain verified.
func TestDebugVarsEndpointExposesVCacheGovernorQualityOverHTTP(t *testing.T) {
	s := newTestServer(t)
	s.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage: agent.Usage{
			PromptTokens:             100,
			CompletionTokens:         4,
			CacheReadInputTokens:     40000,
			CacheCreationInputTokens: 500,
		},
	}}

	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	var chat ChatResponse
	code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
		Model:    "test-model",
		Messages: []agent.Message{{Role: "user", Content: "warm then read this prefix"}},
	}, &chat)
	if code != http.StatusOK {
		t.Fatalf("chat status = %d, want 200", code)
	}

	r, err := http.Get(ts.URL + "/debug/vars")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(r.Body)
	r.Body.Close()
	if err != nil {
		t.Fatalf("read /debug/vars body: %v", err)
	}
	if r.StatusCode != http.StatusOK {
		t.Fatalf("/debug/vars status = %d, want 200", r.StatusCode)
	}

	// The omitempty wiring must actually fire on real traffic.
	if !strings.Contains(string(raw), `"vcache_governor_quality"`) {
		t.Fatalf("served /debug/vars omitted the vcache_governor_quality block after a cache turn:\n%s", raw)
	}

	var vars debugVarsResponse
	if err := json.Unmarshal(raw, &vars); err != nil {
		t.Fatalf("decode /debug/vars: %v", err)
	}
	q := vars.VCacheGovQuality
	if q == nil {
		t.Fatal("a served chat turn with cache activity must expose vcache_governor_quality over HTTP")
	}
	// Non-forgeable: the chain the live loop wrote must verify when read back off the wire.
	if !q.ChainVerified {
		t.Fatalf("live governor journal failed verification over HTTP; broke at seq %d", q.ChainBreakSeq)
	}
	if q.Records < 1 {
		t.Fatalf("records = %d, want >= 1 (one served turn journals one verdict)", q.Records)
	}
	// Scorable: the score is present, bounded, and consistent with the kept bits.
	if q.Score < 0 || q.Score > 1 {
		t.Fatalf("score %v outside [0,1]", q.Score)
	}
	if want := float64(q.Kept) / float64(q.Records); q.Score != want {
		t.Fatalf("score %v != kept/records %v", q.Score, want)
	}
	if len(q.ByDecision) == 0 {
		t.Fatal("a verified chain must publish the per-decision breakdown over HTTP")
	}
	// Law A2 labeling survives serialization: the score is fak's DECISION, not a provider
	// counter, so an operator scraping the endpoint cannot mistake it for OBSERVED truth.
	if q.Provenance != "DECISION" {
		t.Fatalf("provenance = %q, want DECISION", q.Provenance)
	}
	if q.Schema != vcacheGovernorQualitySchema {
		t.Fatalf("schema = %q, want %q", q.Schema, vcacheGovernorQualitySchema)
	}
}
