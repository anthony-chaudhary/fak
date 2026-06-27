package gateway

// vcache_families_http_test.go -- the END-TO-END HTTP guard for the per-family live
// observe view (#935, #788 follow-on).
//
// The unit tests in vcache_families_test.go prove the recorder, the render function, and
// the byte-identity with `fak vcache observe`. They stop short of the issue's literal
// acceptance ("a live gateway RUN exposes the per-family observe view"): that the served
// /debug/vars JSON actually carries the `vcache_families` block after real traffic. The
// existing /debug/vars endpoint test drives a /v1/fak/syscall, which never populates the
// vcache window, so the HTTP serialization of this block was the one uncovered seam. This
// closes it: a chat turn flows through the full live chain (HTTP -> handler ->
// logInferenceTurn -> observeVCacheTurn -> window) and the per-family view is then read
// back off the wire from /debug/vars.

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// TestDebugVarsEndpointExposesVCacheFamiliesOverHTTP proves the per-family observe view is
// reachable over the live wire: a served chat turn carrying provider cache activity makes
// /debug/vars emit the `vcache_families` block, with the per-family governor verdict and
// the OBSERVED/DECISION provenance labels intact. This is the #935 acceptance at the HTTP
// boundary the in-process unit tests do not cross.
func TestDebugVarsEndpointExposesVCacheFamiliesOverHTTP(t *testing.T) {
	s := newTestServer(t)
	// A completion whose usage carries both a cache WRITE and a cache READ, so the live
	// per-family block is active (the no-phantom guard only mints a block once a turn
	// shows provider cache activity).
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

	// The block must be present in the raw served JSON (the omitempty wiring fired).
	if !strings.Contains(string(raw), `"vcache_families"`) {
		t.Fatalf("served /debug/vars omitted the vcache_families block after a cache turn:\n%s", raw)
	}

	var vars debugVarsResponse
	if err := json.Unmarshal(raw, &vars); err != nil {
		t.Fatalf("decode /debug/vars: %v", err)
	}
	fams := vars.VCacheFamilies
	if fams == nil {
		t.Fatal("a served chat turn with cache activity must expose vcache_families over HTTP")
	}
	if fams.TurnsObserved != 1 {
		t.Fatalf("turns_observed = %d, want 1 (one served turn)", fams.TurnsObserved)
	}
	if fams.FamilyCount != 1 || len(fams.Families) != 1 {
		t.Fatalf("family_count = %d, families = %d, want exactly one prefix family", fams.FamilyCount, len(fams.Families))
	}
	// The live per-family slice must carry the M5 governor verdict (a DECISION) and the
	// economics status (OBSERVED-derived), not just the family key.
	if fams.Families[0].GovernorDecision == "" {
		t.Fatalf("live family row is missing the governor verdict: %+v", fams.Families[0])
	}
	if fams.Families[0].Status == "" {
		t.Fatalf("live family row is missing the economics status: %+v", fams.Families[0])
	}
	// Provenance labels must survive serialization so an operator scraping /debug/vars
	// sees who owns each value (Law A2: every value carries an OBSERVED/DECISION label).
	if fams.Provenance["hit_rate"] != string(vcacheobserve.Observed) {
		t.Fatalf("hit_rate provenance = %q, want OBSERVED: %+v", fams.Provenance["hit_rate"], fams.Provenance)
	}
	if fams.Provenance["governor_decision"] != string(vcacheobserve.Decision) {
		t.Fatalf("governor_decision provenance = %q, want DECISION: %+v", fams.Provenance["governor_decision"], fams.Provenance)
	}
}
