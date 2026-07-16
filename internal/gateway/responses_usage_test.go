package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// responses_usage_test.go is the #4776 witness: the client-facing Responses wire
// (POST /v1/responses) must preserve the UPSTREAM provider's
// input_tokens_details.cached_tokens instead of silently dropping it to a fabricated
// zero. Before the fix, responsesUsage carried only input/output/total, so a Codex
// fak-route session recorded cached_input == 0 even when the gateway observed the
// upstream cache hit — making route comparisons and provider-dollar accounting
// misleading. These tests fail on the pre-fix three-field projection and pass after it.

// TestResponsesUsageFromPreservesCachedInputDetails covers the four DoD fixtures at the
// projection seam: an upstream cached hit, a WITNESSED zero (the provider reported a
// counter whose value is 0), an OMITTED/unknown detail, and a local/in-kernel turn with
// no provider counter. The witnessed zero renders `"cached_tokens":0` (present, not
// dropped); the omitted and local cases drop the whole input_tokens_details subobject so
// a consumer reads "unknown", never a fabricated measured zero.
func TestResponsesUsageFromPreservesCachedInputDetails(t *testing.T) {
	cases := []struct {
		name       string
		usage      agent.Usage
		wantSub    string // required wire substring when a detail is expected
		wantDetail bool
		wantCached int
	}{
		{
			name:       "upstream_cached_hit",
			usage:      agent.Usage{PromptTokens: 1000, CompletionTokens: 20, TotalTokens: 1020, PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 800}},
			wantSub:    `"input_tokens_details":{"cached_tokens":800}`,
			wantDetail: true,
			wantCached: 800,
		},
		{
			name:       "witnessed_zero",
			usage:      agent.Usage{PromptTokens: 500, CompletionTokens: 10, TotalTokens: 510, PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 0}},
			wantSub:    `"input_tokens_details":{"cached_tokens":0}`,
			wantDetail: true,
			wantCached: 0,
		},
		{
			name:       "input_tokens_details_fallback",
			usage:      agent.Usage{PromptTokens: 100, CompletionTokens: 5, TotalTokens: 105, InputTokensDetails: &agent.UsageTokenDetails{CachedTokens: 42}},
			wantSub:    `"input_tokens_details":{"cached_tokens":42}`,
			wantDetail: true,
			wantCached: 42,
		},
		{
			name:       "omitted_unknown_detail",
			usage:      agent.Usage{PromptTokens: 300, CompletionTokens: 7, TotalTokens: 307},
			wantDetail: false,
		},
		{
			name:       "local_in_kernel_no_counter",
			usage:      agent.Usage{PromptTokens: 50, CompletionTokens: 5, TotalTokens: 55},
			wantDetail: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ru := responsesUsageFrom(c.usage)
			raw, err := json.Marshal(ru)
			if err != nil {
				t.Fatal(err)
			}
			// The input/output/total triple is always forwarded verbatim.
			if ru.InputTokens != c.usage.PromptTokens || ru.OutputTokens != c.usage.CompletionTokens || ru.TotalTokens != c.usage.TotalTokens {
				t.Fatalf("token triple not forwarded verbatim: wire=%+v from usage=%+v", ru, c.usage)
			}
			if c.wantDetail {
				if ru.InputTokensDetails == nil {
					t.Fatalf("expected input_tokens_details forwarded, got nil; wire=%s", raw)
				}
				if ru.InputTokensDetails.CachedTokens != c.wantCached {
					t.Fatalf("cached_tokens = %d, want %d", ru.InputTokensDetails.CachedTokens, c.wantCached)
				}
				if !strings.Contains(string(raw), c.wantSub) {
					t.Fatalf("wire = %s, want substring %s", raw, c.wantSub)
				}
			} else {
				if ru.InputTokensDetails != nil {
					t.Fatalf("no provider counter, but a detail was synthesized: %+v", ru.InputTokensDetails)
				}
				if strings.Contains(string(raw), "input_tokens_details") {
					t.Fatalf("omitted/unknown detail must NOT synthesize input_tokens_details on the wire: %s", raw)
				}
			}
		})
	}
}

// TestResponsesRouteForwardsUpstreamCachedInputParity is the DoD parity fixture: the same
// deterministic upstream Responses usage flows through (A) direct parsing via the real
// inbound adapter and (B) POST /v1/responses, and the two must agree on input/output/total
// AND cached_tokens at the Codex-visible wire. The raw-bytes assertion is the #4776 core:
// the pre-fix three-field responsesUsage cannot carry input_tokens_details, so the wire
// substring is absent before the change and present after.
func TestResponsesRouteForwardsUpstreamCachedInputParity(t *testing.T) {
	// A deterministic upstream Responses `usage` reporting a real prompt-cache hit —
	// the exact shape internal/agent/adapters_test.go feeds ProviderOpenAIResponses.
	const upstreamResponsesJSON = `{"status":"completed","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":13,"output_tokens":2,"total_tokens":15,"input_tokens_details":{"cached_tokens":6}}}`

	// Path A — direct parsing through the real inbound Responses adapter.
	adapter, err := agent.NewTranscriptAdapter(agent.ProviderOpenAIResponses)
	if err != nil {
		t.Fatal(err)
	}
	directComp, err := adapter.ParseResponse([]byte(upstreamResponsesJSON))
	if err != nil {
		t.Fatalf("direct parse: %v", err)
	}
	if got := directComp.Usage.CachedPromptTokens(); got != 6 {
		t.Fatalf("direct-parsed cached tokens = %d, want 6", got)
	}

	// Path B — the same upstream usage through POST /v1/responses.
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        directComp.Usage,
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json",
		strings.NewReader(`{"model":"m","input":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	raw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", httpResp.StatusCode, raw)
	}

	// The #4776 wire witness: the cached-input detail reaches the Codex-visible bytes.
	if !strings.Contains(string(raw), `"input_tokens_details":{"cached_tokens":6}`) {
		t.Fatalf("Codex-visible wire dropped the cached-input detail (#4776): %s", raw)
	}

	var resp responsesResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("decode 200 body: %v: %s", err, raw)
	}
	if resp.Usage.InputTokens != directComp.Usage.PromptTokens ||
		resp.Usage.OutputTokens != directComp.Usage.CompletionTokens ||
		resp.Usage.TotalTokens != directComp.Usage.TotalTokens {
		t.Fatalf("token triple parity failed: wire=%+v vs direct in/out/total=%d/%d/%d",
			resp.Usage, directComp.Usage.PromptTokens, directComp.Usage.CompletionTokens, directComp.Usage.TotalTokens)
	}
	if resp.Usage.InputTokensDetails == nil ||
		resp.Usage.InputTokensDetails.CachedTokens != directComp.Usage.CachedPromptTokens() {
		t.Fatalf("cached-input parity failed: wire=%+v want cached=%d",
			resp.Usage.InputTokensDetails, directComp.Usage.CachedPromptTokens())
	}
}

// TestResponsesStreamTerminalEventCarriesCachedInputDetails proves the SSE contract:
// intermediate events (response.created) MAY omit usage, but the terminal
// response.completed event carries the SAME cache details as buffered mode — the wire
// Codex actually reads to record cached input on a streamed turn.
func TestResponsesStreamTerminalEventCarriesCachedInputDetails(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "hello"},
		FinishReason: "stop",
		Usage: agent.Usage{
			PromptTokens:        1000,
			CompletionTokens:    20,
			TotalTokens:         1020,
			PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 800},
		},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	httpResp, err := http.Post(ts.URL+"/v1/responses", "application/json",
		strings.NewReader(`{"model":"m","input":"hi","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", httpResp.StatusCode, body)
	}

	events := parseTypedSSE(t, string(body))
	var created, completed typedSSEEvent
	for _, ev := range events {
		switch ev.Event {
		case "response.created":
			created = ev
		case "response.completed":
			completed = ev
		}
	}
	if created.Event == "" || completed.Event == "" {
		t.Fatalf("missing created/completed events: %v", events)
	}
	// Intermediate event omits usage details (the created envelope zeroes usage).
	if strings.Contains(created.Data, "input_tokens_details") {
		t.Errorf("response.created must omit usage details, got: %s", created.Data)
	}
	// Terminal event carries the cache detail, identical to buffered mode.
	if !strings.Contains(completed.Data, `"input_tokens_details":{"cached_tokens":800}`) {
		t.Errorf("terminal response.completed dropped the cached-input detail (#4776): %s", completed.Data)
	}
}
