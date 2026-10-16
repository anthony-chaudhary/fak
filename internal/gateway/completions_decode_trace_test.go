package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func postLegacyCompletion(t *testing.T, srv *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, req)
	return rr
}

func TestLegacyCompletionsNativeDecodeTokenIDsOptIn(t *testing.T) {
	srv := chatDecodeTraceNativeServer(t)
	rr := postLegacyCompletion(t, srv, `{"model":"synthetic-live","prompt":"trace and tokens","max_tokens":4,"temperature":0,"fak_decode_trace":true,"fak":{"native_decode_token_ids":true}}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp struct {
		Object string `json:"object"`
		Usage  struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
		Fak *struct {
			DecodeTrace *struct {
				Schema string            `json:"schema"`
				Events []json.RawMessage `json:"events"`
			} `json:"decode_trace"`
			TokenIDs *struct {
				Schema   string `json:"schema"`
				Engine   string `json:"engine"`
				TokenIDs []int  `json:"token_ids"`
			} `json:"native_decode_token_ids"`
		} `json:"fak"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Object != "text_completion" || resp.Fak == nil || resp.Fak.DecodeTrace == nil || resp.Fak.TokenIDs == nil {
		t.Fatalf("response missing requested trace or token IDs: %s", rr.Body.String())
	}
	ids := resp.Fak.TokenIDs
	if ids.Schema != agent.NativeDecodeTokenIDsSchema || ids.Engine != agent.NativeDecodeTokenIDsEngine {
		t.Fatalf("token-ID provenance = schema %q engine %q", ids.Schema, ids.Engine)
	}
	if len(ids.TokenIDs) != 4 || resp.Usage.CompletionTokens != 4 || len(resp.Fak.DecodeTrace.Events) != 4 {
		t.Fatalf("token_ids/completion/events = %d/%d/%d, want 4 each", len(ids.TokenIDs), resp.Usage.CompletionTokens, len(resp.Fak.DecodeTrace.Events))
	}
}

func TestLegacyCompletionsWithoutOptInHasNoFakKey(t *testing.T) {
	srv := newTestServer(t)
	planner := &chatDecodeTraceCountingPlanner{native: true}
	srv.planner = planner
	rr := postLegacyCompletion(t, srv, `{"prompt":"x"}`)
	if rr.Code != http.StatusOK || planner.calls != 1 || bytes.Contains(rr.Body.Bytes(), []byte(`"fak"`)) {
		t.Fatalf("status/calls/body = %d/%d/%s, want 200/1/no fak key", rr.Code, planner.calls, rr.Body.String())
	}
}

func TestLegacyCompletionsDecodeTraceRejectsBeforeInference(t *testing.T) {
	cases := []struct {
		name   string
		native bool
		body   string
	}{
		{name: "stream", native: true, body: `{"prompt":"x","stream":true,"fak_decode_trace":true}`},
		{name: "proxy", native: false, body: `{"prompt":"x","fak_decode_trace":true}`},
		{name: "token-ids-without-trace", native: true, body: `{"prompt":"x","fak":{"native_decode_token_ids":true}}`},
		{name: "receipt-unsupported", native: true, body: `{"prompt":"x","fak_decode_trace":true,"fak":{"native_inference_receipt":true}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			planner := &chatDecodeTraceCountingPlanner{native: tc.native}
			srv.planner = planner
			rr := postLegacyCompletion(t, srv, tc.body)
			if rr.Code != http.StatusBadRequest || planner.calls != 0 {
				t.Fatalf("status/calls/body = %d/%d/%s, want 400/0", rr.Code, planner.calls, rr.Body.String())
			}
		})
	}
}
