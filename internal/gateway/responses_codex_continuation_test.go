package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// responses_codex_continuation_test.go is the END-TO-END regression witness for the
// guarded-Codex turn-2 HTTP 400 (#3865). The isolated adapter test
// (agent.TestOpenAIResponsesContinuationKeepsCallIDOutOfItemID) proves the OUTBOUND
// marshaler omits the item id; the gateway proxy tests prove the verdict pass runs on
// the wire. NEITHER exercises the seam guarded Codex actually hits: a real Responses
// continuation entering POST /v1/responses, canonicalized, and RE-MARSHALED to an
// upstream that enforces OpenAI's real rule ("input[].id must begin with 'fc_'"). The
// gateway proxy tests use a stub planner, so the outbound adapter — where the #3865 bug
// lived — is never invoked. This test wires the REAL openAIResponsesAdapter planner to a
// strict capture-upstream so the whole guarded-Codex round-trip is witnessed.
//
// Codex sends a function_call input item carrying BOTH id ("fc_...", the provider-owned
// output-item id) AND call_id ("call_...", the result binder). Before #3865 the outbound
// marshaler copied call_id into the item id, producing:
//
//	Invalid 'input[N].id': 'call_...'. Expected an ID that begins with 'fc'.
//
// which upstream answered 400 → the gateway surfaced a non-200 → Codex crashed on turn 2.
// The strict upstream below reproduces that exact refusal, so this test goes RED without
// the fix (gateway != 200) and GREEN with it.

// strictResponsesUpstream mimics the load-bearing slice of OpenAI's Responses ingress
// validation: any input item whose id is present but does not begin with "fc_" is a 400,
// exactly as the live API rejected the pre-#3865 continuation. It records the last body it
// received so the test can assert the outbound continuation shape directly.
type strictResponsesUpstream struct {
	mu       sync.Mutex
	lastBody []byte
}

func (u *strictResponsesUpstream) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		u.mu.Lock()
		u.lastBody = body
		u.mu.Unlock()

		var req struct {
			Input []struct {
				Type   string `json:"type"`
				ID     string `json:"id"`
				CallID string `json:"call_id"`
			} `json:"input"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprintf(w, `{"error":{"message":"undecodable body: %s"}}`, err)
			return
		}
		for i, it := range req.Input {
			if it.ID != "" && !strings.HasPrefix(it.ID, "fc_") {
				// The exact refusal the live Responses API returns for a bad item id.
				w.WriteHeader(http.StatusBadRequest)
				fmt.Fprintf(w, `{"error":{"message":"Invalid 'input[%d].id': '%s'. Expected an ID that begins with 'fc'."}}`, i, it.ID)
				return
			}
		}
		// A minimal well-formed completed Responses turn ParseResponse accepts.
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"status":"completed","model":"gpt-5.6-sol","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	}
}

func (u *strictResponsesUpstream) body() []byte {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.lastBody
}

// TestResponsesGuardedCodexContinuationSurvivesStrictUpstream is the guarded-Codex
// dogfood witness: a turn-2 Responses continuation (message → function_call{id:"fc_..",
// call_id:"call_.."} → function_call_output) enters the gateway, is canonicalized, and is
// re-marshaled to an upstream that enforces the real "id must begin with 'fc_'" rule. The
// gateway must return 200 (Codex does not crash), and the captured outbound continuation
// must preserve call_id while NOT smuggling the "call_.." value into the item id.
func TestResponsesGuardedCodexContinuationSurvivesStrictUpstream(t *testing.T) {
	upstream := &strictResponsesUpstream{}
	up := httptest.NewServer(upstream.handler())
	defer up.Close()

	planner, err := agent.NewProviderHTTPPlanner("openai-responses", up.URL, "gpt-5.6-sol", "test-key")
	if err != nil {
		t.Fatalf("build responses planner: %v", err)
	}

	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	const callID = "call_E9wv3KvrwBNMbZ9nEJKFNq0J"
	code, resp := postResponses(t, ts.URL, map[string]any{
		"model": "gpt-5.6-sol",
		"input": []map[string]any{
			{"type": "message", "role": "user", "content": "run a tool"},
			{
				"type":      "function_call",
				"id":        "fc_realProviderItemId",
				"call_id":   callID,
				"name":      "shell_command",
				"arguments": `{"command":"Get-Location"}`,
			},
			{"type": "function_call_output", "call_id": callID, "output": `{"ok":true}`},
		},
		"tools": []map[string]any{
			{"type": "function", "name": "shell_command"},
		},
	})

	// Primary witness: the guarded-Codex continuation did NOT surface an upstream 400.
	// Before #3865 the outbound item id was the "call_.." value, the strict upstream
	// answered 400, and the gateway surfaced a non-200 here — the turn-2 crash.
	if code != http.StatusOK {
		t.Fatalf("gateway status = %d, want 200 (guarded-Codex turn-2 continuation must not 400 upstream); this is the #3865 crash if the outbound item id carries the call_ value", code)
	}
	if resp.Status != "completed" {
		t.Fatalf("response status = %q, want completed", resp.Status)
	}

	// Corroborating witness: inspect the exact bytes fak sent upstream. The function_call
	// item must keep call_id and must NOT set an item id to the "call_.." value (the
	// forbidden shape). fak discards the client-supplied "fc_" item id on ingress and
	// omits it on egress, so the outbound id is empty here.
	raw := upstream.body()
	if len(raw) == 0 {
		t.Fatal("upstream captured no body: the gateway never reached the real outbound adapter")
	}
	var out struct {
		Input []struct {
			Type   string `json:"type"`
			ID     string `json:"id"`
			CallID string `json:"call_id"`
			Name   string `json:"name"`
		} `json:"input"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode captured upstream body: %v\n%s", err, raw)
	}
	var sawCall bool
	for i, it := range out.Input {
		if it.Type != "function_call" {
			continue
		}
		sawCall = true
		if it.CallID != callID {
			t.Errorf("outbound input[%d].call_id = %q, want %q (the result binder must be preserved)", i, it.CallID, callID)
		}
		if it.ID != "" && !strings.HasPrefix(it.ID, "fc_") {
			t.Errorf("outbound input[%d].id = %q: a non-fc_ item id is the #3865 400 (call_id must not leak into the item id)", i, it.ID)
		}
	}
	if !sawCall {
		t.Fatalf("no function_call item in the outbound continuation; the assistant tool call was dropped: %s", raw)
	}
}
