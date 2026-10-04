package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func postWire(t *testing.T, srv *Server, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, req)
	return rr
}

// siblingWire is one non-chat inference surface that resolves its route through
// prepareChatRoute and then decodes on s.planner.
type siblingWire struct {
	name string
	req  func(model string) (path, body string)
}

func siblingWires() []siblingWire {
	return []siblingWire{
		{"messages", func(m string) (string, string) {
			return "/v1/messages", `{"model":"` + m + `","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}`
		}},
		{"responses", func(m string) (string, string) {
			return "/v1/responses", `{"model":"` + m + `","input":"hi","max_output_tokens":5}`
		}},
		{"completions", func(m string) (string, string) {
			return "/v1/completions", `{"model":"` + m + `","prompt":"hi","max_tokens":5}`
		}},
		{"gemini", func(m string) (string, string) {
			return "/v1beta/models/" + m + ":generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`
		}},
	}
}

// TestSiblingWiresNativeEngineRefuseForeignModel extends the chat-route model
// identity guard to every sibling inference wire: on a single native engine a
// foreign requested model is refused with a typed 400 model_mismatch before any
// inference, while the served name and the engine id keep serving, and a
// non-native (proxy, #82 pass-through) planner is never gated.
func TestSiblingWiresNativeEngineRefuseForeignModel(t *testing.T) {
	const foreign = "deepseek-ai.DeepSeek-V4.1-Flash"
	cases := []struct {
		name     string
		native   bool
		model    string
		wantCode int
		wantErr  string
		wantCall int
	}{
		{"foreign model refused", true, foreign, http.StatusBadRequest, "model_mismatch", 0},
		{"served model allowed", true, "test-model", http.StatusOK, "", 1},
		{"engine id allowed", true, "counting", http.StatusOK, "", 1},
		{"non-native planner passes through", false, foreign, http.StatusOK, "", 1},
	}
	for _, wire := range siblingWires() {
		for _, tc := range cases {
			t.Run(wire.name+"/"+tc.name, func(t *testing.T) {
				srv := newTestServer(t)
				planner := &chatDecodeTraceCountingPlanner{native: tc.native}
				srv.planner = planner
				path, body := wire.req(tc.model)
				rr := postWire(t, srv, path, body)
				if rr.Code != tc.wantCode || planner.calls != tc.wantCall {
					t.Fatalf("status/calls = %d/%d, want %d/%d; body=%s", rr.Code, planner.calls, tc.wantCode, tc.wantCall, rr.Body.String())
				}
				if tc.wantErr != "" {
					if got := chatErrorCode(t, rr.Body.Bytes()); got != tc.wantErr {
						t.Fatalf("error code = %q, want %q; body=%s", got, tc.wantErr, rr.Body.String())
					}
				}
			})
		}
	}
}

// TestSiblingWiresInKernelPlannerRefuseForeignModel drives the production
// in-kernel planner so each sibling guard is witnessed on the real engine type.
func TestSiblingWiresInKernelPlannerRefuseForeignModel(t *testing.T) {
	for _, wire := range siblingWires() {
		t.Run(wire.name, func(t *testing.T) {
			srv := nativeReceiptServer(t)
			path, body := wire.req("deepseek-ai.DeepSeek-V4.1-Flash")
			rr := postWire(t, srv, path, body)
			if rr.Code != http.StatusBadRequest || chatErrorCode(t, rr.Body.Bytes()) != "model_mismatch" {
				t.Fatalf("foreign model: status=%d body=%s, want 400 model_mismatch", rr.Code, rr.Body.String())
			}
			path, body = wire.req("synthetic-live")
			if rr = postWire(t, srv, path, body); rr.Code != http.StatusOK {
				t.Fatalf("served model: status=%d body=%s, want 200", rr.Code, rr.Body.String())
			}
		})
	}
}

// TestChatCompletionsInKernelEchoesServedModel pins that the in-kernel engine
// reports the model it actually decoded: an accepted spelling that differs from
// the served name (case here; a registry alias in production) is answered under
// the served model, not echoed back verbatim.
func TestChatCompletionsInKernelEchoesServedModel(t *testing.T) {
	srv := nativeReceiptServer(t)
	rr := postNativeReceipt(t, srv, `{"model":"SYNTHETIC-LIVE","messages":[{"role":"user","content":"hi"}],"max_tokens":2}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Model != "synthetic-live" {
		t.Fatalf("response model = %q, want served model %q", resp.Model, "synthetic-live")
	}
}

// TestResponsesLiveStreamKeepsOneModel pins constant-model SSE on the live
// Responses wire: an accepted spelling that differs from the served name must
// not open the stream under one model and complete it under another.
func TestResponsesLiveStreamKeepsOneModel(t *testing.T) {
	srv := nativeReceiptServer(t)
	rr := postWire(t, srv, "/v1/responses", `{"model":"SYNTHETIC-LIVE","input":"hi","max_output_tokens":2,"stream":true}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	seen := map[string]bool{}
	for _, line := range strings.Split(rr.Body.String(), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var ev struct {
			Response *struct {
				Model string `json:"model"`
			} `json:"response"`
		}
		if json.Unmarshal([]byte(data), &ev) == nil && ev.Response != nil && ev.Response.Model != "" {
			seen[ev.Response.Model] = true
		}
	}
	if len(seen) != 1 {
		t.Fatalf("stream announced models %v, want exactly one; body=%s", seen, rr.Body.String())
	}
}
