package gateway

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func chatErrorCode(t *testing.T, body []byte) string {
	t.Helper()
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return ""
	}
	return env.Error.Code
}

// TestChatCompletionsNativeEngineRefusesForeignModel pins model identity on the
// single-native-engine chat route: a requested model that is neither the served
// model nor the engine's own id is refused with a typed 400 model_mismatch before
// inference, while the served name, the engine id, an omitted model, non-native
// planners (#82 pass-through) and the dual proxy side keep serving.
func TestChatCompletionsNativeEngineRefusesForeignModel(t *testing.T) {
	const foreign = "deepseek-ai/DeepSeek-V4.1-Flash"
	cases := []struct {
		name     string
		native   bool
		dual     bool
		body     string
		wantCode int
		wantErr  string
		wantCall int
	}{
		{"foreign model refused", true, false, `{"model":"` + foreign + `","messages":[{"role":"user","content":"hi"}],"max_tokens":5}`, http.StatusBadRequest, "model_mismatch", 0},
		{"foreign model stream refused", true, false, `{"model":"` + foreign + `","stream":true,"messages":[{"role":"user","content":"hi"}],"max_tokens":5}`, http.StatusBadRequest, "model_mismatch", 0},
		{"served model allowed", true, false, `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`, http.StatusOK, "", 1},
		{"engine id allowed", true, false, `{"model":"counting","messages":[{"role":"user","content":"hi"}]}`, http.StatusOK, "", 1},
		{"omitted model allowed", true, false, `{"messages":[{"role":"user","content":"hi"}]}`, http.StatusOK, "", 1},
		{"non-native planner passes through", false, false, `{"model":"` + foreign + `","messages":[{"role":"user","content":"hi"}]}`, http.StatusOK, "", 1},
		{"served model case-insensitive", true, false, `{"model":"TEST-MODEL","messages":[{"role":"user","content":"hi"}]}`, http.StatusOK, "", 1},
		{"dual proxy side passes through", true, true, `{"model":"remote-model","messages":[{"role":"user","content":"hi"}]}`, http.StatusOK, "", 0},
		{"dual local alias serves locally", true, true, `{"model":"local","messages":[{"role":"user","content":"hi"}]}`, http.StatusOK, "", 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			planner := &chatDecodeTraceCountingPlanner{native: tc.native}
			srv.planner = planner
			if tc.dual {
				dual, err := NewDualPlanner(agent.NewMockPlanner("remote"), planner, "")
				if err != nil {
					t.Fatal(err)
				}
				srv.planner = dual
			}
			rr := postNativeReceipt(t, srv, tc.body)
			if rr.Code != tc.wantCode || planner.calls != tc.wantCall {
				t.Fatalf("status/calls = %d/%d, want %d/%d; body=%s", rr.Code, planner.calls, tc.wantCode, tc.wantCall, rr.Body.String())
			}
			if got := chatErrorCode(t, rr.Body.Bytes()); got != tc.wantErr {
				t.Fatalf("error code = %q, want %q", got, tc.wantErr)
			}
		})
	}
}

// TestChatCompletionsInKernelPlannerRefusesForeignModel drives the production
// in-kernel planner so the guard is witnessed on the real native engine type.
func TestChatCompletionsInKernelPlannerRefusesForeignModel(t *testing.T) {
	srv := nativeReceiptServer(t)
	rr := postNativeReceipt(t, srv, `{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"hi"}],"max_tokens":2}`)
	if rr.Code != http.StatusBadRequest || chatErrorCode(t, rr.Body.Bytes()) != "model_mismatch" {
		t.Fatalf("foreign model: status=%d body=%s, want 400 model_mismatch", rr.Code, rr.Body.String())
	}
	rr = postNativeReceipt(t, srv, `{"model":"synthetic-live","messages":[{"role":"user","content":"hi"}],"max_tokens":2}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("served model: status=%d body=%s, want 200", rr.Code, rr.Body.String())
	}
}

// TestChatCompletionsNativeEngineRegistryAliases pins that registry aliases of
// the served artifact are the same model on the wire (the Pi defaults send
// qwen38:27b-q4 to a server started as qwen38:27b) while a different artifact,
// size, or foreign name is refused.
func TestChatCompletionsNativeEngineRegistryAliases(t *testing.T) {
	cases := []struct {
		requested string
		wantCode  int
	}{
		{"qwen38:27b-q4", http.StatusOK},
		{"Qwen38:27B", http.StatusOK},
		{"qwen38:27b-q2k", http.StatusBadRequest},
		{"qwen38:70b", http.StatusBadRequest},
		{"deepseek-ai/DeepSeek-V4.1-Flash", http.StatusBadRequest},
	}
	for _, tc := range cases {
		t.Run(tc.requested, func(t *testing.T) {
			srv := newTestServer(t)
			srv.model = "qwen38:27b"
			srv.planner = &chatDecodeTraceCountingPlanner{native: true}
			rr := postNativeReceipt(t, srv, `{"model":"`+tc.requested+`","messages":[{"role":"user","content":"hi"}]}`)
			if rr.Code != tc.wantCode {
				t.Fatalf("status = %d, want %d; body=%s", rr.Code, tc.wantCode, rr.Body.String())
			}
		})
	}
}
