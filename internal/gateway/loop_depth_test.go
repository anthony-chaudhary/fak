package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func decodeLoopDepthChatRequest(t *testing.T, body string) ChatRequest {
	t.Helper()
	var req ChatRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("decode %s: %v", body, err)
	}
	return req
}

func stubModelMaxLoops(t *testing.T, caps map[string]int) {
	t.Helper()
	prev := modelMaxLoops
	modelMaxLoops = func(model string) int {
		if n, ok := caps[model]; ok {
			return n
		}
		return 1
	}
	t.Cleanup(func() { modelMaxLoops = prev })
}

// fak-test:runtime fast est=5ms lane=default
func TestResolveChatLoopDepthContract(t *testing.T) {
	stubModelMaxLoops(t, map[string]int{"looped-stub": 4})
	cases := []struct {
		name       string
		body       string
		wantStatus int
		want       *LoopDepthReceipt
	}{
		{"plain model, no loops: untouched", `{"model":"plain"}`, 0, nil},
		{"plain model, effort only: untouched", `{"model":"plain","reasoning_effort":"high"}`, 0, nil},
		{"plain model, loops 1 admitted", `{"model":"plain","loops":1}`, 0, &LoopDepthReceipt{Loops: 1, MaxLoops: 1, Source: LoopDepthSourceRequest}},
		{"plain model, loops 2 refused", `{"model":"plain","loops":2}`, http.StatusBadRequest, nil},
		{"looped, top-level loops", `{"model":"looped-stub","loops":3}`, 0, &LoopDepthReceipt{Loops: 3, MaxLoops: 4, Source: LoopDepthSourceRequest}},
		{"looped, vllm_xargs.loops", `{"model":"looped-stub","vllm_xargs":{"loops":2,"other":"x"}}`, 0, &LoopDepthReceipt{Loops: 2, MaxLoops: 4, Source: LoopDepthSourceRequest}},
		{"looped, both shapes agree", `{"model":"looped-stub","loops":2,"vllm_xargs":{"loops":2}}`, 0, &LoopDepthReceipt{Loops: 2, MaxLoops: 4, Source: LoopDepthSourceRequest}},
		{"looped, both shapes disagree", `{"model":"looped-stub","loops":2,"vllm_xargs":{"loops":3}}`, http.StatusBadRequest, nil},
		{"looped, above max", `{"model":"looped-stub","loops":5}`, http.StatusBadRequest, nil},
		{"looped, zero", `{"model":"looped-stub","loops":0}`, http.StatusBadRequest, nil},
		{"looped, xargs not object", `{"model":"looped-stub","vllm_xargs":"loops=2"}`, http.StatusBadRequest, nil},
		{"looped, xargs loops not int", `{"model":"looped-stub","vllm_xargs":{"loops":"2"}}`, http.StatusBadRequest, nil},
		{"looped, null xargs is absent", `{"model":"looped-stub","vllm_xargs":null}`, 0, &LoopDepthReceipt{Loops: 4, MaxLoops: 4, Source: LoopDepthSourceDefault}},
		{"looped, no loops: full depth", `{"model":"looped-stub","reasoning_effort":"low"}`, 0, &LoopDepthReceipt{Loops: 4, MaxLoops: 4, Source: LoopDepthSourceDefault}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			got, ok := resolveChatLoopDepth(rec, decodeLoopDepthChatRequest(t, tc.body))
			if tc.wantStatus != 0 {
				if ok || rec.Code != tc.wantStatus || !strings.Contains(rec.Body.String(), codeLoopsOutOfRange) {
					t.Fatalf("ok=%v status=%d body=%s, want refusal %d with %s", ok, rec.Code, rec.Body.String(), tc.wantStatus, codeLoopsOutOfRange)
				}
				return
			}
			if !ok || rec.Body.Len() != 0 {
				t.Fatalf("ok=%v body=%s, want admitted with no write", ok, rec.Body.String())
			}
			if (got == nil) != (tc.want == nil) || (got != nil && *got != *tc.want) {
				t.Fatalf("receipt = %+v, want %+v", got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestStampLoopDepthReceiptKeepsAbsentResponseUnchanged(t *testing.T) {
	var resp ChatResponse
	stampLoopDepthReceipt(&resp, nil)
	if resp.Fak != nil {
		t.Fatalf("nil receipt created fak ext: %+v", resp.Fak)
	}
	stampLoopDepthReceipt(&resp, &LoopDepthReceipt{Loops: 2, MaxLoops: 4, Source: LoopDepthSourceRequest})
	out, err := json.Marshal(resp.Fak)
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"loop_depth":{"loops":2,"max_loops":4,"source":"request"}}`; string(out) != want {
		t.Fatalf("fak ext = %s, want %s", out, want)
	}
}
