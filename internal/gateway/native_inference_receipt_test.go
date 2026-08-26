package gateway

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func nativeReceiptServer(t *testing.T) *Server {
	t.Helper()
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_SEED", "9070")
	cfg := kvmmuSynthCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	tok := newByteLevelTokenizer(t)
	srv, err := New(Config{InKernelModel: m, Tokenizer: tok, Model: "synthetic-live"})
	if err != nil {
		t.Fatal(err)
	}
	return srv
}

func postNativeReceipt(t *testing.T, srv *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, req)
	return rr
}

func TestNativeInferenceReceiptProductionPath(t *testing.T) {
	srv := nativeReceiptServer(t)
	messages := []agent.Message{{Role: agent.RoleUser, Content: "measure"}}
	zero := 0.0
	body, _ := json.Marshal(ChatRequest{Model: "synthetic-live", Messages: messages, MaxTokens: 64, Temperature: &zero, Fak: &FakRequestExt{NativeInferenceReceipt: true}})
	rr := postNativeReceipt(t, srv, string(body))
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Fak == nil || resp.Fak.NativeInferenceReceipt == nil {
		t.Fatalf("opt-in response missing fak.native_inference_receipt: %s", rr.Body.String())
	}
	receipt := resp.Fak.NativeInferenceReceipt
	if len(receipt.TokenIDs) != 64 || len(receipt.TokenLogprobs) != len(receipt.TokenIDs) || resp.Usage.CompletionTokens != len(receipt.TokenIDs) {
		t.Fatalf("token ids/logprobs/usage = %d/%d/%d, want 64 equal entries", len(receipt.TokenIDs), len(receipt.TokenLogprobs), resp.Usage.CompletionTokens)
	}
	for i, lp := range receipt.TokenLogprobs {
		if math.IsNaN(lp) || math.IsInf(lp, 0) || lp > 0 {
			t.Fatalf("logprob[%d]=%v, want finite normalized log probability", i, lp)
		}
	}
	for name, seconds := range map[string]float64{"prefill": receipt.PrefillSeconds, "ttft": receipt.TTFTSeconds, "decode": receipt.DecodeSeconds} {
		if seconds < 0 || math.IsNaN(seconds) || math.IsInf(seconds, 0) {
			t.Fatalf("%s_seconds=%v, want finite non-negative", name, seconds)
		}
	}
	if receipt.Model != "synthetic-live" || receipt.Engine != "inkernel" || receipt.Backend != "cpu-ref" || receipt.ForwardPath != "cpu/reference" || receipt.Q4K || receipt.FallbackActive {
		t.Fatalf("execution identity = %+v, want exact synthetic inkernel cpu/reference without Q4K or fallback", receipt)
	}
}

func TestNativeInferenceReceiptDefaultWireOmission(t *testing.T) {
	srv := nativeReceiptServer(t)
	messages := []agent.Message{{Role: agent.RoleUser, Content: "plain"}}
	zero := 0.0
	body, _ := json.Marshal(ChatRequest{Model: "synthetic-live", Messages: messages, MaxTokens: 2, Temperature: &zero})
	rr := postNativeReceipt(t, srv, string(body))
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	if _, present := raw["fak"]; present || bytes.Contains(rr.Body.Bytes(), []byte("native_inference_receipt")) {
		t.Fatalf("default response changed wire shape: %s", rr.Body.String())
	}
}

func TestNativeInferenceReceiptUnsupportedRequestsFailClosed(t *testing.T) {
	native := nativeReceiptServer(t)
	cases := map[string]string{
		"streaming":   `{"messages":[{"role":"user","content":"x"}],"stream":true,"fak":{"native_inference_receipt":true}}`,
		"temperature": `{"messages":[{"role":"user","content":"x"}],"temperature":0.1,"fak":{"native_inference_receipt":true}}`,
		"top-p":       `{"messages":[{"role":"user","content":"x"}],"top_p":0.5,"fak":{"native_inference_receipt":true}}`,
		"logit-bias":  `{"messages":[{"role":"user","content":"x"}],"logit_bias":{"1":1},"fak":{"native_inference_receipt":true}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			rr := postNativeReceipt(t, native, body)
			if rr.Code != http.StatusBadRequest || bytes.Contains(rr.Body.Bytes(), []byte(`"native_inference_receipt":{`)) {
				t.Fatalf("status=%d body=%s, want fail-closed 400 without receipt", rr.Code, rr.Body.String())
			}
		})
	}

	proxy, err := New(Config{Model: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	rr := postNativeReceipt(t, proxy, `{"messages":[{"role":"user","content":"x"}],"temperature":0,"fak":{"native_inference_receipt":true}}`)
	if rr.Code != http.StatusBadGateway || bytes.Contains(rr.Body.Bytes(), []byte(`"native_inference_receipt":{`)) {
		t.Fatalf("non-native planner status=%d body=%s, want fail-closed 502 without stamped receipt", rr.Code, rr.Body.String())
	}
}

func TestNativeInferenceReceiptJSONShapeUsesChosenTokenArrays(t *testing.T) {
	raw, err := json.Marshal(FakExt{NativeInferenceReceipt: &agent.NativeInferenceReceipt{TokenIDs: []int{7}, TokenLogprobs: []float64{-1}}})
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"native_inference_receipt"`, `"token_ids":[7]`, `"token_logprobs":[-1]`, `"fallback_active":false`} {
		if !bytes.Contains(raw, []byte(field)) {
			t.Fatalf("receipt JSON %s missing %s", raw, field)
		}
	}
}
