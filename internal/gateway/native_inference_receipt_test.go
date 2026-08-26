package gateway

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestNativeReceiptDeterministicHTTPReadback(t *testing.T) {
	receipt := &agent.NativeInferenceReceipt{
		TokenIDs: []int{7, 11}, SelectedTokenLogprobs: []float64{-0.25, -1.5},
		PrefillSeconds: 0.01, DecodeSeconds: 0.02, Model: "native-test",
		Backend: "cpu-ref", ForwardPath: "cpu/reference", Q4K: true,
	}
	planner := &recordingPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop", Usage: agent.Usage{PromptTokens: 3, CompletionTokens: 2},
		NativeInferenceReceipt: receipt,
	}}
	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := []byte(`{"model":"native-test","messages":[{"role":"user","content":"hi"}],"temperature":0,"fak_native_receipt":true}`)
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d", resp.StatusCode)
	}
	var got ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !planner.got.NativeInferenceReceipt {
		t.Fatal("request opt-in did not reach SampleOpt")
	}
	if got.Fak == nil || got.Fak.NativeInferenceReceipt == nil {
		t.Fatal("response omitted fak native receipt")
	}
	r := got.Fak.NativeInferenceReceipt
	if len(r.TokenIDs) != 2 || len(r.SelectedTokenLogprobs) != 2 || r.TokenIDs[0] != 7 || math.IsNaN(r.SelectedTokenLogprobs[0]) {
		t.Fatalf("receipt=%+v", r)
	}
}

func TestNativeReceiptDefaultPathOmitsExtension(t *testing.T) {
	b, err := json.Marshal(ChatResponse{Object: "chat.completion"})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(b, []byte("native_inference_receipt")) {
		t.Fatalf("default response leaked receipt: %s", b)
	}
}
