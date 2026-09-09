package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestAdmissionImpossibleRequestReturns400 witnesses issue #12655: when a single
// request's estimated tokens exceed the scheduler token budget on an idle controller
// (state.tokens == 0 && req.Tokens > capacity), the gateway must return HTTP 400
// Bad Request with type "invalid_request_error" and code "context_length_exceeded",
// NOT HTTP 429 Too Many Requests or rate_limit_error / scheduler_overloaded.
func TestAdmissionImpossibleRequestReturns400(t *testing.T) {
	policy := DefaultAdmissionPolicy()
	policy.TokenBudget = 50
	controller := NewAdmissionController(policy)

	srv := newTestServer(t)
	srv.SetAdmissionController(controller)

	// 1. Direct Acquire test: a request requiring 100 tokens (> TokenBudget 50) on an idle server
	// must return an AdmissionError with VerdictRefused.
	lease, err := controller.Acquire(context.Background(), SeqRequest{
		TraceID: "impossible-trace",
		Tokens:  100,
	})
	if lease != nil {
		lease.Release()
		t.Fatal("Acquire returned a non-nil lease for impossible request; want refusal")
	}
	var admissionErr *AdmissionError
	if !errors.As(err, &admissionErr) {
		t.Fatalf("Acquire err = %v (%T), want *AdmissionError", err, err)
	}
	if admissionErr.Verdict != VerdictRefused {
		t.Fatalf("admissionErr.Verdict = %v, want VerdictRefused (%v)", admissionErr.Verdict, VerdictRefused)
	}
	wantReason := "request tokens 100 exceed scheduler token budget 50"
	if admissionErr.Reason != wantReason {
		t.Fatalf("admissionErr.Reason = %q, want %q", admissionErr.Reason, wantReason)
	}
	if stats := controller.Stats(); stats.Refused != 1 || stats.Shed != 0 {
		t.Fatalf("controller stats = %+v, want Refused=1 Shed=0", stats)
	}

	// 2. admissionErrorStatus mapping check
	status, code, msg, ok := admissionErrorStatus(admissionErr)
	if !ok {
		t.Fatal("admissionErrorStatus returned ok=false for VerdictRefused AdmissionError")
	}
	if status != http.StatusBadRequest {
		t.Fatalf("admissionErrorStatus status = %d, want %d (HTTP 400 Bad Request)", status, http.StatusBadRequest)
	}
	if code != "context_length_exceeded" {
		t.Fatalf("admissionErrorStatus code = %q, want %q", code, "context_length_exceeded")
	}
	if !strings.Contains(msg, wantReason) {
		t.Fatalf("admissionErrorStatus msg = %q, want containing %q", msg, wantReason)
	}

	// 3. Offer test on impossible request
	v := controller.Offer(SeqRequest{TraceID: "impossible-offer", Tokens: 100})
	if v != VerdictRefused {
		t.Fatalf("Offer = %v, want VerdictRefused", v)
	}
	if v.HTTPStatus() != http.StatusBadRequest {
		t.Fatalf("VerdictRefused.HTTPStatus() = %d, want %d", v.HTTPStatus(), http.StatusBadRequest)
	}
	if v.String() != "refused" {
		t.Fatalf("VerdictRefused.String() = %q, want \"refused\"", v.String())
	}

	// 4. HTTP wire contract: POST /v1/chat/completions on an idle server with impossible tokens
	// must return HTTP 400 Bad Request with type="invalid_request_error" and
	// code="context_length_exceeded", NOT HTTP 429 rate_limit_error or scheduler_overloaded.
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	payload := map[string]any{
		"model": "test-model",
		"messages": []map[string]string{
			{"role": "user", "content": strings.Repeat("A", 300)}, // 300 chars / 4 = 75 tokens > 50
		},
		"max_tokens": 100,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}

	httpReq, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		t.Fatalf("POST chat: %v", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("HTTP status = %d, want 400 Bad Request (NOT 429); body: %s", resp.StatusCode, string(respBytes))
	}

	var errResp struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(respBytes, &errResp); err != nil {
		t.Fatalf("failed to decode response JSON: %v, raw: %s", err, string(respBytes))
	}

	if errResp.Error.Type != "invalid_request_error" {
		t.Errorf("error.type = %q, want %q", errResp.Error.Type, "invalid_request_error")
	}
	if errResp.Error.Code != "context_length_exceeded" {
		t.Errorf("error.code = %q, want %q", errResp.Error.Code, "context_length_exceeded")
	}
	if strings.Contains(string(respBytes), "rate_limit_error") {
		t.Errorf("response body must NOT contain rate_limit_error: %s", string(respBytes))
	}
	if strings.Contains(string(respBytes), "scheduler_overloaded") {
		t.Errorf("response body must NOT contain scheduler_overloaded: %s", string(respBytes))
	}

	// 5. Verify transient queue overload still sheds with 429 rate_limit_error
	lease30, err := controller.Acquire(context.Background(), SeqRequest{TraceID: "t1", Tokens: 30})
	if err != nil || lease30 == nil {
		t.Fatalf("Acquire(30) = (%v, %v), want lease", lease30, err)
	}

	controller.SetMaxWaiting(1)
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	go func() {
		_, _ = controller.Acquire(ctx2, SeqRequest{TraceID: "t2", Tokens: 30})
	}()
	if !awaitAdmissionWaiting(controller, 1, 2*time.Second) {
		t.Fatal("t2 never entered waiting queue")
	}

	// Third request exceeding waiting queue -> sheds with VerdictShed (HTTP 429)
	_, errShed := controller.Acquire(context.Background(), SeqRequest{TraceID: "t3", Tokens: 30})
	var shedErr *AdmissionError
	if !errors.As(errShed, &shedErr) || shedErr.Verdict != VerdictShed {
		t.Fatalf("transient shed err = %v, want VerdictShed", errShed)
	}
	shedStatus, shedCode, _, _ := admissionErrorStatus(shedErr)
	if shedStatus != http.StatusTooManyRequests || shedCode != "scheduler_overloaded" {
		t.Fatalf("transient shed status/code = (%d, %q), want (429, scheduler_overloaded)", shedStatus, shedCode)
	}
	lease30.Release()
	cancel2()
}
