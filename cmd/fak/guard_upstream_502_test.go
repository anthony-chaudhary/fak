package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/journal"
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

func TestGuardUpstreamFailureObserver_RecordsAuditJournal502(t *testing.T) {
	audit := journal.OpenMemory()
	const traceID = "test-trace-502"
	obs := guardUpstreamFailureObserver(audit, traceID, nil, nil, nil)

	obs(gateway.UpstreamFailureReceipt{
		HTTPStatus:        http.StatusBadGateway,
		EmittingLayer:     "local_proxy",
		TargetID:          "upstream-model",
		Cause:             "bad gateway",
		ProviderRequestID: "req-provider-1",
		ProxyRequestID:    "req-proxy-1",
		Attempt:           1,
		RetryBudget:       3,
	})

	rows := audit.Recent(0)
	if len(rows) != 1 {
		t.Fatalf("expected 1 audit row, got %d", len(rows))
	}
	if rows[0].Kind != "UPSTREAM_502" {
		t.Fatalf("expected row Kind to be UPSTREAM_502, got %q", rows[0].Kind)
	}
	if rows[0].TraceID != traceID {
		t.Fatalf("expected row TraceID to be %q, got %q", traceID, rows[0].TraceID)
	}
	var receipt gateway.UpstreamFailureReceipt
	if err := json.Unmarshal([]byte(rows[0].Reason), &receipt); err != nil {
		t.Fatalf("failed to unmarshal receipt from row Reason: %v", err)
	}
	if receipt.HTTPStatus != http.StatusBadGateway {
		t.Fatalf("expected HTTPStatus %d, got %d", http.StatusBadGateway, receipt.HTTPStatus)
	}
	if receipt.EmittingLayer != "local_proxy" {
		t.Fatalf("expected EmittingLayer local_proxy, got %q", receipt.EmittingLayer)
	}

	// Non-502 failure records UPSTREAM_FAILURE
	obs(gateway.UpstreamFailureReceipt{
		HTTPStatus:    http.StatusInternalServerError,
		EmittingLayer: "provider",
		TargetID:      "upstream-model",
		Cause:         "server error",
		Attempt:       2,
		RetryBudget:   3,
	})
	rows = audit.Recent(0)
	if len(rows) != 2 {
		t.Fatalf("expected 2 audit rows, got %d", len(rows))
	}
	if rows[1].Kind != "UPSTREAM_FAILURE" {
		t.Fatalf("expected row Kind to be UPSTREAM_FAILURE, got %q", rows[1].Kind)
	}
}

func TestGuardUpstreamFailureObserver_UpdatesWireGaugeAndCrashRetry(t *testing.T) {
	wireGauge := &guardWireErrorGauge{}
	now := time.Now()
	if wireGauge.Recent(now) {
		t.Fatal("expected wireGauge.Recent to be false initially")
	}

	var (
		logCalled        bool
		debugStatsCalled bool
	)
	logf := func(format string, args ...any) {
		logCalled = true
	}
	debugStats := func(format string, args ...any) {
		debugStatsCalled = true
	}

	obs := guardUpstreamFailureObserver(nil, "test-trace-gauge", wireGauge, logf, debugStats)
	obs(gateway.UpstreamFailureReceipt{
		HTTPStatus:    http.StatusBadGateway,
		EmittingLayer: "transport",
		TargetID:      "upstream-host",
		Cause:         "connection reset by peer",
		Attempt:       1,
		RetryBudget:   2,
	})

	if !logCalled {
		t.Error("expected logf to be called on failure receipt")
	}
	if !debugStatsCalled {
		t.Error("expected debugStats to be called on 502 failure receipt")
	}
	if !wireGauge.Recent(time.Now()) {
		t.Fatal("expected wireGauge.Recent to be true after 502 failure receipt")
	}

	// Verify that guardMaybeRetryTransientWireCrash recognizes the observed transient failure
	runErr := wireCrashExitErr(t)
	command := []string{"claude", "-p", "investigate issue"}
	transientObserved := wireGauge.Consume(time.Now())
	if !transientObserved {
		t.Fatal("expected wireGauge.Consume to return true")
	}
	next, ok := guardMaybeRetryTransientWireCrash(runErr, nil, command, "claude", transientObserved, 0, 2, true, nil)
	if !ok {
		t.Fatal("expected guardMaybeRetryTransientWireCrash to return ok=true when transient was observed")
	}
	if len(next) == 0 || next[len(next)-1] != "--continue" {
		t.Fatalf("expected next command to carry --continue, got: %v", next)
	}

	if wireGauge.Recent(time.Now()) {
		t.Fatal("expected wireGauge.Recent to be false after Consume")
	}
}

func TestHandleResponses_RenderTurnDebugErrorOn502(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"upstream down"}`, http.StatusBadGateway)
	}))
	defer upstream.Close()

	var (
		mu          sync.Mutex
		debugOutput strings.Builder
	)
	debugSink := func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		debugOutput.WriteString(fmt.Sprintf(format, args...) + "\n")
	}

	srv, err := gateway.New(gateway.Config{
		EngineID:    "mock",
		Model:       "test-model",
		BaseURL:     upstream.URL + "/v1",
		Provider:    "openai",
		VDSO:        true,
		DebugStatsf: debugSink,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}

	reqBody := []byte(`{"model":"test-model","input":"hello world"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected HTTP 502, got %d (body: %s)", rec.Code, rec.Body.String())
	}

	mu.Lock()
	got := debugOutput.String()
	mu.Unlock()

	if !strings.Contains(got, "FAILED") {
		t.Fatalf("expected debug stats output to contain FAILED, got: %q", got)
	}
	if !strings.Contains(got, "wire=openai_responses") {
		t.Fatalf("expected debug stats output to contain wire=openai_responses, got: %q", got)
	}
}

func TestHandleChatCompletions_RenderTurnDebugErrorOn502(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"upstream down"}`, http.StatusBadGateway)
	}))
	defer upstream.Close()

	var (
		mu          sync.Mutex
		debugOutput strings.Builder
	)
	debugSink := func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		debugOutput.WriteString(fmt.Sprintf(format, args...) + "\n")
	}

	srv, err := gateway.New(gateway.Config{
		EngineID:    "mock",
		Model:       "test-model",
		BaseURL:     upstream.URL + "/v1",
		Provider:    "openai",
		VDSO:        true,
		DebugStatsf: debugSink,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}

	reqBody := []byte(`{"model":"test-model","messages":[{"role":"user","content":"hello world"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected HTTP 502, got %d (body: %s)", rec.Code, rec.Body.String())
	}

	mu.Lock()
	got := debugOutput.String()
	mu.Unlock()

	if !strings.Contains(got, "FAILED") {
		t.Fatalf("expected debug stats output to contain FAILED, got: %q", got)
	}
	if !strings.Contains(got, "wire=openai_chat_completions") {
		t.Fatalf("expected debug stats output to contain wire=openai_chat_completions, got: %q", got)
	}
}

func TestHandleAnthropicMessages_RenderTurnDebugErrorOn502(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"upstream down"}`, http.StatusBadGateway)
	}))
	defer upstream.Close()

	var (
		mu          sync.Mutex
		debugOutput strings.Builder
	)
	debugSink := func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		debugOutput.WriteString(fmt.Sprintf(format, args...) + "\n")
	}

	srv, err := gateway.New(gateway.Config{
		EngineID:    "mock",
		Model:       "test-model",
		BaseURL:     upstream.URL,
		Provider:    "anthropic",
		VDSO:        true,
		DebugStatsf: debugSink,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}

	reqBody := []byte(`{"model":"test-model","max_tokens":100,"messages":[{"role":"user","content":"hello world"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected HTTP 502, got %d (body: %s)", rec.Code, rec.Body.String())
	}

	mu.Lock()
	got := debugOutput.String()
	mu.Unlock()

	if !strings.Contains(got, "FAILED") {
		t.Fatalf("expected debug stats output to contain FAILED, got: %q", got)
	}
	if !strings.Contains(got, "wire=anthropic_messages") {
		t.Fatalf("expected debug stats output to contain wire=anthropic_messages, got: %q", got)
	}
}
