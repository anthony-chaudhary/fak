package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestCapturedPageRendersOperatingStatesAndSecondSkin(t *testing.T) {
	response := httptest.NewRecorder()
	request := httptest.NewRequest("GET", "/", nil)
	handler(newStore()).ServeHTTP(response, request)
	body := response.Body.String()
	for _, want := range []string{
		"Local, bounded, yours.",
		"aria-live=\"polite\"",
		"approval.requested",
		"Approval run",
		"Failure run",
		"data-skin",
		"body[data-skin=\"minimal\"]",
		"p.text",
		"p.summary",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("captured browser render missing %q", want)
		}
	}
	if got := response.Header().Get("Content-Security-Policy"); !strings.Contains(got, "default-src 'self'") {
		t.Fatalf("CSP=%q", got)
	}
}

func TestNormalRunEventsAreOrderedAndResumeExcludesCursor(t *testing.T) {
	s := newStore()
	runID := s.create("prove it")
	events := s.after(runID, 0)
	if len(events) != 8 {
		t.Fatalf("events=%d", len(events))
	}
	for i, event := range events {
		if err := event.Validate(); err != nil {
			t.Fatalf("event %d: %v", i, err)
		}
		if event.Sequence != uint64(i+1) {
			t.Fatalf("sequence=%d want %d", event.Sequence, i+1)
		}
	}
	var message harnesskit.MessagePayload
	if err := json.Unmarshal(events[3].Payload, &message); err != nil {
		t.Fatal(err)
	}
	if message.Text != "offline reply: prove it" {
		t.Fatalf("message=%q", message.Text)
	}
	resumed := s.after(runID, 6)
	if len(resumed) != 2 || resumed[0].Sequence != 7 || resumed[0].Type != harnesskit.EventArtifactPublished {
		t.Fatalf("resumed=%v", resumed)
	}
}

func TestApprovalRequiresMatchingOneShotDecision(t *testing.T) {
	s := newStore()
	runID := s.create("approval: inspect workspace")
	if got := s.after(runID, 0); len(got) != 3 || got[2].Type != harnesskit.EventApprovalRequested {
		t.Fatalf("initial=%v", got)
	}
	if err := s.resolve(runID, "wrong", "approve"); err == nil {
		t.Fatal("mismatched approval accepted")
	}
	if err := s.resolve(runID, "approval-1", "approve"); err != nil {
		t.Fatal(err)
	}
	got := s.after(runID, 3)
	if len(got) != 4 || got[0].Type != harnesskit.EventApprovalResolved || got[1].Type != harnesskit.EventToolCompleted {
		t.Fatalf("resolved=%v", got)
	}
	if err := s.resolve(runID, "approval-1", "approve"); err == nil {
		t.Fatal("approval replay accepted")
	}
}

func TestFailureIsTypedAndTerminal(t *testing.T) {
	s := newStore()
	events := s.after(s.create("failure: demonstrate"), 0)
	if len(events) != 3 || events[1].Type != harnesskit.EventError || events[2].Type != harnesskit.EventRunCompleted {
		t.Fatalf("events=%v", events)
	}
	var failure harnesskit.ErrorPayload
	if err := json.Unmarshal(events[1].Payload, &failure); err != nil {
		t.Fatal(err)
	}
	if failure.Code != "OFFLINE_DEMO_FAILURE" || !failure.Retryable {
		t.Fatalf("failure=%+v", failure)
	}
}

func TestSelfcheckDrivesRenderRunApprovalFailureAndReconnect(t *testing.T) {
	var out strings.Builder
	if err := selfcheck(&out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"HARNESS_WEB_SELFCHECK ok", "protocol=fak.harness.run/v1", "normal=8", "resumed=2", "approval=4", "failure=3", "skins=2", "html_sha256="} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("receipt missing %q: %s", want, out.String())
		}
	}
}

func TestPersistentStoreReopensRunAndExclusiveCursor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sessions.json")
	first, err := newPersistentStore(path)
	if err != nil {
		t.Fatal(err)
	}
	runID := first.create("persist me")
	if got := len(first.after(runID, 0)); got != 8 {
		t.Fatalf("initial events=%d", got)
	}

	reopened, err := newPersistentStore(path)
	if err != nil {
		t.Fatal(err)
	}
	got := reopened.after(runID, 6)
	if len(got) != 2 || got[0].Sequence != 7 || got[1].Sequence != 8 {
		t.Fatalf("reopened exclusive cursor=%v", got)
	}
}

func TestLiveAdapterProjectsNativeSSEWithoutProviderTypes(t *testing.T) {
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Trace-Id"); got != "live-1" {
			t.Errorf("trace=%q", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		frames := []struct{ name, data string }{
			{"message_start", `{"type":"message_start"}`},
			{"turn_started", `{"type":"turn_started","seq":1,"turn":1}`},
			{"tool_started", `{"type":"tool_started","seq":2,"call_id":"read-1","tool":"Read"}`},
			{"result_admitted", `{"type":"result_admitted","seq":3,"call_id":"read-1","tool":"Read"}`},
			{"content_block_delta", `{"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}`},
		}
		for _, f := range frames {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", f.name, f.data)
		}
	}))
	defer gateway.Close()
	adapter := &liveAdapter{baseURL: gateway.URL, client: gateway.Client()}
	events, err := adapter.run(context.Background(), "live-1", "inspect")
	if err != nil {
		t.Fatal(err)
	}
	var kinds []harnesskit.EventType
	for _, e := range events {
		kinds = append(kinds, e.Type)
		if err := e.Validate(); err != nil {
			t.Fatal(err)
		}
	}
	want := []harnesskit.EventType{harnesskit.EventRunStarted, harnesskit.EventMessageStarted, harnesskit.EventToolStarted, harnesskit.EventToolCompleted, harnesskit.EventMessageDelta, harnesskit.EventMessageCompleted, harnesskit.EventRunCompleted}
	if !reflect.DeepEqual(kinds, want) {
		t.Fatalf("kinds=%v want=%v", kinds, want)
	}
}

func TestLiveAdapterFailureBecomesTypedRunFailure(t *testing.T) {
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { http.Error(w, "provider down", http.StatusBadGateway) }))
	defer gateway.Close()
	s := newStore()
	ts := httptest.NewServer(handlerWithLive(s, &liveAdapter{baseURL: gateway.URL, client: gateway.Client()}))
	defer ts.Close()
	runID, err := postRun(ts.Client(), ts.URL, "inspect")
	if err != nil {
		t.Fatal(err)
	}
	events := s.after(runID, 0)
	if len(events) != 3 || events[1].Type != harnesskit.EventError || events[2].Type != harnesskit.EventRunCompleted {
		t.Fatalf("events=%v", events)
	}
	var failure harnesskit.ErrorPayload
	if err := events[1].DecodePayload(&failure); err != nil {
		t.Fatal(err)
	}
	if failure.Code != "LIVE_FAK_ERROR" || !failure.Retryable {
		t.Fatalf("failure=%+v", failure)
	}
}
