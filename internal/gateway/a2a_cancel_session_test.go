package gateway

// a2a_cancel_session_test.go — the de-simulated A2A cancel (#2758, epic #2753). A
// message carrying a session_id binds its task to a live served session: the task
// stays "running" (the run IS the task; nothing completes synchronously), and
// POST /a2a/v1/tasks/{id}/cancel drives the REAL session to Draining through the
// injected control seam — the same seam the /v1/fak/session route uses — before the
// record flips. Fail-closed both ways: a bound task with no seam wired, or whose
// session cancel is refused, keeps its record unchanged (a "canceled" record over a
// still-running session is exactly the simulation this de-simulates away).

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// postA2AMessage sends one A2A message and returns the created task's id and state.
func postA2AMessage(t *testing.T, srv *Server, content map[string]interface{}) (string, string) {
	t.Helper()
	body, err := json.Marshal(map[string]interface{}{
		"message_id": "m-2758",
		"from":       "peer-agent",
		"content":    content,
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/a2a/v1/messages", bytes.NewReader(body))
	req.Header.Set("X-Caller-ID", "peer-agent")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("send-message status = %d, body = %s", w.Code, w.Body.String())
	}
	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode send-message response: %v", err)
	}
	taskID, _ := resp["task_id"].(string)
	state, _ := resp["state"].(string)
	if taskID == "" {
		t.Fatalf("send-message returned no task_id: %v", resp)
	}
	return taskID, state
}

func cancelA2ATask(t *testing.T, srv *Server, taskID string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/a2a/v1/tasks/"+taskID+"/cancel", nil)
	req.Header.Set("X-Caller-ID", "peer-agent")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	return w
}

func getA2ATaskState(t *testing.T, srv *Server, taskID string) string {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/a2a/v1/tasks/"+taskID, nil)
	req.Header.Set("X-Caller-ID", "peer-agent")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("get-task status = %d, body = %s", w.Code, w.Body.String())
	}
	var task a2aTask
	if err := json.NewDecoder(w.Body).Decode(&task); err != nil {
		t.Fatalf("decode task: %v", err)
	}
	return task.State
}

func TestA2ACancelBoundTaskCancelsRealSession(t *testing.T) {
	srv := newTestServer(t)
	gotTrace, gotVerb, gotRun := "", "", ""
	srv.controlSession = func(_ context.Context, traceID, verb string, req SessionControlRequest) (SessionState, bool, error) {
		gotTrace, gotVerb, gotRun = traceID, verb, req.Run
		return SessionState{TraceID: traceID, Run: "draining", Rev: 2}, true, nil
	}

	taskID, state := postA2AMessage(t, srv, map[string]interface{}{
		"method":     "laptop.status",
		"session_id": "sess-a2a-2758",
	})
	// Bound = the run is the task: nothing completed synchronously.
	if state != "running" {
		t.Fatalf("bound task state = %q, want %q (a session-bound task must not simulate synchronous completion)", state, "running")
	}

	w := cancelA2ATask(t, srv, taskID)
	if w.Code != http.StatusOK {
		t.Fatalf("cancel status = %d, body = %s", w.Code, w.Body.String())
	}
	// The cancel reached the REAL session through the control seam.
	if gotTrace != "sess-a2a-2758" || gotVerb != "run" || gotRun != "draining" {
		t.Fatalf("control seam saw (trace=%q verb=%q run=%q), want (sess-a2a-2758, run, draining)", gotTrace, gotVerb, gotRun)
	}
	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode cancel response: %v", err)
	}
	sess, ok := resp["session"].(map[string]interface{})
	if !ok || sess["run"] != "draining" {
		t.Fatalf("cancel response session = %v, want the seam's post-cancel drive state (run=draining)", resp["session"])
	}
	if got := getA2ATaskState(t, srv, taskID); got != "canceled" {
		t.Fatalf("post-cancel task state = %q, want canceled", got)
	}
}

func TestA2ACancelBoundTaskFailsClosed(t *testing.T) {
	t.Run("no control seam wired", func(t *testing.T) {
		srv := newTestServer(t)
		srv.controlSession = nil
		taskID, _ := postA2AMessage(t, srv, map[string]interface{}{
			"method":     "laptop.status",
			"session_id": "sess-a2a-noseam",
		})
		if w := cancelA2ATask(t, srv, taskID); w.Code != http.StatusConflict {
			t.Fatalf("cancel status = %d, want 409 (bound task without a seam must refuse a record-only cancel)", w.Code)
		}
		if got := getA2ATaskState(t, srv, taskID); got != "running" {
			t.Fatalf("task state = %q after refused cancel, want running (record must not flip)", got)
		}
	})
	t.Run("session cancel refused", func(t *testing.T) {
		srv := newTestServer(t)
		srv.controlSession = func(_ context.Context, traceID, _ string, _ SessionControlRequest) (SessionState, bool, error) {
			return SessionState{TraceID: traceID, Run: "stopped"}, false, nil // terminal session
		}
		taskID, _ := postA2AMessage(t, srv, map[string]interface{}{
			"method":     "laptop.status",
			"session_id": "sess-a2a-terminal",
		})
		if w := cancelA2ATask(t, srv, taskID); w.Code != http.StatusConflict {
			t.Fatalf("cancel status = %d, want 409 (refused session cancel must not flip the record)", w.Code)
		}
		if got := getA2ATaskState(t, srv, taskID); got != "running" {
			t.Fatalf("task state = %q after refused cancel, want running", got)
		}
	})
}

// TestA2AUnboundTaskKeepsRecordOnlyBehavior pins the unbound path: no session_id
// means no live run backs the task, so the record still completes synchronously and
// cancel keeps its historical record-only semantics (a completed record refuses).
func TestA2AUnboundTaskKeepsRecordOnlyBehavior(t *testing.T) {
	srv := newTestServer(t)
	called := false
	srv.controlSession = func(_ context.Context, _, _ string, _ SessionControlRequest) (SessionState, bool, error) {
		called = true
		return SessionState{}, true, nil
	}
	taskID, state := postA2AMessage(t, srv, map[string]interface{}{"method": "laptop.status"})
	if state != "completed" {
		t.Fatalf("unbound task state = %q, want completed (the record-only path is unchanged)", state)
	}
	if w := cancelA2ATask(t, srv, taskID); w.Code != http.StatusConflict {
		t.Fatalf("cancel of a completed unbound task = %d, want 409", w.Code)
	}
	if called {
		t.Fatal("unbound task cancel reached the session control seam — it must stay record-only")
	}
}
