package gateway

// a2a_cancel_test.go — the de-simulated A2A cancel witness (#2758, epic #2753).
// Historically POST /a2a/v1/tasks/{id}/cancel only flipped the in-memory task
// record's State field — a peer agent's cancel "succeeded" while the run it named
// kept going. These tests prove the act-path is now real: a task bound to a served
// session (the message's session_id) drives the REAL session's drive-state through
// the injected control seam, asserted on the session table itself, and the handler
// fails CLOSED when the real cancel cannot be applied — a canceled record over a
// still-running session is exactly the simulation this path no longer performs.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// tableControlSession mirrors the cmd/fak controlSession seam over a real
// session.Table — the "run" verb parsed and applied via Transition, exactly the
// production wiring shape — so the test asserts the REAL state transition, not a
// stub's echo.
func tableControlSession(tbl *session.Table) SessionControlFunc {
	return func(_ context.Context, traceID, verb string, req SessionControlRequest) (SessionState, bool, error) {
		if verb != "run" {
			return SessionState{}, false, fmt.Errorf("unexpected control verb %q", verb)
		}
		run, ok := session.ParseRunState(req.Run)
		if !ok {
			return SessionState{}, false, fmt.Errorf("unknown run-state %q", req.Run)
		}
		st, applied := tbl.Transition(traceID, run, req.Reason)
		return SessionState{TraceID: st.TraceID, Run: st.Run.String(), Reason: st.Reason, Rev: st.Rev}, applied, nil
	}
}

// a2aSendSessionBound POSTs one session-bound A2A message and returns the created
// task id and reported state.
func a2aSendSessionBound(t *testing.T, h http.Handler, trace string) (taskID, state string) {
	t.Helper()
	body := fmt.Sprintf(`{"message_id":"m-%s","from":"peer-agent","content":{"method":"laptop.check","session_id":%q}}`, trace, trace)
	req := httptest.NewRequest(http.MethodPost, "/a2a/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("send-message: status %d, body %s", w.Code, w.Body.String())
	}
	var resp struct {
		TaskID string `json:"task_id"`
		State  string `json:"state"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("send-message response: %v", err)
	}
	return resp.TaskID, resp.State
}

// TestA2ACancelStopsRealSession is the #2758 done-condition witness: a peer
// agent's tasks/cancel moves a REAL served session to its canceled drive-state —
// observed on the session table, not on the task record's self-report.
func TestA2ACancelStopsRealSession(t *testing.T) {
	const trace = "sess-2758-a2a-cancel"
	tbl := session.NewTable()
	tbl.Decide(trace) // seed a live Running record — the run a peer will cancel

	srv := newTestServer(t)
	srv.controlSession = tableControlSession(tbl)
	h := srv.Handler()

	taskID, state := a2aSendSessionBound(t, h, trace)
	// A session-bound task REPRESENTS the run: its act-path is not simulated, so
	// it must not report instant completion.
	if state != "running" {
		t.Fatalf("session-bound task state at create = %q, want %q (a bound task must not simulate completion)", state, "running")
	}
	if st := tbl.Get(trace); st.Run != session.Running {
		t.Fatalf("send-message moved the session to %v — only cancel may drive the run", st.Run)
	}

	req := httptest.NewRequest(http.MethodPost, "/a2a/v1/tasks/"+taskID+"/cancel", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("cancel: status %d, body %s", w.Code, w.Body.String())
	}

	// THE de-simulation assertion: the REAL session's drive-state moved — the
	// cancel enqueued the drain (Draining; the loop finalizes it to Stopped at its
	// next boundary, the consumption proven loop-side by the #2766 cancel witness).
	if st := tbl.Get(trace); st.Run != session.Draining {
		t.Fatalf("post-cancel session run-state = %v, want Draining (the A2A cancel did not reach the real session)", st.Run)
	}

	// The record flip is the FOLLOWER of the real cancel, not the substitute.
	getReq := httptest.NewRequest(http.MethodGet, "/a2a/v1/tasks/"+taskID, nil)
	gw := httptest.NewRecorder()
	h.ServeHTTP(gw, getReq)
	var got struct {
		State string `json:"state"`
	}
	if err := json.Unmarshal(gw.Body.Bytes(), &got); err != nil {
		t.Fatalf("get-task response: %v", err)
	}
	if got.State != "canceled" {
		t.Fatalf("task state after real cancel = %q, want %q", got.State, "canceled")
	}
}

// TestA2ACancelFailsClosed pins the no-simulation guarantee on the refusing paths:
// when the real session cancel cannot be applied — the session is already terminal,
// or no control seam is wired at all — the handler refuses and the task record is
// NOT flipped to canceled.
func TestA2ACancelFailsClosed(t *testing.T) {
	t.Run("terminal session refuses; task record unchanged", func(t *testing.T) {
		const trace = "sess-2758-a2a-terminal"
		tbl := session.NewTable()
		tbl.Decide(trace)
		srv := newTestServer(t)
		srv.controlSession = tableControlSession(tbl)
		h := srv.Handler()

		taskID, _ := a2aSendSessionBound(t, h, trace)
		if _, ok := tbl.Transition(trace, session.Stopped, "done"); !ok {
			t.Fatal("arranging terminal state refused")
		}

		req := httptest.NewRequest(http.MethodPost, "/a2a/v1/tasks/"+taskID+"/cancel", nil)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Code != http.StatusConflict {
			t.Fatalf("cancel of a terminal-session task: status %d, want 409; body %s", w.Code, w.Body.String())
		}
		gw := httptest.NewRecorder()
		h.ServeHTTP(gw, httptest.NewRequest(http.MethodGet, "/a2a/v1/tasks/"+taskID, nil))
		var got struct {
			State string `json:"state"`
		}
		if err := json.Unmarshal(gw.Body.Bytes(), &got); err != nil {
			t.Fatalf("get-task response: %v", err)
		}
		if got.State == "canceled" {
			t.Fatal("task record flipped to canceled although the real session cancel was refused — that is the simulation this path must not perform")
		}
	})

	t.Run("no control seam wired refuses; task record unchanged", func(t *testing.T) {
		const trace = "sess-2758-a2a-unwired"
		srv := newTestServer(t) // no ControlSession wired
		h := srv.Handler()

		taskID, _ := a2aSendSessionBound(t, h, trace)
		req := httptest.NewRequest(http.MethodPost, "/a2a/v1/tasks/"+taskID+"/cancel", nil)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Code != http.StatusConflict {
			t.Fatalf("cancel of a bound task without a control seam: status %d, want 409; body %s", w.Code, w.Body.String())
		}
		gw := httptest.NewRecorder()
		h.ServeHTTP(gw, httptest.NewRequest(http.MethodGet, "/a2a/v1/tasks/"+taskID, nil))
		var got struct {
			State string `json:"state"`
		}
		if err := json.Unmarshal(gw.Body.Bytes(), &got); err != nil {
			t.Fatalf("get-task response: %v", err)
		}
		if got.State == "canceled" {
			t.Fatal("task record flipped to canceled with no session control wired — a record-only cancel of a bound task is the forbidden simulation")
		}
	})
}
