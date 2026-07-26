package main

// agent_goal_endpoint.go — the agent-runtime spine handler shape (#3258, epic
// #3256, workstream B): POST /v1/fak/agent/sessions accepts a JSON goal
// envelope and streams back one governed agent-loop session as NDJSON events
// (one JSON object per line, flushed as emitted).
//
// This slice proves the HTTP contract as a standalone tested unit: the loop is
// an INJECTED seam (agentGoalLoop), not a live planner/kernel — the handler
// owns envelope validation with a closed refusal vocabulary, the stream
// framing, and the session.start / call / session.end / error wire shape that
// the in-gateway governed loop (agent.RunArm behind the gateway assembly)
// plugs into. Wiring onto the served gateway mux is deliberately NOT done
// here; the gateway-side route composes with this wire contract when it lands.
//
// Wire shape:
//
//	{"event":"session.start", "goal":...}
//	{"event":"call", ...}          (per governed step the loop emits)
//	{"event":"session.end"}        (loop returned nil)
//	{"event":"error", "error":...} (loop failed after the stream opened)
//
// Closed pre-stream refusals (JSON body {"error":"<REASON>: detail"}):
//
//	405 METHOD_NOT_ALLOWED — anything but POST
//	400 BAD_GOAL_ENVELOPE  — body is not a JSON object
//	400 EMPTY_GOAL         — envelope decoded but the goal is blank

import (
	"encoding/json"
	"net/http"
	"strings"
)

// agentGoalPath is the route the issue names for the spine endpoint.
const agentGoalPath = "/v1/fak/agent/sessions"

// agentGoalRequest is the POST body: the goal the governed loop is seeded with.
type agentGoalRequest struct {
	Goal string `json:"goal"`
}

// agentGoalEvent is one streamed NDJSON line of a governed session.
type agentGoalEvent struct {
	Event  string `json:"event"`
	Goal   string `json:"goal,omitempty"`
	Tool   string `json:"tool,omitempty"`
	Detail string `json:"detail,omitempty"`
	Error  string `json:"error,omitempty"`
}

// agentGoalLoop is the injected governed-loop seam: it receives the validated
// goal and an emit callback that frames each event onto the live stream. A nil
// return closes the stream with session.end; an error closes it with an
// in-stream error event (the response status is already committed by then).
type agentGoalLoop func(goal string, emit func(agentGoalEvent) error) error

// newAgentGoalHandler builds the POST-a-goal streaming handler around the
// given loop seam.
func newAgentGoalHandler(loop agentGoalLoop) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			agentGoalRefuse(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED: POST "+agentGoalPath)
			return
		}
		var req agentGoalRequest
		dec := json.NewDecoder(r.Body)
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil {
			agentGoalRefuse(w, http.StatusBadRequest, "BAD_GOAL_ENVELOPE: "+err.Error())
			return
		}
		if strings.TrimSpace(req.Goal) == "" {
			agentGoalRefuse(w, http.StatusBadRequest, "EMPTY_GOAL: body must carry a non-empty \"goal\"")
			return
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		emit := func(ev agentGoalEvent) error {
			line, err := json.Marshal(ev)
			if err != nil {
				return err
			}
			if _, err := w.Write(append(line, '\n')); err != nil {
				return err
			}
			if flusher != nil {
				flusher.Flush()
			}
			return nil
		}

		// The stream opens before the loop runs: session.start is on the wire
		// immediately, then the loop drives per-step events through emit.
		if err := emit(agentGoalEvent{Event: "session.start", Goal: req.Goal}); err != nil {
			return
		}
		if err := loop(req.Goal, emit); err != nil {
			_ = emit(agentGoalEvent{Event: "error", Error: err.Error()})
			return
		}
		_ = emit(agentGoalEvent{Event: "session.end"})
	})
}

// agentGoalRefuse writes a closed-vocabulary pre-stream refusal as JSON.
func agentGoalRefuse(w http.ResponseWriter, status int, reason string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": reason})
}
