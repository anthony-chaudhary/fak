package main

// agent_goal_endpoint_test.go — the POST-a-goal streaming contract (#3258):
// a valid goal envelope streams session.start → per-step call events → a
// terminal session.end as NDJSON, driven end-to-end over a real HTTP server
// against an injected stub loop; malformed input is refused pre-stream with
// the closed reason vocabulary; a failing loop closes the stream with an
// in-stream error event.

import (
	"bufio"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAgentGoalEndpoint(t *testing.T) {
	scriptedLoop := func(goal string, emit func(agentGoalEvent) error) error {
		if err := emit(agentGoalEvent{Event: "call", Tool: "read_file", Detail: "goal=" + goal}); err != nil {
			return err
		}
		return emit(agentGoalEvent{Event: "call", Tool: "list_dir"})
	}

	t.Run("goal streams governed events", func(t *testing.T) {
		srv := httptest.NewServer(newAgentGoalHandler(scriptedLoop))
		defer srv.Close()

		resp, err := http.Post(srv.URL+agentGoalPath, "application/json", strings.NewReader(`{"goal":"tidy the workspace"}`))
		if err != nil {
			t.Fatalf("POST: %v", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
		if ct := resp.Header.Get("Content-Type"); ct != "application/x-ndjson" {
			t.Fatalf("content-type = %q, want application/x-ndjson", ct)
		}

		var events []agentGoalEvent
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			var ev agentGoalEvent
			if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
				t.Fatalf("line %d is not one JSON event: %v (%q)", len(events), err, sc.Text())
			}
			events = append(events, ev)
		}
		if err := sc.Err(); err != nil {
			t.Fatalf("stream read: %v", err)
		}

		want := []agentGoalEvent{
			{Event: "session.start", Goal: "tidy the workspace"},
			{Event: "call", Tool: "read_file", Detail: "goal=tidy the workspace"},
			{Event: "call", Tool: "list_dir"},
			{Event: "session.end"},
		}
		if len(events) != len(want) {
			t.Fatalf("streamed %d events, want %d: %+v", len(events), len(want), events)
		}
		for i := range want {
			if events[i] != want[i] {
				t.Fatalf("event[%d] = %+v, want %+v", i, events[i], want[i])
			}
		}
	})

	t.Run("malformed body refused with closed reason", func(t *testing.T) {
		srv := httptest.NewServer(newAgentGoalHandler(scriptedLoop))
		defer srv.Close()

		for name, tc := range map[string]struct {
			method, body, wantReason string
			wantStatus               int
		}{
			"not json":     {http.MethodPost, `{not json`, "BAD_GOAL_ENVELOPE", http.StatusBadRequest},
			"blank goal":   {http.MethodPost, `{"goal":"  "}`, "EMPTY_GOAL", http.StatusBadRequest},
			"wrong method": {http.MethodGet, ``, "METHOD_NOT_ALLOWED", http.StatusMethodNotAllowed},
		} {
			req, err := http.NewRequest(tc.method, srv.URL+agentGoalPath, strings.NewReader(tc.body))
			if err != nil {
				t.Fatalf("%s: build request: %v", name, err)
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatalf("%s: do: %v", name, err)
			}
			var refusal map[string]string
			if err := json.NewDecoder(resp.Body).Decode(&refusal); err != nil {
				t.Fatalf("%s: refusal is not JSON: %v", name, err)
			}
			resp.Body.Close()
			if resp.StatusCode != tc.wantStatus {
				t.Fatalf("%s: status = %d, want %d", name, resp.StatusCode, tc.wantStatus)
			}
			if !strings.HasPrefix(refusal["error"], tc.wantReason) {
				t.Fatalf("%s: refusal %q does not carry closed reason %q", name, refusal["error"], tc.wantReason)
			}
		}
	})

	t.Run("loop failure lands as in-stream error event", func(t *testing.T) {
		failing := func(string, func(agentGoalEvent) error) error { return errors.New("planner unavailable") }
		w := httptest.NewRecorder()
		newAgentGoalHandler(failing).ServeHTTP(w, httptest.NewRequest(http.MethodPost, agentGoalPath, strings.NewReader(`{"goal":"g"}`)))
		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (stream already committed)", w.Code)
		}
		lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
		if len(lines) != 2 {
			t.Fatalf("streamed %d lines, want 2 (session.start, error): %q", len(lines), lines)
		}
		var last agentGoalEvent
		if err := json.Unmarshal([]byte(lines[1]), &last); err != nil {
			t.Fatalf("terminal line: %v", err)
		}
		if last.Event != "error" || last.Error != "planner unavailable" {
			t.Fatalf("terminal event = %+v, want error/planner unavailable", last)
		}
	})
}
