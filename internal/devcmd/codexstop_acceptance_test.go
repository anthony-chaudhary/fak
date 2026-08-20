package devcmd

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

type scriptedAppServer struct {
	messages []appServerMessage
	err      error
	sent     []any
}

func (s *scriptedAppServer) Send(v any) error { s.sent = append(s.sent, v); return nil }
func (s *scriptedAppServer) Receive(context.Context) (appServerMessage, error) {
	if len(s.messages) == 0 {
		return appServerMessage{}, s.err
	}
	m := s.messages[0]
	s.messages = s.messages[1:]
	return m, nil
}
func (*scriptedAppServer) Close() error { return nil }

func raw(v any) json.RawMessage { b, _ := json.Marshal(v); return b }
func lifecycle(method, id, status string, entries ...map[string]string) appServerMessage {
	params := map[string]any{
		"threadId": "thread-1", "turnId": "turn-1",
		"run": map[string]any{
			"id": id, "eventName": "stop", "handlerType": "command", "source": "plugin",
			"sourcePath": "C:/plugin/hooks.json", "displayOrder": 6, "status": status,
			"startedAt": int64(1787099000), "completedAt": int64(1787099001), "entries": entries,
		},
	}
	return appServerMessage{Method: method, Params: raw(params)}
}

func baseMessages(extra ...appServerMessage) []appServerMessage {
	m := []appServerMessage{{ID: 2, Result: raw(map[string]any{"thread": map[string]string{"id": "thread-1"}})}, {ID: 3, Result: raw(map[string]any{"turn": map[string]string{"id": "turn-1"}})}}
	m = append(m, extra...)
	m = append(m, appServerMessage{Method: "turn/completed"})
	return m
}

func TestRunStopAcceptanceCompleted(t *testing.T) {
	s := &scriptedAppServer{messages: baseMessages(lifecycle("hook/started", "stop:6:C:/plugin/hooks.json", "running"), lifecycle("hook/completed", "stop:6:C:/plugin/hooks.json", "completed"))}
	r := runStopAcceptance(context.Background(), s, "C:/home", "C:/work", "codex", "prompt", "completed")
	if r.Verdict != "PASS" || r.Stop.Denominator != 1 || r.Stop.Succeeded != 1 || len(s.sent) != 4 {
		t.Fatalf("report=%+v sent=%d", r, len(s.sent))
	}
}

func TestRunStopAcceptanceIntentionalBlock(t *testing.T) {
	s := &scriptedAppServer{messages: baseMessages(lifecycle("hook/started", "stop:6:C:/plugin/hooks.json", "running"), lifecycle("hook/completed", "stop:6:C:/plugin/hooks.json", "blocked"))}
	r := runStopAcceptance(context.Background(), s, "h", "w", "codex", "prompt", "blocked")
	if r.Verdict != "PASS" || r.Stop.Blocked != 1 {
		t.Fatalf("report=%+v", r)
	}
}

func TestRunStopAcceptanceIntentionalBlockDoesNotRequireTurnCompleted(t *testing.T) {
	messages := baseMessages(
		lifecycle("hook/started", "stop:7:C:/plugin/hooks.json", "running"),
		lifecycle("hook/started", "stop:6:C:/plugin/hooks.json", "running"),
		lifecycle("hook/completed", "stop:6:C:/plugin/hooks.json", "blocked"),
	)
	s := &scriptedAppServer{messages: messages[:5]}
	r := runStopAcceptance(context.Background(), s, "home", "workspace", "codex", "prompt", "blocked")
	if r.Verdict != "PASS" || r.Stop.Blocked != 1 || r.Stop.Skipped != 1 {
		t.Fatalf("report=%+v", r)
	}
}
func TestRunStopAcceptanceFailsOnHandlerFailureAndInvalidJSON(t *testing.T) {
	for _, tc := range []struct {
		name    string
		entries []map[string]string
		reason  string
		invalid int
		failed  int
	}{
		{"exit", []map[string]string{{"kind": "error", "text": "hook exited with code 1"}}, "STOP_FAILED", 0, 1},
		{"json", []map[string]string{{"kind": "error", "text": "invalid JSON from hook stdout"}}, "STOP_INVALID_JSON", 1, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &scriptedAppServer{messages: baseMessages(lifecycle("hook/completed", "stop:6:C:/plugin/hooks.json", "failed", tc.entries...))}
			r := runStopAcceptance(context.Background(), s, "h", "w", "codex", "p", "completed")
			if r.Verdict != "FAIL" || r.Stop.InvalidJSON != tc.invalid || r.Stop.Failed != tc.failed || !containsString(r.Reasons, tc.reason) {
				t.Fatalf("report=%+v", r)
			}
		})
	}
}

func TestRunStopAcceptanceFailsOnUnfinishedAndTimeout(t *testing.T) {
	s := &scriptedAppServer{messages: baseMessages(lifecycle("hook/started", "stop:6:C:/plugin/hooks.json", "running"))}
	r := runStopAcceptance(context.Background(), s, "h", "w", "codex", "p", "completed")
	if r.Stop.Unknown != 1 || !containsString(r.Reasons, "STOP_UNKNOWN") {
		t.Fatalf("report=%+v", r)
	}
	s = &scriptedAppServer{err: context.DeadlineExceeded}
	r = runStopAcceptance(context.Background(), s, "h", "w", "codex", "p", "completed")
	if !containsString(r.Reasons, "TIMEOUT") {
		t.Fatalf("report=%+v", r)
	}
}

func TestRunStopAcceptanceRejectsProtocolErrors(t *testing.T) {
	s := &scriptedAppServer{messages: []appServerMessage{{ID: 2, Error: &appServerError{Code: -1, Message: "no thread"}}}, err: errors.New("closed")}
	r := runStopAcceptance(context.Background(), s, "h", "w", "codex", "p", "completed")
	if !containsString(r.Reasons, "THREAD_START_FAILED") || r.Detail != "no thread" {
		t.Fatalf("report=%+v", r)
	}
}

func containsString(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
