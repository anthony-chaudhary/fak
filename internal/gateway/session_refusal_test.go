package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSessionLifecycleRefusalRecoveryContract(t *testing.T) {
	tests := []struct {
		state, action string
		terminal      bool
		message       string
	}{
		{"paused", "resume", false, "held, not killed"},
		{"draining", "wait_for_drain", false, "is draining"},
		{"stopped", "start_new_session", true, "is stopped"},
	}
	for _, tt := range tests {
		t.Run(tt.state, func(t *testing.T) {
			rec := httptest.NewRecorder()
			writeSessionRefusal(rec, SessionState{TraceID: "session-123", Run: tt.state, Reason: "operator_reason"})
			if rec.Code != http.StatusConflict {
				t.Fatalf("status=%d", rec.Code)
			}
			var body struct {
				Error    struct{ Message, Code string } `json:"error"`
				Recovery SessionLifecycleRecovery       `json:"recovery"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if body.Recovery.State != tt.state || body.Recovery.SessionID != "session-123" || body.Recovery.NextAction != tt.action || body.Recovery.Terminal != tt.terminal || body.Recovery.Retryable {
				t.Fatalf("recovery=%+v", body.Recovery)
			}
			if !strings.Contains(body.Error.Message, tt.message) || !strings.Contains(body.Error.Message, "operator_reason") {
				t.Fatalf("message=%q", body.Error.Message)
			}
			if body.Error.Code != "session_"+tt.state {
				t.Fatalf("code=%q", body.Error.Code)
			}
		})
	}
}
