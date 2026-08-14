package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSessionReferenceClientsShareCorpusAndRecoveryAddresses(t *testing.T) {
	state := SessionState{TraceID: "parity-session", Run: "RUNNING", Rev: 1}
	var s *Server
	s, _ = New(Config{EngineID: "mock", Model: "test", ObserveSession: func(context.Context, string) SessionState { return state }, SteerSession: func(_ context.Context, _, _, text string) error {
		s.RecordSessionTerminalOutput("parity-session", []byte("input:"+text+"\n"))
		state.Rev++
		return nil
	}})
	s.RestoreSessionClientState("parity-session", []byte("ready\n"), []SessionEffect{{ID: "effect-1", Verdict: SessionEffectUncertain, Check: "read back"}})
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()
	call := func(method, path string, body any) (int, map[string]any) {
		var r io.Reader
		if body != nil {
			b, _ := json.Marshal(body)
			r = bytes.NewReader(b)
		}
		req, _ := http.NewRequest(method, ts.URL+path, r)
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var v map[string]any
		json.NewDecoder(resp.Body).Decode(&v)
		return resp.StatusCode, v
	}
	_, first := call(http.MethodPost, "/v1/fak/session/parity-session/attach", map[string]any{"client_kind": "terminal", "since": 0})
	descriptor := first["descriptor"].(map[string]any)
	actions := descriptor["actions"].([]any)
	if len(actions) != len(sessionCapabilityCorpus) {
		t.Fatalf("actions=%d want=%d", len(actions), len(sessionCapabilityCorpus))
	}
	attachment := first["attachment_id"].(string)
	epoch := descriptor["execution_epoch"].(string)
	cursor := uint64(first["cursor"].(float64))
	call(http.MethodPost, "/v1/fak/session/parity-session/input", map[string]any{"attachment_id": attachment, "execution_epoch": epoch, "text": "once"})
	call(http.MethodPost, "/v1/fak/session/parity-session/detach", map[string]any{"attachment_id": attachment})
	_, browser := call(http.MethodPost, "/v1/fak/session/parity-session/attach", map[string]any{"client_kind": "browser", "since": cursor})
	if browser["cursor"].(float64) != float64(cursor+1) {
		t.Fatalf("replay address=%v want=%d", browser["cursor"], cursor+1)
	}
	bd := browser["descriptor"].(map[string]any)
	if bd["capability_digest"] != descriptor["capability_digest"] {
		t.Fatal("reference clients disagree on capability corpus")
	}
	if !strings.Contains(bd["terminal"].(map[string]any)["transcript"].(string), "input:once") {
		t.Fatal("disconnect/replay lost transcript")
	}
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/fak/session/parity-session/open", nil)
	s.Handler().ServeHTTP(rr, req)
	html := rr.Body.String()
	for _, action := range sessionCapabilityCorpus {
		if !strings.Contains(html, "data-action='+x.id") && !strings.Contains(html, action.ID) {
			t.Fatal("browser renderer lacks generated action corpus")
		}
	}
}

func TestSessionRecoveryScenarioCorpusIsComplete(t *testing.T) {
	want := []string{"disconnect_replay", "duplicate_delivery", "stale_epoch", "lease_transfer", "pending_approval", "source_process_loss", "interrupted_cutover"}
	if strings.Join(SessionRecoveryScenarios, ",") != strings.Join(want, ",") {
		t.Fatalf("scenarios=%v", SessionRecoveryScenarios)
	}
}
