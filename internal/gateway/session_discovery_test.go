package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestSessionDiscoveryRemoteAttachReplayAndEpochRotation(t *testing.T) {
	var acted string
	s, err := New(Config{EngineID: "mock", Model: "mock", SteerSession: func(_ context.Context, _, _, text string) error { acted = text; return nil }, ObserveSession: func(context.Context, string) SessionState {
		return SessionState{TraceID: "roaming", Run: "RUNNING", Rev: 1}
	}})
	if err != nil {
		t.Fatal(err)
	}
	s.ConfigureSessionClientAuth("local-only")
	s.RestoreSessionClientState("roaming", []byte("same transcript\r\n"), nil)
	pub, err := s.PublishSessionDiscovery("roaming", "https://relay.example.test/session/roaming", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if pub.AccessToken == "" || strings.Contains(discoveryJSON(t, pub.Record), pub.AccessToken) {
		t.Fatal("record leaked bearer")
	}

	request := func(method, path, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+pub.AccessToken)
		rr := httptest.NewRecorder()
		s.Handler().ServeHTTP(rr, req)
		return rr
	}
	discovered := request(http.MethodGet, "/v1/fak/discovery/roaming", "")
	if discovered.Code != http.StatusOK {
		t.Fatalf("discover=%d %s", discovered.Code, discovered.Body.String())
	}
	attach := request(http.MethodPost, "/v1/fak/session/roaming/attach", `{"client_kind":"isolated-device","since":0}`)
	if attach.Code != http.StatusOK {
		t.Fatalf("attach=%d %s", attach.Code, attach.Body.String())
	}
	var attached SessionClientAttachResponse
	if err := json.Unmarshal(attach.Body.Bytes(), &attached); err != nil {
		t.Fatal(err)
	}
	if attached.Descriptor.Terminal.Transcript != "same transcript\r\n" {
		t.Fatalf("transcript=%q", attached.Descriptor.Terminal.Transcript)
	}
	action := request(http.MethodPost, "/v1/fak/session/roaming/input", fmt.Sprintf(`{"attachment_id":%q,"execution_epoch":%q,"text":"answer from second device"}`, attached.AttachmentID, attached.Descriptor.ExecutionEpoch))
	if action.Code != http.StatusOK {
		t.Fatalf("action=%d %s", action.Code, action.Body.String())
	}
	if acted != "answer from second device" {
		t.Fatalf("acted=%q", acted)
	}

	s.clientRuntime().mu.Lock()
	sess := s.clientRuntime().sessions["roaming"]
	oldEpoch := sess.executionEpoch
	s.clientRuntime().mu.Unlock()
	if err := s.ConfigureSessionMove("roaming", SessionPlacement{Provider: "p1", AccountRef: "a1", Model: "m1", Compute: "c1", Capabilities: []string{"describe", "attach", "replay", "act", "detach"}, ContextLimit: 1, BudgetAvailable: 1, ComputeAvailable: true}, SessionMoveHooks{RequestSafePoint: func(context.Context, string) error { return nil }, AdmitDestination: func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error { return nil }, RestoreDestination: func(context.Context, SessionMoveCheckpoint) error { return nil }}); err != nil {
		t.Fatal(err)
	}
	move, err := s.MoveSession(context.Background(), "roaming", SessionMoveRequest{ExecutionEpoch: oldEpoch, Destination: SessionPlacement{Provider: "p2", AccountRef: "a2", Model: "m2", Compute: "c2", Capabilities: []string{"describe", "attach", "replay", "act", "detach"}, ContextLimit: 1, BudgetAvailable: 1, ComputeAvailable: true}})
	if err != nil {
		t.Fatal(err)
	}
	rotated := request(http.MethodGet, "/v1/fak/discovery/roaming", "")
	if rotated.Code != http.StatusOK {
		t.Fatalf("rotated discover=%d %s", rotated.Code, rotated.Body.String())
	}
	var record SessionDiscoveryRecord
	json.Unmarshal(rotated.Body.Bytes(), &record)
	if record.ExecutionEpoch != move.Descriptor.ExecutionEpoch || record.ExecutionEpoch == pub.Record.ExecutionEpoch || record.Generation <= pub.Record.Generation {
		t.Fatalf("record=%+v move=%+v", record, move.Descriptor)
	}
	reattach := request(http.MethodPost, "/v1/fak/session/roaming/attach", `{"client_kind":"isolated-device","since":0}`)
	if reattach.Code != http.StatusOK {
		t.Fatalf("reattach=%d %s", reattach.Code, reattach.Body.String())
	}
	var resumed SessionClientAttachResponse
	json.Unmarshal(reattach.Body.Bytes(), &resumed)
	if string(resumed.Descriptor.Terminal.Transcript) != "same transcript\r\n" || len(resumed.Descriptor.Effects) != 0 {
		t.Fatalf("duplicate or lost state: %+v", resumed)
	}
}

func TestSessionDiscoveryRefusals(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "mock", ObserveSession: func(context.Context, string) SessionState { return SessionState{TraceID: "s", Run: "RUNNING", Rev: 1} }})
	if err != nil {
		t.Fatal(err)
	}
	s.ConfigureSessionClientAuth("local-only")
	s.RestoreSessionClientState("s", nil, nil)
	pub, err := s.PublishSessionDiscovery("s", "https://relay.example.test/s", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	get := func(token string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, "/v1/fak/discovery/s", nil)
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		rr := httptest.NewRecorder()
		s.Handler().ServeHTTP(rr, req)
		return rr
	}
	if rr := get("wrong"); rr.Code != http.StatusUnauthorized || !strings.Contains(rr.Body.String(), "DISCOVERY_UNAUTHORIZED") {
		t.Fatalf("unauthorized=%d %s", rr.Code, rr.Body.String())
	}
	if !s.RevokeSessionDiscovery("s") {
		t.Fatal("revoke")
	}
	if rr := get(pub.AccessToken); rr.Code != http.StatusUnauthorized || !strings.Contains(rr.Body.String(), "DISCOVERY_REVOKED") {
		t.Fatalf("revoked=%d %s", rr.Code, rr.Body.String())
	}
	exp, _ := s.PublishSessionDiscovery("s", "https://relay.example.test/s", time.Nanosecond)
	time.Sleep(time.Millisecond)
	if rr := get(exp.AccessToken); rr.Code != http.StatusUnauthorized || !strings.Contains(rr.Body.String(), "DISCOVERY_EXPIRED") {
		t.Fatalf("expired=%d %s", rr.Code, rr.Body.String())
	}

	s.clientRuntime().mu.Lock()
	s.clientRuntime().sessions["s"].discovery.Record.ExecutionEpoch = "stale"
	s.clientRuntime().mu.Unlock()
	if rr := get(exp.AccessToken); rr.Code != http.StatusUnauthorized && rr.Code != http.StatusConflict {
		t.Fatalf("stale=%d %s", rr.Code, rr.Body.String())
	}
}

func TestSessionDiscoveryRecordExcludesSecrets(t *testing.T) {
	s, err := New(Config{EngineID: "mock", Model: "mock", ObserveSession: func(context.Context, string) SessionState { return SessionState{TraceID: "s", Run: "RUNNING", Rev: 1} }})
	if err != nil {
		t.Fatal(err)
	}
	s.RestoreSessionClientState("s", nil, nil)
	pub, err := s.PublishSessionDiscovery("s", "https://relay.example.test/s", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	body := discoveryJSON(t, pub.Record)
	for _, forbidden := range []string{pub.AccessToken, "credential", "private_hostname", "provider_secret"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("record leaked %q: %s", forbidden, body)
		}
	}
	if _, err := s.PublishSessionDiscovery("s", "https://user:secret@relay.example.test/s", time.Minute); err == nil {
		t.Fatal("credentialed relay accepted")
	}
}

func discoveryJSON(t *testing.T, v any) string {
	t.Helper()
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(v); err != nil {
		t.Fatal(err)
	}
	return b.String()
}
