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

func TestSessionClientTerminalBrowserShareIdentityReplayAndLease(t *testing.T) {
	state := SessionState{TraceID: "logical-7", Run: "RUNNING", Rev: 4}
	inputs := []string{}
	s := &Server{
		observeSession: func(context.Context, string) SessionState { return state },
		steerSession: func(_ context.Context, traceID, principal, text string) error {
			if traceID != "logical-7" {
				t.Fatalf("input trace=%q", traceID)
			}
			inputs = append(inputs, principal+":"+text)
			state.Rev++
			return nil
		},
		sessionFeed: newSessionFeed(32),
	}
	s.PublishSessionRevision(state)

	desc := requestSessionClient[SessionClientDescriptor](t, s, http.MethodGet, "/v1/fak/session/logical-7/client", nil, http.StatusOK)
	terminal := requestSessionClient[SessionClientAttachResponse](t, s, http.MethodPost, "/v1/fak/session/logical-7/attach", SessionClientAttachRequest{ClientKind: "terminal", Since: 0}, http.StatusOK)
	browser := requestSessionClient[SessionClientAttachResponse](t, s, http.MethodPost, "/v1/fak/session/logical-7/attach", SessionClientAttachRequest{ClientKind: "browser", Since: terminal.Cursor}, http.StatusOK)
	if desc.SessionID != terminal.Descriptor.SessionID || desc.SessionID != browser.Descriptor.SessionID {
		t.Fatalf("logical ids diverged: %#v %#v %#v", desc.SessionID, terminal.Descriptor.SessionID, browser.Descriptor.SessionID)
	}
	if desc.ExecutionEpoch != terminal.Descriptor.ExecutionEpoch || desc.ExecutionEpoch != browser.Descriptor.ExecutionEpoch {
		t.Fatal("clients did not attach to one execution epoch")
	}
	if desc.CapabilityDigest != terminal.Descriptor.CapabilityDigest || desc.CapabilityDigest != browser.Descriptor.CapabilityDigest {
		t.Fatal("capability digests diverged")
	}
	if !terminal.InputLease || browser.InputLease {
		t.Fatalf("single writer violated terminal=%t browser=%t", terminal.InputLease, browser.InputLease)
	}

	refused := requestSessionClientError(t, s, http.MethodPost, "/v1/fak/session/logical-7/input", SessionClientActionRequest{AttachmentID: browser.AttachmentID, ExecutionEpoch: desc.ExecutionEpoch, Text: "duplicate"}, http.StatusConflict)
	if refused != "LEASE_NOT_HELD" {
		t.Fatalf("refusal=%q", refused)
	}
	requestSessionClient[map[string]any](t, s, http.MethodPost, "/v1/fak/session/logical-7/detach", SessionClientDetachRequest{AttachmentID: terminal.AttachmentID}, http.StatusOK)
	browser = requestSessionClient[SessionClientAttachResponse](t, s, http.MethodPost, "/v1/fak/session/logical-7/attach", SessionClientAttachRequest{ClientKind: "browser", Since: terminal.Cursor}, http.StatusOK)
	if !browser.InputLease {
		t.Fatal("browser did not acquire released input lease")
	}
	action := requestSessionClient[SessionClientAttachResponse](t, s, http.MethodPost, "/v1/fak/session/logical-7/input", SessionClientActionRequest{AttachmentID: browser.AttachmentID, ExecutionEpoch: desc.ExecutionEpoch, Text: "continue", Principal: "browser"}, http.StatusOK)
	if len(inputs) != 1 || inputs[0] != "browser:continue" {
		t.Fatalf("inputs=%v", inputs)
	}
	requestSessionClient[map[string]any](t, s, http.MethodPost, "/v1/fak/session/logical-7/detach", SessionClientDetachRequest{AttachmentID: browser.AttachmentID}, http.StatusOK)
	terminalReconnect := requestSessionClient[SessionClientAttachResponse](t, s, http.MethodPost, "/v1/fak/session/logical-7/attach", SessionClientAttachRequest{ClientKind: "terminal", Since: terminal.Cursor}, http.StatusOK)
	if len(terminalReconnect.Events) != 1 || terminalReconnect.Events[0].Seq != action.Cursor {
		t.Fatalf("reconnect events=%#v action_cursor=%d", terminalReconnect.Events, action.Cursor)
	}
	if terminalReconnect.Descriptor.State.Rev != 5 {
		t.Fatalf("reconnected rev=%d", terminalReconnect.Descriptor.State.Rev)
	}
}

func TestSessionClientStaleEpochAndCapturedBrowserSurface(t *testing.T) {
	state := SessionState{TraceID: "logical-ui", Run: "RUNNING", Rev: 1}
	s := &Server{observeSession: func(context.Context, string) SessionState { return state }, steerSession: func(context.Context, string, string, string) error { return nil }, sessionFeed: newSessionFeed(8)}
	s.PublishSessionRevision(state)
	attached := requestSessionClient[SessionClientAttachResponse](t, s, http.MethodPost, "/v1/fak/session/logical-ui/attach", SessionClientAttachRequest{ClientKind: "terminal"}, http.StatusOK)
	code := requestSessionClientError(t, s, http.MethodPost, "/v1/fak/session/logical-ui/input", SessionClientActionRequest{AttachmentID: attached.AttachmentID, ExecutionEpoch: "old", Text: "x"}, http.StatusConflict)
	if code != "STALE_EPOCH" {
		t.Fatalf("code=%q", code)
	}

	req := httptest.NewRequest(http.MethodGet, "http://gateway.test/v1/fak/session/logical-ui/open", nil)
	rr := httptest.NewRecorder()
	s.handleFakSession(rr, req)
	body := rr.Body.String()
	for _, want := range []string{"fak session", "Full advertised capabilities", "Shared event tail", "/client", "/attach", "/input", "input lease"} {
		if !strings.Contains(body, want) {
			t.Fatalf("captured browser render missing %q\n%s", want, body)
		}
	}
}

func requestSessionClient[T any](t *testing.T, s *Server, method, target string, body any, status int) T {
	t.Helper()
	var rd *bytes.Reader
	if body == nil {
		rd = bytes.NewReader(nil)
	} else {
		b, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		rd = bytes.NewReader(b)
	}
	req := httptest.NewRequest(method, "http://gateway.test"+target, rd)
	rr := httptest.NewRecorder()
	s.handleFakSession(rr, req)
	if rr.Code != status {
		t.Fatalf("%s %s status=%d want=%d body=%s", method, target, rr.Code, status, rr.Body.String())
	}
	var out T
	if err := json.Unmarshal(rr.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode %s: %v body=%s", target, err, rr.Body.String())
	}
	return out
}

func requestSessionClientError(t *testing.T, s *Server, method, target string, body any, status int) string {
	t.Helper()
	out := requestSessionClient[struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}](t, s, method, target, body, status)
	return out.Error.Code
}

func TestSessionClientAttachFiltersCrossSessionEvents(t *testing.T) {
	states := map[string]SessionState{
		"session-a": {TraceID: "session-a", Run: "RUNNING", Rev: 1},
		"session-b": {TraceID: "session-b", Run: "RUNNING", Rev: 1},
	}
	s, err := New(Config{EngineID: "mock", Model: "mock", ObserveSession: func(_ context.Context, id string) SessionState { return states[id] }})
	if err != nil {
		t.Fatal(err)
	}
	s.PublishSessionRevision(SessionState{TraceID: "session-b", Run: "PAUSED", Rev: 2})
	s.PublishSessionRevision(SessionState{TraceID: "session-a", Run: "RUNNING", Rev: 2})
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	var attached SessionClientAttachResponse
	postClientJSON(t, ts.URL+"/v1/fak/session/session-a/attach", map[string]any{"client_kind": "terminal"}, http.StatusOK, &attached)
	if len(attached.Events) != 1 || attached.Events[0].TraceID != "session-a" {
		t.Fatalf("cross-session event bleed in attach: %+v", attached.Events)
	}
	if attached.Cursor != 2 || attached.Descriptor.EventHead != 2 {
		t.Fatalf("global replay cursor must advance across filtered events: %+v", attached)
	}
}

func TestSessionClientTranscriptAndUncertainEffectSurviveDetach(t *testing.T) {
	state := SessionState{TraceID: "durable-a", Run: "RUNNING", Rev: 1}
	s, err := New(Config{EngineID: "mock", Model: "mock", ObserveSession: func(context.Context, string) SessionState { return state }})
	if err != nil {
		t.Fatal(err)
	}
	transcript := []byte("printf 'same bytes'\r\nsame bytes\r\n")
	s.RecordSessionTerminalOutput("durable-a", transcript[:13])
	s.RecordSessionTerminalOutput("durable-a", transcript[13:])
	if err := s.BeginSessionEffect("durable-a", "effect-before", "create marker", "test -e marker"); err != nil {
		t.Fatal(err)
	}
	if err := s.ResolveSessionEffect("durable-a", "effect-before", SessionEffectKnownNotRun); err != nil {
		t.Fatal(err)
	}
	if err := s.BeginSessionEffect("durable-a", "effect-after", "send payment", "query payment id"); err != nil {
		t.Fatal(err)
	}

	ts := httptest.NewServer(s.Handler())
	defer ts.Close()
	var first SessionClientAttachResponse
	postClientJSON(t, ts.URL+"/v1/fak/session/durable-a/attach", map[string]any{"client_kind": "terminal"}, http.StatusOK, &first)
	postClientJSON(t, ts.URL+"/v1/fak/session/durable-a/detach", map[string]any{"attachment_id": first.AttachmentID}, http.StatusOK, nil)
	var reopened SessionClientAttachResponse
	postClientJSON(t, ts.URL+"/v1/fak/session/durable-a/attach", map[string]any{"client_kind": "browser"}, http.StatusOK, &reopened)

	if got := reopened.Descriptor.Terminal.Transcript; got != string(transcript) {
		t.Fatalf("transcript changed across detach: %q want %q", got, transcript)
	}
	if reopened.Descriptor.Terminal.ByteLength != len(transcript) || reopened.Descriptor.Terminal.Digest != terminalView(transcript).Digest {
		t.Fatalf("terminal byte witness mismatch: %+v", reopened.Descriptor.Terminal)
	}
	if len(reopened.Descriptor.Effects) != 2 || reopened.Descriptor.Effects[0].Verdict != SessionEffectUncertain || reopened.Descriptor.Effects[1].Verdict != SessionEffectKnownNotRun {
		t.Fatalf("effect recovery classification mismatch: %+v", reopened.Descriptor.Effects)
	}
	if reopened.Descriptor.Effects[0].Check != "query payment id" {
		t.Fatalf("uncertain effect omitted checkable decision: %+v", reopened.Descriptor.Effects[0])
	}
	if strings.Contains(reopened.Descriptor.Terminal.Transcript, "send payment") {
		t.Fatal("uncertain effect was replayed into terminal transcript")
	}
}

func postClientJSON(t *testing.T, endpoint string, body any, wantStatus int, dst any) {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(endpoint, "application/json", bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != wantStatus {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("POST %s status=%d want=%d body=%s", endpoint, resp.StatusCode, wantStatus, data)
	}
	if dst != nil {
		if err := json.NewDecoder(resp.Body).Decode(dst); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSessionClientRestoreUsesNewEpochAndNoOldLease(t *testing.T) {
	state := SessionState{TraceID: "restart-a", Run: "RUNNING", Rev: 3}
	old, err := New(Config{EngineID: "mock", Model: "mock", ObserveSession: func(context.Context, string) SessionState { return state }})
	if err != nil {
		t.Fatal(err)
	}
	oldServer := httptest.NewServer(old.Handler())
	var first SessionClientAttachResponse
	postClientJSON(t, oldServer.URL+"/v1/fak/session/restart-a/attach", map[string]any{"client_kind": "terminal"}, http.StatusOK, &first)
	oldServer.Close()

	restarted, err := New(Config{EngineID: "mock", Model: "mock", ObserveSession: func(context.Context, string) SessionState { return state }})
	if err != nil {
		t.Fatal(err)
	}
	if err := restarted.RestoreSessionClientState("restart-a", []byte("restored\r\n"), []SessionEffect{{ID: "effect-1", Verdict: SessionEffectUncertain, Check: "query receipt"}}); err != nil {
		t.Fatal(err)
	}
	newServer := httptest.NewServer(restarted.Handler())
	defer newServer.Close()
	var second SessionClientAttachResponse
	postClientJSON(t, newServer.URL+"/v1/fak/session/restart-a/attach", map[string]any{"client_kind": "browser"}, http.StatusOK, &second)
	if second.Descriptor.ExecutionEpoch == first.Descriptor.ExecutionEpoch {
		t.Fatal("restart reused old writer epoch")
	}
	if second.Descriptor.Terminal.Transcript != "restored\r\n" || second.Descriptor.Effects[0].Verdict != SessionEffectUncertain {
		t.Fatalf("restore mismatch: %+v", second.Descriptor)
	}
	postClientJSON(t, newServer.URL+"/v1/fak/session/restart-a/input", SessionClientActionRequest{AttachmentID: first.AttachmentID, ExecutionEpoch: first.Descriptor.ExecutionEpoch, Text: "stale"}, http.StatusConflict, nil)
}

func TestSessionClientLocalAuthAndWorkspaceFence(t *testing.T) {
	state := SessionState{TraceID: "bound-a", Run: "RUNNING", Rev: 1}
	s, err := New(Config{EngineID: "mock", Model: "mock", ObserveSession: func(context.Context, string) SessionState { return state }})
	if err != nil {
		t.Fatal(err)
	}
	s.ConfigureSessionClientAuth("per-user-secret")
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()
	request := func(token, workspace string) int {
		payload, _ := json.Marshal(SessionClientAttachRequest{ClientKind: "terminal", WorkspaceID: workspace})
		req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/fak/session/bound-a/attach", bytes.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")
		if token != "" {
			req.Header.Set(SessionClientTokenHeader, token)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		return resp.StatusCode
	}
	if got := request("", "workspace-a"); got != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status=%d", got)
	}
	if got := request("wrong", "workspace-a"); got != http.StatusUnauthorized {
		t.Fatalf("wrong token status=%d", got)
	}
	if got := request("per-user-secret", "workspace-a"); got != http.StatusOK {
		t.Fatalf("authenticated status=%d", got)
	}
	if got := request("per-user-secret", "workspace-b"); got != http.StatusConflict {
		t.Fatalf("cross-workspace status=%d", got)
	}
}

func TestSessionClientReportsExactNonRecoverableDependency(t *testing.T) {
	state := SessionState{TraceID: "reboot-a", Run: "PAUSED", Rev: 1}
	s, err := New(Config{EngineID: "mock", Model: "mock", ObserveSession: func(context.Context, string) SessionState { return state }})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RestoreSessionClientStateWithDependency("reboot-a", nil, nil, "adapter identity unavailable"); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()
	var got SessionClientAttachResponse
	postClientJSON(t, ts.URL+"/v1/fak/session/reboot-a/attach", SessionClientAttachRequest{ClientKind: "packaged"}, http.StatusOK, &got)
	if got.Descriptor.RecoveryDependency != "adapter identity unavailable" {
		t.Fatalf("dependency=%q", got.Descriptor.RecoveryDependency)
	}
}

func TestSessionClientKeyboardDecisionAndCloseRoutes(t *testing.T) {
	state := SessionState{TraceID: "keys-a", Run: "RUNNING", Rev: 1}
	s, err := New(Config{EngineID: "mock", Model: "mock", ObserveSession: func(context.Context, string) SessionState { return state }})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()
	var attached SessionClientAttachResponse
	postClientJSON(t, ts.URL+"/v1/fak/session/keys-a/attach", SessionClientAttachRequest{ClientKind: "keyboard"}, http.StatusOK, &attached)
	for _, decision := range []string{"APPROVE", "DENY"} {
		postClientJSON(t, ts.URL+"/v1/fak/session/keys-a/decision", SessionClientDecisionRequest{AttachmentID: attached.AttachmentID, Decision: decision}, http.StatusOK, nil)
	}
	postClientJSON(t, ts.URL+"/v1/fak/session/keys-a/close", SessionClientDetachRequest{AttachmentID: attached.AttachmentID}, http.StatusOK, nil)
	postClientJSON(t, ts.URL+"/v1/fak/session/keys-a/input", SessionClientActionRequest{AttachmentID: attached.AttachmentID, ExecutionEpoch: attached.Descriptor.ExecutionEpoch, Text: "after close"}, http.StatusUnauthorized, nil)
}
