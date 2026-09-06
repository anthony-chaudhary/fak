package harnessweb

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/goalregistry"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

func TestCapturedPageRendersOperatingStatesAndSecondSkin(t *testing.T) {
	response := httptest.NewRecorder()
	request := httptest.NewRequest("GET", "/", nil)
	handler(newStore()).ServeHTTP(response, request)
	body := response.Body.String()
	for _, want := range []string{
		"Harness overview",
		"Web gateway",
		"Agent stats",
		"Goals",
		"Live dashboards",
		"Run agent",
		"aria-live=\"polite\"",
		"approval.requested",
		"Approval run",
		"Failure run",
		"data-skin",
		"body[data-skin=\"minimal\"]",
		"p.text",
		"p.summary",
		"id=\"gateway-startup\"",
		"Gateway startup",
		"Startup phases",
		"Model load",
		"Startup messages",
		"legacy gateway fallback",
		"Structured startup state is unavailable on this older gateway",
		"copy.textContent=message.text",
		"inspect typed startup messages",
		"@media(max-width:600px){.startup-summary{grid-template-columns:1fr}}",
		"const local=data.local_work||{}",
		"worktrees=local.worker_worktrees||{}",
		"cleanup ready, not complete",
		"landed with independent commit witness",
		"target.append(row(identity,detail,item.complete?\"complete\":\"evidence\"))",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("captured browser render missing %q", want)
		}
	}
	if strings.Contains(body, "Local, bounded, yours.") || strings.Contains(body, "separately built product UI") {
		t.Fatal("captured browser render still leads with promotional implementation copy")
	}
	if got := response.Header().Get("Content-Security-Policy"); !strings.Contains(got, "default-src 'self'") {
		t.Fatalf("CSP=%q", got)
	}
}

func TestStatusAndPageDistinguishOfflineProofRunsFromLiveWork(t *testing.T) {
	store := newStore()
	for _, id := range []string{"local-1", "local-2", "local-3", "local-4", "live-5", "imported-run"} {
		if err := store.replace(id, []harnesskit.Envelope{event(id, 1, harnesskit.EventRunCompleted, harnesskit.RunPayload{Status: "completed"})}); err != nil {
			t.Fatal(err)
		}
	}

	ui := httptest.NewServer(handler(store))
	defer ui.Close()
	response, err := ui.Client().Get(ui.URL + "/api/status")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var status struct {
		Agents agentOverview `json:"agents"`
	}
	if err := json.NewDecoder(response.Body).Decode(&status); err != nil {
		t.Fatal(err)
	}
	if status.Agents.TotalRuns != 6 || status.Agents.LiveRuns != 1 || status.Agents.OfflineDemoRuns != 4 {
		t.Fatalf("run provenance counts = total %d, live %d, offline-demo %d", status.Agents.TotalRuns, status.Agents.LiveRuns, status.Agents.OfflineDemoRuns)
	}
	wantSources := map[string]string{"local-1": "offline-demo", "live-5": "live", "imported-run": "legacy-unknown"}
	for _, run := range status.Agents.RecentRuns {
		if want, ok := wantSources[run.RunID]; ok && run.Source != want {
			t.Errorf("source for %s = %q, want %q", run.RunID, run.Source, want)
		}
	}

	pageResponse, err := ui.Client().Get(ui.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	defer pageResponse.Body.Close()
	body, err := io.ReadAll(pageResponse.Body)
	if err != nil {
		t.Fatal(err)
	}
	render := string(body)
	for _, want := range []string{"Live stored runs", "Offline proof runs", "not live activity", "legacy-unknown"} {
		if !strings.Contains(render, want) {
			t.Errorf("render missing %q", want)
		}
	}
	if strings.Contains(render, `"Stored runs"`) {
		t.Error("render still labels the combined total as Stored runs")
	}
}

func TestStatusProjectsAgentStatsGoalsAndLiveDashboardLinks(t *testing.T) {
	goals := goalregistry.Store{Path: filepath.Join(t.TempDir(), "goals.json")}
	if _, err := goals.Create("Make the harness operational", "", goalregistry.Provenance{Actor: "operator", Authority: "user"}, nil); err != nil {
		t.Fatal(err)
	}
	blocked, err := goals.Create("Recover the stalled worker", "", goalregistry.Provenance{Actor: "operator", Authority: "user"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := goals.Transition(blocked.GoalID, goalregistry.Blocked, goalregistry.OutcomeEvidence{
		Class: goalregistry.IndependentWitness, Author: "test", Reference: "fixture",
	}); err != nil {
		t.Fatal(err)
	}

	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/debug/vars" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"sessions": []map[string]any{
				{"trace_id": "agent-main", "run": "running", "turns_left": 12, "last_tool": "Read"},
				{"trace_id": "agent-review", "run": "paused", "turns_left": 4},
			},
			"fleet": map[string]any{"machines": 3, "sessions": 7},
			"startup": map[string]any{
				"status": "ready", "started_at": "2026-08-29T12:00:00Z", "ready_at": "2026-08-29T12:00:04Z",
				"time_to_ready_seconds": 4.25, "unaccounted_seconds": 0.15,
				"phases": []map[string]any{{"name": "listener-bind", "seconds": 0.125, "provenance": "measured", "stage": "gateway-boot"}},
				"model_load": map[string]any{
					"source": "qwen3.8.gguf", "mode": "native", "total_seconds": 3.5, "bytes": 1048576, "tensors": 24, "bottleneck": "upload",
					"load_paths": []map[string]any{{"quant_type": "Q4_K", "class": "resident", "resident_tensors": 22, "resident_bytes": 900000, "dequant_tensors": 2, "dequant_bytes": 148576}},
				},
				"messages": []map[string]any{{"source": "model", "kind": "load", "level": "warn", "text": "weights & cache <pending>"}},
			},
		})
	}))
	defer gateway.Close()

	s := newStore()
	s.create("complete a run")
	s.create("failure: prove the failure total")
	s.create("approval: wait for operator")
	ui := httptest.NewServer(handlerWithSources(s, &liveAdapter{baseURL: gateway.URL, client: gateway.Client()}, goals))
	defer ui.Close()
	response, err := ui.Client().Get(ui.URL + "/api/status")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var got struct {
		Mode       string          `json:"mode"`
		Gateway    gatewayOverview `json:"gateway"`
		Agents     agentOverview   `json:"agents"`
		Goals      goalOverview    `json:"goals"`
		Dashboards []dashboardLink `json:"dashboards"`
	}
	if err := json.NewDecoder(response.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Mode != "live" || !got.Gateway.Reachable || got.Gateway.URL != gateway.URL {
		t.Fatalf("gateway=%+v mode=%q", got.Gateway, got.Mode)
	}
	if got.Gateway.FleetMachines != 3 || got.Gateway.FleetSessions != 7 || len(got.Agents.LiveSessions) != 2 {
		t.Fatalf("live overview gateway=%+v agents=%+v", got.Gateway, got.Agents)
	}
	if got.Gateway.Startup == nil || got.Gateway.Startup.Status != "ready" || len(got.Gateway.Startup.Phases) != 1 || got.Gateway.Startup.ModelLoad == nil || len(got.Gateway.Startup.Messages) != 1 {
		t.Fatalf("typed gateway startup not projected: %+v", got.Gateway.Startup)
	}
	if got.Gateway.Startup.ModelLoad.Mode != "native" || got.Gateway.Startup.ModelLoad.LoadPaths[0].QuantType != "Q4_K" || got.Gateway.Startup.Messages[0].Text != "weights & cache <pending>" {
		t.Fatalf("startup detail lost: %+v", got.Gateway.Startup)
	}
	if got.Agents.TotalRuns != 3 || got.Agents.Completed != 1 || got.Agents.Failed != 1 || got.Agents.AwaitingApproval != 1 {
		t.Fatalf("run stats=%+v", got.Agents)
	}
	if !got.Goals.Readable || got.Goals.Total != 2 || got.Goals.Active != 1 || got.Goals.Blocked != 1 {
		t.Fatalf("goals=%+v", got.Goals)
	}
	if len(got.Dashboards) != 8 || got.Dashboards[0].Label != "Web gateway" || got.Dashboards[0].URL != gateway.URL+"/" {
		t.Fatalf("dashboards=%+v", got.Dashboards)
	}
}

func TestLiveAdapterPreservesOlderGatewayStartupFallback(t *testing.T) {
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/debug/vars" {
			http.NotFound(w, r)
			return
		}
		_, _ = io.WriteString(w, `{"sessions":[],"startup_report":"legacy prose remains at the gateway"}`)
	}))
	defer gateway.Close()
	got, sessions := (&liveAdapter{baseURL: gateway.URL, client: gateway.Client()}).overview(context.Background())
	if !got.Reachable || got.Startup != nil || len(sessions) != 0 {
		t.Fatalf("legacy gateway fallback changed: gateway=%+v sessions=%+v", got, sessions)
	}
}

func TestPublicGatewayURLDropsCredentialsAndQuery(t *testing.T) {
	if got := publicGatewayURL("https://user:secret@example.test:8443/base/?token=private#part"); got != "https://example.test:8443/base" {
		t.Fatalf("public gateway URL=%q", got)
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
	for _, want := range []string{"HARNESS_WEB_SELFCHECK ok", "protocol=fak.harness.run/v1", "normal=8", "resumed=2", "approval=4", "failure=3", "skins=2", "runs=3", "goals=1", "dashboards=8", "html_sha256="} {
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

func TestPageDeepLinksPersistedRun(t *testing.T) {
	if !strings.Contains(page, `query.get("run")`) || !strings.Contains(page, `history.replaceState`) {
		t.Fatal("page does not deep-link persisted run")
	}
}

func TestWebApprovalResolution(t *testing.T) {
	now := time.Now()
	approvalEnv := harnesskit.Envelope{
		Type: harnesskit.EventApprovalRequested,
		Payload: []byte(`{
			"approval_id": "app-web-test-1",
			"tool_name": "Bash",
			"command": "make test-fast",
			"target_path": "/home/user/fak",
			"risk_explanation": "execution of host test suite"
		}`),
	}
	app, err := ParseSessionApproval(approvalEnv)
	if err != nil {
		t.Fatalf("parse approval event: %v", err)
	}

	cardAwaiting := SessionCard{
		ID:                 "sess-awaiting",
		Provider:           "codex",
		Workspace:          "/home/user/fak",
		State:              sessionAwaitingApproval,
		PendingInteraction: "approval requested",
		PendingApproval:    app,
		LastEventAt:        now,
		HasInputLease:      true,
		Capabilities:       map[string]SessionCapability{"cancel": {Enabled: true}},
	}
	cardIdle := SessionCard{
		ID:            "sess-idle",
		Provider:      "codex",
		Workspace:     "/home/user/fak",
		State:         sessionIdle,
		LastEventAt:   now,
		HasInputLease: true,
		Capabilities:  map[string]SessionCapability{"resume": {Enabled: true}},
	}

	source := &fixtureSessionSource{
		cards: []SessionCard{cardAwaiting, cardIdle},
	}
	s := newStore()
	ts := httptest.NewServer(handlerWithSessionSource(s, nil, nil, source))
	defer ts.Close()

	// 1. Verify rendering of approval elements via GET /api/sessions
	res, err := ts.Client().Get(ts.URL + "/api/sessions")
	if err != nil {
		t.Fatal(err)
	}
	var sessResp struct {
		HTML string `json:"html"`
	}
	if err := json.NewDecoder(res.Body).Decode(&sessResp); err != nil {
		t.Fatal(err)
	}
	res.Body.Close()

	for _, want := range []string{
		"session-approval-modal",
		"Action approval required",
		"app-web-test-1",
		"Bash",
		"make test-fast",
		"/home/user/fak",
		"execution of host test suite",
		`data-approval-action="accept"`,
		`data-approval-action="decline"`,
		`data-action="accept"`,
		`data-action="decline"`,
		`<form class="approval-form session-approval-controls" action="/api/sessions/sess-awaiting/approval" method="POST"`,
	} {
		if !strings.Contains(sessResp.HTML, want) {
			t.Fatalf("session markup missing %q:\n%s", want, sessResp.HTML)
		}
	}

	assertJSONError := func(res *http.Response, expectedStatus int, expectedSubstr string) {
		t.Helper()
		defer res.Body.Close()
		if res.StatusCode != expectedStatus {
			t.Fatalf("expected status %d, got %d", expectedStatus, res.StatusCode)
		}
		if ct := res.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
			t.Fatalf("expected Content-Type application/json, got %q", ct)
		}
		var errPayload map[string]string
		if err := json.NewDecoder(res.Body).Decode(&errPayload); err != nil {
			t.Fatalf("failed to decode JSON error: %v", err)
		}
		if expectedSubstr != "" && !strings.Contains(errPayload["error"], expectedSubstr) {
			t.Fatalf("expected error containing %q, got %q", expectedSubstr, errPayload["error"])
		}
	}

	// 2. Validate rejection of malformed JSON
	badJSONRes, err := ts.Client().Post(ts.URL+"/api/sessions/sess-awaiting/approval", "application/json", strings.NewReader("{broken"))
	if err != nil {
		t.Fatal(err)
	}
	assertJSONError(badJSONRes, http.StatusBadRequest, "invalid approval payload: invalid JSON")

	// 3. Validate rejection of invalid resolution token
	badRes, err := ts.Client().Post(ts.URL+"/api/sessions/sess-awaiting/approval", "application/json", strings.NewReader(`{"resolution":"maybe"}`))
	if err != nil {
		t.Fatal(err)
	}
	assertJSONError(badRes, http.StatusBadRequest, "invalid approval resolution")

	// 4. Validate rejection of unknown session ID
	unknownRes, err := ts.Client().Post(ts.URL+"/api/sessions/nonexistent/approval", "application/json", strings.NewReader(`{"resolution":"accept"}`))
	if err != nil {
		t.Fatal(err)
	}
	assertJSONError(unknownRes, http.StatusNotFound, "logical session not found")

	// 5. Validate rejection of session not awaiting approval
	idleRes, err := ts.Client().Post(ts.URL+"/api/sessions/sess-idle/approval", "application/json", strings.NewReader(`{"resolution":"accept"}`))
	if err != nil {
		t.Fatal(err)
	}
	assertJSONError(idleRes, http.StatusConflict, "session is not awaiting approval")

	// 6. Connect SSE client to verify real-time events
	sseReq, err := http.NewRequest(http.MethodGet, ts.URL+"/api/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	sseRes, err := ts.Client().Do(sseReq)
	if err != nil {
		t.Fatal(err)
	}
	defer sseRes.Body.Close()
	if sseRes.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for SSE stream, got %d", sseRes.StatusCode)
	}
	if ct := sseRes.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("expected Content-Type text/event-stream, got %q", ct)
	}

	// 7. Successful accept submission
	acceptBody := `{"resolution":"accept","reason":"operator verified host safety","approval_id":"app-web-test-1"}`
	goodRes, err := ts.Client().Post(ts.URL+"/api/sessions/sess-awaiting/approval", "application/json", strings.NewReader(acceptBody))
	if err != nil {
		t.Fatal(err)
	}
	defer goodRes.Body.Close()
	if goodRes.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(goodRes.Body)
		t.Fatalf("expected 200 for accept submission, got %d: %s", goodRes.StatusCode, body)
	}
	var resData map[string]any
	if err := json.NewDecoder(goodRes.Body).Decode(&resData); err != nil {
		t.Fatal(err)
	}
	if resData["status"] != "accepted" || resData["session_id"] != "sess-awaiting" || resData["resolution"] != "accept" {
		t.Fatalf("unexpected accept response: %+v", resData)
	}

	// Verify dispatch to SessionSource
	if len(source.approvals) != 1 {
		t.Fatalf("expected 1 dispatched approval, got %d", len(source.approvals))
	}
	dispatched := source.approvals[0]
	if dispatched.SessionID != "sess-awaiting" || dispatched.ApprovalID != "app-web-test-1" || dispatched.Resolution != "accept" || dispatched.Reason != "operator verified host safety" {
		t.Fatalf("unexpected dispatched approval: %+v", dispatched)
	}

	// 8. Successful decline submission
	source.cards[0].State = sessionAwaitingApproval
	source.cards[0].PendingApproval = app
	declineBody := `{"resolution":"decline","reason":"too risky"}`
	decRes, err := ts.Client().Post(ts.URL+"/api/sessions/sess-awaiting/approval", "application/json", strings.NewReader(declineBody))
	if err != nil {
		t.Fatal(err)
	}
	decRes.Body.Close()
	if decRes.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for decline submission, got %d", decRes.StatusCode)
	}
	if len(source.approvals) != 2 || source.approvals[1].Resolution != "decline" {
		t.Fatalf("expected 2 dispatched approvals with decline, got %+v", source.approvals)
	}

	// 9. Successful form urlencoded submission
	source.cards[0].State = sessionAwaitingApproval
	source.cards[0].PendingApproval = app
	formBody := "resolution=accept&reason=approved+via+form&approval_id=app-web-test-1"
	formRes, err := ts.Client().Post(ts.URL+"/api/sessions/sess-awaiting/approval", "application/x-www-form-urlencoded", strings.NewReader(formBody))
	if err != nil {
		t.Fatal(err)
	}
	formRes.Body.Close()
	if formRes.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for form submission, got %d", formRes.StatusCode)
	}
	if len(source.approvals) != 3 || source.approvals[2].Resolution != "accept" || source.approvals[2].Reason != "approved via form" {
		t.Fatalf("expected 3 dispatched approvals with form accept, got %+v", source.approvals)
	}

	// 10. Test session-scoped SSE streaming hook: GET /v1/fak/sessions/{id}/events
	scopedSSEReq, err := http.NewRequest(http.MethodGet, ts.URL+"/v1/fak/sessions/sess-awaiting/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	scopedSSERes, err := ts.Client().Do(scopedSSEReq)
	if err != nil {
		t.Fatal(err)
	}
	defer scopedSSERes.Body.Close()
	if scopedSSERes.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for scoped SSE stream, got %d", scopedSSERes.StatusCode)
	}

	// 11. Test event streaming ingestion hook: POST /v1/fak/sessions/{id}/events
	eventHookPayload := `{"type":"approval.requested","approval_id":"app-hook-1","tool_name":"Bash","command":"make ci","target_path":"/repo","risk_explanation":"high risk"}`
	hookRes, err := ts.Client().Post(ts.URL+"/v1/fak/sessions/sess-awaiting/events", "application/json", strings.NewReader(eventHookPayload))
	if err != nil {
		t.Fatal(err)
	}
	hookRes.Body.Close()
	if hookRes.StatusCode != http.StatusAccepted {
		t.Fatalf("expected 202 for event hook, got %d", hookRes.StatusCode)
	}

	// 12. Test store-backed session approval resolution when source is nil
	storeOnly := newStore()
	runID := storeOnly.create("approval: inspect workspace")
	storeTS := httptest.NewServer(handler(storeOnly))
	defer storeTS.Close()

	storeRes, err := storeTS.Client().Post(storeTS.URL+"/api/sessions/"+runID+"/approval", "application/json", strings.NewReader(`{"resolution":"accept","approval_id":"approval-1"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer storeRes.Body.Close()
	if storeRes.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(storeRes.Body)
		t.Fatalf("expected 200 for store-backed session approval, got %d: %s", storeRes.StatusCode, body)
	}

	// 13. Test direct 3-argument coordinator resolution
	coord := &mockCoordinator{}
	coordHandler := HandleSessionApproval(coord, nil)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/sessions/sess-coord/approval", strings.NewReader(`{"resolution":"accept","reason":"coordinator authorized"}`))
	req.SetPathValue("id", "sess-coord")
	coordHandler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for coordinator resolution, got %d: %s", rec.Code, rec.Body.String())
	}
	if coord.resolvedID != "sess-coord" || coord.resolution != "accept" || coord.reason != "coordinator authorized" {
		t.Fatalf("coordinator unexpected values: id=%q, res=%q, reason=%q", coord.resolvedID, coord.resolution, coord.reason)
	}
}

type nonResolverSource struct {
	cards []SessionCard
}

func (n *nonResolverSource) Sessions(context.Context) ([]SessionCard, error) {
	return n.cards, nil
}

func (n *nonResolverSource) Control(context.Context, SessionControlRequest) error {
	return nil
}

func TestSessionApprovalNonResolverSourceFallbackAndError(t *testing.T) {
	s := newStore()
	runID := s.create("approval: inspect workspace")
	cardWithStore := SessionCard{
		ID:                 runID,
		Provider:           "codex",
		Workspace:          "/test/ws",
		State:              sessionAwaitingApproval,
		PendingInteraction: "approval requested",
		PendingApproval: &SessionApproval{
			ApprovalID: "approval-1",
		},
		HasInputLease: true,
	}
	cardWithoutStore := SessionCard{
		ID:                 "sess-no-store",
		Provider:           "codex",
		Workspace:          "/test/ws",
		State:              sessionAwaitingApproval,
		PendingInteraction: "approval requested",
		PendingApproval: &SessionApproval{
			ApprovalID: "approval-orphan",
		},
		HasInputLease: true,
	}

	source := &nonResolverSource{
		cards: []SessionCard{cardWithStore, cardWithoutStore},
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/sessions/{id}/approval", HandleSessionApproval(source, s))
	mux.HandleFunc("GET /api/sessions/events", handleSessionSSE(false))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// 1. Invoking approval for sess-no-store (no store fallback) must return HTTP 501 Not Implemented,
	// never falsely reporting resolved: true.
	noStoreRes, err := ts.Client().Post(
		ts.URL+"/api/sessions/sess-no-store/approval",
		"application/json",
		strings.NewReader(`{"decision":"approve","approval_id":"approval-orphan"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer noStoreRes.Body.Close()
	if noStoreRes.StatusCode != http.StatusNotImplemented {
		body, _ := io.ReadAll(noStoreRes.Body)
		t.Fatalf("expected 501 Not Implemented for non-resolver session source without store fallback, got %d: %s", noStoreRes.StatusCode, body)
	}
	var noStoreBody map[string]any
	if err := json.NewDecoder(noStoreRes.Body).Decode(&noStoreBody); err != nil {
		t.Fatal(err)
	}
	if noStoreBody["resolved"] == true {
		t.Fatalf("endpoint falsely reported resolved=true when resolver is unsupported and no store fallback exists")
	}

	// 2. Invoking approval for runID with wrong approval ID triggers store fallback and returns 409 Conflict.
	badStoreRes, err := ts.Client().Post(
		ts.URL+"/api/sessions/"+runID+"/approval",
		"application/json",
		strings.NewReader(`{"resolution":"accept","approval_id":"wrong-approval"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer badStoreRes.Body.Close()
	if badStoreRes.StatusCode != http.StatusConflict {
		body, _ := io.ReadAll(badStoreRes.Body)
		t.Fatalf("expected 409 Conflict when store fallback resolution fails, got %d: %s", badStoreRes.StatusCode, body)
	}
	var badStoreBody map[string]any
	if err := json.NewDecoder(badStoreRes.Body).Decode(&badStoreBody); err != nil {
		t.Fatal(err)
	}
	if badStoreBody["resolved"] == true {
		t.Fatalf("endpoint falsely reported resolved=true on failed store resolution")
	}

	// Subscribe an SSE client to verify that store fallback triggers a card broadcast
	resetSessionHubForTest()
	defer resetSessionHubForTest()

	sseReq, err := http.NewRequest(http.MethodGet, ts.URL+"/api/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	sseRes, err := ts.Client().Do(sseReq)
	if err != nil {
		t.Fatal(err)
	}
	defer sseRes.Body.Close()
	sseReader := bufio.NewReader(sseRes.Body)
	evConn, _ := readSSEEvent(t, sseReader)
	if evConn != "connected" {
		t.Fatalf("expected connected event on SSE, got %q", evConn)
	}

	// 3. Invoking approval for runID with valid approval ID triggers store fallback and resolves approval in store.
	goodRes, err := ts.Client().Post(
		ts.URL+"/api/sessions/"+runID+"/approval",
		"application/json",
		strings.NewReader(`{"resolution":"accept","approval_id":"approval-1"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer goodRes.Body.Close()
	if goodRes.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(goodRes.Body)
		t.Fatalf("expected 200 OK for successful store fallback, got %d: %s", goodRes.StatusCode, body)
	}
	var goodBody map[string]any
	if err := json.NewDecoder(goodRes.Body).Decode(&goodBody); err != nil {
		t.Fatal(err)
	}
	if goodBody["status"] != "accepted" || goodBody["resolved"] != true || goodBody["session_id"] != runID {
		t.Fatalf("unexpected response body for store fallback: %+v", goodBody)
	}

	// Verify that resolveViaStore triggered a card broadcast (session_update event)
	evUpdate, updateData := readSSEEvent(t, sseReader)
	if evUpdate != "session_update" {
		t.Fatalf("expected session_update event from card broadcast on store fallback, got %q", evUpdate)
	}
	if !strings.Contains(updateData, runID) {
		t.Fatalf("expected session_update payload to contain runID %q: %s", runID, updateData)
	}

	// Verify that approval_resolved event follows the card broadcast
	evResolved, resolvedData := readSSEEvent(t, sseReader)
	if evResolved != "approval_resolved" {
		t.Fatalf("expected approval_resolved event on store fallback, got %q", evResolved)
	}
	if !strings.Contains(resolvedData, runID) || !strings.Contains(resolvedData, "accept") {
		t.Fatalf("unexpected approval_resolved payload: %s", resolvedData)
	}

	// Verify that the run in store was genuinely resolved
	s.mu.RLock()
	st := s.runs[runID]
	s.mu.RUnlock()
	if st == nil || !st.resolved {
		t.Fatalf("expected store run %s to be resolved, got %+v", runID, st)
	}
}

type storeAwareSessionSource struct {
	s     *store
	runID string
}

func (m *storeAwareSessionSource) Sessions(context.Context) ([]SessionCard, error) {
	state := sessionAwaitingApproval
	var app *SessionApproval
	interaction := "approval requested"
	if m.s != nil {
		m.s.mu.RLock()
		st := m.s.runs[m.runID]
		if st != nil && st.resolved {
			state = sessionWorking
			interaction = ""
		} else {
			app = &SessionApproval{
				ApprovalID: "approval-store-1",
				ToolName:   "Bash",
				Command:    "make test-fast",
			}
		}
		m.s.mu.RUnlock()
	}
	return []SessionCard{
		{
			ID:                 m.runID,
			Provider:           "codex",
			Workspace:          "/test/ws",
			State:              state,
			PendingInteraction: interaction,
			PendingApproval:    app,
			LastEventAt:        time.Now(),
			HasInputLease:      true,
			Capabilities:       map[string]SessionCapability{"cancel": {Enabled: true}},
		},
	}, nil
}

func (m *storeAwareSessionSource) Control(context.Context, SessionControlRequest) error {
	return nil
}

func TestSessionApprovalStoreFallbackBroadcastsUpdatedCards(t *testing.T) {
	resetSessionHubForTest()
	defer resetSessionHubForTest()

	s := newStore()
	runID := s.create("approval: inspect workspace")
	source := &storeAwareSessionSource{s: s, runID: runID}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/sessions/events", handleSessionSSE(false))
	mux.HandleFunc("POST /api/sessions/{id}/approval", HandleSessionApproval(source, s))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sseReq, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/api/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	sseRes, err := ts.Client().Do(sseReq)
	if err != nil {
		t.Fatal(err)
	}
	defer sseRes.Body.Close()

	reader := bufio.NewReader(sseRes.Body)
	evConn, _ := readSSEEvent(t, reader)
	if evConn != "connected" {
		t.Fatalf("expected connected event, got %q", evConn)
	}

	// Post approval resolution targeting the store run (store fallback resolution path)
	postBody := `{"resolution":"accept","approval_id":"approval-1"}`
	res, err := ts.Client().Post(ts.URL+"/api/sessions/"+runID+"/approval", "application/json", strings.NewReader(postBody))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(res.Body)
		t.Fatalf("expected 200 OK, got %d: %s", res.StatusCode, body)
	}

	// 1. Immediately assert receipt of session_update containing updated cards
	evType, evData := readSSEEvent(t, reader)
	if evType != "session_update" {
		t.Fatalf("expected immediate session_update event on store fallback, got %q", evType)
	}

	var updatePayload struct {
		Sessions []SessionCard `json:"sessions"`
		HTML     string        `json:"html"`
	}
	if err := json.Unmarshal([]byte(evData), &updatePayload); err != nil {
		t.Fatalf("failed to decode session_update payload: %v", err)
	}

	if len(updatePayload.Sessions) != 1 {
		t.Fatalf("expected 1 session card in update, got %d", len(updatePayload.Sessions))
	}
	updatedCard := updatePayload.Sessions[0]
	if updatedCard.ID != runID {
		t.Fatalf("expected card ID %q, got %q", runID, updatedCard.ID)
	}
	if updatedCard.State != sessionWorking {
		t.Fatalf("expected card state %q, got %q", sessionWorking, updatedCard.State)
	}
	if updatedCard.PendingApproval != nil {
		t.Fatalf("expected nil PendingApproval after resolution, got %+v", updatedCard.PendingApproval)
	}
	if strings.Contains(updatePayload.HTML, "Action approval required") {
		t.Fatalf("broadcast HTML unexpectedly contains approval modal:\n%s", updatePayload.HTML)
	}

	// 2. Assert receipt of approval_resolved event
	evResolvedType, evResolvedData := readSSEEvent(t, reader)
	if evResolvedType != "approval_resolved" {
		t.Fatalf("expected approval_resolved event, got %q", evResolvedType)
	}
	if !strings.Contains(evResolvedData, runID) || !strings.Contains(evResolvedData, "accept") {
		t.Fatalf("unexpected approval_resolved payload: %s", evResolvedData)
	}

	// 3. Confirm store run resolution
	s.mu.RLock()
	st := s.runs[runID]
	s.mu.RUnlock()
	if st == nil || !st.resolved {
		t.Fatalf("store run %q was not marked resolved", runID)
	}
}

func TestSessionApprovalStoreFallbackNilSourceHubBroadcast(t *testing.T) {
	resetSessionHubForTest()
	defer resetSessionHubForTest()

	s := newStore()
	runID := s.create("approval: inspect workspace")

	initialCard := SessionCard{
		ID:                 runID,
		Provider:           "codex",
		Workspace:          "/test/ws",
		State:              sessionAwaitingApproval,
		PendingInteraction: "approval requested",
		PendingApproval: &SessionApproval{
			ApprovalID: "approval-1",
			ToolName:   "Bash",
			Command:    "make test-fast",
		},
		LastEventAt:   time.Now(),
		HasInputLease: true,
		Capabilities:  map[string]SessionCapability{"cancel": {Enabled: true}},
	}
	normInitial, _ := normalizeSessionCards([]SessionCard{initialCard})
	markupInitial, _ := renderSessionCardsHTML(normInitial, time.Now(), false)

	defaultSessionHub.mu.Lock()
	defaultSessionHub.lastCards = cloneSessionCards(normInitial)
	defaultSessionHub.lastHTML = markupInitial
	defaultSessionHub.mu.Unlock()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/sessions/events", handleSessionSSE(false))
	mux.HandleFunc("POST /api/sessions/{id}/approval", HandleSessionApproval(nil, s))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sseReq, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/api/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	sseRes, err := ts.Client().Do(sseReq)
	if err != nil {
		t.Fatal(err)
	}
	defer sseRes.Body.Close()

	reader := bufio.NewReader(sseRes.Body)
	evConn, _ := readSSEEvent(t, reader)
	if evConn != "connected" {
		t.Fatalf("expected connected event, got %q", evConn)
	}
	evInit, _ := readSSEEvent(t, reader)
	if evInit != "session_update" {
		t.Fatalf("expected initial session_update, got %q", evInit)
	}

	postBody := `{"resolution":"accept","approval_id":"approval-1"}`
	res, err := ts.Client().Post(ts.URL+"/api/sessions/"+runID+"/approval", "application/json", strings.NewReader(postBody))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(res.Body)
		t.Fatalf("expected 200 OK, got %d: %s", res.StatusCode, body)
	}

	// 1. Immediately assert receipt of session_update containing updated cards
	evType, evData := readSSEEvent(t, reader)
	if evType != "session_update" {
		t.Fatalf("expected immediate session_update event on store fallback with nil source, got %q", evType)
	}

	var updatePayload struct {
		Sessions []SessionCard `json:"sessions"`
		HTML     string        `json:"html"`
	}
	if err := json.Unmarshal([]byte(evData), &updatePayload); err != nil {
		t.Fatalf("failed to decode session_update payload: %v", err)
	}

	if len(updatePayload.Sessions) != 1 {
		t.Fatalf("expected 1 session card in update, got %d", len(updatePayload.Sessions))
	}
	updatedCard := updatePayload.Sessions[0]
	if updatedCard.ID != runID {
		t.Fatalf("expected card ID %q, got %q", runID, updatedCard.ID)
	}
	if updatedCard.State != sessionWorking {
		t.Fatalf("expected card state %q, got %q", sessionWorking, updatedCard.State)
	}
	if updatedCard.PendingApproval != nil {
		t.Fatalf("expected nil PendingApproval after resolution, got %+v", updatedCard.PendingApproval)
	}
	if strings.Contains(updatePayload.HTML, "Action approval required") {
		t.Fatalf("broadcast HTML unexpectedly contains approval modal:\n%s", updatePayload.HTML)
	}

	// 2. Assert receipt of approval_resolved event
	evResolvedType, evResolvedData := readSSEEvent(t, reader)
	if evResolvedType != "approval_resolved" {
		t.Fatalf("expected approval_resolved event, got %q", evResolvedType)
	}
	if !strings.Contains(evResolvedData, runID) || !strings.Contains(evResolvedData, "accept") {
		t.Fatalf("unexpected approval_resolved payload: %s", evResolvedData)
	}

	// 3. Confirm store run resolution
	s.mu.RLock()
	st := s.runs[runID]
	s.mu.RUnlock()
	if st == nil || !st.resolved {
		t.Fatalf("store run %q was not marked resolved", runID)
	}
}

type mockCoordinator struct {
	sessionsCalled bool
	resolvedID     string
	resolution     string
	reason         string
}

func (m *mockCoordinator) ResolveApproval(sessionID string, resolution string, reason string) error {
	m.resolvedID = sessionID
	m.resolution = resolution
	m.reason = reason
	return nil
}

type erroringSessionSource struct {
	sessionsErr error
	controlErr  error
	cards       []SessionCard
}

func (m *erroringSessionSource) Sessions(context.Context) ([]SessionCard, error) {
	if m.sessionsErr != nil {
		return nil, m.sessionsErr
	}
	return m.cards, nil
}

func (m *erroringSessionSource) Control(context.Context, SessionControlRequest) error {
	return m.controlErr
}

func (m *erroringSessionSource) ResolveApproval(context.Context, SessionApprovalRequest) error {
	return nil
}

func TestSessionApprovalAndControlJSONErrorPayloads(t *testing.T) {
	now := time.Now()
	workingCard := SessionCard{
		ID:            "sess-active",
		Provider:      "codex",
		State:         sessionWorking,
		LastEventAt:   now,
		HasInputLease: true,
		Capabilities:  map[string]SessionCapability{"interrupt": {Enabled: true}},
	}
	idleCard := SessionCard{
		ID:            "sess-idle",
		Provider:      "codex",
		State:         sessionIdle,
		LastEventAt:   now,
		HasInputLease: true,
		Capabilities:  map[string]SessionCapability{"interrupt": {Enabled: false, UnavailableReason: "session is idle"}},
	}

	src := &erroringSessionSource{
		cards: []SessionCard{workingCard, idleCard},
	}
	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, src))
	defer ts.Close()

	nilTS := httptest.NewServer(handlerWithSessionSource(nil, nil, nil, nil))
	defer nilTS.Close()

	failingSrc := &erroringSessionSource{
		sessionsErr: fmt.Errorf("session store backend failure"),
	}
	failingTS := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, failingSrc))
	defer failingTS.Close()

	controlErrSrc := &erroringSessionSource{
		cards:      []SessionCard{workingCard},
		controlErr: fmt.Errorf("rpc dispatch failure"),
	}
	controlErrTS := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, controlErrSrc))
	defer controlErrTS.Close()

	assertJSONErrorResponse := func(res *http.Response, wantStatus int, wantErrSubstr string) {
		t.Helper()
		defer res.Body.Close()
		if res.StatusCode != wantStatus {
			t.Fatalf("expected HTTP status %d, got %d", wantStatus, res.StatusCode)
		}
		ct := res.Header.Get("Content-Type")
		if !strings.HasPrefix(ct, "application/json") {
			t.Fatalf("expected Content-Type application/json, got %q", ct)
		}
		var payload struct {
			Error string `json:"error"`
		}
		if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode JSON response: %v", err)
		}
		if payload.Error == "" {
			t.Fatalf("expected non-empty error field in JSON response, got %+v", payload)
		}
		if wantErrSubstr != "" && !strings.Contains(payload.Error, wantErrSubstr) {
			t.Fatalf("expected error containing %q, got %q", wantErrSubstr, payload.Error)
		}
	}

	t.Run("approval invalid session id returns JSON 400", func(t *testing.T) {
		res, err := ts.Client().Post(ts.URL+"/api/sessions/%20/approval", "application/json", strings.NewReader(`{"resolution":"accept"}`))
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusBadRequest, "invalid session id")
	})

	t.Run("approval malformed form payload returns JSON 400", func(t *testing.T) {
		res, err := ts.Client().Post(ts.URL+"/api/sessions/sess-active/approval", "application/x-www-form-urlencoded", strings.NewReader("resolution=%zz"))
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusBadRequest, "invalid form payload")
	})

	t.Run("approval malformed JSON returns JSON 400", func(t *testing.T) {
		res, err := ts.Client().Post(ts.URL+"/api/sessions/sess-active/approval", "application/json", strings.NewReader("{broken-json"))
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusBadRequest, "invalid approval payload: invalid JSON")
	})

	t.Run("approval invalid resolution returns JSON 400", func(t *testing.T) {
		res, err := ts.Client().Post(ts.URL+"/api/sessions/sess-active/approval", "application/json", strings.NewReader(`{"resolution":"maybe"}`))
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusBadRequest, `invalid approval resolution: must be "accept" or "decline"`)
	})

	t.Run("approval logical session not found returns JSON 404", func(t *testing.T) {
		res, err := ts.Client().Post(ts.URL+"/api/sessions/nonexistent/approval", "application/json", strings.NewReader(`{"resolution":"accept"}`))
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusNotFound, "logical session not found")
	})

	t.Run("approval session not awaiting returns JSON 409", func(t *testing.T) {
		res, err := ts.Client().Post(ts.URL+"/api/sessions/sess-active/approval", "application/json", strings.NewReader(`{"resolution":"accept"}`))
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusConflict, "session is not awaiting approval")
	})

	t.Run("approval sessions listing error returns JSON 503", func(t *testing.T) {
		res, err := failingTS.Client().Post(failingTS.URL+"/api/sessions/sess-active/approval", "application/json", strings.NewReader(`{"resolution":"accept"}`))
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusServiceUnavailable, "session store backend failure")
	})

	t.Run("approval disconnected authority returns JSON 503", func(t *testing.T) {
		res, err := nilTS.Client().Post(nilTS.URL+"/api/sessions/sess-active/approval", "application/json", strings.NewReader(`{"resolution":"accept"}`))
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusServiceUnavailable, "session authority is not connected")
	})

	t.Run("control invalid session id returns JSON 400", func(t *testing.T) {
		res, err := ts.Client().Post(ts.URL+"/api/sessions/%20/controls/interrupt", "application/json", nil)
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusBadRequest, "invalid session id")
	})

	t.Run("control unknown action returns JSON 404", func(t *testing.T) {
		res, err := ts.Client().Post(ts.URL+"/api/sessions/sess-active/controls/unknown_action", "application/json", nil)
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusNotFound, "unknown session control")
	})

	t.Run("control logical session not found returns JSON 404", func(t *testing.T) {
		res, err := ts.Client().Post(ts.URL+"/api/sessions/nonexistent/controls/interrupt", "application/json", nil)
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusNotFound, "logical session not found")
	})

	t.Run("control disabled capability returns JSON 409", func(t *testing.T) {
		res, err := ts.Client().Post(ts.URL+"/api/sessions/sess-idle/controls/interrupt", "application/json", nil)
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusConflict, "session is idle")
	})

	t.Run("control execution failure returns JSON 409", func(t *testing.T) {
		res, err := controlErrTS.Client().Post(controlErrTS.URL+"/api/sessions/sess-active/controls/interrupt", "application/json", nil)
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusConflict, "rpc dispatch failure")
	})

	t.Run("control sessions listing error returns JSON 503", func(t *testing.T) {
		res, err := failingTS.Client().Post(failingTS.URL+"/api/sessions/sess-active/controls/interrupt", "application/json", nil)
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusServiceUnavailable, "session store backend failure")
	})

	t.Run("control disconnected authority returns JSON 503", func(t *testing.T) {
		res, err := nilTS.Client().Post(nilTS.URL+"/api/sessions/sess-active/controls/interrupt", "application/json", nil)
		if err != nil {
			t.Fatal(err)
		}
		assertJSONErrorResponse(res, http.StatusServiceUnavailable, "session authority is not connected")
	})
}
