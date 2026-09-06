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
	source := &fixtureSessionSource{
		cards: []SessionCard{
			{
				ID:                 "sess-appr-1",
				Provider:           "codex",
				Workspace:          `C:\work\fak`,
				ThreadCoordinate:   "thread-coord-123456",
				ExecutionEpoch:     1,
				State:              sessionAwaitingApproval,
				PendingInteraction: "workspace file edit requires approval",
				LastEventAt:        now,
				HasInputLease:      true,
				Capabilities:       map[string]SessionCapability{"cancel": {Enabled: true}},
				Approval: &SessionApprovalDetails{
					ApprovalID: "appr-001",
					ToolName:   "write_file",
					TargetPath: "internal/harnessweb/web.go",
					RiskReason: "modifies core server routing",
				},
			},
			{
				ID:                 "sess-appr-2",
				Provider:           "codex",
				Workspace:          `C:\work\fak`,
				ThreadCoordinate:   "thread-coord-654321",
				ExecutionEpoch:     1,
				State:              sessionAwaitingApproval,
				PendingInteraction: "command execution requires approval",
				LastEventAt:        now,
				HasInputLease:      true,
				Capabilities:       map[string]SessionCapability{"cancel": {Enabled: true}},
				Approval: &SessionApprovalDetails{
					ApprovalID: "appr-002",
					ToolName:   "execute_bash",
					TargetPath: "scripts/deploy.sh",
					RiskReason: "runs deployment script",
				},
			},
		},
	}

	s := newStore()
	ts := httptest.NewServer(handlerWithSessionSource(s, nil, nil, source))
	defer ts.Close()
	client := ts.Client()

	// 1. Verify end-to-end receipt of approval requests and rendering of approval elements in session card HTML.
	res, err := client.Get(ts.URL + "/api/sessions")
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("GET /api/sessions status = %d", res.StatusCode)
	}
	var sessList struct {
		Sessions []SessionCard `json:"sessions"`
		HTML     string        `json:"html"`
	}
	if err := json.NewDecoder(res.Body).Decode(&sessList); err != nil {
		t.Fatal(err)
	}
	if len(sessList.Sessions) != 2 {
		t.Fatalf("got %d sessions, want 2", len(sessList.Sessions))
	}
	for _, want := range []string{
		"sess-appr-1",
		"write_file",
		"internal/harnessweb/web.go",
		"modifies core server routing",
		"Accept",
		"Decline",
		`data-approval-action="accept"`,
		`data-approval-action="decline"`,
		`data-session-id="sess-appr-1"`,
		"approval-tool",
		"approval-target-path",
		"approval-risk-reason",
	} {
		if !strings.Contains(sessList.HTML, want) {
			t.Errorf("session card HTML missing %q:\n%s", want, sessList.HTML)
		}
	}

	// 2. Connect to SSE stream and verify initial event delivery.
	sseCtx, cancelSSE := context.WithCancel(context.Background())
	defer cancelSSE()
	sseReq, err := http.NewRequestWithContext(sseCtx, http.MethodGet, ts.URL+"/api/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	sseResp, err := client.Do(sseReq)
	if err != nil {
		t.Fatal(err)
	}
	defer sseResp.Body.Close()
	if ct := sseResp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("SSE Content-Type = %q, want text/event-stream", ct)
	}

	sseScanner := bufio.NewScanner(sseResp.Body)
	var initialSSE strings.Builder
	for sseScanner.Scan() {
		line := sseScanner.Text()
		if line == "" {
			break
		}
		initialSSE.WriteString(line + "\n")
	}
	if !strings.Contains(initialSSE.String(), "event: session_cards") || !strings.Contains(initialSSE.String(), "sess-appr-1") {
		t.Fatalf("initial SSE frame missing session_cards: %s", initialSSE.String())
	}

	// 3. Test validation for POST /api/sessions/{id}/approval.
	// 3a. Invalid JSON payload.
	badJSONResp, err := client.Post(ts.URL+"/api/sessions/sess-appr-1/approval", "application/json", strings.NewReader(`{invalid`))
	if err != nil {
		t.Fatal(err)
	}
	badJSONResp.Body.Close()
	if badJSONResp.StatusCode != http.StatusBadRequest {
		t.Errorf("invalid json status = %d, want 400", badJSONResp.StatusCode)
	}

	// 3b. Invalid resolution value.
	badResResp, err := client.Post(ts.URL+"/api/sessions/sess-appr-1/approval", "application/json", strings.NewReader(`{"resolution":"maybe"}`))
	if err != nil {
		t.Fatal(err)
	}
	badResResp.Body.Close()
	if badResResp.StatusCode != http.StatusBadRequest {
		t.Errorf("invalid resolution status = %d, want 400", badResResp.StatusCode)
	}

	// 3c. Unknown session id.
	notFoundResp, err := client.Post(ts.URL+"/api/sessions/unknown-session/approval", "application/json", strings.NewReader(`{"resolution":"accept"}`))
	if err != nil {
		t.Fatal(err)
	}
	notFoundResp.Body.Close()
	if notFoundResp.StatusCode != http.StatusNotFound {
		t.Errorf("unknown session status = %d, want 404", notFoundResp.StatusCode)
	}

	// 4. Successful handling of POST /api/sessions/{id}/approval with resolution: "accept".
	acceptBody := strings.NewReader(`{"resolution":"accept","reason":"verified code change safe"}`)
	acceptResp, err := client.Post(ts.URL+"/api/sessions/sess-appr-1/approval", "application/json", acceptBody)
	if err != nil {
		t.Fatal(err)
	}
	defer acceptResp.Body.Close()
	if acceptResp.StatusCode != http.StatusOK {
		t.Fatalf("accept status = %d, want 200", acceptResp.StatusCode)
	}
	var acceptResult struct {
		Status     string `json:"status"`
		SessionID  string `json:"session_id"`
		Resolution string `json:"resolution"`
		Reason     string `json:"reason"`
		Resolved   bool   `json:"resolved"`
	}
	if err := json.NewDecoder(acceptResp.Body).Decode(&acceptResult); err != nil {
		t.Fatal(err)
	}
	if acceptResult.Status != "accepted" || acceptResult.SessionID != "sess-appr-1" || acceptResult.Resolution != "accept" || !acceptResult.Resolved {
		t.Fatalf("acceptResult = %+v", acceptResult)
	}

	// Verify session authority state updated.
	if len(source.approvals) != 1 {
		t.Fatalf("source approvals count = %d, want 1", len(source.approvals))
	}
	if source.approvals[0].SessionID != "sess-appr-1" || source.approvals[0].Resolution != "accept" || source.approvals[0].Reason != "verified code change safe" {
		t.Fatalf("recorded approval = %+v", source.approvals[0])
	}
	if source.cards[0].State != sessionWorking {
		t.Fatalf("card state = %q, want %q", source.cards[0].State, sessionWorking)
	}

	// 5. Verify live SSE event arrived with updated session state.
	var updateSSE strings.Builder
	for sseScanner.Scan() {
		line := sseScanner.Text()
		if line == "" {
			if updateSSE.Len() > 0 {
				break
			}
			continue
		}
		updateSSE.WriteString(line + "\n")
	}
	if !strings.Contains(updateSSE.String(), "event: session_cards") || !strings.Contains(updateSSE.String(), "working") {
		t.Fatalf("update SSE frame missing updated session card: %s", updateSSE.String())
	}

	// 6. Successful handling of POST /api/sessions/{id}/approval with resolution: "decline".
	declineBody := strings.NewReader(`{"resolution":"decline","reason":"risky script rejected"}`)
	declineResp, err := client.Post(ts.URL+"/api/sessions/sess-appr-2/approval", "application/json", declineBody)
	if err != nil {
		t.Fatal(err)
	}
	defer declineResp.Body.Close()
	if declineResp.StatusCode != http.StatusOK {
		t.Fatalf("decline status = %d, want 200", declineResp.StatusCode)
	}
	if len(source.approvals) != 2 || source.approvals[1].SessionID != "sess-appr-2" || source.approvals[1].Resolution != "decline" {
		t.Fatalf("recorded decline approval = %+v", source.approvals[1])
	}
	if source.cards[1].State != sessionCancelled {
		t.Fatalf("card 2 state = %q, want %q", source.cards[1].State, sessionCancelled)
	}

	// 7. Test approval resolution on store run (fallback to store when session is in store).
	storeRunID := s.create("approval: inspect workspace")
	storeApprResp, err := client.Post(ts.URL+"/api/sessions/"+storeRunID+"/approval", "application/json", strings.NewReader(`{"resolution":"accept","reason":"store operator approved"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer storeApprResp.Body.Close()
	if storeApprResp.StatusCode != http.StatusOK {
		t.Fatalf("store approval status = %d, want 200", storeApprResp.StatusCode)
	}
	var storeApprResult struct {
		Status   string `json:"status"`
		Resolved bool   `json:"resolved"`
	}
	if err := json.NewDecoder(storeApprResp.Body).Decode(&storeApprResult); err != nil {
		t.Fatal(err)
	}
	if storeApprResult.Status != "accepted" || !storeApprResult.Resolved {
		t.Fatalf("storeApprResult = %+v", storeApprResult)
	}
}
