package harnessweb

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

type fixtureSessionSource struct {
	mu        sync.Mutex
	err       error
	cards     []SessionCard
	controls  []SessionControlRequest
	approvals []SessionApprovalRequest
}

func (s *fixtureSessionSource) Sessions(context.Context) ([]SessionCard, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return nil, s.err
	}
	return append([]SessionCard(nil), s.cards...), nil
}

func (s *fixtureSessionSource) Control(_ context.Context, request SessionControlRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.controls = append(s.controls, request)
	return nil
}

func (s *fixtureSessionSource) ResolveApproval(_ context.Context, request SessionApprovalRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.approvals = append(s.approvals, request)
	for i := range s.cards {
		if s.cards[i].ID == request.SessionID {
			s.cards[i].State = sessionWorking
			s.cards[i].PendingInteraction = ""
			s.cards[i].PendingApproval = nil
		}
	}
	return nil
}

func (s *fixtureSessionSource) ApplyApproval(sessionID string, app *SessionApproval) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.cards {
		if s.cards[i].ID == sessionID {
			s.cards[i].ApplyApproval(app)
		}
	}
}

func allSessionFixtures(now time.Time) []SessionCard {
	states := []SessionState{sessionWorking, sessionAwaitingApproval, sessionAwaitingInput, sessionIdle, sessionDisconnected, sessionCancelled, sessionFailed}
	cards := make([]SessionCard, 0, len(states))
	for i, state := range states {
		cards = append(cards, SessionCard{
			ID: "logical-" + string(state), Provider: "codex", Workspace: `C:\work\safe`,
			ThreadCoordinate: "thread-sensitive-coordinate-123456", ExecutionEpoch: uint64(i + 1),
			State: state, LastEventAt: now.Add(-time.Duration(i+1) * time.Minute), Model: "gpt-5.6-codex",
			Usage: &SessionUsage{InputTokens: int64(i + 10), OutputTokens: int64(i + 20)}, HasInputLease: true,
			Capabilities: map[string]SessionCapability{"open": {Enabled: true}, "resume": {Enabled: state == sessionIdle, UnavailableReason: "state does not permit resume"}, "interrupt": {Enabled: state == sessionWorking, UnavailableReason: "not running"}, "cancel": {Enabled: state == sessionAwaitingApproval, UnavailableReason: "not cancellable"}, "archive": {Enabled: state == sessionCancelled || state == sessionFailed, UnavailableReason: "terminal state required"}},
		})
	}
	cards[1].PendingInteraction = "approval requested"
	cards[2].PendingInteraction = "user input requested"
	return cards
}

func TestSessionCardsCapturedDesktopNoColorAndNarrowWitness(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	markup, err := renderSessionCardsHTML(allSessionFixtures(now), now, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"working", "awaiting approval", "awaiting input", "idle", "disconnected", "cancelled", "failed", "approval requested", "user input requested", "state-text", `role="listitem"`, `tabindex="0"`, `tabindex="-1"`, `aria-labelledby="session-title-0"`, `aria-describedby="session-detail-0"`, `<span class="sr-only">State: </span>`, `aria-label="Controls for session logical-working"`} {
		if !strings.Contains(markup, want) {
			t.Errorf("captured render missing %q\n%s", want, markup)
		}
	}
	for _, forbidden := range []string{"thread-sensitive-coordinate-123456", "full prompt", "command arguments"} {
		if strings.Contains(markup, forbidden) {
			t.Errorf("captured overview leaked %q", forbidden)
		}
	}
	for _, want := range []string{"@media(max-width:560px)", "minmax(0,1fr)", "min-height:44px", `role="list"`, `aria-live="polite"`, `aria-busy="true"`, `"ArrowDown"`, `e.key==="Home"`, `e.key==="End"`, "moveFocus(next)", "loadSessions(id)"} {
		if !strings.Contains(page, want) {
			t.Errorf("page lacks responsive or accessible session behavior %q", want)
		}
	}
}

func TestSessionControlTargetsOnlySelectedLogicalSession(t *testing.T) {
	now := time.Now()
	source := &fixtureSessionSource{cards: []SessionCard{
		{ID: "session-a", Provider: "codex", State: sessionWorking, LastEventAt: now, HasInputLease: true, Capabilities: map[string]SessionCapability{"interrupt": {Enabled: true}}},
		{ID: "session-b", Provider: "codex", State: sessionWorking, LastEventAt: now, HasInputLease: true, Capabilities: map[string]SessionCapability{"interrupt": {Enabled: true}}},
	}}
	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, source))
	defer ts.Close()
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/api/sessions/session-b/controls/interrupt", nil)
	res, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusAccepted {
		t.Fatalf("status=%d", res.StatusCode)
	}
	if len(source.controls) != 1 || source.controls[0] != (SessionControlRequest{SessionID: "session-b", Action: "interrupt"}) {
		t.Fatalf("controls=%+v", source.controls)
	}
}

func TestSessionControlRequiresAdvertisedCapabilityAndInputLease(t *testing.T) {
	now := time.Now()
	source := &fixtureSessionSource{cards: []SessionCard{{ID: "leased", Provider: "codex", State: sessionIdle, LastEventAt: now, Capabilities: map[string]SessionCapability{"resume": {Enabled: true}}}}}
	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, source))
	defer ts.Close()
	res, err := ts.Client().Post(ts.URL+"/api/sessions/leased/controls/resume", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusConflict {
		t.Fatalf("status=%d", res.StatusCode)
	}
	if len(source.controls) != 0 {
		t.Fatalf("unauthorized control reached authority: %+v", source.controls)
	}
}

func TestLiveTwoSessionCapturedRenderUpdatesPendingAndCancelled(t *testing.T) {
	now := time.Now()
	source := &fixtureSessionSource{cards: []SessionCard{
		{ID: "approval-live", Provider: "codex", Workspace: "alpha", State: sessionAwaitingApproval, PendingInteraction: "approval requested", LastEventAt: now, HasInputLease: true, Capabilities: map[string]SessionCapability{"cancel": {Enabled: true}}},
		{ID: "cancelled-live", Provider: "codex", Workspace: "beta", State: sessionCancelled, LastEventAt: now, HasInputLease: true, Capabilities: map[string]SessionCapability{"archive": {Enabled: true}}},
	}}
	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, source))
	defer ts.Close()
	body := getBody(t, ts.Client(), ts.URL+"/api/sessions?no_color=1")
	for _, want := range []string{"approval-live", "approval requested", "cancelled-live", "cancelled", `data-session-id=\"approval-live\"`, `data-session-id=\"cancelled-live\"`} {
		if !strings.Contains(body, want) {
			t.Errorf("live capture missing %q: %s", want, body)
		}
	}
}

func getBody(t *testing.T, client *http.Client, endpoint string) string {
	t.Helper()
	res, err := client.Get(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	data, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestParseSessionApprovalDetails(t *testing.T) {
	// 1. Parse from Envelope
	envelope := harnesskit.Envelope{
		Type: harnesskit.EventApprovalRequested,
		Payload: []byte(`{
			"approval_id": "app-101",
			"tool_name": "Bash",
			"command": "git clean -fd",
			"target_path": "/workspace/fak",
			"risk_explanation": "destructive deletion of untracked files"
		}`),
	}
	app, err := ParseSessionApproval(envelope)
	if err != nil {
		t.Fatalf("parse envelope: %v", err)
	}
	if app.ApprovalID != "app-101" {
		t.Errorf("approval id=%q, want app-101", app.ApprovalID)
	}
	if app.ToolName != "Bash" {
		t.Errorf("tool name=%q, want Bash", app.ToolName)
	}
	if app.Command != "git clean -fd" || app.Arguments != "git clean -fd" {
		t.Errorf("command=%q, args=%q", app.Command, app.Arguments)
	}
	if app.TargetPath != "/workspace/fak" {
		t.Errorf("target path=%q, want /workspace/fak", app.TargetPath)
	}
	if app.RiskExplanation != "destructive deletion of untracked files" {
		t.Errorf("risk explanation=%q", app.RiskExplanation)
	}

	// 2. Parse from harnesskit.ApprovalPayload
	payload := harnesskit.ApprovalPayload{
		ApprovalID:   "app-102",
		Kind:         "patch",
		Summary:      "apply proposed patch",
		Scope:        "/workspace/fak/cmd",
		Risk:         "high",
		PolicyReason: "touches core package",
	}
	app2, err := ParseSessionApproval(payload)
	if err != nil {
		t.Fatalf("parse payload: %v", err)
	}
	if app2.ApprovalID != "app-102" || app2.ToolName != "patch" || app2.Command != "apply proposed patch" || app2.TargetPath != "/workspace/fak/cmd" || app2.RiskExplanation != "high: touches core package" {
		t.Fatalf("unexpected parsed approval from ApprovalPayload: %+v", app2)
	}

	// 3. ApplyApprovalEvent to SessionCard
	card := SessionCard{ID: "session-test", Provider: "codex", State: sessionWorking}
	if err := card.ApplyApprovalEvent(envelope); err != nil {
		t.Fatalf("apply approval event: %v", err)
	}
	if card.State != sessionAwaitingApproval {
		t.Errorf("state=%q, want awaiting_approval", card.State)
	}
	if card.PendingApproval == nil || card.PendingApproval.ApprovalID != "app-101" {
		t.Fatalf("pending approval on card: %+v", card.PendingApproval)
	}
}

func TestSessionCardApprovalRendering(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	approval := &SessionApproval{
		ApprovalID:      "approval-exec-1",
		ToolName:        "Bash",
		Command:         "git reset --hard HEAD~1",
		TargetPath:      "/home/user/fak",
		RiskExplanation: "destructive history reset",
	}
	cardWithApproval := SessionCard{
		ID:                 "session-approval-rendered",
		Provider:           "codex",
		Workspace:          "/home/user/fak",
		State:              sessionAwaitingApproval,
		PendingInteraction: "approval requested",
		PendingApproval:    approval,
		LastEventAt:        now,
		HasInputLease:      true,
		Capabilities:       map[string]SessionCapability{"cancel": {Enabled: true}},
	}
	cardWorking := SessionCard{
		ID:            "session-working-test",
		Provider:      "codex",
		Workspace:     "/home/user/fak",
		State:         sessionWorking,
		LastEventAt:   now,
		HasInputLease: true,
		Capabilities:  map[string]SessionCapability{"interrupt": {Enabled: true}},
	}

	markup, err := renderSessionCardsHTML([]SessionCard{cardWithApproval, cardWorking}, now, false)
	if err != nil {
		t.Fatal(err)
	}

	// Verify structured approval details and interactive controls on cardWithApproval
	for _, want := range []string{
		"session-approval-modal",
		"Action approval required",
		"approval-exec-1",
		"Bash",
		"git reset --hard HEAD~1",
		"/home/user/fak",
		"destructive history reset",
		`data-approval-action="accept"`,
		`data-approval-action="decline"`,
		`data-action="accept"`,
		`data-action="decline"`,
		`<form class="approval-form session-approval-controls" action="/api/sessions/session-approval-rendered/approval" method="POST"`,
	} {
		if !strings.Contains(markup, want) {
			t.Errorf("markup missing %q\n%s", want, markup)
		}
	}

	// Verify working card does NOT render approval controls
	if strings.Contains(markup, `/api/sessions/session-working-test/approval`) {
		t.Error("working session unexpectedly rendered approval controls")
	}

	// Verify card with awaiting_approval state but nil PendingApproval also renders actionable controls
	cardAwaitingNoStruct := SessionCard{
		ID:                 "session-awaiting-no-struct",
		Provider:           "codex",
		Workspace:          "/home/user/fak",
		State:              sessionAwaitingApproval,
		PendingInteraction: "operator approval required for high risk bash command",
		LastEventAt:        now,
		HasInputLease:      true,
		Capabilities:       map[string]SessionCapability{"cancel": {Enabled: true}},
	}
	markupNoStruct, err := renderSessionCardsHTML([]SessionCard{cardAwaitingNoStruct}, now, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"session-approval-modal",
		"Action approval required",
		"operator approval required for high risk bash command",
		`data-approval-action="accept"`,
		`data-approval-action="decline"`,
		`action="/api/sessions/session-awaiting-no-struct/approval"`,
	} {
		if !strings.Contains(markupNoStruct, want) {
			t.Errorf("markup for awaiting session without struct missing %q:\n%s", want, markupNoStruct)
		}
	}
}

func TestSessionApprovalEndpointValidationAndResolution(t *testing.T) {
	now := time.Now()
	source := &fixtureSessionSource{cards: []SessionCard{
		{
			ID:                 "session-app-1",
			Provider:           "codex",
			State:              sessionAwaitingApproval,
			PendingInteraction: "approval requested",
			PendingApproval: &SessionApproval{
				ApprovalID:      "app-uuid-1",
				ToolName:        "Edit",
				Command:         "edit file.go",
				TargetPath:      "/tmp/file.go",
				RiskExplanation: "workspace mutation",
			},
			LastEventAt:   now,
			HasInputLease: true,
			Capabilities:  map[string]SessionCapability{"cancel": {Enabled: true}},
		},
		{
			ID:            "session-idle-1",
			Provider:      "codex",
			State:         sessionIdle,
			LastEventAt:   now,
			HasInputLease: true,
			Capabilities:  map[string]SessionCapability{"resume": {Enabled: true}},
		},
	}}

	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, source))
	defer ts.Close()

	// 1. Invalid resolution
	res, err := ts.Client().Post(ts.URL+"/api/sessions/session-app-1/approval", "application/json", strings.NewReader(`{"resolution":"maybe"}`))
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for invalid resolution, got %d", res.StatusCode)
	}

	// 2. Non-existent session
	res, err = ts.Client().Post(ts.URL+"/api/sessions/nonexistent/approval", "application/json", strings.NewReader(`{"resolution":"accept"}`))
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 for nonexistent session, got %d", res.StatusCode)
	}

	// 3. Session not awaiting approval
	res, err = ts.Client().Post(ts.URL+"/api/sessions/session-idle-1/approval", "application/json", strings.NewReader(`{"resolution":"accept"}`))
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusConflict {
		t.Fatalf("expected 409 for idle session, got %d", res.StatusCode)
	}

	// 4. Successful accept resolution
	res, err = ts.Client().Post(ts.URL+"/api/sessions/session-app-1/approval", "application/json", strings.NewReader(`{"resolution":"accept","reason":"approved by test"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for accept, got %d", res.StatusCode)
	}
	var resp map[string]any
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp["status"] != "accepted" || resp["session_id"] != "session-app-1" || resp["resolution"] != "accept" {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if len(source.approvals) != 1 || source.approvals[0].Resolution != "accept" || source.approvals[0].ApprovalID != "app-uuid-1" {
		t.Fatalf("unexpected source approvals: %+v", source.approvals)
	}
}

func TestSessionCardsSSEHeartbeatOnIdleStream(t *testing.T) {
	restore := setSSEHeartbeatInterval(25 * time.Millisecond)
	defer restore()

	source := &fixtureSessionSource{cards: []SessionCard{
		{ID: "sess-heartbeat", Provider: "codex", State: sessionWorking, LastEventAt: time.Now()},
	}}
	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, source))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/api/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("expected Content-Type text/event-stream, got %q", ct)
	}

	reader := bufio.NewReader(resp.Body)
	heartbeatSeen := make(chan bool, 1)
	go func() {
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				return
			}
			if strings.HasPrefix(strings.TrimSpace(line), ": ping") {
				heartbeatSeen <- true
				return
			}
		}
	}()

	select {
	case <-heartbeatSeen:
		// Heartbeat received successfully
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for keep-alive SSE heartbeat comment on idle stream")
	}

	cancel()
}

func TestSessionCardsBroadcasterErrorResilience(t *testing.T) {
	resetSessionHubForTest()
	defer resetSessionHubForTest()

	now := time.Now()
	initialCards := []SessionCard{
		{ID: "session-resilient-1", Provider: "codex", State: sessionWorking, LastEventAt: now},
		{ID: "session-resilient-2", Provider: "codex", State: sessionIdle, LastEventAt: now},
	}
	source := &fixtureSessionSource{
		cards: initialCards,
	}

	subCh := SubscribeSessionEvents("")
	defer UnsubscribeSessionEvents(subCh)

	// 1. Initial successful broadcast populates card state
	broadcastCards(source)

	current := CurrentCards()
	if len(current) != 2 {
		t.Fatalf("expected 2 populated cards, got %d", len(current))
	}
	if current[0].ID != "session-resilient-1" || current[1].ID != "session-resilient-2" {
		t.Fatalf("unexpected populated cards: %+v", current)
	}

	// Verify initial broadcast from subscriber emits session_update
	select {
	case msg := <-subCh:
		if !strings.Contains(string(msg), "event: session_update") {
			t.Fatalf("expected broadcast event to be session_update, got: %s", string(msg))
		}
		if !strings.Contains(string(msg), "session-resilient-1") {
			t.Fatalf("initial broadcast missing card: %s", string(msg))
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for initial broadcast")
	}

	// Verify no secondary duplicate broadcast event was emitted
	select {
	case dup := <-subCh:
		t.Fatalf("unexpected duplicate broadcast event received: %s", string(dup))
	default:
	}

	// 2. Introduce transient error in source.Sessions()
	source.mu.Lock()
	source.err = fmt.Errorf("transient network timeout")
	source.mu.Unlock()

	// Calling broadcastCards during transient error must NOT wipe out populated card state
	broadcastCards(source)

	currentAfterError := CurrentCards()
	if len(currentAfterError) != 2 {
		t.Fatalf("transient error wiped out populated card state: got %d cards, want 2", len(currentAfterError))
	}
	if currentAfterError[0].ID != "session-resilient-1" || currentAfterError[1].ID != "session-resilient-2" {
		t.Fatalf("card state corrupted: %+v", currentAfterError)
	}

	// Verify no empty wipeout event was broadcast to subscribers
	select {
	case msg := <-subCh:
		if strings.Contains(string(msg), `"sessions":[]`) || strings.Contains(string(msg), "No authoritative sessions are reporting") {
			t.Fatalf("transient error broadcast empty card state wiping out client: %s", string(msg))
		}
	case <-time.After(100 * time.Millisecond):
		// Expected: no wipe-out event sent
	}

	// 3. Clear transient error and verify recovery
	source.mu.Lock()
	source.err = nil
	source.cards = append(source.cards, SessionCard{
		ID: "session-resilient-3", Provider: "codex", State: sessionWorking, LastEventAt: time.Now(),
	})
	source.mu.Unlock()

	broadcastCards(source)

	currentRecovered := CurrentCards()
	if len(currentRecovered) != 3 {
		t.Fatalf("expected 3 cards after recovery, got %d", len(currentRecovered))
	}
	if currentRecovered[2].ID != "session-resilient-3" {
		t.Fatalf("expected session-resilient-3, got: %+v", currentRecovered)
	}
}

func TestSessionBroadcasterErrorResilience(t *testing.T) {
	broadcaster := newSessionBroadcaster()
	subCh, unsub := broadcaster.subscribe()
	defer unsub()

	errSource := &fixtureSessionSource{
		err: fmt.Errorf("backend service unavailable"),
	}

	broadcaster.broadcastCards(errSource)

	select {
	case msg := <-subCh:
		t.Fatalf("sessionBroadcaster unexpectedly broadcast card snapshot on error: %s", string(msg))
	case <-time.After(60 * time.Millisecond):
		// Expected: nothing broadcasted
	}

	validSource := &fixtureSessionSource{
		cards: []SessionCard{
			{ID: "session-broadcaster-ok", Provider: "codex", State: sessionWorking, LastEventAt: time.Now()},
		},
	}
	broadcaster.broadcastCards(validSource)

	select {
	case msg := <-subCh:
		if !strings.Contains(string(msg), "session-broadcaster-ok") {
			t.Fatalf("expected broadcast to contain card id, got: %s", string(msg))
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for broadcast from valid source")
	}
}

func readNextSSEEvent(reader *bufio.Reader) (string, string, error) {
	var eventType, data string
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return "", "", err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if eventType != "" || data != "" {
				return eventType, data, nil
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			if data != "" {
				data += "\n"
			}
			data += strings.TrimPrefix(line, "data: ")
		}
	}
}

func readSSEEvent(t *testing.T, reader *bufio.Reader) (string, string) {
	t.Helper()
	evType, data, err := readNextSSEEvent(reader)
	if err != nil {
		t.Fatalf("error reading SSE stream: %v", err)
	}
	return evType, data
}

func TestSessionCardsSSEInitialConnectionEmitsSessionUpdate(t *testing.T) {
	resetSessionHubForTest()
	defer resetSessionHubForTest()

	now := time.Now()
	source := &fixtureSessionSource{cards: []SessionCard{
		{ID: "sess-sse-init", Provider: "codex", State: sessionWorking, LastEventAt: now},
	}}

	broadcastCards(source)

	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, source))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/api/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}

	reader := bufio.NewReader(resp.Body)
	event1Type, data1 := readSSEEvent(t, reader)
	if event1Type != "connected" {
		t.Fatalf("expected first event to be connected, got %q (data: %s)", event1Type, data1)
	}

	event2Type, data2 := readSSEEvent(t, reader)
	if event2Type != "session_update" {
		t.Fatalf("expected second event to be session_update, got %q (data: %s)", event2Type, data2)
	}
	var payload struct {
		Sessions []SessionCard `json:"sessions"`
		HTML     string        `json:"html"`
	}
	if err := json.Unmarshal([]byte(data2), &payload); err != nil {
		t.Fatalf("failed to parse session_update json data: %v", err)
	}
	if len(payload.Sessions) != 1 || payload.Sessions[0].ID != "sess-sse-init" {
		t.Fatalf("unexpected sessions payload: %+v", payload.Sessions)
	}
	if !strings.Contains(payload.HTML, "sess-sse-init") {
		t.Fatalf("expected html markup to contain session id: %s", payload.HTML)
	}
}

func TestSessionSSE_ScopedReplayFiltering(t *testing.T) {
	resetSessionHubForTest()
	defer resetSessionHubForTest()

	now := time.Now()
	source := &fixtureSessionSource{cards: []SessionCard{
		{ID: "session-alpha", Provider: "codex", State: sessionWorking, LastEventAt: now},
		{ID: "session-beta", Provider: "claude", State: sessionIdle, LastEventAt: now},
	}}

	broadcastCards(source)

	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, source))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 1. Scoped subscription for session-alpha
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/v1/fak/sessions/session-alpha/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}

	reader := bufio.NewReader(resp.Body)
	event1Type, data1 := readSSEEvent(t, reader)
	if event1Type != "connected" {
		t.Fatalf("expected first event to be connected, got %q (data: %s)", event1Type, data1)
	}
	if !strings.Contains(data1, "session-alpha") {
		t.Fatalf("expected connected event to include session-alpha, got %s", data1)
	}

	event2Type, data2 := readSSEEvent(t, reader)
	if event2Type != "session_update" {
		t.Fatalf("expected second event to be session_update, got %q (data: %s)", event2Type, data2)
	}

	var payload struct {
		Sessions []SessionCard `json:"sessions"`
		HTML     string        `json:"html"`
	}
	if err := json.Unmarshal([]byte(data2), &payload); err != nil {
		t.Fatalf("failed to parse session_update json data: %v", err)
	}

	if len(payload.Sessions) != 1 || payload.Sessions[0].ID != "session-alpha" {
		t.Fatalf("expected only session-alpha in sessions payload, got %d cards: %+v", len(payload.Sessions), payload.Sessions)
	}
	if !strings.Contains(payload.HTML, "session-alpha") {
		t.Fatalf("expected html markup to contain session-alpha: %s", payload.HTML)
	}
	if strings.Contains(payload.HTML, "session-beta") {
		t.Fatalf("html markup leaked session-beta: %s", payload.HTML)
	}
	if strings.Contains(data2, "session-beta") {
		t.Fatalf("raw session_update data leaked session-beta: %s", data2)
	}

	// 2. Scoped subscription for an unknown/non-existent session
	reqUnknown, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/v1/fak/sessions/session-nonexistent/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	respUnknown, err := ts.Client().Do(reqUnknown)
	if err != nil {
		t.Fatal(err)
	}
	defer respUnknown.Body.Close()

	if respUnknown.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", respUnknown.StatusCode)
	}

	readerUnknown := bufio.NewReader(respUnknown.Body)
	ev1Type, ev1Data := readSSEEvent(t, readerUnknown)
	if ev1Type != "connected" || !strings.Contains(ev1Data, "session-nonexistent") {
		t.Fatalf("unexpected connected event: %s (%s)", ev1Type, ev1Data)
	}

	ev2Type, ev2Data := readSSEEvent(t, readerUnknown)
	if ev2Type != "session_update" {
		t.Fatalf("expected session_update event, got %q", ev2Type)
	}
	var payloadUnknown struct {
		Sessions []SessionCard `json:"sessions"`
		HTML     string        `json:"html"`
	}
	if err := json.Unmarshal([]byte(ev2Data), &payloadUnknown); err != nil {
		t.Fatalf("failed to parse session_update json data: %v", err)
	}
	if len(payloadUnknown.Sessions) != 0 {
		t.Fatalf("expected 0 sessions for nonexistent session, got %d", len(payloadUnknown.Sessions))
	}
	if payloadUnknown.HTML != "" {
		t.Fatalf("expected empty/omitted HTML payload, got %q", payloadUnknown.HTML)
	}
	if strings.Contains(ev2Data, "session-alpha") || strings.Contains(ev2Data, "session-beta") {
		t.Fatalf("raw session_update data leaked sessions for nonexistent stream: %s", ev2Data)
	}

	// 3. Global subscription for /v1/fak/sessions/events receives all cards
	reqGlobalV1, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/v1/fak/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	respGlobalV1, err := ts.Client().Do(reqGlobalV1)
	if err != nil {
		t.Fatal(err)
	}
	defer respGlobalV1.Body.Close()

	if respGlobalV1.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", respGlobalV1.StatusCode)
	}
	readerGlobalV1 := bufio.NewReader(respGlobalV1.Body)
	g1Ev1Type, g1Ev1Data := readSSEEvent(t, readerGlobalV1)
	if g1Ev1Type != "connected" {
		t.Fatalf("expected connected event, got %q (data: %s)", g1Ev1Type, g1Ev1Data)
	}
	g1Ev2Type, g1Ev2Data := readSSEEvent(t, readerGlobalV1)
	if g1Ev2Type != "session_update" {
		t.Fatalf("expected session_update event, got %q", g1Ev2Type)
	}
	var payloadGlobalV1 struct {
		Sessions []SessionCard `json:"sessions"`
		HTML     string        `json:"html"`
	}
	if err := json.Unmarshal([]byte(g1Ev2Data), &payloadGlobalV1); err != nil {
		t.Fatalf("failed to parse session_update json data: %v", err)
	}
	if len(payloadGlobalV1.Sessions) != 2 {
		t.Fatalf("expected 2 sessions for global v1 stream, got %d", len(payloadGlobalV1.Sessions))
	}
	if !strings.Contains(payloadGlobalV1.HTML, "session-alpha") || !strings.Contains(payloadGlobalV1.HTML, "session-beta") {
		t.Fatalf("expected html markup to contain both sessions: %s", payloadGlobalV1.HTML)
	}

	// 4. Global subscription for /api/sessions/events receives all cards
	reqGlobalAPI, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/api/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	respGlobalAPI, err := ts.Client().Do(reqGlobalAPI)
	if err != nil {
		t.Fatal(err)
	}
	defer respGlobalAPI.Body.Close()

	if respGlobalAPI.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", respGlobalAPI.StatusCode)
	}
	readerGlobalAPI := bufio.NewReader(respGlobalAPI.Body)
	g2Ev1Type, g2Ev1Data := readSSEEvent(t, readerGlobalAPI)
	if g2Ev1Type != "connected" {
		t.Fatalf("expected connected event, got %q (data: %s)", g2Ev1Type, g2Ev1Data)
	}
	g2Ev2Type, g2Ev2Data := readSSEEvent(t, readerGlobalAPI)
	if g2Ev2Type != "session_update" {
		t.Fatalf("expected session_update event, got %q", g2Ev2Type)
	}
	var payloadGlobalAPI struct {
		Sessions []SessionCard `json:"sessions"`
		HTML     string        `json:"html"`
	}
	if err := json.Unmarshal([]byte(g2Ev2Data), &payloadGlobalAPI); err != nil {
		t.Fatalf("failed to parse session_update json data: %v", err)
	}
	if len(payloadGlobalAPI.Sessions) != 2 {
		t.Fatalf("expected 2 sessions for global api stream, got %d", len(payloadGlobalAPI.Sessions))
	}
	if !strings.Contains(payloadGlobalAPI.HTML, "session-alpha") || !strings.Contains(payloadGlobalAPI.HTML, "session-beta") {
		t.Fatalf("expected html markup to contain both sessions: %s", payloadGlobalAPI.HTML)
	}
}

func TestSessionSSE_ScopedDynamicBroadcastFiltering(t *testing.T) {
	resetSessionHubForTest()
	defer resetSessionHubForTest()

	now := time.Now()
	source := &fixtureSessionSource{cards: []SessionCard{
		{ID: "session-alpha", Provider: "codex", State: sessionWorking, LastEventAt: now},
		{ID: "session-beta", Provider: "claude", State: sessionIdle, LastEventAt: now},
	}}

	broadcastCards(source)

	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, source))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 1. Connect scoped SSE client for session-alpha
	reqScoped, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/v1/fak/sessions/session-alpha/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	respScoped, err := ts.Client().Do(reqScoped)
	if err != nil {
		t.Fatal(err)
	}
	defer respScoped.Body.Close()
	if respScoped.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200 for scoped client, got %d", respScoped.StatusCode)
	}

	// 2. Connect global SSE client for all sessions
	reqGlobal, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/v1/fak/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	respGlobal, err := ts.Client().Do(reqGlobal)
	if err != nil {
		t.Fatal(err)
	}
	defer respGlobal.Body.Close()
	if respGlobal.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200 for global client, got %d", respGlobal.StatusCode)
	}

	type sseMsg struct {
		eventType string
		data      string
	}

	scopedReader := bufio.NewReader(respScoped.Body)
	scopedEvents := make(chan sseMsg, 16)
	go func() {
		for {
			evType, evData, err := readNextSSEEvent(scopedReader)
			if err != nil {
				return
			}
			scopedEvents <- sseMsg{eventType: evType, data: evData}
		}
	}()

	globalReader := bufio.NewReader(respGlobal.Body)
	globalEvents := make(chan sseMsg, 16)
	go func() {
		for {
			evType, evData, err := readNextSSEEvent(globalReader)
			if err != nil {
				return
			}
			globalEvents <- sseMsg{eventType: evType, data: evData}
		}
	}()

	// Verify initial connection events on scoped stream
	select {
	case ev := <-scopedEvents:
		if ev.eventType != "connected" || !strings.Contains(ev.data, "session-alpha") {
			t.Fatalf("expected connected event for session-alpha, got: %+v", ev)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for scoped connected event")
	}
	select {
	case ev := <-scopedEvents:
		if ev.eventType != "session_update" || !strings.Contains(ev.data, "session-alpha") {
			t.Fatalf("expected initial session_update for session-alpha, got: %+v", ev)
		}
		if strings.Contains(ev.data, "session-beta") {
			t.Fatalf("initial replay leaked session-beta on scoped stream: %s", ev.data)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for scoped initial session_update")
	}

	// Verify initial connection events on global stream
	select {
	case ev := <-globalEvents:
		if ev.eventType != "connected" {
			t.Fatalf("expected global connected event, got: %+v", ev)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for global connected event")
	}
	select {
	case ev := <-globalEvents:
		if ev.eventType != "session_update" || !strings.Contains(ev.data, "session-alpha") || !strings.Contains(ev.data, "session-beta") {
			t.Fatalf("expected global initial session_update with all sessions, got: %+v", ev)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for global initial session_update")
	}

	// 3. Dynamic broadcast via broadcastCards: updates include new session-gamma
	source.mu.Lock()
	source.cards = []SessionCard{
		{ID: "session-alpha", Provider: "codex", State: sessionWorking, LastEventAt: time.Now()},
		{ID: "session-beta", Provider: "claude", State: sessionWorking, LastEventAt: time.Now()},
		{ID: "session-gamma", Provider: "codex", State: sessionIdle, LastEventAt: time.Now()},
	}
	source.mu.Unlock()

	broadcastCards(source)

	// Global stream MUST receive this dynamic broadcast with all sessions
	select {
	case ev := <-globalEvents:
		if ev.eventType != "session_update" {
			t.Fatalf("expected session_update on global stream, got: %s", ev.eventType)
		}
		if !strings.Contains(ev.data, "session-gamma") {
			t.Fatalf("expected session-gamma in dynamic broadcast on global stream, got: %s", ev.data)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for dynamic session_update on global stream")
	}

	// Scoped stream for session-alpha MUST NOT receive this global broadcast
	select {
	case ev := <-scopedEvents:
		t.Fatalf("scoped stream unexpectedly received global broadcast: %+v", ev)
	case <-time.After(100 * time.Millisecond):
		// Expected: no event delivered to scoped subscriber
	}

	// 4. Scoped event targeted to session-beta
	BroadcastSessionEvent("session-beta", "approval_requested", []byte(`{"session_id":"session-beta","approval_id":"app-beta"}`))

	// Scoped stream for session-alpha MUST NOT receive session-beta's event
	select {
	case ev := <-scopedEvents:
		t.Fatalf("scoped stream for session-alpha unexpectedly received event for session-beta: %+v", ev)
	case <-time.After(100 * time.Millisecond):
		// Expected: no event delivered
	}

	// Global stream MUST receive session-beta's event
	select {
	case ev := <-globalEvents:
		if ev.eventType != "approval_requested" || !strings.Contains(ev.data, "app-beta") {
			t.Fatalf("expected approval_requested for session-beta on global stream, got: %+v", ev)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for session-beta event on global stream")
	}

	// 5. Scoped event targeted to session-alpha
	BroadcastSessionEvent("session-alpha", "approval_requested", []byte(`{"session_id":"session-alpha","approval_id":"app-alpha"}`))

	// Scoped stream for session-alpha MUST receive its own event
	select {
	case ev := <-scopedEvents:
		if ev.eventType != "approval_requested" || !strings.Contains(ev.data, "app-alpha") {
			t.Fatalf("expected approval_requested for session-alpha on scoped stream, got: %+v", ev)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for session-alpha event on scoped stream")
	}

	// Global stream MUST also receive session-alpha's event
	select {
	case ev := <-globalEvents:
		if ev.eventType != "approval_requested" || !strings.Contains(ev.data, "app-alpha") {
			t.Fatalf("expected approval_requested for session-alpha on global stream, got: %+v", ev)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for session-alpha event on global stream")
	}

	// 6. Direct sessionHub subscriber filtering check
	subScoped := SubscribeSessionEvents("session-alpha")
	defer UnsubscribeSessionEvents(subScoped)
	subGlobal := SubscribeSessionEvents("")
	defer UnsubscribeSessionEvents(subGlobal)

	BroadcastSessionUpdate([]byte(`{"dynamic":"global_update"}`))

	select {
	case msg := <-subScoped:
		t.Fatalf("scoped hub subscriber unexpectedly received global BroadcastSessionUpdate: %s", string(msg))
	default:
	}

	select {
	case msg := <-subGlobal:
		if !strings.Contains(string(msg), "global_update") {
			t.Fatalf("expected global hub subscriber to receive BroadcastSessionUpdate, got: %s", string(msg))
		}
	default:
		t.Fatal("global hub subscriber did not receive BroadcastSessionUpdate")
	}
}

func TestSessionCardApprovalFormURLPathEscapesSessionID(t *testing.T) {
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	slashSessionID := "worker/leaf-1"
	approval := &SessionApproval{
		ApprovalID:      "app-slash-1",
		ToolName:        "Bash",
		Command:         "go test ./...",
		TargetPath:      "/workspace/fak",
		RiskExplanation: "command execution",
	}
	card := SessionCard{
		ID:                 slashSessionID,
		Provider:           "codex",
		Workspace:          "/workspace/fak",
		State:              sessionAwaitingApproval,
		PendingInteraction: "approval requested",
		PendingApproval:    approval,
		LastEventAt:        now,
		HasInputLease:      true,
		Capabilities:       map[string]SessionCapability{"cancel": {Enabled: true}},
	}

	markup, err := renderSessionCardsHTML([]SessionCard{card}, now, false)
	if err != nil {
		t.Fatalf("renderSessionCardsHTML failed: %v", err)
	}

	// Verify approval form URL has path-escaped session ID
	wantAction := `action="/api/sessions/worker%2Fleaf-1/approval"`
	if !strings.Contains(markup, wantAction) {
		t.Errorf("rendered markup missing path-escaped action %q\n%s", wantAction, markup)
	}
	unwantedAction := `action="/api/sessions/worker/leaf-1/approval"`
	if strings.Contains(markup, unwantedAction) {
		t.Errorf("rendered markup contains unescaped slash action %q\n%s", unwantedAction, markup)
	}

	// Verify data-session-id preserves unescaped logical session ID for UI/DOM binding
	if !strings.Contains(markup, `data-session-id="worker/leaf-1"`) {
		t.Errorf("rendered markup missing data-session-id with unescaped logical session ID")
	}

	// Verify end-to-end routing to the path-escaped URL
	source := &fixtureSessionSource{cards: []SessionCard{card}}
	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, source))
	defer ts.Close()

	// 1. Submit JSON resolution to path-escaped URL
	escapedURL := fmt.Sprintf("%s/api/sessions/%s/approval", ts.URL, url.PathEscape(slashSessionID))
	res, err := ts.Client().Post(escapedURL, "application/json", strings.NewReader(`{"resolution":"accept","approval_id":"app-slash-1"}`))
	if err != nil {
		t.Fatalf("POST to escaped approval URL failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200 for escaped approval URL, got %d", res.StatusCode)
	}
	if len(source.approvals) != 1 || source.approvals[0].SessionID != slashSessionID || source.approvals[0].Resolution != "accept" {
		t.Fatalf("source did not receive expected approval: %+v", source.approvals)
	}

	// 2. Submit form URL-encoded resolution to path-escaped URL (matching HTML form submission)
	card.State = sessionAwaitingApproval
	card.PendingApproval = approval
	source.cards = []SessionCard{card}
	formBody := strings.NewReader("resolution=decline&approval_id=app-slash-1&reason=denied+by+test")
	formRes, err := ts.Client().Post(escapedURL, "application/x-www-form-urlencoded", formBody)
	if err != nil {
		t.Fatalf("POST form to escaped approval URL failed: %v", err)
	}
	defer formRes.Body.Close()
	if formRes.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200 for form POST to escaped approval URL, got %d", formRes.StatusCode)
	}
	if len(source.approvals) != 2 || source.approvals[1].SessionID != slashSessionID || source.approvals[1].Resolution != "decline" {
		t.Fatalf("source did not receive expected second approval: %+v", source.approvals)
	}
}

func TestSessionCardCapabilitiesDeepCopyPreventsDataRaces(t *testing.T) {
	initialCaps := map[string]SessionCapability{
		"open":      {Enabled: true},
		"interrupt": {Enabled: true},
		"cancel":    {Enabled: false, UnavailableReason: "not cancellable"},
	}

	card := SessionCard{
		ID:           "session-deep-copy-test",
		Provider:     "codex",
		State:        sessionWorking,
		Capabilities: initialCaps,
	}

	// 1. Structural independence: normalizeSessionCards deep-copies Capabilities
	normalized, err := normalizeSessionCards([]SessionCard{card})
	if err != nil {
		t.Fatalf("normalizeSessionCards failed: %v", err)
	}
	cloned := card.Clone()

	// Mutating the original map must not mutate normalized or cloned
	initialCaps["archive"] = SessionCapability{Enabled: true}
	initialCaps["interrupt"] = SessionCapability{Enabled: false}

	if _, exists := normalized[0].Capabilities["archive"]; exists {
		t.Errorf("normalized card saw mutation on original map ('archive' key added)")
	}
	if !normalized[0].Capabilities["interrupt"].Enabled {
		t.Errorf("normalized card 'interrupt' was unexpectedly mutated through original map")
	}
	if _, exists := cloned.Capabilities["archive"]; exists {
		t.Errorf("cloned card saw mutation on original map ('archive' key added)")
	}
	if !cloned.Capabilities["interrupt"].Enabled {
		t.Errorf("cloned card 'interrupt' was unexpectedly mutated through original map")
	}

	// Mutating normalized card must not mutate cloned or original
	normalized[0].Capabilities["resume"] = SessionCapability{Enabled: true}
	if _, exists := cloned.Capabilities["resume"]; exists {
		t.Errorf("cloned card saw mutation from normalized card ('resume' key added)")
	}
	if _, exists := initialCaps["resume"]; exists {
		t.Errorf("original map saw mutation from normalized card ('resume' key added)")
	}

	// 2. Concurrency test: concurrent map writes on original map while readers
	// iterate over normalized/cloned cards' Capabilities (e.g. SSE broadcast & HTML rendering).
	mutableMap := map[string]SessionCapability{
		"open":      {Enabled: true},
		"interrupt": {Enabled: true},
		"cancel":    {Enabled: false},
	}
	sourceCard := SessionCard{
		ID:           "session-race-target",
		Provider:     "codex",
		State:        sessionWorking,
		Capabilities: mutableMap,
	}

	normalizedCards, err := normalizeSessionCards([]SessionCard{sourceCard})
	if err != nil {
		t.Fatalf("normalizeSessionCards: %v", err)
	}
	cardCopy := sourceCard.Clone()

	var wg sync.WaitGroup
	var readerWg sync.WaitGroup
	stopWriter := make(chan struct{})

	// Writer goroutine: constantly writes, updates, and deletes from original map
	wg.Add(1)
	go func() {
		defer wg.Done()
		i := 0
		for {
			select {
			case <-stopWriter:
				return
			default:
				k := fmt.Sprintf("cap-%d", i%8)
				mutableMap[k] = SessionCapability{Enabled: i%2 == 0}
				if i > 5 {
					delete(mutableMap, fmt.Sprintf("cap-%d", (i+3)%8))
				}
				i++
			}
		}
	}()

	// Reader goroutines: simulate concurrent HTML rendering, SSE broadcasting (json.Marshal),
	// and map iteration over normalized & cloned cards
	for g := 0; g < 4; g++ {
		readerWg.Add(1)
		go func() {
			defer readerWg.Done()
			for iter := 0; iter < 150; iter++ {
				// Direct map iteration
				for k, v := range normalizedCards[0].Capabilities {
					_ = k
					_ = v.Enabled
				}
				for k, v := range cardCopy.Capabilities {
					_ = k
					_ = v.Enabled
				}
				// HTML render (sessionAction reads card.Capabilities)
				if _, err := renderSessionCardsHTML(normalizedCards, time.Now(), false); err != nil {
					t.Errorf("renderSessionCardsHTML error: %v", err)
				}
				// JSON marshal (simulates SSE broadcast / API JSON response serialization)
				if _, err := json.Marshal(normalizedCards); err != nil {
					t.Errorf("json.Marshal error: %v", err)
				}
				// Further cloning
				_ = normalizedCards[0].Clone()
			}
		}()
	}

	readerWg.Wait()
	close(stopWriter)
	wg.Wait()
}

func TestSessionEventHookApprovalBroadcastValidCardInventoryAndReplay(t *testing.T) {
	resetSessionHubForTest()
	defer resetSessionHubForTest()

	now := time.Now()
	sessionID := "session-hook-approval-test"
	initialCard := SessionCard{
		ID:          sessionID,
		Provider:    "codex",
		Workspace:   "/workspace/fak",
		State:       sessionWorking,
		LastEventAt: now,
		Capabilities: map[string]SessionCapability{
			"cancel": {Enabled: true},
		},
	}
	source := &fixtureSessionSource{cards: []SessionCard{initialCard}}

	ts := httptest.NewServer(handlerWithSessionSource(newStore(), nil, nil, source))
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 1. Connect initial SSE client to /api/sessions/events
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/api/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for SSE stream, got %d", resp.StatusCode)
	}

	reader := bufio.NewReader(resp.Body)
	evConnected, _ := readSSEEvent(t, reader)
	if evConnected != "connected" {
		t.Fatalf("expected first event to be connected, got %q", evConnected)
	}

	// 2. Post an approval event to /api/sessions/{id}/events
	approvalJSON := `{
		"type": "approval.requested",
		"approval_id": "app-hook-99",
		"tool_name": "Bash",
		"command": "git reset --hard HEAD",
		"target_path": "/workspace/fak",
		"risk_explanation": "hard reset of working tree"
	}`
	postResp, err := ts.Client().Post(ts.URL+"/api/sessions/"+sessionID+"/events", "application/json", strings.NewReader(approvalJSON))
	if err != nil {
		t.Fatal(err)
	}
	defer postResp.Body.Close()
	if postResp.StatusCode != http.StatusAccepted {
		t.Fatalf("expected 202 Accepted for event hook, got %d", postResp.StatusCode)
	}

	// 3. Receive approval_requested event on SSE channel
	evApprovalType, evApprovalData := readSSEEvent(t, reader)
	if evApprovalType != "approval_requested" {
		t.Fatalf("expected approval_requested event, got %q (data: %s)", evApprovalType, evApprovalData)
	}
	if !strings.Contains(evApprovalData, "app-hook-99") {
		t.Fatalf("expected approval data to contain app-hook-99, got %s", evApprovalData)
	}

	// 4. Receive session_update event and verify it is a valid card inventory payload (sessions slice + html)
	evUpdateType, evUpdateData := readSSEEvent(t, reader)
	if evUpdateType != "session_update" {
		t.Fatalf("expected session_update event, got %q (data: %s)", evUpdateType, evUpdateData)
	}

	var updatePayload struct {
		Sessions []SessionCard `json:"sessions"`
		HTML     string        `json:"html"`
	}
	if err := json.Unmarshal([]byte(evUpdateData), &updatePayload); err != nil {
		t.Fatalf("failed to parse session_update json: %v (raw: %s)", err, evUpdateData)
	}
	if len(updatePayload.Sessions) != 1 {
		t.Fatalf("expected 1 session card in session_update payload, got %d", len(updatePayload.Sessions))
	}
	card := updatePayload.Sessions[0]
	if card.ID != sessionID {
		t.Fatalf("card ID = %q, want %q", card.ID, sessionID)
	}
	if card.State != sessionAwaitingApproval {
		t.Fatalf("card State = %q, want %q", card.State, sessionAwaitingApproval)
	}
	if card.PendingApproval == nil || card.PendingApproval.ApprovalID != "app-hook-99" {
		t.Fatalf("card PendingApproval = %+v, want app-hook-99", card.PendingApproval)
	}
	if card.PendingApproval.ToolName != "Bash" || card.PendingApproval.Command != "git reset --hard HEAD" {
		t.Fatalf("card PendingApproval details mismatch: %+v", card.PendingApproval)
	}
	if !strings.Contains(updatePayload.HTML, "app-hook-99") {
		t.Fatalf("session_update HTML lacks approval ID: %s", updatePayload.HTML)
	}
	if !strings.Contains(updatePayload.HTML, sessionID) {
		t.Fatalf("session_update HTML lacks session ID: %s", updatePayload.HTML)
	}

	// 5. Connect a new SSE client to verify replayed state receives updated card state
	reqReplay, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/api/sessions/events", nil)
	if err != nil {
		t.Fatal(err)
	}
	respReplay, err := ts.Client().Do(reqReplay)
	if err != nil {
		t.Fatal(err)
	}
	defer respReplay.Body.Close()

	readerReplay := bufio.NewReader(respReplay.Body)
	rEv1Type, _ := readSSEEvent(t, readerReplay)
	if rEv1Type != "connected" {
		t.Fatalf("expected connected event on reconnect, got %q", rEv1Type)
	}

	rEv2Type, rEv2Data := readSSEEvent(t, readerReplay)
	if rEv2Type != "session_update" {
		t.Fatalf("expected replayed session_update event, got %q (data: %s)", rEv2Type, rEv2Data)
	}

	var replayedPayload struct {
		Sessions []SessionCard `json:"sessions"`
		HTML     string        `json:"html"`
	}
	if err := json.Unmarshal([]byte(rEv2Data), &replayedPayload); err != nil {
		t.Fatalf("failed to parse replayed session_update json: %v (raw: %s)", err, rEv2Data)
	}
	if len(replayedPayload.Sessions) != 1 {
		t.Fatalf("expected 1 replayed session card, got %d", len(replayedPayload.Sessions))
	}
	rCard := replayedPayload.Sessions[0]
	if rCard.ID != sessionID || rCard.State != sessionAwaitingApproval || rCard.PendingApproval == nil || rCard.PendingApproval.ApprovalID != "app-hook-99" {
		t.Fatalf("replayed card mismatch: %+v", rCard)
	}
	if !strings.Contains(replayedPayload.HTML, "app-hook-99") {
		t.Fatalf("replayed HTML lacks approval ID: %s", replayedPayload.HTML)
	}

	// 6. Verify CurrentCards reflects updated state
	current := CurrentCards()
	if len(current) != 1 || current[0].ID != sessionID || current[0].State != sessionAwaitingApproval || current[0].PendingApproval == nil || current[0].PendingApproval.ApprovalID != "app-hook-99" {
		t.Fatalf("CurrentCards() mismatch: %+v", current)
	}
}
