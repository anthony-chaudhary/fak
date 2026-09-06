package harnessweb

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

type fixtureSessionSource struct {
	mu        sync.Mutex
	cards     []SessionCard
	controls  []SessionControlRequest
	approvals []SessionApprovalRequest
}

func (s *fixtureSessionSource) Sessions(context.Context) ([]SessionCard, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
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
