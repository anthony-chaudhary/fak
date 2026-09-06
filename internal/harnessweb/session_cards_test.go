package harnessweb

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
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
	for i := range s.cards {
		if s.cards[i].ID == request.SessionID {
			s.approvals = append(s.approvals, request)
			if request.Resolution == "accept" || request.Resolution == "approve" {
				s.cards[i].State = sessionWorking
			} else {
				s.cards[i].State = sessionCancelled
			}
			s.cards[i].PendingInteraction = ""
			s.cards[i].Approval = nil
			return nil
		}
	}
	return fmt.Errorf("session not found: %s", request.SessionID)
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
