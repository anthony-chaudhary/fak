package harnessweb

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"
)

// SessionSource is the renderer-side boundary to authoritative session state.
// Implementations remain responsible for policy, leases, and provider wire types.
type SessionSource interface {
	Sessions(context.Context) ([]SessionCard, error)
	Control(context.Context, SessionControlRequest) error
}

// SessionApprovalResolver is an optional extension for SessionSource implementations
// that support resolving pending interactive approvals.
type SessionApprovalResolver interface {
	ResolveApproval(context.Context, SessionApprovalRequest) error
}

type SessionApprovalRequest struct {
	SessionID  string `json:"session_id"`
	Resolution string `json:"resolution"`
	Reason     string `json:"reason,omitempty"`
}

type SessionApprovalDetails struct {
	ApprovalID string `json:"approval_id,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`
	Tool       string `json:"tool,omitempty"`
	TargetPath string `json:"target_path,omitempty"`
	RiskReason string `json:"risk_reason,omitempty"`
}

func (a *SessionApprovalDetails) EffectiveTool() string {
	if a == nil {
		return ""
	}
	if a.ToolName != "" {
		return a.ToolName
	}
	return a.Tool
}

type SessionState string

const (
	sessionWorking          SessionState = "working"
	sessionAwaitingApproval SessionState = "awaiting_approval"
	sessionAwaitingInput    SessionState = "awaiting_input"
	sessionIdle             SessionState = "idle"
	sessionDisconnected     SessionState = "disconnected"
	sessionCancelled        SessionState = "cancelled"
	sessionFailed           SessionState = "failed"
)

type SessionCapability struct {
	Enabled           bool   `json:"enabled"`
	UnavailableReason string `json:"unavailable_reason,omitempty"`
}

type SessionUsage struct {
	InputTokens  int64 `json:"input_tokens,omitempty"`
	OutputTokens int64 `json:"output_tokens,omitempty"`
}

type SessionCard struct {
	ID                 string                       `json:"id"`
	Provider           string                       `json:"provider"`
	Workspace          string                       `json:"workspace"`
	ThreadCoordinate   string                       `json:"thread_coordinate,omitempty"`
	ExecutionEpoch     uint64                       `json:"execution_epoch"`
	State              SessionState                 `json:"state"`
	PendingInteraction string                       `json:"pending_interaction,omitempty"`
	LastEventAt        time.Time                    `json:"last_event_at"`
	Model              string                       `json:"model,omitempty"`
	Usage              *SessionUsage                `json:"usage,omitempty"`
	HasInputLease      bool                         `json:"has_input_lease"`
	Capabilities       map[string]SessionCapability `json:"capabilities"`
	Approval           *SessionApprovalDetails      `json:"approval,omitempty"`
}

type SessionControlRequest struct {
	SessionID string `json:"session_id"`
	Action    string `json:"action"`
}

var sessionActions = []string{"open", "resume", "interrupt", "cancel", "archive"}

type sessionBroadcaster struct {
	mu      sync.RWMutex
	clients map[chan []byte]struct{}
}

func newSessionBroadcaster() *sessionBroadcaster {
	return &sessionBroadcaster{
		clients: make(map[chan []byte]struct{}),
	}
}

func (b *sessionBroadcaster) subscribe() (chan []byte, func()) {
	b.mu.Lock()
	defer b.mu.Unlock()
	ch := make(chan []byte, 16)
	b.clients[ch] = struct{}{}
	return ch, func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		delete(b.clients, ch)
	}
}

func (b *sessionBroadcaster) broadcast(msg []byte) {
	if b == nil {
		return
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	for ch := range b.clients {
		select {
		case ch <- msg:
		default:
		}
	}
}

func (b *sessionBroadcaster) broadcastCards(source SessionSource) {
	if b == nil || source == nil {
		return
	}
	var c []SessionCard
	if cards, err := source.Sessions(context.Background()); err == nil {
		if norm, err := normalizeSessionCards(cards); err == nil {
			c = norm
		}
	}
	markup, err := renderSessionCardsHTML(c, time.Now(), false)
	if err != nil {
		return
	}
	data, err := json.Marshal(map[string]any{"sessions": c, "html": markup})
	if err != nil {
		return
	}
	b.broadcast([]byte(fmt.Sprintf("event: session_cards\ndata: %s\n\n", data)))
}

func (c SessionCard) validate() error {
	if strings.TrimSpace(c.ID) == "" {
		return fmt.Errorf("session card: empty logical session id")
	}
	switch c.State {
	case sessionWorking, sessionAwaitingApproval, sessionAwaitingInput, sessionIdle, sessionDisconnected, sessionCancelled, sessionFailed:
	default:
		return fmt.Errorf("session card %q: unknown state %q", c.ID, c.State)
	}
	return nil
}

func sessionAction(c SessionCard, action string) SessionCapability {
	capability, ok := c.Capabilities[action]
	if !ok {
		return SessionCapability{UnavailableReason: "not advertised by session"}
	}
	if capability.Enabled && action != "open" && !c.HasInputLease {
		return SessionCapability{UnavailableReason: "input lease is held elsewhere"}
	}
	if !capability.Enabled && capability.UnavailableReason == "" {
		capability.UnavailableReason = "unavailable"
	}
	return capability
}

func normalizeSessionCards(cards []SessionCard) ([]SessionCard, error) {
	out := append([]SessionCard(nil), cards...)
	seen := map[string]bool{}
	for i := range out {
		if err := out[i].validate(); err != nil {
			return nil, err
		}
		if seen[out[i].ID] {
			return nil, fmt.Errorf("session card: duplicate logical session id %q", out[i].ID)
		}
		seen[out[i].ID] = true
		// A provider coordinate is deliberately shortened at this boundary. Full
		// thread IDs and transcript/argument content never enter overview markup.
		out[i].ThreadCoordinate = shortCoordinate(out[i].ThreadCoordinate)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func shortCoordinate(value string) string {
	value = strings.TrimSpace(value)
	if len(value) <= 12 {
		return value
	}
	return value[:6] + "…" + value[len(value)-4:]
}

func sessionStateLabel(state SessionState) string {
	return strings.ReplaceAll(string(state), "_", " ")
}

func sessionAge(now, at time.Time) string {
	if at.IsZero() {
		return "unknown"
	}
	d := now.Sub(at)
	if d < 0 {
		d = 0
	}
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm", int(d.Minutes()))
	}
	return fmt.Sprintf("%dh", int(d.Hours()))
}

// renderSessionCardsHTML is also the captured-render witness seam. The same
// escaped markup is returned by the live endpoint and asserted by fixtures.
func renderSessionCardsHTML(cards []SessionCard, now time.Time, noColor bool) (string, error) {
	cards, err := normalizeSessionCards(cards)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	if len(cards) == 0 {
		return `<p class="empty">No authoritative sessions are reporting.</p>`, nil
	}
	for index, c := range cards {
		stateClass := "state-" + strings.ReplaceAll(string(c.State), "_", "-")
		if noColor {
			stateClass = "state-text"
		}
		tabIndex := -1
		if index == 0 {
			tabIndex = 0
		}
		titleID := fmt.Sprintf("session-title-%d", index)
		detailID := fmt.Sprintf("session-detail-%d", index)
		fmt.Fprintf(&b, `<article class="session-card" role="listitem" tabindex="%d" data-session-id="%s" aria-labelledby="%s" aria-describedby="%s">`, tabIndex, html.EscapeString(c.ID), titleID, detailID)
		fmt.Fprintf(&b, `<div class="session-title" id="%s"><strong>%s</strong><span class="provider">%s</span><span class="session-state %s"><span class="sr-only">State: </span>%s</span></div>`, titleID, html.EscapeString(c.ID), html.EscapeString(c.Provider), stateClass, html.EscapeString(sessionStateLabel(c.State)))
		fmt.Fprintf(&b, `<span class="sr-only" id="%s">%s session %s details and controls</span>`, detailID, html.EscapeString(c.Provider), html.EscapeString(c.ID))
		fmt.Fprintf(&b, `<dl><div><dt>Workspace</dt><dd>%s</dd></div><div><dt>Thread</dt><dd>%s</dd></div><div><dt>Epoch</dt><dd>%d</dd></div><div><dt>Last event</dt><dd>%s ago</dd></div>`, html.EscapeString(c.Workspace), html.EscapeString(c.ThreadCoordinate), c.ExecutionEpoch, sessionAge(now, c.LastEventAt))
		if c.Model != "" {
			fmt.Fprintf(&b, `<div><dt>Model</dt><dd>%s</dd></div>`, html.EscapeString(c.Model))
		}
		if c.Usage != nil {
			fmt.Fprintf(&b, `<div><dt>Usage</dt><dd>%d in / %d out</dd></div>`, c.Usage.InputTokens, c.Usage.OutputTokens)
		}
		b.WriteString(`</dl>`)
		if c.PendingInteraction != "" {
			fmt.Fprintf(&b, `<p class="pending"><strong>Needs action:</strong> %s</p>`, html.EscapeString(c.PendingInteraction))
		}
		if c.Approval != nil {
			tool := c.Approval.EffectiveTool()
			b.WriteString(`<div class="session-approval"><dl class="approval-details">`)
			if tool != "" {
				fmt.Fprintf(&b, `<div><dt>Tool</dt><dd class="approval-tool">%s</dd></div>`, html.EscapeString(tool))
			}
			if c.Approval.TargetPath != "" {
				fmt.Fprintf(&b, `<div><dt>Target path</dt><dd class="approval-target-path">%s</dd></div>`, html.EscapeString(c.Approval.TargetPath))
			}
			if c.Approval.RiskReason != "" {
				fmt.Fprintf(&b, `<div><dt>Risk reason</dt><dd class="approval-risk-reason">%s</dd></div>`, html.EscapeString(c.Approval.RiskReason))
			}
			b.WriteString(`</dl>`)
			fmt.Fprintf(&b, `<div class="approval-actions" role="group" aria-label="Approval resolution for session %s">`, html.EscapeString(c.ID))
			fmt.Fprintf(&b, `<button type="button" class="button-accept" data-approval-action="accept" data-session-id="%s" aria-label="Accept approval for session %s">Accept</button>`, html.EscapeString(c.ID), html.EscapeString(c.ID))
			fmt.Fprintf(&b, `<button type="button" class="button-decline" data-approval-action="decline" data-session-id="%s" aria-label="Decline approval for session %s">Decline</button>`, html.EscapeString(c.ID), html.EscapeString(c.ID))
			b.WriteString(`</div></div>`)
		} else if c.State == sessionAwaitingApproval {
			fmt.Fprintf(&b, `<div class="session-approval"><div class="approval-actions" role="group" aria-label="Approval resolution for session %s">`, html.EscapeString(c.ID))
			fmt.Fprintf(&b, `<button type="button" class="button-accept" data-approval-action="accept" data-session-id="%s" aria-label="Accept approval for session %s">Accept</button>`, html.EscapeString(c.ID), html.EscapeString(c.ID))
			fmt.Fprintf(&b, `<button type="button" class="button-decline" data-approval-action="decline" data-session-id="%s" aria-label="Decline approval for session %s">Decline</button>`, html.EscapeString(c.ID), html.EscapeString(c.ID))
			b.WriteString(`</div></div>`)
		}
		fmt.Fprintf(&b, `<div class="session-controls" role="group" aria-label="Controls for session %s">`, html.EscapeString(c.ID))
		for _, action := range sessionActions {
			capability := sessionAction(c, action)
			label := strings.ToUpper(action[:1]) + action[1:]
			if capability.Enabled {
				fmt.Fprintf(&b, `<button type="button" data-session-action="%s" data-session-id="%s" aria-label="%s session %s">%s</button>`, action, html.EscapeString(c.ID), label, html.EscapeString(c.ID), label)
			} else {
				fmt.Fprintf(&b, `<button type="button" disabled aria-disabled="true" title="%s" aria-label="%s session %s unavailable: %s">%s</button>`, html.EscapeString(capability.UnavailableReason), label, html.EscapeString(c.ID), html.EscapeString(capability.UnavailableReason), label)
			}
		}
		b.WriteString(`</div></article>`)
	}
	return b.String(), nil
}

func installSessionRoutes(mux *http.ServeMux, source SessionSource, broadcaster *sessionBroadcaster) {
	mux.HandleFunc("GET /api/sessions", func(w http.ResponseWriter, r *http.Request) {
		if source == nil {
			writeSessionJSON(w, http.StatusOK, map[string]any{"sessions": []SessionCard{}, "html": `<p class="empty">Session authority is not connected.</p>`})
			return
		}
		cards, err := source.Sessions(r.Context())
		if err != nil {
			writeSessionJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
			return
		}
		cards, err = normalizeSessionCards(cards)
		if err != nil {
			writeSessionJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
			return
		}
		markup, err := renderSessionCardsHTML(cards, time.Now(), r.URL.Query().Get("no_color") == "1")
		if err != nil {
			writeSessionJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
			return
		}
		writeSessionJSON(w, http.StatusOK, map[string]any{"sessions": cards, "html": markup})
	})
	mux.HandleFunc("GET /api/sessions/events", func(w http.ResponseWriter, r *http.Request) {
		handleSessionSSE(w, r, source, broadcaster)
	})
	mux.HandleFunc("GET /api/sessions/stream", func(w http.ResponseWriter, r *http.Request) {
		handleSessionSSE(w, r, source, broadcaster)
	})
	mux.HandleFunc("POST /api/sessions/{id}/controls/{action}", func(w http.ResponseWriter, r *http.Request) {
		if source == nil {
			writeSessionJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "session authority is not connected"})
			return
		}
		id, err := url.PathUnescape(r.PathValue("id"))
		if err != nil || id == "" {
			http.Error(w, "invalid session id", http.StatusBadRequest)
			return
		}
		action := r.PathValue("action")
		valid := false
		for _, candidate := range sessionActions {
			valid = valid || candidate == action
		}
		if !valid {
			http.Error(w, "unknown session control", http.StatusNotFound)
			return
		}
		cards, err := source.Sessions(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
		cards, err = normalizeSessionCards(cards)
		if err != nil {
			writeSessionJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
			return
		}
		var selected *SessionCard
		for i := range cards {
			if cards[i].ID == id {
				selected = &cards[i]
				break
			}
		}
		if selected == nil {
			http.Error(w, "logical session not found", http.StatusNotFound)
			return
		}
		capability := sessionAction(*selected, action)
		if !capability.Enabled {
			writeSessionJSON(w, http.StatusConflict, map[string]string{"error": capability.UnavailableReason})
			return
		}
		if err := source.Control(r.Context(), SessionControlRequest{SessionID: id, Action: action}); err != nil {
			writeSessionJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}
		if broadcaster != nil {
			broadcaster.broadcastCards(source)
		}
		writeSessionJSON(w, http.StatusAccepted, SessionControlRequest{SessionID: id, Action: action})
	})
}

func handleSessionSSE(w http.ResponseWriter, r *http.Request, source SessionSource, broadcaster *sessionBroadcaster) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	var cards []SessionCard
	if source != nil {
		if c, err := source.Sessions(r.Context()); err == nil {
			if norm, err := normalizeSessionCards(c); err == nil {
				cards = norm
			}
		}
	}
	markup, _ := renderSessionCardsHTML(cards, time.Now(), r.URL.Query().Get("no_color") == "1")
	initialData, _ := json.Marshal(map[string]any{
		"sessions": cards,
		"html":     markup,
	})
	fmt.Fprintf(w, "event: session_cards\ndata: %s\n\n", initialData)
	flusher.Flush()

	if broadcaster == nil {
		<-r.Context().Done()
		return
	}

	ch, unsubscribe := broadcaster.subscribe()
	defer unsubscribe()

	for {
		select {
		case <-r.Context().Done():
			return
		case msg, ok := <-ch:
			if !ok {
				return
			}
			_, _ = w.Write(msg)
			flusher.Flush()
		}
	}
}

func writeSessionJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
