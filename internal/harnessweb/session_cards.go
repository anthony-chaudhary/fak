package harnessweb

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// SessionSource is the renderer-side boundary to authoritative session state.
// Implementations remain responsible for policy, leases, and provider wire types.
type SessionSource interface {
	Sessions(context.Context) ([]SessionCard, error)
	Control(context.Context, SessionControlRequest) error
	ResolveApproval(context.Context, SessionApprovalRequest) error
}

// SessionApproval captures structured details of a pending approval requested by a session.
type SessionApproval struct {
	ApprovalID      string `json:"approval_id"`
	ToolName        string `json:"tool_name,omitempty"`
	Command         string `json:"command,omitempty"`
	Arguments       string `json:"arguments,omitempty"`
	TargetPath      string `json:"target_path,omitempty"`
	RiskExplanation string `json:"risk_explanation,omitempty"`
}

// SessionApprovalRequest conveys an approval resolution to an authoritative session source.
type SessionApprovalRequest struct {
	SessionID  string `json:"session_id"`
	ApprovalID string `json:"approval_id,omitempty"`
	Resolution string `json:"resolution"` // "accept" | "decline"
	Reason     string `json:"reason,omitempty"`
}

// SessionApprovalResolver resolves pending approvals for a session.
type SessionApprovalResolver interface {
	ResolveApproval(context.Context, SessionApprovalRequest) error
}

// SessionApprovalFunc is an adapter allowing the use of ordinary functions as approval resolvers.
type SessionApprovalFunc func(context.Context, SessionApprovalRequest) error

func (f SessionApprovalFunc) ResolveApproval(ctx context.Context, req SessionApprovalRequest) error {
	return f(ctx, req)
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
	PendingApproval    *SessionApproval             `json:"pending_approval,omitempty"`
	LastEventAt        time.Time                    `json:"last_event_at"`
	Model              string                       `json:"model,omitempty"`
	Usage              *SessionUsage                `json:"usage,omitempty"`
	HasInputLease      bool                         `json:"has_input_lease"`
	Capabilities       map[string]SessionCapability `json:"capabilities"`
}

// ApplyApproval associates a structured pending approval with this session card
// and transitions its state to sessionAwaitingApproval.
func (c *SessionCard) ApplyApproval(app *SessionApproval) {
	if app == nil {
		return
	}
	c.PendingApproval = app
	c.State = sessionAwaitingApproval
	if c.PendingInteraction == "" {
		c.PendingInteraction = "approval requested"
	}
}

// ApplyApprovalEvent extracts structured approval details from an approval.requested
// envelope and applies them to this session card.
func (c *SessionCard) ApplyApprovalEvent(env harnesskit.Envelope) error {
	app, err := ParseSessionApproval(env)
	if err != nil {
		return err
	}
	c.ApplyApproval(app)
	return nil
}

// ParseSessionApproval extracts structured approval details (tool name, arguments/command,
// target path, risk explanation, approval ID) from an approval.requested event, payload, or JSON.
func ParseSessionApproval(v any) (*SessionApproval, error) {
	if v == nil {
		return nil, fmt.Errorf("session approval: nil input")
	}

	var rawBytes []byte
	switch val := v.(type) {
	case harnesskit.Envelope:
		rawBytes = val.Payload
	case *harnesskit.Envelope:
		if val == nil {
			return nil, fmt.Errorf("session approval: nil envelope")
		}
		rawBytes = val.Payload
	case []byte:
		rawBytes = val
	case string:
		rawBytes = []byte(val)
	case harnesskit.ApprovalPayload:
		cmd := val.Summary
		risk := firstNonEmpty(val.Risk, val.PolicyReason, val.Prompt, val.Consequence)
		if val.Risk != "" && val.PolicyReason != "" {
			risk = val.Risk + ": " + val.PolicyReason
		}
		return &SessionApproval{
			ApprovalID:      val.ApprovalID,
			ToolName:        val.Kind,
			Command:         cmd,
			Arguments:       cmd,
			TargetPath:      firstNonEmpty(val.Scope, val.Workspace),
			RiskExplanation: risk,
		}, nil
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("session approval: marshal error: %w", err)
		}
		rawBytes = data
	}

	if len(rawBytes) == 0 {
		return nil, fmt.Errorf("session approval: empty payload")
	}

	var raw struct {
		ApprovalID      string          `json:"approval_id"`
		ApprovalId      string          `json:"approvalId"`
		ID              string          `json:"id"`
		ToolName        string          `json:"tool_name"`
		Tool            string          `json:"tool"`
		Name            string          `json:"name"`
		Kind            string          `json:"kind"`
		Command         string          `json:"command"`
		Arguments       json.RawMessage `json:"arguments"`
		Args            json.RawMessage `json:"args"`
		Summary         string          `json:"summary"`
		TargetPath      string          `json:"target_path"`
		Path            string          `json:"path"`
		Scope           string          `json:"scope"`
		Workspace       string          `json:"workspace"`
		GrantRoot       string          `json:"grantRoot"`
		Grant_Root      string          `json:"grant_root"`
		Cwd             string          `json:"cwd"`
		RiskExplanation string          `json:"risk_explanation"`
		Risk            string          `json:"risk"`
		PolicyReason    string          `json:"policy_reason"`
		Reason          string          `json:"reason"`
		Prompt          string          `json:"prompt"`
		Consequence     string          `json:"consequence"`
	}

	if err := json.Unmarshal(rawBytes, &raw); err != nil {
		return nil, fmt.Errorf("session approval: unmarshal error: %w", err)
	}

	approvalID := firstNonEmpty(raw.ApprovalID, raw.ApprovalId, raw.ID)
	toolName := firstNonEmpty(raw.ToolName, raw.Tool, raw.Name, raw.Kind)
	targetPath := firstNonEmpty(raw.TargetPath, raw.Path, raw.Scope, raw.GrantRoot, raw.Grant_Root, raw.Cwd, raw.Workspace)

	cmd := firstNonEmpty(raw.Command, raw.Summary)
	args := extractRawString(raw.Arguments)
	if args == "" {
		args = extractRawString(raw.Args)
	}
	if cmd != "" && args == "" {
		args = cmd
	}
	if args != "" && cmd == "" {
		cmd = args
	}

	risk := raw.RiskExplanation
	if risk == "" {
		if raw.Risk != "" && raw.PolicyReason != "" {
			risk = raw.Risk + ": " + raw.PolicyReason
		} else {
			risk = firstNonEmpty(raw.PolicyReason, raw.Reason, raw.Risk, raw.Prompt, raw.Consequence)
		}
	}

	if approvalID == "" && toolName == "" && cmd == "" && targetPath == "" && risk == "" {
		return nil, fmt.Errorf("session approval: no structured approval details found")
	}

	return &SessionApproval{
		ApprovalID:      approvalID,
		ToolName:        toolName,
		Command:         cmd,
		Arguments:       args,
		TargetPath:      targetPath,
		RiskExplanation: risk,
	}, nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if trimmed := strings.TrimSpace(v); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func extractRawString(rm json.RawMessage) string {
	if len(rm) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(rm, &s); err == nil {
		return strings.TrimSpace(s)
	}
	return strings.TrimSpace(string(rm))
}

type SessionControlRequest struct {
	SessionID string `json:"session_id"`
	Action    string `json:"action"`
}

var sessionActions = []string{"open", "resume", "interrupt", "cancel", "archive"}

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

// Clone returns a deep copy of the session card, allocating a separate map for
// Capabilities to prevent data races between concurrent readers and writers.
func (c SessionCard) Clone() SessionCard {
	cp := c
	if c.Capabilities != nil {
		cp.Capabilities = make(map[string]SessionCapability, len(c.Capabilities))
		for k, v := range c.Capabilities {
			cp.Capabilities[k] = v
		}
	}
	if c.PendingApproval != nil {
		app := *c.PendingApproval
		cp.PendingApproval = &app
	}
	if c.Usage != nil {
		u := *c.Usage
		cp.Usage = &u
	}
	return cp
}

// cloneSessionCards returns a deep copy of the given slice of session cards.
func cloneSessionCards(cards []SessionCard) []SessionCard {
	if cards == nil {
		return nil
	}
	out := make([]SessionCard, len(cards))
	for i := range cards {
		out[i] = cards[i].Clone()
	}
	return out
}

func normalizeSessionCards(cards []SessionCard) ([]SessionCard, error) {
	if cards == nil {
		return nil, nil
	}
	out := make([]SessionCard, len(cards))
	seen := map[string]bool{}
	for i := range cards {
		out[i] = cards[i].Clone()
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
		if c.State == sessionAwaitingApproval || c.PendingApproval != nil {
			approvalID := ""
			if c.PendingApproval != nil && c.PendingApproval.ApprovalID != "" {
				approvalID = c.PendingApproval.ApprovalID
			} else {
				approvalID = "approval-" + c.ID
			}
			fmt.Fprintf(&b, `<div class="session-approval-modal session-approval" role="region" aria-label="Approval required for session %s">`, html.EscapeString(c.ID))
			fmt.Fprintf(&b, `<div class="approval-title"><strong>Action approval required</strong></div>`)
			if c.PendingApproval != nil {
				fmt.Fprintf(&b, `<dl class="approval-details">`)
				if c.PendingApproval.ApprovalID != "" {
					fmt.Fprintf(&b, `<div><dt>Approval ID</dt><dd class="approval-id">%s</dd></div>`, html.EscapeString(c.PendingApproval.ApprovalID))
				}
				if c.PendingApproval.ToolName != "" {
					fmt.Fprintf(&b, `<div><dt>Tool</dt><dd class="approval-tool">%s</dd></div>`, html.EscapeString(c.PendingApproval.ToolName))
				}
				cmd := c.PendingApproval.Command
				if cmd == "" {
					cmd = c.PendingApproval.Arguments
				}
				if cmd != "" {
					fmt.Fprintf(&b, `<div><dt>Command</dt><dd class="approval-command"><code>%s</code></dd></div>`, html.EscapeString(cmd))
				}
				if c.PendingApproval.TargetPath != "" {
					fmt.Fprintf(&b, `<div><dt>Target path</dt><dd class="approval-target-path">%s</dd></div>`, html.EscapeString(c.PendingApproval.TargetPath))
				}
				if c.PendingApproval.RiskExplanation != "" {
					fmt.Fprintf(&b, `<div><dt>Risk explanation</dt><dd class="approval-risk">%s</dd></div>`, html.EscapeString(c.PendingApproval.RiskExplanation))
				}
				fmt.Fprintf(&b, `</dl>`)
			}
			fmt.Fprintf(&b, `<form class="approval-form session-approval-controls" action="/api/sessions/%s/approval" method="POST" data-session-id="%s" data-approval-id="%s">`, url.PathEscape(c.ID), html.EscapeString(c.ID), html.EscapeString(approvalID))
			fmt.Fprintf(&b, `<input type="hidden" name="session_id" value="%s">`, html.EscapeString(c.ID))
			fmt.Fprintf(&b, `<input type="hidden" name="approval_id" value="%s">`, html.EscapeString(approvalID))
			fmt.Fprintf(&b, `<button type="submit" name="resolution" value="accept" class="button-approval-accept" data-approval-action="accept" data-action="accept" data-session-id="%s" data-approval-id="%s" aria-label="Accept approval for session %s">Accept</button>`, html.EscapeString(c.ID), html.EscapeString(approvalID), html.EscapeString(c.ID))
			fmt.Fprintf(&b, `<button type="submit" name="resolution" value="decline" class="button-approval-decline" data-approval-action="decline" data-action="decline" data-session-id="%s" data-approval-id="%s" aria-label="Decline approval for session %s">Decline</button>`, html.EscapeString(c.ID), html.EscapeString(approvalID), html.EscapeString(c.ID))
			fmt.Fprintf(&b, `</form></div>`)
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

type sseSubscriber struct {
	ch        chan []byte
	sessionID string
}

type sessionHub struct {
	mu          sync.RWMutex
	subscribers map[chan []byte]*sseSubscriber
	lastCards   []SessionCard
	lastHTML    string
}

var defaultSessionHub = &sessionHub{
	subscribers: make(map[chan []byte]*sseSubscriber),
}

func (h *sessionHub) subscribe(sessionID string) chan []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	ch := make(chan []byte, 32)
	h.subscribers[ch] = &sseSubscriber{ch: ch, sessionID: sessionID}
	return ch
}

func (h *sessionHub) unsubscribe(ch chan []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.subscribers[ch]; ok {
		delete(h.subscribers, ch)
		close(ch)
	}
}

func (h *sessionHub) broadcastSession(sessionID string, eventType string, data []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	msg := []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, string(data)))
	for ch, sub := range h.subscribers {
		if sub.sessionID == "" || sessionID == "" || sub.sessionID == sessionID {
			select {
			case ch <- msg:
			default:
			}
		}
	}
}

func (h *sessionHub) broadcast(eventType string, data []byte) {
	h.broadcastSession("", eventType, data)
}

// BroadcastSessionUpdate notifies all connected SSE clients of a session update.
func BroadcastSessionUpdate(data []byte) {
	defaultSessionHub.broadcast("session_update", data)
}

// BroadcastApprovalResolved notifies all connected SSE clients that an approval was resolved.
func BroadcastApprovalResolved(data []byte) {
	defaultSessionHub.broadcast("approval_resolved", data)
}

// BroadcastApprovalRequested notifies all connected SSE clients that an approval is requested.
func BroadcastApprovalRequested(data []byte) {
	defaultSessionHub.broadcast("approval_requested", data)
}

// BroadcastSessionEvent notifies SSE clients listening to a specific session (and global subscribers).
func BroadcastSessionEvent(sessionID string, eventType string, data []byte) {
	defaultSessionHub.broadcastSession(sessionID, eventType, data)
}

// SubscribeSessionEvents registers a channel to receive live SSE events for a session
// (or all sessions if sessionID is empty).
func SubscribeSessionEvents(sessionID string) chan []byte {
	return defaultSessionHub.subscribe(sessionID)
}

// UnsubscribeSessionEvents removes an SSE subscriber channel.
func UnsubscribeSessionEvents(ch chan []byte) {
	defaultSessionHub.unsubscribe(ch)
}

// CurrentCards returns the latest populated session cards held in the default hub.
func CurrentCards() []SessionCard {
	return defaultSessionHub.currentCards()
}

func (h *sessionHub) currentCards() []SessionCard {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return cloneSessionCards(h.lastCards)
}

func (h *sessionHub) resetForTest() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.lastCards = nil
	h.lastHTML = ""
}

func resetSessionHubForTest() {
	defaultSessionHub.resetForTest()
}

// broadcastCards fetches the latest session cards from source, renders the HTML markup,
// and broadcasts the update to all connected SSE clients. If source.Sessions() returns
// a transient error, previously populated card state is preserved and not wiped out.
func broadcastCards(source SessionSource) {
	defaultSessionHub.broadcastCards(source)
}

// BroadcastCards fetches the latest session cards from source and broadcasts them to clients,
// preserving existing populated card state on transient errors.
func BroadcastCards(source SessionSource) {
	defaultSessionHub.broadcastCards(source)
}

func (h *sessionHub) broadcastCards(source SessionSource) {
	if h == nil || source == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cards, err := source.Sessions(ctx)
	if err != nil {
		// Transient error from source.Sessions(): preserve existing populated card state.
		// Do not overwrite lastCards with empty/nil and do not broadcast empty markup.
		return
	}
	norm, err := normalizeSessionCards(cards)
	if err != nil {
		return
	}
	markup, err := renderSessionCardsHTML(norm, time.Now(), false)
	if err != nil {
		return
	}
	h.mu.Lock()
	h.lastCards = cloneSessionCards(norm)
	h.lastHTML = markup
	h.mu.Unlock()

	data, err := json.Marshal(map[string]any{"sessions": norm, "html": markup})
	if err != nil {
		return
	}
	h.broadcast("session_update", data)
}

// sessionBroadcaster manages active SSE subscriptions and broadcasts session card updates.
type sessionBroadcaster struct {
	hub *sessionHub
}

func newSessionBroadcaster() *sessionBroadcaster {
	return &sessionBroadcaster{hub: defaultSessionHub}
}

func (b *sessionBroadcaster) broadcastCards(source SessionSource) {
	if b == nil || b.hub == nil {
		defaultSessionHub.broadcastCards(source)
		return
	}
	b.hub.broadcastCards(source)
}

func (b *sessionBroadcaster) subscribe() (chan []byte, func()) {
	if b == nil || b.hub == nil {
		ch := defaultSessionHub.subscribe("")
		return ch, func() { defaultSessionHub.unsubscribe(ch) }
	}
	ch := b.hub.subscribe("")
	return ch, func() { b.hub.unsubscribe(ch) }
}

var (
	sseHeartbeatMu       sync.RWMutex
	sseHeartbeatInterval = 15 * time.Second
)

func getSSEHeartbeatInterval() time.Duration {
	sseHeartbeatMu.RLock()
	defer sseHeartbeatMu.RUnlock()
	return sseHeartbeatInterval
}

func setSSEHeartbeatInterval(d time.Duration) func() {
	sseHeartbeatMu.Lock()
	old := sseHeartbeatInterval
	sseHeartbeatInterval = d
	sseHeartbeatMu.Unlock()
	return func() {
		sseHeartbeatMu.Lock()
		sseHeartbeatInterval = old
		sseHeartbeatMu.Unlock()
	}
}

// handleSessionSSE serves Server-Sent Events (SSE) for session cards and approval notifications.
// If scoped is true, it only streams events matching the path parameter {id} and global broadcasts.
// It emits periodic comment heartbeats (: ping\n\n) every 15 seconds during idle periods to prevent
// intermediate proxies from closing idle connections.
func handleSessionSSE(scoped bool) http.HandlerFunc {
	return handleSessionSSEWithInterval(scoped, getSSEHeartbeatInterval())
}

func handleSessionSSEWithInterval(scoped bool, interval time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusBadRequest)
			return
		}
		sessionID := ""
		if scoped {
			var err error
			sessionID, err = url.PathUnescape(r.PathValue("id"))
			if err != nil || strings.TrimSpace(sessionID) == "" {
				http.Error(w, "invalid session id", http.StatusBadRequest)
				return
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")

		ch := defaultSessionHub.subscribe(sessionID)
		defer defaultSessionHub.unsubscribe(ch)

		if sessionID != "" {
			fmt.Fprintf(w, "event: connected\ndata: {\"status\":\"connected\",\"session_id\":%q}\n\n", sessionID)
		} else {
			fmt.Fprintf(w, "event: connected\ndata: {\"status\":\"connected\"}\n\n")
		}
		defaultSessionHub.mu.RLock()
		lastCards := cloneSessionCards(defaultSessionHub.lastCards)
		lastHTML := defaultSessionHub.lastHTML
		defaultSessionHub.mu.RUnlock()
		if len(lastCards) > 0 || lastHTML != "" {
			cards := lastCards
			htmlPayload := lastHTML
			if sessionID != "" {
				filteredCards := make([]SessionCard, 0)
				for _, card := range lastCards {
					if card.ID == sessionID {
						filteredCards = append(filteredCards, card)
						break
					}
				}
				cards = filteredCards
				if len(filteredCards) > 0 {
					noColor := false
					if r != nil && r.URL != nil {
						noColor = r.URL.Query().Get("no_color") == "1"
					}
					if markup, err := renderSessionCardsHTML(filteredCards, time.Now(), noColor); err == nil {
						htmlPayload = markup
					} else {
						htmlPayload = ""
					}
				} else {
					htmlPayload = ""
				}
			}

			payload := map[string]any{"sessions": cards}
			if htmlPayload != "" {
				payload["html"] = htmlPayload
			} else if sessionID == "" {
				payload["html"] = lastHTML
			}
			if data, err := json.Marshal(payload); err == nil {
				fmt.Fprintf(w, "event: session_update\ndata: %s\n\n", data)
			}
		}
		flusher.Flush()

		if interval <= 0 {
			interval = getSSEHeartbeatInterval()
		}
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		ctx := r.Context()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if _, err := io.WriteString(w, ": ping\n\n"); err != nil {
					return
				}
				flusher.Flush()
			case msg, ok := <-ch:
				if !ok {
					return
				}
				if _, err := w.Write(msg); err != nil {
					return
				}
				flusher.Flush()
				ticker.Reset(interval)
			}
		}
	}
}

func installSessionRoutes(mux *http.ServeMux, source SessionSource) {
	installSessionRoutesWithStore(mux, source, nil)
}

func installSessionRoutesWithStore(mux *http.ServeMux, source SessionSource, s *store) {
	mux.HandleFunc("GET /api/sessions/events", handleSessionSSE(false))
	mux.HandleFunc("GET /v1/fak/sessions/events", handleSessionSSE(false))
	mux.HandleFunc("GET /v1/fak/sessions/{id}/events", handleSessionSSE(true))
	mux.HandleFunc("POST /v1/fak/sessions/{id}/events", handleSessionEventHook(source, s))
	mux.HandleFunc("POST /api/sessions/{id}/events", handleSessionEventHook(source, s))

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
		defaultSessionHub.mu.Lock()
		defaultSessionHub.lastCards = cloneSessionCards(cards)
		defaultSessionHub.lastHTML = markup
		defaultSessionHub.mu.Unlock()
		writeSessionJSON(w, http.StatusOK, map[string]any{"sessions": cards, "html": markup})
	})
	mux.HandleFunc("POST /api/sessions/{id}/controls/{action}", func(w http.ResponseWriter, r *http.Request) {
		if source == nil {
			writeSessionJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "session authority is not connected"})
			return
		}
		id, err := url.PathUnescape(r.PathValue("id"))
		if err != nil || strings.TrimSpace(id) == "" {
			writeSessionJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid session id"})
			return
		}
		action := r.PathValue("action")
		valid := false
		for _, candidate := range sessionActions {
			valid = valid || candidate == action
		}
		if !valid {
			writeSessionJSON(w, http.StatusNotFound, map[string]string{"error": "unknown session control"})
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
		var selected *SessionCard
		for i := range cards {
			if cards[i].ID == id {
				selected = &cards[i]
				break
			}
		}
		if selected == nil {
			writeSessionJSON(w, http.StatusNotFound, map[string]string{"error": "logical session not found"})
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
		broadcastCards(source)
		writeSessionJSON(w, http.StatusAccepted, SessionControlRequest{SessionID: id, Action: action})
	})
	mux.HandleFunc("POST /api/sessions/{id}/approval", handleSessionApproval(source, s))
}

func writeSessionJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
