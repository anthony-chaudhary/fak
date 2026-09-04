package codexsession

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// ApprovalPolicy is the fak capability-floor seam. Adapters must not present an
// allow button until this check has allowed the exact proposed scope.
type ApprovalPolicy func(ApprovalRequest) PolicyDecision

// PolicyDecision represents the capability-floor evaluation outcome, indicating whether
// an approval request may proceed along with reason classification and assessed risk level.
type PolicyDecision struct {
	Allow  bool
	Reason string
	Risk   string
}

// ApprovalRequest encapsulates contextual metadata for a pending action requiring authorization,
// including target thread, turn, item, workspace bounds, proposed summary, and blast-radius consequences.
type ApprovalRequest struct {
	ApprovalID  string
	Kind        string
	ThreadID    string
	TurnID      string
	ItemID      string
	RequestID   string
	Workspace   string
	Summary     string
	Scope       string
	Reason      string
	Consequence string
}

// ApprovalResolution conveys the authoritative decision for a pending approval request,
// validated against an active input lease and bound to an execution epoch and principal.
type ApprovalResolution struct {
	InputID    string
	ApprovalID string
	Decision   string
	Scope      string
	Principal  string
	LeaseValid bool
	Epoch      uint64
}

// ApprovalJournalEntry records an immutable audit trail entry capturing approval identity,
// kind, status, rationale, and underlying capability-floor decisions for compliance tracking.
type ApprovalJournalEntry struct {
	ApprovalID         string `json:"approval_id"`
	Kind               string `json:"kind"`
	Status             string `json:"status"`
	Reason             string `json:"reason"`
	Scope              string `json:"scope"`
	ThreadID           string `json:"thread_id"`
	TurnID             string `json:"turn_id"`
	ItemID             string `json:"item_id"`
	RequestID          string `json:"request_id"`
	FakCapabilityFloor string `json:"fak_capability_floor"`
	CodexSandboxPolicy string `json:"codex_sandbox_policy"`
}

type pendingApproval struct {
	request  ApprovalRequest
	rpcID    json.RawMessage
	deadline time.Time
	epoch    uint64
	policy   PolicyDecision
}

// Approval error conditions returned during approval validation and resolution.
var (
	// ErrApprovalUnknown indicates that no pending or historical approval matches the requested ID.
	ErrApprovalUnknown = errors.New("codexsession: unknown approval")
	// ErrApprovalDuplicate indicates that the approval or input lease identifier was already consumed.
	ErrApprovalDuplicate = errors.New("codexsession: approval already resolved")
	// ErrApprovalStale indicates that the approval deadline elapsed or the execution epoch changed.
	ErrApprovalStale = errors.New("codexsession: stale approval response")
	// ErrApprovalUnauthorized indicates that caller lacks a valid principal identity or input lease.
	ErrApprovalUnauthorized = errors.New("codexsession: approval requires a valid input lease and principal")
	// ErrApprovalScope indicates that the resolved permission scope exceeds the originally proposed bounds.
	ErrApprovalScope = errors.New("codexsession: approval scope exceeds proposed scope")
)

func approvalKind(method string) string {
	switch method {
	case "item/commandExecution/requestApproval":
		return "command"
	case "item/fileChange/requestApproval":
		return "patch"
	default:
		return "unknown"
	}
}

func (a *Adapter) handleApprovalRequest(m rpcMessage) error {
	kind := approvalKind(m.Method)
	var p struct {
		ThreadID   string `json:"threadId"`
		TurnID     string `json:"turnId"`
		ItemID     string `json:"itemId"`
		ApprovalID string `json:"approvalId"`
		RequestID  string `json:"requestId"`
		Reason     string `json:"reason"`
		Command    string `json:"command"`
		Cwd        string `json:"cwd"`
		GrantRoot  string `json:"grantRoot"`
	}
	if err := json.Unmarshal(m.Params, &p); err != nil {
		return a.rejectRPC(m.ID, "invalid_request", "malformed approval request")
	}
	rid := p.RequestID
	if rid == "" {
		rid = p.ApprovalID
	}
	if rid == "" {
		rid = string(m.ID)
	}
	aid := strings.Join([]string{p.ThreadID, p.TurnID, p.ItemID, rid}, ":")
	scope := p.Cwd
	summary := p.Command
	consequence := "run the proposed command once"
	if kind == "patch" {
		scope = p.GrantRoot
		summary = "apply proposed workspace patch"
		consequence = "apply the proposed patch once"
	}
	if scope == "" {
		scope = a.cfg.Workspace
	}
	req := ApprovalRequest{ApprovalID: aid, Kind: kind, ThreadID: p.ThreadID, TurnID: p.TurnID, ItemID: p.ItemID, RequestID: rid, Workspace: a.cfg.Workspace, Summary: summary, Scope: scope, Reason: p.Reason, Consequence: consequence}
	decision := PolicyDecision{Allow: false, Reason: "approval policy unavailable", Risk: "unknown"}
	if kind == "unknown" {
		decision.Reason = "unknown_request_kind"
	} else if a.cfg.ApprovalPolicy != nil {
		decision = a.cfg.ApprovalPolicy(req)
	}
	status := "pending"
	if !decision.Allow {
		status = "denied"
	}
	a.mu.Lock()
	if _, exists := a.pending[aid]; exists {
		a.mu.Unlock()
		return a.rejectRPC(m.ID, "duplicate_request", "duplicate approval identity")
	}
	if _, done := a.resolved[aid]; done {
		a.mu.Unlock()
		return a.rejectRPC(m.ID, "stale_request", "approval identity already consumed")
	}
	pa := pendingApproval{request: req, rpcID: append(json.RawMessage(nil), m.ID...), deadline: a.now().Add(a.cfg.ApprovalTimeout), epoch: a.epoch, policy: decision}
	if decision.Allow {
		a.pending[aid] = pa
	} else {
		a.resolved[aid] = struct{}{}
	}
	a.mu.Unlock()
	payload := harnesskit.ApprovalPayload{ApprovalID: aid, Prompt: req.Reason, Status: status, Scope: req.Scope, Kind: req.Kind, Summary: req.Summary, Workspace: req.Workspace, Risk: decision.Risk, Consequence: req.Consequence, PolicyReason: decision.Reason, FakCapabilityFloor: map[bool]string{true: "allow", false: "deny"}[decision.Allow], CodexSandboxPolicy: "codex sandbox/approval policy remains independently enforced"}
	if err := a.emit(harnesskit.EventApprovalRequested, aid, p.TurnID, payload); err != nil {
		return err
	}
	a.journal(req, status, decision.Reason)
	if !decision.Allow {
		return a.rejectRPC(m.ID, map[bool]string{true: "unknown_request_kind", false: "policy_denied"}[kind == "unknown"], decision.Reason)
	}
	return nil
}

// ResolveApproval evaluates and applies an external approval decision against pending requests,
// enforcing input lease validity, non-replayability, epoch freshness, and scope confinement.
// Precondition: resolution must specify non-empty input lease identifier, valid epoch, and principal identity.
// Postcondition: matching approval is consumed, removed from pending queue, and marked resolved permanently.
func (a *Adapter) ResolveApproval(r ApprovalResolution) error {
	if r.InputID == "" {
		return ErrApprovalUnauthorized
	}
	a.mu.Lock()
	if _, ok := a.inputIDs[r.InputID]; ok {
		a.mu.Unlock()
		return ErrApprovalDuplicate
	}
	pa, ok := a.pending[r.ApprovalID]
	if !ok {
		_, done := a.resolved[r.ApprovalID]
		a.mu.Unlock()
		if done {
			return ErrApprovalDuplicate
		}
		return ErrApprovalUnknown
	}
	if r.Epoch != pa.epoch || a.epoch != pa.epoch || !a.now().Before(pa.deadline) {
		delete(a.pending, r.ApprovalID)
		a.resolved[r.ApprovalID] = struct{}{}
		a.mu.Unlock()
		_ = a.respond(pa.rpcID, "decline", "stale_or_expired")
		a.resolvedEvent(pa, "expired", "stale_or_expired")
		return ErrApprovalStale
	}
	if !r.LeaseValid || r.Principal == "" {
		a.mu.Unlock()
		return ErrApprovalUnauthorized
	}
	if r.Scope != "" && r.Scope != pa.request.Scope {
		a.mu.Unlock()
		return ErrApprovalScope
	}
	if r.Decision != "approve" && r.Decision != "deny" {
		a.mu.Unlock()
		return fmt.Errorf("%w: invalid decision", ErrApprovalUnknown)
	}
	delete(a.pending, r.ApprovalID)
	a.resolved[r.ApprovalID] = struct{}{}
	a.inputIDs[r.InputID] = struct{}{}
	a.mu.Unlock()
	codexDecision := "decline"
	status := "denied"
	if r.Decision == "approve" {
		codexDecision = "accept"
		status = "approved"
	}
	if err := a.respond(pa.rpcID, codexDecision, ""); err != nil {
		a.resolvedEvent(pa, "denied", "adapter_crash")
		return err
	}
	a.resolvedEvent(pa, status, "human_decision")
	return nil
}

// ExpireApprovals scans all pending approval requests and automatically declines any that have
// exceeded their configured deadline, transitioning their state to expired and emitting audit events.
// Precondition: adapter mutex protects concurrent iteration and removal of expired pending records.
// Postcondition: any approval whose deadline is before current timestamp is declined and recorded expired.
func (a *Adapter) ExpireApprovals() {
	now := a.now()
	var expired []pendingApproval
	a.mu.Lock()
	for id, p := range a.pending {
		if !now.Before(p.deadline) {
			delete(a.pending, id)
			a.resolved[id] = struct{}{}
			expired = append(expired, p)
		}
	}
	a.mu.Unlock()
	for _, p := range expired {
		_ = a.respond(p.rpcID, "decline", "timeout")
		a.resolvedEvent(p, "expired", "timeout")
	}
}

func (a *Adapter) failPending(reason string) {
	var ps []pendingApproval
	a.mu.Lock()
	for id, p := range a.pending {
		delete(a.pending, id)
		a.resolved[id] = struct{}{}
		ps = append(ps, p)
	}
	a.epoch++
	a.mu.Unlock()
	for _, p := range ps {
		_ = a.respond(p.rpcID, "decline", reason)
		a.resolvedEvent(p, "denied", reason)
	}
}
func (a *Adapter) resolvedEvent(p pendingApproval, status, reason string) {
	_ = a.emit(harnesskit.EventApprovalResolved, p.request.ApprovalID, p.request.TurnID, harnesskit.ApprovalPayload{ApprovalID: p.request.ApprovalID, Status: status, Scope: p.request.Scope, Kind: p.request.Kind, Summary: p.request.Summary, Workspace: p.request.Workspace, Risk: p.policy.Risk, Consequence: p.request.Consequence, PolicyReason: reason, FakCapabilityFloor: "allow", CodexSandboxPolicy: "codex sandbox/approval policy remains independently enforced"})
	a.journal(p.request, status, reason)
}
func (a *Adapter) journal(r ApprovalRequest, status, reason string) {
	if a.cfg.ApprovalJournal != nil {
		a.cfg.ApprovalJournal(ApprovalJournalEntry{ApprovalID: r.ApprovalID, Kind: r.Kind, Status: status, Reason: reason, Scope: r.Scope, ThreadID: r.ThreadID, TurnID: r.TurnID, ItemID: r.ItemID, RequestID: r.RequestID, FakCapabilityFloor: "additional capability floor", CodexSandboxPolicy: "independent Codex sandbox/approval policy"})
	}
}
func (a *Adapter) respond(id json.RawMessage, decision, reason string) error {
	result := map[string]any{"decision": decision}
	if reason != "" {
		result["reason"] = reason
	}
	return a.writeRPC(map[string]any{"jsonrpc": "2.0", "id": id, "result": result})
}
func (a *Adapter) rejectRPC(id json.RawMessage, code, message string) error {
	if err := a.writeRPC(map[string]any{"jsonrpc": "2.0", "id": id, "result": map[string]any{"decision": "decline", "reason": code}}); err != nil {
		return err
	}
	return nil
}
func (a *Adapter) writeRPC(v any) error {
	a.writeMu.Lock()
	defer a.writeMu.Unlock()
	a.mu.Lock()
	w := a.stdin
	a.mu.Unlock()
	if w == nil {
		return errors.New("codexsession: adapter disconnected")
	}
	return json.NewEncoder(w).Encode(v)
}
