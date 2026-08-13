package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

const sessionMoveSchema = "fak.session-move.v1"

type SessionMovePhase string

const (
	SessionMoveSafePointRequested  SessionMovePhase = "SAFE_POINT_REQUESTED"
	SessionMoveCheckpointed        SessionMovePhase = "CHECKPOINTED"
	SessionMoveDestinationAdmitted SessionMovePhase = "DESTINATION_ADMITTED"
	SessionMoveRestored            SessionMovePhase = "RESTORED"
	SessionMoveCutoverCommitted    SessionMovePhase = "CUTOVER_COMMITTED"
)

// SessionPlacement names replaceable runtime coordinates. AccountRef is a
// destination-resolved reference: credentials and provider tokens never enter
// this structure, a checkpoint, or a movement event.
type SessionPlacement struct {
	Provider             string   `json:"provider"`
	AccountRef           string   `json:"account_ref"`
	Model                string   `json:"model"`
	Compute              string   `json:"compute"`
	Capabilities         []string `json:"capabilities,omitempty"`
	ContextLimit         int64    `json:"context_limit,omitempty"`
	BudgetAvailable      int64    `json:"budget_available,omitempty"`
	ComputeAvailable     bool     `json:"compute_available"`
	CacheLineage         string   `json:"cache_lineage,omitempty"`
	SemanticDegradations []string `json:"semantic_degradations,omitempty"`
}

type SessionMoveRequest struct {
	Schema          string           `json:"schema,omitempty"`
	ExecutionEpoch  string           `json:"execution_epoch"`
	Destination     SessionPlacement `json:"destination"`
	RequiredCaps    []string         `json:"required_capabilities,omitempty"`
	RequiredContext int64            `json:"required_context,omitempty"`
	RequiredBudget  int64            `json:"required_budget,omitempty"`
}

type SessionMoveTransition struct {
	Phase          SessionMovePhase `json:"phase"`
	SessionID      string           `json:"session_id"`
	SourceEpoch    string           `json:"source_epoch"`
	Destination    SessionPlacement `json:"destination"`
	CheckpointHash string           `json:"checkpoint_hash,omitempty"`
	At             time.Time        `json:"at"`
}

type SessionMoveCheckpoint struct {
	Schema      string           `json:"schema"`
	SessionID   string           `json:"session_id"`
	SourceEpoch string           `json:"source_epoch"`
	EventHead   uint64           `json:"event_head"`
	Source      SessionPlacement `json:"source"`
	Destination SessionPlacement `json:"destination"`
	Terminal    []byte           `json:"terminal,omitempty"`
	Effects     []SessionEffect  `json:"effects,omitempty"`
	CreatedAt   time.Time        `json:"created_at"`
	Digest      string           `json:"digest"`
}

type SessionMoveDelta struct {
	CapabilityAdded      []string `json:"capability_added,omitempty"`
	CapabilityRemoved    []string `json:"capability_removed,omitempty"`
	CacheLineageChanged  bool     `json:"cache_lineage_changed"`
	SemanticDegradations []string `json:"semantic_degradations,omitempty"`
}

type SessionMoveResponse struct {
	Schema      string                  `json:"schema"`
	Descriptor  SessionClientDescriptor `json:"descriptor"`
	Source      SessionPlacement        `json:"source"`
	Destination SessionPlacement        `json:"destination"`
	Transitions []SessionMoveTransition `json:"transitions"`
	Delta       SessionMoveDelta        `json:"delta"`
}

// SessionMoveHooks connect the protocol state machine to provider/account/model
// routing and compute placement. Nil admission/restore hooks fail closed.
type SessionMoveHooks struct {
	RequestSafePoint    func(context.Context, string) error
	AdmitDestination    func(context.Context, string, SessionMoveCheckpoint, SessionMoveRequest) error
	RestoreDestination  func(context.Context, SessionMoveCheckpoint) error
	CommitDestination   func(context.Context, SessionMoveCheckpoint) error
	RollbackDestination func(context.Context, SessionMoveCheckpoint) error
	RecordTransition    func(context.Context, SessionMoveTransition) error
}

type sessionMoveState struct {
	placement SessionPlacement
	last      []SessionMoveTransition
	moving    bool
}

type SessionMoveError struct {
	Code        string
	Message     string
	Status      int
	Transitions []SessionMoveTransition
}

func (e *SessionMoveError) Error() string { return e.Message }

// SessionMoveJournalHooks persists every movement transition through the
// crash-safe work-session journal. The destination carries references and
// capability disclosures only; no credential material is accepted.
func SessionMoveJournalHooks(path, writerEpoch string) func(context.Context, SessionMoveTransition) error {
	return func(_ context.Context, transition SessionMoveTransition) error {
		destination := sessionjournal.PlacementIdentity{Provider: transition.Destination.Provider, AccountRef: transition.Destination.AccountRef, Model: transition.Destination.Model, Compute: transition.Destination.Compute, Capabilities: append([]string(nil), transition.Destination.Capabilities...), CacheLineage: transition.Destination.CacheLineage, SemanticDegradations: append([]string(nil), transition.Destination.SemanticDegradations...)}
		return sessionjournal.AppendWorkEvent(path, sessionjournal.WorkEvent{SessionID: transition.SessionID, Kind: sessionjournal.WorkMoveTransitionEvent, WriterEpoch: writerEpoch, MovePhase: string(transition.Phase), SourceEpoch: transition.SourceEpoch, Destination: &destination, Checkpoint: transition.CheckpointHash})
	}
}

// ConfigureSessionMove installs the source placement and destination adapters for one logical session.
func (s *Server) ConfigureSessionMove(sessionID string, placement SessionPlacement, hooks SessionMoveHooks) error {
	if s == nil || strings.TrimSpace(sessionID) == "" {
		return errors.New("session move configuration requires session_id")
	}
	if err := validatePlacement(placement, nil, 0, 0); err != nil {
		return err
	}
	rt := s.clientRuntime()
	rt.mu.Lock()
	defer rt.mu.Unlock()
	sess := rt.sessionLocked(sessionID)
	sess.placement = normalizedPlacement(placement)
	sess.moveHooks = hooks
	return nil
}

// MoveSession performs a fail-closed, journalable epoch cutover while preserving logical identity.
func (s *Server) MoveSession(ctx context.Context, sessionID string, req SessionMoveRequest) (SessionMoveResponse, error) {
	if s == nil || strings.TrimSpace(sessionID) == "" {
		return SessionMoveResponse{}, moveError(http.StatusBadRequest, "MOVE_INVALID", "session_id is required", nil)
	}
	if err := validatePlacement(req.Destination, req.RequiredCaps, req.RequiredContext, req.RequiredBudget); err != nil {
		return SessionMoveResponse{}, moveError(http.StatusUnprocessableEntity, "DESTINATION_REJECTED", err.Error(), nil)
	}
	rt := s.clientRuntime()
	rt.mu.Lock()
	sess := rt.sessionLocked(sessionID)
	if sess.moving {
		rt.mu.Unlock()
		return SessionMoveResponse{}, moveError(http.StatusConflict, "MOVE_IN_PROGRESS", "another movement owns this logical session", nil)
	}
	sourceEpoch := sess.executionEpoch
	if req.ExecutionEpoch != sourceEpoch {
		rt.mu.Unlock()
		return SessionMoveResponse{}, moveError(http.StatusConflict, "STALE_EPOCH", "session execution epoch changed; describe and retry", nil)
	}
	sess.moving = true
	source := clonePlacement(sess.placement)
	hooks := sess.moveHooks
	terminal := append([]byte(nil), sess.transcript...)
	effects := append([]SessionEffect(nil), effectsFromMap(sess.effects)...)
	rt.mu.Unlock()

	committed := false
	defer func() {
		if committed {
			return
		}
		rt.mu.Lock()
		if current := rt.sessions[sessionID]; current != nil && current.executionEpoch == sourceEpoch {
			current.moving = false
		}
		rt.mu.Unlock()
	}()

	var transitions []SessionMoveTransition
	record := func(phase SessionMovePhase, digest string) error {
		t := SessionMoveTransition{Phase: phase, SessionID: sessionID, SourceEpoch: sourceEpoch, Destination: publicPlacement(req.Destination), CheckpointHash: digest, At: time.Now().UTC()}
		if hooks.RecordTransition != nil {
			if err := hooks.RecordTransition(ctx, t); err != nil {
				return err
			}
		}
		transitions = append(transitions, t)
		return nil
	}
	if err := record(SessionMoveSafePointRequested, ""); err != nil {
		return SessionMoveResponse{}, moveError(http.StatusServiceUnavailable, "MOVE_JOURNAL_FAILED", err.Error(), transitions)
	}
	state := s.observeSession(ctx, sessionID)
	if hooks.RequestSafePoint == nil {
		if strings.EqualFold(state.Run, "RUNNING") || strings.EqualFold(state.Run, "STARTING") {
			return SessionMoveResponse{}, moveError(http.StatusConflict, "MOVE_UNSAFE", "active turn has no safe-point provider", transitions)
		}
	} else if err := hooks.RequestSafePoint(ctx, sessionID); err != nil {
		return SessionMoveResponse{}, moveError(http.StatusConflict, "MOVE_UNSAFE", err.Error(), transitions)
	}
	_, eventHead := s.sessionChangesFor(sessionID, 0)
	checkpoint := SessionMoveCheckpoint{Schema: sessionMoveSchema, SessionID: sessionID, SourceEpoch: sourceEpoch, EventHead: eventHead, Source: publicPlacement(source), Destination: publicPlacement(req.Destination), Terminal: terminal, Effects: effects, CreatedAt: time.Now().UTC()}
	checkpoint.Digest = checkpointDigest(checkpoint)
	if err := record(SessionMoveCheckpointed, checkpoint.Digest); err != nil {
		return SessionMoveResponse{}, moveError(http.StatusServiceUnavailable, "MOVE_JOURNAL_FAILED", err.Error(), transitions)
	}
	if hooks.AdmitDestination == nil {
		return SessionMoveResponse{}, moveError(http.StatusNotImplemented, "DESTINATION_ADMISSION_UNAVAILABLE", "destination admission is not configured", transitions)
	}
	if err := hooks.AdmitDestination(ctx, sessionID, checkpoint, req); err != nil {
		return SessionMoveResponse{}, moveError(http.StatusUnprocessableEntity, "DESTINATION_REJECTED", err.Error(), transitions)
	}
	if err := record(SessionMoveDestinationAdmitted, checkpoint.Digest); err != nil {
		return SessionMoveResponse{}, moveError(http.StatusServiceUnavailable, "MOVE_JOURNAL_FAILED", err.Error(), transitions)
	}
	if hooks.RestoreDestination == nil {
		return SessionMoveResponse{}, moveError(http.StatusNotImplemented, "DESTINATION_RESTORE_UNAVAILABLE", "destination restore is not configured", transitions)
	}
	if err := hooks.RestoreDestination(ctx, checkpoint); err != nil {
		rollbackMove(ctx, hooks, checkpoint)
		return SessionMoveResponse{}, moveError(http.StatusBadGateway, "DESTINATION_RESTORE_FAILED", err.Error(), transitions)
	}
	if err := record(SessionMoveRestored, checkpoint.Digest); err != nil {
		rollbackMove(ctx, hooks, checkpoint)
		return SessionMoveResponse{}, moveError(http.StatusServiceUnavailable, "MOVE_JOURNAL_FAILED", err.Error(), transitions)
	}
	if hooks.CommitDestination != nil {
		if err := hooks.CommitDestination(ctx, checkpoint); err != nil {
			rollbackMove(ctx, hooks, checkpoint)
			return SessionMoveResponse{}, moveError(http.StatusBadGateway, "CUTOVER_FAILED", err.Error(), transitions)
		}
	}

	newEpoch := moveEpoch(sessionID, sourceEpoch, checkpoint.Digest)
	rt.mu.Lock()
	sess = rt.sessions[sessionID]
	if sess == nil || sess.executionEpoch != sourceEpoch || !sess.moving {
		rt.mu.Unlock()
		rollbackMove(ctx, hooks, checkpoint)
		return SessionMoveResponse{}, moveError(http.StatusConflict, "STALE_EPOCH", "source epoch changed before cutover", transitions)
	}
	sess.executionEpoch = newEpoch
	sess.placement = normalizedPlacement(req.Destination)
	sess.moving = false
	for id, att := range rt.attachments {
		if att.SessionID == sessionID {
			delete(rt.attachments, id)
		}
	}
	delete(rt.leases, sessionID)
	rt.mu.Unlock()
	if err := record(SessionMoveCutoverCommitted, checkpoint.Digest); err != nil {
		// The destination is now authoritative. Preserve the committed epoch and
		// return a typed post-cutover recovery condition rather than reviving source.
		committed = true
		return SessionMoveResponse{}, moveError(http.StatusServiceUnavailable, "CUTOVER_JOURNAL_RECOVERY_REQUIRED", err.Error(), transitions)
	}
	committed = true
	rt.mu.Lock()
	sess.lastMove = append([]SessionMoveTransition(nil), transitions...)
	rt.mu.Unlock()
	desc, _ := s.sessionClientDescriptorForContext(ctx, sessionID)
	return SessionMoveResponse{Schema: sessionMoveSchema, Descriptor: desc, Source: publicPlacement(source), Destination: publicPlacement(req.Destination), Transitions: transitions, Delta: placementDelta(source, req.Destination)}, nil
}

func rollbackMove(ctx context.Context, hooks SessionMoveHooks, checkpoint SessionMoveCheckpoint) {
	if hooks.RollbackDestination != nil {
		_ = hooks.RollbackDestination(ctx, checkpoint)
	}
}

func moveError(status int, code, message string, transitions []SessionMoveTransition) *SessionMoveError {
	return &SessionMoveError{Status: status, Code: code, Message: message, Transitions: append([]SessionMoveTransition(nil), transitions...)}
}

func validatePlacement(p SessionPlacement, required []string, contextLimit, budget int64) error {
	if strings.TrimSpace(p.Provider) == "" || strings.TrimSpace(p.AccountRef) == "" || strings.TrimSpace(p.Model) == "" || strings.TrimSpace(p.Compute) == "" {
		return errors.New("provider, account_ref, model, and compute are required")
	}
	if !p.ComputeAvailable {
		return errors.New("compute is unavailable")
	}
	if contextLimit > 0 && p.ContextLimit < contextLimit {
		return fmt.Errorf("context limit %d is below required %d", p.ContextLimit, contextLimit)
	}
	if budget > 0 && p.BudgetAvailable < budget {
		return fmt.Errorf("budget %d is below required %d", p.BudgetAvailable, budget)
	}
	have := map[string]bool{}
	for _, cap := range p.Capabilities {
		have[strings.TrimSpace(cap)] = true
	}
	for _, cap := range required {
		if !have[strings.TrimSpace(cap)] {
			return fmt.Errorf("required capability %q is unavailable", cap)
		}
	}
	return nil
}

func normalizedPlacement(p SessionPlacement) SessionPlacement {
	p.Provider, p.AccountRef, p.Model, p.Compute = strings.TrimSpace(p.Provider), strings.TrimSpace(p.AccountRef), strings.TrimSpace(p.Model), strings.TrimSpace(p.Compute)
	p.Capabilities = sortedUnique(p.Capabilities)
	p.SemanticDegradations = sortedUnique(p.SemanticDegradations)
	return p
}
func clonePlacement(p SessionPlacement) SessionPlacement {
	p.Capabilities = append([]string(nil), p.Capabilities...)
	p.SemanticDegradations = append([]string(nil), p.SemanticDegradations...)
	return p
}
func publicPlacement(p SessionPlacement) SessionPlacement {
	return clonePlacement(normalizedPlacement(p))
}
func sortedUnique(in []string) []string {
	m := map[string]bool{}
	for _, v := range in {
		if v = strings.TrimSpace(v); v != "" {
			m[v] = true
		}
	}
	out := make([]string, 0, len(m))
	for v := range m {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}
func checkpointDigest(c SessionMoveCheckpoint) string {
	c.Digest = ""
	b, _ := json.Marshal(c)
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}
func moveEpoch(sessionID, sourceEpoch, digest string) string {
	sum := sha256.Sum256([]byte(sessionID + "\x00" + sourceEpoch + "\x00" + digest + "\x00" + time.Now().UTC().Format(time.RFC3339Nano)))
	return "epoch-" + hex.EncodeToString(sum[:8])
}
func placementDelta(a, b SessionPlacement) SessionMoveDelta {
	aa, bb := map[string]bool{}, map[string]bool{}
	for _, v := range a.Capabilities {
		aa[v] = true
	}
	for _, v := range b.Capabilities {
		bb[v] = true
	}
	d := SessionMoveDelta{CacheLineageChanged: a.CacheLineage != b.CacheLineage, SemanticDegradations: sortedUnique(b.SemanticDegradations)}
	for v := range bb {
		if !aa[v] {
			d.CapabilityAdded = append(d.CapabilityAdded, v)
		}
	}
	for v := range aa {
		if !bb[v] {
			d.CapabilityRemoved = append(d.CapabilityRemoved, v)
		}
	}
	sort.Strings(d.CapabilityAdded)
	sort.Strings(d.CapabilityRemoved)
	return d
}

func (s *Server) handleSessionMove(w http.ResponseWriter, r *http.Request, sessionID string) bool {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST /v1/fak/session/{id}/move")
		return true
	}
	var req SessionMoveRequest
	if !decodeSessionClientJSON(w, r, &req) {
		return true
	}
	resp, err := s.MoveSession(r.Context(), sessionID, req)
	if err != nil {
		var me *SessionMoveError
		if errors.As(err, &me) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(me.Status)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": me.Code, "message": me.Message, "transitions": me.Transitions})
			return true
		}
		writeErrCode(w, http.StatusInternalServerError, "MOVE_FAILED", err.Error())
		return true
	}
	writeJSON(w, http.StatusOK, resp)
	return true
}

func effectsFromMap(in map[string]SessionEffect) []SessionEffect {
	out := make([]SessionEffect, 0, len(in))
	for _, effect := range in {
		out = append(out, effect)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}
