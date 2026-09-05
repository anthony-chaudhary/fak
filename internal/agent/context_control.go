package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// context_control.go — bounded agent context controls and queries for the native agent
// harness (issue #8583).
//
// Exposes the context_control tool so the model can inspect, query, pin, release, retune
// retrieval scope, express advisory budget preferences, and request restoration of elided
// or released context spans under strict kernel safety and capability bounds.
//
// Mandatory safety/capability policy (system prompts, root goals, security boundaries,
// hard budget ceilings) is strictly decoupled from advisory model preferences. Releasing
// or overriding mandatory invariants is refused with closed refusal tokens.

const (
	// ToolContextControl is the planner/model-facing tool name.
	ToolContextControl = "context_control"

	// EngineContextControl is the registered kernel engine ID.
	EngineContextControl = "agent.context_control"

	// RungNameContextControl is the adjudicator link name.
	RungNameContextControl = "context_control"

	// contextControlRank places the gate after cheap syntax rungs and coding tools,
	// before the default monitor.
	contextControlRank = 23

	// Typed context control operations.
	ActionInspect          = "inspect"
	ActionQuery            = "query"
	ActionPin              = "pin"
	ActionRelease          = "release"
	ActionRetrievalScope   = "retrieval_scope"
	ActionBudgetPreference = "budget_preference"
	ActionRestore          = "restore"

	// Status outcomes for ContextControlReceipt.
	StatusAccepted = "accepted"
	StatusRefused  = "refused"
	StatusError    = "error"

	// Default envelope bounds.
	DefaultMinBudgetTokens = 512
	DefaultMaxBudgetTokens = 128000
	DefaultMaxHorizonTurns = 50
	DefaultMaxTTLSeconds   = 86400
	DefaultSpanTokens      = 250
)

// SupportedControlKinds lists the supported typed context control operations.
var SupportedControlKinds = []string{
	ActionInspect,
	ActionQuery,
	ActionPin,
	ActionRelease,
	ActionRetrievalScope,
	ActionBudgetPreference,
	ActionRestore,
}

// MalformedRequestClasses enumerates the recognized classes of malformed or refused requests.
var MalformedRequestClasses = []string{
	"unknown_action",
	"over_budget",
	"missing_required_params",
	"negative_budget",
	"invalid_digest",
	"policy_protected_mutation",
}

// CountTypedControlKinds returns the number of supported typed control kinds.
func CountTypedControlKinds() int {
	return len(SupportedControlKinds)
}

// CountMalformedRequestClasses returns the number of recognized malformed request classes.
func CountMalformedRequestClasses() int {
	return len(MalformedRequestClasses)
}

// ContextControlRequest represents incoming arguments for the context_control tool.
type ContextControlRequest struct {
	Action         string   `json:"action"`
	SpanIDs        []string `json:"span_ids,omitempty"`
	Digest         string   `json:"digest,omitempty"`
	Budget         *int     `json:"budget,omitempty"`
	Query          string   `json:"query,omitempty"`
	IdempotencyKey string   `json:"idempotency_key,omitempty"`
	Scope          string   `json:"scope,omitempty"`
	Horizon        int      `json:"horizon,omitempty"`
	TTLSeconds     int      `json:"ttl_seconds,omitempty"`
	Provenance     string   `json:"provenance,omitempty"`
}

// ContextControlReceipt is the typed receipt returned by every context_control operation.
type ContextControlReceipt struct {
	Status     string         `json:"status"`            // "accepted" | "refused" | "error"
	Action     string         `json:"action"`            // requested or executed action
	Reason     string         `json:"reason,omitempty"`  // closed refusal token e.g. MALFORMED, OVERSIZE, POLICY_BLOCK
	Idempotent bool           `json:"idempotent"`        // true if returned from an idempotent replayed request
	Details    map[string]any `json:"details,omitempty"` // structured operation details
}

// MandatoryContextPolicy defines immutable safety and capability constraints enforced by the kernel.
type MandatoryContextPolicy struct {
	MaxBudgetTokens   int      `json:"max_budget_tokens"`
	MinBudgetTokens   int      `json:"min_budget_tokens"`
	MaxHorizonTurns   int      `json:"max_horizon_turns"`
	MaxTTLSeconds     int      `json:"max_ttl_seconds"`
	ProtectedSpanIDs  []string `json:"protected_span_ids"`
	ProtectedPrefixes []string `json:"protected_prefixes"`
	AllowedScopes     []string `json:"allowed_scopes"`
}

// DefaultMandatoryContextPolicy returns standard kernel safety limits.
func DefaultMandatoryContextPolicy() MandatoryContextPolicy {
	return MandatoryContextPolicy{
		MaxBudgetTokens:   DefaultMaxBudgetTokens,
		MinBudgetTokens:   DefaultMinBudgetTokens,
		MaxHorizonTurns:   DefaultMaxHorizonTurns,
		MaxTTLSeconds:     DefaultMaxTTLSeconds,
		ProtectedSpanIDs:  []string{"sys:prompt", "sys:boundary", "goal:root", "safety:floor"},
		ProtectedPrefixes: []string{"sys:", "safety:", "policy:"},
		AllowedScopes:     []string{"all", "repo", "evidence", "turn", "session", "tools", "recent", "code", "docs"},
	}
}

// AdvisoryContextSettings records agent-tunable preferences within mandatory policy boundaries.
type AdvisoryContextSettings struct {
	PreferredBudget *int   `json:"preferred_budget,omitempty"`
	RetrievalScope  string `json:"retrieval_scope,omitempty"`
	Horizon         int    `json:"horizon,omitempty"`
	TTLSeconds      int    `json:"ttl_seconds,omitempty"`
}

// PinnedSpan tracks an active pinned context span.
type PinnedSpan struct {
	SpanID     string    `json:"span_id"`
	Digest     string    `json:"digest,omitempty"`
	Provenance string    `json:"provenance,omitempty"`
	PinnedAt   time.Time `json:"pinned_at"`
	ExpiresAt  time.Time `json:"expires_at,omitempty"`
	Horizon    int       `json:"horizon,omitempty"`
	Tokens     int       `json:"tokens,omitempty"`
}

// ReleasedSpan records a span explicitly released by the agent.
type ReleasedSpan struct {
	SpanID     string    `json:"span_id"`
	ReleasedAt time.Time `json:"released_at"`
}

type idempotencyRecord struct {
	reqHash string
	receipt ContextControlReceipt
}

// ContextControlSnapshot represents an exported read-only view of context control state.
type ContextControlSnapshot struct {
	ActivePins      []string                `json:"active_pins"`
	ActiveReleases  []string                `json:"active_releases"`
	PreferredBudget *int                    `json:"preferred_budget,omitempty"`
	RetrievalScope  string                  `json:"retrieval_scope,omitempty"`
	Horizon         int                     `json:"horizon,omitempty"`
	EstimatedTokens int                     `json:"estimated_tokens"`
	MandatoryPolicy MandatoryContextPolicy  `json:"mandatory_policy"`
	Advisory        AdvisoryContextSettings `json:"advisory"`
}

// NativeContextControlAdapter provides native agent harness integration.
type NativeContextControlAdapter interface {
	Name() string
	ApplyToTurn(ctx context.Context, turn int, planner *CtxViewPlanner) error
	Snapshot() ContextControlSnapshot
}

type nativeTurnLoopAdapter struct {
	state *ContextControlState
}

func (a *nativeTurnLoopAdapter) Name() string {
	return "native_turn_loop"
}

func (a *nativeTurnLoopAdapter) ApplyToTurn(ctx context.Context, turn int, planner *CtxViewPlanner) error {
	if a == nil || a.state == nil || planner == nil {
		return nil
	}
	a.state.mu.RLock()
	pref := a.state.advisory.PreferredBudget
	a.state.mu.RUnlock()
	if pref != nil && *pref > 0 {
		planner.Budget = *pref
	}
	return nil
}

func (a *nativeTurnLoopAdapter) Snapshot() ContextControlSnapshot {
	if a == nil || a.state == nil {
		return ContextControlSnapshot{}
	}
	return a.state.Snapshot()
}

// ContextControlState holds thread-safe context control and query state for a session.
type ContextControlState struct {
	mu              sync.RWMutex
	policy          MandatoryContextPolicy
	advisory        AdvisoryContextSettings
	pinned          map[string]PinnedSpan
	released        map[string]ReleasedSpan
	restored        map[string]time.Time
	knownDigests    map[string]string // digest -> span_id
	idempotency     map[string]idempotencyRecord
	adapters        []NativeContextControlAdapter
	estimatedTokens int
}

// ContextControlOption configures initialization of ContextControlState.
type ContextControlOption func(*ContextControlState)

// WithMandatoryPolicy overrides the default mandatory safety policy.
func WithMandatoryPolicy(p MandatoryContextPolicy) ContextControlOption {
	return func(s *ContextControlState) {
		s.policy = p
	}
}

// WithInitialBudget configures an initial preferred budget.
func WithInitialBudget(tokens int) ContextControlOption {
	return func(s *ContextControlState) {
		s.advisory.PreferredBudget = &tokens
	}
}

// WithKnownDigest seeds a known provenance digest and associated span ID.
func WithKnownDigest(digest, spanID string) ContextControlOption {
	return func(s *ContextControlState) {
		s.knownDigests[digest] = spanID
	}
}

// WithHarnessAdapter attaches a custom native harness adapter.
func WithHarnessAdapter(a NativeContextControlAdapter) ContextControlOption {
	return func(s *ContextControlState) {
		if a != nil {
			s.adapters = append(s.adapters, a)
		}
	}
}

// NewContextControlState constructs a new initialized ContextControlState.
func NewContextControlState(opts ...ContextControlOption) *ContextControlState {
	st := &ContextControlState{
		policy:       DefaultMandatoryContextPolicy(),
		pinned:       make(map[string]PinnedSpan),
		released:     make(map[string]ReleasedSpan),
		restored:     make(map[string]time.Time),
		knownDigests: make(map[string]string),
		idempotency:  make(map[string]idempotencyRecord),
	}
	st.advisory = AdvisoryContextSettings{
		RetrievalScope: "all",
		Horizon:        1,
	}
	st.adapters = []NativeContextControlAdapter{&nativeTurnLoopAdapter{state: st}}

	for _, opt := range opts {
		if opt != nil {
			opt(st)
		}
	}
	return st
}

// CountNativeHarnessAdapters returns the count of registered native harness adapters.
func (s *ContextControlState) CountNativeHarnessAdapters() int {
	if s == nil {
		return 1
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.adapters) == 0 {
		return 1
	}
	return len(s.adapters)
}

// CountNativeHarnessAdapters returns the adapter count for the active context control state.
func CountNativeHarnessAdapters() int {
	st := armedContextControl.Load()
	if st == nil {
		return 1
	}
	return st.CountNativeHarnessAdapters()
}

// Snapshot returns a point-in-time snapshot of the context control state.
func (s *ContextControlState) Snapshot() ContextControlSnapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()

	pins := make([]string, 0, len(s.pinned))
	for id := range s.pinned {
		pins = append(pins, id)
	}
	sort.Strings(pins)

	rels := make([]string, 0, len(s.released))
	for id := range s.released {
		rels = append(rels, id)
	}
	sort.Strings(rels)

	var budgetCopy *int
	if s.advisory.PreferredBudget != nil {
		v := *s.advisory.PreferredBudget
		budgetCopy = &v
	}

	return ContextControlSnapshot{
		ActivePins:      pins,
		ActiveReleases:  rels,
		PreferredBudget: budgetCopy,
		RetrievalScope:  s.advisory.RetrievalScope,
		Horizon:         s.advisory.Horizon,
		EstimatedTokens: s.estimatedTokens,
		MandatoryPolicy: s.policy,
		Advisory:        s.advisory,
	}
}

func hashRequest(req ContextControlRequest) string {
	h := sha256.New()
	h.Write([]byte(req.Action))
	h.Write([]byte{0})
	for _, s := range req.SpanIDs {
		h.Write([]byte(s))
		h.Write([]byte{0})
	}
	h.Write([]byte(req.Digest))
	h.Write([]byte{0})
	if req.Budget != nil {
		h.Write([]byte(strconv.Itoa(*req.Budget)))
	}
	h.Write([]byte{0})
	h.Write([]byte(req.Query))
	h.Write([]byte{0})
	h.Write([]byte(req.Scope))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(req.Horizon)))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(req.TTLSeconds)))
	h.Write([]byte{0})
	h.Write([]byte(req.Provenance))
	return hex.EncodeToString(h.Sum(nil))
}

func (s *ContextControlState) isProtectedSpanLocked(id string) bool {
	for _, p := range s.policy.ProtectedSpanIDs {
		if id == p {
			return true
		}
	}
	for _, prefix := range s.policy.ProtectedPrefixes {
		if strings.HasPrefix(id, prefix) {
			return true
		}
	}
	return false
}

func (s *ContextControlState) isAllowedScopeLocked(scope string) bool {
	if len(s.policy.AllowedScopes) == 0 {
		return true
	}
	for _, a := range s.policy.AllowedScopes {
		if strings.EqualFold(a, scope) {
			return true
		}
	}
	return false
}

func (s *ContextControlState) recalculateTokensLocked() {
	total := 0
	for _, p := range s.pinned {
		if p.Tokens > 0 {
			total += p.Tokens
		} else {
			total += DefaultSpanTokens
		}
	}
	s.estimatedTokens = total
}

// Execute handles a validated ContextControlRequest with concurrency safety and idempotency tracking.
func (s *ContextControlState) Execute(ctx context.Context, req ContextControlRequest) ContextControlReceipt {
	s.mu.Lock()
	defer s.mu.Unlock()

	reqHash := hashRequest(req)
	if req.IdempotencyKey != "" {
		if record, ok := s.idempotency[req.IdempotencyKey]; ok {
			if record.reqHash == reqHash {
				res := record.receipt
				res.Idempotent = true
				return res
			}
			return ContextControlReceipt{
				Status:     StatusRefused,
				Action:     req.Action,
				Reason:     abi.ReasonName(abi.ReasonMalformed),
				Idempotent: false,
				Details:    map[string]any{"error": "idempotency key reused with conflicting request parameters"},
			}
		}
	}

	action := strings.ToLower(strings.TrimSpace(req.Action))
	var receipt ContextControlReceipt

	switch action {
	case ActionInspect, "status":
		pins := make([]string, 0, len(s.pinned))
		for id := range s.pinned {
			pins = append(pins, id)
		}
		sort.Strings(pins)

		releases := make([]string, 0, len(s.released))
		for id := range s.released {
			releases = append(releases, id)
		}
		sort.Strings(releases)

		restored := make([]string, 0, len(s.restored))
		for id := range s.restored {
			restored = append(restored, id)
		}
		sort.Strings(restored)

		receipt = ContextControlReceipt{
			Status: StatusAccepted,
			Action: ActionInspect,
			Details: map[string]any{
				"active_pins":        pins,
				"active_releases":    releases,
				"restored_spans":     restored,
				"advisory_settings":  s.advisory,
				"mandatory_policy":   s.policy,
				"estimated_tokens":   s.estimatedTokens,
				"retrieval_scope":    s.advisory.RetrievalScope,
				"total_pinned_count": len(s.pinned),
			},
		}

	case ActionQuery:
		if req.Budget != nil && *req.Budget < 0 {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionQuery,
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": "query budget cannot be negative"},
			}
			break
		}

		matchedSpans := make([]string, 0)
		qLower := strings.ToLower(strings.TrimSpace(req.Query))
		for id, pin := range s.pinned {
			if qLower == "" || strings.Contains(strings.ToLower(id), qLower) || strings.Contains(strings.ToLower(pin.Provenance), qLower) {
				matchedSpans = append(matchedSpans, id)
			}
		}
		sort.Strings(matchedSpans)

		suggested := len(matchedSpans) * DefaultSpanTokens
		if suggested < s.policy.MinBudgetTokens {
			suggested = s.policy.MinBudgetTokens
		}
		if suggested > s.policy.MaxBudgetTokens {
			suggested = s.policy.MaxBudgetTokens
		}

		receipt = ContextControlReceipt{
			Status: StatusAccepted,
			Action: ActionQuery,
			Details: map[string]any{
				"query":            req.Query,
				"matched_spans":    matchedSpans,
				"matched_count":    len(matchedSpans),
				"suggested_budget": suggested,
				"retrieval_scope":  s.advisory.RetrievalScope,
			},
		}

	case ActionPin:
		if len(req.SpanIDs) == 0 {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionPin,
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": "span_ids parameter is required and cannot be empty for pin operation"},
			}
			break
		}

		if req.Digest != "" {
			cleanDigest := strings.TrimSpace(req.Digest)
			if len(cleanDigest) < 8 || strings.ContainsAny(cleanDigest, " \t\r\n") {
				receipt = ContextControlReceipt{
					Status:  StatusRefused,
					Action:  ActionPin,
					Reason:  abi.ReasonName(abi.ReasonMalformed),
					Details: map[string]any{"error": "invalid digest format: must be at least 8 non-whitespace characters"},
				}
				break
			}
		}

		if req.TTLSeconds < 0 {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionPin,
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": "ttl_seconds cannot be negative"},
			}
			break
		}
		if req.TTLSeconds > s.policy.MaxTTLSeconds {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionPin,
				Reason:  abi.ReasonName(abi.ReasonOversize),
				Details: map[string]any{"error": fmt.Sprintf("ttl_seconds %d exceeds maximum policy limit %d", req.TTLSeconds, s.policy.MaxTTLSeconds)},
			}
			break
		}

		if req.Horizon < 0 {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionPin,
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": "horizon cannot be negative"},
			}
			break
		}
		if req.Horizon > s.policy.MaxHorizonTurns {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionPin,
				Reason:  abi.ReasonName(abi.ReasonOversize),
				Details: map[string]any{"error": fmt.Sprintf("horizon %d exceeds maximum policy limit %d", req.Horizon, s.policy.MaxHorizonTurns)},
			}
			break
		}

		// Check budget ceiling
		currentTokens := s.estimatedTokens
		newTokens := currentTokens + len(req.SpanIDs)*DefaultSpanTokens
		if newTokens > s.policy.MaxBudgetTokens {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionPin,
				Reason:  abi.ReasonName(abi.ReasonOversize),
				Details: map[string]any{"error": fmt.Sprintf("pinning %d spans exceeds max budget ceiling (%d > %d)", len(req.SpanIDs), newTokens, s.policy.MaxBudgetTokens)},
			}
			break
		}

		pinnedIDs := make([]string, 0, len(req.SpanIDs))
		for _, id := range req.SpanIDs {
			trimmed := strings.TrimSpace(id)
			if trimmed == "" {
				continue
			}
			delete(s.released, trimmed)
			var expiresAt time.Time
			if req.TTLSeconds > 0 {
				expiresAt = time.Now().Add(time.Duration(req.TTLSeconds) * time.Second)
			}
			s.pinned[trimmed] = PinnedSpan{
				SpanID:     trimmed,
				Digest:     req.Digest,
				Provenance: req.Provenance,
				PinnedAt:   time.Now(),
				ExpiresAt:  expiresAt,
				Horizon:    req.Horizon,
				Tokens:     DefaultSpanTokens,
			}
			if req.Digest != "" {
				s.knownDigests[req.Digest] = trimmed
			}
			pinnedIDs = append(pinnedIDs, trimmed)
		}
		s.recalculateTokensLocked()

		receipt = ContextControlReceipt{
			Status: StatusAccepted,
			Action: ActionPin,
			Details: map[string]any{
				"pinned_spans":     pinnedIDs,
				"total_pinned":     len(s.pinned),
				"estimated_tokens": s.estimatedTokens,
			},
		}

	case ActionRelease:
		if len(req.SpanIDs) == 0 {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionRelease,
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": "span_ids parameter is required and cannot be empty for release operation"},
			}
			break
		}

		// Enforce mandatory safety/capability policy: protected spans cannot be released.
		for _, id := range req.SpanIDs {
			trimmed := strings.TrimSpace(id)
			if s.isProtectedSpanLocked(trimmed) {
				receipt = ContextControlReceipt{
					Status:  StatusRefused,
					Action:  ActionRelease,
					Reason:  abi.ReasonName(abi.ReasonPolicyBlock),
					Details: map[string]any{"error": fmt.Sprintf("cannot release mandatory safety/capability protected span: %q", trimmed)},
				}
				return s.recordReceiptLocked(req.IdempotencyKey, reqHash, receipt)
			}
		}

		releasedIDs := make([]string, 0, len(req.SpanIDs))
		for _, id := range req.SpanIDs {
			trimmed := strings.TrimSpace(id)
			if trimmed == "" {
				continue
			}
			delete(s.pinned, trimmed)
			s.released[trimmed] = ReleasedSpan{
				SpanID:     trimmed,
				ReleasedAt: time.Now(),
			}
			releasedIDs = append(releasedIDs, trimmed)
		}
		s.recalculateTokensLocked()

		receipt = ContextControlReceipt{
			Status: StatusAccepted,
			Action: ActionRelease,
			Details: map[string]any{
				"released_spans":   releasedIDs,
				"total_released":   len(s.released),
				"total_pinned":     len(s.pinned),
				"estimated_tokens": s.estimatedTokens,
			},
		}

	case ActionRetrievalScope, "scope":
		scope := strings.ToLower(strings.TrimSpace(req.Scope))
		if scope == "" {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionRetrievalScope,
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": "scope parameter cannot be empty"},
			}
			break
		}

		if !s.isAllowedScopeLocked(scope) {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionRetrievalScope,
				Reason:  abi.ReasonName(abi.ReasonPolicyBlock),
				Details: map[string]any{"error": fmt.Sprintf("retrieval scope %q is not permitted by mandatory policy", scope)},
			}
			break
		}

		s.advisory.RetrievalScope = scope
		receipt = ContextControlReceipt{
			Status: StatusAccepted,
			Action: ActionRetrievalScope,
			Details: map[string]any{
				"retrieval_scope": scope,
			},
		}

	case ActionBudgetPreference, "budget":
		if req.Budget == nil {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionBudgetPreference,
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": "budget parameter must be provided"},
			}
			break
		}
		if *req.Budget < 0 {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionBudgetPreference,
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": "budget cannot be negative"},
			}
			break
		}
		if *req.Budget < s.policy.MinBudgetTokens {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionBudgetPreference,
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": fmt.Sprintf("budget %d is below mandatory minimum floor %d", *req.Budget, s.policy.MinBudgetTokens)},
			}
			break
		}
		if *req.Budget > s.policy.MaxBudgetTokens {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionBudgetPreference,
				Reason:  abi.ReasonName(abi.ReasonOversize),
				Details: map[string]any{"error": fmt.Sprintf("budget %d exceeds mandatory maximum ceiling %d", *req.Budget, s.policy.MaxBudgetTokens)},
			}
			break
		}

		bVal := *req.Budget
		s.advisory.PreferredBudget = &bVal
		receipt = ContextControlReceipt{
			Status: StatusAccepted,
			Action: ActionBudgetPreference,
			Details: map[string]any{
				"preferred_budget": bVal,
			},
		}

	case ActionRestore, "restore_request":
		if len(req.SpanIDs) == 0 && req.Digest == "" {
			receipt = ContextControlReceipt{
				Status:  StatusRefused,
				Action:  ActionRestore,
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": "restore requires either span_ids or digest"},
			}
			break
		}

		if req.Digest != "" {
			cleanDigest := strings.TrimSpace(req.Digest)
			if len(cleanDigest) < 8 || strings.ContainsAny(cleanDigest, " \t\r\n") {
				receipt = ContextControlReceipt{
					Status:  StatusRefused,
					Action:  ActionRestore,
					Reason:  abi.ReasonName(abi.ReasonMalformed),
					Details: map[string]any{"error": "invalid digest format for restore: must be at least 8 non-whitespace characters"},
				}
				break
			}
			if _, ok := s.knownDigests[cleanDigest]; !ok {
				receipt = ContextControlReceipt{
					Status:  StatusRefused,
					Action:  ActionRestore,
					Reason:  abi.ReasonName(abi.ReasonUnwitnessed),
					Details: map[string]any{"error": fmt.Sprintf("unknown or unprovenanced digest %q for restore", cleanDigest)},
				}
				break
			}
		}

		restoredIDs := make([]string, 0, len(req.SpanIDs)+1)
		for _, id := range req.SpanIDs {
			trimmed := strings.TrimSpace(id)
			if trimmed == "" {
				continue
			}
			delete(s.released, trimmed)
			s.restored[trimmed] = time.Now()
			s.pinned[trimmed] = PinnedSpan{
				SpanID:   trimmed,
				PinnedAt: time.Now(),
				Tokens:   DefaultSpanTokens,
			}
			restoredIDs = append(restoredIDs, trimmed)
		}

		if req.Digest != "" {
			if id, ok := s.knownDigests[req.Digest]; ok {
				delete(s.released, id)
				s.restored[id] = time.Now()
				s.pinned[id] = PinnedSpan{
					SpanID:   id,
					Digest:   req.Digest,
					PinnedAt: time.Now(),
					Tokens:   DefaultSpanTokens,
				}
				restoredIDs = append(restoredIDs, id)
			}
		}
		s.recalculateTokensLocked()

		receipt = ContextControlReceipt{
			Status: StatusAccepted,
			Action: ActionRestore,
			Details: map[string]any{
				"restored_spans": restoredIDs,
				"total_pinned":   len(s.pinned),
			},
		}

	default:
		receipt = ContextControlReceipt{
			Status:  StatusRefused,
			Action:  req.Action,
			Reason:  abi.ReasonName(abi.ReasonMisroute),
			Details: map[string]any{"error": fmt.Sprintf("unknown context control action: %q", req.Action)},
		}
	}

	return s.recordReceiptLocked(req.IdempotencyKey, reqHash, receipt)
}

func (s *ContextControlState) recordReceiptLocked(idempKey, reqHash string, receipt ContextControlReceipt) ContextControlReceipt {
	if idempKey != "" {
		s.idempotency[idempKey] = idempotencyRecord{
			reqHash: reqHash,
			receipt: receipt,
		}
	}
	return receipt
}

// ---------------------------------------------------------------------------
// Adjudicator Gate & Engine Implementation
// ---------------------------------------------------------------------------

type contextControlGate struct{}

func (contextControlGate) Caps() []abi.Capability { return nil }

func (contextControlGate) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	st := armedContextControl.Load()
	if st == nil {
		return abi.Verdict{Kind: abi.VerdictDefer, By: RungNameContextControl}
	}
	if c.Tool == ToolContextControl {
		c.Engine = EngineContextControl
		return abi.Verdict{Kind: abi.VerdictAllow, By: RungNameContextControl}
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: RungNameContextControl}
}

type contextControlEngine struct {
	state *ContextControlState
}

func (e *contextControlEngine) Caps() []abi.Capability { return nil }
func (e *contextControlEngine) WeightBearing() bool    { return false }

func (e *contextControlEngine) getState() *ContextControlState {
	if e != nil && e.state != nil {
		return e.state
	}
	return armedContextControl.Load()
}

func (e *contextControlEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body, _ := decodeCallArgs(ctx, c.Args)
	st := e.getState()
	if st == nil {
		errResp, _ := json.Marshal(ContextControlReceipt{
			Status:  StatusError,
			Action:  c.Tool,
			Reason:  abi.ReasonName(abi.ReasonDefaultDeny),
			Details: map[string]any{"error": "context control state is unarmed"},
		})
		return engineResult(ctx, c, body, errResp, true, EngineContextControl), nil
	}

	var req ContextControlRequest
	if len(body) > 0 {
		if err := json.Unmarshal(body, &req); err != nil {
			errResp, _ := json.Marshal(ContextControlReceipt{
				Status:  StatusRefused,
				Action:  "unknown",
				Reason:  abi.ReasonName(abi.ReasonMalformed),
				Details: map[string]any{"error": fmt.Sprintf("invalid arguments JSON: %v", err)},
			})
			return engineResult(ctx, c, body, errResp, false, EngineContextControl), nil
		}
	}

	receipt := st.Execute(ctx, req)
	respBytes, _ := json.Marshal(receipt)
	isErr := receipt.Status == StatusError
	return engineResult(ctx, c, body, respBytes, isErr, EngineContextControl), nil
}

// ---------------------------------------------------------------------------
// Arm / Disarm / Catalog
// ---------------------------------------------------------------------------

var (
	armedContextControl        atomic.Pointer[ContextControlState]
	contextControlGateOnce     sync.Once
	contextControlEngineOnce   sync.Once
	activeContextControlEngine *contextControlEngine
)

// ArmContextControl initializes context_control, registers its engine, installs the
// adjudicator gate once, and returns the planner-facing ToolDef declarations.
func ArmContextControl(opts ...ContextControlOption) ([]ToolDef, error) {
	st := NewContextControlState(opts...)
	armedContextControl.Store(st)

	contextControlEngineOnce.Do(func() {
		activeContextControlEngine = &contextControlEngine{}
		abi.RegisterEngine(EngineContextControl, activeContextControlEngine)
	})

	contextControlGateOnce.Do(func() {
		abi.RegisterAdjudicator(contextControlRank, contextControlGate{})
	})

	return ContextControlCatalog(), nil
}

// DisarmContextControl unarms context_control, restoring the unarmed state.
func DisarmContextControl() {
	armedContextControl.Store(nil)
}

// GetActiveContextControlState returns the active ContextControlState, or nil if unarmed.
func GetActiveContextControlState() *ContextControlState {
	return armedContextControl.Load()
}

// ContextControlCatalog renders the context_control tool as loop ToolDefs. Empty when unarmed.
func ContextControlCatalog() []ToolDef {
	if armedContextControl.Load() == nil {
		return nil
	}
	return contextControlToolDefs()
}

func contextControlToolDefs() []ToolDef {
	return []ToolDef{
		{
			Type: "function",
			Function: ToolDefFunction{
				Name: ToolContextControl,
				Description: "Expose bounded agent context controls and queries: inspect, query, pin, release, " +
					"retrieval_scope, budget_preference, and restore operations with structured receipts.",
				Parameters: rawSchema(`{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["inspect", "query", "pin", "release", "retrieval_scope", "budget_preference", "restore"],
      "description": "The context control operation to perform: inspect, query, pin, release, retrieval_scope, budget_preference, or restore."
    },
    "span_ids": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Target span IDs for pin, release, or restore operations."
    },
    "digest": {
      "type": "string",
      "description": "Content digest for provenance verification or restore requests."
    },
    "budget": {
      "type": "integer",
      "description": "Advisory context token budget preference or cap."
    },
    "query": {
      "type": "string",
      "description": "Context query or prediction intent text to inspect or score spans."
    },
    "idempotency_key": {
      "type": "string",
      "description": "Unique key to ensure idempotent request processing across turns."
    },
    "scope": {
      "type": "string",
      "description": "Optional retrieval scope name (e.g. repo, evidence, turn, session)."
    },
    "horizon": {
      "type": "integer",
      "description": "Optional lookahead horizon in turns."
    },
    "ttl_seconds": {
      "type": "integer",
      "description": "Optional time-to-live in seconds for pinned spans."
    },
    "provenance": {
      "type": "string",
      "description": "Optional provenance source or evidence origin tag."
    }
  },
  "required": ["action"]
}`),
			},
		},
	}
}

// contextControlMeta returns the vDSO / consistency metadata for context_control.
func contextControlMeta(tool string) (map[string]string, bool) {
	if armedContextControl.Load() == nil {
		return nil, false
	}
	if tool == ToolContextControl {
		return map[string]string{
			"readOnlyHint":   "false",
			"idempotentHint": "true",
			"consistency":    "STRICT",
		}, true
	}
	return nil, false
}

// contextControlAllow returns admitted tool names when context control is armed.
func contextControlAllow() []string {
	if armedContextControl.Load() == nil {
		return nil
	}
	return []string{ToolContextControl}
}
