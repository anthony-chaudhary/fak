package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
)

// GoalContinuationSession manages byte-stable prompt prefixes across continuation turns (#10671).
// Per-turn progress and budget deltas are separated into trailing, append-only messages
// so the goal prefix remains byte-identical turn-over-turn, preserving provider prompt cache.
// It also tracks tool denials and enforces turn budget (MaxTurns) and stall circuit breakers (MaxConsecutiveNoProgress) (#11639).
type GoalContinuationSession struct {
	GoalID                   string
	GoalStatement            string
	stablePrefix             string
	lastWorldHash            string
	turnsDelivered           int
	MaxTurns                 int
	MaxConsecutiveNoProgress int
	BranchBlocked            bool
	StallReason              string
	StallDetector            *StallDetector
	consecutiveNoProgress    int
	denialsTotal             int
	denialsByReason          map[string]int
	denials                  []string
}

const (
	// DefaultMaxTurns is the default maximum session turn budget before continuation halts.
	DefaultMaxTurns = 100

	// DefaultMaxConsecutiveNoProgress is the default consecutive turn threshold for stalling.
	DefaultMaxConsecutiveNoProgress = 5

	// StallReasonMaxConsecutiveNoProgress indicates circuit breaker tripped due to consecutive zero progress.
	StallReasonMaxConsecutiveNoProgress = "max_consecutive_no_progress_tripped"

	// StallReasonMaxTurnsExceeded indicates session halted due to exceeding maximum turns budget.
	StallReasonMaxTurnsExceeded = "max_turns_exceeded"
)

// StallDetector monitors turn-by-turn progress deltas and trips a circuit breaker
// when consecutive turns show no forward progress or state delta (#11639).
type StallDetector struct {
	MaxConsecutiveNoProgress int
	ConsecutiveNoProgress    int
	BranchBlocked            bool
	StallReason              string
}

// NewStallDetector initializes a StallDetector with a maximum consecutive no-progress threshold.
func NewStallDetector(maxConsecutive int) *StallDetector {
	if maxConsecutive <= 0 {
		maxConsecutive = DefaultMaxConsecutiveNoProgress
	}
	return &StallDetector{
		MaxConsecutiveNoProgress: maxConsecutive,
	}
}

// RecordProgress updates progress state and returns whether the circuit breaker tripped.
func (s *StallDetector) RecordProgress(hasProgress bool) bool {
	if hasProgress {
		s.ConsecutiveNoProgress = 0
		return s.BranchBlocked
	}
	s.ConsecutiveNoProgress++
	if s.ConsecutiveNoProgress >= s.MaxConsecutiveNoProgress {
		s.BranchBlocked = true
		if s.StallReason == "" {
			s.StallReason = StallReasonMaxConsecutiveNoProgress
		}
	}
	return s.BranchBlocked
}

// IsStalled reports whether the circuit breaker has tripped.
func (s *StallDetector) IsStalled() bool {
	return s.BranchBlocked
}

// Reset clears the consecutive no-progress counter and unblocks the circuit breaker.
func (s *StallDetector) Reset() {
	s.ConsecutiveNoProgress = 0
	s.BranchBlocked = false
	s.StallReason = ""
}

// TurnProgress encapsulates per-turn progress signals to evaluate forward motion.
type TurnProgress struct {
	HasProgress    bool              // Explicit progress indicator
	ToolExecutions int               // Tool executions during the turn (0 = zero progress)
	StateDelta     bool              // Environment / world state delta observed
	Env            map[string]string // Current environment map (evaluated against lastWorldHash)
	Status         string            // Turn status string (e.g. "completed", "in_progress", "blocked")
}

// ProgressDelta is an alias for TurnProgress.
type ProgressDelta = TurnProgress

// GoalContinuationOption configures optional parameters for GoalContinuationSession.
type GoalContinuationOption func(*GoalContinuationSession)

// WithMaxTurns configures the session budget.
func WithMaxTurns(maxTurns int) GoalContinuationOption {
	return func(g *GoalContinuationSession) {
		g.MaxTurns = maxTurns
	}
}

// WithMaxConsecutiveNoProgress configures the stall detector threshold.
func WithMaxConsecutiveNoProgress(maxConsecutive int) GoalContinuationOption {
	return func(g *GoalContinuationSession) {
		g.MaxConsecutiveNoProgress = maxConsecutive
		if g.StallDetector != nil {
			g.StallDetector.MaxConsecutiveNoProgress = maxConsecutive
		}
	}
}

// WithStallDetector configures a custom StallDetector.
func WithStallDetector(detector *StallDetector) GoalContinuationOption {
	return func(g *GoalContinuationSession) {
		g.StallDetector = detector
		if detector != nil && detector.MaxConsecutiveNoProgress > 0 {
			g.MaxConsecutiveNoProgress = detector.MaxConsecutiveNoProgress
		}
	}
}

// NewGoalContinuationSession initializes a goal continuation session with byte-stable prefix text,
// default turn budgets, and a runtime stall detector.
func NewGoalContinuationSession(goalID, goalStatement string, opts ...GoalContinuationOption) *GoalContinuationSession {
	cleaned := strings.TrimSpace(goalStatement)
	prefix := fmt.Sprintf("[fak:goal id=%s] %s", strings.TrimSpace(goalID), cleaned)
	detector := NewStallDetector(DefaultMaxConsecutiveNoProgress)
	sess := &GoalContinuationSession{
		GoalID:                   goalID,
		GoalStatement:            cleaned,
		stablePrefix:             prefix,
		MaxTurns:                 DefaultMaxTurns,
		MaxConsecutiveNoProgress: DefaultMaxConsecutiveNoProgress,
		StallDetector:            detector,
		denialsByReason:          make(map[string]int),
	}
	for _, opt := range opts {
		opt(sess)
	}
	return sess
}

func (g *GoalContinuationSession) effectiveMaxTurns() int {
	if g.MaxTurns > 0 {
		return g.MaxTurns
	}
	return DefaultMaxTurns
}

func (g *GoalContinuationSession) effectiveMaxConsecutiveNoProgress() int {
	if g.MaxConsecutiveNoProgress > 0 {
		return g.MaxConsecutiveNoProgress
	}
	if g.StallDetector != nil && g.StallDetector.MaxConsecutiveNoProgress > 0 {
		return g.StallDetector.MaxConsecutiveNoProgress
	}
	return DefaultMaxConsecutiveNoProgress
}

// CanContinue reports whether continuation may proceed without being blocked by turn budget or stall breaker.
func (g *GoalContinuationSession) CanContinue() bool {
	if g.BranchBlocked {
		return false
	}
	if g.turnsDelivered >= g.effectiveMaxTurns() {
		return false
	}
	return true
}

// IsBranchBlocked reports whether the session continuation branch is blocked.
func (g *GoalContinuationSession) IsBranchBlocked() bool {
	return g.BranchBlocked
}

// TurnsDelivered returns the number of continuation turns processed.
func (g *GoalContinuationSession) TurnsDelivered() int {
	return g.turnsDelivered
}

// ConsecutiveNoProgress returns the number of consecutive turns without progress.
func (g *GoalContinuationSession) ConsecutiveNoProgress() int {
	return g.consecutiveNoProgress
}

// EvaluateStallState checks whether the session is currently stalled or blocked.
func (g *GoalContinuationSession) EvaluateStallState() bool {
	return g.BranchBlocked
}

// CheckStall checks whether the session is currently stalled or blocked.
func (g *GoalContinuationSession) CheckStall() bool {
	return g.BranchBlocked
}

// RecordDenial tracks a tool execution denial for the session.
func (g *GoalContinuationSession) RecordDenial(args ...string) {
	if g.denialsByReason == nil {
		g.denialsByReason = make(map[string]int)
	}
	g.denialsTotal++
	var reason string
	if len(args) == 1 {
		reason = args[0]
	} else if len(args) >= 2 {
		reason = args[1]
	} else {
		reason = "unknown"
	}
	g.denialsByReason[reason]++
	g.denials = append(g.denials, strings.Join(args, ":"))
}

// DenialsTotal returns the total count of tool execution denials.
func (g *GoalContinuationSession) DenialsTotal() int {
	return g.denialsTotal
}

// DenialsByReason returns a copy of the denials-by-reason map.
func (g *GoalContinuationSession) DenialsByReason() map[string]int {
	out := make(map[string]int, len(g.denialsByReason))
	for k, v := range g.denialsByReason {
		out[k] = v
	}
	return out
}

// RecordTurnProgress records progress signals for a turn and returns whether BranchBlocked is tripped.
func (g *GoalContinuationSession) RecordTurnProgress(args ...any) bool {
	if g.BranchBlocked {
		return true
	}

	hasProgress := false
	if len(args) > 0 {
		switch v := args[0].(type) {
		case bool:
			hasProgress = v
		case TurnProgress:
			hasProgress = v.HasProgress || v.ToolExecutions > 0 || v.StateDelta
			if v.Env != nil {
				h := computeEnvHash(v.Env)
				if g.lastWorldHash != "" && h != g.lastWorldHash {
					hasProgress = true
				}
				g.lastWorldHash = h
			}
			if v.Status == "blocked" || v.Status == "stalled" || v.Status == "branch_blocked" {
				hasProgress = false
			}
		case int:
			hasProgress = v > 0
			if len(args) > 1 {
				if delta, ok := args[1].(bool); ok {
					hasProgress = hasProgress || delta
				}
			}
		case map[string]string:
			h := computeEnvHash(v)
			hasProgress = g.lastWorldHash != "" && h != g.lastWorldHash
			g.lastWorldHash = h
		case string:
			s := strings.ToLower(strings.TrimSpace(v))
			if s == "progress" || s == "advanced" || s == "done" || s == "completed" {
				hasProgress = true
			}
		}
	}

	g.turnsDelivered++

	maxThreshold := g.effectiveMaxConsecutiveNoProgress()
	if g.StallDetector != nil {
		g.StallDetector.MaxConsecutiveNoProgress = maxThreshold
		g.StallDetector.RecordProgress(hasProgress)
		g.consecutiveNoProgress = g.StallDetector.ConsecutiveNoProgress
		if g.StallDetector.BranchBlocked {
			g.BranchBlocked = true
			g.StallReason = g.StallDetector.StallReason
			if g.StallReason == "" {
				g.StallReason = StallReasonMaxConsecutiveNoProgress
			}
		}
	} else {
		if hasProgress {
			g.consecutiveNoProgress = 0
		} else {
			g.consecutiveNoProgress++
			if g.consecutiveNoProgress >= maxThreshold {
				g.BranchBlocked = true
				g.StallReason = StallReasonMaxConsecutiveNoProgress
			}
		}
	}

	if g.turnsDelivered >= g.effectiveMaxTurns() {
		g.BranchBlocked = true
		if g.StallReason == "" {
			g.StallReason = StallReasonMaxTurnsExceeded
		}
		if g.StallDetector != nil {
			g.StallDetector.BranchBlocked = true
			g.StallDetector.StallReason = g.StallReason
		}
	}

	return g.BranchBlocked
}

// RecordTurn is an alias for RecordTurnProgress.
func (g *GoalContinuationSession) RecordTurn(hasProgress bool) bool {
	return g.RecordTurnProgress(hasProgress)
}

// StableGoalMessage returns the invariant goal message whose content is guaranteed byte-identical
// across all turns in the session.
func (g *GoalContinuationSession) StableGoalMessage() Message {
	return Message{
		Role:    RoleUser,
		Content: g.stablePrefix,
	}
}

// TrailingProgressDeltaMessage produces a trailing progress message containing turn-specific
// budget, step count, or status information without mutating the leading prompt prefix.
func (g *GoalContinuationSession) TrailingProgressDeltaMessage(turn int, budgetRemaining int, status string) Message {
	effectiveStatus := strings.TrimSpace(status)
	if g.BranchBlocked {
		effectiveStatus = "branch_blocked"
	}
	return Message{
		Role:    RoleUser,
		Content: fmt.Sprintf("[fak:progress turn=%d budget_remaining=%d status=%s]", turn, budgetRemaining, effectiveStatus),
	}
}

// WorldStateUpdate represents an environment/world-state injection.
type WorldStateUpdate struct {
	Full      bool              `json:"full"`
	Snapshot  map[string]string `json:"snapshot,omitempty"`
	Delta     map[string]string `json:"delta,omitempty"`
	DriftSeen bool              `json:"drift_seen"`
}

// FormatWorldState emits world state as a diff/partial by default (#10671).
// A full injection is emitted only on the first turn (turn 0), explicit drift, or forced request.
// If the session turn budget is exceeded or consecutive turns exhibit no state delta exceeding
// the MaxConsecutiveNoProgress threshold, BranchBlocked is tripped and further continuations are suppressed (#11639).
func (g *GoalContinuationSession) FormatWorldState(currentEnv map[string]string, forceFull bool) (WorldStateUpdate, Message) {
	if g.BranchBlocked || g.turnsDelivered >= g.effectiveMaxTurns() {
		g.BranchBlocked = true
		if g.StallReason == "" {
			g.StallReason = StallReasonMaxTurnsExceeded
		}
		if g.StallDetector != nil {
			g.StallDetector.BranchBlocked = true
			if g.StallDetector.StallReason == "" {
				g.StallDetector.StallReason = g.StallReason
			}
		}
		return WorldStateUpdate{
				Full:      false,
				DriftSeen: false,
			}, Message{
				Role:    RoleSystem,
				Content: fmt.Sprintf("<codex_internal_context source=\"world_state\" full=\"false\" status=\"branch_blocked\">[fak:continuation_halted reason=%s]</codex_internal_context>", g.StallReason),
			}
	}

	currentHash := computeEnvHash(currentEnv)
	isFirst := g.turnsDelivered == 0
	drift := g.lastWorldHash != "" && g.lastWorldHash != currentHash

	g.turnsDelivered++
	g.lastWorldHash = currentHash

	// Track state delta for stall detector
	if isFirst || drift {
		g.consecutiveNoProgress = 0
		if g.StallDetector != nil {
			g.StallDetector.ConsecutiveNoProgress = 0
		}
	} else {
		// Identical world state hash without drift
		maxThreshold := g.effectiveMaxConsecutiveNoProgress()
		if g.StallDetector != nil {
			g.StallDetector.MaxConsecutiveNoProgress = maxThreshold
			g.StallDetector.RecordProgress(false)
			g.consecutiveNoProgress = g.StallDetector.ConsecutiveNoProgress
			if g.StallDetector.BranchBlocked {
				g.BranchBlocked = true
				g.StallReason = g.StallDetector.StallReason
				if g.StallReason == "" {
					g.StallReason = StallReasonMaxConsecutiveNoProgress
				}
			}
		} else {
			g.consecutiveNoProgress++
			if g.consecutiveNoProgress >= maxThreshold {
				g.BranchBlocked = true
				g.StallReason = StallReasonMaxConsecutiveNoProgress
			}
		}
	}

	if g.turnsDelivered >= g.effectiveMaxTurns() {
		g.BranchBlocked = true
		if g.StallReason == "" {
			g.StallReason = StallReasonMaxTurnsExceeded
		}
		if g.StallDetector != nil {
			g.StallDetector.BranchBlocked = true
			g.StallDetector.StallReason = g.StallReason
		}
	}

	if isFirst || forceFull || drift {
		// Full baseline snapshot
		snapCopy := make(map[string]string, len(currentEnv))
		for k, v := range currentEnv {
			snapCopy[k] = v
		}
		up := WorldStateUpdate{
			Full:      true,
			Snapshot:  snapCopy,
			DriftSeen: drift,
		}
		return up, Message{
			Role:    RoleSystem,
			Content: fmt.Sprintf("<codex_internal_context source=\"world_state\" full=\"true\">%s</codex_internal_context>", serializeEnv(snapCopy)),
		}
	}

	// Partial diff update
	up := WorldStateUpdate{
		Full:      false,
		Delta:     map[string]string{"hash": currentHash},
		DriftSeen: false,
	}
	return up, Message{
		Role:    RoleSystem,
		Content: fmt.Sprintf("<codex_internal_context source=\"world_state\" full=\"false\" hash=\"%s\"></codex_internal_context>", currentHash),
	}
}

func computeEnvHash(env map[string]string) string {
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	h := sha256.New()
	for _, k := range keys {
		h.Write([]byte(k))
		h.Write([]byte("="))
		h.Write([]byte(env[k]))
		h.Write([]byte(";"))
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

func serializeEnv(env map[string]string) string {
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%s", k, env[k]))
	}
	return strings.Join(parts, "; ")
}
