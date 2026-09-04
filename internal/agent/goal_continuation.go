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
type GoalContinuationSession struct {
	GoalID         string
	GoalStatement  string
	stablePrefix   string
	lastWorldHash  string
	turnsDelivered int
}

// NewGoalContinuationSession initializes a goal continuation session with byte-stable prefix text.
func NewGoalContinuationSession(goalID, goalStatement string) *GoalContinuationSession {
	cleaned := strings.TrimSpace(goalStatement)
	prefix := fmt.Sprintf("[fak:goal id=%s] %s", strings.TrimSpace(goalID), cleaned)
	return &GoalContinuationSession{
		GoalID:        goalID,
		GoalStatement: cleaned,
		stablePrefix:  prefix,
	}
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
	return Message{
		Role:    RoleUser,
		Content: fmt.Sprintf("[fak:progress turn=%d budget_remaining=%d status=%s]", turn, budgetRemaining, strings.TrimSpace(status)),
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
func (g *GoalContinuationSession) FormatWorldState(currentEnv map[string]string, forceFull bool) (WorldStateUpdate, Message) {
	currentHash := computeEnvHash(currentEnv)
	isFirst := g.turnsDelivered == 0
	drift := g.lastWorldHash != "" && g.lastWorldHash != currentHash

	g.turnsDelivered++
	g.lastWorldHash = currentHash

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
