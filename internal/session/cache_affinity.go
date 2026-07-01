package session

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

const (
	// CacheAffinityPreserve means a continuation should keep using the same opaque
	// routing/cache-affinity key as its parent lineage.
	CacheAffinityPreserve = "preserve"
)

// CacheAffinityDecision is the auditable cache-affinity handoff stamped when a
// context-budget reset mints a continuation id. It is deliberately provider-neutral:
// vcachegov owns provider-specific header derivation, while session owns the lineage
// decision that says whether a hidden reset preserved affinity.
type CacheAffinityDecision struct {
	Action      string `json:"action,omitempty"`
	AffinityKey string `json:"affinity_key,omitempty"`
	FromTraceID string `json:"from_trace_id,omitempty"`
	ToTraceID   string `json:"to_trace_id,omitempty"`
	Reason      string `json:"reason,omitempty"`
}

// IsZero reports whether the decision is absent, for json omitzero.
func (d CacheAffinityDecision) IsZero() bool {
	return d.Action == "" && d.AffinityKey == "" && d.FromTraceID == "" && d.ToTraceID == "" && d.Reason == ""
}

func cacheAffinityForContinuation(parent State, child, reason string) CacheAffinityDecision {
	if child == "" {
		return CacheAffinityDecision{}
	}
	parentTrace := strings.TrimSpace(parent.TraceID)
	if parentTrace == "" {
		return CacheAffinityDecision{}
	}
	key := parent.CacheAffinity.AffinityKey
	if key == "" {
		key = deriveCacheAffinityKey(parentTrace)
	}
	if reason == "" {
		reason = ReasonBudgetReset
	}
	return CacheAffinityDecision{
		Action:      CacheAffinityPreserve,
		AffinityKey: key,
		FromTraceID: parentTrace,
		ToTraceID:   strings.TrimSpace(child),
		Reason:      reason,
	}
}

func deriveCacheAffinityKey(rootTrace string) string {
	h := sha256.Sum256([]byte("fak-session-cache-affinity\x00" + strings.TrimSpace(rootTrace)))
	return hex.EncodeToString(h[:])[:32]
}
