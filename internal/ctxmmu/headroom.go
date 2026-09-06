package ctxmmu

import (
	"crypto/sha256"
)

// Standard headroom actions and default thresholds.
const (
	ActionNone            = "none"
	ActionTier1Tombstone  = "tier1_tombstone"
	ActionTier2Compaction = "tier2_compaction"

	DefaultEscalationThreshold = 0.85
	DefaultTier1Threshold      = 0.70

	// Tier1TombstoneMarker is the standardized mechanical tombstone marker
	// for cleared tool execution payloads.
	Tier1TombstoneMarker = "[Old tool result content cleared]"
)

// HeadroomPolicy defines context window headroom monitoring and escalation thresholds.
type HeadroomPolicy struct {
	// EscalationThreshold is the fraction of maxCapacity at or above which Tier 2
	// emergency compaction is triggered (escalation buffer to prevent cliff aborts).
	// Defaults to DefaultEscalationThreshold (0.85).
	EscalationThreshold float64 `json:"escalation_threshold,omitempty"`

	// Tier1Threshold is the fraction of maxCapacity at or above which Tier 1
	// mechanical tombstoning is recommended. Defaults to DefaultTier1Threshold (0.70).
	Tier1Threshold float64 `json:"tier1_threshold,omitempty"`

	// TargetRatio is the desired post-compaction utilization ratio used to calculate
	// ReclaimTargetTokens. Defaults to Tier1Threshold (0.70).
	TargetRatio float64 `json:"target_ratio,omitempty"`
}

// DefaultHeadroomPolicy returns a HeadroomPolicy initialized with standard default thresholds.
func DefaultHeadroomPolicy() HeadroomPolicy {
	return HeadroomPolicy{
		EscalationThreshold: DefaultEscalationThreshold,
		Tier1Threshold:      DefaultTier1Threshold,
		TargetRatio:         DefaultTier1Threshold,
	}
}

// HeadroomAssessment represents the evaluated context headroom state.
type HeadroomAssessment struct {
	Ratio               float64 `json:"ratio"`
	ExceedsEscalation   bool    `json:"exceeds_escalation"`
	RecommendedAction   string  `json:"recommended_action"`
	ReclaimTargetTokens int     `json:"reclaim_target_tokens"`
}

// AssessHeadroom evaluates current token usage against effective context capacity
// according to the given policy. It implements two-tier compaction advice:
//   - Under 70%: "none" (safe working window)
//   - 70% - 85%: "tier1_tombstone" (mechanical clearance of older tool results)
//   - >= 85%: "tier2_compaction" (headroom escalation buffer triggered to prevent cliff aborts)
func AssessHeadroom(currentTokens int, maxCapacity int, policy HeadroomPolicy) HeadroomAssessment {
	if maxCapacity <= 0 || currentTokens <= 0 {
		return HeadroomAssessment{
			Ratio:               0,
			ExceedsEscalation:   false,
			RecommendedAction:   ActionNone,
			ReclaimTargetTokens: 0,
		}
	}

	escalationThresh := policy.EscalationThreshold
	if escalationThresh <= 0 {
		escalationThresh = DefaultEscalationThreshold
	}

	tier1Thresh := policy.Tier1Threshold
	if tier1Thresh <= 0 {
		tier1Thresh = DefaultTier1Threshold
	}

	targetRatio := policy.TargetRatio
	if targetRatio <= 0 {
		targetRatio = tier1Thresh
	}

	ratio := float64(currentTokens) / float64(maxCapacity)

	var (
		exceedsEscalation bool
		action            string
		reclaimTarget     int
	)

	if ratio >= escalationThresh {
		exceedsEscalation = true
		action = ActionTier2Compaction
	} else if ratio >= tier1Thresh {
		exceedsEscalation = false
		action = ActionTier1Tombstone
	} else {
		exceedsEscalation = false
		action = ActionNone
	}

	if action != ActionNone {
		targetTokens := int(float64(maxCapacity) * targetRatio)
		if currentTokens > targetTokens {
			reclaimTarget = currentTokens - targetTokens
		}
	}

	return HeadroomAssessment{
		Ratio:               ratio,
		ExceedsEscalation:   exceedsEscalation,
		RecommendedAction:   action,
		ReclaimTargetTokens: reclaimTarget,
	}
}

// PrefixPreservingByteSplice performs Tier 1 prefix-preserving byte-splice compaction
// on token pages, replacing middle-turn tool outputs with compact tombstones while
// guaranteeing that token 0 system instructions and tool schemas remain bit-for-bit verbatim.
// Returns a slice of compacted pages and the total number of tokens reclaimed.
func PrefixPreservingByteSplice(pages []TokenPage, activeKeepTurns int) ([]TokenPage, int) {
	if len(pages) == 0 {
		return nil, 0
	}
	if activeKeepTurns < 1 {
		activeKeepTurns = 2
	}

	// Find the maximum turn index to determine the active window boundary.
	maxTurn := 0
	for i := range pages {
		if !pages[i].Kind.IsPrefix() && pages[i].TurnIndex > maxTurn {
			maxTurn = pages[i].TurnIndex
		}
	}

	activeWindowStart := maxTurn - activeKeepTurns + 1
	if activeWindowStart < 1 {
		activeWindowStart = 1
	}

	compacted := make([]TokenPage, len(pages))
	copy(compacted, pages)

	tokensReclaimed := 0
	markerBytes := []byte(Tier1TombstoneMarker)
	markerTokens := EstimateTokens(markerBytes)

	for i := range compacted {
		p := &compacted[i]

		// Prefix, active window, or pinned -> preserve bit-for-bit verbatim.
		if p.Kind.IsPrefix() || p.TurnIndex < 1 || p.TurnIndex >= activeWindowStart || p.Pinned {
			continue
		}

		// Middle turns: mechanically tombstone uncompacted tool results.
		if p.Kind == PageKindToolResult && !p.Tombstone.Active {
			origBytes := len(p.Content)
			origTokens := p.Tokens
			sum := sha256.Sum256(p.Content)

			p.Tombstone = Tombstone{
				Active:         true,
				Digest:         sum,
				OriginalBytes:  origBytes,
				OriginalTokens: origTokens,
				Tool:           p.ToolName,
				Summary:        Tier1TombstoneMarker,
			}

			p.Content = append([]byte(nil), markerBytes...)
			p.Tokens = markerTokens

			if origTokens > markerTokens {
				tokensReclaimed += (origTokens - markerTokens)
			}
		}
	}

	return compacted, tokensReclaimed
}
