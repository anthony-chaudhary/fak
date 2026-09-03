package agenticbench

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const AttemptLineageSchema = "fak.agentic-benchmark-attempt-lineage.v1"

// Reason constants for attempt lineage validation and refusal.
const (
	ReasonTimeoutRetryForbidden = "TIMEOUT_RETRY_FORBIDDEN"
	ReasonRetryRegimeMismatch   = "RETRY_REGIME_MISMATCH"
	ReasonHiddenAttempt         = "HIDDEN_ATTEMPT"
	ReasonDuplicateOrdinal      = "DUPLICATE_ORDINAL"
	ReasonNonContiguousOrdinals = "NON_CONTIGUOUS_ORDINALS"
	ReasonDuplicateAttemptID    = "DUPLICATE_ATTEMPT_ID"
	ReasonDuplicateArtifactRoot = "DUPLICATE_ARTIFACT_ROOT"
	ReasonInvalidTerminalStatus = "INVALID_TERMINAL_STATUS"
	ReasonMaxAttemptsExceeded   = "MAX_ATTEMPTS_EXCEEDED"
	ReasonRetryForbidden        = "RETRY_FORBIDDEN"
	ReasonInvalidSelectionRule  = "INVALID_SELECTION_RULE"
	ReasonInvalidRetryRegime    = "INVALID_RETRY_REGIME"
	ReasonUnitIDMismatch        = "UNIT_ID_MISMATCH"
	ReasonEmptyAttempts         = "EMPTY_ATTEMPTS"
	ReasonInvalidScore          = "INVALID_SCORE"
	ReasonInvalidRetryReason    = "INVALID_RETRY_REASON"
	ReasonAggregateMismatch     = "AGGREGATE_MISMATCH"
)

type NativeRetryRegime string

const (
	RetryOnFailureOnly NativeRetryRegime = "failure_only" // ALE native: failures retry, timeouts do NOT
	NoRetry            NativeRetryRegime = "no_retry"
	RetryAll           NativeRetryRegime = "retry_all"
)

type SelectionRule string

const (
	FinalAttempt SelectionRule = "final_attempt" // official score is the final attempt's score
	BestOfN      SelectionRule = "best_of_n"     // must be declared explicitly
)

const (
	TerminalStatusSuccess = "success"
	TerminalStatusFailed  = "failed"
	TerminalStatusTimeout = "timeout"
	TerminalStatusError   = "error"
)

// AttemptReceipt records the complete execution, cost, duration, and artifact
// identity for one attempt within a benchmark unit run.
type AttemptReceipt struct {
	AttemptID      string   `json:"attempt_id"`
	Ordinal        int      `json:"ordinal"`
	RunID          string   `json:"run_id"`
	UnitID         string   `json:"unit_id"`
	IsClean        bool     `json:"is_clean"`
	TerminalStatus string   `json:"terminal_status"`
	Score          *float64 `json:"score"`
	CostUSD        float64  `json:"cost_usd"`
	Tokens         int64    `json:"tokens"`
	DurationMS     int64    `json:"duration_ms"`
	ArtifactRoot   string   `json:"artifact_root"`
	ArtifactSHA256 string   `json:"artifact_sha256"`
	RetryReason    string   `json:"retry_reason,omitempty"`
}

// AggregateAccounting represents total resource spend and selected outcome across all attempts.
type AggregateAccounting struct {
	TotalCostUSD      float64  `json:"total_cost_usd"`
	TotalTokens       int64    `json:"total_tokens"`
	TotalDurationMS   int64    `json:"total_duration_ms"`
	TotalAttempts     int      `json:"total_attempts"`
	FinalScore        *float64 `json:"final_score"`
	SelectedAttemptID string   `json:"selected_attempt_id,omitempty"`
}

// AttemptLineagePacket captures full attempt lineage, retry policy, and aggregate spend
// for an external benchmark unit.
type AttemptLineagePacket struct {
	Schema          string              `json:"schema,omitempty"`
	UnitID          string              `json:"unit_id"`
	RetryRegime     NativeRetryRegime   `json:"retry_policy"`
	MaxAttempts     int                 `json:"max_attempts"`
	SelectionRule   SelectionRule       `json:"selection_rule"`
	Attempts        []AttemptReceipt    `json:"attempts"`
	Aggregate       AggregateAccounting `json:"aggregate"`
	DiscoveredRoots []string            `json:"discovered_roots,omitempty"`
}

// LineageError records structured reasons for attempt lineage refusals and validation failures.
type LineageError struct {
	Reason string
	Detail string
}

func (e *LineageError) Error() string {
	return e.Reason + ": " + e.Detail
}

func lineageError(reason, detail string) *LineageError {
	return &LineageError{Reason: reason, Detail: detail}
}

// IsLineageReason tests if an error carries a given refusal reason.
func IsLineageReason(err error, reason string) bool {
	var target *LineageError
	return errors.As(err, &target) && target.Reason == reason
}

// ComputeAggregate computes total cost, tokens, and duration across all attempts,
// ensuring failed attempts are never hidden from spend totals, and sets the final score
// based on the declared selection rule.
func ComputeAggregate(packet *AttemptLineagePacket) {
	if packet == nil {
		return
	}
	if packet.SelectionRule == "" {
		packet.SelectionRule = FinalAttempt
	}
	var (
		totalCostUSD    float64
		totalTokens     int64
		totalDurationMS int64
	)
	for _, att := range packet.Attempts {
		totalCostUSD += att.CostUSD
		totalTokens += att.Tokens
		totalDurationMS += att.DurationMS
	}
	// Round to microdollars to eliminate IEEE-754 precision artifacts.
	totalCostUSD = math.Round(totalCostUSD*1e6) / 1e6

	agg := AggregateAccounting{
		TotalCostUSD:    totalCostUSD,
		TotalTokens:     totalTokens,
		TotalDurationMS: totalDurationMS,
		TotalAttempts:   len(packet.Attempts),
	}

	if len(packet.Attempts) > 0 {
		switch packet.SelectionRule {
		case BestOfN:
			var bestAttempt *AttemptReceipt
			var maxScore float64
			for i := range packet.Attempts {
				att := &packet.Attempts[i]
				if att.Score != nil {
					if bestAttempt == nil || bestAttempt.Score == nil || *att.Score > maxScore {
						bestAttempt = att
						maxScore = *att.Score
					}
				}
			}
			if bestAttempt != nil {
				agg.FinalScore = bestAttempt.Score
				agg.SelectedAttemptID = bestAttempt.AttemptID
			} else {
				last := &packet.Attempts[len(packet.Attempts)-1]
				agg.FinalScore = nil
				agg.SelectedAttemptID = last.AttemptID
			}
		case FinalAttempt:
			fallthrough
		default:
			last := &packet.Attempts[len(packet.Attempts)-1]
			agg.FinalScore = last.Score
			agg.SelectedAttemptID = last.AttemptID
		}
	}
	packet.Aggregate = agg
}

// ValidateAttemptLineage enforces attempt lineage invariants: contiguous ordinals,
// unique IDs and roots, valid terminal statuses, retry policy bounds, and exact
// match with discovered attempt directories.
func ValidateAttemptLineage(packet AttemptLineagePacket) error {
	if strings.TrimSpace(packet.UnitID) == "" {
		return lineageError(ReasonUnitIDMismatch, "packet unit_id must be non-empty")
	}

	switch packet.RetryRegime {
	case RetryOnFailureOnly, NoRetry, RetryAll:
	default:
		return lineageError(ReasonInvalidRetryRegime, fmt.Sprintf("invalid retry policy %q", packet.RetryRegime))
	}

	switch packet.SelectionRule {
	case FinalAttempt, BestOfN:
	case "":
		// Allowed: defaults to FinalAttempt
	default:
		return lineageError(ReasonInvalidSelectionRule, fmt.Sprintf("invalid selection rule %q", packet.SelectionRule))
	}

	if packet.MaxAttempts < 1 {
		return lineageError(ReasonMaxAttemptsExceeded, fmt.Sprintf("max_attempts must be >= 1, got %d", packet.MaxAttempts))
	}

	if len(packet.Attempts) == 0 {
		return lineageError(ReasonEmptyAttempts, "attempt lineage packet must contain at least one attempt")
	}

	if len(packet.Attempts) > packet.MaxAttempts {
		return lineageError(ReasonMaxAttemptsExceeded, fmt.Sprintf("attempt count %d exceeds max_attempts %d", len(packet.Attempts), packet.MaxAttempts))
	}

	if packet.Attempts[0].Ordinal > 1 {
		return lineageError(ReasonHiddenAttempt, fmt.Sprintf("attempt 1 is missing from packet; first attempt has ordinal %d", packet.Attempts[0].Ordinal))
	}
	if packet.Attempts[0].Ordinal < 1 {
		return lineageError(ReasonNonContiguousOrdinals, fmt.Sprintf("ordinal %d must be positive (1-based)", packet.Attempts[0].Ordinal))
	}

	seenAttemptIDs := make(map[string]bool, len(packet.Attempts))
	seenArtifactRoots := make(map[string]bool, len(packet.Attempts))

	for i, receipt := range packet.Attempts {
		if strings.TrimSpace(receipt.UnitID) != "" && receipt.UnitID != packet.UnitID {
			return lineageError(ReasonUnitIDMismatch, fmt.Sprintf("attempt %d unit_id %q does not match packet unit_id %q", receipt.Ordinal, receipt.UnitID, packet.UnitID))
		}

		if strings.TrimSpace(receipt.AttemptID) == "" {
			return lineageError(ReasonDuplicateAttemptID, fmt.Sprintf("attempt %d has empty attempt_id", receipt.Ordinal))
		}
		if seenAttemptIDs[receipt.AttemptID] {
			return lineageError(ReasonDuplicateAttemptID, fmt.Sprintf("duplicate attempt_id %q", receipt.AttemptID))
		}
		seenAttemptIDs[receipt.AttemptID] = true

		if strings.TrimSpace(receipt.ArtifactRoot) == "" {
			return lineageError(ReasonDuplicateArtifactRoot, fmt.Sprintf("attempt %d has empty artifact_root", receipt.Ordinal))
		}
		cleanRoot := filepath.Clean(receipt.ArtifactRoot)
		if seenArtifactRoots[cleanRoot] {
			return lineageError(ReasonDuplicateArtifactRoot, fmt.Sprintf("duplicate artifact_root %q", receipt.ArtifactRoot))
		}
		seenArtifactRoots[cleanRoot] = true

		if i > 0 {
			prevOrdinal := packet.Attempts[i-1].Ordinal
			if receipt.Ordinal == prevOrdinal {
				return lineageError(ReasonDuplicateOrdinal, fmt.Sprintf("duplicate ordinal %d at index %d", receipt.Ordinal, i))
			}
			if receipt.Ordinal != prevOrdinal+1 {
				return lineageError(ReasonNonContiguousOrdinals, fmt.Sprintf("non-contiguous ordinal %d follows %d", receipt.Ordinal, prevOrdinal))
			}
		}

		switch receipt.TerminalStatus {
		case TerminalStatusSuccess, TerminalStatusFailed, TerminalStatusTimeout, TerminalStatusError:
		default:
			return lineageError(ReasonInvalidTerminalStatus, fmt.Sprintf("invalid terminal status %q on attempt %d", receipt.TerminalStatus, receipt.Ordinal))
		}

		if receipt.Ordinal == 1 && strings.TrimSpace(receipt.RetryReason) != "" {
			return lineageError(ReasonInvalidRetryReason, "attempt 1 must not carry a retry_reason")
		}

		if receipt.Score != nil {
			if math.IsNaN(*receipt.Score) || math.IsInf(*receipt.Score, 0) || *receipt.Score < 0.0 || *receipt.Score > 1.0 {
				return lineageError(ReasonInvalidScore, fmt.Sprintf("score %v on attempt %d is invalid (must be in [0, 1])", *receipt.Score, receipt.Ordinal))
			}
		}
	}

	if packet.RetryRegime == NoRetry && len(packet.Attempts) > 1 {
		return lineageError(ReasonRetryForbidden, fmt.Sprintf("policy %s forbids retry, but packet contains %d attempts", packet.RetryRegime, len(packet.Attempts)))
	}

	for i := 0; i < len(packet.Attempts)-1; i++ {
		prev := packet.Attempts[i]
		next := packet.Attempts[i+1]

		if prev.TerminalStatus == TerminalStatusTimeout && packet.RetryRegime == RetryOnFailureOnly {
			return lineageError(ReasonTimeoutRetryForbidden, fmt.Sprintf("attempt %d timed out; retry attempt %d is forbidden under %s policy", prev.Ordinal, next.Ordinal, packet.RetryRegime))
		}

		selectionRule := packet.SelectionRule
		if selectionRule == "" {
			selectionRule = FinalAttempt
		}
		if prev.TerminalStatus == TerminalStatusSuccess && selectionRule != BestOfN {
			return lineageError(ReasonRetryForbidden, fmt.Sprintf("attempt %d succeeded; subsequent attempt %d is forbidden under %s selection rule", prev.Ordinal, next.Ordinal, selectionRule))
		}
	}

	if len(packet.DiscoveredRoots) > 0 {
		if err := ReconcileDiscoveredAttempts(packet, packet.DiscoveredRoots); err != nil {
			return err
		}
	}

	if packet.Aggregate.TotalAttempts > 0 {
		expected := packet
		ComputeAggregate(&expected)
		if packet.Aggregate.TotalAttempts != expected.Aggregate.TotalAttempts {
			return lineageError(ReasonAggregateMismatch, fmt.Sprintf("aggregate total_attempts %d does not match %d", packet.Aggregate.TotalAttempts, expected.Aggregate.TotalAttempts))
		}
		if packet.Aggregate.TotalTokens != expected.Aggregate.TotalTokens {
			return lineageError(ReasonAggregateMismatch, fmt.Sprintf("aggregate total_tokens %d does not match sum %d", packet.Aggregate.TotalTokens, expected.Aggregate.TotalTokens))
		}
		if packet.Aggregate.TotalDurationMS != expected.Aggregate.TotalDurationMS {
			return lineageError(ReasonAggregateMismatch, fmt.Sprintf("aggregate total_duration_ms %d does not match sum %d", packet.Aggregate.TotalDurationMS, expected.Aggregate.TotalDurationMS))
		}
		if math.Abs(packet.Aggregate.TotalCostUSD-expected.Aggregate.TotalCostUSD) > 1e-4 {
			return lineageError(ReasonAggregateMismatch, fmt.Sprintf("aggregate total_cost_usd %f does not match sum %f", packet.Aggregate.TotalCostUSD, expected.Aggregate.TotalCostUSD))
		}
	}

	return nil
}

// CompareArmLineage verifies that two arms are comparable by ensuring equal retry policies
// and maximum attempt limits, refusing comparison with RETRY_REGIME_MISMATCH if they diverge.
func CompareArmLineage(armA, armB AttemptLineagePacket) error {
	if armA.RetryRegime != armB.RetryRegime {
		return lineageError(ReasonRetryRegimeMismatch, fmt.Sprintf("retry policy mismatch: arm A has %q, arm B has %q", armA.RetryRegime, armB.RetryRegime))
	}
	if armA.MaxAttempts != armB.MaxAttempts {
		return lineageError(ReasonRetryRegimeMismatch, fmt.Sprintf("max attempts mismatch: arm A has %d, arm B has %d", armA.MaxAttempts, armB.MaxAttempts))
	}
	ruleA := armA.SelectionRule
	if ruleA == "" {
		ruleA = FinalAttempt
	}
	ruleB := armB.SelectionRule
	if ruleB == "" {
		ruleB = FinalAttempt
	}
	if ruleA != ruleB {
		return lineageError(ReasonRetryRegimeMismatch, fmt.Sprintf("selection rule mismatch: arm A has %q, arm B has %q", ruleA, ruleB))
	}
	if len(armA.Attempts) > 0 {
		if err := ValidateAttemptLineage(armA); err != nil {
			return fmt.Errorf("arm A lineage invalid: %w", err)
		}
	}
	if len(armB.Attempts) > 0 {
		if err := ValidateAttemptLineage(armB); err != nil {
			return fmt.Errorf("arm B lineage invalid: %w", err)
		}
	}
	return nil
}

// ReconcileDiscoveredAttempts checks that every discovered attempt directory
// matches a declared attempt in the packet without hidden or missing directories.
func ReconcileDiscoveredAttempts(packet AttemptLineagePacket, discoveredRoots []string) error {
	declaredRoots := make(map[string]bool, len(packet.Attempts))
	for _, att := range packet.Attempts {
		declaredRoots[filepath.Clean(att.ArtifactRoot)] = true
	}
	for _, disc := range discoveredRoots {
		cleanDisc := filepath.Clean(disc)
		if !declaredRoots[cleanDisc] {
			return lineageError(ReasonHiddenAttempt, fmt.Sprintf("discovered attempt directory %q is missing from declared packet attempts", disc))
		}
	}
	if len(discoveredRoots) != len(packet.Attempts) {
		return lineageError(ReasonHiddenAttempt, fmt.Sprintf("discovered %d attempt directories, but packet declares %d", len(discoveredRoots), len(packet.Attempts)))
	}
	return nil
}

// DiscoverAttemptDirectories returns sorted directory paths under a unit directory.
func DiscoverAttemptDirectories(unitDir string) ([]string, error) {
	entries, err := os.ReadDir(unitDir)
	if err != nil {
		return nil, fmt.Errorf("read unit directory: %w", err)
	}
	var dirs []string
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dirs = append(dirs, filepath.Join(unitDir, entry.Name()))
	}
	return dirs, nil
}

// ReadAttemptLineagePacket reads and unmarshals an AttemptLineagePacket from disk.
func ReadAttemptLineagePacket(path string) (AttemptLineagePacket, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return AttemptLineagePacket{}, fmt.Errorf("read attempt lineage packet: %w", err)
	}
	var packet AttemptLineagePacket
	if err := json.Unmarshal(b, &packet); err != nil {
		return AttemptLineagePacket{}, fmt.Errorf("unmarshal attempt lineage packet: %w", err)
	}
	return packet, nil
}
