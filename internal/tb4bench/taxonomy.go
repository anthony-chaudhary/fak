package tb4bench

import (
	"errors"
	"fmt"
)

// FailureReason is a closed enum representing task outcomes.
type FailureReason string

const (
	ReasonSolved            FailureReason = "SOLVED"
	ReasonTestFailed        FailureReason = "TEST_FAILED"
	ReasonTimeoutAgent      FailureReason = "TIMEOUT_AGENT"
	ReasonTimeoutOracle     FailureReason = "TIMEOUT_ORACLE"
	ReasonContextExhausted  FailureReason = "CONTEXT_EXHAUSTED"
	ReasonToolCallMalformed FailureReason = "TOOL_CALL_MALFORMED"
	ReasonPolicyBlock       FailureReason = "POLICY_BLOCK"
	ReasonContainerCrash    FailureReason = "CONTAINER_CRASH"
	ReasonServerCrash       FailureReason = "SERVER_CRASH"
	ReasonUnclassified      FailureReason = "UNCLASSIFIED" // Forbidden in official runs
)

var closedReasons = map[FailureReason]bool{
	ReasonSolved:            true,
	ReasonTestFailed:        true,
	ReasonTimeoutAgent:      true,
	ReasonTimeoutOracle:     true,
	ReasonContextExhausted:  true,
	ReasonToolCallMalformed: true,
	ReasonPolicyBlock:       true,
	ReasonContainerCrash:    true,
	ReasonServerCrash:       true,
}

// AllValidReasons returns a slice of all permissible non-unclassified reasons.
func AllValidReasons() []FailureReason {
	return []FailureReason{
		ReasonSolved,
		ReasonTestFailed,
		ReasonTimeoutAgent,
		ReasonTimeoutOracle,
		ReasonContextExhausted,
		ReasonToolCallMalformed,
		ReasonPolicyBlock,
		ReasonContainerCrash,
		ReasonServerCrash,
	}
}

// IsValidReason checks if a reason belongs to the closed valid taxonomy.
func IsValidReason(r FailureReason) bool {
	return closedReasons[r]
}

// ValidateReason returns an error if reason is unclassified or unknown.
func ValidateReason(r FailureReason) error {
	if r == ReasonUnclassified {
		return errors.New("UNCLASSIFIED failure reason is strictly forbidden in official TB4 runs")
	}
	if !closedReasons[r] {
		return fmt.Errorf("unknown failure reason %q not in closed vocabulary", r)
	}
	return nil
}
