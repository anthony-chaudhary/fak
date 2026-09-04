package schedcontract

import (
	"crypto/subtle"
	"errors"
	"fmt"
	"strings"
	"time"
)

// Schema identifies the canonical versioned specification for scheduler contracts.
const Schema = "fak.sched-contract/1"

var (
	// ErrInvalidSchema indicates that the contract payload schema does not match the expected version.
	ErrInvalidSchema = errors.New("invalid or mismatched scheduler contract schema")

	// ErrInvalidPriority indicates an unknown or unassigned execution priority tier.
	ErrInvalidPriority = errors.New("invalid task scheduling priority tier")

	// ErrExpiredDeadline indicates that the task deadline is zero or has already elapsed.
	ErrExpiredDeadline = errors.New("task scheduling deadline is zero or has already expired")

	// ErrInvalidToken indicates an execution token failing structural or cryptographic validation.
	ErrInvalidToken = errors.New("execution token validation failed")

	// ErrConstraintViolation indicates an impossible or contradictory scheduling constraint set.
	ErrConstraintViolation = errors.New("schedule constraint violation detected")

	// ErrInvariantViolated indicates that a required runtime contract invariant was breached.
	ErrInvariantViolated = errors.New("scheduler contract invariant violated")
)

// Priority defines the execution urgency tier of a scheduled agent unit of work.
type Priority string

const (
	// PriorityBackground designates non-urgent background batch work executed only under idle capacity.
	PriorityBackground Priority = "background"

	// PriorityLow designates low-priority background maintenance tasks.
	PriorityLow Priority = "low"

	// PriorityNormal designates standard interactive agent execution turns.
	PriorityNormal Priority = "normal"

	// PriorityHigh designates latency-sensitive turns with elevated dispatch preference.
	PriorityHigh Priority = "high"

	// PriorityCritical designates pre-emptive or incident mitigation runs requiring immediate allocation.
	PriorityCritical Priority = "critical"
)

// Valid verifies whether the priority corresponds to a recognized system scheduling tier.
func (p Priority) Valid() bool {
	switch p {
	case PriorityBackground, PriorityLow, PriorityNormal, PriorityHigh, PriorityCritical:
		return true
	default:
		return false
	}
}

// Rank calculates the numerical ordering for priority comparison, with higher values indicating greater scheduling urgency.
func (p Priority) Rank() int {
	switch p {
	case PriorityBackground:
		return 10
	case PriorityLow:
		return 20
	case PriorityNormal:
		return 30
	case PriorityHigh:
		return 40
	case PriorityCritical:
		return 50
	default:
		return -1
	}
}

// ExecutionToken encapsulates authorized lease credentials permitting task execution within a designated repo lane.
type ExecutionToken struct {
	TokenID      string    `json:"token_id"`
	Issuer       string    `json:"issuer"`
	Subject      string    `json:"subject"`
	Lane         string    `json:"lane"`
	IssuedAt     time.Time `json:"issued_at"`
	ExpiresAt    time.Time `json:"expires_at"`
	Capabilities []string  `json:"capabilities"`
	Signature    string    `json:"signature"`
	Nonce        string    `json:"nonce"`
}

// Validate verifies structural completeness, temporal validity, and cryptographic nonces on the execution token.
//
// Fail-closed guard: Any missing token identifier, issuer, signature, or nonce results in immediate failure.
// Invariant: The token issuance timestamp must strictly precede its expiration boundary.
func (t ExecutionToken) Validate(now time.Time) error {
	if strings.TrimSpace(t.TokenID) == "" {
		return fmt.Errorf("%w: missing token id", ErrInvalidToken)
	}
	if strings.TrimSpace(t.Issuer) == "" {
		return fmt.Errorf("%w: missing token issuer", ErrInvalidToken)
	}
	if strings.TrimSpace(t.Subject) == "" {
		return fmt.Errorf("%w: missing token subject", ErrInvalidToken)
	}
	if strings.TrimSpace(t.Signature) == "" {
		return fmt.Errorf("%w: missing cryptographic signature", ErrInvalidToken)
	}
	if strings.TrimSpace(t.Nonce) == "" {
		return fmt.Errorf("%w: missing replay protection nonce", ErrInvalidToken)
	}
	if t.IssuedAt.IsZero() || t.ExpiresAt.IsZero() {
		return fmt.Errorf("%w: zero timestamp in token validity window", ErrInvalidToken)
	}
	if !t.ExpiresAt.After(t.IssuedAt) {
		return fmt.Errorf("%w: token expiration %v must be after issuance %v", ErrInvalidToken, t.ExpiresAt, t.IssuedAt)
	}
	if !now.IsZero() {
		if now.Before(t.IssuedAt) {
			return fmt.Errorf("%w: token cannot be accepted before issuance time %v", ErrInvalidToken, t.IssuedAt)
		}
		if !now.Before(t.ExpiresAt) {
			return fmt.Errorf("%w: token expired at %v (current time: %v)", ErrInvalidToken, t.ExpiresAt, now)
		}
	}
	return nil
}

// HasPermit checks whether the token grants permission for the specified functional capability permit.
func (t ExecutionToken) HasPermit(permitName string) bool {
	target := strings.TrimSpace(strings.ToLower(permitName))
	for _, c := range t.Capabilities {
		if strings.TrimSpace(strings.ToLower(c)) == target {
			return true
		}
	}
	return false
}

// VerifySignature compares the token signature against an expected authorization signature using constant-time evaluation.
//
// Fail-closed guard: Signature comparison executes in constant time to prevent timing attacks during token verification.
func (t ExecutionToken) VerifySignature(expectedSignature string) bool {
	if t.Signature == "" || expectedSignature == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(t.Signature), []byte(expectedSignature)) == 1
}

// ScheduleConstraints governs admission boundaries and resource ceilings for task execution.
type ScheduleConstraints struct {
	MaxConcurrency       int           `json:"max_concurrency"`
	MinWaitDuration      time.Duration `json:"min_wait_duration"`
	MaxLeaseDuration     time.Duration `json:"max_lease_duration"`
	AllowedLanes         []string      `json:"allowed_lanes"`
	RequiredCapabilities []string      `json:"required_capabilities"`
	MemoryLimitBytes     int64         `json:"memory_limit_bytes"`
	ExclusiveOnly        bool          `json:"exclusive_only"`
}

// Validate checks that the constraints define an achievable and non-contradictory bounding envelope.
//
// Invariant: The maximum lease duration must strictly exceed the minimum wait duration.
func (c ScheduleConstraints) Validate() error {
	if c.MaxConcurrency < 0 {
		return fmt.Errorf("%w: max concurrency cannot be negative", ErrConstraintViolation)
	}
	if c.MinWaitDuration < 0 {
		return fmt.Errorf("%w: min wait duration cannot be negative", ErrConstraintViolation)
	}
	if c.MaxLeaseDuration < 0 {
		return fmt.Errorf("%w: max lease duration cannot be negative", ErrConstraintViolation)
	}
	if c.MaxLeaseDuration > 0 && c.MinWaitDuration > 0 && c.MaxLeaseDuration < c.MinWaitDuration {
		return fmt.Errorf("%w: max lease duration %v is smaller than min wait duration %v", ErrConstraintViolation, c.MaxLeaseDuration, c.MinWaitDuration)
	}
	if c.MemoryLimitBytes < 0 {
		return fmt.Errorf("%w: memory limit bytes cannot be negative", ErrConstraintViolation)
	}
	return nil
}

// AllowsLane checks whether the given lane satisfies the whitelist constraint when restrictions are active.
func (c ScheduleConstraints) AllowsLane(lane string) bool {
	if len(c.AllowedLanes) == 0 {
		return true
	}
	target := strings.TrimSpace(strings.ToLower(lane))
	for _, l := range c.AllowedLanes {
		if strings.TrimSpace(strings.ToLower(l)) == target {
			return true
		}
	}
	return false
}

// ExecutionContract defines the formal admission pact between the agent queue dispatcher and execution workers.
type ExecutionContract struct {
	ContractID  string              `json:"contract_id"`
	Schema      string              `json:"schema"`
	TaskID      string              `json:"task_id"`
	Lane        string              `json:"lane"`
	Priority    Priority            `json:"priority"`
	Deadline    time.Time           `json:"deadline"`
	Token       ExecutionToken      `json:"token"`
	Constraints ScheduleConstraints `json:"constraints"`
	CreatedAt   time.Time           `json:"created_at"`
}

// Validate performs structural, chronological, and constraint verification on the execution contract.
//
// Contract: All execution contracts must maintain non-zero task identifiers and valid positive timeouts before admission.
// Fail-closed guard: Any corrupted schema, blank identifiers, or invalid tokens fail admission immediately.
func (c ExecutionContract) Validate(now time.Time) error {
	if c.Schema != Schema {
		return fmt.Errorf("%w: got %q, expected %q", ErrInvalidSchema, c.Schema, Schema)
	}
	if strings.TrimSpace(c.ContractID) == "" {
		return fmt.Errorf("%w: missing contract id", ErrConstraintViolation)
	}
	if strings.TrimSpace(c.TaskID) == "" {
		return fmt.Errorf("%w: missing task id", ErrConstraintViolation)
	}
	if strings.TrimSpace(c.Lane) == "" {
		return fmt.Errorf("%w: missing lane specification", ErrConstraintViolation)
	}
	if !c.Priority.Valid() {
		return fmt.Errorf("%w: unknown tier %q", ErrInvalidPriority, c.Priority)
	}
	if c.Deadline.IsZero() {
		return fmt.Errorf("%w: deadline cannot be zero", ErrExpiredDeadline)
	}
	if !now.IsZero() && !c.Deadline.After(now) {
		return fmt.Errorf("%w: deadline %v is at or before current evaluation time %v", ErrExpiredDeadline, c.Deadline, now)
	}
	if err := c.Token.Validate(now); err != nil {
		return fmt.Errorf("contract token invalid: %w", err)
	}
	if err := c.Constraints.Validate(); err != nil {
		return fmt.Errorf("contract constraints invalid: %w", err)
	}
	if !c.Constraints.AllowsLane(c.Lane) {
		return fmt.Errorf("%w: lane %q not permitted by schedule constraints", ErrConstraintViolation, c.Lane)
	}
	return nil
}

// CheckInvariants enforces cross-field consistency invariants across the contract, token, and constraints.
//
// Invariant: The execution token lease duration must not expire prior to the scheduled evaluation deadline.
// Invariant: The token lane must match the contract target lane when the token specifies a lane constraint.
// Invariant: Critical priority tasks require exclusive execution constraints or explicit security capabilities.
// Contract: Schedule constraints must not allow maximum lease duration to exceed remaining time until deadline.
func CheckInvariants(c *ExecutionContract, now time.Time) error {
	if c == nil {
		return fmt.Errorf("%w: nil contract reference", ErrInvariantViolated)
	}

	// Invariant 1: Structural validity must pass before evaluating cross-field relationships.
	if err := c.Validate(now); err != nil {
		return fmt.Errorf("%w: underlying contract validation failed: %v", ErrInvariantViolated, err)
	}

	// Invariant 2: The execution token must remain valid at least until the task deadline.
	if c.Token.ExpiresAt.Before(c.Deadline) {
		return fmt.Errorf("%w: token expiration %v occurs before task deadline %v", ErrInvariantViolated, c.Token.ExpiresAt, c.Deadline)
	}

	// Invariant 3: If the execution token binds to an explicit lane, it must match the contract lane.
	if strings.TrimSpace(c.Token.Lane) != "" {
		if !strings.EqualFold(strings.TrimSpace(c.Token.Lane), strings.TrimSpace(c.Lane)) {
			return fmt.Errorf("%w: token lane %q does not match contract lane %q", ErrInvariantViolated, c.Token.Lane, c.Lane)
		}
	}

	// Invariant 4: Required capabilities in constraints must all be satisfied by the token.
	for _, reqCap := range c.Constraints.RequiredCapabilities {
		if !c.Token.HasPermit(reqCap) {
			return fmt.Errorf("%w: missing required execution capability %q", ErrInvariantViolated, reqCap)
		}
	}

	// Invariant 5: Critical priority tasks demand exclusive execution or dedicated emergency capability.
	if c.Priority == PriorityCritical {
		hasEmergencyCap := c.Token.HasPermit("emergency_override") || c.Token.HasPermit("critical_preempt")
		if !c.Constraints.ExclusiveOnly && !hasEmergencyCap {
			return fmt.Errorf("%w: critical priority requires exclusive execution mode or emergency capability", ErrInvariantViolated)
		}
	}

	// Invariant 6: MaxLeaseDuration cannot exceed the remaining duration until the deadline.
	if !now.IsZero() && c.Constraints.MaxLeaseDuration > 0 {
		remaining := c.Deadline.Sub(now)
		if c.Constraints.MaxLeaseDuration > remaining {
			return fmt.Errorf("%w: max lease duration %v exceeds time remaining until deadline %v", ErrInvariantViolated, c.Constraints.MaxLeaseDuration, remaining)
		}
	}

	return nil
}
