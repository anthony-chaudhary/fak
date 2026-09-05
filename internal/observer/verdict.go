package observer

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// Closed step classification vocabulary.
type StepVerdict string

const (
	StepAdvance StepVerdict = "STEP_ADVANCE"
	StepChurn   StepVerdict = "STEP_CHURN"
	StepRegress StepVerdict = "STEP_REGRESS"

	// Direct uppercase aliases for explicit closed vocabulary matching.
	STEP_ADVANCE = StepAdvance
	STEP_CHURN   = StepChurn
	STEP_REGRESS = StepRegress
)

// IsValid reports whether v is a member of the closed StepVerdict vocabulary.
func (v StepVerdict) IsValid() bool {
	switch v {
	case StepAdvance, StepChurn, StepRegress:
		return true
	default:
		return false
	}
}

// StepVerdicts returns the complete closed vocabulary of StepVerdict.
func StepVerdicts() []StepVerdict {
	return []StepVerdict{StepAdvance, StepChurn, StepRegress}
}

// ParseStepVerdict parses and strictly validates a StepVerdict against the closed vocabulary.
func ParseStepVerdict(s string) (StepVerdict, error) {
	v := StepVerdict(strings.ToUpper(strings.TrimSpace(s)))
	if !v.IsValid() {
		return "", fmt.Errorf("%w: %q (allowed: %v)", ErrInvalidStepVerdict, s, StepVerdicts())
	}
	return v, nil
}

func (v StepVerdict) String() string {
	return string(v)
}

// Witness verification types for mutation claims.
type WitnessVerdict string

const (
	WitnessDiffConfirmed    WitnessVerdict = "WITNESS_DIFF_CONFIRMED"
	WitnessUnwitnessedClaim WitnessVerdict = "WITNESS_UNWITNESSED_CLAIM"

	// Direct uppercase aliases for explicit witness verification type matching.
	WITNESS_DIFF_CONFIRMED    = WitnessDiffConfirmed
	WITNESS_UNWITNESSED_CLAIM = WitnessUnwitnessedClaim
)

// IsValid reports whether w is a member of the closed WitnessVerdict vocabulary.
func (w WitnessVerdict) IsValid() bool {
	switch w {
	case WitnessDiffConfirmed, WitnessUnwitnessedClaim:
		return true
	default:
		return false
	}
}

// WitnessVerdicts returns the complete closed vocabulary of WitnessVerdict.
func WitnessVerdicts() []WitnessVerdict {
	return []WitnessVerdict{WitnessDiffConfirmed, WitnessUnwitnessedClaim}
}

// ParseWitnessVerdict parses and strictly validates a WitnessVerdict against the closed vocabulary.
func ParseWitnessVerdict(s string) (WitnessVerdict, error) {
	w := WitnessVerdict(strings.ToUpper(strings.TrimSpace(s)))
	if !w.IsValid() {
		return "", fmt.Errorf("%w: %q (allowed: %v)", ErrInvalidWitnessVerdict, s, WitnessVerdicts())
	}
	return w, nil
}

func (w WitnessVerdict) String() string {
	return string(w)
}

// Sentinel refusal and barrier errors.
var (
	ErrPoolClosed            = errors.New("observer: pool is closed")
	ErrPoolStopped           = errors.New("observer: pool is stopped")
	ErrPoolNotStarted        = errors.New("observer: pool is not started")
	ErrQueueFull             = errors.New("observer: work queue is full")
	ErrBarrierTimeout        = errors.New("observer: barrier timeout exceeded")
	ErrChurnRefused          = errors.New("observer: step refused due to churn loop")
	ErrRegressRefused        = errors.New("observer: step refused due to regression loop")
	ErrUnwitnessedDiff       = errors.New("observer: mutating step lacks confirmed diff witness")
	ErrInvalidStepVerdict    = errors.New("observer: invalid step verdict; not in closed vocabulary")
	ErrInvalidWitnessVerdict = errors.New("observer: invalid witness verdict; not in closed vocabulary")
)

// StepObservation holds execution, classification, and witness facts for an agent step.
type StepObservation struct {
	ID             string         `json:"id,omitempty"`
	SessionID      string         `json:"session_id,omitempty"`
	Tool           string         `json:"tool"`
	Args           any            `json:"args,omitempty"`
	Result         any            `json:"result,omitempty"`
	StepVerdict    StepVerdict    `json:"step_verdict"`
	WitnessVerdict WitnessVerdict `json:"witness_verdict"`
	Duration       time.Duration  `json:"duration"`
	Timestamp      time.Time      `json:"timestamp"`
	Reason         string         `json:"reason,omitempty"`
	Mutating       bool           `json:"mutating,omitempty"`
	Diff           string         `json:"diff,omitempty"`
	Error          string         `json:"error,omitempty"`
	CachedPrefix   bool           `json:"cached_prefix,omitempty"`
	BarrierLatency time.Duration  `json:"barrier_latency,omitempty"`
}

// Validate checks that if StepVerdict or WitnessVerdict is populated, it belongs to the closed vocabulary.
func (o StepObservation) Validate() error {
	if o.StepVerdict != "" && !o.StepVerdict.IsValid() {
		return fmt.Errorf("%w: %q", ErrInvalidStepVerdict, o.StepVerdict)
	}
	if o.WitnessVerdict != "" && !o.WitnessVerdict.IsValid() {
		return fmt.Errorf("%w: %q", ErrInvalidWitnessVerdict, o.WitnessVerdict)
	}
	return nil
}

// IsMutating reports whether the observation represents a mutating operation.
func (o StepObservation) IsMutating() bool {
	if o.Mutating {
		return true
	}
	return IsMutatingTool(o.Tool)
}

// IsReadOnly reports whether the observation represents a read-only exploration turn.
func (o StepObservation) IsReadOnly() bool {
	if o.Mutating {
		return false
	}
	return IsReadOnlyTool(o.Tool)
}

// Summary returns a concise diagnostic string for the observation.
func (o StepObservation) Summary() string {
	return fmt.Sprintf("[%s/%s] tool=%s mutating=%v cached=%v latency=%s reason=%s",
		o.StepVerdict, o.WitnessVerdict, o.Tool, o.IsMutating(), o.CachedPrefix, o.Duration, o.Reason)
}

// IsReadOnlyTool reports whether tool is a read-only exploration tool (Read, Grep, Glob).
func IsReadOnlyTool(tool string) bool {
	t := strings.ToLower(strings.TrimSpace(tool))
	switch t {
	case "read", "grep", "glob", "fak_read", "list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource":
		return true
	default:
		return false
	}
}

// IsMutatingTool reports whether tool is a mutating operation (Edit, Write, git commit, etc.).
func IsMutatingTool(tool string) bool {
	if IsReadOnlyTool(tool) {
		return false
	}
	t := strings.ToLower(strings.TrimSpace(tool))
	switch t {
	case "edit", "write", "bash", "git commit", "commit", "fak_syscall":
		return true
	default:
		if strings.HasPrefix(t, "git ") {
			if strings.Contains(t, "status") || strings.Contains(t, "diff") || strings.Contains(t, "log") || strings.Contains(t, "rev-parse") {
				return false
			}
			return true
		}
		return false
	}
}
