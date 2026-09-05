package stopgate

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Mode controls gate enforcement: off, shadow, or enforce.
type Mode string

const (
	ModeOff     Mode = "off"
	ModeShadow  Mode = "shadow"
	ModeEnforce Mode = "enforce"
)

// NormalizeMode normalizes raw strings into a valid Mode with a given default.
func NormalizeMode(raw string, defaultMode Mode) (Mode, error) {
	s := strings.ToLower(strings.TrimSpace(raw))
	switch s {
	case "":
		return defaultMode, nil
	case string(ModeOff):
		return ModeOff, nil
	case string(ModeShadow):
		return ModeShadow, nil
	case string(ModeEnforce):
		return ModeEnforce, nil
	default:
		return "", fmt.Errorf("invalid stopgate mode %q (want off, shadow, or enforce)", raw)
	}
}

// Action represents the gate's lifecycle instruction.
type Action string

const (
	ActionAllow    Action = "allow"    // allow stop to proceed (exit 0)
	ActionContinue Action = "continue" // block stop and continue session (exit 2)
	ActionStop     Action = "stop"     // stop session
)

// Stage is the rung of the graduated back-off ladder.
type Stage string

const (
	StageAllow  Stage = "allow"   // clean completion: allow stop
	StageNudge  Stage = "nudge"   // gentle alternative suggestion
	StageWarn   Stage = "warn"    // relevance-decision warning
	StageFinal  Stage = "final"   // final auto-continue
	StageGiveUp Stage = "give-up" // bounded stand-down: allow stop
)

func (s Stage) String() string {
	return string(s)
}

// Kind is the coarse category of the stop decision.
type Kind string

const (
	KindClean     Kind = "clean"
	KindContinue  Kind = "continue"
	KindStandDown Kind = "stand-down"
	KindOff       Kind = "off"
	KindShadow    Kind = "shadow"
	KindFailOpen  Kind = "fail-open"
)

func (k Kind) String() string {
	return string(k)
}

// Disposition is the closed set of terminal outcomes the stop gate can produce.
type Disposition string

const (
	// Clean stops
	DispCleanCompletion Disposition = "clean_completion"
	DispCleanWrapup     Disposition = "clean_wrapup"

	// Continues
	DispToolFeedbackContinue     Disposition = "tool_feedback_continue"
	DispDenyAllContinue          Disposition = "deny_all_continue"
	DispSameIssueContinue        Disposition = "same_issue_continue"
	DispHandoffBlock             Disposition = "handoff_block"
	DispClaimUnwitnessedContinue Disposition = "claim_unwitnessed_continue"

	// Stand-downs / Give-ups
	DispBlindGiveUp            Disposition = "blind_give_up"
	DispSameIssueGiveUp        Disposition = "same_issue_give_up"
	DispToolFeedbackGiveUp     Disposition = "tool_feedback_give_up"
	DispHandoffGiveUp          Disposition = "handoff_give_up"
	DispClaimUnwitnessedGiveUp Disposition = "claim_unwitnessed_give_up"

	// Witness outcomes
	DispClaimWitnessed     Disposition = "claim_witnessed"
	DispClaimWitnessShadow Disposition = "claim_witness_shadow"

	// Mode / System outcomes
	DispModeOff  Disposition = "mode_off"
	DispShadow   Disposition = "shadow"
	DispFailOpen Disposition = "fail_open"
)

func (d Disposition) String() string {
	return string(d)
}

// Kind maps a Disposition to its coarse classification Kind.
func (d Disposition) Kind() Kind {
	switch d {
	case DispCleanCompletion, DispCleanWrapup, DispClaimWitnessed:
		return KindClean
	case DispToolFeedbackContinue, DispDenyAllContinue, DispSameIssueContinue, DispHandoffBlock, DispClaimUnwitnessedContinue:
		return KindContinue
	case DispBlindGiveUp, DispSameIssueGiveUp, DispToolFeedbackGiveUp, DispHandoffGiveUp, DispClaimUnwitnessedGiveUp:
		return KindStandDown
	case DispModeOff:
		return KindOff
	case DispShadow, DispClaimWitnessShadow:
		return KindShadow
	default:
		return KindFailOpen
	}
}

// WitnessClaim captures an assertion that work was finished.
type WitnessClaim struct {
	Claimed   bool   `json:"claimed"`
	Witnessed bool   `json:"witnessed"`
	Commit    string `json:"commit,omitempty"`
	Reason    string `json:"reason,omitempty"`
	Detail    string `json:"detail,omitempty"`
}

// Decision represents the outcome of evaluating a stop gate.
type Decision struct {
	Action      Action      `json:"action"`
	Stage       Stage       `json:"stage"`
	Disposition Disposition `json:"disposition"`
	Kind        Kind        `json:"kind"`
	Blocked     bool        `json:"blocked"`
	ExitCode    int         `json:"exit_code"`
	Signal      string      `json:"signal,omitempty"` // "clean", "blind", "same-issue", "tool-feedback", "witness"
	Depth       int         `json:"depth,omitempty"`
	Bound       int         `json:"bound,omitempty"`
	Guidance    string      `json:"guidance,omitempty"`     // continuation prompt fed back to model
	OperatorMsg string      `json:"operator_msg,omitempty"` // operator-facing stand-down note
	Reason      string      `json:"reason,omitempty"`
	Note        string      `json:"note,omitempty"`
}

// ShouldContinue returns true if the decision dictates continuing execution.
func (d Decision) ShouldContinue() bool {
	return d.Action == ActionContinue || (d.Blocked && d.ExitCode == 2)
}

// BoundaryRefusalReceipt records verified evidence of a boundary refusal from the capability floor.
type BoundaryRefusalReceipt struct {
	Tool        string         `json:"tool,omitempty"`
	Reason      string         `json:"reason,omitempty"`      // abi.ReasonCode name or dos.toml refusal token
	Disposition string         `json:"disposition,omitempty"` // TERMINAL, RETRYABLE, WAIT, etc.
	Detail      string         `json:"detail,omitempty"`
	Signature   string         `json:"signature,omitempty"`
	Verified    bool           `json:"verified,omitempty"`
	Token       string         `json:"token,omitempty"`       // optional legacy token
	ReasonCode  abi.ReasonCode `json:"reason_code,omitempty"` // optional typed code
	Terminal    bool           `json:"terminal,omitempty"`    // explicit assertion of terminal boundary
	Transient   bool           `json:"transient,omitempty"`   // explicit assertion of transient hurdle
}

// BoundaryInput contains the telemetry and state for turn boundary adjudication.
type BoundaryInput struct {
	SessionID               string
	Turn                    int
	ConsecutiveDenyAll      int
	ConsecutiveSameIssue    int
	UseSameIssue            bool
	ConsecutiveToolFeedback int
	NotedNoAllowedPath      bool
	WitnessClaim            *WitnessClaim
	WitnessBlockCount       int
	FinalGate               func() (satisfied bool, missingWitness string)
	StopHookActive          bool

	// Verified boundary refusal evidence required when NotedNoAllowedPath is true.
	RefusalReceipt         *BoundaryRefusalReceipt
	BoundaryRefusalReceipt *BoundaryRefusalReceipt
	RefusalToken           string
	ReasonCode             abi.ReasonCode
}
