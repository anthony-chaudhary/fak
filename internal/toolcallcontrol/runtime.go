package toolcallcontrol

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Mode selects whether runtime verdicts are disabled, observed, or applied.
type Mode string

const (
	ModeOff     Mode = "off"
	ModeShadow  Mode = "shadow"
	ModeEnforce Mode = "enforce"
)

// ParseMode is fail-safe for optimization: malformed values disable control.
func ParseMode(raw string) Mode {
	switch Mode(strings.ToLower(strings.TrimSpace(raw))) {
	case ModeShadow:
		return ModeShadow
	case ModeEnforce:
		return ModeEnforce
	default:
		return ModeOff
	}
}

// RuntimeState is the bounded per-session knowledge used by the live hook.
// Epoch advances after any potentially state-changing tool, invalidating reuse.
type RuntimeState struct {
	Epoch        uint64               `json:"epoch"`
	Observations []RuntimeObservation `json:"observations,omitempty"`
}

// RuntimeObservation records a completed read at one mutation epoch.
type RuntimeObservation struct {
	Fingerprint string `json:"fingerprint"`
	Epoch       uint64 `json:"epoch"`
	ResultRef   string `json:"result_ref,omitempty"`
}

// RuntimeInput is the provider-neutral subset available at a pre/post tool seam.
type RuntimeInput struct {
	SessionID   string
	CallID      string
	Tool        string
	Args        json.RawMessage
	ReadOnly    bool
	Succeeded   bool
	ResultRef   string
	PromptUnits int64
	Declaration OutcomeDeclaration
	ExitCode    *int
	Output      json.RawMessage
}

// OutcomeClass is the closed outcome vocabulary carried by post-tool receipts.
// ExpectedNegative is a class only when a caller declared that expectation;
// text heuristics can refine an unexpected failure but can never make it healthy.
type OutcomeClass string

const (
	OutcomeSuccess                  OutcomeClass = "success"
	OutcomeExpectedNegative         OutcomeClass = "expected_negative"
	OutcomeGuardRefusal             OutcomeClass = "guard_refusal"
	OutcomeTestFailure              OutcomeClass = "test_failure"
	OutcomeTimeoutInterruption      OutcomeClass = "timeout_interruption"
	OutcomeContractDefect           OutcomeClass = "contract_defect"
	OutcomeUnexpectedCommandFailure OutcomeClass = "unexpected_command_failure"
	OutcomeUnknown                  OutcomeClass = "unknown"
)

// OutcomeProjection is the operator-facing grouping. Only a structural
// expected-negative declaration enters the non-red expected_negative group.
type OutcomeProjection string

const (
	ProjectionSuccess           OutcomeProjection = "success"
	ProjectionExpectedNegative  OutcomeProjection = "expected_negative"
	ProjectionUnexpectedFailure OutcomeProjection = "unexpected_failure"
)

// OutcomeDeclaration is out-of-band call-site intent. ExpectedNegativeSet
// distinguishes an explicit false from an absent marker so contradictory
// declarations can fail closed instead of being guessed through.
type OutcomeDeclaration struct {
	ExpectedNegative    bool
	ExpectedNegativeSet bool
	Class               OutcomeClass
	Invalid             bool
}

// RuntimeReceipt records classification without changing the tool's result.
// ExitCode and Output retain the underlying evidence even when the operator
// projection moves an expected negative out of the unexpected-failure group.
type RuntimeReceipt struct {
	Class            OutcomeClass      `json:"class"`
	UnderlyingClass  OutcomeClass      `json:"underlying_class,omitempty"`
	Projection       OutcomeProjection `json:"projection"`
	ExpectedNegative bool              `json:"expected_negative"`
	Succeeded        bool              `json:"succeeded"`
	ExitCode         *int              `json:"exit_code,omitempty"`
	Output           json.RawMessage   `json:"output,omitempty"`
	Reason           string            `json:"reason"`
}

// ClassifyOutcome deterministically projects a completed call into the closed
// receipt vocabulary. Invalid declarations and ambiguous heuristic evidence
// remain visible as unknown unexpected failures.
func ClassifyOutcome(in RuntimeInput) RuntimeReceipt {
	receipt := RuntimeReceipt{
		Succeeded:        in.Succeeded,
		ExpectedNegative: in.Declaration.ExpectedNegative,
		ExitCode:         cloneInt(in.ExitCode),
		Output:           append(json.RawMessage(nil), in.Output...),
	}
	decl := in.Declaration
	if decl.Invalid || !validDeclaredOutcome(decl.Class) {
		return unexpectedReceipt(receipt, OutcomeUnknown, "invalid_outcome_declaration")
	}
	if decl.Class == OutcomeExpectedNegative {
		if decl.ExpectedNegativeSet && !decl.ExpectedNegative {
			return unexpectedReceipt(receipt, OutcomeUnknown, "contradictory_outcome_declaration")
		}
		decl.ExpectedNegative = true
		decl.ExpectedNegativeSet = true
	}
	receipt.ExpectedNegative = decl.ExpectedNegative

	if in.Succeeded {
		if decl.ExpectedNegative || (decl.Class != "" && decl.Class != OutcomeSuccess) {
			return unexpectedReceipt(receipt, OutcomeUnknown, "declared_negative_succeeded")
		}
		receipt.Class = OutcomeSuccess
		receipt.Projection = ProjectionSuccess
		receipt.Reason = "tool_succeeded"
		return receipt
	}

	if decl.Class == OutcomeSuccess {
		return unexpectedReceipt(receipt, OutcomeUnknown, "declared_success_failed")
	}
	if decl.ExpectedNegative {
		underlying, reason := inferredFailure(in)
		if decl.Class != "" && decl.Class != OutcomeExpectedNegative {
			underlying, reason = decl.Class, "declared_failure_class"
		}
		if underlying == OutcomeUnknown {
			return unexpectedReceipt(receipt, OutcomeUnknown, reason)
		}
		receipt.Class = OutcomeExpectedNegative
		receipt.UnderlyingClass = underlying
		receipt.Projection = ProjectionExpectedNegative
		receipt.Reason = "declared_expected_negative"
		return receipt
	}
	if decl.Class != "" {
		return unexpectedReceipt(receipt, decl.Class, "declared_failure_class")
	}
	class, reason := inferredFailure(in)
	return unexpectedReceipt(receipt, class, reason)
}

func validDeclaredOutcome(class OutcomeClass) bool {
	switch class {
	case "", OutcomeSuccess, OutcomeExpectedNegative, OutcomeGuardRefusal,
		OutcomeTestFailure, OutcomeTimeoutInterruption, OutcomeContractDefect,
		OutcomeUnexpectedCommandFailure:
		return true
	default:
		return false
	}
}

func unexpectedReceipt(receipt RuntimeReceipt, class OutcomeClass, reason string) RuntimeReceipt {
	receipt.Class = class
	receipt.Projection = ProjectionUnexpectedFailure
	receipt.Reason = reason
	return receipt
}

func inferredFailure(in RuntimeInput) (OutcomeClass, string) {
	text := strings.ToLower(string(in.Args) + "\n" + string(in.Output))
	var signals []OutcomeClass
	add := func(class OutcomeClass, matched bool) {
		if matched {
			signals = append(signals, class)
		}
	}
	add(OutcomeGuardRefusal, containsAny(text,
		"policy_block", "guard_refusal", "guard refusal", `"permissiondecision":"deny"`,
		"permission decision: deny", "refused by guard", "preview refusal"))
	add(OutcomeTestFailure, containsAny(text,
		"go test", "make test", "test.ps1", "pytest", "cargo test", "npm test", "pnpm test", "yarn test",
		"--- fail:", "\nfail\t"))
	add(OutcomeTimeoutInterruption, timeoutExitCode(in.ExitCode) || containsAny(text,
		`"timed_out":true`, `"interrupted":true`, "timed out", "deadline exceeded", "operation canceled",
		"operation cancelled", "interrupted"))
	add(OutcomeContractDefect, containsAny(text,
		"invalid tool arguments", "invalid arguments", "contract defect", "schema validation", "missing required argument"))

	switch len(signals) {
	case 0:
		return OutcomeUnexpectedCommandFailure, "unclassified_command_failure"
	case 1:
		return signals[0], "typed_failure_evidence"
	default:
		return OutcomeUnknown, "ambiguous_failure_evidence"
	}
}

func containsAny(text string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(text, needle) {
			return true
		}
	}
	return false
}

func timeoutExitCode(code *int) bool {
	if code == nil {
		return false
	}
	switch *code {
	case 124, 130, 137, -1073741510:
		return true
	default:
		return false
	}
}

func cloneInt(value *int) *int {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

// RuntimeVerdict is emitted for every shadow/enforce pre-tool evaluation.
type RuntimeVerdict struct {
	Mode               Mode   `json:"mode"`
	Action             Action `json:"action"`
	Reason             string `json:"reason"`
	Fingerprint        string `json:"fingerprint,omitempty"`
	ResultRef          string `json:"result_ref,omitempty"`
	Epoch              uint64 `json:"epoch"`
	Applied            bool   `json:"applied"`
	ReplayUnitsSaved   int64  `json:"replay_units_saved"`
	ReplaySquaredSaved string `json:"replay_squared_saved,omitempty"`
	PromptBucket       string `json:"prompt_bucket,omitempty"`
}

// Before applies exact-fresh reuse against state. Shadow records the same
// counterfactual without changing execution.
func Before(mode Mode, state RuntimeState, in RuntimeInput) RuntimeVerdict {
	v := RuntimeVerdict{Mode: mode, Action: Allow, Reason: "mode_off", Epoch: state.Epoch}
	if mode == ModeOff {
		return v
	}
	if strings.TrimSpace(in.Tool) == "" || len(in.Args) == 0 {
		v.Reason = "malformed_fail_open"
		return v
	}
	if !in.ReadOnly {
		v.Reason = "mutation_not_suppressed"
		return v
	}
	v.Fingerprint = runtimeFingerprint(in.Tool, in.Args, state.Epoch)
	v.Reason = "novel_at_epoch"
	for i := len(state.Observations) - 1; i >= 0; i-- {
		o := state.Observations[i]
		if o.Epoch == state.Epoch && o.Fingerprint == v.Fingerprint {
			v.Action = Reuse
			v.Reason = "exact_fresh_result"
			v.ResultRef = o.ResultRef
			v.Applied = mode == ModeEnforce
			if in.PromptUnits > 0 {
				v.ReplayUnitsSaved = in.PromptUnits
				v.ReplaySquaredSaved = squareDecimal(in.PromptUnits)
				v.PromptBucket = promptBucket(in.PromptUnits)
			}
			return v
		}
	}
	return v
}

// After learns successful reads and invalidates all reads after any mutation.
// The observation bound prevents session state from growing with context length.
func After(state RuntimeState, in RuntimeInput, maxObservations int) RuntimeState {
	if maxObservations <= 0 {
		maxObservations = 128
	}
	if !in.ReadOnly {
		state.Epoch++
		state.Observations = nil
		return state
	}
	if !in.Succeeded || strings.TrimSpace(in.Tool) == "" || len(in.Args) == 0 {
		return state
	}
	o := RuntimeObservation{
		Fingerprint: runtimeFingerprint(in.Tool, in.Args, state.Epoch),
		Epoch:       state.Epoch,
		ResultRef:   firstNonEmpty(in.ResultRef, in.CallID),
	}
	for i := range state.Observations {
		if state.Observations[i].Fingerprint == o.Fingerprint && state.Observations[i].Epoch == o.Epoch {
			state.Observations[i] = o
			return state
		}
	}
	state.Observations = append(state.Observations, o)
	if len(state.Observations) > maxObservations {
		state.Observations = append([]RuntimeObservation(nil), state.Observations[len(state.Observations)-maxObservations:]...)
	}
	return state
}

func runtimeFingerprint(tool string, args json.RawMessage, epoch uint64) string {
	return fingerprint(tool, string(args), fmt.Sprintf("epoch:%d", epoch))
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func promptBucket(units int64) string {
	switch {
	case units < 16_000:
		return "lt16k"
	case units < 64_000:
		return "16k_64k"
	case units < 128_000:
		return "64k_128k"
	default:
		return "gte128k"
	}
}
