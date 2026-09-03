package agentopt

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// CascadeTier identifies the cost and optimization tier of a model target in the cascade.
type CascadeTier string

const (
	// TierFast represents low-latency, lightweight, cost-efficient models.
	TierFast CascadeTier = "fast"
	// TierSmall represents compact, parameter-efficient small models.
	TierSmall CascadeTier = "small"
	// TierStandard represents general-purpose, balanced models.
	TierStandard CascadeTier = "standard"
	// TierFrontier represents high-optimization, large frontier models.
	TierFrontier CascadeTier = "frontier"
)

// Aspect defines the domain or criteria focus for an aspect-based cascade step.
type Aspect string

const (
	AspectGeneral   Aspect = "general"
	AspectSyntax    Aspect = "syntax"
	AspectSchema    Aspect = "schema"
	AspectTest      Aspect = "test"
	AspectReasoning Aspect = "reasoning"
)

// TargetModel identifies a model destination within a cascade tier.
type TargetModel struct {
	Name        string      `json:"name"`
	Tier        CascadeTier `json:"tier"`
	CostWeight  float64     `json:"cost_weight,omitempty"`
	Description string      `json:"description,omitempty"`
}

// CascadeStep represents one stage in a progressive model cascade.
type CascadeStep struct {
	Target      TargetModel `json:"target"`
	Aspect      Aspect      `json:"aspect,omitempty"`
	Description string      `json:"description,omitempty"`
}

// CascadeAttempt records the telemetry and validation evidence of one execution in the cascade.
type CascadeAttempt struct {
	StepIndex    int           `json:"step_index"`
	Target       TargetModel   `json:"target"`
	Prompt       string        `json:"prompt"`
	Output       string        `json:"output,omitempty"`
	Valid        bool          `json:"valid"`
	FailureError string        `json:"failure_error,omitempty"`
	EvidenceInfo string        `json:"evidence_info,omitempty"`
	Duration     time.Duration `json:"duration"`
}

// CascadeResult summarizes the overall outcome of a model cascade execution.
type CascadeResult struct {
	FinalTier CascadeTier      `json:"final_tier"`
	Escalated bool             `json:"escalated"`
	Attempts  []CascadeAttempt `json:"attempts"`
	Output    string           `json:"output"`
	Error     error            `json:"error,omitempty"`
}

// EvidenceProvider is an optional interface implemented by validation errors to yield structured evidence information.
type EvidenceProvider interface {
	Evidence() string
}

// CascadeValidationError encapsulates a failure reason and deterministic evidence information.
type CascadeValidationError struct {
	Reason       string `json:"reason"`
	EvidenceData string `json:"evidence,omitempty"`
}

func (e *CascadeValidationError) Error() string {
	if e.EvidenceData != "" && e.EvidenceData != e.Reason {
		return fmt.Sprintf("%s: %s", e.Reason, e.EvidenceData)
	}
	return e.Reason
}

func (e *CascadeValidationError) Evidence() string {
	return e.EvidenceData
}

// NewValidationError constructs an error carrying deterministic evidence data.
func NewValidationError(reason, evidence string) *CascadeValidationError {
	return &CascadeValidationError{
		Reason:       reason,
		EvidenceData: evidence,
	}
}

// CascadeValidatorFn evaluates whether the model output satisfies deterministic criteria.
// It returns nil if the output is valid, or an error describing the failure.
type CascadeValidatorFn func(ctx context.Context, output string) error

// CascadeInvokerFn invokes a model target with the given prompt.
type CascadeInvokerFn func(ctx context.Context, target TargetModel, prompt string) (string, error)

// EscalationFormatter formats the escalated prompt passed to higher-tier models upon validation failure.
type EscalationFormatter func(originalPrompt string, lastAttempt CascadeAttempt) string

// DefaultEscalationFormatter formats an escalated prompt with captured failure error and evidence information.
func DefaultEscalationFormatter(originalPrompt string, lastAttempt CascadeAttempt) string {
	var sb strings.Builder
	sb.WriteString(originalPrompt)
	sb.WriteString("\n\n[Escalation Context]")
	sb.WriteString(fmt.Sprintf("\nPrevious tier (%s, model %s) failed validation.", lastAttempt.Target.Tier, lastAttempt.Target.Name))
	if lastAttempt.FailureError != "" {
		sb.WriteString(fmt.Sprintf("\nFailure Error: %s", lastAttempt.FailureError))
	}
	if lastAttempt.EvidenceInfo != "" && lastAttempt.EvidenceInfo != lastAttempt.FailureError {
		sb.WriteString(fmt.Sprintf("\nEvidence Details: %s", lastAttempt.EvidenceInfo))
	}
	if lastAttempt.Output != "" {
		sb.WriteString("\nPrevious Rejected Output:\n")
		sb.WriteString(lastAttempt.Output)
	}
	sb.WriteString("\nPlease correct the failure and satisfy all validation criteria.")
	return sb.String()
}

// ModelCascadeController coordinates aspect-based cascading with automated fallback escalation.
type ModelCascadeController struct {
	Steps            []CascadeStep
	FormatEscalation EscalationFormatter
}

// NewModelCascadeController constructs a cascade controller with the specified steps.
func NewModelCascadeController(steps ...CascadeStep) *ModelCascadeController {
	copied := make([]CascadeStep, len(steps))
	copy(copied, steps)
	return &ModelCascadeController{
		Steps: copied,
	}
}

// DefaultModelCascadeController returns a controller configured with standard Fast -> Standard -> Frontier tiers.
func DefaultModelCascadeController() *ModelCascadeController {
	return NewModelCascadeController(
		CascadeStep{
			Target: TargetModel{
				Name:       "tier-fast-model",
				Tier:       TierFast,
				CostWeight: 0.1,
			},
			Aspect: AspectGeneral,
		},
		CascadeStep{
			Target: TargetModel{
				Name:       "tier-standard-model",
				Tier:       TierStandard,
				CostWeight: 0.5,
			},
			Aspect: AspectGeneral,
		},
		CascadeStep{
			Target: TargetModel{
				Name:       "tier-frontier-model",
				Tier:       TierFrontier,
				CostWeight: 1.0,
			},
			Aspect: AspectGeneral,
		},
	)
}

// NewAspectCascadeController returns a cascade controller tuned for a specific aspect.
func NewAspectCascadeController(aspect Aspect) *ModelCascadeController {
	return NewModelCascadeController(
		CascadeStep{
			Target: TargetModel{
				Name:       fmt.Sprintf("%s-fast-model", aspect),
				Tier:       TierFast,
				CostWeight: 0.1,
			},
			Aspect: aspect,
		},
		CascadeStep{
			Target: TargetModel{
				Name:       fmt.Sprintf("%s-standard-model", aspect),
				Tier:       TierStandard,
				CostWeight: 0.5,
			},
			Aspect: aspect,
		},
		CascadeStep{
			Target: TargetModel{
				Name:       fmt.Sprintf("%s-frontier-model", aspect),
				Tier:       TierFrontier,
				CostWeight: 1.0,
			},
			Aspect: aspect,
		},
	)
}

// Cascade executes progressive generation attempts starting with the most cost-efficient tier.
// If output validation fails, it automatically escalates to a higher-tier model along with
// the captured failure error and evidence information.
func (c *ModelCascadeController) Cascade(
	ctx context.Context,
	prompt string,
	validatorFn CascadeValidatorFn,
	invokerFn CascadeInvokerFn,
) CascadeResult {
	if invokerFn == nil {
		return CascadeResult{
			Error: errors.New("invokerFn cannot be nil"),
		}
	}

	steps := c.Steps
	if len(steps) == 0 {
		steps = DefaultModelCascadeController().Steps
	}

	formatter := c.FormatEscalation
	if formatter == nil {
		formatter = DefaultEscalationFormatter
	}

	attempts := make([]CascadeAttempt, 0, len(steps))
	currentPrompt := prompt
	var lastErr error

	for i, step := range steps {
		if ctx.Err() != nil {
			return CascadeResult{
				FinalTier: step.Target.Tier,
				Escalated: i > 0,
				Attempts:  attempts,
				Error:     ctx.Err(),
			}
		}

		start := time.Now()
		output, err := invokerFn(ctx, step.Target, currentPrompt)
		duration := time.Since(start)

		if err != nil {
			attempt := CascadeAttempt{
				StepIndex:    i,
				Target:       step.Target,
				Prompt:       currentPrompt,
				Output:       "",
				Valid:        false,
				FailureError: err.Error(),
				EvidenceInfo: fmt.Sprintf("invocation failure: %v", err),
				Duration:     duration,
			}
			attempts = append(attempts, attempt)
			lastErr = err

			if ctx.Err() != nil {
				return CascadeResult{
					FinalTier: step.Target.Tier,
					Escalated: i > 0,
					Attempts:  attempts,
					Error:     ctx.Err(),
				}
			}

			currentPrompt = formatter(prompt, attempt)
			continue
		}

		var valErr error
		var evidence string
		if validatorFn != nil {
			valErr = validatorFn(ctx, output)
		}

		if valErr != nil {
			evidence = valErr.Error()
			if wp, ok := valErr.(EvidenceProvider); ok {
				evidence = wp.Evidence()
			}

			attempt := CascadeAttempt{
				StepIndex:    i,
				Target:       step.Target,
				Prompt:       currentPrompt,
				Output:       output,
				Valid:        false,
				FailureError: valErr.Error(),
				EvidenceInfo: evidence,
				Duration:     duration,
			}
			attempts = append(attempts, attempt)
			lastErr = valErr

			currentPrompt = formatter(prompt, attempt)
			continue
		}

		// Validation succeeded
		attempt := CascadeAttempt{
			StepIndex: i,
			Target:    step.Target,
			Prompt:    currentPrompt,
			Output:    output,
			Valid:     true,
			Duration:  duration,
		}
		attempts = append(attempts, attempt)

		return CascadeResult{
			FinalTier: step.Target.Tier,
			Escalated: i > 0,
			Attempts:  attempts,
			Output:    output,
			Error:     nil,
		}
	}

	finalTier := TierFrontier
	if len(steps) > 0 {
		finalTier = steps[len(steps)-1].Target.Tier
	}
	var lastOutput string
	if len(attempts) > 0 {
		lastOutput = attempts[len(attempts)-1].Output
	}

	return CascadeResult{
		FinalTier: finalTier,
		Escalated: len(attempts) > 1,
		Attempts:  attempts,
		Output:    lastOutput,
		Error:     fmt.Errorf("cascade exhausted: all %d steps failed: %w", len(steps), lastErr),
	}
}

// SyntaxValidator wraps a syntax check into a CascadeValidatorFn.
func SyntaxValidator(checker func(string) error) CascadeValidatorFn {
	return func(ctx context.Context, output string) error {
		if checker == nil {
			return nil
		}
		if err := checker(output); err != nil {
			return NewValidationError("syntax validation failure", err.Error())
		}
		return nil
	}
}

// JSONSyntaxValidator verifies that output is syntactically valid JSON.
func JSONSyntaxValidator() CascadeValidatorFn {
	return func(ctx context.Context, output string) error {
		var raw any
		if err := json.Unmarshal([]byte(output), &raw); err != nil {
			return NewValidationError("malformed JSON syntax", err.Error())
		}
		return nil
	}
}

// SchemaValidationAdapter integrates a SchemaValidator for tool call argument verification.
func SchemaValidationAdapter(validator *SchemaValidator, toolName string) CascadeValidatorFn {
	return func(ctx context.Context, output string) error {
		if validator == nil {
			return nil
		}
		res := validator.ValidateToolCall(toolName, []byte(output))
		if !res.Valid {
			evidence := strings.Join(res.Violations, "; ")
			return NewValidationError("schema validation failure", evidence)
		}
		return nil
	}
}

// TestExecutionValidator wraps test execution verification into a CascadeValidatorFn.
func TestExecutionValidator(tester func(ctx context.Context, output string) (bool, string, error)) CascadeValidatorFn {
	return func(ctx context.Context, output string) error {
		if tester == nil {
			return nil
		}
		passed, evidence, err := tester(ctx, output)
		if err != nil {
			return NewValidationError("test execution error", err.Error())
		}
		if !passed {
			return NewValidationError("test execution failed", evidence)
		}
		return nil
	}
}

// CompositeCascadeValidator runs multiple validators in sequence, stopping on the first failure.
func CompositeCascadeValidator(validators ...CascadeValidatorFn) CascadeValidatorFn {
	return func(ctx context.Context, output string) error {
		for _, v := range validators {
			if v != nil {
				if err := v(ctx, output); err != nil {
					return err
				}
			}
		}
		return nil
	}
}
