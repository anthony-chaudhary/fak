package agentopt

import (
	"context"
	"fmt"
)

// IterationStep represents the observed state and test results of one reasoning cycle.
type IterationStep struct {
	StepIndex       int     `json:"step_index"`
	StepOutput      string  `json:"step_output"`
	ConfidenceScore float64 `json:"confidence_score"`
	PassedTests     int     `json:"passed_tests"`
	TotalTests      int     `json:"total_tests"`
	TokensConsumed  int     `json:"tokens_consumed"`
}

// ExitCondition decides whether search should terminate immediately based on step evidence.
type ExitCondition interface {
	ShouldExit(step IterationStep) (exit bool, reason string)
}

// AllTestsPassedRule terminates early as soon as all declared tests pass.
type AllTestsPassedRule struct{}

// ShouldExit evaluates whether all tests passed.
func (r AllTestsPassedRule) ShouldExit(step IterationStep) (bool, string) {
	if step.TotalTests > 0 && step.PassedTests == step.TotalTests {
		return true, fmt.Sprintf("all %d tests passed successfully", step.TotalTests)
	}
	return false, ""
}

// ConfidenceThresholdRule terminates early when confidence meets or exceeds a target.
type ConfidenceThresholdRule struct {
	MinConfidence float64
}

// ShouldExit evaluates whether the confidence threshold is met.
func (r ConfidenceThresholdRule) ShouldExit(step IterationStep) (bool, string) {
	if step.ConfidenceScore >= r.MinConfidence {
		return true, fmt.Sprintf("confidence score %g meets or exceeds threshold %g", step.ConfidenceScore, r.MinConfidence)
	}
	return false, ""
}

// FunctionalRule allows specifying an arbitrary condition closure.
type FunctionalRule struct {
	Check func(step IterationStep) (bool, string)
}

// ShouldExit delegates to the provided check function.
func (r FunctionalRule) ShouldExit(step IterationStep) (bool, string) {
	if r.Check != nil {
		return r.Check(step)
	}
	return false, ""
}

// EarlyExitReport records execution metrics and savings from early termination.
type EarlyExitReport struct {
	CompletedIterations  int           `json:"completed_iterations"`
	MaxIterations        int           `json:"max_iterations"`
	EarlyExited          bool          `json:"early_exited"`
	ExitReason           string        `json:"exit_reason,omitempty"`
	FinalStep            IterationStep `json:"final_step"`
	SavedIterations      int           `json:"saved_iterations"`
	TotalTokensConsumed  int           `json:"total_tokens_consumed"`
	EstimatedSavedTokens int           `json:"estimated_saved_tokens"`
}

// EarlyExitController coordinates iterative reasoning search with early-exit gating.
type EarlyExitController struct {
	MaxIterations int
	Rules         []ExitCondition
}

// NewEarlyExitController constructs an iterative loop controller.
func NewEarlyExitController(maxIterations int, rules ...ExitCondition) *EarlyExitController {
	if maxIterations <= 0 {
		maxIterations = 1
	}
	return &EarlyExitController{
		MaxIterations: maxIterations,
		Rules:         rules,
	}
}

// Run executes the iterative step generator until an exit condition is satisfied or MaxIterations is reached.
func (c *EarlyExitController) Run(
	ctx context.Context,
	stepFn func(ctx context.Context, iter int) (IterationStep, error),
) (EarlyExitReport, error) {
	report := EarlyExitReport{
		MaxIterations: c.MaxIterations,
	}

	var lastStep IterationStep

	for iter := 1; iter <= c.MaxIterations; iter++ {
		if err := ctx.Err(); err != nil {
			return report, err
		}

		step, err := stepFn(ctx, iter)
		if err != nil {
			return report, err
		}
		step.StepIndex = iter
		lastStep = step
		report.CompletedIterations = iter
		report.TotalTokensConsumed += step.TokensConsumed

		for _, rule := range c.Rules {
			if exit, reason := rule.ShouldExit(step); exit {
				report.EarlyExited = true
				report.ExitReason = reason
				report.FinalStep = step
				report.SavedIterations = c.MaxIterations - iter
				if iter > 0 {
					avgTokens := report.TotalTokensConsumed / iter
					report.EstimatedSavedTokens = avgTokens * report.SavedIterations
				}
				return report, nil
			}
		}
	}

	report.FinalStep = lastStep
	report.EarlyExited = false
	report.SavedIterations = 0
	return report, nil
}
