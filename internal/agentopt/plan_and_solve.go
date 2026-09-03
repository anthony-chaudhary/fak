package agentopt

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Family 10: Workflowning algorithms & world models.
//
// Workflow-and-solve decomposes high-level objectives into ordered steps
// with typed preconditions and postconditions, enforcing deterministic
// verification prior to advancing across steps.

// WorkflowState defines the operational state of the workflowning state machine.
type WorkflowState string

const (
	WorkflowStateIdle          WorkflowState = "idle"
	WorkflowStateWorkflowning  WorkflowState = "workflowning"
	WorkflowStateExecuting     WorkflowState = "executing"
	WorkflowStateVerifying     WorkflowState = "verifying"
	WorkflowStateTransitioning WorkflowState = "transitioning"
	WorkflowStateBlocked       WorkflowState = "blocked"
	WorkflowStateCompleted     WorkflowState = "completed"
	WorkflowStateFailed        WorkflowState = "failed"
)

// StepStatus indicates the execution status of an individual step.
type StepStatus string

const (
	StepStatusPending StepStatus = "pending"
	StepStatusRunning StepStatus = "running"
	StepStatusSuccess StepStatus = "success"
	StepStatusFailed  StepStatus = "failed"
	StepStatusBlocked StepStatus = "blocked"
	StepStatusSkipped StepStatus = "skipped"
)

// StepEnv holds runtime context and shared key-value state across step executions.
type StepEnv struct {
	context.Context
	state   map[string]any
	results []StepResult
	mu      sync.RWMutex
}

// NewStepEnv constructs a new StepEnv with an initialized state map.
func NewStepEnv(ctx context.Context) *StepEnv {
	if ctx == nil {
		ctx = context.Background()
	}
	return &StepEnv{
		Context: ctx,
		state:   make(map[string]any),
		results: make([]StepResult, 0),
	}
}

// Set stores a value in the shared state map.
func (sc *StepEnv) Set(key string, val any) {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.state[key] = val
}

// Get retrieves a value from the shared state map.
func (sc *StepEnv) Get(key string) (any, bool) {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	val, ok := sc.state[key]
	return val, ok
}

// RecordResult appends an executed step result to context history.
func (sc *StepEnv) RecordResult(res StepResult) {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.results = append(sc.results, res)
}

// Results returns a copy of all recorded step results.
func (sc *StepEnv) Results() []StepResult {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	copied := make([]StepResult, len(sc.results))
	copy(copied, sc.results)
	return copied
}

// StepCondition asserts a typed requirement before or after step execution.
type StepCondition struct {
	Name        string                                         `json:"name"`
	Description string                                         `json:"description,omitempty"`
	Assert      func(ctx context.Context, senv *StepEnv) error `json:"-"`
}

// NewStepCondition constructs a StepCondition with the provided name and assertion.
func NewStepCondition(name string, assertFn func(ctx context.Context, senv *StepEnv) error) StepCondition {
	return StepCondition{
		Name:   name,
		Assert: assertFn,
	}
}

// RequireStateKey returns a condition ensuring a specific key exists in step context state.
func RequireStateKey(key string) StepCondition {
	return StepCondition{
		Name:        "require_state_key_" + key,
		Description: "Verify state key exists: " + key,
		Assert: func(ctx context.Context, senv *StepEnv) error {
			if _, ok := senv.Get(key); !ok {
				return fmt.Errorf("required state key missing: %s", key)
			}
			return nil
		},
	}
}

// ExecuteFn performs the action of a step.
type ExecuteFn func(ctx context.Context, senv *StepEnv) (any, error)

// VerifyFn deterministically checks step output and context state.
type VerifyFn func(ctx context.Context, senv *StepEnv, output any) (bool, error)

// StepSpec specifies one unit of decomposed workflow work with preconditions and postconditions.
type StepSpec struct {
	StepIndex     int           `json:"step_index"`
	Title         string        `json:"title"`
	Precondition  StepCondition `json:"precondition"`
	Postcondition StepCondition `json:"postcondition"`
	ExecuteFn     ExecuteFn     `json:"-"`
	VerifyFn      VerifyFn      `json:"-"`
}

// StepResult captures the output and verification state of an executed step.
type StepResult struct {
	StepIndex int           `json:"step_index"`
	Title     string        `json:"title"`
	Status    StepStatus    `json:"status"`
	Output    any           `json:"output,omitempty"`
	Verified  bool          `json:"verified"`
	Error     string        `json:"error,omitempty"`
	Duration  time.Duration `json:"duration"`
}

// StepExecutionReport summarizes execution across all steps in a workflow.
type StepExecutionReport struct {
	Objective      string        `json:"objective,omitempty"`
	CompletedSteps int           `json:"completed_steps"`
	TotalSteps     int           `json:"total_steps"`
	Success        bool          `json:"success"`
	Blocked        bool          `json:"blocked"`
	BlockedReason  string        `json:"blocked_reason,omitempty"`
	Results        []StepResult  `json:"results"`
	Duration       time.Duration `json:"duration"`
}

// StateTransition records a state movement within the state machine.
type StateTransition struct {
	From      WorkflowState `json:"from"`
	To        WorkflowState `json:"to"`
	StepIndex int           `json:"step_index"`
	Timestamp time.Time     `json:"timestamp"`
	Reason    string        `json:"reason,omitempty"`
}

// ObjectiveDecomposer converts a high-level objective into ordered StepSpecs.
type ObjectiveDecomposer interface {
	Decompose(ctx context.Context, objective string) ([]StepSpec, error)
}

// ObjectiveDecomposerFunc adapts a function to the ObjectiveDecomposer interface.
type ObjectiveDecomposerFunc func(ctx context.Context, objective string) ([]StepSpec, error)

// Decompose calls the underlying function.
func (f ObjectiveDecomposerFunc) Decompose(ctx context.Context, objective string) ([]StepSpec, error) {
	return f(ctx, objective)
}

// DecomposeObjective decomposes text into ordered StepSpecs.
func DecomposeObjective(ctx context.Context, objective string) ([]StepSpec, error) {
	trimmed := strings.TrimSpace(objective)
	if trimmed == "" {
		return nil, fmt.Errorf("objective cannot be empty")
	}

	lines := strings.Split(trimmed, "\n")
	steps := make([]StepSpec, 0, len(lines))

	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}

		cleanTitle := line
		cleanTitle = strings.TrimLeft(cleanTitle, "-* ")
		if dotIdx := strings.Index(cleanTitle, "."); dotIdx > 0 && dotIdx < 4 {
			prefix := cleanTitle[:dotIdx]
			isNum := true
			for _, r := range prefix {
				if r < '0' || r > '9' {
					isNum = false
					break
				}
			}
			if isNum {
				cleanTitle = strings.TrimSpace(cleanTitle[dotIdx+1:])
			}
		}

		stepIdx := len(steps) + 1
		steps = append(steps, StepSpec{
			StepIndex: stepIdx,
			Title:     cleanTitle,
			Precondition: StepCondition{
				Name: fmt.Sprintf("step_%d_precondition", stepIdx),
			},
			Postcondition: StepCondition{
				Name: fmt.Sprintf("step_%d_postcondition", stepIdx),
			},
		})
	}

	if len(steps) == 0 {
		return nil, fmt.Errorf("no steps extracted from objective")
	}
	return steps, nil
}

// DecomposedSolveMachine executes ordered workflow steps with deterministic step verification.
type DecomposedSolveMachine struct {
	CurrentState                WorkflowState
	CurrentStep                 int
	RequireExplicitVerification bool
	Decomposer                  ObjectiveDecomposer
	Transitions                 []StateTransition
	mu                          sync.RWMutex
}

// NewDecomposedSolveMachine creates an initialized state machine.
func NewDecomposedSolveMachine() *DecomposedSolveMachine {
	return &DecomposedSolveMachine{
		CurrentState: WorkflowStateIdle,
		Transitions:  make([]StateTransition, 0),
	}
}

// State returns the current machine state.
func (sm *DecomposedSolveMachine) State() WorkflowState {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.CurrentState
}

// transition records and updates machine state.
func (sm *DecomposedSolveMachine) transition(to WorkflowState, stepIndex int, reason string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	transition := StateTransition{
		From:      sm.CurrentState,
		To:        to,
		StepIndex: stepIndex,
		Timestamp: time.Now(),
		Reason:    reason,
	}
	sm.Transitions = append(sm.Transitions, transition)
	sm.CurrentState = to
	sm.CurrentStep = stepIndex
}

// Decompose decomposes an objective into steps using the configured or default decomposer.
func (sm *DecomposedSolveMachine) Decompose(ctx context.Context, objective string) ([]StepSpec, error) {
	sm.transition(WorkflowStateWorkflowning, 0, "decomposing objective into steps")
	if sm.Decomposer != nil {
		return sm.Decomposer.Decompose(ctx, objective)
	}
	return DecomposeObjective(ctx, objective)
}

// ExecuteSteps executes steps sequentially, verifying each step before advancing.
func (sm *DecomposedSolveMachine) ExecuteSteps(ctx context.Context, steps []StepSpec) StepExecutionReport {
	startTime := time.Now()
	report := StepExecutionReport{
		TotalSteps: len(steps),
		Results:    make([]StepResult, 0, len(steps)),
	}

	if len(steps) == 0 {
		report.Success = true
		sm.transition(WorkflowStateCompleted, 0, "zero steps executed")
		return report
	}

	senv := NewStepEnv(ctx)

	for i, step := range steps {
		stepIdx := step.StepIndex
		if stepIdx == 0 {
			stepIdx = i + 1
		}

		if cErr := ctx.Err(); cErr != nil {
			sm.transition(WorkflowStateFailed, stepIdx, cErr.Error())
			report.Blocked = true
			report.BlockedReason = cErr.Error()
			report.Success = false
			sm.markSubsequentBlocked(&report, steps, i, "context cancelled")
			report.Duration = time.Since(startTime)
			return report
		}

		// 1. Precondition check
		if step.Precondition.Assert != nil {
			if preErr := step.Precondition.Assert(ctx, senv); preErr != nil {
				failureMsg := fmt.Sprintf("precondition failed: %v", preErr)
				res := StepResult{
					StepIndex: stepIdx,
					Title:     step.Title,
					Status:    StepStatusFailed,
					Verified:  false,
					Error:     failureMsg,
				}
				report.Results = append(report.Results, res)
				senv.RecordResult(res)

				sm.transition(WorkflowStateBlocked, stepIdx, failureMsg)
				report.Blocked = true
				report.BlockedReason = failureMsg
				report.Success = false
				sm.markSubsequentBlocked(&report, steps, i+1, fmt.Sprintf("blocked by step %d precondition failure", stepIdx))
				report.Duration = time.Since(startTime)
				return report
			}
		}

		// 2. Execute step
		sm.transition(WorkflowStateExecuting, stepIdx, fmt.Sprintf("executing step %d", stepIdx))
		stepStart := time.Now()
		var output any
		var execErr error
		if step.ExecuteFn != nil {
			output, execErr = step.ExecuteFn(ctx, senv)
		}
		stepDuration := time.Since(stepStart)

		if execErr != nil {
			failureMsg := fmt.Sprintf("execution failed: %v", execErr)
			res := StepResult{
				StepIndex: stepIdx,
				Title:     step.Title,
				Status:    StepStatusFailed,
				Output:    output,
				Verified:  false,
				Error:     failureMsg,
				Duration:  stepDuration,
			}
			report.Results = append(report.Results, res)
			senv.RecordResult(res)

			sm.transition(WorkflowStateBlocked, stepIdx, failureMsg)
			report.Blocked = true
			report.BlockedReason = failureMsg
			report.Success = false
			sm.markSubsequentBlocked(&report, steps, i+1, fmt.Sprintf("blocked by step %d execution failure", stepIdx))
			report.Duration = time.Since(startTime)
			return report
		}

		// 3. Deterministic step verification
		sm.transition(WorkflowStateVerifying, stepIdx, fmt.Sprintf("verifying step %d", stepIdx))
		verified := false
		var verifyErr error

		if step.VerifyFn != nil {
			verified, verifyErr = step.VerifyFn(ctx, senv, output)
		} else if sm.RequireExplicitVerification {
			verifyErr = fmt.Errorf("explicit verification required but VerifyFn is nil")
			verified = false
		} else {
			verified = true
		}

		if verifyErr != nil || !verified {
			reason := "step verification failed"
			if verifyErr != nil {
				reason = fmt.Sprintf("verification error: %v", verifyErr)
			} else if !verified {
				reason = "step verification check returned false"
			}

			res := StepResult{
				StepIndex: stepIdx,
				Title:     step.Title,
				Status:    StepStatusFailed,
				Output:    output,
				Verified:  false,
				Error:     reason,
				Duration:  stepDuration,
			}
			report.Results = append(report.Results, res)
			senv.RecordResult(res)

			sm.transition(WorkflowStateBlocked, stepIdx, reason)
			report.Blocked = true
			report.BlockedReason = fmt.Sprintf("step %d un-verified: %s", stepIdx, reason)
			report.Success = false
			sm.markSubsequentBlocked(&report, steps, i+1, fmt.Sprintf("blocked by un-verified step %d", stepIdx))
			report.Duration = time.Since(startTime)
			return report
		}

		// 4. Postcondition check
		if step.Postcondition.Assert != nil {
			if postErr := step.Postcondition.Assert(ctx, senv); postErr != nil {
				failureMsg := fmt.Sprintf("postcondition failed: %v", postErr)
				res := StepResult{
					StepIndex: stepIdx,
					Title:     step.Title,
					Status:    StepStatusFailed,
					Output:    output,
					Verified:  false,
					Error:     failureMsg,
					Duration:  stepDuration,
				}
				report.Results = append(report.Results, res)
				senv.RecordResult(res)

				sm.transition(WorkflowStateBlocked, stepIdx, failureMsg)
				report.Blocked = true
				report.BlockedReason = failureMsg
				report.Success = false
				sm.markSubsequentBlocked(&report, steps, i+1, fmt.Sprintf("blocked by step %d postcondition failure", stepIdx))
				report.Duration = time.Since(startTime)
				return report
			}
		}

		// 5. Successful transition to next step
		sm.transition(WorkflowStateTransitioning, stepIdx, fmt.Sprintf("step %d verified, transitioning", stepIdx))
		res := StepResult{
			StepIndex: stepIdx,
			Title:     step.Title,
			Status:    StepStatusSuccess,
			Output:    output,
			Verified:  true,
			Duration:  stepDuration,
		}
		report.Results = append(report.Results, res)
		senv.RecordResult(res)
		report.CompletedSteps++
	}

	sm.transition(WorkflowStateCompleted, len(steps), "all steps verified and completed")
	report.Success = true
	report.Blocked = false
	report.Duration = time.Since(startTime)
	return report
}

// markSubsequentBlocked fills subsequent unexecuted steps with blocked status.
func (sm *DecomposedSolveMachine) markSubsequentBlocked(report *StepExecutionReport, steps []StepSpec, startIdx int, reason string) {
	for j := startIdx; j < len(steps); j++ {
		idx := steps[j].StepIndex
		if idx == 0 {
			idx = j + 1
		}
		report.Results = append(report.Results, StepResult{
			StepIndex: idx,
			Title:     steps[j].Title,
			Status:    StepStatusBlocked,
			Verified:  false,
			Error:     reason,
		})
	}
}

// DecomposedSolveRunner coordinates step execution using DecomposedSolveMachine.
type DecomposedSolveRunner struct {
	StateMachine *DecomposedSolveMachine
}

// NewDecomposedSolveRunner constructs a DecomposedSolveRunner.
func NewDecomposedSolveRunner() *DecomposedSolveRunner {
	return &DecomposedSolveRunner{
		StateMachine: NewDecomposedSolveMachine(),
	}
}

// ExecuteSteps executes steps through the runner's state machine.
func (r *DecomposedSolveRunner) ExecuteSteps(ctx context.Context, steps []StepSpec) StepExecutionReport {
	return r.StateMachine.ExecuteSteps(ctx, steps)
}
