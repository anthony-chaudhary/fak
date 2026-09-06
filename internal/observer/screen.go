package observer

import (
	"context"
	"errors"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ObserverSemanticScreen implements abi.SemanticScreen to wire the in-kernel observer
// into the context-MMU screen chain. It evaluates tool results and mutating tool calls,
// returning ScreenQuarantine on churn loops, regression loops, or unwitnessed mutating claims.
type ObserverSemanticScreen struct {
	pool *Pool
}

var (
	_ abi.SemanticScreen = ObserverSemanticScreen{}
	_ abi.SemanticScreen = (*ObserverSemanticScreen)(nil)
)

// NewObserverSemanticScreen creates a new ObserverSemanticScreen backed by the given Pool.
// If pool is nil, a default Pool with RequireWitnessDiff enabled is created.
func NewObserverSemanticScreen(pool *Pool) *ObserverSemanticScreen {
	if pool == nil {
		pool = NewPool(Config{
			RequireWitnessDiff: true,
		})
	}
	return &ObserverSemanticScreen{pool: pool}
}

// Pool returns the underlying observer Pool.
func (s *ObserverSemanticScreen) Pool() *Pool {
	return s.pool
}

// Register registers this screen with the kernel's ABI semantic screen registry.
func (s *ObserverSemanticScreen) Register() {
	abi.RegisterSemanticScreen(s)
}

// VerifyToolCall inspects a tool call before execution against the observer's session state
// and context MMU state markers (e.g. active churn/regress flags or quarantined MMU state).
// If the session is already flagged for churn or regression, or if the mutating call violates
// context MMU invariants, it returns ScreenQuarantine to prevent unverified mutation execution.
func (s ObserverSemanticScreen) VerifyToolCall(ctx context.Context, c *abi.ToolCall) abi.ScreenAdvice {
	if c == nil || s.pool == nil || ctx.Err() != nil {
		return abi.ScreenAdvice{Disposition: abi.ScreenAllow}
	}

	sessionID := extractSessionID(c)
	sess := s.pool.getOrCreateSession(sessionID)

	sess.mu.Lock()
	flaggedRegress := sess.flaggedRegress
	flaggedChurn := sess.flaggedChurn
	sess.mu.Unlock()

	if flaggedRegress {
		return abi.ScreenAdvice{
			Disposition: abi.ScreenQuarantine,
			Reason:      abi.ReasonIntegrityRefuted,
			Digest:      "pre-execution check: session flagged in regression loop",
			By:          "observer:pre_exec_regress",
		}
	}
	if flaggedChurn {
		return abi.ScreenAdvice{
			Disposition: abi.ScreenQuarantine,
			Reason:      abi.ReasonTrustViolation,
			Digest:      "pre-execution check: session flagged in churn loop",
			By:          "observer:pre_exec_churn",
		}
	}

	if IsMutatingTool(c.Tool) || (c.Meta != nil && c.Meta["mutating"] == "true") {
		if c.Meta != nil {
			if c.Meta["mmu_quarantined"] == "true" || c.Meta["taint"] == "quarantined" {
				return abi.ScreenAdvice{
					Disposition: abi.ScreenQuarantine,
					Reason:      abi.ReasonTrustViolation,
					Digest:      "pre-execution check: mutating tool call in quarantined context MMU state",
					By:          "observer:pre_exec_mmu_quarantined",
				}
			}
			if c.Meta["require_diff"] == "true" && c.Meta["diff"] == "" && len(c.Args.Inline) == 0 {
				return abi.ScreenAdvice{
					Disposition: abi.ScreenQuarantine,
					Reason:      abi.ReasonUnwitnessed,
					Digest:      "pre-execution check: mutating tool call lacks diff witness",
					By:          "observer:pre_exec_unwitnessed",
				}
			}
		}
	}

	return abi.ScreenAdvice{Disposition: abi.ScreenAllow}
}

// ScreenResult inspects a result body that survived the regex floor and returns
// an advisory disposition to the context MMU.
func (s ObserverSemanticScreen) ScreenResult(ctx context.Context, c *abi.ToolCall, body []byte) abi.ScreenAdvice {
	if s.pool == nil {
		return abi.ScreenAdvice{Disposition: abi.ScreenAllow}
	}

	if ctx.Err() != nil {
		return abi.ScreenAdvice{Disposition: abi.ScreenAllow}
	}

	tool := ""
	sessionID := "default"
	var args any
	diff := ""
	mutating := false

	if c != nil {
		tool = c.Tool
		sessionID = extractSessionID(c)
		if c.Meta != nil {
			if a, ok := c.Meta["args"]; ok {
				args = a
			} else if a, ok := c.Meta["arguments"]; ok {
				args = a
			}
			if d, ok := c.Meta["diff"]; ok {
				diff = d
			}
			if m, ok := c.Meta["mutating"]; ok && (m == "true" || m == "1") {
				mutating = true
			}
		}
		if args == nil && len(c.Args.Inline) > 0 {
			args = string(c.Args.Inline)
		}
	}

	if !mutating {
		mutating = IsMutatingTool(tool)
	}

	resultStr := string(body)
	if diff == "" && mutating && hasDiffInResult(resultStr) {
		diff = resultStr
	}

	errStr := ""
	if c != nil && c.Meta != nil {
		if e, ok := c.Meta["error"]; ok {
			errStr = e
		}
	}
	if errStr == "" && isResultError(resultStr) {
		errStr = resultStr
	}

	obs := StepObservation{
		SessionID: sessionID,
		Tool:      tool,
		Args:      args,
		Result:    resultStr,
		Diff:      diff,
		Error:     errStr,
		Mutating:  mutating,
	}

	res, err := s.pool.ObserveSyncBarrier(ctx, obs)

	if errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
		digest := res.Reason
		if digest == "" {
			digest = "observer sync barrier context deadline exceeded"
		}
		return abi.ScreenAdvice{
			Disposition: abi.ScreenQuarantine,
			Reason:      abi.ReasonIntegrityRefuted,
			Digest:      digest,
			By:          "observer:context_deadline",
		}
	}

	if errors.Is(err, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) || ctx.Err() != nil {
		digest := res.Reason
		if digest == "" {
			digest = "observer sync barrier context canceled"
		}
		return abi.ScreenAdvice{
			Disposition: abi.ScreenQuarantine,
			Reason:      abi.ReasonIntegrityRefuted,
			Digest:      digest,
			By:          "observer:context_canceled",
		}
	}

	if errors.Is(err, ErrBarrierTimeout) {
		if obs.IsMutating() {
			digest := res.Reason
			if digest == "" {
				digest = "observer sync barrier timed out waiting for in-flight tasks"
			}
			return abi.ScreenAdvice{
				Disposition: abi.ScreenQuarantine,
				Reason:      abi.ReasonIntegrityRefuted,
				Digest:      digest,
				By:          "observer:barrier_timeout",
			}
		}
		return abi.ScreenAdvice{
			Disposition: abi.ScreenAllow,
			By:          "observer:advance",
		}
	}

	if errors.Is(err, ErrUnwitnessedDiff) || (obs.IsMutating() && res.WitnessVerdict == WitnessUnwitnessedClaim && s.pool.cfg.RequireWitnessDiff) {
		digest := res.Reason
		if digest == "" {
			digest = "mutating step lacks confirmed diff witness"
		}
		return abi.ScreenAdvice{
			Disposition: abi.ScreenQuarantine,
			Reason:      abi.ReasonUnwitnessed,
			Digest:      digest,
			By:          "observer:unwitnessed_claim",
		}
	}

	if errors.Is(err, ErrRegressRefused) || res.StepVerdict == StepRegress {
		digest := res.Reason
		if digest == "" {
			digest = "step refused due to regression loop"
		}
		return abi.ScreenAdvice{
			Disposition: abi.ScreenQuarantine,
			Reason:      abi.ReasonIntegrityRefuted,
			Digest:      digest,
			By:          "observer:step_regress",
		}
	}

	if errors.Is(err, ErrChurnRefused) || res.StepVerdict == StepChurn {
		digest := res.Reason
		if digest == "" {
			digest = "step refused due to churn loop"
		}
		return abi.ScreenAdvice{
			Disposition: abi.ScreenQuarantine,
			Reason:      abi.ReasonTrustViolation,
			Digest:      digest,
			By:          "observer:step_churn",
		}
	}

	return abi.ScreenAdvice{
		Disposition: abi.ScreenAllow,
		By:          "observer:advance",
	}
}

func extractSessionID(c *abi.ToolCall) string {
	if c == nil {
		return "default"
	}
	if c.TraceID != "" {
		return c.TraceID
	}
	if c.Meta != nil {
		if sid, ok := c.Meta["session_id"]; ok && sid != "" {
			return sid
		}
		if sid, ok := c.Meta["session"]; ok && sid != "" {
			return sid
		}
	}
	return "default"
}
