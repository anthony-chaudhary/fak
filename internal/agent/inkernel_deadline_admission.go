package agent

import (
	"context"
	"errors"
)

type executionDeadlineKey struct{}

type executionDeadlineAdmission struct {
	planner *InKernelPlanner
	check   func(context.Context, int, int, int) error
}

// ExecutionDeadlineAdmissionSupported reports whether the normal request-owned
// prefix acquisition path can supply deadline credit. Speculative routes retain
// cold admission until their own acquisition boundaries carry this contract.
func (p *InKernelPlanner) ExecutionDeadlineAdmissionSupported() bool {
	return p != nil && p.SpeculativeEngine() == nil && p.MetalMTPCoordinator() == nil
}

// WithExecutionDeadlineAdmission binds a request's deadline check to this exact
// planner. The check receives prompt, usable cached-prefix, and output token
// counts after the request owns its cache state, before prefill or decode. It may
// run again on retry, when both cache ownership and remaining time have changed.
func (p *InKernelPlanner) WithExecutionDeadlineAdmission(ctx context.Context, check func(context.Context, int, int, int) error) context.Context {
	return context.WithValue(ctx, executionDeadlineKey{}, executionDeadlineAdmission{planner: p, check: check})
}

func (p *InKernelPlanner) checkExecutionDeadline(ctx context.Context, prompt, cached, output int) error {
	admission, ok := ctx.Value(executionDeadlineKey{}).(executionDeadlineAdmission)
	if !ok || admission.check == nil {
		return nil
	}
	if admission.planner != p {
		return errors.New("native deadline admission belongs to a different planner")
	}
	if err := ctx.Err(); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	return admission.check(ctx, prompt, cached, output)
}

// Preparation can observe cancellation before the owned-prefix seam. Translate
// only exhaustion of this bound request's deadline, without inventing cache
// credit or changing unrelated internal timeouts and explicit cancellation.
func (p *InKernelPlanner) executionDeadlineError(ctx context.Context, err error) error {
	if errors.Is(err, context.DeadlineExceeded) && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		if refusal := p.checkExecutionDeadline(ctx, 0, 0, 0); refusal != nil {
			return refusal
		}
	}
	return err
}
