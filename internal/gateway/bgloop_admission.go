package gateway

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/bgloop"
)

// AdmitBackgroundWake adapts background-loop wake reconstitution to the same
// bounded admission controller used by foreground gateway work. Queued wakes
// wait for capacity instead of being refused merely because a cohort woke at
// once; only policy denial, impossible token demand, queue exhaustion, or
// context cancellation can refuse one.
func AdmitBackgroundWake(controller *AdmissionController, tokens int) func(context.Context, bgloop.WakeRequest) (func(), error) {
	return func(ctx context.Context, wake bgloop.WakeRequest) (func(), error) {
		if controller == nil {
			return nil, nil
		}
		lease, err := controller.Acquire(ctx, SeqRequest{TraceID: "bgloop-wake:" + wake.Job.JobID(), Tokens: tokens})
		if err != nil {
			return nil, err
		}
		return lease.Release, nil
	}
}
