package microagent

import "context"

// RetryFeedback is the evidence seam for bounded retry. A Microagent opts in by
// implementing it; the host passes the exact error returned by the failed Step
// before it will call Step again. Implementations append that evidence to their
// model/tool transcript (or equivalent state). Returning an error refuses the
// retry and retires the agent with both failures preserved.
//
// The interface deliberately has no success path and no string-only summary:
// callers receive the original error value, so retry cannot silently become a
// blind replay. Config.MaxRetries remains off by default.
type RetryFeedback interface {
	RetryFeedback(ctx context.Context, evidence error) error
}
