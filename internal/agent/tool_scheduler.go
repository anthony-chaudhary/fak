package agent

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/session"
)

const toolCallSkippedByCancellation = session.ReasonInterrupted

type scheduledToolCall struct {
	call ToolCall
	run  func(context.Context) (string, error)
}

type scheduledToolResult struct {
	call    ToolCall
	content string
	err     error
	started bool
}

type completedToolCall struct {
	index   int
	content string
	err     error
}

// runScheduledToolCalls admits at most parallelism bodies at once. Cancellation
// freezes admission, drains every admitted body, and closes the remaining call/result
// pairs with typed skipped receipts. The returned slice always follows model order,
// independent of body completion order.
func runScheduledToolCalls(ctx context.Context, parallelism int, calls []scheduledToolCall) []scheduledToolResult {
	results := make([]scheduledToolResult, len(calls))
	if len(calls) == 0 {
		return results
	}
	if parallelism < 1 {
		parallelism = 1
	}
	if parallelism > len(calls) {
		parallelism = len(calls)
	}

	completed := make(chan completedToolCall, parallelism)
	next, running := 0, 0
	cancelled := ctx.Err() != nil

	launch := func(index int) {
		job := calls[index]
		results[index] = scheduledToolResult{call: job.call, started: true}
		running++
		go func() {
			content, err := job.run(ctx)
			completed <- completedToolCall{index: index, content: content, err: err}
		}()
	}
	fill := func() {
		for !cancelled && running < parallelism && next < len(calls) {
			if ctx.Err() != nil {
				cancelled = true
				break
			}
			launch(next)
			next++
		}
	}

	fill()
	for running > 0 {
		var outcome completedToolCall
		if cancelled {
			outcome = <-completed
		} else {
			select {
			case <-ctx.Done():
				cancelled = true
				continue
			case outcome = <-completed:
			}
		}
		results[outcome.index].content = outcome.content
		results[outcome.index].err = outcome.err
		running--
		if ctx.Err() != nil {
			cancelled = true
		}
		fill()
	}

	for ; next < len(calls); next++ {
		results[next] = scheduledToolResult{
			call: calls[next].call,
			content: ToolReceipt{
				Status:      ToolResultSkipped,
				Reason:      toolCallSkippedByCancellation,
				Disposition: "RETRYABLE",
				Fix:         "re-issue the call in a new turn if it is still needed",
				Detail:      "skipped before dispatch because cancellation froze the queued call set; never dispatched",
			}.JSON(),
		}
	}
	return results
}
