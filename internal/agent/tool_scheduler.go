package agent

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/session"
)

const toolCallSkippedByCancellation = session.ReasonInterrupted

type toolEffectClass uint8

const (
	toolEffectExclusive toolEffectClass = iota
	toolEffectSafe
)

type scheduledToolCall struct {
	call   ToolCall
	effect toolEffectClass
	run    func(context.Context) (string, error)
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

// toolEffectFor fails closed: only calls carrying both native read-only and
// idempotent attestations may overlap.
func toolEffectFor(tool string) toolEffectClass {
	meta := metaFor(tool)
	if meta["readOnlyHint"] == "true" && meta["idempotentHint"] == "true" {
		return toolEffectSafe
	}
	return toolEffectExclusive
}

// runScheduledToolCalls overlaps contiguous effect-safe bodies. Exclusive calls
// run alone and form barriers. Results always return in model order, independent
// of body completion order.
func runScheduledToolCalls(ctx context.Context, parallelism int, calls []scheduledToolCall) []scheduledToolResult {
	results := make([]scheduledToolResult, len(calls))
	if parallelism < 1 {
		parallelism = 1
	}
	for start := 0; start < len(calls); {
		end := start + 1
		limit := 1
		if calls[start].effect == toolEffectSafe {
			limit = parallelism
			for end < len(calls) && calls[end].effect == toolEffectSafe {
				end++
			}
		}
		runScheduledToolBatch(ctx, limit, calls[start:end], results[start:end])
		start = end
	}
	return results
}

func runScheduledToolBatch(ctx context.Context, parallelism int, calls []scheduledToolCall, results []scheduledToolResult) {
	if len(calls) == 0 {
		return
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
		results[next] = scheduledToolResult{call: calls[next].call, content: ToolReceipt{
			Status: ToolResultSkipped, Reason: toolCallSkippedByCancellation, Disposition: "RETRYABLE",
			Fix:    "re-issue the call in a new turn if it is still needed",
			Detail: "skipped before dispatch because cancellation froze the queued call set; never dispatched",
		}.JSON()}
	}
}
