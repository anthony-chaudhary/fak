package agent

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/session"
)

const toolCallSkippedByCancellation = session.ReasonInterrupted

// maxParallelToolCalls bounds one model turn's in-process tool fan-out. Keeping
// the ceiling small prevents a malformed or over-eager turn from turning tool
// overlap into an unbounded goroutine/resource burst.
const maxParallelToolCalls = 4

type toolEffectClass uint8

const (
	toolEffectExclusive toolEffectClass = iota
	toolEffectSafe
)

type scheduledToolCall struct {
	call   ToolCall
	effect toolEffectClass
	admit  func() *scheduledToolSkip
	start  func()
	run    func(context.Context) (string, error)
	commit func(scheduledToolResult) error
}

type scheduledToolResult struct {
	call    ToolCall
	content string
	err     error
	started bool
	skip    *scheduledToolSkip
}

type scheduledToolSkip struct {
	reason        string
	by            string
	disposition   string
	detail        string
	suppressTrace bool
}

func (s scheduledToolSkip) receipt() string {
	fix := "re-issue the call in a new turn if it is still needed"
	if s.disposition == "TERMINAL" {
		fix = "raise the session tool-call budget before issuing more calls"
	}
	return ToolReceipt{
		Status: ToolResultSkipped, Reason: s.reason, Disposition: s.disposition,
		Fix: fix, Detail: s.detail,
	}.JSON()
}

func interruptedToolSkip() *scheduledToolSkip {
	return &scheduledToolSkip{
		reason: toolCallSkippedByCancellation, by: "tool-scheduler/interruption", disposition: "RETRYABLE",
		detail: "skipped before dispatch because cancellation froze the queued call set; never dispatched",
	}
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
// run alone and form barriers. Each segment is committed in model order before
// the next segment starts, so a failed read admission can never be crossed by a
// later mutation. Results always return in model order.
func runScheduledToolCalls(ctx context.Context, parallelism int, calls []scheduledToolCall) ([]scheduledToolResult, error) {
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
		frozen := runScheduledToolBatch(ctx, limit, calls[start:end], results[start:end])
		for i := start; i < end; i++ {
			if calls[i].commit != nil {
				if err := calls[i].commit(results[i]); err != nil {
					return results, fmt.Errorf("commit tool call %q: %w", calls[i].call.ID, err)
				}
			}
		}
		if frozen != nil {
			for i := end; i < len(calls); i++ {
				results[i] = scheduledToolResult{call: calls[i].call, content: frozen.receipt(), skip: frozen}
				if calls[i].commit != nil {
					if err := calls[i].commit(results[i]); err != nil {
						return results, fmt.Errorf("commit tool call %q: %w", calls[i].call.ID, err)
					}
				}
			}
			return results, nil
		}
		start = end
	}
	return results, nil
}

func runScheduledToolBatch(ctx context.Context, parallelism int, calls []scheduledToolCall, results []scheduledToolResult) *scheduledToolSkip {
	if len(calls) == 0 {
		return nil
	}
	if parallelism > len(calls) {
		parallelism = len(calls)
	}
	completed := make(chan completedToolCall, parallelism)
	next, running := 0, 0
	cancelled := ctx.Err() != nil
	var frozen *scheduledToolSkip
	launch := func(index int) bool {
		job := calls[index]
		if job.admit != nil {
			if skip := job.admit(); skip != nil {
				frozen = skip
				results[index] = scheduledToolResult{call: job.call, content: skip.receipt(), skip: skip}
				return false
			}
		}
		if job.start != nil {
			job.start()
		}
		results[index] = scheduledToolResult{call: job.call, started: true}
		running++
		go func() {
			content, err := job.run(ctx)
			completed <- completedToolCall{index: index, content: content, err: err}
		}()
		return true
	}
	fill := func() {
		for !cancelled && frozen == nil && running < parallelism && next < len(calls) {
			if ctx.Err() != nil {
				cancelled = true
				break
			}
			if !launch(next) {
				next++
				break
			}
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
		skip := frozen
		if skip == nil {
			skip = interruptedToolSkip()
		}
		results[next] = scheduledToolResult{call: calls[next].call, content: skip.receipt(), skip: skip}
	}
	if frozen != nil {
		return frozen
	}
	if cancelled {
		return interruptedToolSkip()
	}
	return nil
}
