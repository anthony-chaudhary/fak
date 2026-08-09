package streamrules

import (
	"fmt"
	"strings"
	"time"
)

const RetryDelay = 50 * time.Millisecond

type InterruptHooks struct {
	Abort          func(toolCallID, reason string)
	DiscardPartial func(toolCallID string)
	Inject         func(message string)
	Resume         func(toolCallID string) error
	Resolve        func(toolCallID string)
}

type Retry struct {
	token            uint64
	toolCallID       string
	promptGeneration uint64
	targetMessageID  string
}

type Runtime struct {
	enabled bool
	next    uint64
	pending map[string]Retry
}

func NewRuntime(enabled bool) *Runtime {
	return &Runtime{enabled: enabled, pending: make(map[string]Retry)}
}

func (r *Runtime) Begin(match Match, promptGeneration uint64, targetMessageID string, hooks InterruptHooks) (Retry, bool) {
	if r == nil || !r.enabled || !match.Interrupt || strings.TrimSpace(match.SubstituteAction) == "" || match.Key.ToolCallID == "" {
		return Retry{}, false
	}
	r.next++
	retry := Retry{token: r.next, toolCallID: match.Key.ToolCallID, promptGeneration: promptGeneration, targetMessageID: targetMessageID}
	r.pending[retry.toolCallID] = retry
	if hooks.Abort != nil {
		hooks.Abort(retry.toolCallID, "substitute-action:"+match.Rule)
	}
	if hooks.DiscardPartial != nil {
		hooks.DiscardPartial(retry.toolCallID)
	}
	if hooks.Inject != nil {
		hooks.Inject(substituteMessage(match.SubstituteAction))
	}
	return retry, true
}

// Schedule defers the retry so stream callbacks can unwind before resume. The
// current prompt and target identities are read only when the timer fires.
func (r *Runtime) Schedule(retry Retry, promptGeneration func() uint64, targetMessageID func() string, hooks InterruptHooks) *time.Timer {
	return time.AfterFunc(RetryDelay, func() {
		var generation uint64
		if promptGeneration != nil {
			generation = promptGeneration()
		}
		var messageID string
		if targetMessageID != nil {
			messageID = targetMessageID()
		}
		r.Attempt(retry, generation, messageID, hooks)
	})
}

func (r *Runtime) Cancel(callID string) {
	if r != nil {
		delete(r.pending, callID)
	}
}

func (r *Runtime) Attempt(retry Retry, promptGeneration uint64, targetMessageID string, hooks InterruptHooks) bool {
	if r == nil {
		resolve(hooks, retry.toolCallID)
		return false
	}
	pending, abortPending := r.pending[retry.toolCallID]
	current := abortPending && pending.token == retry.token && retry.promptGeneration == promptGeneration && retry.targetMessageID == targetMessageID
	delete(r.pending, retry.toolCallID)
	if !current {
		resolve(hooks, retry.toolCallID)
		return false
	}
	ok := hooks.Resume != nil && hooks.Resume(retry.toolCallID) == nil
	resolve(hooks, retry.toolCallID)
	return ok
}

func substituteMessage(action string) string {
	return fmt.Sprintf("%s\n\nContinue from the updated result.", strings.TrimSpace(action))
}

func resolve(hooks InterruptHooks, callID string) {
	if hooks.Resolve != nil {
		hooks.Resolve(callID)
	}
}
