package agent

// loop_session.go — the session-control seam for the agent turn loop. It threads a
// per-session DRIVE state (internal/session.Table) into RunArm as an OPTIONAL
// trailing option, so the loop reads its budget/pace/run-state each turn instead of
// running blindly to a fixed maxTurns. With no option passed, runConfig.table is nil
// and session.Table.Decide is a permissive no-op (nil receiver) — so every existing
// caller and the default loop are byte-for-byte unchanged. This is the live-loop
// half of docs/notes/SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md.

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// RunOption configures an optional behavior of RunArm / Run. The zero set of options
// is the historical behavior; each option opts into one capability (today: a session
// drive-state table). It is the variadic-options idiom so adding a capability never
// breaks an existing positional call site.
type RunOption func(*runConfig)

// runConfig is the resolved option set for one RunArm invocation. The zero value is
// the historical loop (nil table => permissive Decide => no per-turn gate).
type runConfig struct {
	table *session.Table
	trace string
}

// WithSessionTable wires a per-session drive-state table and the trace id this run is
// keyed under into RunArm. Each turn boundary the loop calls table.Decide(trace) to
// gate the turn on the session's live run-state + budget + pace, and Debit reports the
// turn's token usage back. A nil table is accepted (it degrades to the historical
// loop), so a caller can pass the option unconditionally.
func WithSessionTable(table *session.Table, trace string) RunOption {
	return func(c *runConfig) {
		c.table = table
		c.trace = trace
	}
}

// resolveRunConfig folds the options into a runConfig.
func resolveRunConfig(opts []RunOption) runConfig {
	var c runConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&c)
		}
	}
	return c
}

// gateTurn applies the per-turn session gate at a turn boundary. It returns the
// per-turn output-token cap to lower into the planner (0 = no cap / planner default),
// whether the loop should PROCEED with this turn, and the stop reason when it should
// not. A nil table proceeds with no cap (the historical loop). When pace asks for an
// inter-turn gap, gateTurn sleeps it here (respecting ctx cancellation) so a throttled
// session is paced without the loop body needing to know about timing.
//
// PAUSED is a non-terminal hold: gateTurn returns proceed=false with the PAUSED reason
// so the loop stops THIS run cleanly (the harness loop is single-shot; a long-lived
// gateway loop would instead wait and re-Decide — that wait belongs to the gateway
// integration, #555, not the A/B harness). DRAINING/STOPPED/budget-exhausted return
// proceed=false with a terminal reason.
func (c runConfig) gateTurn(ctx contextLike) (maxTokens int, proceed bool, reason string) {
	if c.table == nil {
		return 0, true, ""
	}
	v := c.table.Decide(c.trace)
	if !v.Proceed {
		return 0, false, v.Reason
	}
	if v.MinGapMs > 0 {
		select {
		case <-ctx.Done():
			return 0, false, v.Reason
		case <-time.After(time.Duration(v.MinGapMs) * time.Millisecond):
		}
	}
	return v.MaxTokens, true, ""
}

// debitTurn reports a completed turn's usage to the session table so the output and
// context budget axes decrement. A nil table is a no-op.
func (c runConfig) debitTurn(usage Usage) {
	if c.table != nil {
		c.table.DebitUsage(c.trace, session.Usage{
			OutputTokens:  usage.CompletionTokens,
			ContextTokens: usage.ContextWindowTokens(),
		})
	}
}

// sampleOptsFor turns a per-turn output-token cap into the variadic SampleOpt slice
// for p.Complete. A non-positive cap returns NO options, so the planner call is
// byte-identical to the pre-seam p.Complete(ctx, messages, tools) — the historical
// path is untouched. A positive cap lowers WithMaxTokens, capping THIS turn's output
// (the pace throttle), which WithMaxTokens itself further guards (it no-ops on n<=0).
func sampleOptsFor(maxTokens int) []SampleOpt {
	if maxTokens <= 0 {
		return nil
	}
	return []SampleOpt{WithMaxTokens(maxTokens)}
}

// contextLike is the narrow slice of context.Context gateTurn needs (Done). It lets
// the helper be unit-tested with a fake without importing a real context, and keeps
// this file's surface honest about what it uses.
type contextLike interface {
	Done() <-chan struct{}
}
