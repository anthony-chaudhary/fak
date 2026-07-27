package agent

import (
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// spliceTurnDirectives folds every operator- and session-sourced directive drained at
// THIS turn boundary into the turn's input, in the fixed order runArm established:
// operator steer (#850), objective redirect (#2755), tightened constraint floor
// (#2756), then the context-spike advisory (#2197) riding closest to the model's next
// turn. Each drain is a no-op without a wired trace/table/gate, so a run with nothing
// enqueued gets its messages back byte-for-byte unchanged — this is code motion out of
// runArm (the god-function growth gate), not a behavior change.
func spliceTurnDirectives(cfg runConfig, messages []Message) []Message {
	// Steer splice (#850): a running session drains any operator steer enqueued on
	// the a2achan Session-locale bus and folds it into THIS turn's input. With no
	// trace wired (or an empty mailbox) this is a no-op, so the historical loop is
	// byte-for-byte unchanged. This is the consumer half #760 deferred.
	if steer := cfg.drainSteer(); steer != "" {
		messages = append(messages, Message{Role: RoleUser, Content: steer})
	}

	// Redirect apply (#2755): a running session drains any operator redirect /
	// set-objective op enqueued on the sessionctl mailbox and folds it into the
	// live per-session objective, carried into THIS turn as a first-class
	// objective directive (a SYSTEM message, categorically distinct from the
	// user-message steer splice above). This is the semantic op the epic names —
	// the objective CHANGES; it is not another interpreted operator turn. A no-op
	// without a wired trace or a set objective, so the historical loop is
	// byte-for-byte unchanged.
	if objective := cfg.applyRedirect(); objective != "" {
		messages = append(messages, Message{Role: RoleSystem, Content: objective})
	}

	// Constraint apply (#2756): a running session drains any operator
	// add-constraint op enqueued on the sessionctl mailbox at this boundary —
	// never mid-turn — folds it into the live per-session floor, and carries
	// the tightened floor's standing notice as a SYSTEM directive so the model
	// knows what narrowed. The floor itself binds at tool dispatch in runArm
	// (constraintDenied). A no-op without a wired trace or a constrained
	// floor, so the historical loop is byte-for-byte unchanged.
	if floor := cfg.applyConstraints(); floor != "" {
		messages = append(messages, Message{Role: RoleSystem, Content: floor})
	}

	// Context-spike nudge (#2197): when the session's cost ring shows the context
	// window grew suddenly and materially last turn, splice the advisory into THIS
	// turn's input so the model corrects its own context use (windowed reads,
	// summarize-don't-carry) before any gate has to act. Rides after any operator
	// steer, closest to the model's next turn. Advisory only, and a no-op without
	// a wired trace/table/gate — the historical loop is byte-for-byte unchanged.
	if nudge := cfg.contextNudge(); nudge != "" {
		sessionctl.RecordContextAdvisoryNext(cfg.trace, nudge)
		messages = append(messages, Message{Role: RoleUser, Content: nudge})
	}

	return messages
}
