package main

// guard_handoff_mode.go — the attended-interactive default for the Stop-hook task-handoff gate.
//
// The task-handoff gate (--task-handoff, ENFORCE by default) demands a valid fak.task-handoff.v1
// JSON on EVERY clean Stop and BLOCKS the stop until one is written. That structured-handoff
// discipline is right for an UNATTENDED headless/fleet worker (`fak guard -- claude -p …`), where
// the dispatched run must leave a witnessed handoff for the next worker. But on an ATTENDED
// interactive `fak guard -- claude` — a human at Claude Code's full-screen TUI — the SAME gate
// fires every time the agent yields a turn: it prints the multi-line "task handoff required"
// message onto the shared terminal (spam in the agent's UI) and refuses to let the turn end, so
// the agent cannot hand control back to the operator. That is friction the interactive operator
// never asked for, and the dominant reason a plain `fak guard -- claude` feels noisy and slow to
// settle into a usable state. The same friction appears in a local one-shot smoke (`fak guard
// --probe -- claude -p "say pong"`): the operator wants to see the guarded wire work, not prove
// fleet continuity. The explicit --probe flag gets that smoke path out of the handoff gate without
// changing the default for real headless workers.

// guardTaskHandoffEffectiveMode resolves the effective task-handoff gate mode for a session.
//
// When the operator did NOT set --task-handoff explicitly AND either (a) the wrapped child is an
// interactive agent (a TUI session, i.e. NOT a `-p`/--print one-shot) or (b) guard is in explicit
// --probe mode, the gate defaults OFF. An explicit --task-handoff value always wins (the operator
// opted in), and ordinary headless children keep the configured mode (enforce by default), so fleet
// continuity is unchanged.
//
// This mirrors the attended-interactive auto-suppression already used for the `fak info` split
// pane (guardSplitEnabled) and the per-turn debug line (guardDebugStatsToSharedStderr). It leans on
// childInteractive ALONE — not the stdin-terminal probe those two also use — because the gate is
// inappropriate for ANY interactive TUI child regardless of how the host launched it (a wrapper
// that redirects only stdin must not silently re-arm a per-turn handoff demand); the headless
// fleet path is distinguished cleanly by its `-p` flag, which makes childInteractive false unless
// the operator opted into the local smoke path with --probe.
func guardTaskHandoffEffectiveMode(configured string, explicitlySet, childInteractive, probeMode bool) string {
	if !explicitlySet && (childInteractive || probeMode) {
		return guardPreCompactModeOff
	}
	return configured
}
