package main

// serve_session_tag.go — the PRODUCER for sessionctl's broadcast tag registry (#5640,
// epic #5632). The registry, the selector resolver and the fail-closed rule all shipped
// with #2764; nothing ever wrote to the registry. That made half the documented fleetbus
// selector grammar inert: serve_fleetbus.go builds a sessionctl.BroadcastSelector
// straight from a fleet directive's --lane/--wave/--label, resolved it against an
// always-empty map, and matched zero sessions on every instance — indistinguishable, to
// an operator, from an empty fleet.
//
// What this file adds is the missing call, not a design. Every served session crosses
// decideSession (the host side of the gateway's beginServedSessionTurn admission gate),
// so that is the one boundary that sees every trace a fleet directive can reach on this
// process. Each admitted trace is tagged there from the ROUTING IDENTITY THE PROCESS WAS
// LAUNCHED UNDER, and the tag is dropped when the session reaches its terminal state.
//
// Where the metadata comes from. A serve/guard process is launched by the dispatch
// worker launcher for ONE lane's work, so the lane/wave/label are process facts, read
// from the environment (FAK_SESSION_LANE / FAK_SESSION_WAVE / FAK_SESSION_LABELS) — the
// same FAK_SESSION_* family the launcher already threads FAK_SESSION_ID through. That is
// the metadata "available at the admission site"; nothing on the request wire carries a
// per-session lane today, and inventing a header for one would be a grammar change, not
// this fix.
//
// Admission FILLS, it never OVERWRITES. decideSession runs once per TURN, not once per
// session, so an unconditional TagSession would re-stamp the process identity over a
// more precise tag every turn. An in-process spawn site that knows better (a wave driver
// tagging its cohort, a test seeding two lanes on one process) calls
// sessionctl.TagSession first and keeps it; admission only covers the traces nobody
// claimed.
//
// Fail-closed is preserved end to end: an unconfigured process declares no routing
// identity, tags nothing, and its sessions keep matching no selector — exactly as today.
// Turning the registry on is an operator's explicit act.
//
// Spawn sites that remain UNTAGGED by this change, named rather than silently omitted:
// sessions created outside a serve process (issue-scoped out), and any serve launched
// with no FAK_SESSION_LANE/WAVE/LABELS in its environment. Both are reported by the
// zero-match refusal serve_fleetbus.go's noMatchDetail already writes ("none of this
// instance's N live session(s) are tagged lane=..."), which is what keeps a zero-match
// broadcast distinguishable from a successful one.

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// The environment a launcher declares this process's session routing identity in. They
// join the FAK_SESSION_* family (FAK_SESSION_ID, FAK_SESSION_REGISTRY, ...) because they
// describe the SESSIONS this process serves, not the process itself — the fleetbus
// instance axes (machine, role) already cover the process.
const (
	serveSessionLaneEnv   = "FAK_SESSION_LANE"
	serveSessionWaveEnv   = "FAK_SESSION_WAVE"
	serveSessionLabelsEnv = "FAK_SESSION_LABELS"
)

// serveSessionTagMeta resolves this process's declared session routing identity. Read
// from the environment on each call rather than cached at init: a guard child is
// relaunched in-process across a budget reset (guard_child.go re-execs with a fresh
// FAK_SESSION_ID), and a cached identity would outlive the launch it described. The zero
// meta means "this process declares no routing identity" — TagSession ignores it, so an
// unconfigured serve is byte-for-byte the pre-#5640 behavior.
func serveSessionTagMeta() sessionctl.BroadcastMeta {
	return sessionctl.BroadcastMeta{
		Lane:   strings.TrimSpace(os.Getenv(serveSessionLaneEnv)),
		Wave:   strings.TrimSpace(os.Getenv(serveSessionWaveEnv)),
		Labels: parseServeSessionLabels(os.Getenv(serveSessionLabelsEnv)),
	}
}

// parseServeSessionLabels splits the comma-separated label list, dropping blanks. A
// label is an exact, case-sensitive token the selector matches whole (BroadcastSelector
// is a closed operator token, never a pattern), so only surrounding space is trimmed.
func parseServeSessionLabels(raw string) []string {
	var out []string
	for _, part := range strings.Split(raw, ",") {
		if l := strings.TrimSpace(part); l != "" {
			out = append(out, l)
		}
	}
	return out
}

// tagServedSessionAdmit is the admission-boundary tagger: it records trace's routing
// metadata for broadcast selection, and drops it once the session has ended.
//
// The ORDER matters. The teardown check runs first so a Decide that finalizes a draining
// session to Stopped clears the tag on that same boundary instead of re-tagging a corpse
// — a stale tag on a dead trace is how a selector comes to "match" a session that cannot
// take the op. Only Stopped clears: Draining and Terminating each advance one more
// boundary and must stay selectable, because "terminate the sessions that are still
// draining" is a real operator move.
//
// Tagging is fill-only (see the file header) and cheap: one map read under the registry
// mutex per turn for an already-tagged trace.
func tagServedSessionAdmit(trace string, st session.State) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return
	}
	if st.Run == session.Stopped {
		sessionctl.ClearSessionTag(trace)
		return
	}
	if _, tagged := sessionctl.SessionTag(trace); tagged {
		return
	}
	sessionctl.TagSession(trace, serveSessionTagMeta())
}

// dropServedSessionTagIfEnded is the teardown half on the paths that end a session
// WITHOUT a following admission: an operator POSTing run=stopped through the control
// route never produces another decideSession for that trace, so nothing else would ever
// clear it. Idempotent, and a no-op for every non-terminal state.
func dropServedSessionTagIfEnded(trace string, st session.State) {
	if st.Run != session.Stopped {
		return
	}
	if trace = strings.TrimSpace(trace); trace != "" {
		sessionctl.ClearSessionTag(trace)
	}
}
