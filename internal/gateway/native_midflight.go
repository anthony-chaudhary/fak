package gateway

// native_midflight.go — the TRANSPORT half of #2403 (harness-native epic #2388,
// program #2387): the session control plane's mid-flight verbs reach the LIVE owned
// run.
//
// #5148 made the owned loop's state legible — typed progress SSE (turn_started /
// call_adjudicated / tool_started / result_admitted / turn_done) interleaved with the
// text deltas, each tagged with the request trace. #5158 built the write half's LOOP
// layer — agent.MidflightVerbs, the per-run mailbox drained at the next CLEAN turn
// boundary (never mid-tool, never mid-adjudication) with a tamper-evident journal.
//
// What was missing is the wire between them. MidflightVerbs' own contract names "a
// transport (route / CLI)" that enqueues onto it, and there was none: the mailbox had
// ZERO callers outside internal/agent, so an operator watching a run over the typed
// stream still had no way to act on what they saw. Streaming carried the loop's state
// out; nothing carried a decision back in.
//
// This file is that transport. Every owned-loop run registers a mailbox under its
// request trace for the life of the run, and
//
//	POST /v1/fak/session/{trace_id}/{interrupt | drop-pending-call | set-budget}
//
// looks the live run up by that same trace and enqueues. The loop applies the verb at
// its next boundary exactly as #5158 specified — this route adds REACHABILITY, never a
// second policy. The key an operator POSTs to is the key they already hold: the
// `session` tag on every typed progress event.
//
// Fail-closed, mirroring the sibling /steer verb (#3528): a serve process that owns no
// agent loop, or a trace with no live run, is a 409 with a closed reason — never a 202
// for a verb nothing will ever drain.

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// midflightRuns is the live per-trace mid-flight mailbox registry: the lookup a
// control-plane verb crosses to reach the run it names. Keyed by request trace — the
// same id the typed progress events carry — so watching a run and steering it use one
// identifier. The zero value is ready to use.
type midflightRuns struct {
	mu   sync.Mutex
	runs map[string]*agent.MidflightVerbs
}

// register opens a mailbox for one owned-loop run and returns it alongside the release
// the caller defers. An empty trace is UNADDRESSABLE — there is no key an operator
// could POST to — so it registers nothing and returns a nil mailbox, which
// agent.WithMidflightVerbs accepts as the historical (mailbox-free) loop.
//
// Concurrent runs sharing one trace are last-registered-wins for lookup; release drops
// the entry only while it is still this run's, so a finishing run never unregisters its
// successor. The mailbox itself is sealed by the loop (agent's own defer at the arm's
// return), so a verb that races a run's end refuses with the closed
// CONTROL_SESSION_TERMINAL token rather than silently vanishing.
func (m *midflightRuns) register(trace string) (*agent.MidflightVerbs, func()) {
	if trace == "" {
		return nil, func() {}
	}
	v := agent.NewMidflightVerbs()
	m.mu.Lock()
	if m.runs == nil {
		m.runs = map[string]*agent.MidflightVerbs{}
	}
	m.runs[trace] = v
	m.mu.Unlock()
	return v, func() {
		m.mu.Lock()
		if m.runs[trace] == v {
			delete(m.runs, trace)
		}
		m.mu.Unlock()
	}
}

// lookup returns the live mailbox registered under trace, or nil when no owned run
// holds it.
func (m *midflightRuns) lookup(trace string) *agent.MidflightVerbs {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.runs[trace]
}

// midflightRunOption opens this run's mailbox and returns the RunOption that wires it
// plus the release to defer. Called from nativeRunOptions, so every owned-loop
// entrypoint — the buffered and streamed native /v1/messages turns and the
// agent-sessions run alike — is addressable by the same control-plane verbs.
func (s *Server) midflightRunOption(reqTrace string) (agent.RunOption, func()) {
	v, release := s.midflight.register(reqTrace)
	return agent.WithMidflightVerbs(v), release
}
