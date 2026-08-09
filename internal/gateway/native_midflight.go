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
	"net/http"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
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

// isMidflightVerb reports whether a control-plane verb belongs to the mid-flight
// mailbox rather than the drive-state table. The vocabulary is agent's own closed set,
// referenced rather than restated so the route can never drift from the loop that
// drains it; an unknown verb still falls through to the drive-state control path
// unchanged.
func isMidflightVerb(verb string) bool {
	switch verb {
	case agent.MidflightInterrupt, agent.MidflightDropPendingCall, agent.MidflightSetBudget:
		return true
	}
	return false
}

// midflightRunOption opens this run's mailbox and returns the RunOption that wires it
// plus the release to defer. Called from nativeRunOptions, so every owned-loop
// entrypoint — the buffered and streamed native /v1/messages turns and the
// agent-sessions run alike — is addressable by the same control-plane verbs.
func (s *Server) midflightRunOption(reqTrace string) (agent.RunOption, func()) {
	v, release := s.midflight.register(reqTrace)
	return agent.WithMidflightVerbs(v), release
}

// handleFakSessionMidflight applies one mid-flight verb to the LIVE owned run named by
// traceID. Accepted verbs are enqueued (202) and land at the loop's next clean turn
// boundary; the response carries the run's verb journal so the caller sees the
// tamper-evident record of the command it just issued, not merely an ack.
func (s *Server) handleFakSessionMidflight(w http.ResponseWriter, r *http.Request, traceID, verb string) {
	// The honest-contract gate the sibling /steer verb applies (#3528): a mid-flight
	// verb is only ever DRAINED by an owned agent loop. The default proxy serve forwards
	// a single upstream turn and owns no loop, so an enqueue there would be an
	// accepted-but-never-applied phantom. Refuse at ingress rather than return a false
	// 202, so the operator learns the verb will not land instead of trusting a lie.
	if !s.ownsSessionLoop() {
		writeErrCode(w, http.StatusConflict, "midflight_no_owned_loop",
			"MIDFLIGHT_NO_OWNED_LOOP: this serve process forwards proxy turns and owns no agent loop to "+
				"apply a mid-flight verb at a turn boundary; start the gateway with --native, which runs "+
				"agent.RunArm and drains the mid-flight mailbox at each boundary")
		return
	}
	verbs := s.midflight.lookup(traceID)
	if verbs == nil {
		// No live run under this trace. A mid-flight verb lands at a RUNNING loop's next
		// boundary; with no such loop there is no boundary to land at, and queuing it for
		// a future run would apply an operator's stop to a run they never saw.
		writeErrCode(w, http.StatusConflict, "midflight_no_live_run",
			"MIDFLIGHT_NO_LIVE_RUN: no owned loop is currently running under this trace; a mid-flight "+
				"verb lands at a live run's next turn boundary and there is none to land at")
		return
	}
	var req SessionControlRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}

	var refusal *session.ControlRefusal
	switch verb {
	case agent.MidflightInterrupt:
		refusal = verbs.Interrupt()
	case agent.MidflightDropPendingCall:
		// The named call is the whole verb — an unnamed drop would silently skip nothing
		// (agent ignores an empty id) and read as accepted, so it is a request-shape error.
		callID := strings.TrimSpace(req.CallID)
		if callID == "" {
			writeErr(w, http.StatusBadRequest, "call_id is required for drop-pending-call")
			return
		}
		refusal = verbs.DropPendingCall(callID)
	case agent.MidflightSetBudget:
		if req.Budget == nil {
			writeErr(w, http.StatusBadRequest, "budget is required for set-budget")
			return
		}
		// Same projection the drive-state `budget` verb uses in cmd/fak: the priced spend
		// axis rides the budget wire (#2762), so carrying it here keeps a mid-flight write
		// from silently clearing a live spend ceiling.
		refusal = verbs.SetBudget(session.Budget{
			TurnsLeft:           req.Budget.TurnsLeft,
			TokensLeft:          req.Budget.TokensLeft,
			ContextTokensLeft:   req.Budget.ContextTokensLeft,
			SpendMicroCentsLeft: req.Budget.SpendMicroCentsLeft,
			SpendMicroCentsCap:  req.Budget.SpendMicroCentsCap,
		})
	}
	if refusal != nil {
		// The arm returned between the lookup and the enqueue: the mailbox is sealed and
		// refuses with the closed CONTROL_SESSION_TERMINAL token — the same terminal-session
		// refusal the drive-state verbs give, mapped to the same 409.
		writeErrCode(w, http.StatusConflict, "midflight_refused", refusal.Reason+": "+refusal.Detail)
		return
	}
	s.logf("gateway: session %s midflight %s queued for the next turn boundary", traceID, verb)
	writeJSON(w, http.StatusAccepted, map[string]any{
		"trace_id": traceID,
		"verb":     verb,
		"queued":   true,
		"journal":  verbs.Journal(),
	})
}
