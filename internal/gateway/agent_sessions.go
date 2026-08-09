package gateway

// agent_sessions.go — the agent-runtime spine endpoint (#3258, epic #3256,
// workstream B). POST /v1/fak/agent/sessions accepts a goal and streams back ONE
// kernel-governed owned-loop session as NDJSON events: the industry's "trusted
// application runtime owns the agent loop" pattern, served from the same gateway
// assembly every other route rides.
//
// The loop is agent.RunGovernedArm — RunArm(fak=true) over the server's planner
// (offline mock, --gguf in-kernel, or the proxy upstream), so every proposed tool
// call crosses the in-kernel syscall boundary (adjudication, vDSO, quarantine)
// and the endpoint runs OFFLINE in CI against the deterministic mock. The wired
// nativeRunOptions thread the live session gate / stop gate / route manifest in,
// exactly as the native /v1/messages loop does.
//
// Wire shape (one JSON object per line, flushed as emitted):
//
//	{"event":"session.start", "trace_id":..., "goal":..., "max_turns":N}
//	{"event":"call", "trace_id":..., "call":{turn,tool,verdict,reason,by,...}}   (per adjudicated call)
//	{"event":"session.end", "trace_id":..., "metrics":{ArmMetrics}}
//	{"event":"error", "trace_id":..., "error":"..."}                             (loop failure, in-stream)
//
// Honest fence for THIS slice: session.start is flushed before the loop runs (the
// stream opens immediately), and the per-call rows + terminal metrics are flushed
// once the governed run completes — the loop records its trace synchronously and
// exposes it post-run. Incremental per-turn emission rides the typed
// loop-progress observer seam (#5148) when it lands; the wire contract here is
// what that hardening (#B6, #2388) extends, not replaces.

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// AgentSessionRequest is the body of POST /v1/fak/agent/sessions: the goal the
// governed loop is seeded with, and an optional per-session turn cap. MaxTurns
// <= 0 (or above the server's configured native cap) falls back to that cap.
type AgentSessionRequest struct {
	Goal     string `json:"goal"`
	MaxTurns int    `json:"max_turns,omitempty"`
}

// AgentSessionEvent is one NDJSON line of a streamed governed session. Event is
// the closed discriminator (session.start | call | session.end | error); exactly
// the fields that event carries are populated.
type AgentSessionEvent struct {
	Event    string            `json:"event"`
	TraceID  string            `json:"trace_id,omitempty"`
	Goal     string            `json:"goal,omitempty"`
	MaxTurns int               `json:"max_turns,omitempty"`
	Call     *agent.CallTrace  `json:"call,omitempty"`
	Metrics  *agent.ArmMetrics `json:"metrics,omitempty"`
	Error    string            `json:"error,omitempty"`
}

// agentSessionMaxTurns resolves the effective turn cap for one session: the
// client's requested cap when positive and within the server's configured native
// cap, else that server cap (DefaultNativeMaxTurns when unconfigured). The
// client can narrow the budget, never widen it.
func (s *Server) agentSessionMaxTurns(requested int) int {
	cap := nativeMaxTurnsOr(s.nativeMaxTurns)
	if requested > 0 && requested < cap {
		return requested
	}
	return cap
}

// handleFakAgentSessions serves POST /v1/fak/agent/sessions: decode the goal,
// open the NDJSON stream, drive one kernel-governed owned loop to a terminal
// state, and stream the witnessed decision trace + ArmMetrics back. Request
// shape errors (wrong method, malformed body, missing goal) fail as plain JSON
// errors before any stream bytes are written; a loop failure after the stream
// opened rides in-stream as a terminal error event with the upstream detail kept
// off the wire (the same trust boundary writeUpstreamErr enforces).
func (s *Server) handleFakAgentSessions(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	var req AgentSessionRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}
	if strings.TrimSpace(req.Goal) == "" {
		writeErr(w, http.StatusBadRequest, "goal: field required")
		return
	}
	maxTurns := s.agentSessionMaxTurns(req.MaxTurns)
	reqTrace := s.useHTTPTrace(w, r, "")

	h := w.Header()
	h.Set("Content-Type", "application/x-ndjson")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	emit := func(ev AgentSessionEvent) {
		raw, err := json.Marshal(ev)
		if err != nil {
			return
		}
		if _, err := w.Write(append(raw, '\n')); err != nil {
			return
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	emit(AgentSessionEvent{Event: "session.start", TraceID: reqTrace, Goal: req.Goal, MaxTurns: maxTurns})

	began := time.Now()
	ensureGovernedRungs()
	opts, release := s.nativeRunOptions(r.Context(), reqTrace)
	defer release()
	m, calls, err := agent.RunGovernedArm(r.Context(), s.planner, req.Goal, maxTurns, opts...)
	if err != nil {
		// The stream is already open (200 written), so the failure is reported as the
		// terminal in-stream event. Detail goes to the operator log only — the raw
		// upstream error must not cross the trust boundary to the caller.
		s.logf("gateway: agent session loop error (trace %s): %v", reqTrace, err)
		emit(AgentSessionEvent{Event: "error", TraceID: reqTrace, Error: "agent loop error"})
		return
	}
	for i := range calls {
		emit(AgentSessionEvent{Event: "call", TraceID: reqTrace, Call: &calls[i]})
	}
	s.logInferenceTurn(reqTrace, "fak_agent_sessions", true, agent.Usage{
		PromptTokens:     m.PromptTokens,
		CompletionTokens: m.CompletionTokens,
	}, nativeFinishReason(m), time.Since(began), false)
	emit(AgentSessionEvent{Event: "session.end", TraceID: reqTrace, Metrics: &m})
}
