package gateway

// session_admit.go — the PROXY-path enforcement of the session control surface. The
// control routes (#620) let an operator write a served session's DRIVE state; fak's
// OWN agent turn loop honors it at each turn boundary (agent.RunArm + WithSessionTable).
// On the PROXIED serve/guard path, each /v1/{chat,messages,generateContent} request is
// the same boundary: beginServedSessionTurn asks the injected session.Table.Decide to
// admit/refuse the request, debit TurnsLeft, and hand back pace caps before the model
// turn runs; debitServedSessionTurn reports the post-response usage so output/context
// budgets are exhausted at the right boundary.
//
// That is what makes "cancel a request in flight" and "budget a served session" real
// on the flagship path — the operator POSTs draining/stopped or budget/pace changes,
// and the agent's subsequent calls are refused or throttled at the boundary, cleanly,
// with the reason.
//
// HONEST SCOPE. This refuses the NEXT request for the session; an already-open upstream
// round-trip rides its own request context (the operator's stop takes effect at the
// next call boundary, never mid-stream — the same boundary discipline the design owns).
// It keys on the request TraceID. An operator can target a session when the client
// sends a stable X-Trace-Id OR the host configured a stable DefaultTraceID (as guard
// does for wrapped CLIs); a minted per-request gw-<n> remains, by construction, not
// externally addressable. THROTTLED is admitted (pace shapes fak's own loop, not proxy
// admission); only the non-advancing states refuse.

import (
	"context"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const (
	sessionReasonBudgetContext = "BUDGET_CONTEXT_EXHAUSTED"
	sessionReasonBudgetTokens  = "BUDGET_TOKENS_EXHAUSTED"
)

type servedSessionTurn struct {
	traceID   string
	state     SessionState
	maxTokens int
	minGapMs  int
}

// beginServedSessionTurn applies the live session gate to one proxied model request.
// When DecideSession is wired, this is the mutating boundary: session.Table.Decide
// debits TurnsLeft, resolves pause/drain/stop/budget exhaustion, and returns the pace
// caps for THIS request. When only ObserveSession is wired, it falls back to the
// shipped run-state admission guard. With neither hook, it is fail-open and leaves the
// historical request path unchanged.
func (s *Server) beginServedSessionTurn(ctx context.Context, trace string) (servedSessionTurn, bool, bool) {
	turn := servedSessionTurn{traceID: trace}
	if trace == "" {
		return turn, true, false
	}
	if s.decideSession != nil {
		v := s.decideSession(ctx, trace)
		turn.state = v.State
		turn.maxTokens = v.MaxTokens
		turn.minGapMs = v.MinGapMs
		if !v.Proceed {
			if turn.state.TraceID == "" {
				turn.state.TraceID = trace
			}
			if turn.state.Reason == "" {
				turn.state.Reason = v.Reason
			}
			return turn, false, false
		}
		if turn.minGapMs > 0 {
			timer := time.NewTimer(time.Duration(turn.minGapMs) * time.Millisecond)
			defer timer.Stop()
			select {
			case <-ctx.Done():
				return turn, false, true
			case <-timer.C:
			}
		}
		return turn, true, false
	}
	ok, st := s.sessionAdmits(ctx, trace)
	turn.state = st
	return turn, ok, false
}

// isBudgetResetReason reports whether a refused session state is one the human-like
// reset should fire on: a long-context exhaustion (which mints a continuation id) or
// an output-token exhaustion. These are the budget drains a fresh window continues
// past; an operator pause/stop is NOT reset (the operator meant to halt it).
func isBudgetResetReason(st SessionState) bool {
	return st.ContinuationID != "" ||
		st.Reason == sessionReasonBudgetContext ||
		st.Reason == sessionReasonBudgetTokens
}

// maybeResetOnBudget is the opt-in auto-reset boundary. When a served request was
// refused for a budget drain AND Config.ResetOnBudget is wired, it asks the host to
// distill a carryover seed and re-arm a fresh session, then returns the fresh trace
// and the seed messages to splice ahead of the live request so the client transparently
// continues. ok=false means "fall back to the historical 409 refusal" — either the
// refusal was not a budget drain, the hook is not wired, or the host declined. The
// gateway never imports internal/session or internal/sessionreset; the host owns both.
func (s *Server) maybeResetOnBudget(ctx context.Context, st SessionState, messages []agent.Message) (newTrace string, seed []agent.Message, ok bool) {
	if s.resetOnBudget == nil || !isBudgetResetReason(st) {
		return "", nil, false
	}
	return s.resetOnBudget(ctx, st.TraceID, messages)
}

// spliceSeed prepends the carryover seed to a live transcript, keeping any leading
// system message(s) at the very top (a provider expects the system prompt first). The
// seed's continuation recap lands AFTER the system framing and BEFORE the historical
// user/assistant turns, so the fresh window reads: system prompt -> "here's the
// carried-over context" -> the (now budget-fit) recent turns. An empty seed is the
// identity. The original slice is not mutated (a fresh slice is returned).
func spliceSeed(seed, messages []agent.Message) []agent.Message {
	if len(seed) == 0 {
		return messages
	}
	lead := 0
	for lead < len(messages) && messages[lead].Role == agent.RoleSystem {
		lead++
	}
	out := make([]agent.Message, 0, len(messages)+len(seed))
	out = append(out, messages[:lead]...) // leading system framing
	out = append(out, seed...)            // the carryover recap
	out = append(out, messages[lead:]...) // the historical turns
	return out
}

// maxTokensFor lowers a client's requested max_tokens by the session pace cap. A
// zero/non-positive value on either side means "no cap from that side"; when both are
// present the smaller cap wins, so session pace can never raise a client-requested
// limit.
func (t servedSessionTurn) maxTokensFor(requestMax int) int {
	switch {
	case t.maxTokens <= 0:
		return requestMax
	case requestMax <= 0:
		return t.maxTokens
	case t.maxTokens < requestMax:
		return t.maxTokens
	default:
		return requestMax
	}
}

// debitServedSessionTurn reports the provider usage after a served model request.
// Usage is known only post-response; session.Table.DebitUsage records the debit
// now, and the next Decide takes any normal budget-exhaustion stop at the boundary.
func (s *Server) debitServedSessionTurn(ctx context.Context, turn servedSessionTurn, usage agent.Usage) {
	su := sessionUsageFromAgent(usage)
	if s.debitSession == nil || turn.traceID == "" || (su.CompletionTokens <= 0 && su.ContextTokens <= 0) {
		return
	}
	s.debitSession(ctx, turn.traceID, su)
}

func sessionUsageFromAgent(u agent.Usage) SessionUsage {
	return SessionUsage{
		PromptTokens:             u.PromptTokens,
		CompletionTokens:         u.CompletionTokens,
		ContextTokens:            u.ContextWindowTokens(),
		CacheReadInputTokens:     u.CacheReadInputTokens,
		CacheCreationInputTokens: u.CacheCreationInputTokens,
	}
}

// sessionAdmits reports whether a proxied request for trace may proceed under the
// operator-controlled DRIVE state. Fail-OPEN: with no observeSession wired (the route
// disabled) or an empty trace, and for the advancing states (running/throttled/unknown),
// it returns true and the request path is byte-for-byte the pre-control behavior. It
// returns false only for the operator-set non-advancing states (paused/draining/stopped),
// carrying the state so the refusal can name why.
func (s *Server) sessionAdmits(ctx context.Context, trace string) (bool, SessionState) {
	if s.observeSession == nil || trace == "" {
		return true, SessionState{}
	}
	st := s.observeSession(ctx, trace)
	switch st.Run {
	case "paused", "draining", "stopped":
		return false, st
	default:
		return true, st
	}
}

// writeSessionRefusal emits the 409 a proxied request gets when an operator has held or
// stopped its session. 409 Conflict (not 503): the request is well-formed; the session
// STATE refuses it — the same status the control routes return for a terminal/stale-CAS
// write. The error code is "session_<state>" so a client can branch on it, and the
// operator's reason token (if any) rides the message.
func writeSessionRefusal(w http.ResponseWriter, st SessionState) {
	msg := "session " + st.TraceID + " is " + st.Run + " (operator control); request refused"
	if st.Reason != "" {
		msg += ": " + st.Reason
	}
	body := map[string]any{
		"error": map[string]any{"message": msg, "type": errType(http.StatusConflict), "code": "session_" + st.Run, "param": nil},
	}
	if st.ContinuationID != "" || st.Reason == sessionReasonBudgetContext {
		body["session"] = st
		body["reset"] = SessionResetDirective{
			Action:      "restart_fresh_session",
			FromTraceID: st.TraceID,
			ToTraceID:   st.ContinuationID,
			Reason:      st.Reason,
			Required: []string{
				"dump_session_image",
				"start_fresh_process",
				"rehydrate_planned_view",
				"reuse_provider_cache_when_legal",
			},
			Note: "context budget exhausted; continue under the continuation_id in a fresh model window",
		}
	}
	writeJSON(w, http.StatusConflict, body)
}

// SessionResetDirective is the machine-readable handoff a supervisor gets when the
// long-context budget drains a served session. fak does not kill or relaunch the
// child itself here; it gives the host a deterministic continuation id and the
// required fresh-window actions.
type SessionResetDirective struct {
	Action      string   `json:"action"`
	FromTraceID string   `json:"from_trace_id,omitempty"`
	ToTraceID   string   `json:"to_trace_id,omitempty"`
	Reason      string   `json:"reason,omitempty"`
	Required    []string `json:"required_actions,omitempty"`
	Note        string   `json:"note,omitempty"`
}
