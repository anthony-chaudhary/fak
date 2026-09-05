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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/harnessversion"
	"github.com/anthony-chaudhary/fak/internal/lifecycle"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
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

// beginServedRequest establishes the request context and trace before admitting
// one served turn. Every model-facing HTTP wire crosses this same boundary.
func (s *Server) beginServedRequest(w http.ResponseWriter, r *http.Request) (context.Context, string, servedSessionTurn, bool, bool) {
	ctx := r.Context()
	trace := s.useHTTPTrace(w, r, "")
	if s != nil {
		router := s.HarnessRouter()
		if router != nil && r != nil {
			sessionID := strings.TrimSpace(r.Header.Get("X-Fak-Session-Id"))
			if sessionID == "" {
				sessionID = strings.TrimSpace(r.Header.Get("X-Trace-Id"))
			}
			if sessionID == "" {
				sessionID = trace
			}
			var pathParam string
			if r.URL != nil {
				pathParam = r.URL.Path
			}
			selectedVersion, _ := router.Route(sessionID, r.Header.Get(harnessversion.HeaderHarnessVersion), pathParam)
			if selectedVersion != "" && w != nil {
				w.Header().Set(harnessversion.HeaderHarnessVersion, selectedVersion)
			}
		}
	}
	turn, ok, canceled := s.beginServedSessionTurn(ctx, trace)
	return ctx, trace, turn, ok, canceled
}

// admitServedRequest applies the reset-capable admission path shared by the
// OpenAI and Anthropic request wires. It owns refusal rendering so each wire
// cannot drift in cancellation, budget-reset, or coherence-reset behavior.
func (s *Server) admitServedRequest(w http.ResponseWriter, r *http.Request, messages []agent.Message) (context.Context, string, []agent.Message, servedSessionTurn, bool) {
	ctx, trace, turn, ok, canceled := s.beginServedRequest(w, r)
	if canceled {
		return ctx, trace, messages, turn, false
	}
	if !ok {
		newTrace, resetMessages, resetTurn, resetOK, resetCanceled, reset := s.applyBudgetReset(ctx, turn.state, messages)
		if !reset {
			writeSessionRefusal(w, turn.state)
			return ctx, trace, messages, turn, false
		}
		trace, messages, turn, ok, canceled = newTrace, resetMessages, resetTurn, resetOK, resetCanceled
		if canceled || !ok {
			if !canceled {
				writeSessionRefusal(w, turn.state)
			}
			return ctx, trace, messages, turn, false
		}
	}

	trace, messages, turn, ok, canceled = s.resetOnCoherenceIfArmed(ctx, trace, messages, turn)
	if canceled || !ok {
		if !canceled {
			writeSessionRefusal(w, turn.state)
		}
		return ctx, trace, messages, turn, false
	}

	if c := s.ResearchArmCoordinator(); c != nil {
		path := ""
		if r != nil && r.URL != nil {
			path = r.URL.Path
		}
		armLease, err := c.Admit(ctx, r, path, trace)
		if err != nil {
			if w != nil {
				w.Header().Set("Retry-After", "1")
				writeErr(w, http.StatusTooManyRequests, err.Error())
			}
			return ctx, trace, messages, turn, false
		}
		if armLease != nil {
			ctx = withArmLease(ctx, armLease)
			go func() {
				<-ctx.Done()
				armLease.Done(0, ctx.Err())
			}()
		}
	}

	return ctx, trace, messages, turn, true
}

// beginServedSessionTurn applies the live session gate to one proxied model request.
// When DecideSession is wired, this is the mutating boundary: session.Table.Decide
// debits TurnsLeft, resolves pause/drain/stop/budget exhaustion, and returns the pace
// caps for THIS request. When only ObserveSession is wired, it falls back to the
// shipped run-state admission guard. With neither hook, it is fail-open and leaves the
// historical request path unchanged.
func (s *Server) beginServedSessionTurn(ctx context.Context, trace string) (servedSessionTurn, bool, bool) {
	turn := servedSessionTurn{traceID: trace}
	appendSessionLedger(trace, "turn_begin", nil)
	if trace == "" {
		return turn, true, false
	}
	// Deployment SESSION CEILING (#3425, epic #3256): when the operator configured a
	// maximum concurrent-governed-session count (FAK_MAX_SESSIONS) and the box is at/over
	// it, backpressure a NEW session's turn here — before the model is consulted — with
	// the closed reason SESSION_CEILING_SATURATED. A trace already resident in the session
	// registry is never refused (sessionCeilingRefusal), so an in-flight loop is never
	// sacrificed to admit a new one. Inert (nil) when no ceiling is configured, so the
	// default serve path is byte-for-byte historical. See session_ceiling.go.
	if ref := s.sessionCeilingRefusal(ctx, trace); ref != nil {
		turn.state = *ref
		return turn, false, false
	}
	// COMPACTION THRASH (#2424): a session whose window refilled to the compaction limit on
	// ctxThrashConsecutiveRefills consecutive turns is spending a compaction per reply and
	// making no headway — the lever is firing and the traffic is undoing it. The closed
	// verdict COMPACTION_THRASH is taken here, at the same pre-model boundary the ceiling
	// uses, so the stop lands cleanly instead of after another full window is re-shipped.
	// Inert (nil) unless the operator armed FAK_COMPACTION_THRASH_STOP; the verdict is
	// COUNTED either way, so a deploy watches the row before it arms the stop. See ctxadvice.go.
	if ref := s.compactionThrashRefusal(trace); ref != nil {
		turn.state = *ref
		return turn, false, false
	}
	// Control-plane SPEND CAP (#3273): if a scope this trace belongs to (session / agent
	// / team / tenant) has crossed its versioned budget, the kernel refuses the turn here
	// — before the model is consulted — carrying the closed reason and the pause/kill run
	// state. BUDGET_SPEND_EXCEEDED is NOT a reset reason (isBudgetResetReason), so the
	// caller falls through to writeSessionRefusal (a hard 409 stop), never the human-like
	// continue the context/token drains take. No-op when no governor is attached.
	if brk := s.spendBreach(trace); brk != nil {
		turn.state = SessionState{TraceID: trace, Run: spendBreachRunToken(brk.Action), Reason: brk.Reason}
		return turn, false, false
	}
	var v SessionVerdict
	hasDecide := false
	if s.decideSession != nil {
		v = s.decideSession(ctx, trace)
		hasDecide = true
	} else if s.table != nil {
		v = toGatewaySessionVerdict(s.table.Decide(trace))
		hasDecide = true
	}
	if hasDecide {
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

// applyBudgetReset lowers the host reset transaction onto the physical gateway
// boundary: splice the exact seed, then admit the fresh child. The shared Next row
// is applied only after both effects succeed; every terminal no-render outcome is
// independently readable as a refusal.
func (s *Server) applyBudgetReset(ctx context.Context, st SessionState, messages []agent.Message) (string, []agent.Message, servedSessionTurn, bool, bool, bool) {
	newTrace, seed, reset := s.maybeResetOnBudget(ctx, st, messages)
	if !reset {
		return "", messages, servedSessionTurn{}, false, false, false
	}
	messages = spliceSeed(seed, messages)
	turn, ok, canceled := s.beginServedSessionTurn(ctx, newTrace)
	recordResetSeedNext(newTrace, seed, turn, ok, canceled)
	return newTrace, messages, turn, ok, canceled, true
}

// armCoherenceReset latches a hard reset for trace on its NEXT admitted turn — the actuation the
// #3159 non-holding-rewrite escalation arms. Bounded generationally (like resetHealth) so a
// long-running gateway minting a trace per session cannot grow the latch set without limit. Empty
// trace is a no-op.
func (s *Server) armCoherenceReset(trace string) {
	if s == nil || trace == "" {
		return
	}
	s.resetHealthMu.Lock()
	if s.coherenceResetArmed == nil {
		s.coherenceResetArmed = make(map[string]bool)
	}
	if len(s.coherenceResetArmed) >= maxResetHealthSessions {
		s.coherenceResetArmed = make(map[string]bool) // generational reset, like resetHealth
	}
	s.coherenceResetArmed[trace] = true
	s.resetHealthMu.Unlock()
}

// consumeCoherenceReset reports whether trace was armed for a coherence reset and clears the latch
// in the same critical section, so a given escalation actuates at most once. Empty/unarmed → false.
func (s *Server) consumeCoherenceReset(trace string) bool {
	if s == nil || trace == "" {
		return false
	}
	s.resetHealthMu.Lock()
	armed := s.coherenceResetArmed[trace]
	if armed {
		delete(s.coherenceResetArmed, trace)
	}
	s.resetHealthMu.Unlock()
	return armed
}

// maybeResetOnCoherence is the #3159 actuation twin of maybeResetOnBudget, but on the ADMITTED
// path: when a prior turn's non-holding-rewrite escalation armed a hard reset for trace AND the
// host wired the opt-in ResetOnBudget callback, it consumes the latch, asks the host to distill a
// carryover seed + re-arm a fresh session, zeroes the reset-health cooldown, and counts the
// actuated reset. ok=false means "nothing armed, or no resetter wired, or the host declined" — the
// caller proceeds unchanged. Consuming the latch BEFORE invoking the host (and the fresh trace
// getting a fresh per-trace coordinator) prevents a reset loop: the new trace starts with a zero
// non-holding streak, so it cannot immediately re-escalate.
func (s *Server) maybeResetOnCoherence(ctx context.Context, trace string, messages []agent.Message) (newTrace string, seed []agent.Message, ok bool) {
	if s.resetOnBudget == nil || !s.consumeCoherenceReset(trace) {
		return "", nil, false
	}
	newTrace, seed, ok = s.resetOnBudget(ctx, trace, messages)
	if ok {
		s.resetHealthReset(newTrace)
		s.metrics.recordCoherenceResetFired()
	}
	return newTrace, seed, ok
}

// resetOnCoherenceIfArmed runs the coherence reset on the admitted path and re-admits on the fresh
// trace, mirroring the budget-reset block. When nothing is armed it returns its inputs unchanged
// with ok=true, canceled=false (a cheap no-op). Callers reassign trace/messages/sessionTurn/ok/
// canceled from the returns: canceled=true ⇒ return immediately; ok=false ⇒ the fresh trace refused
// (write the refusal, mirroring the budget path's never-loop guard).
func (s *Server) resetOnCoherenceIfArmed(ctx context.Context, trace string, messages []agent.Message, sessionTurn servedSessionTurn) (string, []agent.Message, servedSessionTurn, bool, bool) {
	newTrace, seed, reset := s.maybeResetOnCoherence(ctx, trace, messages)
	if !reset {
		return trace, messages, sessionTurn, true, false
	}
	messages = spliceSeed(seed, messages)
	st, ok, canceled := s.beginServedSessionTurn(ctx, newTrace)
	recordResetSeedNext(newTrace, seed, st, ok, canceled)
	return newTrace, messages, st, ok, canceled
}

func recordResetSeedNext(trace string, seed []agent.Message, turn servedSessionTurn, ok, canceled bool) {
	payload := ""
	if len(seed) > 0 {
		payload = seed[0].Content
	}
	result := sessionctl.ApplyResult{Applied: ok && !canceled}
	if canceled {
		result.Refusal = "fresh child admission canceled"
	} else if !ok {
		result.Refusal = "fresh child admission refused: " + turn.state.Reason
	}
	sessionctl.RecordBudgetResetNextResult(trace, payload, result)
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
// turnDur is the turn's real wall-clock duration (0 = unknown); it feeds the
// throughput axis's sustained-rate observation (#2762), so a turn that produced
// no tokens but took real time (a stall) is still reported.
func (s *Server) debitServedSessionTurn(ctx context.Context, turn servedSessionTurn, usage agent.Usage, turnDur time.Duration, messages []agent.Message) {
	su := sessionUsageFromAgent(usage)
	if turnDur > 0 {
		su.DurationNanos = int64(turnDur)
	}
	// Fold this turn's provider usage into the control-plane spend cap (#3273) FIRST and
	// independently of the session-control table: the spend governor works on any served
	// path (serve/guard-as-client), whether or not the operator wired the DRIVE-state
	// DebitSession hook. A breach fires its counter + webhook here; the NEXT turn's
	// beginServedSessionTurn refuses on the accumulated total. No-op when unattached.
	s.chargeSpend(turn.traceID, usage)
	if armLease := armLeaseFromContext(ctx); armLease != nil {
		armLease.Done(su.CompletionTokens+su.ContextTokens, nil)
	}
	if (s.debitSession == nil && s.table == nil) || turn.traceID == "" || (su.CompletionTokens <= 0 && su.ContextTokens <= 0 && su.DurationNanos <= 0) {
		return
	}
	var st SessionState
	if s.debitSession != nil {
		st = s.debitSession(ctx, turn.traceID, su)
	} else if s.table != nil {
		sessUsage := session.Usage{
			OutputTokens:  su.CompletionTokens,
			ContextTokens: su.ContextTokens,
			DurationNanos: su.DurationNanos,
		}
		st = toGatewaySessionState(s.table.DebitUsage(turn.traceID, sessUsage))
	}
	if s.budgetDrained != nil && isBudgetResetReason(st) {
		s.budgetDrained(ctx, st, append([]agent.Message(nil), messages...))
	}
}

// accountStreamedTurn records the post-turn accounting the streaming paths share
// but the buffered path folds into s.complete: inference metrics, the planner
// request-memory observation, and the session-usage debit. began is the turn start
// (for the inference duration).
//
// reqModel is the model id the CLIENT asked for — the same input the planner routed
// on, which is why the attribution it produces cannot disagree with the side that
// actually served the turn. Deliberately not the model the response reported: a
// local decode may not echo one back, and an empty id would then be read as vendor.
func (s *Server) accountStreamedTurn(ctx context.Context, turn servedSessionTurn, comp *agent.Completion, messages []agent.Message, began time.Time, reqModel string) {
	s.metrics.observeInferenceUsageServed(s.servedLocality(reqModel), comp.Usage, comp.FinishReason, time.Since(began))
	s.observePlannerRequestMemory()
	s.debitServedSessionTurn(ctx, turn, comp.Usage, time.Since(began), messages)
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
	// The non-advancing states are the shared lifecycle phases other than Running:
	// Paused/Draining/Stopped. The tokens are SOURCED from internal/lifecycle (not
	// re-spelled) so a vocabulary rename can never silently drift this admission gate
	// away from the served session's own RunState.String() (#912). Throttled is a
	// session-only pace extra and stays advancing; armed/disabled/unknown fail open.
	switch st.Run {
	case lifecycle.TokenPaused, lifecycle.TokenDraining, lifecycle.TokenStopped:
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
	// Deployment session-ceiling backpressure (#3425) is NOT operator DRIVE control: the
	// request is well-formed and the session is not held — the DEPLOYMENT is at capacity.
	// Emit 503 (Service Unavailable) with a Retry-After hint and the closed reason token,
	// so a client/autoscaler backs off and retries rather than reading a 409 as a terminal
	// operator stop. In-flight sessions are unaffected; only this new session is shed.
	if st.Reason == ReasonSessionCeilingSaturated {
		w.Header().Set("Retry-After", "5")
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"error": map[string]any{
				"message": "deployment at its configured concurrent-session ceiling (FAK_MAX_SESSIONS); new session refused — retry when headroom frees. Sessions already in flight are unaffected.",
				"type":    errType(http.StatusServiceUnavailable),
				"code":    "session_ceiling_saturated",
				"param":   nil,
			},
			"reason": st.Reason,
		})
		return
	}
	// COMPACTION_THRASH (#2424) is not operator DRIVE control either: nobody paused this
	// session — its own window refilled to the compaction limit turn after turn. Still 409
	// (terminal, the same class as an operator stop), but with its own code and a message
	// naming the measured run and the only continuation that helps, so a supervisor does not
	// go looking for an operator who never acted. See ctxadvice.go.
	if st.Reason == ReasonCompactionThrash {
		writeJSON(w, http.StatusConflict, compactionThrashRefusalBody(st))
		return
	}
	msg := "session " + st.TraceID + " is " + st.Run + " (operator control); request refused"
	recovery := SessionLifecycleRecovery{State: st.Run, SessionID: st.TraceID}
	switch st.Run {
	case "paused":
		msg = "session " + st.TraceID + " is held, not killed; session continuity is preserved until an operator resumes it"
		recovery.Terminal = false
		recovery.Retryable = false
		recovery.NextAction = "resume"
	case "draining":
		recovery.Terminal = false
		recovery.Retryable = false
		recovery.NextAction = "wait_for_drain"
	case "stopped":
		recovery.Terminal = true
		recovery.Retryable = false
		recovery.NextAction = "start_new_session"
	default:
		recovery.Terminal = true
		recovery.Retryable = false
	}
	if st.Reason != "" {
		msg += ": " + st.Reason
	}
	body := map[string]any{
		"error":    map[string]any{"message": msg, "type": errType(http.StatusConflict), "code": "session_" + st.Run, "param": nil},
		"recovery": recovery,
	}
	if st.ContinuationID != "" || st.Reason == sessionReasonBudgetContext {
		body["session"] = st
		body["reset"] = SessionResetDirective{
			Action:        "restart_fresh_session",
			FromTraceID:   st.TraceID,
			ToTraceID:     st.ContinuationID,
			Reason:        st.Reason,
			CacheAffinity: st.CacheAffinity,
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

// SessionLifecycleRecovery is the stable recovery contract attached to ordinary
// operator-controlled lifecycle refusals. NextAction is a typed token, never a
// shell command; clients target SessionID through the control API.
type SessionLifecycleRecovery struct {
	State      string `json:"state"`
	Terminal   bool   `json:"terminal"`
	Retryable  bool   `json:"retryable"`
	SessionID  string `json:"session_id"`
	NextAction string `json:"next_action,omitempty"`
}

// SessionResetDirective is the machine-readable handoff a supervisor gets when the
// long-context budget drains a served session. fak does not kill or relaunch the
// child itself here; it gives the host a deterministic continuation id and the
// required fresh-window actions.
type SessionResetDirective struct {
	Action        string               `json:"action"`
	FromTraceID   string               `json:"from_trace_id,omitempty"`
	ToTraceID     string               `json:"to_trace_id,omitempty"`
	Reason        string               `json:"reason,omitempty"`
	CacheAffinity SessionCacheAffinity `json:"cache_affinity,omitempty,omitzero"`
	Required      []string             `json:"required_actions,omitempty"`
	Note          string               `json:"note,omitempty"`
}

func appendSessionLedger(trace, kind string, content []byte) {
	if trace == "" {
		return
	}
	l, err := sessionledger.OpenDefault()
	if err != nil {
		return
	}
	if len(content) == 0 {
		content = []byte("{}")
	}
	_, _ = l.Append(trace, kind, content)
}

// turnLedgerSummary is what the ledger records for a served turn: the SHAPE of the
// request, never the request. Persisting req.Raw here meant a full conversation
// context (~200 KB and rising) landed in the ledger every turn; the ledger is a
// provenance chain, and a byte count plus a digest witnesses the same turn for a
// fixed ~200 bytes. sessionledger.Elide would clamp an oversized blob anyway --
// summarizing at the source keeps what we DO store useful instead of a bare hash.
func turnLedgerSummary(req *agent.AnthropicMessagesRequest) []byte {
	if req == nil {
		return nil
	}
	sum := sha256.Sum256(req.Raw)
	b, err := json.Marshal(struct {
		Model    string `json:"model,omitempty"`
		Messages int    `json:"messages"`
		Tools    int    `json:"tools,omitempty"`
		Stream   bool   `json:"stream,omitempty"`
		RawBytes int    `json:"raw_bytes"`
		RawSHA   string `json:"raw_sha256"`
	}{
		Model:    req.Model,
		Messages: len(req.Messages),
		Tools:    len(req.Tools),
		Stream:   req.Stream,
		RawBytes: len(req.Raw),
		RawSHA:   hex.EncodeToString(sum[:]),
	})
	if err != nil {
		return nil
	}
	return b
}

func (t servedSessionTurn) complete() { appendSessionLedger(t.traceID, "turn_complete", nil) }

// SetSpendGovernor wires the control-plane spend cap (#3273) onto the Server. scopeOf maps
// a request trace to its scope hierarchy (tenant/team/agent/session); a nil resolver
// defaults to session-only (Session=trace), the shape a single-tenant serve needs. A nil
// governor detaches it (the request path goes byte-for-byte historical). Guarded by
// admissionMu alongside SetTokenRateGate. A nil receiver is a no-op.
func (s *Server) SetSpendGovernor(g *SpendGovernor, scopeOf func(trace string) ScopeKey) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	s.spendGovernor = g
	s.spendScopeOf = scopeOf
	s.admissionMu.Unlock()
}

// spendScopeFor resolves the governor and the scope key for a trace under the configured
// resolver. Returns a nil governor when none is attached or the trace is empty (the
// fail-open, historical path).
func (s *Server) spendScopeFor(trace string) (*SpendGovernor, ScopeKey) {
	s.admissionMu.RLock()
	g := s.spendGovernor
	resolve := s.spendScopeOf
	s.admissionMu.RUnlock()
	if g == nil || trace == "" {
		return nil, ScopeKey{}
	}
	if resolve != nil {
		return g, resolve(trace)
	}
	return g, ScopeKey{Session: trace}
}

// spendBreach reports the spend-cap breach (if any) that must refuse this trace's next
// served turn — a pure read of the accumulated per-scope totals. nil ⇒ admit.
func (s *Server) spendBreach(trace string) *SpendBreach {
	g, key := s.spendScopeFor(trace)
	if g == nil {
		return nil
	}
	return g.Evaluate(key)
}

// chargeSpend folds one served turn's provider usage into the spend cap for every scope
// the trace belongs to. No-op when no governor is attached.
func (s *Server) chargeSpend(trace string, usage agent.Usage) {
	g, key := s.spendScopeFor(trace)
	if g == nil {
		return
	}
	g.Charge(key, spendCostFromUsage(usage))
}

// spendCostFromUsage folds a completion's provider-reported usage into the spend
// increment, using the provider-neutral split (UncachedPromptTokens + CachedPromptTokens
// == the full resident prompt on every provider) so a cache-heavy turn is not double- or
// under-counted, plus the completion and cache-write axes. The dollar axis is left 0 (the
// token cap is the axis this increment charges; per-model USD pricing is a follow-on).
func spendCostFromUsage(u agent.Usage) SpendCost {
	return SpendCost{
		InputTokens:      int64(u.UncachedPromptTokens()),
		OutputTokens:     int64(u.CompletionTokens),
		CacheReadTokens:  int64(u.CachedPromptTokens()),
		CacheWriteTokens: int64(u.CacheCreationInputTokens),
	}
}

// spendBreachRunToken maps a breach action onto the session DRIVE run-state token the
// refusal carries: kill is terminal (stopped), pause drains and is operator-resumable
// (paused). Sourced from internal/lifecycle so the vocabulary can never drift from the
// session-admit gate that reads the same tokens.
func spendBreachRunToken(a SpendAction) string {
	if a == SpendActionKill {
		return lifecycle.TokenStopped
	}
	return lifecycle.TokenPaused
}

// toGatewaySessionState projects internal/session.State into gateway SessionState.
func toGatewaySessionState(s session.State) SessionState {
	return toGatewaySessionStateAt(s, time.Now())
}

func toGatewaySessionStateAt(s session.State, now time.Time) SessionState {
	return SessionState{
		TraceID:    s.TraceID,
		Run:        s.Run.String(),
		TokensUsed: s.Cost.TotalTokens(),
		TokenUsage: s.Cost.TotalTokens(),
		Budget: SessionBudget{
			TurnsLeft:             s.Budget.TurnsLeft,
			TokensLeft:            s.Budget.TokensLeft,
			ContextTokensLeft:     s.Budget.ContextTokensLeft,
			ContextTokensCap:      s.Budget.ContextTokensCap,
			ResidentContextTokens: s.Cost.LatestContextTokens(),
			SpendMicroCentsLeft:   s.Budget.SpendMicroCentsLeft,
			SpendMicroCentsCap:    s.Budget.SpendMicroCentsCap,
		},
		Priority:       s.Priority,
		Pace:           SessionPace{MaxTokensPerTurn: s.Pace.MaxTokensPerTurn, MinTurnGapMs: s.Pace.MinTurnGapMs},
		Reason:         s.Reason,
		ContinuationID: s.ContinuationID,
		ParentTrace:    s.ParentTrace,
		Generation:     s.Generation,
		CacheAffinity: SessionCacheAffinity{
			Action:      s.CacheAffinity.Action,
			AffinityKey: s.CacheAffinity.AffinityKey,
			FromTraceID: s.CacheAffinity.FromTraceID,
			ToTraceID:   s.CacheAffinity.ToTraceID,
			Reason:      s.CacheAffinity.Reason,
		},
		ResetTransaction: toGatewayResetTransaction(s.ResetTransaction),
		ProviderBoundary: SessionProviderBoundary{
			Schema:            s.ProviderBoundary.Schema,
			Provider:          s.ProviderBoundary.Provider,
			Source:            s.ProviderBoundary.Source,
			PreviousTrace:     s.ProviderBoundary.PreviousTrace,
			ProviderSessionID: s.ProviderBoundary.ProviderSessionID,
		},
		Assumptions: toGatewaySessionAssumptions(s.Assumptions),
		Time:        toGatewaySessionTime(s.Time, now),
		Throughput: SessionThroughput{
			ExpectedTokensPerSec: s.Throughput.ExpectedTokensPerSec,
			MinTokensPerSec:      s.Throughput.MinTokensPerSec,
			ObservedTokensPerSec: s.Throughput.ObservedTokensPerSec(),
		},
		Rev: s.Rev,
	}
}

func toGatewaySessionTime(tb session.TimeBudget, now time.Time) SessionTime {
	q := tb.Query(now)
	elapsed := tb.Elapsed(now)
	if !q.Bounded && elapsed <= 0 {
		return SessionTime{}
	}
	return SessionTime{
		Bounded:          q.Bounded,
		Exceeded:         q.Exceeded,
		ElapsedSeconds:   int64(elapsed / time.Second),
		RemainingSeconds: int64(q.Remaining / time.Second),
		LimitSeconds:     int64(q.Limit / time.Second),
	}
}

func toGatewaySessionAssumptions(in []session.Assumption) []SessionAssumption {
	if len(in) == 0 {
		return nil
	}
	out := make([]SessionAssumption, 0, len(in))
	for _, a := range in {
		out = append(out, SessionAssumption{
			Key:        a.Key,
			Statement:  a.Statement,
			Source:     a.Source,
			Confidence: a.Confidence,
			Expiry:     a.Expiry,
			SourceRef:  a.SourceRef,
		})
	}
	return out
}

func toGatewayResetTransaction(tx session.ResetTransaction) SessionResetTransaction {
	out := SessionResetTransaction{
		Schema:       tx.Schema,
		OldTrace:     tx.OldTrace,
		NewTrace:     tx.NewTrace,
		SeedDigest:   tx.SeedDigest,
		Contributors: append([]string(nil), tx.Contributors...),
		BudgetRearm: SessionResetBudgetRearm{
			TurnsLeft:         tx.BudgetRearm.TurnsLeft,
			TokensLeft:        tx.BudgetRearm.TokensLeft,
			ContextTokensLeft: tx.BudgetRearm.ContextTokensLeft,
			ContextTokensCap:  tx.BudgetRearm.ContextTokensCap,
		},
		WarmPrefixDigest: tx.WarmPrefixDigest,
	}
	if len(tx.OmittedSpans) > 0 {
		out.OmittedSpans = make([]SessionResetOmittedSpan, 0, len(tx.OmittedSpans))
		for _, span := range tx.OmittedSpans {
			out.OmittedSpans = append(out.OmittedSpans, SessionResetOmittedSpan{
				Index:  span.Index,
				Role:   span.Role,
				Digest: span.Digest,
				Reason: span.Reason,
			})
		}
	}
	return out
}

func toGatewaySessionVerdict(v session.Verdict) SessionVerdict {
	return SessionVerdict{
		Proceed:   v.Proceed,
		MaxTokens: v.MaxTokens,
		MinGapMs:  v.MinGapMs,
		State:     toGatewaySessionState(v.State),
		Stop:      v.Stop,
		Reason:    v.Reason,
	}
}
