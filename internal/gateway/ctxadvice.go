package gateway

// ctxadvice.go — the CONSUMER arm of the managed-context value API (#2424). ctxvalue.go
// already computes a closed step_advice verdict (any/bounded/checkpoint/rebuild) every
// served turn, but it was PULL-ONLY at GET /v1/fak/ctxvalue and the fak_context_value MCP
// tool: an agent that never asked never heard it, so the advice was dead weight exactly
// when it mattered — a window about to turn over. This file wires the already-computed
// verdict to two surfaces, in the record → view → gate shape ctxexpense.go prescribes so a
// detector is never built without something to act on it:
//
//   - PUSH (on by default): when the verdict enters checkpoint or rebuild, one in-band
//     [fak] advisory rides the next Anthropic response (ctxAdviceNoteOnce, spliced in
//     messages.go beside the other once-per-state notes). Deduped per STATE ENTRY, not per
//     session: the model hears it once on the way IN to a pressure state and again only if
//     the session leaves and re-enters one. The text is a pure function of the SAME
//     CtxStepAdvice the HTTP/MCP read returns (ctxAdviceNote), so the pushed line and the
//     pulled verdict for one trace can never disagree.
//
//   - THRASH (detection + view on by default, stop opt-in): a session whose window refills
//     to the compaction limit ctxThrashConsecutiveRefills turns RUNNING is not being helped
//     by the lever — the transform sheds, the next turn refills, and the session burns a
//     compaction per reply making no headway. That is the closed session verdict
//     COMPACTION_THRASH, counted always (recordCompactionThrash → the CompactionBailReasons
//     row on /metrics) and, only when the operator arms FAK_COMPACTION_THRASH_STOP, taken
//     at the session gate (compactionThrashRefusal, wired into beginServedSessionTurn).
//
// HONEST SCOPE. The advisory is advice: it never refuses a request, never rewrites the
// body, and the machine-readable verdict still rides every report whether or not the note
// fired. The thrash STOP is inert until armed, so the default serve path is byte-for-byte
// historical; the thrash COUNT is not gated, because a signal an operator cannot see is
// the thing #2424 exists to fix.

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/lifecycle"
	"github.com/anthony-chaudhary/fak/internal/numfmt"
)

// ReasonCompactionThrash is the closed session verdict for a window that refills to the
// compaction limit turn after turn. Declared in dos.toml as [reasons.COMPACTION_THRASH];
// the binding is asserted by ctxadvice_test.go so dos_check_reason resolves it.
const ReasonCompactionThrash = "COMPACTION_THRASH"

// ctxThrashConsecutiveRefills is how many BACK-TO-BACK context-event turns constitute
// thrash. Three, not two: a pair of consecutive fires is an ordinary burst (a big tool
// result lands, the lever sheds, the next turn is still over the line). By the third the
// session has spent three replies' worth of compaction and the window is still at the
// limit — the lever is not the binding constraint any more, the traffic is.
const ctxThrashConsecutiveRefills = 3

// envCompactionThrashStop arms the thrash STOP at the session gate. Unset/false = detection
// and telemetry only (the default), so a deploy sees the verdict before anything acts on it
// — the soak-first posture FAK_CTX_EXPENSE_GATE takes for the expense advisory.
const envCompactionThrashStop = "FAK_COMPACTION_THRASH_STOP"

// ctxAdviceNote renders the model-facing advisory for one step verdict, or "" when the
// verdict is not one the model needs pushed (any/bounded/unknown are readable on request
// and say nothing urgent). Pure and deterministic — same verdict, byte-identical string —
// so the pushed line is bindable to the pulled /v1/fak/ctxvalue report for the same trace,
// which is the whole point: one verdict, two deliveries, never two stories.
func ctxAdviceNote(a CtxStepAdvice) string {
	if a.StepClass != StepClassCheckpoint && a.StepClass != StepClassRebuild {
		return ""
	}
	return fmt.Sprintf("[fak] context step advice: step_class=%s basis=%s — %s. %s",
		a.StepClass, a.Basis, a.Reason, a.Affordance)
}

// ctxAdviceNoteOnce is the PUSH gate: the in-band advisory this trace's CURRENT step
// verdict owes the model, or "" when there is nothing new to say. Deduped per STATE ENTRY
// via sessionCtxValue.noticedClass — every call records the class it saw, and a note is
// returned only when the class CHANGED into checkpoint or rebuild. So a session that sits
// at checkpoint for twelve turns is told once; one that compacts, recovers, and crowds
// again is told once per entry. A nil server, an empty trace, or a trace with no served
// turn is a no-op (no phantom advisory for a session the gateway has not observed).
func (s *Server) ctxAdviceNoteOnce(trace string) string {
	trace = strings.TrimSpace(trace)
	if s == nil || trace == "" {
		return ""
	}
	s.ctxValueMu.Lock()
	defer s.ctxValueMu.Unlock()
	v, ok := s.ctxValue[trace]
	if !ok || v.turns == 0 {
		return ""
	}
	advice := adviseCtxStep(s.ctxValueStateLocked(v))
	if advice.StepClass == v.noticedClass {
		return "" // same state as the last look — already said
	}
	v.noticedClass = advice.StepClass
	return ctxAdviceNote(advice)
}

// compactionThrashStreak reports how many consecutive context-event turns this trace has
// taken — the run length the COMPACTION_THRASH verdict is graded on. 0 for an unknown or
// empty trace, so an unobserved session can never read as thrashing.
func (s *Server) compactionThrashStreak(trace string) int {
	trace = strings.TrimSpace(trace)
	if s == nil || trace == "" {
		return 0
	}
	s.ctxValueMu.Lock()
	defer s.ctxValueMu.Unlock()
	if v, ok := s.ctxValue[trace]; ok {
		return v.consecutiveEvents
	}
	return 0
}

// compactionThrashRefusal reports the session state (if any) that must refuse this trace's
// next served turn because its window keeps refilling to the compaction limit. nil ⇒ admit.
// Inert unless the operator armed FAK_COMPACTION_THRASH_STOP, so the default serve path is
// byte-for-byte historical — the verdict is still COUNTED either way (observeCtxValue), so
// an operator can watch the row before deciding to arm the stop.
func (s *Server) compactionThrashRefusal(trace string) *SessionState {
	if !envEnabled(envCompactionThrashStop) {
		return nil
	}
	if s.compactionThrashStreak(trace) < ctxThrashConsecutiveRefills {
		return nil
	}
	return &SessionState{TraceID: trace, Run: lifecycle.TokenStopped, Reason: ReasonCompactionThrash}
}

// compactionThrashRefusalBody renders the 409 a thrash-stopped session's next request gets.
// It is NOT the operator-control refusal writeSessionRefusal renders by default: nobody
// paused this session, its own window did — so the message names the measured run and the
// only continuation that helps (a fresh window seeded from durable state) instead of
// telling a supervisor to look for an operator who never acted.
func compactionThrashRefusalBody(st SessionState) map[string]any {
	return map[string]any{
		"error": map[string]any{
			"message": "session " + st.TraceID + " stopped: the context window refilled to the compaction limit " +
				numfmt.ItoaInt(ctxThrashConsecutiveRefills) + " consecutive turns (compaction thrash) — compaction is no longer buying headroom; " +
				"restart from durable state in a fresh window",
			"type":  errType(http.StatusConflict),
			"code":  "session_compaction_thrash",
			"param": nil,
		},
		"reason": st.Reason,
	}
}
