package gateway

// ctxexpense.go — the EXPENSE arm of the managed-context value API (sibling of the
// step_advice arm in ctxvalue.go): a per-session verdict on whether a session has
// become UNUSUALLY EXPENSIVE, graded on the total volume of context that would be
// re-sent on ONE turn (the ESTIMATED as-sent footprint from ctxfootprint.go:
// system + tools + history + tail). Every turn re-ships that whole volume, so a
// session whose per-turn floor has ballooned is paying that cost on every reply —
// the number worth warning on before the window is spent.
//
// This is the "warns/blocks for unusually expensive sessions" surface, wired
// record → view → gate (the shape [[detection-without-enforcement]] prescribes so
// the detector is never built without a gate to act on it):
//
//   - record: assessCtxExpense folds the per-turn volume + thresholds into a closed
//     3-level verdict. Pure, so the policy is unit-testable without a Server.
//   - view (ON by default): the verdict rides the CtxValueReport (so the
//     fak_context_value MCP tool + GET /v1/fak/ctxvalue carry it) AND the per-turn
//     --debug-stats operator line (expense=warn|block …, absent when ok). Both are
//     read-only observability — an operator sees the warn, the agent can query it.
//   - gate (OFF by default, soak-first): behind FAK_CTX_EXPENSE_GATE, a block-tier
//     verdict emits ONE in-band [fak] advisory per session telling the agent to
//     checkpoint and end the turn (ctxExpenseNoteOnce). Never refuses the request
//     — the passthrough body is untouched; the note is the only actuation.
//
// Provenance (Law A2): the verdict is a DECISION taken over an ESTIMATED input
// (the ~4-char/token footprint). It is NEVER conflated with the OBSERVED resident
// token counters in CtxValueReport.Tokens — those say how full the window is; this
// says how expensive every turn's re-send has become.

import (
	"fmt"
	"strings"
)

// ExpenseLevel is the closed expense vocabulary. Every value is simultaneously
// emittable (the report/line carries it), interpretable (the reason names the
// deciding numbers), and — for block — actionable behind the opt-in gate.
type ExpenseLevel string

const (
	// ExpenseOK — the per-turn volume is below the warn line (or no footprint /
	// thresholds to judge on). Nothing to surface; the line/field stays absent.
	ExpenseOK ExpenseLevel = "ok"
	// ExpenseWarn — the per-turn as-sent volume crossed the warn line: the session
	// is getting expensive; keep new residency (large reads/tool results) deliberate.
	ExpenseWarn ExpenseLevel = "warn"
	// ExpenseBlock — the per-turn volume crossed the block line: every reply now
	// re-ships an unusually large floor. Checkpoint durable state and end the turn;
	// behind the soak gate this also emits the in-band advisory.
	ExpenseBlock ExpenseLevel = "block"
)

// Default per-turn as-sent volume thresholds, in ESTIMATED tokens (footprint
// TotalBytes / estBytesPerToken). Chosen against a ~200k-token window so a normal
// session (fresh floor ~41k) never trips warn, while a session re-sending most of
// the window every turn does. A deploy overrides via Config.CtxExpense*Tokens; the
// pure policy treats a 0 threshold as "that tier off".
const (
	ctxExpenseWarnTokensDefault  = 120_000
	ctxExpenseBlockTokensDefault = 180_000
)

// CtxExpense is the DECISION block surfaced beside the OBSERVED tokens in a
// CtxValueReport. TurnTokens is the ESTIMATED total as-sent volume this turn (the
// number the verdict is graded on); WarnTokens/BlockTokens are the lines it was
// graded against (0 = that tier disabled).
type CtxExpense struct {
	Level       ExpenseLevel `json:"level"` // ok | warn | block
	Basis       string       `json:"basis"` // per_turn_volume | none
	Reason      string       `json:"reason"`
	TurnTokens  int          `json:"turn_tokens"` // ESTIMATED total as-sent volume (system+tools+history+tail)
	WarnTokens  int          `json:"warn_tokens"`
	BlockTokens int          `json:"block_tokens"`
	Provenance  string       `json:"provenance"` // DECISION (over an ESTIMATED per-turn volume)
}

// ctxExpenseState is the folded, pure input assessCtxExpense decides on — split out
// so the policy is unit-testable without a Server (the adviseCtxStep pattern). A 0
// threshold means that tier is disabled; a <=0 TurnTokens means no as-sent
// footprint has been observed yet.
type ctxExpenseState struct {
	TurnTokens  int
	WarnTokens  int
	BlockTokens int
}

// assessCtxExpense is the pure expense policy, rung by rung:
//
//  1. no footprint yet          → ok/none (nothing to grade; no phantom warn)
//  2. no thresholds configured  → ok/none (the expense edge is not judgeable here)
//  3. >= block line             → block  (per-turn re-send is unusually large)
//  4. >= warn line              → warn   (the session is getting expensive)
//  5. otherwise                 → ok     (below the warn line)
//
// A tier with a 0 threshold is skipped, so a deploy can run warn-only (block=0) or
// block-only (warn=0) without the other tier ever firing.
func assessCtxExpense(st ctxExpenseState) CtxExpense {
	e := CtxExpense{
		Level:       ExpenseOK,
		Basis:       "none",
		TurnTokens:  maxNonNeg(st.TurnTokens),
		WarnTokens:  maxNonNeg(st.WarnTokens),
		BlockTokens: maxNonNeg(st.BlockTokens),
		Provenance:  "DECISION",
	}
	switch {
	case st.TurnTokens <= 0:
		e.Reason = "no as-sent footprint observed yet; per-turn volume is not measurable"
		return e
	case st.WarnTokens <= 0 && st.BlockTokens <= 0:
		e.Reason = "no expense thresholds configured; the per-turn cost edge is not judged"
		return e
	}
	e.Basis = "per_turn_volume"
	vol := HumanTokenEquiv(float64(st.TurnTokens))
	switch {
	case st.BlockTokens > 0 && st.TurnTokens >= st.BlockTokens:
		e.Level = ExpenseBlock
		e.Reason = fmt.Sprintf("per-turn as-sent volume %s tok >= block line %s; every reply re-ships this floor — checkpoint durable state and end the turn",
			vol, HumanTokenEquiv(float64(st.BlockTokens)))
	case st.WarnTokens > 0 && st.TurnTokens >= st.WarnTokens:
		e.Level = ExpenseWarn
		e.Reason = fmt.Sprintf("per-turn as-sent volume %s tok >= warn line %s; the session is getting expensive — keep new residency deliberate",
			vol, HumanTokenEquiv(float64(st.WarnTokens)))
	default:
		line := st.WarnTokens
		if line <= 0 {
			line = st.BlockTokens
		}
		e.Reason = fmt.Sprintf("per-turn as-sent volume %s tok below the %s expense line", vol, HumanTokenEquiv(float64(line)))
	}
	return e
}

// ctxExpenseThresholdOr maps a Config threshold to the effective server threshold:
// 0 (the struct zero value) takes the built-in default, so expense DETECTION is
// on by default; a positive value overrides; a negative value disables that tier
// (the explicit off, mirroring how --compact-history-budget=0 opts out of a
// default-on lever without collapsing 0 into "default").
func ctxExpenseThresholdOr(cfg, def int) int {
	switch {
	case cfg == 0:
		return def
	case cfg < 0:
		return 0
	default:
		return cfg
	}
}

// ctxExpenseStateFrom folds a session's latest as-sent footprint and the server's
// thresholds into the pure policy input. A nil footprint (no Anthropic-passthrough
// turn yet) yields TurnTokens 0, which assessCtxExpense reads as "not measurable".
func ctxExpenseStateFrom(fp *ctxFootprintBytes, warn, block int) ctxExpenseState {
	turnTokens := 0
	if r := fp.report(); r != nil {
		turnTokens = r.TotalBytes / estBytesPerToken
	}
	return ctxExpenseState{TurnTokens: turnTokens, WarnTokens: warn, BlockTokens: block}
}

// ctxExpenseFor renders one session's expense verdict from its latest footprint +
// the server thresholds. Caller holds ctxValueMu (it reads v.footprint).
func (s *Server) ctxExpenseForLocked(v *sessionCtxValue) CtxExpense {
	return assessCtxExpense(ctxExpenseStateFrom(v.footprint, s.ctxExpenseWarn, s.ctxExpenseBlock))
}

// peekCtxExpense scores a trace's CURRENT expense WITHOUT minting a record (the
// peekResetHealth pattern), so the per-turn debug render can carry the verdict on
// the same line the operator already watches. ok is false when the session has no
// record yet, so the render omits the field rather than printing a phantom.
func (s *Server) peekCtxExpense(trace string) (CtxExpense, bool) {
	if s == nil || strings.TrimSpace(trace) == "" {
		return CtxExpense{}, false
	}
	s.ctxValueMu.Lock()
	defer s.ctxValueMu.Unlock()
	v, ok := s.ctxValue[trace]
	if !ok {
		return CtxExpense{}, false
	}
	return s.ctxExpenseForLocked(v), true
}

// ctxExpenseNoteOnce is the GATE: the opt-in, once-per-session in-band [fak]
// advisory a block-tier verdict emits when FAK_CTX_EXPENSE_GATE is on. It mirrors
// resultAdmissionNoteOnce exactly — a no-op for the empty trace, deduped per
// session so the paragraph fires once (the machine-readable verdict still rides the
// report every turn), and bounded by the same reaper. Returns "" when the gate is
// off, the session is not block-tier, or the note already fired this session; the
// caller prepends a non-empty return as a text block on the Anthropic response.
func (s *Server) ctxExpenseNoteOnce(trace string) string {
	if s == nil || !s.ctxExpenseGate || strings.TrimSpace(trace) == "" {
		return ""
	}
	e, ok := s.peekCtxExpense(trace)
	if !ok || e.Level != ExpenseBlock {
		return ""
	}
	s.ctxExpenseNotedMu.Lock()
	defer s.ctxExpenseNotedMu.Unlock()
	if s.ctxExpenseNoted == nil {
		s.ctxExpenseNoted = map[string]struct{}{}
	}
	if _, seen := s.ctxExpenseNoted[trace]; seen {
		return ""
	}
	if len(s.ctxExpenseNoted) >= maxResetHealthSessions {
		for k := range s.ctxExpenseNoted {
			delete(s.ctxExpenseNoted, k)
			break
		}
	}
	s.ctxExpenseNoted[trace] = struct{}{}
	return "[fak] " + e.Reason + " (context-expense gate)"
}

// formatExpenseField renders the per-turn debug-line expense suffix. It returns ""
// for the ok tier so a healthy session's line stays clean — the field's presence is
// itself the signal, matching the glanceable posture of the nudge=/safety= fields.
func formatExpenseField(e CtxExpense, have bool) string {
	if !have || e.Level == ExpenseOK || e.Basis != "per_turn_volume" {
		return ""
	}
	line := e.WarnTokens
	if e.Level == ExpenseBlock && e.BlockTokens > 0 {
		line = e.BlockTokens
	}
	return fmt.Sprintf(" expense=%s vol=%s/%s", e.Level,
		HumanTokenEquiv(float64(e.TurnTokens)), HumanTokenEquiv(float64(line)))
}
