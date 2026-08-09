package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/negframe"
	"github.com/anthony-chaudhary/fak/internal/sessionsteer"
)

// ctxvalue.go — the managed-context arm of the guard's value API: one rolling,
// content-free record per session trace of how a LONG session consumes its context
// window, folded into a multi-LEVEL report the agent INSIDE the session can query
// (GET /v1/fak/ctxvalue, or the fak_context_value MCP tool) to size its next step
// against the window it has left.
//
// Three levels, three provenances (Law A2 — every value carries its owner):
//
//   - tokens  (OBSERVED): the provider-relayed resident-prompt split per turn — how
//     full the window is NOW, and how fast it grows per reply. Same normalized axes
//     as the vcache families view (UncachedPromptTokens + CachedPromptTokens +
//     cache_creation), so a codex session reads identically to a Claude one.
//   - turns   (WITNESSED): counts fak takes itself — every served turn, and every
//     CONTEXT EVENT (a compaction / planned-view rewrite fak's own outbound
//     transform fired; the gateway records its own action, never a guess about the
//     harness).
//   - session (WITNESSED): the whole arc — cumulative prefill volume, output
//     tokens, and a closed lifecycle phase.
//
// The derived turns-to-next-event estimate is FORECAST, and step_advice is a
// DECISION from a closed vocabulary. Advice-only: nothing here acts on the answer,
// nothing feeds the request path. The token thresholds stay in lockstep with the
// per-turn human nudge (compactionNudgeNearPercent in debug_stats.go) so the agent
// query and the `fak info` pane can never disagree about when to checkpoint.

// maxCtxValueSessions bounds the per-session table exactly like resetHealth /
// sessionPlanners: on overflow the table takes a generational reset, so a gateway
// minting a fresh trace per session cannot grow it without limit.
const maxCtxValueSessions = 8192

// ctxValueWindow is how many recent turns the resident-growth slope averages over.
// It matches resetHealthWindow so the two per-session rolling views move on the
// same horizon.
const ctxValueWindow = 8

// StepClass is the closed step-advice vocabulary. Every value is simultaneously
// emittable (the report carries it), interpretable (the reason names the deciding
// numbers), and inert (advice-only; the gateway never enforces it).
type StepClass string

const (
	// StepClassAny — wide headroom: a large, multi-file step fits in the window.
	StepClassAny StepClass = "any"
	// StepClassBounded — moderate pressure: take a single-concern step and keep new
	// residency (large reads, long tool results) deliberate.
	StepClassBounded StepClass = "bounded"
	// StepClassCheckpoint — the window is nearly spent: land in-flight state (commit,
	// write the ledger/plan down) BEFORE the next context event rewrites the window.
	StepClassCheckpoint StepClass = "checkpoint"
	// StepClassRebuild — a context event just fired: the cheapest first step is
	// re-anchoring from durable state (plan, ledger, index), not starting a wide step
	// against a just-rewritten window.
	StepClassRebuild StepClass = "rebuild"
	// StepClassUnknown — no evidence to decide on (no turns, or no budget and no
	// event ever observed). Fail-closed: the report says why instead of guessing.
	StepClassUnknown StepClass = "unknown"
)

// Step-advice thresholds. The checkpoint percent is the SAME constant the per-turn
// debug nudge uses (nudge=checkpoint-soon), so the two surfaces agree by
// construction. The turn floors tighten the class when the FORECAST says the next
// context event is closer than the percent alone implies (a fast-growing window
// can cross 80% in two replies).
const (
	ctxStepCheckpointPercent = compactionNudgeNearPercent // 80
	ctxStepBoundedPercent    = 50
	ctxStepCheckpointTurns   = 4.0
	ctxStepBoundedTurns      = 12.0
	// ctxRebuildWindowTurns is how many turns after a context event the advice stays
	// on rebuild — long enough to re-anchor, short enough not to stall the session.
	ctxRebuildWindowTurns = 2
	// Cadence shares grade turns-since-event against the session's own average
	// event cycle when no token budget is configured (the turn-level fallback rung).
	ctxCadenceCheckpointShare = 0.8
	ctxCadenceBoundedShare    = 0.5
)

// sessionCtxValue is one session's rolling managed-context accumulator. Every field
// is a count or a token sum; none carries prompt content. Mutated only under
// Server.ctxValueMu.
type sessionCtxValue struct {
	turns              int  // every served turn (all wires), not just compacted ones
	contextEvents      int  // turns where fak's own compaction/planned-view transform fired
	turnsSinceEvent    int  // turns since the last context event; == turns when none yet
	lastTurnEvent      bool // the most recent turn was a context event
	ttl1hActive        bool // the 1h prompt-cache TTL-upgrade rung fired for this session
	ttl1hMessagePrefix bool // that rung upgraded at least one message-prefix breakpoint

	// consecutiveEvents is the run of BACK-TO-BACK context-event turns: the window
	// refilling to the limit as fast as the transform sheds it. Any turn that fired no
	// context event resets it, so it measures a RUN and never a lifetime total. At
	// ctxThrashConsecutiveRefills it is the COMPACTION_THRASH verdict (ctxadvice.go).
	consecutiveEvents int
	// noticedClass is the step class the in-band advisory last reported to the model for
	// this session, so the push fires once per ENTRY into a pressure state rather than
	// every turn the session sits in one (ctxAdviceNoteOnce).
	noticedClass StepClass

	// ring holds the last ctxValueWindow turns' resident-token counts within the
	// CURRENT window era; a context event clears it, so the growth slope never spans
	// a rewrite (which would read as negative growth and poison the forecast).
	ring  [ctxValueWindow]int
	count int
	next  int

	lastResident int
	peakResident int

	totalOutput        int64 // cumulative completion tokens (the session's reply volume)
	totalResidentTurns int64 // cumulative resident tokens across turns (prefill volume)

	// footprint is the latest inbound ESTIMATED structural split of the as-sent
	// floor (#3233), captured at the pre-transform anchor. nil until the first
	// Anthropic-passthrough turn. ESTIMATED, never conflated with the OBSERVED
	// token counters above.
	footprint *ctxFootprintBytes
}

// push records one turn's resident-token count into the growth ring.
func (v *sessionCtxValue) push(resident int) {
	if v.count < ctxValueWindow {
		v.count++
	}
	v.ring[v.next] = resident
	v.next = (v.next + 1) % ctxValueWindow
}

// resetRing starts a fresh window era (called when a context event fires).
func (v *sessionCtxValue) resetRing() {
	v.count = 0
	v.next = 0
}

// growthPerTurn is the resident-token slope over the ring: (newest-oldest)/(spanned
// turns). 0 means "no usable slope" (fewer than two turns this era, or a flat/
// shrinking window) — callers treat 0 as no-forecast, never as a real rate.
func (v *sessionCtxValue) growthPerTurn() float64 {
	if v.count < 2 {
		return 0
	}
	oldest := v.ring[(v.next-v.count+2*ctxValueWindow)%ctxValueWindow]
	newest := v.ring[(v.next-1+ctxValueWindow)%ctxValueWindow]
	if newest <= oldest {
		return 0
	}
	return float64(newest-oldest) / float64(v.count-1)
}

// ctxValueState is the folded, pure input adviseCtxStep decides on — split out so
// the policy is unit-testable without a Server (the reset_score.go pattern).
type ctxValueState struct {
	Turns           int
	ContextEvents   int
	TurnsSinceEvent int
	LastTurnEvent   bool
	Resident        int
	Budget          int
	GrowthPerTurn   float64 // <= 0 means unknown
}

// estTurnsToEvent is the FORECAST rungs share: headroom over growth. 0 = no
// forecast (no budget, no growth, or already past the budget).
func (st ctxValueState) estTurnsToEvent() float64 {
	if st.Budget <= 0 || st.GrowthPerTurn <= 0 {
		return 0
	}
	headroom := st.Budget - st.Resident
	if headroom <= 0 {
		return 0
	}
	return float64(headroom) / st.GrowthPerTurn
}

// CtxValueTokens is the token LEVEL: how full the window is and how fast it grows.
type CtxValueTokens struct {
	ResidentTokens     int               `json:"resident_tokens"`
	PeakResidentTokens int               `json:"peak_resident_tokens"`
	BudgetTokens       int               `json:"budget_tokens"` // 0 = no resident budget configured
	Headroom           *CtxValueHeadroom `json:"headroom,omitempty"`
	GrowthPerTurn      float64           `json:"growth_tokens_per_turn"` // 0 = no usable slope yet
	Provenance         string            `json:"provenance"`             // OBSERVED (provider-relayed counters)
}

// CtxValueHeadroom is present only when a resident budget is configured, so a
// zero can never masquerade as a measured "no headroom".
type CtxValueHeadroom struct {
	Tokens int     `json:"tokens"` // budget - resident; negative = past the cut line
	Pct    float64 `json:"pct"`    // share of the budget still free (may be negative)
}

// CtxValueTurns is the reply-loop LEVEL: turn counts and the context-event cadence.
type CtxValueTurns struct {
	TurnsObserved          int     `json:"turns_observed"`
	ContextEvents          int     `json:"context_events"`
	TurnsSinceContextEvent int     `json:"turns_since_context_event"`
	AvgTurnsBetweenEvents  float64 `json:"avg_turns_between_events,omitempty"` // 0 = no event yet
	EstTurnsToContextEvent float64 `json:"est_turns_to_context_event,omitempty"`
	EstProvenance          string  `json:"est_provenance,omitempty"` // FORECAST when the estimate is set
	Provenance             string  `json:"provenance"`               // WITNESSED (fak counted its own turns/events)
}

// CtxValueSession is the whole-arc LEVEL: cumulative volumes and a closed phase.
type CtxValueSession struct {
	TotalOutputTokens       int64  `json:"total_output_tokens"`
	TotalResidentTokenTurns int64  `json:"total_resident_token_turns"`
	Phase                   string `json:"phase"`      // fresh | building | cruising | crowding | post_event
	Provenance              string `json:"provenance"` // WITNESSED
}

// CtxStepAdvice is the DECISION: a closed step class plus the basis and the
// deciding numbers, so the agent can weigh it rather than obey it.
type CtxStepAdvice struct {
	StepClass  StepClass `json:"step_class"`
	Affordance string    `json:"affordance"`
	Basis      string    `json:"basis"` // token_headroom | event_cadence | context_event | none
	Reason     string    `json:"reason"`
	Provenance string    `json:"provenance"` // DECISION
}

// Cache-window horizon (#2446). Prompt-cache economics gate self-pacing wakeups:
// a session that wakes at TTL+1s pays a full re-prefill the TTL-30s wake avoided.
// Anthropic's default prompt-cache window is 5 minutes; the stable-head 1h
// TTL-upgrade rung (maybeUpgradeAnthropicCacheTTL1H) extends it to one hour.
const (
	cacheTTLWindow5mMs int64 = 5 * 60 * 1000
	cacheTTLWindow1hMs int64 = 60 * 60 * 1000
)

// CtxValueWakeup is the ADVICE-ONLY cache-window horizon: the latest a self-pacing
// scheduler can fire the next turn and still land inside the prompt-cache window,
// plus the priced re-prefill it forfeits by knowingly committing past it. Like
// step_advice, nothing enforces it — loop skills and schedulers consume it. The
// provenance splits by owner (Law A2): the TTL window length is OBSERVED
// (provider-owned); the tier decision and the horizon arithmetic are WITNESSED
// (fak's own upgrade rung + math).
type CtxValueWakeup struct {
	TTLTier                   string `json:"ttl_tier"`                     // "5m" | "1h" — which cache window applies
	WindowMs                  int64  `json:"window_ms"`                    // the cache TTL window length in ms
	WindowProvenance          string `json:"window_provenance"`            // OBSERVED (provider-owned TTL)
	WakeupByMs                int64  `json:"wakeup_by_ms"`                 // latest next-turn offset from the last served turn (ms) that stays in-window
	PastWindowReprefillTokens int    `json:"past_window_reprefill_tokens"` // resident prompt re-prefilled uncached if the next turn lands past the window
	HorizonProvenance         string `json:"horizon_provenance"`           // WITNESSED (fak upgrade decision + arithmetic)
}

// ctxWakeupHorizon folds the session's cache-window tier into the wakeup horizon.
// ttl1h reflects whether the 1h TTL-upgrade rung fired for this session; resident
// is the prompt that re-prefills if the next turn lands past the window. Pure, so
// the horizon is unit-testable without a Server (the adviseCtxStep pattern).
func ctxWakeupHorizon(ttl1h bool, resident int) CtxValueWakeup {
	tier, window := "5m", cacheTTLWindow5mMs
	if ttl1h {
		tier, window = "1h", cacheTTLWindow1hMs
	}
	return CtxValueWakeup{
		TTLTier:                   tier,
		WindowMs:                  window,
		WindowProvenance:          "OBSERVED",
		WakeupByMs:                window,
		PastWindowReprefillTokens: maxNonNeg(resident),
		HorizonProvenance:         "WITNESSED",
	}
}

// CtxValueReport is one session's multi-level managed-context report.
type CtxValueReport struct {
	Schema     string          `json:"schema"`
	TraceID    string          `json:"trace_id"`
	Tokens     CtxValueTokens  `json:"tokens"`
	Turns      CtxValueTurns   `json:"turns"`
	Session    CtxValueSession `json:"session"`
	StepAdvice CtxStepAdvice   `json:"step_advice"`
	Wakeup     CtxValueWakeup  `json:"wakeup"`
	// Footprint is the ESTIMATED where-did-they-go split of the as-sent floor
	// (#3233): system + built-in tools + MCP tools, taken at the pre-transform
	// anchor. nil until the first Anthropic-passthrough turn. Composed with, never
	// substituted for, the OBSERVED Tokens above (Law A2).
	Footprint *CtxValueFootprint `json:"footprint,omitempty"`
	// Expense is the DECISION on whether the session has become UNUSUALLY EXPENSIVE,
	// graded on Footprint's total per-turn as-sent volume against the warn/block
	// lines (ctxexpense.go). Always present; Level is "ok"/Basis "none" until a
	// footprint exists to grade. This is the "warns/blocks" verdict the --debug-stats
	// line and the opt-in block advisory read from.
	Expense CtxExpense `json:"expense"`
}

// CtxValueSnapshot is the multi-session HTTP body: every tracked session's report.
type CtxValueSnapshot struct {
	Schema       string           `json:"schema"`
	BudgetTokens int              `json:"budget_tokens"`
	Sessions     []CtxValueReport `json:"sessions"`
}

const ctxValueSchema = "fak-ctxvalue-report/1"

// ContextValueRequest is the fak_context_value MCP argument shape. An omitted
// trace_id resolves through traceFor to the gateway default trace — under
// `fak guard` that is the wrapped session itself, so the agent asking "how much
// window do I have left?" needs no out-of-band identity.
type ContextValueRequest struct {
	TraceID string `json:"trace_id"`
}

// adviseCtxStep is the pure step-advice policy, rung by rung:
//
//  1. no turns                → unknown (nothing observed, nothing to decide on)
//  2. just after an event     → rebuild (re-anchor from durable state first)
//  3. budget configured       → token_headroom: percent rungs (80/50, shared with
//     the debug nudge) tightened by the FORECAST turn floors (4/12)
//  4. events but no budget    → event_cadence: grade turns-since-event against the
//     session's own average event cycle (0.8/0.5 shares)
//  5. neither                 → unknown (the window edge is not visible from here;
//     fail closed instead of guessing "any")
func adviseCtxStep(st ctxValueState) CtxStepAdvice {
	a := CtxStepAdvice{Provenance: "DECISION"}
	switch {
	case st.Turns == 0:
		a.StepClass, a.Basis = StepClassUnknown, "none"
		a.Reason = "no served turns observed for this session yet"
	case st.LastTurnEvent || (st.ContextEvents > 0 && st.TurnsSinceEvent < ctxRebuildWindowTurns):
		a.StepClass, a.Basis = StepClassRebuild, "context_event"
		a.Reason = fmt.Sprintf("context event %d turn(s) ago rewrote the window; re-anchor from durable state before a wide step", st.TurnsSinceEvent)
	case st.Budget > 0 && st.Resident > 0:
		a.Basis = "token_headroom"
		usedPct := float64(st.Resident) * 100 / float64(st.Budget)
		est := st.estTurnsToEvent()
		switch {
		case st.Resident*100 >= st.Budget*ctxStepCheckpointPercent || (est > 0 && est < ctxStepCheckpointTurns):
			a.StepClass = StepClassCheckpoint
		case st.Resident*100 >= st.Budget*ctxStepBoundedPercent || (est > 0 && est < ctxStepBoundedTurns):
			a.StepClass = StepClassBounded
		default:
			a.StepClass = StepClassAny
		}
		a.Reason = fmt.Sprintf("resident %s of %s budget (%.0f%% used)",
			HumanTokenEquiv(float64(st.Resident)), HumanTokenEquiv(float64(st.Budget)), usedPct)
		if est > 0 {
			a.Reason += fmt.Sprintf(", est %.1f turn(s) to the next context event", est)
		}
	case st.ContextEvents > 0:
		a.Basis = "event_cadence"
		avg := float64(st.Turns) / float64(st.ContextEvents)
		share := float64(st.TurnsSinceEvent) / avg
		switch {
		case share >= ctxCadenceCheckpointShare:
			a.StepClass = StepClassCheckpoint
		case share >= ctxCadenceBoundedShare:
			a.StepClass = StepClassBounded
		default:
			a.StepClass = StepClassAny
		}
		a.Reason = fmt.Sprintf("no token budget configured; %d turn(s) since the last context event vs a %.1f-turn average cycle", st.TurnsSinceEvent, avg)
	default:
		a.StepClass, a.Basis = StepClassUnknown, "none"
		a.Reason = "no resident budget configured and no context event observed; the window edge is not visible from the guard"
	}
	headroom := int64(0)
	if st.Budget > st.Resident {
		headroom = int64(st.Budget - st.Resident)
	}
	a.Affordance = negframe.Reframe(sessionsteer.StepAdviceAffordance(string(a.StepClass), headroom))
	return a
}

// ctxPhase folds the same state into the closed session-level phase.
func ctxPhase(st ctxValueState) string {
	switch {
	case st.Turns == 0:
		return "fresh"
	case st.LastTurnEvent || (st.ContextEvents > 0 && st.TurnsSinceEvent < ctxRebuildWindowTurns):
		return "post_event"
	case st.Budget > 0 && st.Resident*100 >= st.Budget*ctxStepCheckpointPercent:
		return "crowding"
	case st.Budget > 0 && st.Resident*100 >= st.Budget*ctxStepBoundedPercent:
		return "cruising"
	case st.Turns < 3:
		return "fresh"
	default:
		return "building"
	}
}

// observeCtxValue rolls one served turn into the session's managed-context record.
// Called from logInferenceTurn on EVERY served turn (all wires), before any log/
// debug sink gating — the same always-on posture as observeVCacheTurn. A nil
// server or empty trace is a safe no-op.
func (s *Server) observeCtxValue(trace string, uncachedPrompt, cacheRead, cacheCreation, completion int, compacted bool) {
	if s == nil || strings.TrimSpace(trace) == "" {
		return
	}
	resident := maxNonNeg(uncachedPrompt) + maxNonNeg(cacheRead) + maxNonNeg(cacheCreation)
	s.ctxValueMu.Lock()
	defer s.ctxValueMu.Unlock()
	v := s.ctxValueForLocked(trace)
	if v == nil {
		return
	}
	v.turns++
	if compacted {
		v.contextEvents++
		v.turnsSinceEvent = 0
		v.lastTurnEvent = true
		v.resetRing() // the growth slope never spans a window rewrite
		// COMPACTION THRASH (#2424): count the RUN of back-to-back context events. Recorded
		// on the turn the run REACHES the line (== not >=) so one thrashing stretch books
		// exactly one verdict however long it goes on — a session stuck at the limit must
		// read as one sick session, not as a counter that climbs with the transcript.
		v.consecutiveEvents++
		if v.consecutiveEvents == ctxThrashConsecutiveRefills {
			s.metrics.recordCompactionThrash()
		}
	} else {
		v.turnsSinceEvent++
		v.lastTurnEvent = false
		v.consecutiveEvents = 0
	}
	v.push(resident)
	v.lastResident = resident
	if resident > v.peakResident {
		v.peakResident = resident
	}
	v.totalOutput += int64(maxNonNeg(completion))
	v.totalResidentTurns += int64(resident)
	advice := adviseCtxStep(ctxValueState{Turns: v.turns, ContextEvents: v.contextEvents, TurnsSinceEvent: v.turnsSinceEvent, LastTurnEvent: v.lastTurnEvent, Resident: v.lastResident, Budget: s.compactHistoryBudget, GrowthPerTurn: v.growthPerTurn()})
	checkpoint := s.autoCheckpoint
	if checkpoint != nil && (advice.StepClass == StepClassCheckpoint || advice.StepClass == StepClassRebuild) {
		go checkpoint(trace, "compaction")
	}
}

// ctxValueForLocked mints the per-trace record lazily, bounding the table
// generationally on overflow (the resetHealthForLocked policy). Caller holds
// ctxValueMu.
func (s *Server) ctxValueForLocked(trace string) *sessionCtxValue {
	if trace == "" {
		return nil
	}
	if s.ctxValue == nil {
		s.ctxValue = make(map[string]*sessionCtxValue)
	}
	if v, ok := s.ctxValue[trace]; ok {
		return v
	}
	if len(s.ctxValue) >= maxCtxValueSessions {
		s.ctxValue = make(map[string]*sessionCtxValue) // generational reset
	}
	v := &sessionCtxValue{}
	s.ctxValue[trace] = v
	return v
}

// noteCtxValueTTL1h records that the 1h prompt-cache TTL-upgrade rung fired for
// this session, so the wakeup horizon reads the 1h window instead of the 5m
// default from here on. Called from the Anthropic passthrough transform the turn
// the upgrade applies (maybeUpgradeAnthropicCacheTTL1H). A nil server or empty
// trace no-ops; the note-minted record stays out of the multi-session snapshot
// until a real served turn is observed (the no-phantom invariant).
func (s *Server) noteCtxValueTTL1h(trace string) { s.noteCtxValueTTL1hScope(trace, false) }

// noteCtxValueTTL1hScope records whether this trace's admitted 1h layout reaches
// into messages. That scope lets provider cache_creation tokens be split into the
// head-only baseline versus the message-prefix arm without guessing from totals.
func (s *Server) noteCtxValueTTL1hScope(trace string, messagePrefix bool) {
	if s == nil || strings.TrimSpace(trace) == "" {
		return
	}
	s.ctxValueMu.Lock()
	defer s.ctxValueMu.Unlock()
	if v := s.ctxValueForLocked(trace); v != nil {
		v.ttl1hActive = true
		v.ttl1hMessagePrefix = v.ttl1hMessagePrefix || messagePrefix
	}
}

// ttl1hActiveFor reports whether the managed-cache 1h TTL-upgrade rung has fired
// for this trace (noteCtxValueTTL1h), so a cache-creation-token counter can
// attribute a turn's write to the 1h tier (#2179) instead of the unattributed 5m
// default. A trace with no recorded session (never upgraded, or an empty trace
// from a non-session caller) reads false — never true on a guess.
func (s *Server) ttl1hActiveFor(trace string) bool {
	if s == nil || strings.TrimSpace(trace) == "" {
		return false
	}
	s.ctxValueMu.Lock()
	defer s.ctxValueMu.Unlock()
	if v, ok := s.ctxValue[trace]; ok {
		return v.ttl1hActive
	}
	return false
}

// ttl1hMessagePrefixFor reports whether the admitted 1h layout for trace includes
// a message-prefix breakpoint. False is the head-only control arm.
func (s *Server) ttl1hMessagePrefixFor(trace string) bool {
	if s == nil || strings.TrimSpace(trace) == "" {
		return false
	}
	s.ctxValueMu.Lock()
	defer s.ctxValueMu.Unlock()
	if v, ok := s.ctxValue[trace]; ok {
		return v.ttl1hMessagePrefix
	}
	return false
}

// ctxValueStateLocked folds the accumulator into the pure policy input.
func (s *Server) ctxValueStateLocked(v *sessionCtxValue) ctxValueState {
	return ctxValueState{
		Turns:           v.turns,
		ContextEvents:   v.contextEvents,
		TurnsSinceEvent: v.turnsSinceEvent,
		LastTurnEvent:   v.lastTurnEvent,
		Resident:        v.lastResident,
		Budget:          s.compactHistoryBudget,
		GrowthPerTurn:   v.growthPerTurn(),
	}
}

// ctxValueReportLocked renders one session's multi-level report. Caller holds
// ctxValueMu.
func (s *Server) ctxValueReportLocked(trace string, v *sessionCtxValue) CtxValueReport {
	st := s.ctxValueStateLocked(v)
	r := CtxValueReport{
		Schema:  ctxValueSchema,
		TraceID: trace,
		Tokens: CtxValueTokens{
			ResidentTokens:     v.lastResident,
			PeakResidentTokens: v.peakResident,
			BudgetTokens:       st.Budget,
			GrowthPerTurn:      st.GrowthPerTurn,
			Provenance:         "OBSERVED",
		},
		Turns: CtxValueTurns{
			TurnsObserved:          v.turns,
			ContextEvents:          v.contextEvents,
			TurnsSinceContextEvent: v.turnsSinceEvent,
			Provenance:             "WITNESSED",
		},
		Session: CtxValueSession{
			TotalOutputTokens:       v.totalOutput,
			TotalResidentTokenTurns: v.totalResidentTurns,
			Phase:                   ctxPhase(st),
			Provenance:              "WITNESSED",
		},
		StepAdvice: adviseCtxStep(st),
		Wakeup:     ctxWakeupHorizon(v.ttl1hActive, v.lastResident),
	}
	if st.Budget > 0 {
		r.Tokens.Headroom = &CtxValueHeadroom{
			Tokens: st.Budget - st.Resident,
			Pct:    float64(st.Budget-st.Resident) * 100 / float64(st.Budget),
		}
	}
	if v.contextEvents > 0 {
		r.Turns.AvgTurnsBetweenEvents = float64(v.turns) / float64(v.contextEvents)
	}
	if est := st.estTurnsToEvent(); est > 0 {
		r.Turns.EstTurnsToContextEvent = est
		r.Turns.EstProvenance = "FORECAST"
	}
	// The ESTIMATED as-sent footprint (#3233), surfaced beside the OBSERVED tokens.
	r.Footprint = v.footprint.report()
	// The DECISION on per-turn expense, graded on that footprint's total volume.
	r.Expense = s.ctxExpenseForLocked(v)
	return r
}

// CtxValueReportFor is the single-session read the MCP tool serves. A trace with
// no served turns returns a decidable zero report (step_class=unknown with the
// reason), never an error — the agent always gets an answer it can act on.
func (s *Server) CtxValueReportFor(trace string) CtxValueReport {
	trace = strings.TrimSpace(trace)
	if s == nil || trace == "" {
		return CtxValueReport{Schema: ctxValueSchema, StepAdvice: adviseCtxStep(ctxValueState{}), Wakeup: ctxWakeupHorizon(false, 0)}
	}
	s.ctxValueMu.Lock()
	defer s.ctxValueMu.Unlock()
	if v, ok := s.ctxValue[trace]; ok {
		return s.ctxValueReportLocked(trace, v)
	}
	r := CtxValueReport{Schema: ctxValueSchema, TraceID: trace}
	r.Session.Phase = ctxPhase(ctxValueState{})
	r.Session.Provenance = "WITNESSED"
	r.Tokens.Provenance = "OBSERVED"
	r.Turns.Provenance = "WITNESSED"
	r.StepAdvice = adviseCtxStep(ctxValueState{Budget: s.compactHistoryBudget})
	r.Wakeup = ctxWakeupHorizon(false, 0)
	// No footprint yet ⇒ an ok/none expense verdict (nothing to grade), carrying the
	// configured lines so a reader sees the thresholds even before the first turn.
	r.Expense = assessCtxExpense(ctxExpenseState{WarnTokens: s.ctxExpenseWarn, BlockTokens: s.ctxExpenseBlock})
	return r
}

// ctxValueSnapshot renders every tracked session (optionally filtered to one
// trace), sorted by trace id for a stable wire order. No sessions means an empty
// list — a session only appears once a served turn was observed (no phantoms).
func (s *Server) ctxValueSnapshot(traceFilter string) CtxValueSnapshot {
	snap := CtxValueSnapshot{Schema: ctxValueSchema, Sessions: []CtxValueReport{}}
	if s == nil {
		return snap
	}
	snap.BudgetTokens = s.compactHistoryBudget
	s.ctxValueMu.Lock()
	defer s.ctxValueMu.Unlock()
	for trace, v := range s.ctxValue {
		if v.turns == 0 {
			continue // note-minted (e.g. the TTL rung) but no served turn yet — no phantom
		}
		if traceFilter != "" && trace != traceFilter {
			continue
		}
		snap.Sessions = append(snap.Sessions, s.ctxValueReportLocked(trace, v))
	}
	sort.Slice(snap.Sessions, func(i, j int) bool { return snap.Sessions[i].TraceID < snap.Sessions[j].TraceID })
	return snap
}

// handleFakCtxValue serves GET /v1/fak/ctxvalue: the managed-context value report
// for every tracked session, or one session with ?trace=<id>.
func (s *Server) handleFakCtxValue(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s.ctxValueSnapshot(strings.TrimSpace(r.URL.Query().Get("trace"))))
}
