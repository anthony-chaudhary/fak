package gateway

import "encoding/json"

// reset_shadow.go -- the LIVE, SHADOW-mode wiring of the per-session compaction-health reset
// policy (#792, epic #788). reset_score.go shipped the PURE decision surface (ResetScore over a
// CacheHealthState); this file supplies the issue's other two deliverables:
//
//   1. a PER-SESSION rolling CacheHealthState the gateway fills from the provider's OBSERVED
//      cache counters on every compacted turn (the struct existed; nothing populated it), and
//   2. the SHADOW emission -- a recommend-only log line + a ResetReason-labeled metric that
//      records "would a reset beat another cut right now?" and NEVER acts on the answer.
//
// Nothing here resets a session. The policy stays CUT-BY-DEFAULT until shadow evidence supports
// enabling reset; observeResetHealth only rolls state, scores it, and reports the verdict.
//
// PROVENANCE (the same split the compaction metrics keep): the VERDICT (ShouldReset/Score/Reason)
// is WITNESSED -- fak computed it from its own policy over inputs it controls. The INPUTS it folds
// (cache_read / input / cache_creation tokens) are OBSERVED -- relayed verbatim from the upstream,
// never a fak claim. The shadow log/metric label the verdict a RECOMMENDATION, so a stale_prefix
// reading is never mistaken for fak having reset anything (it has not -- shadow mode is inert).

// maxResetHealthSessions bounds the per-session rolling-health table exactly like the
// sessionPlanners map (gateway.go), so a long-running gateway minting a fresh trace per session
// cannot grow it without limit. On overflow the table takes a generational reset (the same
// cheap, correctness-preserving policy sessionPlannerFor uses): the rolling health of an
// evicted session simply restarts at unknown_provider, never a stale verdict.
const maxResetHealthSessions = 8192

// resetHealthWindow is how many recent compacted turns the rolling read-ratio averages over. It
// matches the reset-cooldown scale (DefaultResetCooldownTurns) so the window a verdict sees and
// the hysteresis that gates it move on the same horizon: a stale stretch shorter than the window
// cannot dominate the average, and a recovery shows up within one cooldown.
const resetHealthWindow = 8

// neverResetSentinel is TurnsSinceReset for a session that has never reset. It is large so the
// cooldown gate (ResetScore rung 3/4) is always satisfied for a first-ever recommendation -- a
// session that never reset has no flapping to guard against. An actual reset (resetHealthReset)
// zeroes it, after which the cooldown holds the next recommendation back for CooldownTurns.
const neverResetSentinel = 1 << 30

// sessionResetHealth is one session's rolling, content-free compaction-health accumulator -- the
// live source the gateway folds into a CacheHealthState for ResetScore. Every field is a count or
// a token sum; none carries prompt content, so the shadow log of the derived verdict is safe.
// Mutated only under Server.resetHealthMu by the single in-flight turn for that trace.
type sessionResetHealth struct {
	observedTurns   int  // compacted turns seen for this session (the ResetScore warmup gate)
	turnsSinceReset int  // turns since this session last reset (neverResetSentinel until it has)
	hasSignal       bool // a provider reported non-zero prompt/cache counters on some observed turn

	// ring holds the last resetHealthWindow turns' (cacheRead, input, creation) token counts; the
	// running sums let RecentReadRatio be computed in O(1) per turn without rescanning the window.
	ring         [resetHealthWindow]resetTurnTokens
	count        int // turns pushed (caps at resetHealthWindow once full)
	next         int // next ring slot to overwrite (circular)
	sumCacheRead int
	sumInput     int
	sumCreation  int
}

// resetTurnTokens is one compacted turn's OBSERVED prompt-token split, relayed from the provider.
type resetTurnTokens struct {
	cacheRead int
	input     int
	creation  int
}

// push records one compacted turn's observed tokens, evicting the oldest when the window is full
// and keeping the running sums in lockstep. It is the only mutation of the ring/sum invariants.
func (h *sessionResetHealth) push(t resetTurnTokens) {
	if h.count == resetHealthWindow {
		old := h.ring[h.next]
		h.sumCacheRead -= old.cacheRead
		h.sumInput -= old.input
		h.sumCreation -= old.creation
	} else {
		h.count++
	}
	h.ring[h.next] = t
	h.sumCacheRead += t.cacheRead
	h.sumInput += t.input
	h.sumCreation += t.creation
	h.next = (h.next + 1) % resetHealthWindow
}

// state folds the rolling accumulator into the pure CacheHealthState ResetScore consumes.
// RecentReadRatio is the OBSERVED share of the prompt the provider served from cache over the
// window: cache_read / (cache_read + input + creation). A still-landing prefix keeps it high; a
// stale one craters it toward 0 -- exactly the signal the cut-vs-reset crossover turns on.
func (h *sessionResetHealth) state() CacheHealthState {
	total := h.sumCacheRead + h.sumInput + h.sumCreation
	ratio := 0.0
	if total > 0 {
		ratio = float64(h.sumCacheRead) / float64(total)
	}
	return CacheHealthState{
		ObservedTurns:     h.observedTurns,
		RecentReadRatio:   ratio,
		HasProviderSignal: h.hasSignal,
		ProviderIdleHint:  false, // no provider exposes a TTL/idle hint today; leave it off
		TurnsSinceReset:   h.turnsSinceReset,
	}
}

// resetHealthFor returns the rolling-health record for a trace, minting one lazily and bounding
// the table generationally on overflow. It returns nil for an empty trace (the single-session
// default has no per-session health to roll). Caller holds resetHealthMu.
func (s *Server) resetHealthForLocked(trace string) *sessionResetHealth {
	if trace == "" {
		return nil
	}
	if s.resetHealth == nil {
		s.resetHealth = make(map[string]*sessionResetHealth)
	}
	if h, ok := s.resetHealth[trace]; ok {
		return h
	}
	if len(s.resetHealth) >= maxResetHealthSessions {
		s.resetHealth = make(map[string]*sessionResetHealth) // generational reset, like sessionPlanners
	}
	h := &sessionResetHealth{turnsSinceReset: neverResetSentinel}
	s.resetHealth[trace] = h
	return h
}

// observeResetHealth rolls one compacted turn's OBSERVED provider counters into the session's
// rolling health, scores it through the cut-by-default policy, and emits the verdict in SHADOW
// mode (recommend-only -- it acts on nothing). It returns the decision and the rolled state so a
// caller that wants the per-turn view (the --debug-stats render, #793) reuses this one roll
// rather than re-deriving it. A nil server / empty trace is a safe no-op returning the zero
// (unknown_provider) verdict.
func (s *Server) observeResetHealth(trace string, input, cacheRead, creation int) (ResetDecision, CacheHealthState) {
	if s == nil || trace == "" {
		return ResetDecision{Reason: ResetReasonUnknown}, CacheHealthState{}
	}
	s.resetHealthMu.Lock()
	h := s.resetHealthForLocked(trace)
	if h == nil {
		s.resetHealthMu.Unlock()
		return ResetDecision{Reason: ResetReasonUnknown}, CacheHealthState{}
	}
	h.observedTurns++
	if h.turnsSinceReset < neverResetSentinel {
		h.turnsSinceReset++
	}
	if input+cacheRead+creation > 0 {
		h.hasSignal = true
	}
	h.push(resetTurnTokens{cacheRead: cacheRead, input: input, creation: creation})
	st := h.state()
	s.resetHealthMu.Unlock()

	d := DefaultResetPolicy().ResetScore(st)
	// Emission is OUTSIDE the lock: the metric takes its own lock and the log may block, neither
	// of which should stall another session's turn. The metric always accumulates; the log only
	// fires when a --log sink is wired (the same gating as logInferenceTurn).
	s.metrics.recordResetShadow(d)
	s.logResetShadow(trace, d, st)
	return d, st
}

// resetHealthReset zeroes a session's cooldown clock -- the hook an eventual NON-shadow path
// calls the moment it actually resets a session, so the hysteresis (ResetScore rung 3/4) then
// holds the next recommendation back for CooldownTurns. Shadow mode never calls it on the live
// path; it makes the cooldown contract complete and is exercised by the test.
func (s *Server) resetHealthReset(trace string) {
	if s == nil || trace == "" {
		return
	}
	s.resetHealthMu.Lock()
	if h := s.resetHealthForLocked(trace); h != nil {
		h.turnsSinceReset = 0
	}
	s.resetHealthMu.Unlock()
}

// logResetShadow writes the recommend-only shadow verdict as a structured, content-free log line
// (gated on the --log sink). It carries the closed reason, the score, and the rolling inputs --
// never any prompt bytes -- so an operator can see the cut-vs-reset pressure build without the
// gateway leaking conversation content.
func (s *Server) logResetShadow(trace string, d ResetDecision, st CacheHealthState) {
	if s == nil || s.logf == nil || trace == "" {
		return
	}
	b, err := json.Marshal(map[string]any{
		"event":               "gateway_reset_shadow",
		"trace_id":            trace,
		"recommend_reset":     d.ShouldReset, // SHADOW: a recommendation; nothing acts on it
		"reset_score":         d.Score,
		"reset_reason":        string(d.Reason),
		"observed_turns":      st.ObservedTurns,
		"recent_read_ratio":   st.RecentReadRatio,
		"turns_since_reset":   st.TurnsSinceReset,
		"has_provider_signal": st.HasProviderSignal,
	})
	if err != nil {
		return
	}
	s.logf("%s", b)
}
