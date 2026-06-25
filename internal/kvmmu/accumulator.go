package kvmmu

import "sort"

// accumulator.go — issue #855, rung 4 of the attention-witness epic (#851).
//
// Rung 2 (#853, attention.go) gives each Segment a scalar Attended: the witnessed
// post-softmax attention mass a span received DURING ONE TURN. A span's value, though,
// spans many turns — it can idle then turn load-bearing, or run hot then die. This file
// adds the temporal dual over that per-turn signal, with two reductions chosen by one knob:
//
//	real-time controller   A_s(t) = λ·A_s(t-1) + a_s(t)   — an EMA (recency-decayed rolling
//	                       sum). With λ<1 this IS the H2O / heavy-hitter signal, but as a
//	                       WITNESSED kernel quantity (see #851's novelty boundary — we did
//	                       not invent heavy-hitters; we witness them from the live forward pass).
//	post-hoc analyst       A_s   = Σ_t a_s(t)             — the undecayed cumulative sum, plus
//	                       the trajectory {a_s(t)} (which turns was the span hot?).
//
// ONE ACCUMULATOR, TWO λ. λ is the only knob separating "hot now" (eviction, #856) from
// "mattered overall" (audit, #857). λ=1 collapses the EMA onto the plain cumulative sum;
// λ=0 keeps only the most recent turn.
//
// The accumulator lives BESIDE the span ledger but does not own it: at a turn boundary it
// reads each live segment's per-turn Attended (ObserveContext) and folds it in; the segment's
// own scalar is cleared by ResetAttention before the next turn refills it. Reset-on-evict
// carries over from rung 2: evict() already zeroes a held span's Segment.Attended, and the
// accumulator's counterpart is Forget(id) — but WHO forgets is a consumer policy, not the
// accumulator's. The real-time controller (#856) forgets an evicted span (it can no longer be
// attended to); the post-hoc analyst (#857) deliberately keeps it (its history is the answer
// to "was that 9000-token blob ever worth its residency?"). The accumulator stays policy-free
// and exposes both EMA/Cumulative and Forget so each consumer picks its own posture.

// defaultTrajectoryCap bounds the per-span trajectory ring so a long session stays O(cap) per
// span, not O(turns). A turn boundary pushes one entry per resident span; the ring keeps the
// most recent cap turns. Chosen generously (a span attended across hundreds of turns is already
// the outlier the post-hoc report wants) but finite so memory cannot grow without bound.
const defaultTrajectoryCap = 512

// TurnMass is one entry in a span's attention trajectory: the witnessed mass it drew in a
// specific (1-based, session-global) turn. The trajectory is dense from a span's first
// observed turn, so a zero Mass is a real "resident but unattended that turn", not a gap.
type TurnMass struct {
	Turn int     `json:"turn"`
	Mass float64 `json:"mass"`
}

// SpanAttention is the read-only snapshot of one span's accumulated attention: the EMA
// (recency-decayed real-time signal), the undecayed Cumulative (post-hoc signal), and the
// bounded Trajectory ring. EMA == Cumulative exactly when λ == 1.
type SpanAttention struct {
	ID         string     `json:"id"`
	EMA        float64    `json:"ema"`
	Cumulative float64    `json:"cumulative"`
	Trajectory []TurnMass `json:"trajectory,omitempty"`
}

// spanAccum is the mutable per-span state behind the accumulator: the two reductions plus the
// bounded trajectory ring (oldest at index 0).
type spanAccum struct {
	ema  float64
	cum  float64
	traj []TurnMass
}

// AttentionAccumulator folds a stream of per-turn per-span witnessed attention masses into a
// recency-decayed EMA and an undecayed cumulative sum, keeping a bounded per-turn trajectory
// for each span. It is deterministic (no clock, no randomness) and not safe for concurrent
// mutation — fold one turn at a time under the same single-flight discipline the rest of the
// bridge assumes for a Context.
type AttentionAccumulator struct {
	lambda  float64
	ringCap int
	turn    int
	spans   map[string]*spanAccum
}

// NewAttentionAccumulator builds an accumulator with EMA decay λ and trajectory ring cap. λ is
// clamped to [0,1] (λ=1 makes the EMA the plain cumulative sum; λ=0 keeps only the last turn).
// A ringCap <= 0 falls back to defaultTrajectoryCap.
func NewAttentionAccumulator(lambda float64, ringCap int) *AttentionAccumulator {
	if lambda < 0 {
		lambda = 0
	}
	if lambda > 1 {
		lambda = 1
	}
	if ringCap <= 0 {
		ringCap = defaultTrajectoryCap
	}
	return &AttentionAccumulator{lambda: lambda, ringCap: ringCap, spans: make(map[string]*spanAccum)}
}

// Observe folds one turn's witnessed per-span mass into the accumulator. It advances the global
// turn counter, decays EVERY known span's EMA by λ (so a resident span that drew no mass this
// turn cools toward zero — the "ran hot then died" case), then adds this turn's mass to the EMA,
// the cumulative sum, and the trajectory ring. A span named for the first time starts at
// A_s = a_s(t). Masses are the raw witnessed Attended scalar (not normalized); a consumer
// normalizes when it needs a ratio.
//
// Pure and deterministic: the same sequence of turnMass maps yields the same accumulator state.
func (a *AttentionAccumulator) Observe(turnMass map[string]float64) {
	a.turn++
	// Decay + add for every span already known. This is what makes the EMA an EMA: an
	// unobserved-this-turn span (mass 0, absent from turnMass) still decays. It also keeps the
	// trajectory dense (a real 0 for "resident but cold"), so post-hoc alignment is by index.
	for id, sp := range a.spans {
		sp.ema *= a.lambda
		m := turnMass[id]
		sp.ema += m
		sp.cum += m
		a.push(sp, m)
	}
	// Spans seen for the first time this turn: A_s starts at a_s(t) (λ·0 + m == m).
	for id, m := range turnMass {
		if _, ok := a.spans[id]; ok {
			continue
		}
		sp := &spanAccum{ema: m, cum: m}
		a.push(sp, m)
		a.spans[id] = sp
	}
}

// push appends one turn's mass to a span's trajectory ring, shifting the oldest entry out once
// the ring is full so the per-span cost stays O(cap), not O(turns), with no reallocation churn.
func (a *AttentionAccumulator) push(sp *spanAccum, m float64) {
	e := TurnMass{Turn: a.turn, Mass: m}
	if len(sp.traj) < a.ringCap {
		sp.traj = append(sp.traj, e)
		return
	}
	copy(sp.traj, sp.traj[1:])
	sp.traj[len(sp.traj)-1] = e
}

// Forget drops a span's accumulated state — the accumulator's side of evict()'s reset-on-evict.
// It is a CONSUMER decision (see the file header): the real-time controller forgets an evicted
// span; the post-hoc analyst keeps it. Idempotent.
func (a *AttentionAccumulator) Forget(id string) {
	delete(a.spans, id)
}

// EMA returns the recency-decayed rolling attention for a span (0 if unknown) — the real-time
// controller signal #856's eviction reads to find the coldest span.
func (a *AttentionAccumulator) EMA(id string) float64 {
	if sp, ok := a.spans[id]; ok {
		return sp.ema
	}
	return 0
}

// Cumulative returns the undecayed total attention a span has drawn (0 if unknown) — the
// post-hoc analyst signal #857's session report reads.
func (a *AttentionAccumulator) Cumulative(id string) float64 {
	if sp, ok := a.spans[id]; ok {
		return sp.cum
	}
	return 0
}

// Trajectory returns a copy of a span's per-turn mass ring (nil if unknown), most recent last.
func (a *AttentionAccumulator) Trajectory(id string) []TurnMass {
	sp, ok := a.spans[id]
	if !ok {
		return nil
	}
	return append([]TurnMass(nil), sp.traj...)
}

// Turns is the number of turns Observe has folded.
func (a *AttentionAccumulator) Turns() int { return a.turn }

// Span returns the accumulated attention snapshot for one span, or false if the accumulator has
// never seen (or has Forgotten) it.
func (a *AttentionAccumulator) Span(id string) (SpanAttention, bool) {
	sp, ok := a.spans[id]
	if !ok {
		return SpanAttention{}, false
	}
	return a.snapshot(id, sp), true
}

// Snapshot returns every known span's accumulated attention, sorted by descending EMA then ID
// (deterministic), so a controller reads the hottest at the head and the coldest at the tail:
// #856 evicts the tail; #857 reports the head.
func (a *AttentionAccumulator) Snapshot() []SpanAttention {
	out := make([]SpanAttention, 0, len(a.spans))
	for id, sp := range a.spans {
		out = append(out, a.snapshot(id, sp))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].EMA != out[j].EMA {
			return out[i].EMA > out[j].EMA
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func (a *AttentionAccumulator) snapshot(id string, sp *spanAccum) SpanAttention {
	return SpanAttention{
		ID:         id,
		EMA:        sp.ema,
		Cumulative: sp.cum,
		Trajectory: append([]TurnMass(nil), sp.traj...),
	}
}

// ObserveContext folds this Context's current per-span Attended mass into the accumulator as one
// turn. It reads only resident (non-held) spans and does NOT clear Attended — call ResetAttention
// at the turn boundary so the NEXT turn's fold sees per-turn mass a_s(t), not cumulative-within-
// session mass. The per-turn S/N consumer (#854's SignalNoiseFromAttention reads AttendedMass)
// must run BEFORE ResetAttention.
func (a *AttentionAccumulator) ObserveContext(c *Context) {
	a.Observe(c.liveSpanMasses())
}

// liveSpanMasses snapshots each live (resident, non-held) segment's current per-turn Attended
// mass, keyed by span id — the input ObserveContext folds at a turn boundary. Held spans are
// excluded (their mass is already zeroed by evict and they can no longer be attended to).
func (c *Context) liveSpanMasses() map[string]float64 {
	m := make(map[string]float64, len(c.segs))
	for _, s := range c.segs {
		if s.Held || s.Len == 0 {
			continue
		}
		m[s.ID] = s.Attended
	}
	return m
}

// ResetAttention clears every live segment's per-turn Attended mass — the turn boundary that
// makes successive ObserveContext folds per-turn (a_s(t)) rather than cumulative-within-session.
// Held spans are already zero. It does not touch the accumulator: the folded EMA/cumulative
// persist; only the raw per-turn scalar the next turn refills is reset.
func (c *Context) ResetAttention() {
	for _, s := range c.segs {
		if !s.Held {
			s.Attended = 0
		}
	}
}
