package kvmmu

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// attention.go — issue #853, rung 2 of the attention-witness epic (#851).
//
// Rung 1 (#852, internal/model/attn_observer.go) emits the post-softmax attention
// weights at the softmax seam as a stream of per-(layer, query, head) rows, each row a
// set of normalized weights over ABSOLUTE key positions. This file attributes that
// stream to SPANS: a weight on key position p is accumulated onto the Segment whose
// recorded [From, From+Len) range owns p. The span-to-KV map already exists — it is the
// same ledger the quarantine/eviction path renumbers — so attribution is a pure read of
// the From/Len partition the bridge already maintains.
//
// This produces the witnessed a_s term #851's S/N formula needs: the attention mass a
// span actually received, rather than the inferred lexical-overlap hit ctxplan's rung-0
// path guesses (the consumer is #854's SignalNoiseFromAttention).
//
// CONSERVATION. The live segments partition the cache positions [0, kv.Len()) with no
// gaps and no overlap (Append lays each new span at From == kv.Len(); evict renumbers
// survivors to close the gap). So every key position a row names is owned by exactly one
// live segment, and the sum of all segments' attributed mass equals the total emitted
// mass — no weight lost, none double-counted. The one exception is a position owned by
// NO live segment (a weight on an evicted/held span, which cannot happen on the
// write-time path but is possible if an observer is fed a stale row): that residual is
// counted in Unattributed rather than silently dropped, so conservation is auditable.
//
// EVICTION COHERENCE. evict() zeroes the evicted span's Attended (its mass is gone — the
// model can no longer attend to a span that is not in the cache) and leaves every
// survivor's Attended untouched. Attended is mass, not a position, so the From
// renumbering evict already does does not touch it: the surviving span keeps exactly the
// mass it accumulated.

// AttributeRow routes one post-softmax attention row onto the segment ledger: for each
// key position keyPositions[i], it adds weights[i] to the Attended accumulator of the
// live segment whose [From, From+Len) range contains that position. Any weight on a
// position owned by no live segment is returned as the unattributed residual (it is NOT
// added to any span), so a caller can assert conservation (sum of span deltas + residual
// == sum of weights).
//
// Efficiency: keyPositions is ascending and contiguous on the witness paths (decode /
// prefill emit kp[i] = j0+i), and the live segments are sorted by From, so a single
// linear cursor over the small live-segment ledger attributes the whole row in
// O(positions + spans) with no per-position search.
func (c *Context) AttributeRow(keyPositions []int, weights []float32) (unattributed float64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	n := len(keyPositions)
	if len(weights) < n {
		n = len(weights)
	}
	if n == 0 {
		return 0
	}
	live := c.liveSegsByFrom()
	si := 0
	for i := 0; i < n; i++ {
		p := keyPositions[i]
		w := float64(weights[i])
		// Advance the cursor to the first live segment that could contain p. Because
		// both the row's positions and the live segments are ascending, the cursor only
		// moves forward across the whole row.
		for si < len(live) && live[si].From+live[si].Len <= p {
			si++
		}
		if si < len(live) && p >= live[si].From && p < live[si].From+live[si].Len {
			live[si].Attended += w
			continue
		}
		unattributed += w
	}
	return unattributed
}

// liveSegsByFrom returns the not-yet-held segments in From order. The ledger is appended
// in From order and evict renumbers survivors in place, so c.segs is already ascending
// by From among live entries; this just filters out the held (Len==0) ones. The slice
// header is fresh but the *Segment pointers alias the ledger — AttributeRow mutates
// Attended through them by design.
func (c *Context) liveSegsByFrom() []*Segment {
	live := make([]*Segment, 0, len(c.segs))
	for _, s := range c.segs {
		if !s.Held && s.Len > 0 {
			live = append(live, s)
		}
	}
	return live
}

// AttentionObserver returns a model.AttnObserver that attributes every emitted row onto
// this Context's span ledger via AttributeRow. Install it on the model for the duration
// of a forward pass (model.SetAttnObserver) to accumulate witnessed per-span attention
// mass across the turn; the layer/query/head dimensions are summed into the span's
// scalar Attended mass (the per-(layer,head) breakdown is rung 4's concern, #855). The
// unattributed residual is discarded here; use AttributeRow directly when a caller wants
// to assert conservation.
//
// Concurrency: model.AttnObserver may fire from parallel attention workers, so the
// returned observer must not be installed while the ledger is being mutated by the
// eviction path on another goroutine. The witness paths emit synchronously from within
// the forward pass; attribute under the same single-flight discipline the rest of the
// bridge assumes (one turn's prefill/decode at a time per Context).
func (c *Context) AttentionObserver() model.AttnObserver {
	return func(_, _, _ int, keyPositions []int, weights []float32) {
		c.AttributeRow(keyPositions, weights)
	}
}

// AttendedMass returns the total witnessed attention mass currently attributed across
// all live segments — the denominator for normalizing each span's a_s into [0,1].
func (c *Context) AttendedMass() float64 {
	var total float64
	for _, s := range c.segs {
		if !s.Held {
			total += s.Attended
		}
	}
	return total
}

// trajCap bounds the per-span trajectory ring: each live segment remembers at most the
// last trajCap per-turn masses {a_s(t)}, so memory is O(trajCap) per span regardless of
// how many turns a session runs. A span hot far in the past still shows a high Cumulative
// and (with lambda<1) a decayed EMA; only the fine-grained "which exact turn" trail is
// windowed. 64 turns is a generous recent-history window for the post-hoc analyst.
const trajCap = 64

// TrajCapForTest exposes the trajectory ring cap to the external _test package so the
// bounded-ring acceptance test can assert the window size without hardcoding the constant.
func TrajCapForTest() int { return trajCap }

// CloseTurn ends the current turn for the rolling accumulators (#855, rung 4): for every
// live segment it folds the turn's accumulated a_s(t) (the in-flight Attended) into the
// two reductions of the SAME stream, appends it to the bounded trajectory ring, then
// resets Attended to 0 for the next turn. lambda is the ONLY knob separating the two
// consumers:
//
//	Cumulative += a_s(t)                 (undecayed — "mattered overall", the audit signal)
//	EMA         = lambda*EMA + a_s(t)    (recency-decayed — "hot now", the eviction signal)
//
// One accumulator, two lambda: pass lambda<1 for the real-time heavy-hitter controller
// (#856 evicts the lowest EMA); pass lambda==1 and the EMA recurrence is exactly the
// cumulative sum, so a post-hoc analyst reading EMA at lambda=1 sees Cumulative (the
// identity the issue requires). Held segments are skipped — an evicted span has no turn
// to close. Deterministic given a fixed per-turn mass stream.
//
// Call once per turn boundary, AFTER the turn's attention has been attributed (the
// AttributeRow / observer calls) and BEFORE the next turn's attribution begins.
func (c *Context) CloseTurn(lambda float64) {
	for _, s := range c.segs {
		if s.Held {
			continue
		}
		a := s.Attended
		s.Cumulative += a
		s.EMA = lambda*s.EMA + a
		s.pushTraj(a)
		s.Attended = 0
	}
}

// pushTraj appends one per-turn mass to the bounded trajectory ring, overwriting the
// oldest entry once the ring is full (trajCap turns of history). O(1) per turn.
func (s *Segment) pushTraj(a float64) {
	if cap(s.traj) < trajCap {
		s.traj = make([]float64, trajCap)
	}
	s.traj[s.trajHead] = a
	s.trajHead = (s.trajHead + 1) % trajCap
	if s.trajLen < trajCap {
		s.trajLen++
	}
}

// Trajectory returns the segment's recent per-turn masses {a_s(t)} in chronological
// order (oldest retained turn first, most recent last) — the reconstruction of WHEN the
// span was hot, for the post-hoc analyst. The slice is a fresh copy (the caller may keep
// it); it holds at most trajCap entries (older turns have aged out of the bounded ring).
// A span with no closed turns returns an empty slice.
func (c *Context) Trajectory(id string) []float64 {
	for _, s := range c.segs {
		if s.ID == id {
			return s.trajectory()
		}
	}
	return nil
}

// trajectory unrolls the ring into chronological order (oldest first). When the ring has
// not yet filled, entries occupy [0, trajLen) and trajHead == trajLen, so the unroll is
// the identity; once full, the oldest entry sits at trajHead and we read wrapping around.
func (s *Segment) trajectory() []float64 {
	out := make([]float64, s.trajLen)
	if s.trajLen == 0 {
		return out
	}
	start := s.trajHead - s.trajLen
	if start < 0 {
		start += cap(s.traj)
	}
	for i := 0; i < s.trajLen; i++ {
		out[i] = s.traj[(start+i)%cap(s.traj)]
	}
	return out
}

// EvictedSpan records one span dropped by the attention-informed controller (#856): its
// id, the EMA it carried when selected, and the number of cache positions removed.
type EvictedSpan struct {
	ID        string
	EMA       float64
	Positions int
}

// EvictColdest drops the coldest-by-attention unpinned spans until at least
// targetPositions cache positions have been freed, or until no unpinned live spans
// remain. Pinned spans are excluded from the candidate set entirely.
func (c *Context) EvictColdest(targetPositions int) []EvictedSpan {
	if targetPositions <= 0 {
		return nil
	}
	cand := make([]*Segment, 0, len(c.segs))
	for _, s := range c.segs {
		if s.Held || s.Len == 0 || s.Pinned {
			continue
		}
		cand = append(cand, s)
	}
	sort.SliceStable(cand, func(i, j int) bool {
		a, b := cand[i], cand[j]
		if a.EMA != b.EMA {
			return a.EMA < b.EMA
		}
		if a.Cumulative != b.Cumulative {
			return a.Cumulative < b.Cumulative
		}
		return a.From < b.From
	})
	var out []EvictedSpan
	freed := 0
	for _, s := range cand {
		if freed >= targetPositions {
			break
		}
		ema, id := s.EMA, s.ID
		n := c.evict(s)
		out = append(out, EvictedSpan{ID: id, EMA: ema, Positions: n})
		freed += n
	}
	return out
}

// EvictUnderBudget applies EvictColdest only when live residency exceeds the budget.
func (c *Context) EvictUnderBudget(budgetPositions int) []EvictedSpan {
	over := c.kv.Len() - budgetPositions
	if over <= 0 {
		return nil
	}
	return c.EvictColdest(over)
}
