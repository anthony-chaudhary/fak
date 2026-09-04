package kvmmu

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// attention.go — attribution of post-softmax attention weights to recorded spans.

// AttributeRow routes one post-softmax attention row onto the segment ledger.
// Returns unattributed residual weight for keys not owned by any live segment.
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

// AttentionObserver returns a model.AttnObserver attributing emitted rows onto Context segments.
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

// CloseTurn ends the current turn for rolling accumulators.
// Updates Cumulative and EMA per-span and resets in-flight Attended.
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

// EvictColdest drops coldest-by-attention unpinned spans until targetPositions are freed.
func (c *Context) EvictColdest(targetPositions int) []EvictedSpan {
	before, cost := c.liveMassLedger()
	defer c.gradeRetention(before, cost)

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

// EvictUnderBudget applies EvictColdest only when live residency exceeds budgetPositions.
func (c *Context) EvictUnderBudget(budgetPositions int) []EvictedSpan {
	over := c.kv.Len() - budgetPositions
	if over <= 0 {
		before, cost := c.liveMassLedger()
		c.gradeRetention(before, cost)
		return nil
	}
	return c.EvictColdest(over)
}
