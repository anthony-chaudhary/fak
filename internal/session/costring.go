package session

// costring.go — the per-session cost RING (issue #756, epic #748 Pillar 2). The most
// useful column `fak ps` can show is cost-PER-ITERATION: the per-turn token spend that
// spikes ~200x the instant an agent enters a runaway loop. Until this file, nothing in
// the drive state remembered a turn's cost — DebitUsage only decremented the live budget
// (usage.go) and fired the #743 budget observer, so a runaway was visible only as "budget
// drained fast", never as "this turn cost 200x the last one".
//
// THE SHAPE. CostRing is a fixed-size, allocation-free ring of the last CostRingSize turns'
// token cost, carried inline on State so it rides Snapshot (the scheduler / `fak ps` read)
// with no side table and no extra map lookup. It is bounded BY CONSTRUCTION — a fixed array
// plus a head index and a count — so a million-turn session keeps exactly the last N entries
// and never grows. DebitUsage pushes one TurnCost per debited turn; Snapshot carries the ring
// out; CostSummary folds it into the render-ready numbers (latest, previous, delta, and the
// spike ratio that is the runaway tell).
//
// FENCE: advisory, never trust. The ring is a cost/observability lever, exactly like Intent
// and Goal — a renderer reads it, no decision GATES on it. The zero ring (an unseen or
// never-debited session) is the safe default: empty, summarizing to all-zero, marshaling
// byte-identically to a pre-ring State via the omitzero tag.

// CostRingSize is the number of recent turns the per-session cost ring retains. It is small
// on purpose: `fak ps` renders the latest cost and a short trailing window, and the ring is
// hot-path-adjacent (pushed under the table lock on every debited turn), so a fixed handful
// of entries keeps the State copy Snapshot makes cheap. Eight turns is enough to SEE a spike
// (the last few "normal" turns next to the runaway one) without carrying a transcript.
const CostRingSize = 8

// TurnCost is one debited turn's token cost — the output tokens it emitted and the
// context/prompt tokens it had to read. These are exactly the two axes DebitUsage already
// receives (Usage), recorded so a renderer can show cost-per-iteration and a supervisor can
// see the per-turn shape of a runaway. Both are the turn's reported usage, not a running
// total.
type TurnCost struct {
	OutputTokens  int `json:"output_tokens"`
	ContextTokens int `json:"context_tokens"`
}

// total is the turn's combined token cost — the single scalar a spike ratio is computed
// over, so a runaway that explodes on either axis (a giant context re-read or a giant
// output) is caught the same way.
func (c TurnCost) total() int { return c.OutputTokens + c.ContextTokens }

// CostRing is the bounded per-session record of the last CostRingSize turns' cost. It is a
// fixed array plus a head cursor and a live count, so it never allocates and never grows:
// the (head+CostRingSize-1) slot is the most recent push, older entries trail backward, and
// once count reaches CostRingSize the oldest is overwritten in place. Carried inline on State
// (omitzero), it rides Snapshot with no side table. The zero CostRing is a valid empty ring.
type CostRing struct {
	Turns [CostRingSize]TurnCost `json:"turns"`
	Head  int                    `json:"head"`  // index of the NEXT write (the oldest entry once full)
	Count int                    `json:"count"` // live entries, capped at CostRingSize
}

// IsZero reports whether the ring holds no recorded turns — the safe default a renderer reads
// as "no cost history yet". It drives the `omitzero` JSON tag so a session that has never been
// debited marshals byte-identically to a pre-ring State.
func (r CostRing) IsZero() bool { return r.Count == 0 }

// push appends one turn's cost, overwriting the oldest entry when the ring is full. It is
// value-semantic (returns the new ring) so DebitUsage mutates its local State copy and writes
// it through putLocked exactly like every other field — no pointer aliasing into the table's
// stored record. O(1), no allocation.
func (r CostRing) push(c TurnCost) CostRing {
	r.Turns[r.Head] = c
	r.Head = (r.Head + 1) % CostRingSize
	if r.Count < CostRingSize {
		r.Count++
	}
	return r
}

// at returns the i-th most recent entry (0 == latest, 1 == the one before it, ...) and whether
// it exists. It walks backward from the write head over the live count, so a half-full ring
// reports only the entries it actually holds — never a zero-value slot that was never written.
func (r CostRing) at(i int) (TurnCost, bool) {
	if i < 0 || i >= r.Count {
		return TurnCost{}, false
	}
	idx := (r.Head - 1 - i + 2*CostRingSize) % CostRingSize
	return r.Turns[idx], true
}

// CostSummary is the render-ready fold of a session's cost ring — the numbers `fak ps` shows
// in a cost-per-iteration column. Latest/Previous are the two most recent turns' combined cost;
// Delta is Latest-Previous (a positive jump is a climbing cost); SpikeRatio is Latest over the
// MAX of the prior window (the runaway tell: a 200x loop reads ~200.0, a steady session reads
// ~1.0). Count is how many turns the ring actually holds. All zero for an empty ring, so a
// fresh session renders blank rather than a divide-by-zero.
type CostSummary struct {
	Latest     int     `json:"latest"`      // most recent turn's total token cost
	Previous   int     `json:"previous"`    // the turn before it (0 if only one recorded)
	Delta      int     `json:"delta"`       // Latest - Previous
	SpikeRatio float64 `json:"spike_ratio"` // Latest / max(prior window); 0 when there is no prior turn
	Count      int     `json:"count"`       // live entries in the ring
}

// CostSummary folds the ring into the render-ready summary. SpikeRatio divides the latest
// turn's total by the LARGEST total among the prior entries (not the immediately-previous
// one) so a single normal turn wedged between two runaway turns cannot mask the spike; it is
// 0 when no prior turn exists (a session's first debit cannot be a spike). The fold is pure and
// reads only the ring, so it is reproducible and table-testable.
func (r CostRing) CostSummary() CostSummary {
	if r.Count == 0 {
		return CostSummary{}
	}
	latest, _ := r.at(0)
	s := CostSummary{Latest: latest.total(), Count: r.Count}
	if r.Count == 1 {
		return s
	}
	prev, _ := r.at(1)
	s.Previous = prev.total()
	s.Delta = s.Latest - s.Previous
	priorMax := 0
	for i := 1; i < r.Count; i++ {
		if c, ok := r.at(i); ok && c.total() > priorMax {
			priorMax = c.total()
		}
	}
	if priorMax > 0 {
		s.SpikeRatio = float64(s.Latest) / float64(priorMax)
	}
	return s
}
