package kvmmu

import "github.com/anthony-chaudhary/fak/internal/model"

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
