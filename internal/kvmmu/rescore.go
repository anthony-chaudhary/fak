package kvmmu

// rescore.go — cheap layer-0 attention re-score over KV-resident candidates.

import "fmt"

// reScorer is the additive backend seam for the cheap re-score, reachable by type
// assertion exactly like the CanEvict verdict: the in-process model.Session backend
// implements it over Kraw; a backend that does not is reported as a typed refusal,
// never a zero score.
type reScorer interface {
	ReScoreSpans(probe []int, spans [][2]int) ([]float64, error)
}

// SpanScore is one candidate's relevance to a probe query, as adjudicated by the
// cheap Kraw re-score. Score is the share of the probe's layer-0 candidate-directed
// attention mass this span received; scores across one ReScore call sum to ~1.0.
type SpanScore struct {
	ID    string
	Score float64
}

// ReScore ranks the named KV-resident segments by their relevance to a NEW probe
// query (token ids), without a forward pass and without touching the cached bytes.
//
// Every candidate must name a live (non-evicted, non-empty) segment: a held or
// unknown id fails the whole call closed rather than silently scoring the survivor
// subset — the caller's Tier-1 narrowing decided that set, so a hole in it is a
// caller bug, not a ranking. Results are parallel to candidateIDs.
func (c *Context) ReScore(probe []int, candidateIDs []string) ([]SpanScore, error) {
	if len(candidateIDs) == 0 {
		return nil, fmt.Errorf("kvmmu: ReScore: no candidate ids")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	rs, ok := c.kv.(reScorer)
	if !ok {
		return nil, fmt.Errorf("kvmmu: ReScore: backend %T does not expose the Kraw re-score seam", c.kv)
	}
	spans := make([][2]int, 0, len(candidateIDs))
	for _, id := range candidateIDs {
		seg := c.liveSegment(id)
		if seg == nil {
			return nil, fmt.Errorf("kvmmu: ReScore: candidate %q is not KV-resident", id)
		}
		spans = append(spans, [2]int{seg.From, seg.Len})
	}
	rel, err := rs.ReScoreSpans(probe, spans)
	if err != nil {
		return nil, err
	}
	if len(rel) != len(candidateIDs) {
		return nil, fmt.Errorf("kvmmu: ReScore: backend returned %d scores for %d candidates", len(rel), len(candidateIDs))
	}
	out := make([]SpanScore, len(candidateIDs))
	for i, id := range candidateIDs {
		out[i] = SpanScore{ID: id, Score: rel[i]}
	}
	return out, nil
}

// liveSegment resolves an id to its live (resident, non-empty) segment, nil when the
// id is unknown, evicted, or empty. Caller holds c.mu.
func (c *Context) liveSegment(id string) *Segment {
	for _, s := range c.segs {
		if s.ID == id && !s.Held && s.Len > 0 {
			return s
		}
	}
	return nil
}
