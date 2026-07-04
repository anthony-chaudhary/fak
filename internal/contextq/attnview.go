package contextq

import (
	"sort"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

// attnview.go — issue #2617, Tier-1 ("keep-a-scalar") of the context-views-at-
// marginal-cost design (docs/notes/CONTEXT-VIEWS-AT-MARGINAL-COST-2026-07-04.md).
//
// The attention witness (#851, shipped) already reduces every live span to
// {Attended, EMA, Cumulative, traj[64]} at ZERO forward-pass cost — the turn-fold
// maintains it (internal/kvmmu/attention.go + accumulator.go). Two consumers read
// that signal today: coldest-span eviction (Context.EvictColdest) and a post-hoc
// report. But it is not QUERYABLE — a caller cannot ask "which spans are above θ?"
// and get back a typed span-ref set the way it can request any other contextq view.
//
// This file exposes the side-car as ON-DEMAND VIEWS. A predicate over the per-span
// scalars kvmmu already keeps returns a set of span refs, each stamped with the SAME
// MaterializationVerdict vocabulary every other contextq view uses. It is a READ
// over the O(spans) table the turn-fold already maintains — NO new attention
// compute, NO AttnObserver install or re-run (the acceptance's zero-forward-pass
// invariant). The refs are score-only handles: the bytes behind a ref fault in
// through the existing recall/contextq page-in gate only when a consumer renders
// them (lazy side-car mode). A resident span is a HIT (its scalar is already in
// hand); an evicted/Held span is a FAULT (its K/V left the cache and must be paged
// back to render); a quarantined source span is a REFUSE for free, inherited from
// the cachemeta taint on its KV entry — the same fail-closed rule GateKVView applies.

// SpanScalar names which per-span attention reduction a view predicate reads. All
// three are kept by the turn-fold at zero marginal cost on kvmmu.Segment:
//
//   - ScalarAttended   a_s(t): the in-flight CURRENT-turn mass (the goal's "attention > θ").
//   - ScalarEMA        recency-decayed rolling mass — "hot now" (the eviction controller's signal).
//   - ScalarCumulative undecayed lifetime mass — "mattered overall" (the post-hoc analyst's signal).
type SpanScalar string

const (
	ScalarAttended   SpanScalar = "attended"
	ScalarEMA        SpanScalar = "ema"
	ScalarCumulative SpanScalar = "cumulative"
)

// read returns the named scalar off a segment. An unknown scalar falls back to the
// per-turn Attended mass (the naive default view).
func (s SpanScalar) read(seg *kvmmu.Segment) float64 {
	switch s {
	case ScalarEMA:
		return seg.EMA
	case ScalarCumulative:
		return seg.Cumulative
	default:
		return seg.Attended
	}
}

// SpanScalarRef is one span a view selected: its logical id, the scalar it matched
// on and that scalar's value, whether its K/V is still resident, and the
// materialization verdict a consumer MUST honor before rendering its bytes. It is
// the queryable analogue of a SliceRef — a score-only handle, never the bytes.
type SpanScalarRef struct {
	SpanID   string                 `json:"span_id"`
	Scalar   SpanScalar             `json:"scalar"`
	Value    float64                `json:"value"`
	Resident bool                   `json:"resident"`
	Verdict  MaterializationVerdict `json:"verdict"`
}

// SpanScalarView reads the per-span attention scalars kvmmu keeps on the driven
// Context and returns the spans that satisfy pred, each stamped with a
// MaterializationVerdict. It does ZERO forward-pass work: it reads only the scalars
// the turn-fold already maintains (Segment.{Attended,EMA,Cumulative}) — no
// AttnObserver is installed or re-run. Held (evicted) spans are included when they
// satisfy the predicate but are stamped FAULT; a quarantined span is stamped REFUSE.
// pred receives the read scalar value AND the segment so a predicate can also gate on
// residency (see DeadWeightView). A nil Context yields no refs. The scan is O(spans);
// the taint join is built once so it is not O(spans × entries).
func SpanScalarView(c *kvmmu.Context, scalar SpanScalar, pred func(value float64, seg *kvmmu.Segment) bool) []SpanScalarRef {
	if c == nil {
		return nil
	}
	taint := taintIndex(c)
	var refs []SpanScalarRef
	for _, seg := range c.Segments() {
		if seg == nil {
			continue
		}
		v := scalar.read(seg)
		if !pred(v, seg) {
			continue
		}
		refs = append(refs, spanScalarRef(seg, scalar, v, taint))
	}
	return refs
}

// ThresholdView returns every span whose `scalar` mass exceeds theta — the goal's
// naive "attention > θ" example, generalized to any of the three scalars. Refs come
// back strongest-first so the load-bearing spans read first.
func ThresholdView(c *kvmmu.Context, scalar SpanScalar, theta float64) []SpanScalarRef {
	refs := SpanScalarView(c, scalar, func(v float64, _ *kvmmu.Segment) bool { return v > theta })
	sort.SliceStable(refs, func(i, j int) bool { return refs[i].Value > refs[j].Value })
	return refs
}

// HottestView returns the top-n spans by `scalar`, descending — generalizing
// report.go's Hottest (over Cumulative) and hot-now (over EMA). Only spans with
// positive mass are candidates; n <= 0 returns all of them.
func HottestView(c *kvmmu.Context, scalar SpanScalar, n int) []SpanScalarRef {
	refs := SpanScalarView(c, scalar, func(v float64, _ *kvmmu.Segment) bool { return v > 0 })
	sort.SliceStable(refs, func(i, j int) bool { return refs[i].Value > refs[j].Value })
	if n > 0 && len(refs) > n {
		refs = refs[:n]
	}
	return refs
}

// DeadWeightView returns the RESIDENT spans whose `scalar` mass is at or below theta
// — cold-but-still-in-cache spans, the eviction-candidate set report.go's DeadWeight
// names. Held spans are excluded (already gone). A pinned span is still reported: it
// is dead weight the controller is forbidden to drop, which is exactly what a caller
// asking "what is idle but resident?" wants surfaced. Refs come back coldest-first.
func DeadWeightView(c *kvmmu.Context, scalar SpanScalar, theta float64) []SpanScalarRef {
	refs := SpanScalarView(c, scalar, func(v float64, seg *kvmmu.Segment) bool {
		return !seg.Held && seg.Len > 0 && v <= theta
	})
	sort.SliceStable(refs, func(i, j int) bool { return refs[i].Value < refs[j].Value })
	return refs
}

// taintIndex maps each tracked KV entry's identity to its taint label, so a span's
// sealed/quarantined status is a single map read rather than a rescan of Entries()
// per span. Spans added via Append carry no KV identity and simply miss the map.
func taintIndex(c *kvmmu.Context) map[cachemeta.EntryID]abi.TaintLabel {
	entries := c.Entries()
	if len(entries) == 0 {
		return nil
	}
	idx := make(map[cachemeta.EntryID]abi.TaintLabel, len(entries))
	for _, e := range entries {
		idx[e.ID] = e.Security.Taint
	}
	return idx
}

// spanScalarRef stamps one selected span with a MaterializationVerdict. The priority
// is REFUSE (a quarantined source span is never served, resident or not) > FAULT (a
// Held span's K/V left the cache and must page in to render) > HIT (a resident span:
// its scalar is already in hand, no page-in needed to answer the view).
func spanScalarRef(seg *kvmmu.Segment, scalar SpanScalar, v float64, taint map[cachemeta.EntryID]abi.TaintLabel) SpanScalarRef {
	resident := !seg.Held && seg.Len > 0
	var kind MaterializationKind
	var reason string
	switch {
	case taint != nil && taint[seg.KV] == abi.TaintQuarantined:
		kind, reason = MaterializationRefuse, "span_sealed_quarantined"
	case resident:
		kind, reason = MaterializationHit, "span_scalar_resident"
	default:
		kind, reason = MaterializationFault, "span_scalar_held_page_in"
	}
	return SpanScalarRef{
		SpanID:   seg.ID,
		Scalar:   scalar,
		Value:    v,
		Resident: resident,
		Verdict: MaterializationVerdict{
			Kind:   kind,
			Reason: reason,
			Step:   seg.From,
			ViewID: "view-attn-" + string(scalar) + "-" + spanTag(seg),
			Entry:  seg.KV,
		},
	}
}

// spanTag is a stable, legible tag for a span's view id: its logical id, or its
// first cache position when the id is empty.
func spanTag(seg *kvmmu.Segment) string {
	if seg.ID != "" {
		return seg.ID
	}
	return "pos" + strconv.Itoa(seg.From)
}
