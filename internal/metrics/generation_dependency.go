package metrics

import (
	"sort"
	"strconv"
	"strings"
)

// Cross-generation dependency edges (issue #1655, gen/second-next).
//
// A generation portfolio is not a set of independent columns: options depend on
// each other, and some of those dependencies cross a horizon boundary. "Future bets
// often depend on next-gen seams" — a gen/future option that builds on a gen/now
// seam is the normal, healthy shape, and finishing the near seam is what unblocks the
// far option's later promotion. The dangerous shape is the mirror image: a NEARER
// option that depends on a FARTHER one. That edge is an inversion — near work that
// cannot mature until an option two horizons out matures first — and it is invisible
// on a per-column roadmap, because each column looks fine on its own.
//
// This file is the SHAPE of the metadata and reporting for those edges, expressed as
// a pure, stdlib-only model (no clock, no disk, no default exposure) in the same idiom
// as generation_roadmap.go and generation_learning_agenda.go. It names a closed edge
// vocabulary keyed off the now->future horizon rank, classifies each edge as forward /
// inverted / intra-generation / unknown-horizon, and renders a deterministic report
// that surfaces the inversions (the promotion blockers) with their demotion criterion.
//
// Generation stays orthogonal to three things, and the model is built to keep them
// separate (OrthogonalityNote, shared with the roadmap dashboard, is rendered in the
// report header):
//
//   - PRIORITY — an edge's kind is decided by horizon RANK alone (now->future), never
//     by the priority of either endpoint. An inverted edge is flagged because the near
//     option cannot mature yet, not because it is more or less important than its
//     dependency. A high-priority gen/now item and a low-priority one invert identically.
//   - SHARED TRUNK — both endpoints live on main. An edge is a label-to-label relation
//     over one trunk, not a cross-branch dependency; nothing here implies a branch or a
//     worktree per generation, and the graph reads a single line of history.
//   - RUNTIME FEATURE GATES — an endpoint's horizon says WHEN it is expected to mature,
//     not WHETHER it is exposed at runtime. An edge classifies maturation dependency,
//     not runtime wiring; default exposure remains an explicit, independent feature gate.
//
// Promotion / demotion / invalidating-assumption for this artifact itself:
//   - Promotion evidence: internal/devindex (or the milestone/parent-link ledger) folds
//     real parent/blocked-by edges between gen/*-labelled issues through Classify, and an
//     operator surface renders ReportEdges so an inverted edge is caught at triage instead
//     of at the near option's due date.
//   - Demotion/retirement evidence: retire this model if measurement shows inversions never
//     occur in practice (every real edge is forward or intra, so the classifier is
//     ceremony), OR if the milestone graph grows its own horizon-aware blocked-by view that
//     subsumes it. A single persistent inverted edge that is never acted on is itself the
//     signal to demote the NEAR endpoint to its dependency's horizon (see DemotionCriterion).
//   - Invalidating assumption: that a dependency's honest horizon is a TOTAL order on the
//     four-horizon rank, so "farther depends on nearer" is always healthy and the reverse is
//     always a risk. If real dependencies are cyclic, or if a nearer option can legitimately
//     depend on a farther one that is already de-risked (a spec published ahead of its
//     implementation horizon), then rank-direction is the wrong signal and the classifier
//     must move to an explicit blocked/unblocked witness before it is promoted.

// EdgeKind is the closed classification of a dependency edge by the horizon rank of
// its endpoints. A value outside this set is a bug, not a new kind.
type EdgeKind string

const (
	// EdgeForward is a farther-horizon option depending on a nearer one
	// (rank(from) > rank(to)) — the healthy shape: the near seam, once finished,
	// unblocks the far option's promotion.
	EdgeForward EdgeKind = "forward"
	// EdgeInverted is a nearer-horizon option depending on a farther one
	// (rank(from) < rank(to)) — the promotion blocker: the near option cannot mature
	// until an option further out does. This is the edge the report exists to surface.
	EdgeInverted EdgeKind = "inverted"
	// EdgeIntra is an edge whose endpoints share a horizon — a real dependency, but not
	// a cross-generation one, so it is reported separately and never flagged.
	EdgeIntra EdgeKind = "intra-generation"
	// EdgeUnknown is an edge with an endpoint outside RoadmapGenerations. Classification
	// fails closed on it rather than guessing a rank.
	EdgeUnknown EdgeKind = "unknown-horizon"
)

// EdgeKinds is the ordered, closed edge vocabulary, so a test or a renderer can
// enumerate every classification an edge can take.
var EdgeKinds = []EdgeKind{EdgeForward, EdgeInverted, EdgeIntra, EdgeUnknown}

// Label is the short human label for an edge kind.
func (k EdgeKind) Label() string {
	switch k {
	case EdgeForward:
		return "Forward"
	case EdgeInverted:
		return "Inverted"
	case EdgeIntra:
		return "Intra-generation"
	case EdgeUnknown:
		return "Unknown horizon"
	default:
		return string(k)
	}
}

// DependencyEdge is one dependency between two portfolio options: From depends on To.
// Each endpoint carries the option's identity and its horizon (a stream in
// RoadmapGenerations). It is a plain data snapshot — a caller folds a real
// parent/blocked-by graph into it; this package does not read disk.
type DependencyEdge struct {
	// From is the dependent option's stable key.
	From string `json:"from"`
	// FromStream is the dependent option's horizon (one of RoadmapGenerations).
	FromStream string `json:"from_stream"`
	// To is the depended-on option's stable key.
	To string `json:"to"`
	// ToStream is the depended-on option's horizon (one of RoadmapGenerations).
	ToStream string `json:"to_stream"`
	// Note is an optional one-line rationale carried into the rendered report.
	Note string `json:"note,omitempty"`
}

// horizonRank returns the now->future rank of a stream (now=0 .. future=len-1) and
// whether it is in the closed horizon vocabulary. It is the single source of horizon
// ordering for the classifier — priority never enters here.
func horizonRank(stream string) (int, bool) {
	for i, s := range RoadmapGenerations {
		if s == stream {
			return i, true
		}
	}
	return -1, false
}

// Kind classifies the edge by the horizon rank of its endpoints. Direction is decided
// by rank alone (now->future); the priority of either endpoint is never consulted.
func (e DependencyEdge) Kind() EdgeKind {
	fr, ok1 := horizonRank(e.FromStream)
	tr, ok2 := horizonRank(e.ToStream)
	if !ok1 || !ok2 {
		return EdgeUnknown
	}
	switch {
	case fr == tr:
		return EdgeIntra
	case fr > tr:
		return EdgeForward
	default:
		return EdgeInverted
	}
}

// CrossesGeneration reports whether the edge spans a horizon boundary — true for a
// forward or inverted edge, false for an intra-generation edge. An unknown-horizon
// edge cannot be placed, so it is not counted as crossing.
func (e DependencyEdge) CrossesGeneration() bool {
	k := e.Kind()
	return k == EdgeForward || k == EdgeInverted
}

// DemotionCriterion returns the honest fix an inverted edge implies, and "" for any
// other kind. An inversion means one of two labels is wrong: either To gains promotion
// evidence and matures early, or From must be demoted to To's horizon because it cannot
// honestly mature before what it depends on. Naming the criterion is what keeps a
// standing inversion from silently becoming a hidden priority inversion.
func (e DependencyEdge) DemotionCriterion() string {
	if e.Kind() != EdgeInverted {
		return ""
	}
	return "gen/" + e.FromStream + " option " + e.From + " depends on farther gen/" + e.ToStream +
		" option " + e.To + ": promote " + e.To + " with promotion evidence, else demote " +
		e.From + " to gen/" + e.ToStream + " (it cannot mature before its dependency)"
}

// DependencyReport is the classified snapshot of a dependency graph: edges grouped by
// kind, in EdgeKinds order within each group by (From,To) for determinism.
type DependencyReport struct {
	// Edges is the classified edge list, sorted by kind then endpoints.
	Edges []classifiedEdge `json:"edges"`
}

// classifiedEdge pairs an edge with its computed kind, so the report carries the
// classification alongside the raw edge.
type classifiedEdge struct {
	DependencyEdge
	Kind EdgeKind `json:"kind"`
}

// Classify folds a raw edge set into a deterministic report: each edge tagged with its
// kind and sorted by (kind, from, to). It is pure — the same edges always produce the
// same report, so a test can assert it and an observability surface can mount it.
func Classify(edges []DependencyEdge) DependencyReport {
	out := make([]classifiedEdge, 0, len(edges))
	for _, e := range edges {
		out = append(out, classifiedEdge{DependencyEdge: e, Kind: e.Kind()})
	}
	rankOf := func(k EdgeKind) int {
		for i, kk := range EdgeKinds {
			if kk == k {
				return i
			}
		}
		return len(EdgeKinds)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if ri, rj := rankOf(out[i].Kind), rankOf(out[j].Kind); ri != rj {
			return ri < rj
		}
		if out[i].From != out[j].From {
			return out[i].From < out[j].From
		}
		return out[i].To < out[j].To
	})
	return DependencyReport{Edges: out}
}

// CountByKind rolls the report up per edge kind. Every kind in EdgeKinds is present,
// including the zero ones — an absent kind must read as "0", not vanish.
func (r DependencyReport) CountByKind() map[EdgeKind]int {
	out := make(map[EdgeKind]int, len(EdgeKinds))
	for _, k := range EdgeKinds {
		out[k] = 0
	}
	for _, e := range r.Edges {
		out[e.Kind]++
	}
	return out
}

// Inversions returns just the inverted edges — the promotion blockers an operator
// must act on. It is the answer to the issue's question: which near options are
// blocked on a farther horizon, and what is the demotion criterion for each.
func (r DependencyReport) Inversions() []classifiedEdge {
	var out []classifiedEdge
	for _, e := range r.Edges {
		if e.Kind == EdgeInverted {
			out = append(out, e)
		}
	}
	return out
}

// Render produces a deterministic text report: the orthogonality header, a per-kind
// count line, then each edge with its endpoints and kind, and — for every inversion —
// its demotion criterion. Pure (no clock, no disk), so a test can assert its bytes.
func (r DependencyReport) Render() string {
	var b strings.Builder
	b.WriteString("Cross-generation dependency edges (" + strconv.Itoa(len(r.Edges)) + " edges)\n")
	b.WriteString(OrthogonalityNote)
	b.WriteString("\n\n")

	b.WriteString("  counts:")
	counts := r.CountByKind()
	for _, k := range EdgeKinds {
		b.WriteString(" " + string(k) + "=" + strconv.Itoa(counts[k]))
	}
	b.WriteString("\n\n")

	b.WriteString("  edges:\n")
	if len(r.Edges) == 0 {
		b.WriteString("    - (none)\n")
	}
	for _, e := range r.Edges {
		b.WriteString("    - " + pad(e.Kind.Label(), edgeKindWidth) + " " +
			"gen/" + e.FromStream + ":" + e.From + " -> gen/" + e.ToStream + ":" + e.To)
		if e.Note != "" {
			b.WriteString(" | " + e.Note)
		}
		b.WriteString("\n")
	}

	inv := r.Inversions()
	if len(inv) > 0 {
		b.WriteString("\n  inversions (promotion blockers):\n")
		for _, e := range inv {
			b.WriteString("    - " + e.DemotionCriterion() + "\n")
		}
	}
	return b.String()
}

const edgeKindWidth = 18
