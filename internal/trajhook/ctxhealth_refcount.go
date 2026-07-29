package trajhook

// ctxhealth_refcount.go — issue #3096: the DETERMINISTIC shed-span refcount
// detector, the context use-after-free signal ctxhealth.go's verdict fold
// anticipates ("as the #3096 span refcount reach the corpus rows, they fold
// into the same verdict here").
//
// "Was the shed content pure value or load-bearing?" is a reference-counting
// question. A context span is safe to shed iff NO LATER turn references it. The
// substrate already exists — spans carry content-addressed ids and a restore
// tombstone (internal/gateway/ctxspans.go, ctxrestore.go) — so this is a pure
// fold over the corpus, no model in the loop: shed a span, then count the later
// turns that reference it (re-read the path, cite the id, pin it). A shed span
// whose refcount ends > 0 is a context USE-AFTER-FREE — the deterministic
// signature of the model getting confused after a trim dropped something still
// live.
//
// It reads the shed/reference events off the OPEN producer channel
// (trajectory.Turn.Labels) rather than a new schema field, so the corpus shape
// is unchanged and a producer (the gateway compaction path) opts a session in by
// stamping the three keys below. Absent those labels, every corpus is a clean
// no-op on this axis — a trace that sheds nothing flags nothing.

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/trajectory"

	"github.com/anthony-chaudhary/fak/internal/numfmt"
)

// GroupByTrace buckets a flat corpus into per-trace turn slices, returning the trace ids
// in FIRST-SEEN order alongside the map. The order slice is the point: ranging a map is
// nondeterministic, so any fold that walks traces needs a stable sequence, and first-seen
// preserves the corpus's own narrative order for a reader.
//
// Turns inside each bucket keep their corpus-relative order; this does NOT sort them by
// Seq. Callers that need call order sort their own bucket, because they differ on whether
// the sort may mutate the grouped slice (`fak traj report` sorts in place; the refcount
// fold sorts a copy). Grouping is the shared part; ordering is not.
//
// Exported because it is the shared preamble of every per-trace fold: this refcount
// detector and `fak traj report` carried byte-identical private copies of the loop.
//
// An empty corpus yields a NIL order (the map is always non-nil). That is deliberate:
// the report caller returned nil here before and only ever ranges it, so nil keeps every
// existing caller byte-identical rather than trading one allocation for a behaviour change.
func GroupByTrace(corpus []trajectory.Turn) (order []string, byTrace map[string][]trajectory.Turn) {
	byTrace = map[string][]trajectory.Turn{}
	for _, t := range corpus {
		if _, seen := byTrace[t.TraceID]; !seen {
			order = append(order, t.TraceID)
		}
		byTrace[t.TraceID] = append(byTrace[t.TraceID], t)
	}
	return order, byTrace
}

const (
	// LabelUseAfterFree is the closed signal a shed-then-referenced span emits: the
	// trim dropped a span a later turn still needed. It is this detector's own flag
	// (the issue names it verbatim), distinct from the trajctl HEALTHY/STALL/DRIFT
	// vocabulary the ctxhealth verdict fold speaks — that fold is #3098's job.
	LabelUseAfterFree = "USE_AFTER_FREE"

	// LabelShedSpan is the corpus label whose value is the content-addressed id of a
	// span SHED on this turn (the compaction/tombstone event). A turn with no such
	// label sheds nothing.
	LabelShedSpan = "ctx_shed"
	// LabelRefSpan is the corpus label whose value is the id of a span this turn
	// REFERENCES — the re-read/cite/pin event whose id is matched against the shed
	// set. A reference at a turn AFTER the span's shed is what bumps the refcount.
	LabelRefSpan = "ctx_ref"
	// LabelRefKind is the OPTIONAL reference kind carried alongside LabelRefSpan:
	// "reread" (the path was read again), "cited" (the id was cited), or "pin" (the
	// span was pinned). Recorded in the finding's reason; it never changes the count.
	LabelRefKind = "ctx_ref_kind"
)

// ShedSpanStat is the deterministic refcount verdict for one shed span within a
// trace: the id, the turn it was shed on, and how many LATER turns referenced it.
// UseAfterFree is the fold (RefCount > 0) — the one bit that says the trim dropped
// a still-live span. RefSeqs/RefKinds carry the witnessing later references, sorted
// for a stable, legible reason.
type ShedSpanStat struct {
	TraceID      string   `json:"trace_id"`
	SpanID       string   `json:"span_id"`
	ShedSeq      int      `json:"shed_seq"`
	RefCount     int      `json:"ref_count"`
	RefSeqs      []int    `json:"ref_seqs,omitempty"`
	RefKinds     []string `json:"ref_kinds,omitempty"`
	UseAfterFree bool     `json:"use_after_free"`
}

// ShedSpanRefcounts is the pure detector core: for every span SHED in the corpus,
// count the turns that reference it AFTER its shed turn (a reference at or before
// the shed is a legitimate resident read and does not count). It is total and
// deterministic — grouped by trace, each trace processed in seq order, output in
// lexical trace-id then shed-seq then span-id order — so the same corpus always
// yields the same stats regardless of the input slice's order. A span shed more
// than once binds to its FIRST shed (the earliest point after which any reference
// is use-after-free). A corpus that sheds nothing returns an empty slice.
func ShedSpanRefcounts(corpus []trajectory.Turn) []ShedSpanStat {
	order, byTrace := GroupByTrace(corpus)

	var out []ShedSpanStat
	for _, id := range order {
		turns := append([]trajectory.Turn(nil), byTrace[id]...)
		sort.SliceStable(turns, func(i, j int) bool { return turns[i].Seq < turns[j].Seq })

		// First shed seq per span id in this trace (earliest wins).
		shedSeq := map[string]int{}
		shedOrder := []string{} // shed ids in first-shed order (input for stable sort)
		for _, t := range turns {
			s := t.Labels[LabelShedSpan]
			if s == "" {
				continue
			}
			if _, ok := shedSeq[s]; !ok {
				shedSeq[s] = t.Seq
				shedOrder = append(shedOrder, s)
			}
		}
		if len(shedSeq) == 0 {
			continue // this trace sheds nothing — a clean no-op
		}

		// Count references that land STRICTLY AFTER the span's shed turn.
		refSeqs := map[string][]int{}
		refKinds := map[string]map[string]bool{}
		for _, t := range turns {
			r := t.Labels[LabelRefSpan]
			if r == "" {
				continue
			}
			shed, ok := shedSeq[r]
			if !ok || t.Seq <= shed {
				continue // never shed, or referenced while still resident (not a UAF)
			}
			refSeqs[r] = append(refSeqs[r], t.Seq)
			if k := t.Labels[LabelRefKind]; k != "" {
				if refKinds[r] == nil {
					refKinds[r] = map[string]bool{}
				}
				refKinds[r][k] = true
			}
		}

		for _, span := range shedOrder {
			seqs := append([]int(nil), refSeqs[span]...)
			sort.Ints(seqs)
			kinds := make([]string, 0, len(refKinds[span]))
			for k := range refKinds[span] {
				kinds = append(kinds, k)
			}
			sort.Strings(kinds)
			out = append(out, ShedSpanStat{
				TraceID:      id,
				SpanID:       span,
				ShedSeq:      shedSeq[span],
				RefCount:     len(seqs),
				RefSeqs:      seqs,
				RefKinds:     kinds,
				UseAfterFree: len(seqs) > 0,
			})
		}
	}

	// Stable output: lexical trace id, then shed seq, then span id — a total order
	// with no map-iteration nondeterminism, invariant to the input slice's order.
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].TraceID != out[j].TraceID {
			return out[i].TraceID < out[j].TraceID
		}
		if out[i].ShedSeq != out[j].ShedSeq {
			return out[i].ShedSeq < out[j].ShedSeq
		}
		return out[i].SpanID < out[j].SpanID
	})
	return out
}

// ShedRefcount returns a CorpusScorer that emits ONE USE_AFTER_FREE finding per
// shed span whose refcount is > 0 — a shed the trim should not have made because a
// later turn still referenced it. A clean session (every shed span at refcount 0)
// emits nothing, exactly like the anomaly scorers stay silent on healthy work. The
// finding is trace-anchored at the SHED turn (Seq = the shed's seq), and its Score
// is the refcount, so Registry.Run surfaces the worst use-after-frees first.
func ShedRefcount() CorpusScorer {
	return func(corpus []trajectory.Turn) []Finding {
		stats := ShedSpanRefcounts(corpus)
		var out []Finding
		for _, s := range stats {
			if !s.UseAfterFree {
				continue
			}
			out = append(out, Finding{
				Label:   LabelUseAfterFree,
				Score:   float64(s.RefCount),
				TraceID: s.TraceID,
				Seq:     s.ShedSeq,
				Reason:  useAfterFreeReason(s),
				Related: s.SpanID,
			})
		}
		return out
	}
}

// useAfterFreeReason renders the deterministic witness sentence for one flagged
// span: which span, when it was shed, and the later turns (and kinds) that kept
// referencing it.
func useAfterFreeReason(s ShedSpanStat) string {
	reason := "shed span " + truncate(s.SpanID, 24) + " (shed at turn " + itoa(s.ShedSeq) +
		") referenced by " + itoa(s.RefCount) + " later turn"
	if s.RefCount != 1 {
		reason += "s"
	}
	if len(s.RefSeqs) > 0 {
		reason += " (turn" + numfmt.PluralSuffix(len(s.RefSeqs)) + " " + joinInts(s.RefSeqs) + ")"
	}
	if len(s.RefKinds) > 0 {
		reason += " via " + joinStrings(s.RefKinds)
	}
	return reason
}

func joinInts(xs []int) string {
	out := ""
	for i, x := range xs {
		if i > 0 {
			out += ","
		}
		out += itoa(x)
	}
	return out
}

func joinStrings(xs []string) string {
	out := ""
	for i, x := range xs {
		if i > 0 {
			out += ","
		}
		out += x
	}
	return out
}
