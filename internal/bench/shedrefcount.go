// shedrefcount.go — the shed-span reference-count detector (issue #3096, epic
// #3095, the context-engineering deterministic-detector spine).
//
// "Was the shed content pure value or load-bearing?" is a reference-counting
// question. A context span is safe to shed IFF no LATER turn references it —
// re-reads the file it held, cites its id, or pins it. Shedding a span that a
// later turn still reaches for is a context USE-AFTER-FREE: the deterministic
// signature of the trim dropping something still live, the shape the model then
// gets confused after. This file makes that signal COUNTABLE with no model in
// the loop.
//
// The mechanism it models is the guard's real one: `internal/gateway/ctxspans.go`
// enumerates the content-addressed spans a trace has dropped, and
// `internal/gateway/ctxrestore.go` stashes each dropped originating-task span
// under its sha256 handle so a later turn can page it back in. Those handles are
// exactly the ids a re-read / cite / pin would name. No refcount is tracked over
// them today, so a shed-then-referenced span goes undetected. This detector is
// the deterministic base the siblings (#3097 BehaviorLens port, #3098 trajhook
// scorer, #3099 live guard surface, #3100 offline benchmark) build the
// confusion/thrash verdict on top of — it is the refcount fold ONLY, not those.
//
// The fold is order-independent and pure: a span's shed turn is the FIRST turn
// that dropped it; its refcount is the number of LATER turns (strictly greater
// than the shed turn) that reference its content by any of the three kinds. A
// reference BEFORE the shed is normal live use and is NOT counted — that is the
// guard against over-flagging (a named confusion risk of the issue). A reference
// to a span that was never shed is not a use-after-free and is ignored.
//
// PROVENANCE (net-true doctrine, docs/standards/net-true-value.md): the checked-in
// transcripts are labeled hermetic FIXTURES, not a live capture. What this
// witnesses is the MEASUREMENT and the discrimination — a known shed-then-
// reference lands refcount > 0 and fires USE_AFTER_FREE per span, and a clean
// session lands every shed span at refcount 0. The OBSERVED seam
// (BuildShedRefcountReportFor) folds the SAME shape from a live gateway capture
// (ctxspans/ctxrestore shed ids × the turn's re-read/cite/pin log) to promote the
// signal toward gen/now. See the report's assumptions / promotion fields.
//
// Re-run: `go test ./internal/bench -run ShedRefcount` (regenerate the golden
// with UPDATE_GOLDEN=1).
package bench

import (
	"encoding/json"
	"sort"
)

// ShedRefcountSchema versions the detector artifact.
const ShedRefcountSchema = "fak.shedrefcount.v1"

// The closed vocabulary of the ways a later turn can reference a shed span's
// content — the three the issue names. A shed event drops the span; a reference
// event reaches for it.
const (
	EventShed = "shed"   // a compaction/trim dropped the span (the ctxrestore tombstone)
	RefReread = "reread" // a later turn re-read the path/file the span held
	RefCite   = "cite"   // a later turn cited the span's content-addressed id
	RefPin    = "pin"    // a later turn pinned the span
)

// SignalUseAfterFree is the deterministic verdict a shed span with refcount > 0
// raises: the trim dropped something a later turn still referenced.
const SignalUseAfterFree = "USE_AFTER_FREE"

// refKinds is the closed reference set, in stable order — the discovery edge for
// per-kind tallies and the over-count guard (only these kinds count).
var refKinds = []string{RefReread, RefCite, RefPin}

// CtxSpanEvent is one entry of a session's context-span lifecycle: a span SHED
// (a trim dropped it) or a later REFERENCE to a span's content (reread | cite |
// pin). Turn is the monotonic turn index the event happened on; SpanID is the
// content-addressed span handle (the same sha256 id ctxspans enumerates and
// ctxrestore keys on).
type CtxSpanEvent struct {
	Turn   int    `json:"turn"`
	Kind   string `json:"kind"`
	SpanID string `json:"span_id"`
}

// ShedSpanTranscript is a named ordered sequence of context-span events — the
// unit the detector folds over. The default fixtures are hermetic; the OBSERVED
// seam feeds the same shape from a live gateway capture.
type ShedSpanTranscript struct {
	Name   string         `json:"name"`
	Events []CtxSpanEvent `json:"events"`
}

// SpanRefcount is one shed span's tally: the turn it was shed and how many LATER
// turns referenced its content, split by kind. UseAfterFree is the fold
// Refcount > 0 — a still-live span the trim dropped.
type SpanRefcount struct {
	SpanID       string         `json:"span_id"`
	ShedTurn     int            `json:"shed_turn"`
	Refcount     int            `json:"refcount"`
	RefsByKind   map[string]int `json:"refs_by_kind,omitempty"`
	RefTurns     []int          `json:"ref_turns,omitempty"`
	UseAfterFree bool           `json:"use_after_free"`
}

// ShedRefcountReport is the folded detector artifact: every shed span's refcount,
// the count of use-after-free spans, and the deterministic signal. Clean is the
// fold "no shed span was referenced after it was shed".
type ShedRefcountReport struct {
	Schema      string         `json:"schema"`
	Provenance  Provenance     `json:"provenance"`
	Transcript  string         `json:"transcript"`
	TotalEvents int            `json:"total_events"`
	ShedSpans   int            `json:"shed_spans"`
	Spans       []SpanRefcount `json:"spans"`
	// UseAfterFreeCount is the number of shed spans with refcount > 0. Signal is
	// SignalUseAfterFree when that count is > 0, else empty — the one-bit verdict.
	UseAfterFreeCount int    `json:"use_after_free_count"`
	Signal            string `json:"signal,omitempty"`
	// Clean is UseAfterFreeCount == 0: every shed span was safe to shed.
	Clean               bool     `json:"clean"`
	Assumptions         []string `json:"assumptions"`
	Promotion           string   `json:"promotion"`
	DemotionRetirement  string   `json:"demotion_or_retirement"`
	InvalidatingUnknown string   `json:"invalidating_assumption"`
}

// BuildShedRefcountReport folds the default known-bad fixture (a session with a
// shed-then-referenced span) — the primary acceptance witness.
func BuildShedRefcountReport() ShedRefcountReport {
	return BuildShedRefcountReportFor(DefaultShedThenReferencedTranscript())
}

// BuildShedRefcountReportFor folds an arbitrary transcript — the seam a live
// gateway capture feeds real shed-id × reference records into.
func BuildShedRefcountReportFor(tr ShedSpanTranscript) ShedRefcountReport {
	// Pass 1: a span's shed turn is the FIRST turn that dropped it (a re-shed
	// after a window rewrite keeps the earliest; the span was already gone).
	shedTurn := map[string]int{}
	for _, e := range tr.Events {
		if e.Kind != EventShed || e.SpanID == "" {
			continue
		}
		if prev, ok := shedTurn[e.SpanID]; !ok || e.Turn < prev {
			shedTurn[e.SpanID] = e.Turn
		}
	}

	// Pass 2: count references that land strictly AFTER the shed turn. A ref at
	// or before the shed turn is live use, not a use-after-free (over-count guard);
	// a ref to a span that was never shed is not a use-after-free (ignored).
	type tally struct {
		byKind   map[string]int
		refTurns []int
	}
	tallies := map[string]*tally{}
	for _, e := range tr.Events {
		if !isRefKind(e.Kind) || e.SpanID == "" {
			continue
		}
		st, shed := shedTurn[e.SpanID]
		if !shed || e.Turn <= st {
			continue
		}
		t := tallies[e.SpanID]
		if t == nil {
			t = &tally{byKind: map[string]int{}}
			tallies[e.SpanID] = t
		}
		t.byKind[e.Kind]++
		t.refTurns = append(t.refTurns, e.Turn)
	}

	spans := make([]SpanRefcount, 0, len(shedTurn))
	uaf := 0
	for id, st := range shedTurn {
		row := SpanRefcount{SpanID: id, ShedTurn: st}
		if t := tallies[id]; t != nil {
			for _, k := range refKinds {
				if n := t.byKind[k]; n > 0 {
					if row.RefsByKind == nil {
						row.RefsByKind = map[string]int{}
					}
					row.RefsByKind[k] = n
					row.Refcount += n
				}
			}
			row.RefTurns = append(row.RefTurns, t.refTurns...)
			sort.Ints(row.RefTurns)
		}
		row.UseAfterFree = row.Refcount > 0
		if row.UseAfterFree {
			uaf++
		}
		spans = append(spans, row)
	}
	// Stable output: by shed turn, then span id.
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].ShedTurn != spans[j].ShedTurn {
			return spans[i].ShedTurn < spans[j].ShedTurn
		}
		return spans[i].SpanID < spans[j].SpanID
	})

	signal := ""
	if uaf > 0 {
		signal = SignalUseAfterFree
	}

	return ShedRefcountReport{
		Schema:            ShedRefcountSchema,
		Provenance:        simulatedShedRefcountProvenance(),
		Transcript:        tr.Name,
		TotalEvents:       len(tr.Events),
		ShedSpans:         len(spans),
		Spans:             spans,
		UseAfterFreeCount: uaf,
		Signal:            signal,
		Clean:             uaf == 0,
		Assumptions: []string{
			"A span's shed turn is the FIRST turn that dropped it; a reference is counted only when it lands STRICTLY after that turn — a reference at or before the shed is normal live use, not a use-after-free. This is the over-count guard the issue's confusion-risk names.",
			"The three reference kinds (reread, cite, pin) are treated as equal-weight increments; the detector counts a use-after-free, it does not rank by how the span was reached. A weighting policy is a later scorer's job (#3098), not this fold's.",
			"Span ids are stable across shed and reference (the ctxspans/ctxrestore sha256 handle), so a later turn's re-read/cite/pin names the SAME id the trim dropped. A capture that re-mints ids per turn would under-count and must resolve ids to the shed handle before folding.",
		},
		Promotion:          "Replace the hermetic transcript with an OBSERVED capture from a live guard session — the gateway ctxspans/ctxrestore shed ids joined against the turn log's re-read (fak_context_restore / file re-open), cite (id echoed in a later turn), and pin events — fed through BuildShedRefcountReportFor. When a live session with a known shed-then-reference reproduces refcount > 0 and the signal fires, the detector promotes toward gen/now.",
		DemotionRetirement: "If a live capture shows the three reference kinds cannot be enumerated deterministically from the turn log without a model in the loop (e.g. a cite that only a semantic match would catch), the deterministic claim is demoted and the signal is reframed as a lower-bound rather than defended as exact.",
		InvalidatingUnknown: "The assumption most likely to flip the result is id stability: if a re-read pages content back under a NEW id (a fresh content-address after an edit) rather than the shed handle, the reference does not match the shed span and the use-after-free is missed — a false-clean, the dangerous direction.",
	}
}

func isRefKind(kind string) bool {
	for _, k := range refKinds {
		if kind == k {
			return true
		}
	}
	return false
}

// simulatedShedRefcountProvenance labels the hermetic-fixture path.
func simulatedShedRefcountProvenance() Provenance {
	return Provenance{
		Kind:        ProvenanceSimulated,
		Command:     "go test ./internal/bench -run ShedRefcount",
		GeneratedBy: "fak/internal/bench.BuildShedRefcountReport",
		Note: "Transcript is a labeled hermetic FIXTURE, not a live capture. It witnesses " +
			"the deterministic MEASUREMENT — a known shed-then-reference lands refcount > 0 " +
			"and fires USE_AFTER_FREE per span, a clean session lands every shed span at " +
			"refcount 0. A live gateway ctxspans/ctxrestore × turn-reference capture feeds " +
			"the same shape via BuildShedRefcountReportFor to promote the signal.",
	}
}

// DefaultShedThenReferencedTranscript is the known-bad acceptance fixture: a
// compaction at turn 3 sheds three spans; two are reached for by LATER turns (a
// re-read + a cite of one, a pin of another) — a use-after-free each — while the
// third is never referenced and stays safe. It discriminates PER span: not every
// shed is a use-after-free.
func DefaultShedThenReferencedTranscript() ShedSpanTranscript {
	return ShedSpanTranscript{
		Name: "shed-then-referenced (known use-after-free)",
		Events: []CtxSpanEvent{
			// Turn 3: a compaction drops three content-addressed spans.
			{Turn: 3, Kind: EventShed, SpanID: "sha-originating-task"},
			{Turn: 3, Kind: EventShed, SpanID: "sha-toolresult-schema"},
			{Turn: 3, Kind: EventShed, SpanID: "sha-scratch-note"},
			// Later turns reach BACK for two of them — the use-after-free.
			{Turn: 5, Kind: RefReread, SpanID: "sha-originating-task"}, // re-reads the dropped task
			{Turn: 6, Kind: RefPin, SpanID: "sha-toolresult-schema"},   // pins the dropped schema
			{Turn: 7, Kind: RefCite, SpanID: "sha-originating-task"},   // cites the dropped task again
			// sha-scratch-note is never referenced after the shed — safe.
		},
	}
}

// DefaultCleanTranscript is the clean-session control: a span is re-read while it
// is still LIVE (turn 1) and only shed afterward (turn 4), and a later pin names
// a span that was never shed. No shed span is referenced after its shed, so every
// shed span lands at refcount 0 — the acceptance's clean half, and the guard that
// live use and non-shed references do not over-flag.
func DefaultCleanTranscript() ShedSpanTranscript {
	return ShedSpanTranscript{
		Name: "clean session (no use-after-free)",
		Events: []CtxSpanEvent{
			{Turn: 1, Kind: RefReread, SpanID: "sha-config"}, // live use BEFORE shed — not counted
			{Turn: 4, Kind: EventShed, SpanID: "sha-config"}, // shed after its last use — safe
			{Turn: 4, Kind: EventShed, SpanID: "sha-stale-log"},
			{Turn: 6, Kind: RefPin, SpanID: "sha-active-doc"}, // pins a span that was never shed — ignored
		},
	}
}

// JSON renders the report as stable, indented JSON (deterministic: no clock, no
// unsorted map iteration in the slices), a re-derivable witness.
func (r ShedRefcountReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}
