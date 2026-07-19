package modelroute

import (
	"fmt"
	"strings"
	"unicode"
)

// spanfill.go — the SPAN demand type: span-level fill-in-the-blank (#4319, spine
// #4296 "demand-driven partial consult"). The small LOCAL model writes the entire
// answer — the bulk fluent scaffolding — but marks the spans it knows are beyond it
// (a specific fact, an exact number, a citation, a tricky proof step, a rare API
// signature) as inline blanks. The blanks become SPAN demands; the big REMOTE model
// fills ONLY those blanks. The load-bearing hard bits get the big model's capacity;
// the fluent connective tissue costs nothing remote.
//
// This is the coarser, semantic-granularity sibling of token-level uncertainty-gated
// consult (#4317): a span carries WHAT KIND of help it needs (its SpanKind), which is
// richer than a bare next-token escalation. The demand is a request, not a promise —
// every fill is accepted or rejected (a number span must actually contain a digit; a
// citation must be non-empty), so a bad fill is caught and reported as UNFILLED, never
// silently trusted (the assumption-ledger discipline, #4106).
//
// Pure, deterministic, stdlib-only: the "big model" is abstracted behind the Filler
// seam, so the whole span→fill→splice→account path is exercisable with no GPU, model,
// or network — the same posture as the resolver it sits beside. The savings witness
// (SpanSavings) reports what the epic asks of this child: the fraction of the answer
// the big model had to generate vs. producing the whole thing itself.
//
// Marker grammar (what the local draft emits): [[SPAN:kind|hint]] or [[SPAN:kind]].
// `kind` is one of the SpanKind values (an unknown kind parses as SpanOther, never an
// error — the local model should not be able to red the pipeline with a typo); `hint`
// is the free-text local context the remote model fills against.

// SpanKind names WHAT KIND of hard span the local model marked — richer than a bare
// "I'm unsure here" so the remote fill (and its acceptance check) can be kind-aware.
type SpanKind string

const (
	SpanFact     SpanKind = "fact"     // a specific fact the local model doesn't hold
	SpanNumber   SpanKind = "number"   // an exact number / quantity / date
	SpanCitation SpanKind = "citation" // a reference that must resolve
	SpanProof    SpanKind = "proof"    // a tricky proof / derivation step
	SpanAPI      SpanKind = "api"      // a rare API signature / exact call
	SpanOther    SpanKind = "other"    // marked hard, unclassified
)

// knownSpanKind reports whether k is one of the named kinds; anything else folds to
// SpanOther at parse time.
func knownSpanKind(k SpanKind) bool {
	switch k {
	case SpanFact, SpanNumber, SpanCitation, SpanProof, SpanAPI, SpanOther:
		return true
	}
	return false
}

// SpanDemand is one blank the local draft marked for the big model to fill. Index is
// its 0-based ordinal in the draft (the key a fill is matched back on).
type SpanDemand struct {
	Index int      `json:"index"`
	Kind  SpanKind `json:"kind"`
	Hint  string   `json:"hint,omitempty"`
}

// segment is one piece of a parsed draft: either literal local text (demand == nil) or
// a blank to be filled (demand != nil).
type segment struct {
	lit    string
	demand *SpanDemand
}

// Draft is a parsed local answer: an ordered run of literal scaffolding and marked
// blanks. The zero Draft is an empty answer.
type Draft struct {
	segments []segment
}

const (
	spanOpen  = "[[SPAN:"
	spanClose = "]]"
)

// ParseDraft splits a local draft into literal scaffolding and SPAN demands. A malformed
// marker — an opener with no closer — is a structural error and fails closed (the local
// model produced a blank the remote side could never resolve). An unknown `kind` is NOT
// an error: it folds to SpanOther, so a kind typo degrades to "help here" rather than
// redding the whole draft.
func ParseDraft(s string) (Draft, error) {
	var d Draft
	next := 0 // demand ordinal
	for {
		open := strings.Index(s, spanOpen)
		if open < 0 {
			if s != "" {
				d.segments = append(d.segments, segment{lit: s})
			}
			return d, nil
		}
		if open > 0 {
			d.segments = append(d.segments, segment{lit: s[:open]})
		}
		rest := s[open+len(spanOpen):]
		end := strings.Index(rest, spanClose)
		if end < 0 {
			return Draft{}, fmt.Errorf("modelroute: unterminated SPAN marker at %q", s[open:])
		}
		body := rest[:end]
		kindStr, hint := body, ""
		if bar := strings.IndexByte(body, '|'); bar >= 0 {
			kindStr, hint = body[:bar], body[bar+1:]
		}
		kind := SpanKind(strings.TrimSpace(kindStr))
		if !knownSpanKind(kind) {
			kind = SpanOther
		}
		dem := SpanDemand{Index: next, Kind: kind, Hint: strings.TrimSpace(hint)}
		next++
		d.segments = append(d.segments, segment{demand: &dem})
		s = rest[end+len(spanClose):]
	}
}

// Demands returns every SPAN the draft marked, in order — the batch to hand the remote
// filler. Empty (nil) when the local model needed no remote help at all.
func (d Draft) Demands() []SpanDemand {
	var out []SpanDemand
	for _, seg := range d.segments {
		if seg.demand != nil {
			out = append(out, *seg.demand)
		}
	}
	return out
}

// LocalChars is the byte count of the fluent scaffolding the local model produced for
// free (the literal segments only) — the connective tissue that costs nothing remote.
func (d Draft) LocalChars() int {
	n := 0
	for _, seg := range d.segments {
		if seg.demand == nil {
			n += len(seg.lit)
		}
	}
	return n
}

// acceptFill decides whether a proposed fill for a demand is trustworthy. Fail-closed:
// an empty fill is always rejected, and a number span must actually carry a digit —
// a bad fill is caught and reported UNFILLED, never spliced in and trusted.
func acceptFill(kind SpanKind, fill string) bool {
	if strings.TrimSpace(fill) == "" {
		return false
	}
	if kind == SpanNumber && !strings.ContainsFunc(fill, unicode.IsDigit) {
		return false
	}
	return true
}

// unfilledMarker is what a rejected/missing span renders as in the resolved answer —
// visibly a hole, so a downstream reader is never handed a silently-fabricated span.
func unfilledMarker(dem SpanDemand) string {
	if dem.Hint != "" {
		return fmt.Sprintf("[[UNFILLED:%s|%s]]", dem.Kind, dem.Hint)
	}
	return fmt.Sprintf("[[UNFILLED:%s]]", dem.Kind)
}

// Resolve splices the remote fills back into the local scaffolding, keyed by demand
// Index. A fill that is missing or fails acceptFill leaves a visible UNFILLED marker and
// is returned in `unfilled` — the caller decides whether a hole is tolerable; it is
// never hidden. The returned answer is always complete text (holes included).
func (d Draft) Resolve(fills map[int]string) (answer string, unfilled []SpanDemand) {
	var b strings.Builder
	for _, seg := range d.segments {
		if seg.demand == nil {
			b.WriteString(seg.lit)
			continue
		}
		dem := *seg.demand
		fill, ok := fills[dem.Index]
		if ok && acceptFill(dem.Kind, fill) {
			b.WriteString(fill)
			continue
		}
		b.WriteString(unfilledMarker(dem))
		unfilled = append(unfilled, dem)
	}
	return b.String(), unfilled
}

// SpanSavings is the witness this child owes the epic: how much of the final answer the
// big model actually had to generate, vs. producing the whole thing itself. RemoteFraction
// is the headline — bytes the remote model generated over the whole answer's bytes; a
// small fraction is the win (fluent bulk local, capacity spent only on the hard bits).
type SpanSavings struct {
	TotalSpans     int     `json:"total_spans"`
	FilledSpans    int     `json:"filled_spans"`
	UnfilledSpans  int     `json:"unfilled_spans"`
	LocalChars     int     `json:"local_chars"`     // scaffolding the local model wrote for free
	RemoteChars    int     `json:"remote_chars"`    // bytes the big model actually generated (accepted fills)
	BigOnlyChars   int     `json:"big_only_chars"`  // the whole answer's bytes = the big-only baseline
	RemoteFraction float64 `json:"remote_fraction"` // RemoteChars / BigOnlyChars (0 when the answer is empty)
}

// Savings measures a resolution against the big-only baseline. BigOnlyChars is the final
// answer length (what the big model would have had to generate alone); RemoteChars counts
// only accepted fills. A rejected/missing fill contributes to UnfilledSpans and its
// UNFILLED-marker bytes count toward the answer but not toward RemoteChars.
func (d Draft) Savings(fills map[int]string) SpanSavings {
	sv := SpanSavings{LocalChars: d.LocalChars()}
	answer, unfilled := d.Resolve(fills)
	sv.BigOnlyChars = len(answer)
	sv.UnfilledSpans = len(unfilled)
	for _, seg := range d.segments {
		if seg.demand == nil {
			continue
		}
		sv.TotalSpans++
		dem := *seg.demand
		if fill, ok := fills[dem.Index]; ok && acceptFill(dem.Kind, fill) {
			sv.FilledSpans++
			sv.RemoteChars += len(fill)
		}
	}
	if sv.BigOnlyChars > 0 {
		sv.RemoteFraction = float64(sv.RemoteChars) / float64(sv.BigOnlyChars)
	}
	return sv
}

// Filler is the big REMOTE model seam: it fills a batch of demands in ONE call (the
// batching the idea turns on — collect every blank, one remote round-trip) and returns
// one fill per demand, in the same order. Abstracting it here is what lets the whole
// path run with no model in the loop.
type Filler interface {
	Fill(demands []SpanDemand) ([]string, error)
}

// FillerFunc adapts a plain function to Filler.
type FillerFunc func(demands []SpanDemand) ([]string, error)

// Fill implements Filler.
func (f FillerFunc) Fill(demands []SpanDemand) ([]string, error) { return f(demands) }

// Consult runs the whole span-fill path: batch the draft's demands to the remote filler
// in one call, splice the fills back, and measure the savings. A filler that returns the
// wrong count is a protocol error (fail closed). When the draft has no demands the filler
// is never called. `unfilled` reports any span the remote side left a hole in.
func Consult(d Draft, f Filler) (answer string, sv SpanSavings, unfilled []SpanDemand, err error) {
	demands := d.Demands()
	fills := map[int]string{}
	if len(demands) > 0 {
		out, ferr := f.Fill(demands)
		if ferr != nil {
			return "", SpanSavings{}, nil, ferr
		}
		if len(out) != len(demands) {
			return "", SpanSavings{}, nil, fmt.Errorf("modelroute: filler returned %d fills for %d demands", len(out), len(demands))
		}
		for i, dem := range demands {
			fills[dem.Index] = out[i]
		}
	}
	answer, unfilled = d.Resolve(fills)
	sv = d.Savings(fills)
	return answer, sv, unfilled, nil
}
