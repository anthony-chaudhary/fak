// Package sessionsearch provides witnessed cross-session recall over the guard
// tool-process journal (#2913). It builds a pure-Go TF-IDF inverted index with
// recency windowing and source weighting, verifies byte-stability of suffix-only
// prompt injection against provider prefix caches, and witnesses downstream recall
// usefulness by measuring token survival into subsequent outcomes.
package sessionsearch

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// Source identifies the recall-priority class of an originating session.
type Source string

const (
	// SourceInteractive marks a human-driven interactive session with full recall weight.
	SourceInteractive Source = "interactive"
	// SourceCron marks a scheduled background session demoted in recall ranking.
	SourceCron Source = "cron"
)

// CronDemotion is the multiplicative recall score penalty applied to cron session hits.
const CronDemotion = 0.5

// NormSource parses a source string into the closed Source vocabulary, defaulting to interactive.
func NormSource(s string) Source {
	if Source(s) == SourceCron {
		return SourceCron
	}
	return SourceInteractive
}

func (s Source) weight() float64 {
	if NormSource(string(s)) == SourceCron {
		return CronDemotion
	}
	return 1
}

// Doc represents an indexable unit of recall with ordinal position and text descriptor.
type Doc struct {
	ID      string `json:"id"`
	Ordinal int    `json:"ordinal"`
	Source  Source `json:"source"`
	Text    string `json:"text"`
}

// Hit contains a matched document, its scored relevance, and surrounding context window.
type Hit struct {
	Doc    Doc     `json:"doc"`
	Score  float64 `json:"score"`
	Window []Doc   `json:"window,omitempty"`
}

type posting struct {
	doc int
	tf  int
}

// Index is a pure-Go TF-IDF inverted index mapping terms to document postings and frequencies.
type Index struct {
	docs     []Doc
	postings map[string][]posting
	df       map[string]int
}

// NewIndex constructs an initialized, empty search index ready for document insertion.
func NewIndex() *Index {
	return &Index{postings: map[string][]posting{}, df: map[string]int{}}
}

// Add indexes a document and updates internal term-frequency postings and document frequencies.
func (ix *Index) Add(d Doc) {
	if ix == nil {
		return
	}
	if ix.postings == nil {
		ix.postings = make(map[string][]posting)
	}
	if ix.df == nil {
		ix.df = make(map[string]int)
	}
	d.Source = NormSource(string(d.Source))
	i := len(ix.docs)
	ix.docs = append(ix.docs, d)
	for term, tf := range termFreq(d.Text) {
		ix.postings[term] = append(ix.postings[term], posting{doc: i, tf: tf})
		ix.df[term]++
	}
}

// Len reports the total count of indexed documents currently stored in the index.
func (ix *Index) Len() int {
	if ix == nil {
		return 0
	}
	return len(ix.docs)
}

func (ix *Index) idf(term string) float64 {
	n := float64(len(ix.docs))
	df := float64(ix.df[term])
	return math.Log((n+1)/(df+1)) + 1
}

// Search queries the inverted index and returns the top-k hits ranked by TF-IDF and source weight.
func (ix *Index) Search(query string, k, window int) []Hit {
	if ix == nil || len(ix.docs) == 0 {
		return nil
	}
	if k <= 0 {
		k = DefaultK
	}
	if window < 0 {
		window = DefaultWindow
	}
	terms := distinctTerms(query)
	if len(terms) == 0 {
		return nil
	}

	scored := map[int]float64{}
	for _, term := range terms {
		idf := ix.idf(term)
		for _, p := range ix.postings[term] {
			scored[p.doc] += float64(p.tf) * idf
		}
	}
	if len(scored) == 0 {
		return nil
	}

	cands := make([]Hit, 0, len(scored))
	for di, raw := range scored {
		d := ix.docs[di]
		cands = append(cands, Hit{Doc: d, Score: raw * d.Source.weight()})
	}
	sort.Slice(cands, func(a, b int) bool {
		if cands[a].Score != cands[b].Score {
			return cands[a].Score > cands[b].Score
		}
		aw, bw := cands[a].Doc.Source.weight(), cands[b].Doc.Source.weight()
		if aw != bw {
			return aw > bw
		}
		if cands[a].Doc.Ordinal != cands[b].Doc.Ordinal {
			return cands[a].Doc.Ordinal < cands[b].Doc.Ordinal
		}
		return cands[a].Doc.ID < cands[b].Doc.ID
	})

	var out []Hit
	covered := map[int]bool{}
	for _, h := range cands {
		if len(out) >= k {
			break
		}
		if covered[h.Doc.Ordinal] {
			continue
		}
		h.Window = ix.windowAround(h.Doc.Ordinal, window)
		for _, w := range h.Window {
			covered[w.Ordinal] = true
		}
		out = append(out, h)
	}
	return out
}

func (ix *Index) windowAround(center, window int) []Doc {
	var w []Doc
	for _, d := range ix.docs {
		if d.Ordinal >= center-window && d.Ordinal <= center+window {
			w = append(w, d)
		}
	}
	sort.Slice(w, func(a, b int) bool { return w[a].Ordinal < w[b].Ordinal })
	return w
}

// DefaultK is the fallback limit on top ranked hits returned by Search.
const DefaultK = 5

// DefaultWindow is the fallback ordinal radius attached as surrounding context to each hit.
const DefaultWindow = 5

// DocsFromJournal parses a toolproc journal reader into searchable Doc descriptors.
func DocsFromJournal(r io.Reader) ([]Doc, error) {
	if r == nil {
		return nil, fmt.Errorf("sessionsearch: reader is nil")
	}
	events, err := toolproc.ParseEvents(r)
	if err != nil {
		return nil, err
	}
	docs := make([]Doc, 0, len(events))
	for i, ev := range events {
		docs = append(docs, Doc{
			ID:      fmt.Sprintf("ev:%d", i),
			Ordinal: i,
			Source:  classifySource(ev.Session),
			Text:    eventText(ev),
		})
	}
	return docs, nil
}

func eventText(ev toolproc.Event) string {
	parts := []string{string(ev.Kind)}
	if ev.Tool != "" {
		parts = append(parts, ev.Tool)
	}
	if ev.Status != "" {
		parts = append(parts, ev.Status)
	}
	if ev.Reason != "" {
		parts = append(parts, ev.Reason)
	}
	if ev.Session != "" {
		parts = append(parts, ev.Session)
	}
	return strings.Join(parts, " ")
}

func classifySource(session string) Source {
	if strings.Contains(strings.ToLower(session), "cron") {
		return SourceCron
	}
	return SourceInteractive
}

// RecalledSpan renders ranked hits and their surrounding windows into formatted text.
func RecalledSpan(hits []Hit) string {
	if len(hits) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("recalled from prior sessions:\n")
	for i, h := range hits {
		src := NormSource(string(h.Doc.Source))
		fmt.Fprintf(&b, "  #%d [%s] %s\n", i+1, src, h.Doc.Text)
		for _, w := range h.Window {
			if w.ID == h.Doc.ID {
				continue
			}
			fmt.Fprintf(&b, "      ~ %s\n", w.Text)
		}
	}
	return b.String()
}

// PrefixProof records cachemeta divergence and confirms prefix cacheability under suffix injection.
type PrefixProof struct {
	Divergence   cachemeta.TurnDivergence `json:"divergence"`
	PrefixTokens int64                    `json:"prefix_tokens"`
	PrefixStable bool                     `json:"prefix_stable"`
}

// Inject appends recalled text to a prompt prefix as a suffix and proves byte stability.
func Inject(prefix []cachemeta.PromptSegment, recalled string) ([]cachemeta.PromptSegment, PrefixProof) {
	injected := make([]cachemeta.PromptSegment, len(prefix), len(prefix)+1)
	copy(injected, prefix)
	injected = append(injected, cachemeta.PromptSegment{
		Kind:    cachemeta.SegMessage,
		Tokens:  EstimateTokens(recalled),
		Content: []byte(recalled),
	})

	d := cachemeta.Diverge(prefix, injected)
	var prefixTokens int64
	for _, s := range prefix {
		prefixTokens += s.Tokens
	}
	proof := PrefixProof{
		Divergence:   d,
		PrefixTokens: prefixTokens,
		PrefixStable: d.FirstDivergeSeg == len(prefix) && !d.SealedStop && d.StableTokens == prefixTokens,
	}
	return injected, proof
}

// EstimateTokens computes a coarse byte-to-token count matching cachemeta conventions.
func EstimateTokens(s string) int64 {
	if t := int64(len(s)) / 4; t > 0 {
		return t
	}
	if s != "" {
		return 1
	}
	return 0
}

// Usefulness measures whether injected recall hits were referenced in downstream output.
type Usefulness struct {
	Hits            int     `json:"hits"`
	Referenced      int     `json:"referenced"`
	Unreferenced    int     `json:"unreferenced"`
	ReferencedRatio float64 `json:"referenced_ratio"`
	Blindness       bool    `json:"blindness"`
}

// WitnessUsefulness evaluates which recalled hits contributed distinctive terms to the outcome.
func WitnessUsefulness(hits []Hit, priorContext, outcome string) Usefulness {
	prior := distinctTermSet(priorContext)
	out := distinctTermSet(outcome)
	u := Usefulness{Hits: len(hits)}
	for _, h := range hits {
		referenced := false
		hitTerms := distinctTermSet(h.Doc.Text)
		for _, w := range h.Window {
			for term := range distinctTermSet(w.Text) {
				hitTerms[term] = true
			}
		}
		for term := range hitTerms {
			if prior[term] {
				continue
			}
			if out[term] {
				referenced = true
				break
			}
		}
		if referenced {
			u.Referenced++
		} else {
			u.Unreferenced++
		}
	}
	if u.Hits > 0 {
		u.ReferencedRatio = float64(u.Referenced) / float64(u.Hits)
		u.Blindness = u.Referenced == 0
	}
	return u
}

// UsefulnessLedgerSchema identifies the schema version for persisted recall usefulness rows.
const UsefulnessLedgerSchema = "fak.sessionsearch-recall-usefulness/1"

// UsefulnessRow represents an append-only JSONL record tracking recall hit outcomes.
type UsefulnessRow struct {
	Schema       string `json:"schema"`
	Hits         int    `json:"hits"`
	Referenced   int    `json:"referenced"`
	Unreferenced int    `json:"unreferenced"`
}

// Row converts a Usefulness measurement into an exportable ledger row.
func (u Usefulness) Row() UsefulnessRow {
	return UsefulnessRow{
		Schema:       UsefulnessLedgerSchema,
		Hits:         u.Hits,
		Referenced:   u.Referenced,
		Unreferenced: u.Unreferenced,
	}
}

// MarshalRow serializes the usefulness measurement into formatted JSON bytes.
func (u Usefulness) MarshalRow() ([]byte, error) { return json.Marshal(u.Row()) }

func termFreq(s string) map[string]int {
	out := map[string]int{}
	for _, t := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(t) > 2 {
			out[t]++
		}
	}
	return out
}

func distinctTerms(s string) []string {
	seen := distinctTermSet(s)
	if len(seen) == 0 {
		return nil
	}
	out := make([]string, 0, len(seen))
	for t := range seen {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

func distinctTermSet(s string) map[string]bool {
	out := map[string]bool{}
	for _, t := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(t) > 2 {
			out[t] = true
		}
	}
	return out
}
