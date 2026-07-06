// Package sessionsearch is witnessed cross-session recall over the guard session
// journal (#2913, parent #2908 — porting the design of Hermes'
// tools/session_search_tool.py to fak's evidence discipline).
//
// # What Hermes has, and the two things it does not
//
// Hermes gives FTS5 cross-session recall: a full-text index over past sessions
// with a ±window, lineage dedup, and a recall order that demotes cron sources
// below interactive ones. That is a good access path. But two properties the
// kernel cares about are missing:
//
//   - Recall that isn't MEASURED can silently regress. An index that returns
//     irrelevant hits (or hits the model never uses) is "recall blindness": the
//     search still runs, it just stops helping, and nothing in the loop notices.
//   - Injecting recalled text mid-conversation BREAKS the provider prefix cache.
//     A provider only rewards a byte-stable prompt PREFIX; splicing recalled
//     bytes ahead of the tail re-bills every token after the splice at full
//     price (the failure class internal/cachemeta.Diverge exists to catch).
//
// # What this leaf adds (fak owns the prefix and measures context value)
//
// This package is a pure fold with three parts, each answering one acceptance
// item of #2913. It is stdlib + two foundation leaves (toolproc for the journal
// contract, cachemeta for the prefix-stability math): no SQLite/FTS5 dependency
// (the module is std-lib-only), so the "FTS5" is a pure-Go TF-IDF inverted index
// — the same access path shape SQLite's FTS5 bm25() gives, without the C dep.
//
//  1. RECALL. An Index over documents lowered from the guard tool-process journal
//     (.fak/toolproc/journal.jsonl, internal/toolproc.Event). Search ranks by a
//     smoothed TF-IDF sum, carries a ±window of neighbouring events, dedups
//     overlapping neighbourhoods, and DEMOTES cron sources below interactive ones
//     (the _order_for_recall behaviour) via a source weight.
//
//  2. CACHE-SAFE INJECTION. Inject appends the recalled span as a fresh SUFFIX
//     segment and never mutates the prefix, then PROVES byte-stability by running
//     the existing prefix bytes through cachemeta.Diverge: a stable injection
//     leaves every prefix segment cacheable (StableTokens == the whole prefix)
//     and diverges only at the appended tail. This is a witness, not a promise.
//
//  3. USEFULNESS WITNESS. WitnessUsefulness measures whether the recalled span
//     was actually REFERENCED downstream: for each hit it takes the tokens the
//     recall ADDED (the hit's tokens minus what the prior context already held)
//     and checks whether any survived into the outcome. Hits>0 with zero
//     referenced is recall blindness — now a counted metric, not a silent
//     regression. The ledger row schema mirrors the memory-value ledger so a
//     higher-tier consumer can fold it exactly like recall_value_witnessed.
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

// Source is the CLOSED recall-priority class of a document's originating session.
// The recall order demotes cron below interactive (Hermes' _order_for_recall): a
// scheduled/background session's hits are lower-value context than a human-driven
// one's, so a tie between them should prefer the interactive hit.
type Source string

const (
	// SourceInteractive is a human-driven session — full recall weight.
	SourceInteractive Source = "interactive"
	// SourceCron is a scheduled/background session — demoted in recall.
	SourceCron Source = "cron"
)

// CronDemotion is the multiplicative recall-weight penalty applied to a cron
// document's score. It is < 1 so a cron hit ranks below an interactive hit of
// equal textual relevance — the ranking form of _order_for_recall's demotion.
// Conservative (halve, not zero): a cron hit is still recallable, just outranked.
const CronDemotion = 0.5

// NormSource maps any source string into the closed vocabulary, failing OPEN to
// interactive: an unknown/absent source is treated as full-weight, never
// silently demoted (demotion is a deliberate signal, not a default).
func NormSource(s string) Source {
	if Source(s) == SourceCron {
		return SourceCron
	}
	return SourceInteractive
}

// weight is the recall-order multiplier for a source (interactive 1, cron demoted).
func (s Source) weight() float64 {
	if NormSource(string(s)) == SourceCron {
		return CronDemotion
	}
	return 1
}

// Doc is one indexable unit of recall: a stable id, its ordinal position in the
// append-ordered journal (the coordinate the ±window walks), the source class
// that sets its recall weight, and the SAFE text the index tokenizes. Text is a
// descriptor, never sealed bytes — the same posture ctxplan.Span.Descriptor takes.
type Doc struct {
	ID      string `json:"id"`
	Ordinal int    `json:"ordinal"`
	Source  Source `json:"source"`
	Text    string `json:"text"`
}

// Hit is one ranked recall result: the matched document, its TF-IDF-after-demotion
// score, and the ±window of neighbouring documents that give it context (Hermes'
// +/-N message window). Window is in ordinal order and includes the hit itself.
type Hit struct {
	Doc    Doc     `json:"doc"`
	Score  float64 `json:"score"`
	Window []Doc   `json:"window,omitempty"`
}

// posting is one (document, term-frequency) entry of a term's posting list.
type posting struct {
	doc int // index into Index.docs
	tf  int // count of the term in that document
}

// Index is the pure-Go inverted index — the FTS5 analogue. It holds the append
// order (docs, for the recency/window walk), the per-term posting lists with
// term frequencies (the TF half of TF-IDF), and the document frequency per term
// (the IDF half). It is append-only (Add) and deterministic (a Search over a
// fixed (index, query, k, window) yields the same hits).
type Index struct {
	docs     []Doc
	postings map[string][]posting
	df       map[string]int // term -> number of docs containing it
}

// NewIndex returns an empty index.
func NewIndex() *Index {
	return &Index{postings: map[string][]posting{}, df: map[string]int{}}
}

// Add indexes one document in O(tokens(doc)) time: it appends to the doc list and,
// for every DISTINCT term, appends a (doc, tf) posting and bumps the term's
// document frequency. Duplicate ids are allowed (the journal can repeat a tool
// name); each Add is its own addressable ordinal.
func (ix *Index) Add(d Doc) {
	d.Source = NormSource(string(d.Source))
	i := len(ix.docs)
	ix.docs = append(ix.docs, d)
	for term, tf := range termFreq(d.Text) {
		ix.postings[term] = append(ix.postings[term], posting{doc: i, tf: tf})
		ix.df[term]++
	}
}

// Len reports the number of indexed documents.
func (ix *Index) Len() int { return len(ix.docs) }

// idf is the smoothed inverse-document-frequency weight of a term: log((N+1)/
// (df+1)) + 1. Always >= 1 (a matched term always contributes), never NaN/Inf,
// and monotonically decreasing in df — the same form ctxplan.Index.idf uses, so
// a common term ("read", "ok") earns ~1 and a rare, discriminating one earns
// more. A term never indexed has df 0.
func (ix *Index) idf(term string) float64 {
	n := float64(len(ix.docs))
	df := float64(ix.df[term])
	return math.Log((n+1)/(df+1)) + 1
}

// Search returns the top-k documents for a free-text query, ranked by the sum
// over the query's distinct terms of tf(term,doc)*idf(term), then multiplied by
// the document's source weight (cron demoted). k<=0 defaults to DefaultK; window
// is the ±ordinal neighbourhood attached to each hit (DefaultWindow if <0).
//
// Ordering is deterministic: score descending, then interactive-before-cron,
// then ordinal ascending, then id — so equal-relevance hits break toward the
// higher-recall-weight, earlier document. Overlapping neighbourhoods are deduped
// (lineage dedup analogue): once a document is inside a selected hit's window, a
// lower-scored hit that falls in the same window is dropped, so the k results
// cover k DISTINCT regions rather than k adjacent rows of one event burst.
func (ix *Index) Search(query string, k, window int) []Hit {
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
			return aw > bw // interactive (weight 1) before cron (demoted)
		}
		if cands[a].Doc.Ordinal != cands[b].Doc.Ordinal {
			return cands[a].Doc.Ordinal < cands[b].Doc.Ordinal
		}
		return cands[a].Doc.ID < cands[b].Doc.ID
	})

	var out []Hit
	covered := map[int]bool{} // ordinals already inside a selected hit's window
	for _, h := range cands {
		if len(out) >= k {
			break
		}
		if covered[h.Doc.Ordinal] {
			continue // lineage dedup: this row is already in a higher hit's neighbourhood
		}
		h.Window = ix.windowAround(h.Doc.Ordinal, window)
		for _, w := range h.Window {
			covered[w.Ordinal] = true
		}
		out = append(out, h)
	}
	return out
}

// windowAround returns the documents whose ordinal is within +/-window of center,
// in ordinal order, including the center. The journal is append-ordered, so this
// is the contiguous message neighbourhood Hermes' +/-N window recalls.
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

// DefaultK is the seed result cap for Search (a handful of distinct regions).
const DefaultK = 5

// DefaultWindow is the seed ±ordinal neighbourhood (Hermes recalls ±5 messages).
const DefaultWindow = 5

// DocsFromJournal lowers a guard tool-process journal (.fak/toolproc/journal.jsonl)
// into indexable documents. It parses through the real toolproc contract
// (ParseEvents fails closed on a bad enum token, so a corrupt journal is refused
// at the boundary, never half-recalled), then renders each event's SAFE fields
// into one searchable line. The ordinal is the event's position; the id is
// "ev:<ordinal>"; the source is classified from the session id (see classify).
func DocsFromJournal(r io.Reader) ([]Doc, error) {
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

// eventText renders one journal event's SAFE fields into a searchable descriptor:
// the kind, the tool, the exit status, the kill reason, and the session — every
// field an operator would query by ("which sessions killed a tool for
// TOOL_DEADLINE_EXCEEDED?"). It never emits payload bytes (the journal has none).
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

// classifySource infers the recall source from a session id: a session whose id
// carries the "cron" marker is a scheduled/background session (demoted), else
// interactive. This is the one heuristic the journal exposes — the toolproc Event
// has no explicit source field — and it fails OPEN to interactive (NormSource).
func classifySource(session string) Source {
	if strings.Contains(strings.ToLower(session), "cron") {
		return SourceCron
	}
	return SourceInteractive
}

// RecalledSpan renders a hit set into ONE deterministic text block — the span a
// caller injects. Each hit is rendered with its rank, source, and window, in the
// order Search returned (highest recall value first). An empty hit set renders "".
func RecalledSpan(hits []Hit) string {
	if len(hits) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("recalled from prior sessions:\n")
	for i, h := range hits {
		fmt.Fprintf(&b, "  #%d [%s] %s\n", i+1, h.Doc.Source, h.Doc.Text)
		for _, w := range h.Window {
			if w.ID == h.Doc.ID {
				continue
			}
			fmt.Fprintf(&b, "      ~ %s\n", w.Text)
		}
	}
	return b.String()
}

// PrefixProof is the byte-stability witness for one injection: the divergence the
// injected turn has against the pre-injection prefix, plus the one-bit verdict.
// PrefixStable is true only when EVERY prefix segment stayed cacheable and the
// turn diverged solely at the appended tail — the definition of a cache-safe
// suffix injection.
type PrefixProof struct {
	Divergence   cachemeta.TurnDivergence `json:"divergence"`
	PrefixTokens int64                    `json:"prefix_tokens"` // cacheable tokens the prefix carried
	PrefixStable bool                     `json:"prefix_stable"`
}

// Inject appends the recalled span to an existing prompt prefix as a fresh SUFFIX
// segment and returns both the new turn and the byte-stability PROOF. It never
// mutates the caller's prefix slice (it copies before append), and it estimates
// the suffix segment's billed tokens from its byte length (~4 B/token, the
// cachemeta relative-accounting convention). The proof runs the prefix bytes back
// through cachemeta.Diverge: a stable injection leaves the whole prefix cacheable
// (StableTokens == sum of the prefix's tokens) and reports the recalled span as
// the only lost/tail tokens.
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
		// Stable iff divergence begins exactly at the tail we appended (every prefix
		// segment matched, none sealed) AND every prefix token stayed cacheable.
		PrefixStable: d.FirstDivergeSeg == len(prefix) && !d.SealedStop && d.StableTokens == prefixTokens,
	}
	return injected, proof
}

// EstimateTokens is the coarse byte->token estimate (~4 B/token, min 1 for
// non-empty), matching cachemeta/recall relative accounting. It is honest for the
// linter's relative cacheable/lost math, never sold as an exact provider count.
func EstimateTokens(s string) int64 {
	if t := int64(len(s)) / 4; t > 0 {
		return t
	}
	if s != "" {
		return 1
	}
	return 0
}

// Usefulness is the recall-usefulness witness for one injected recall: how many
// hits were injected, how many were REFERENCED downstream (a token the recall
// added survived into the outcome), and the derived ratio. Blindness is the one
// failure bit — hits injected but none referenced, the "recall blindness" the
// issue names: recall ran, cost prefix budget, and changed nothing.
type Usefulness struct {
	Hits            int     `json:"hits"`
	Referenced      int     `json:"referenced"`
	Unreferenced    int     `json:"unreferenced"`
	ReferencedRatio float64 `json:"referenced_ratio"`
	Blindness       bool    `json:"blindness"`
}

// WitnessUsefulness measures whether an injected recall was actually used. For
// each hit it computes the DISTINCTIVE tokens the recall contributed — the hit's
// (and its window's) terms MINUS whatever the prior context already carried — and
// counts the hit as referenced if any distinctive term survived into the outcome
// text. Subtracting the prior context is what makes this a witness and not a
// tautology: a hit that only echoes words already in context cannot be credited
// for "changing the outcome". A hit with no distinctive tokens at all is
// unreferenced (it added nothing that could be used).
func WitnessUsefulness(hits []Hit, priorContext, outcome string) Usefulness {
	prior := distinctTermSet(priorContext)
	out := distinctTermSet(outcome)
	u := Usefulness{Hits: len(hits)}
	for _, h := range hits {
		text := h.Doc.Text
		for _, w := range h.Window {
			text += " " + w.Text
		}
		referenced := false
		for term := range distinctTermSet(text) {
			if prior[term] {
				continue // already in context — the recall did not contribute it
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

// UsefulnessLedgerSchema stamps a recall-usefulness ledger row. Versioned /1 so a
// consumer folds it the way memvaluescore folds fak-memory-value-ledger/1.
const UsefulnessLedgerSchema = "fak.sessionsearch-recall-usefulness/1"

// UsefulnessRow is one append-only witness row for the recall-usefulness ledger —
// the counts a higher-tier consumer sums into a frontier (the shape of the
// memory-value ledger). Audit context (intent, session) is the consumer's to add.
type UsefulnessRow struct {
	Schema       string `json:"schema"`
	Hits         int    `json:"hits"`
	Referenced   int    `json:"referenced"`
	Unreferenced int    `json:"unreferenced"`
}

// Row lowers a Usefulness verdict into a schema-stamped ledger row.
func (u Usefulness) Row() UsefulnessRow {
	return UsefulnessRow{
		Schema:       UsefulnessLedgerSchema,
		Hits:         u.Hits,
		Referenced:   u.Referenced,
		Unreferenced: u.Unreferenced,
	}
}

// MarshalRow renders a usefulness row as one JSONL line (no trailing newline) —
// the exact byte form a ledger appender writes.
func (u Usefulness) MarshalRow() ([]byte, error) { return json.Marshal(u.Row()) }

// termFreq is the term-frequency map of a text: lowercased content tokens (letters
// and digits, length > 2) counted with repeats — the same extractive tokenization
// ctxplan/recall/memq share, kept local so this leaf adds no coupling to them.
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

// distinctTerms is the ordered distinct content-token slice of a query.
func distinctTerms(s string) []string {
	seen := distinctTermSet(s)
	out := make([]string, 0, len(seen))
	for t := range seen {
		out = append(out, t)
	}
	sort.Strings(out) // deterministic scan order
	return out
}

// distinctTermSet is the distinct content-token set of a text.
func distinctTermSet(s string) map[string]bool {
	out := map[string]bool{}
	for t := range termFreq(s) {
		out[t] = true
	}
	return out
}
