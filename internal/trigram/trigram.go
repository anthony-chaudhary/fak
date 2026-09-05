// Package trigram is a pure-Go trigram postings index for substring and regex
// search over a document set — the tree + sibling-repo code-search seam (#3437,
// epic #3434). It borrows the two load-bearing ideas from Google Code Search
// (Russ Cox, "Regular Expression Matching with a Trigram Index"):
//
//  1. Index every distinct 3-rune shingle of each document, mapping trigram ->
//     posting list of document ids. A literal can only occur in a document that
//     contains ALL of the literal's trigrams, so the postings intersection is a
//     cheap candidate pre-filter before the exact (and expensive) verify.
//  2. Probe with only the RAREST trigrams. The two lowest-frequency trigrams of a
//     literal already cut the candidate set to near-nothing; intersecting more
//     buys little. And if any trigram of the literal has an EMPTY posting list,
//     the literal cannot exist anywhere — short-circuit to zero candidates without
//     touching a single document.
//
// The index is in-memory and built once from one goroutine, then queried; a repo
// is thousands of files, so a linear postings scan stays cheap and deterministic.
package trigram

import (
	"regexp"
	"regexp/syntax"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// Trigram is three runes packed into a uint64 (21 bits each; Unicode's max code
// point 0x10FFFF fits in 21 bits, and 3*21 = 63 <= 64). Packing keeps the postings
// key a comparable scalar instead of a heap string.
type Trigram uint64

func packTrigram(a, b, c rune) Trigram {
	return Trigram(uint64(a)<<42 | uint64(b)<<21 | uint64(c))
}

// trigrams returns the DISTINCT trigrams of s in first-seen order. A string of
// fewer than three runes has none (it cannot be located by the index and must be
// brute-force verified).
func trigrams(s string) []Trigram {
	rs := []rune(s)
	if len(rs) < 3 {
		return nil
	}
	seen := make(map[Trigram]bool, len(rs))
	out := make([]Trigram, 0, len(rs))
	for i := 0; i+3 <= len(rs); i++ {
		t := packTrigram(rs[i], rs[i+1], rs[i+2])
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	return out
}

type doc struct {
	id      string
	path    string
	content string
}

// Index maps each trigram to the sorted, de-duplicated list of document ids that
// contain it. The zero value is ready to use.
type Index struct {
	docs     []doc
	postings map[Trigram][]int
}

// Result is one matching document plus the 1-based line numbers where the query
// was found (populated by the verify pass).
type Result struct {
	ID    string
	Path  string
	Lines []int
}

// Add indexes a document under id (path is carried through to results). Re-adding
// an existing id is allowed; it appends a second doc, so callers that upsert should
// use distinct ids.
func (ix *Index) Add(id, path, content string) {
	if ix.postings == nil {
		ix.postings = map[Trigram][]int{}
	}
	docID := len(ix.docs)
	ix.docs = append(ix.docs, doc{id: id, path: path, content: content})
	for _, t := range trigrams(content) {
		lst := ix.postings[t]
		// content trigrams are already distinct per doc, so a plain append keeps
		// the posting list sorted (docID is monotonically increasing) and unique.
		ix.postings[t] = append(lst, docID)
	}
}

// DocCount is the number of indexed documents.
func (ix *Index) DocCount() int { return len(ix.docs) }

// SizeBytes returns an estimate of the heap memory used by the index in bytes,
// accounting for document strings, struct headers, and trigram posting lists.
func (ix *Index) SizeBytes() int64 {
	if ix == nil {
		return 0
	}
	var total int64
	total += int64(cap(ix.docs) * 48)
	for _, d := range ix.docs {
		total += int64(len(d.id) + len(d.path) + len(d.content))
	}
	for _, lst := range ix.postings {
		total += int64(48 + cap(lst)*8)
	}
	return total
}

// Compact shrinks postings and doc slice capacities to their exact lengths to minimize memory.
func (ix *Index) Compact() {
	if ix == nil {
		return
	}
	if cap(ix.docs) > len(ix.docs) {
		compactedDocs := make([]doc, len(ix.docs))
		copy(compactedDocs, ix.docs)
		ix.docs = compactedDocs
	}
	for t, lst := range ix.postings {
		if cap(lst) > len(lst) {
			compacted := make([]int, len(lst))
			copy(compacted, lst)
			ix.postings[t] = compacted
		}
	}
}

func (ix *Index) allDocIDs() []int {
	out := make([]int, len(ix.docs))
	for i := range ix.docs {
		out[i] = i
	}
	return out
}

// Candidates returns the document ids that COULD contain literal, using the
// rarest-two-trigram probe. Guarantees:
//
//   - A literal with any trigram whose posting list is empty returns nil — it
//     cannot exist in any document (the freq-0 short-circuit).
//   - A literal shorter than 3 runes cannot be indexed, so every document is a
//     candidate (the caller must verify).
//   - Otherwise the result is a superset of the truly-matching documents (sound:
//     never drops a real match), narrowed by intersecting the two rarest trigrams.
func (ix *Index) Candidates(literal string) []int {
	tris := trigrams(literal)
	if len(tris) == 0 {
		return ix.allDocIDs()
	}
	// Order trigrams by posting-list length (document frequency); a zero-df
	// trigram means the literal is absent everywhere.
	type tf struct {
		t  Trigram
		df int
	}
	tfs := make([]tf, 0, len(tris))
	for _, t := range tris {
		df := len(ix.postings[t])
		if df == 0 {
			return nil // short-circuit: this trigram appears in no document
		}
		tfs = append(tfs, tf{t, df})
	}
	sort.Slice(tfs, func(i, j int) bool { return tfs[i].df < tfs[j].df })

	cand := ix.postings[tfs[0].t]
	if len(tfs) >= 2 {
		cand = intersectSorted(cand, ix.postings[tfs[1].t])
	}
	return append([]int(nil), cand...)
}

// intersectSorted intersects two ascending, de-duplicated int slices.
func intersectSorted(a, b []int) []int {
	out := make([]int, 0, mathx.MinInt(len(a), len(b)))
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i] == b[j]:
			out = append(out, a[i])
			i++
			j++
		case a[i] < b[j]:
			i++
		default:
			j++
		}
	}
	return out
}

// Search returns the documents that contain literal as a substring, verified
// exactly after the trigram pre-filter. Results are ordered by document id.
func (ix *Index) Search(literal string) []Result {
	var out []Result
	for _, id := range ix.Candidates(literal) {
		d := ix.docs[id]
		if lines := lineHitsFunc(d.content, func(line string) bool {
			return strings.Contains(line, literal)
		}); len(lines) > 0 {
			out = append(out, Result{ID: d.id, Path: d.path, Lines: lines})
		}
	}
	return out
}

// Similarity is the Sørensen–Dice coefficient over the DISTINCT trigram sets of a
// and b: 2·|A∩B| / (|A|+|B|), a 0..1 lexical closeness that REUSES the same 3-rune
// shingling the postings index is built on. It is the fuzzy ratio a near-miss or
// synonym lookup falls back to when exact substring matching finds nothing (the
// devindex Search* false-ABSENT fix, #3925), so it lives beside the index rather
// than being re-implemented by every caller.
//
// It is case-sensitive (the index is too); callers that want case-folding lowercase
// both sides first. Two strings too short to shingle (<3 runes) share no trigrams, so
// they compare by exact equality — identical short strings score 1, otherwise 0. Two
// equal strings always score 1 (this also defines empty-vs-empty as identical).
func Similarity(a, b string) float64 {
	if a == b {
		return 1
	}
	ta, tb := trigrams(a), trigrams(b)
	if len(ta) == 0 || len(tb) == 0 {
		return 0 // one side cannot be shingled and they are not equal
	}
	set := make(map[Trigram]bool, len(ta))
	for _, t := range ta {
		set[t] = true
	}
	inter := 0
	for _, t := range tb {
		if set[t] {
			inter++
		}
	}
	return 2 * float64(inter) / float64(len(ta)+len(tb))
}

// SearchRegexp returns the documents matching pattern. It narrows candidates using
// the required literals extracted from the regex (an AND of substrings every match
// must contain), then verifies with the compiled regexp. When no required literal
// can be proven (alternation at top level, or a fully wildcard pattern), it falls
// back to brute force over every document — always sound, never a missed match.
func (ix *Index) SearchRegexp(pattern string) ([]Result, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, err
	}
	cand := ix.regexCandidates(pattern)
	var out []Result
	for _, id := range cand {
		d := ix.docs[id]
		if lines := lineHitsFunc(d.content, re.MatchString); len(lines) > 0 {
			out = append(out, Result{ID: d.id, Path: d.path, Lines: lines})
		}
	}
	return out, nil
}

// regexCandidates computes the candidate document set for a regex by AND-ing the
// candidate sets of every required literal. If none can be proven required it
// returns all documents (brute force).
func (ix *Index) regexCandidates(pattern string) []int {
	lits := requiredLiterals(pattern)
	if len(lits) == 0 {
		return ix.allDocIDs()
	}
	cand := ix.Candidates(lits[0])
	for _, lit := range lits[1:] {
		if len(cand) == 0 {
			break
		}
		cand = intersectSorted(cand, ix.Candidates(lit))
	}
	return cand
}

// requiredLiterals extracts substrings that MUST appear in any string the regex
// matches. It parses the pattern and walks the syntax tree, emitting a literal run
// only from a concatenation of exact-literal nodes; any alternation, star, quest,
// or repeat with a zero lower bound breaks the run (that text is optional). This is
// sound by construction: it never claims a literal is required unless every match
// contains it. Runs shorter than 3 runes are dropped (not indexable).
func requiredLiterals(pattern string) []string {
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return nil
	}
	var out []string
	var walk func(*syntax.Regexp) []rune // returns a required literal run flowing out of this node, if fully literal
	walk = func(r *syntax.Regexp) []rune {
		switch r.Op {
		case syntax.OpLiteral:
			return append([]rune(nil), r.Rune...)
		case syntax.OpConcat:
			var run []rune
			for _, sub := range r.Sub {
				if lit := walk(sub); lit != nil {
					run = append(run, lit...)
					continue
				}
				// a non-literal child ends the current run
				emit(&out, run)
				run = nil
			}
			emit(&out, run)
			return nil
		case syntax.OpCapture:
			if len(r.Sub) == 1 {
				return walk(r.Sub[0])
			}
			return nil
		case syntax.OpPlus:
			// x+ requires at least one x; if x is a literal it is still required.
			if len(r.Sub) == 1 {
				if lit := walk(r.Sub[0]); lit != nil {
					return lit
				}
			}
			return nil
		default:
			// Star / Quest / Repeat / Alternate / CharClass / anchors: nothing under
			// an optional or alternated node is guaranteed to appear in a match, so
			// emit NOTHING and claim no required literal. Treating a min>=1 Repeat as
			// non-required only widens the candidate set — still sound, never a miss.
			return nil
		}
	}
	emit(&out, walk(re))
	return out
}

// emit records run as a required literal when it is long enough to index.
func emit(out *[]string, run []rune) {
	if len(run) >= 3 {
		*out = append(*out, string(run))
	}
}

// lineHitsFunc returns the 1-based line numbers of content for which pred is true.
func lineHitsFunc(content string, pred func(string) bool) []int {
	var lines []int
	for i, line := range strings.Split(content, "\n") {
		if pred(line) {
			lines = append(lines, i+1)
		}
	}
	return lines
}
