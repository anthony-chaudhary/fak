package recall

import "sort"

// journal_index.go — issue #2840: a cross-session recall INDEX over the adjudicated
// journal, with witnessed provenance.
//
// Hermes' session search (tools/session_search_tool.py, hermes #19434) is an FTS5
// index over raw transcripts whose recall ordering (`_order_for_recall`) DEMOTES
// cron/automation sources below interactive ones — a recency / source-type heuristic.
// It indexes raw turns, so it has no notion of WHICH recalled facts were witnessed vs
// merely asserted by a past turn.
//
// fak's journal (internal/journal) is a hash-chained, adjudicated decision log, so
// recall can do better: every indexed row carries PROVENANCE — a witnessed outcome, a
// kept skill, or an un-verified model claim — and the ordering key is provenance-aware,
// not just recency/FTS rank. We adopt Hermes' recency-demotion idea but SUBORDINATE it
// to trust: a witnessed row outranks an un-witnessed claim of equal recency, and an
// un-witnessed claim is NEVER surfaced as fact (JournalHit.AsFact stays false for it).
//
// The two axes are deliberately kept distinct (the issue's stated confusion risk):
// recency-demotion is Hermes' source-type heuristic; provenance ranking is fak's
// witness-status heuristic. This file's primary sort key is provenance; recency is the
// within-tier tie-break.
//
// Scope: this is the recall READ-PATH only. It is NOT the cross-agent census (#2209),
// NOT a change to the journal hash-chain format, and NOT the curator keep/revert reason
// vocabulary (#2841). It adds no adjudication pass — the journal already carries the
// witnessed/kept/claimed structure this indexes (see ProvenanceFromJournal), so this is
// a pure read-side projection.

// Provenance is the adjudicated trust tier of a recalled journal row. Higher value ==
// more trusted, so the enum value IS the trust rank and the zero value is the weakest
// (fail-closed) tier — an un-verified model claim, matching the recall package's
// fail-closed posture elsewhere (abi.FallbackDeny, ErrUnwitnessed).
type Provenance uint8

const (
	// ProvUnverified is the fail-closed default (the uint8 zero value): an un-verified
	// model claim. A past turn ASSERTED it, but no external witness graded it. It is
	// the least-trusted tier and is NEVER surfaced as fact (Provenance.Fact == false).
	ProvUnverified Provenance = iota
	// ProvKept is a kept skill/fact: promoted into durable memory under an adjudicated
	// keep decision (a capability admitted or version-bound by the cap lifecycle, a
	// curator keep). More trusted than a bare claim, less than a directly witnessed
	// outcome — it passed a keep gate but is not itself a graded outcome.
	ProvKept
	// ProvWitnessed is a witnessed outcome: an adjudicated decision-journal row whose
	// result was graded by an external trust witness (a git-verified ship leaf, a
	// require-witness verdict). The most-trusted tier.
	ProvWitnessed
)

// String renders the provenance tag as the issue's own vocabulary.
func (p Provenance) String() string {
	switch p {
	case ProvWitnessed:
		return "witnessed"
	case ProvKept:
		return "kept"
	case ProvUnverified:
		return "unverified"
	default:
		return "unverified"
	}
}

// trustRank is the provenance ordering key — the primary recall sort axis. It is the
// enum value itself (higher == more trusted), so a witnessed row always outranks a
// kept row, which always outranks an un-verified claim.
func (p Provenance) trustRank() int { return int(p) }

// Fact reports whether recall may surface a row of this provenance AS FACT. A witnessed
// outcome and a kept skill both carry adjudicated backing; an un-verified claim does
// not, so Fact is false for it — the issue-#2840 rule that an un-witnessed claim is
// never surfaced as fact.
func (p Provenance) Fact() bool { return p >= ProvKept }

// ProvenanceFromJournal projects an adjudicated journal row's closed status vocabulary
// — the strings internal/journal.Row already renders (its verdict name, its taint name,
// its capability kind) plus whether the row carries a live bounded-disclosure witness —
// onto a recall Provenance, WITHOUT importing the journal package. The mapping is a
// pure string fold, so a caller wires a real journal.Row in with a one-line adapter and
// recall stays decoupled from the journal's on-disk format (this is the read-path only;
// it adds no adjudication pass and does not reach the census work in #2209).
//
// The fold, fail-closed to ProvUnverified (an un-graded assertion is the weakest tier):
//   - a live witness on the row, a WITNESS verdict, or a trusted-taint adjudicated
//     result → witnessed;
//   - otherwise a kept capability (a skill/tool/agent the cap lifecycle admitted or
//     version-bound — a non-empty CapKind) → kept;
//   - anything else (a tainted, un-witnessed assertion) → unverified.
func ProvenanceFromJournal(verdict, taint, capKind string, hasWitness bool) Provenance {
	if hasWitness || verdict == "WITNESS" || taint == "trusted" {
		return ProvWitnessed
	}
	if capKind != "" {
		return ProvKept
	}
	return ProvUnverified
}

// JournalRow is the provenance projection of one adjudicated journal row that the
// recall index consumes: its recency anchor, the text to index, and its adjudicated
// provenance tag. It is a PROJECTION of a journal.Row, not the row itself — recall
// indexes the witnessed/kept/claimed structure the journal already carries, it does not
// re-adjudicate. The Text is expected to be already safe to surface (a reason label /
// descriptor), never sealed bytes.
type JournalRow struct {
	Seq        int        // monotonic recency anchor (higher == more recent) — the recency-demotion axis
	Text       string     // the recall-visible content of the row
	Provenance Provenance // the adjudicated trust tier
}

// defaultMaxJournalRows bounds how many rows a JournalIndex retains. A long-lived
// cross-session index that Adds a row per adjudicated journal decision would grow
// both rows AND the postings map without limit; the bound keeps the index to its
// most RECENT window (the recency-demotion axis already prefers recent rows, so
// aging out the oldest is aligned with the ranker). Below the bound Recall is
// byte-identical; only once the window overflows do the oldest rows drop.
const defaultMaxJournalRows = 1024

// JournalIndex is an FTS-style inverted index over a set of adjudicated journal rows,
// keyed by token, for provenance-aware recall. It is append-only and deterministic: the
// same rows added in the same order always produce the same ranking.
//
// The retained rows are bounded to a recent window (defaultMaxJournalRows): once the
// window overflows, the oldest rows are dropped and the postings map — whose values
// are absolute row indices that all shift on a trim — is rebuilt over the survivors,
// so it can never grow without limit alongside the rows.
type JournalIndex struct {
	rows     []JournalRow
	postings map[string][]int // token -> indices of rows that contain it (the inverted index)
	// maxRows bounds the retained window: 0 = the default cap; <0 = unbounded; >0 =
	// a custom cap. See rowCap.
	maxRows int
}

// NewJournalIndex returns an empty index ready for Add.
func NewJournalIndex() *JournalIndex {
	return &JournalIndex{postings: map[string][]int{}}
}

// rowCap resolves the effective retained-row cap: (cap, true) when bounded,
// (_, false) when the caller disabled the bound (maxRows<0).
func (idx *JournalIndex) rowCap() (max int, bounded bool) {
	switch {
	case idx.maxRows < 0:
		return 0, false
	case idx.maxRows == 0:
		return defaultMaxJournalRows, true
	default:
		return idx.maxRows, true
	}
}

// Add indexes one journal row, posting each distinct token it contains. The token floor
// is applied at query time by Recall (the same length floor the page-side overlap uses),
// so the index and the live Recall() ranker share one relevance notion.
//
// Retention is applied here: the rows are held to a recent window bounded by rowCap.
// A high-water of 2×cap is allowed before a trim so the O(cap) postings rebuild is
// amortized to O(1) per Add — nothing is dropped until the window exceeds 2×cap, so
// Recall is unchanged for any index below that.
func (idx *JournalIndex) Add(row JournalRow) {
	idx.rows = append(idx.rows, row)
	if max, bounded := idx.rowCap(); bounded && len(idx.rows) > 2*max {
		// Drop the oldest rows down to the cap; postings hold absolute row indices
		// that all shift when the front is trimmed, so rebuild the inverted index.
		idx.rows = append([]JournalRow(nil), idx.rows[len(idx.rows)-max:]...)
		idx.reindex()
		return
	}
	// Fast path: post only the new row's tokens at its index.
	idx.postRow(len(idx.rows)-1, row.Text)
}

// postRow posts row i's DISTINCT tokens into the inverted index. The de-duplication is
// load-bearing: a term repeated inside one row must post that row ONCE, or overlap scoring
// would count the same row several times. Both the incremental Add above and the post-trim
// reindex below go through here, so an index rebuilt from scratch is identical to one grown
// a row at a time — which is the property that makes the trim safe.
func (idx *JournalIndex) postRow(i int, text string) {
	seen := map[string]bool{}
	for _, t := range tokenize(text) {
		if seen[t] {
			continue
		}
		seen[t] = true
		idx.postings[t] = append(idx.postings[t], i)
	}
}

// reindex rebuilds the postings map from scratch over the current rows — used after
// a retention trim shifts every row index. O(total tokens in the retained window).
func (idx *JournalIndex) reindex() {
	idx.postings = make(map[string][]int, len(idx.postings))
	for i, r := range idx.rows {
		idx.postRow(i, r.Text)
	}
}

// JournalHit is one provenance-tagged recall result: the row's content, its recency
// anchor, its FTS relevance score, its adjudicated provenance, and whether recall is
// allowed to surface it AS FACT (never true for an un-witnessed claim).
type JournalHit struct {
	Seq        int        `json:"seq"`
	Text       string     `json:"text"`
	Score      int        `json:"score"`      // FTS-style token-overlap relevance (the hard candidacy gate)
	Provenance Provenance `json:"provenance"` // the adjudicated trust tier this row was tagged with
	AsFact     bool       `json:"as_fact"`    // provenance-gated: false for an un-verified claim so a caller never renders it as established fact
}

// Recall returns the top-k rows matching query, ranked PROVENANCE-FIRST: a witnessed
// row outranks a kept row outranks an un-verified claim; within a provenance tier the
// more recent row ranks higher (Hermes' recency-demotion idea, adopted but subordinated
// to trust); ties in recency fall to FTS relevance, then to index order for a stable,
// deterministic result.
//
// Relevance is a HARD candidacy gate — a score-0 (irrelevant) row is never returned, so
// provenance re-ranks WITHIN the relevant set and never widens it (the same discipline
// the page-side Recall uses). Every returned hit is tagged with its provenance, and
// AsFact is set ONLY for a witnessed/kept row: an un-witnessed claim may be returned as
// a relevant lead but is NEVER surfaced as fact.
func (idx *JournalIndex) Recall(query string, k int) []JournalHit {
	scores := idx.score(query)
	cands := make([]int, 0, len(scores))
	for i := range idx.rows {
		if scores[i] > 0 {
			cands = append(cands, i)
		}
	}
	sort.SliceStable(cands, func(a, b int) bool {
		ra, rb := idx.rows[cands[a]], idx.rows[cands[b]]
		if ra.Provenance != rb.Provenance {
			return ra.Provenance.trustRank() > rb.Provenance.trustRank() // provenance is the primary key
		}
		if ra.Seq != rb.Seq {
			return ra.Seq > rb.Seq // within a tier: recency-demotion (more recent first)
		}
		if scores[cands[a]] != scores[cands[b]] {
			return scores[cands[a]] > scores[cands[b]] // then FTS relevance
		}
		return cands[a] < cands[b] // stable: index (recording) order
	})
	// #3940: optional MMR redundancy suppression, applied WITHIN each provenance run so a
	// near-duplicate row loses its top-k slot to a novel one without ever crossing the
	// trust boundary. Unless mmrEnv is armed this does nothing and the ordering above is
	// the final one, byte-identical to pre-#3940. See mmr.go.
	if mmrEnabled() {
		cands = idx.mmrRerank(cands, scores, mmrLambda(), k)
	}
	out := make([]JournalHit, 0, k)
	for _, i := range cands {
		if len(out) >= k {
			break
		}
		r := idx.rows[i]
		out = append(out, JournalHit{
			Seq: r.Seq, Text: r.Text, Score: scores[i],
			Provenance: r.Provenance, AsFact: r.Provenance.Fact(),
		})
	}
	return out
}

// RecallFacts is Recall filtered to rows recall may surface AS FACT: it drops every
// un-verified claim, returning only witnessed/kept rows (still provenance-then-recency
// ranked). It is the "filter by witness status" read verb — the refusal to surface an
// un-witnessed claim as fact expressed as a filter, complementing Recall's ranked mode
// that returns the claim but tags it AsFact=false.
func (idx *JournalIndex) RecallFacts(query string, k int) []JournalHit {
	// Rank over the full candidate set first (so provenance/recency ordering is
	// identical to Recall), then keep only the fact-surfaceable prefix up to k.
	ranked := idx.Recall(query, len(idx.rows))
	out := make([]JournalHit, 0, k)
	for _, h := range ranked {
		if len(out) >= k {
			break
		}
		if h.AsFact {
			out = append(out, h)
		}
	}
	return out
}

// score computes the FTS-style relevance of every row for query: the count of distinct
// query tokens (past the length floor) present in the row, read straight off the
// inverted index. It equals overlap(query, row.Text) for every row, so the index ranker
// and the page-side overlap agree.
func (idx *JournalIndex) score(query string) map[int]int {
	qset := map[string]bool{}
	for _, t := range tokenize(query) {
		if len(t) > 2 {
			qset[t] = true
		}
	}
	scores := map[int]int{}
	for t := range qset {
		for _, i := range idx.postings[t] {
			scores[i]++
		}
	}
	return scores
}
