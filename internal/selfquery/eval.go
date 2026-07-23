package selfquery

import (
	"fmt"
	"sort"
	"strings"
)

// Retrieval-quality evaluation for the card ranker (#3162, epic #1494).
//
// rankCards decides which cards a fak-guarded agent gets back when it asks
// "what can I do for THIS task?". Before this file the only grade on that
// ranker was RELATIVE: ranker_test.go's A/B proved the hybrid retriever beat
// the retained flat lexical baseline (#3235). A relative win says nothing about
// whether the top-K actually contains the right card, so any later score()
// tweak was still a vibe. This harness publishes the ABSOLUTE numbers —
// recall@K and MRR over a labeled intent set — so a ranking change that drops
// retrieval below a recorded floor fails a gate instead of landing silently.
//
// Fixture provenance (read this before quoting a number). #3162 proposed
// seeding the labels for free from "the queries phrasings each capability card
// already registers". No such field exists: FeatureCard carries Kind/Name/
// Summary/Tags/DetailRef/Source/Witness and nothing resembling a registered
// query phrasing, and the cards are synthesized from tool descriptors, memory
// drivers and index rows rather than from any authored intent list. So the
// labels here are HAND-AUTHORED, carried over from the #3235 A/B fixture. That
// is why the set is small, and why EvalResult reports coverage: an
// intent-to-card label is a human judgement about what a user meant, and this
// repo does not have a free source of them.
//
// Grades retrieval, not answer correctness: a hit means the intended card was
// returned in the top K, not that acting on it would work.

// IntentLabel is one ground-truth retrieval label: a natural-language intent
// and the FeatureCard.Name that a user issuing that intent means.
type IntentLabel struct {
	Intent string `json:"intent"`
	Want   string `json:"want"`
}

// Ranker is the retrieval function under grade. rankCards (the production
// hybrid retriever) and any candidate replacement share this shape, so a
// challenger can be graded against the same fixture without touching callers.
type Ranker func([]FeatureCard, string) []FeatureCard

// EvalResult is the graded retrieval report for one ranker over one fixture.
//
// RecallAtK is the fraction of intents whose intended card appeared in the top
// K — "did the agent see the right card at all". MRR is the mean reciprocal
// rank of that card over the FULL ranked list (1.0 = always first, 0.5 =
// typically second); it is deliberately not truncated at K so that a change
// which merely demotes a card, without pushing it out of the window, still
// moves a number. Recall answers "did we surface it", MRR answers "how near
// the top".
//
// CardsCovered/CardsTotal exist to keep the fence honest: they say how much of
// the catalog the fixture actually exercises, so a high score is never read as
// full coverage. Misses names the intents that failed, so a regression report
// points at the specific label to look at rather than just a delta.
type EvalResult struct {
	K            int      `json:"k"`
	N            int      `json:"n"`
	RecallAtK    float64  `json:"recall_at_k"`
	MRR          float64  `json:"mrr"`
	CardsCovered int      `json:"cards_covered"`
	CardsTotal   int      `json:"cards_total"`
	Misses       []string `json:"misses,omitempty"`
}

// Unmeasured renders the share of the catalog the fixture never asks for. The
// issue's fence is to log what is unmeasured rather than imply full coverage,
// so callers print this next to any quoted recall number.
func (r EvalResult) Unmeasured() string {
	return fmt.Sprintf("%d/%d cards carry no labeled intent", r.CardsTotal-r.CardsCovered, r.CardsTotal)
}

// String is the one-line witness form: the numbers plus their coverage caveat.
func (r EvalResult) String() string {
	return fmt.Sprintf("recall@%d=%.3f mrr=%.3f (n=%d intents; %s)", r.K, r.RecallAtK, r.MRR, r.N, r.Unmeasured())
}

// EvalRetrieval grades rank over fixture and reports recall@K and MRR.
//
// An intent whose intended card is absent from the candidate set is still
// counted in the denominator: a card the ranker cannot possibly return is a
// retrieval failure, not an excused sample. That keeps the number honest when
// the catalog shrinks.
func EvalRetrieval(rank Ranker, cards []FeatureCard, fixture []IntentLabel, k int) EvalResult {
	res := EvalResult{K: k, N: len(fixture), CardsTotal: len(cards)}
	if len(fixture) == 0 || k <= 0 {
		return res
	}
	covered := map[string]bool{}
	hits, rrSum := 0, 0.0
	for _, lab := range fixture {
		covered[lab.Want] = true
		pos := rankOf(rank(cards, lab.Intent), lab.Want)
		switch {
		case pos < 0:
			res.Misses = append(res.Misses, lab.Intent)
		case pos < k:
			hits++
			rrSum += 1.0 / float64(pos+1)
		default:
			// Returned, but below the window the agent actually sees: no
			// recall credit, partial MRR credit.
			rrSum += 1.0 / float64(pos+1)
			res.Misses = append(res.Misses, lab.Intent)
		}
	}
	// Count only labeled cards that the catalog can actually serve, so coverage
	// never claims a card the fixture names but the index does not hold.
	present := map[string]bool{}
	for _, c := range cards {
		present[c.Name] = true
	}
	for name := range covered {
		if present[name] {
			res.CardsCovered++
		}
	}
	res.RecallAtK = float64(hits) / float64(len(fixture))
	res.MRR = rrSum / float64(len(fixture))
	sort.Strings(res.Misses)
	return res
}

// rankOf returns the 0-based position of the named card in a ranked list, or
// -1 when the ranker did not return it at all. Names are compared exactly:
// these labels are catalog card names, not user-typed keys, so the fuzzy
// findCard matching would let a near-miss score as a hit.
func rankOf(ranked []FeatureCard, want string) int {
	for i, c := range ranked {
		if c.Name == want {
			return i
		}
	}
	return -1
}

// DefaultIntentFixture is the labeled intent set the ranker is graded on.
//
// Hand-authored (see the provenance note above — there is no registered-query
// field to harvest). The intents mix realistic multi-word asks with sparse
// morphological variants ("compaction" for a card named "compact") so that the
// stemming and IDF rungs, not substring luck, are what decide the score.
var DefaultIntentFixture = []IntentLabel{
	// Realistic multi-word intents any competent ranker should satisfy.
	{Intent: "prove a commit matches its diff", Want: "dos_commit_audit"},
	{Intent: "which agent may take this lane right now", Want: "dos_arbitrate"},
	{Intent: "search for a tool schema on demand", Want: "fak_tools_search"},
	{Intent: "compacting my resident context window", Want: "memory-driver:compact"},
	{Intent: "did this phase actually ship", Want: "dos_verify"},
	{Intent: "commit stamp for a path", Want: "fak index lane"},
	// Sparse morphological intents: the only signal is a variant a flat
	// substring scorer cannot reach ("compaction" is not a substring of
	// "compact").
	{Intent: "compaction", Want: "memory-driver:compact"},
	{Intent: "auditing", Want: "dos_commit_audit"},
	{Intent: "citations", Want: "dos_citation_resolve"},
}

// FormatEvalReport renders a human-readable grade, including the per-intent
// misses that a bare pair of numbers would hide.
func FormatEvalReport(res EvalResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s\n", res)
	if len(res.Misses) == 0 {
		fmt.Fprintf(&b, "misses: none\n")
		return b.String()
	}
	fmt.Fprintf(&b, "misses (intended card outside top %d):\n", res.K)
	for _, m := range res.Misses {
		fmt.Fprintf(&b, "  - %q\n", m)
	}
	return b.String()
}
