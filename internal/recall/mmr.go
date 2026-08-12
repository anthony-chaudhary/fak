package recall

import (
	"os"
	"strconv"
	"strings"
)

// mmr.go — issue #3940: MMR-style redundancy suppression inside the top-k journal
// Recall() selection (the one borrow filed by the Puppetmaster study,
// docs/notes/study-puppetmaster-borrow-scout-2026-07-10.md, row 3 — a Python→Go
// clean-room port of its `finalize_memory_retrieval` reranker).
//
// The defect it closes: journal Recall() ranks provenance → recency → FTS-overlap →
// index order and then takes the top k. Nothing in that key penalizes REDUNDANCY, so k
// near-duplicate rows ("cache lease expired on gateway alpha" / "… beta" / "… gamma")
// all get injected, spending the very context budget recall exists to conserve. Maximal
// Marginal Relevance (Carbonell & Goldstein, 1998) is the standard fix: greedily pick
// each next row to maximize
//
//	lambda*relevance(row) - (1-lambda)*max_similarity(row, already-selected)
//
// so a row that says something the selection already says loses its slot to a novel one.
//
// The fak-specific constraint — and the reason this is not a straight port — is that the
// journal ranker is PROVENANCE-FIRST (journal_index.go): a witnessed outcome must never
// be demoted below an un-verified claim, whatever a diversity term says. So MMR runs
// strictly WITHIN each provenance run of the already-ranked candidate list, never across
// runs. Diversity therefore reorders trust-equals only; the trust boundary is structural,
// not a weight that a small enough lambda could overpower
// (TestRecallMMRNeverPromotesClaimAboveWitness pins that at lambda=0, the most
// diversity-aggressive setting there is).
// Similarity is still measured against EVERY already-selected row, including ones picked
// from a higher tier, so a claim that merely restates a witnessed row already in the
// selection is penalized — it just cannot thereby outrank that witnessed row.
//
// Off by default and env-gated (like Puppetmaster's PUPPETMASTER_MEMORY_MMR): with the
// gate unset, Recall's ordering is byte-identical to before this file existed.
const (
	// mmrEnv gates the redundancy suppressor. Unset/empty/false => off, and Recall keeps
	// its exact pre-#3940 provenance → recency → relevance → index ordering.
	mmrEnv = "FAK_RECALL_MMR"
	// mmrLambdaEnv tunes the relevance/diversity trade-off in [0,1]: 1 = pure relevance
	// (MMR becomes a no-op reordering), 0 = pure novelty. Out-of-range values clamp;
	// an unparseable one falls back to defaultMMRLambda.
	mmrLambdaEnv = "FAK_RECALL_MMR_LAMBDA"
)

// defaultMMRLambda keeps relevance dominant (the borrow's 0.7): the diversity term breaks
// near-ties between comparably relevant rows rather than dragging a weak-but-novel row up
// past a strongly relevant one.
const defaultMMRLambda = 0.7

// mmrPoolFactor bounds the reranked window to a small multiple of k (Puppetmaster reranks
// a 3xlimit pool). Greedy MMR is O(pool^2) in similarity comparisons, so bounding the pool
// keeps the cost proportional to what the caller actually asked for; candidates past the
// pool keep their baseline order and can only matter when the pool is the whole list.
const mmrPoolFactor = 3

// mmrEnabled reports whether the redundancy suppressor is armed. Fail-closed: anything
// that is not an explicit truthy value leaves Recall's committed ordering untouched.
func mmrEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(mmrEnv))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// mmrLambda resolves the relevance weight from the environment, clamped to [0,1].
func mmrLambda() float64 {
	raw := strings.TrimSpace(os.Getenv(mmrLambdaEnv))
	if raw == "" {
		return defaultMMRLambda
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return defaultMMRLambda
	}
	switch {
	case v < 0:
		return 0
	case v > 1:
		return 1
	}
	return v
}

// mmrRerank reorders an ALREADY provenance-ranked candidate list by greedy MMR, within
// each provenance run. cands holds row indices in Recall's baseline order; scores is the
// FTS relevance map Recall already computed (reused, not recomputed — the borrow's
// "reuse the overlap score score() already computes"); k is the caller's top-k.
//
// Determinism: every loop walks slices, and the best-candidate scan keeps the FIRST
// maximum (strict >), so a tie falls back to the baseline order and repeated calls on an
// unchanged index return byte-identical results
// (TestRecallMMRDiversifiesWithinWitnessedTierDeterministically calls Recall twice and
// compares).
func (idx *JournalIndex) mmrRerank(cands []int, scores map[int]int, lambda float64, k int) []int {
	if len(cands) < 2 || k <= 0 {
		return cands
	}
	pool := len(cands)
	if k < len(cands) {
		if p := k * mmrPoolFactor; p < pool {
			pool = p
		}
	}
	if pool < 2 {
		return cands
	}

	toks := make([]map[string]bool, pool)
	maxRel := 0
	for i := 0; i < pool; i++ {
		toks[i] = mmrTokenSet(idx.rows[cands[i]].Text)
		if s := scores[cands[i]]; s > maxRel {
			maxRel = s
		}
	}

	out := make([]int, 0, len(cands))
	taken := make([]bool, pool)
	// sim[j] is the running max similarity of pool candidate j to any already-selected
	// row — maintained incrementally, so each selection costs one O(pool) update pass
	// rather than a rescan of the whole selection.
	sim := make([]float64, pool)

	// Walk the contiguous runs of equal provenance. cands is sorted provenance-first, so
	// a run is exactly one trust tier, and confining selection to it is what makes the
	// "never promote a claim above a witness" property structural.
	for start := 0; start < pool; {
		end := start + 1
		for end < pool && idx.rows[cands[end]].Provenance == idx.rows[cands[start]].Provenance {
			end++
		}
		for picked := start; picked < end; picked++ {
			best, bestScore := -1, 0.0
			for j := start; j < end; j++ {
				if taken[j] {
					continue
				}
				s := lambda*mmrRelevance(scores[cands[j]], maxRel) - (1-lambda)*sim[j]
				if best < 0 || s > bestScore {
					best, bestScore = j, s
				}
			}
			taken[best] = true
			out = append(out, cands[best])
			for j := 0; j < pool; j++ {
				if taken[j] {
					continue
				}
				if v := mmrJaccard(toks[best], toks[j]); v > sim[j] {
					sim[j] = v
				}
			}
		}
		start = end
	}
	return append(out, cands[pool:]...)
}

// mmrRelevance normalizes an FTS overlap score into [0,1] against the strongest candidate
// in the pool, so the relevance and similarity terms are on one scale and lambda means the
// same thing for a one-token query as for a ten-token one.
func mmrRelevance(score, maxScore int) float64 {
	if maxScore <= 0 {
		return 0
	}
	return float64(score) / float64(maxScore)
}

// mmrTokenSet is the distinct-token set of a row's text, past the same >2 length floor the
// index's relevance scoring uses, so "similar" and "relevant" are measured over one
// vocabulary.
func mmrTokenSet(text string) map[string]bool {
	set := map[string]bool{}
	for _, t := range tokenize(text) {
		if len(t) > 2 {
			set[t] = true
		}
	}
	return set
}

// mmrJaccard is word-set Jaccard similarity in [0,1] — the borrow's similarity measure. It
// needs no embedding model or extra state, which keeps recall's read path dependency-free
// and deterministic.
func mmrJaccard(a, b map[string]bool) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	small, large := a, b
	if len(large) < len(small) {
		small, large = large, small
	}
	inter := 0
	for t := range small {
		if large[t] {
			inter++
		}
	}
	if inter == 0 {
		return 0
	}
	return float64(inter) / float64(len(a)+len(b)-inter)
}
