package selfquery

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/simhash"
)

// Reciprocal Rank Fusion (#3436, epic #3434 — the highest-ROI seam: pure
// plumbing over two retrieval arms the repo already ships). RRF fuses N ranked
// lists by RANK, not score:
//
//	fused(item) = Σ_over_lists 1 / (k_list + rank + 1)
//
// Fusing ranks is what lets the BM25 arm (ranker.go, a lexical IDF score) and the
// simhash arm (internal/simhash, a cosine over char-ngram embeddings) combine with
// NO cross-system score normalization — their score scales are incomparable, but
// their ranks are not. This is the same construction codesearch's reciprocal-rank
// fusion uses; DefaultRRFK == 20 reproduces its 1/(20+rank+1) weighting, so a card
// ranked first (rank 0) by two arms scores 1/21 + 1/21.
const (
	// DefaultRRFK is the neutral fusion constant used when a per-list k is not
	// supplied. Larger k flattens the contribution of rank (every position counts
	// nearly the same); smaller k sharpens it (the top few dominate).
	DefaultRRFK = 20.0
)

// QueryClass is the coarse shape of a query, used to bias fusion toward the arm
// that shape trusts (adapt_rrf_k). An identifier-like query ("cardLess") trusts
// exact lexical retrieval; a natural-language query ("how do I rotate logs")
// trusts the semantic arm; a structural query ("foo(.*)") sits between.
type QueryClass int

const (
	ClassSemantic QueryClass = iota
	ClassIdentifier
	ClassStructural
)

// AdaptRRFK returns the (ftsK, vecK) fusion constants for a query class — the
// query-adaptive k table (codesearch adapt_rrf_k). A SMALLER k gives an arm MORE
// influence (its 1/(k+rank+1) terms are larger), so an identifier query hands the
// lexical/BM25 arm the small k; a semantic query keeps the arms balanced.
//
//	identifier -> fts 12, vec 28   (trust exact lexical)
//	structural -> fts 15, vec 25
//	semantic   -> fts 20, vec 20   (balanced)
func AdaptRRFK(class QueryClass) (ftsK, vecK float64) {
	switch class {
	case ClassIdentifier:
		return 12, 28
	case ClassStructural:
		return 15, 25
	default:
		return 20, 20
	}
}

// ClassifyQuery buckets a raw query by shape. Structural (contains code
// metacharacters) is checked first, then identifier (a single identifier-shaped
// token), else semantic. Deterministic and dependency-free — same query, same
// class.
func ClassifyQuery(q string) QueryClass {
	q = strings.TrimSpace(q)
	if q == "" {
		return ClassSemantic
	}
	if strings.ContainsAny(q, "().{}[]$*?:/\\|") {
		return ClassStructural
	}
	if !strings.ContainsAny(q, " \t\n") && isIdentifierToken(q) {
		return ClassIdentifier
	}
	return ClassSemantic
}

func isIdentifierToken(s string) bool {
	for i, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r == '_':
		case r >= '0' && r <= '9':
			if i == 0 {
				return false // an identifier does not start with a digit
			}
		default:
			return false
		}
	}
	return true
}

// fuseKey is a card's identity for fusion: the same (Root, Kind, Name, DetailRef)
// card surfaced by both arms must fold into ONE fused row, or its rank-contributions
// never add. Root is included so a cross-repo union (#3435) keeps same-named cards
// from different checkouts distinct.
func fuseKey(c FeatureCard) string { return c.Root + "\x00" + cardKey(c) }

// scoredCard is a fused row: the card plus its accumulated reciprocal-rank score
// and the best (lowest) rank it achieved in any arm. Exposed to tests so the exact
// 1/(k+rank+1) weighting is checkable, not just the final order.
type scoredCard struct {
	Card     FeatureCard
	Score    float64
	BestRank int
}

// fuseRanked is the scoring core of RRF: it accumulates each card's reciprocal-rank
// contribution across the arms and returns the fused rows in ranked order. RRF wraps
// it to drop the scores.
func fuseRanked(lists [][]FeatureCard, ks []float64) []scoredCard {
	scores := map[string]*scoredCard{}
	order := make([]string, 0)
	for li, list := range lists {
		k := DefaultRRFK
		if li < len(ks) && ks[li] > 0 {
			k = ks[li]
		}
		for rank, c := range list {
			key := fuseKey(c)
			a := scores[key]
			if a == nil {
				a = &scoredCard{Card: c, BestRank: rank}
				scores[key] = a
				order = append(order, key)
			}
			a.Score += 1.0 / (k + float64(rank) + 1.0)
			if rank < a.BestRank {
				a.BestRank = rank
			}
		}
	}
	fused := make([]scoredCard, 0, len(order))
	for _, key := range order {
		fused = append(fused, *scores[key])
	}
	sort.SliceStable(fused, func(i, j int) bool {
		if fused[i].Score != fused[j].Score {
			return fused[i].Score > fused[j].Score
		}
		if fused[i].BestRank != fused[j].BestRank {
			return fused[i].BestRank < fused[j].BestRank
		}
		return cardLess(fused[i].Card, fused[j].Card)
	})
	return fused
}

// RRF fuses ranked card lists by reciprocal rank. ks[i] is the fusion constant for
// lists[i]; a missing or non-positive entry falls back to DefaultRRFK. Cards are
// keyed by fuseKey so a card appearing in several arms accumulates their
// contributions. Output is sorted by fused score descending, tiebroken by best
// (lowest) achieved rank, then cardLess for a total, deterministic order.
func RRF(lists [][]FeatureCard, ks []float64) []FeatureCard {
	fused := fuseRanked(lists, ks)
	out := make([]FeatureCard, len(fused))
	for i, a := range fused {
		out[i] = a.Card
	}
	return out
}

// cardEmbedText is the text the semantic arm embeds for a card — the same fields
// the lexical arm weighs, joined so char-ngram similarity can bridge a query that
// misses the exact stems (a typo, a truncation, a near-synonym).
func cardEmbedText(c FeatureCard) string {
	return strings.Join([]string{c.Name, c.Summary, strings.Join(c.Tags, " "), c.DetailRef}, " ")
}

// simhashArm ranks cards by cosine similarity of their embedded text to the query,
// returning the positive-similarity matches best-first — the second retrieval arm
// RRF fuses against BM25. It is intentionally recall-oriented: it surfaces cards a
// lexical arm misses, and RRF's rank weighting keeps low-similarity noise from
// dominating.
func simhashArm(cards []FeatureCard, q string, k int) []FeatureCard {
	if len(cards) == 0 || strings.TrimSpace(q) == "" {
		return nil
	}
	var ix simhash.Index
	idx := make(map[string]int, len(cards))
	for i, c := range cards {
		id := fuseKey(c)
		idx[id] = i
		ix.AddText(id, cardEmbedText(c), "")
	}
	matches := ix.TopK(simhash.Embed(q), k)
	out := make([]FeatureCard, 0, len(matches))
	for _, m := range matches {
		if m.Score <= 0 {
			continue // orthogonal card: not a retrieval, do not inject as noise
		}
		if i, ok := idx[m.ID]; ok {
			out = append(out, cards[i])
		}
	}
	return out
}

// HybridRRF is the shippable wiring: rank cards by the BM25 arm and the simhash
// arm, then fuse the two by reciprocal rank with query-adaptive k. This is the
// retrieval upgrade the epic buys with the least new surface — both arms already
// exist; RRF only decides how their ranks combine.
func HybridRRF(cards []FeatureCard, q string) []FeatureCard {
	if strings.TrimSpace(q) == "" {
		return nil
	}
	ftsK, vecK := AdaptRRFK(ClassifyQuery(q))
	fts := rankHybrid(cards, q)
	vec := simhashArm(cards, q, len(cards))
	return RRF([][]FeatureCard{fts, vec}, []float64{ftsK, vecK})
}
