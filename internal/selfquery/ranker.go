package selfquery

import (
	"math"
	"sort"
)

// ranker.go is the hybrid BM25 + deterministic-expansion retriever behind
// rankCards (#3235, epic #3229 — the deferral recall lever). It replaces the
// flat lexical token-overlap scorer (score(), retained below as the A/B
// baseline) that gated fak_feature_query / fak_capabilities.
//
// Why the flat scorer was the failure mode: it credited every matched token by
// a fixed per-field weight, so the boilerplate tags EVERY live card carries —
// `live`, `mcp`, `tool` (toolTags) — scored exactly like a discriminating term.
// A generic or tag-shaped query dragged the whole live plane forward, and the
// only fix in the tree was hand-maintained per-driver synonym lists
// (memoryHygieneSynonyms). Deferral (#3231/#3232) makes retrieval load-bearing:
// a tool the model cannot re-find from its intent becomes an invisible
// capability, not merely a longer prompt. So recall has to come from a
// corpus-wide mechanism, not a growing hand list.
//
// Two rungs, both offline and deterministic (same constraint as #3230's
// scorecard — no network, no embedding; an embeddings rung via
// internal/gateway/embeddings.go is a documented follow-on, not v1):
//
//  1. A BM25 lexical rung with IDF computed over the CURRENT candidate corpus,
//     so a term's weight falls as it gets more ubiquitous. `live`/`mcp`/`tool`,
//     present in nearly every live card, collapse to ~zero IDF and stop
//     dominating; a rare, discriminating term dominates instead.
//  2. A deterministic expansion rung — light suffix stemming applied
//     SYMMETRICALLY to the query and the corpus — so morphological variants
//     ("compact" / "compacting" / "compaction") match without a synonym entry.
//     This generalizes what memoryHygieneSynonyms did by hand into a
//     corpus-wide mechanism; the seed list is retained in capabilities.go only
//     as a documented cross-driver semantic cluster (the case pure lexical
//     stemming cannot reach), pending the embeddings rung.
//
// The result is fed through rankCards' existing stable tiebreak (cardLess) so
// output stays deterministic for identical inputs.

// BM25 free parameters (Robertson/Sparck-Jones defaults). k1 controls
// term-frequency saturation; b controls length normalization.
const (
	bm25K1 = 1.5
	// bm25B is set well below the 0.75 prose default: feature cards are SHORT,
	// roughly uniform metadata records (a name, a one-line summary, a handful of
	// tags), not variable-length documents. Aggressive length normalization is
	// designed for corpora where long docs spuriously match more terms; here it
	// would over-reward the shortest card (a bare leaf record whose only tie to
	// a term is a boilerplate tag) over a card that names the term in BOTH its
	// summary and tags — the genuinely more-relevant one. Mild normalization
	// keeps the intent-owning card on top. Independent of the IDF rung that
	// solves the ubiquitous-tag problem.
	bm25B = 0.3

	// Field weights act as the term-frequency multiplier a term earns from the
	// field it appears in: a name/tag hit is worth more raw TF than a summary
	// hit, preserving the flat scorer's field ordering (name/tag >> ref >>
	// summary) inside the BM25 saturation curve rather than as a flat add.
	fwName    = 3.0
	fwTag     = 3.0
	fwRefKind = 1.0
	fwSummary = 1.0

	// exactNameTagBoost rewards a query term that IS a card's name-part or an
	// exact tag, scaled by that term's IDF — so an exact match on a
	// discriminating term (high IDF) reliably tops a mere summary mention,
	// while an exact match on a ubiquitous tag (near-zero IDF) earns almost
	// nothing. This preserves the old scorer's "the driver whose name is in the
	// intent wins" behavior without letting boilerplate tags win by exactness.
	exactNameTagBoost = 4.0
)

// stemToken applies a conservative, deterministic suffix strip so morphological
// variants collapse to a shared key. It is intentionally crude (not a real
// Porter stemmer): correctness matters less than being applied identically to
// both the query and the corpus, which is what makes a variant match.
//
// It iterates to a fixpoint, stripping one suffix per pass, because variants can
// differ by where a single pass lands: "citations" sheds "s" -> "citation",
// while "citation" sheds "ion" -> "citat" — only a second pass on "citation"
// reunites them at "citat". Each pass removes >=1 char and only fires when the
// remainder stays >= 3 chars, so it always terminates and never over-strips a
// short discriminating token ("index", "lane", "live") into a collision.
func stemToken(t string) string {
	for {
		next := stripOneSuffix(t)
		if next == t {
			return t
		}
		t = next
	}
}

func stripOneSuffix(t string) string {
	for _, suf := range stemSuffixes {
		if len(t) >= len(suf)+3 && hasSuffixASCII(t, suf) {
			return t[:len(t)-len(suf)]
		}
	}
	return t
}

// stemSuffixes is the atomic suffix set, longest-first within a pass. The
// over-broad "ation"/"ions" are deliberately absent: iteration reaches the same
// stem via "s"+"ion" without "ation" also chopping "citation" -> "cit".
var stemSuffixes = []string{"ing", "ies", "ion", "ers", "ise", "ize", "ed", "es", "er", "ly", "s"}

func hasSuffixASCII(s, suf string) bool {
	if len(s) < len(suf) {
		return false
	}
	return s[len(s)-len(suf):] == suf
}

// stemAll stems a token slice, dropping empties.
func stemAll(toks []string) []string {
	out := make([]string, 0, len(toks))
	for _, t := range toks {
		s := stemToken(t)
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

// cardTerms builds a card's field-weighted, stemmed term-frequency bag: the
// document BM25 scores against. Every field routes through tokens() (the same
// [a-z0-9] tokenizer the flat scorer used) then stemToken, so "memory-driver:compact"
// contributes {memory, driver, compact} and matches a stemmed query symmetrically.
func cardTerms(c FeatureCard) map[string]float64 {
	bag := map[string]float64{}
	add := func(text string, w float64) {
		for _, tk := range tokens(text) {
			bag[stemToken(tk)] += w
		}
	}
	add(c.Name, fwName)
	for _, tag := range c.Tags {
		add(tag, fwTag)
	}
	add(c.DetailRef+" "+c.Kind+" "+c.Source, fwRefKind)
	add(c.Summary, fwSummary)
	return bag
}

// nameTagStemSet is the set of stemmed name-parts and exact tags of a card —
// the keys eligible for the exact-match boost.
func nameTagStemSet(c FeatureCard) map[string]bool {
	set := map[string]bool{}
	for _, tk := range tokens(c.Name) {
		set[stemToken(tk)] = true
	}
	for _, tag := range c.Tags {
		for _, tk := range tokens(tag) {
			set[stemToken(tk)] = true
		}
	}
	return set
}

// corpusModel is the IDF table + average document length over a candidate set,
// rebuilt per rankCards call (the candidate set differs by plane/query surface,
// and IDF only means anything relative to the set actually being ranked).
type corpusModel struct {
	idf    map[string]float64
	avgLen float64
}

func buildCorpusModel(bags []map[string]float64) corpusModel {
	df := map[string]int{}
	totalLen := 0.0
	for _, bag := range bags {
		l := 0.0
		for t, w := range bag {
			l += w
			df[t]++
		}
		totalLen += l
	}
	n := len(bags)
	idf := make(map[string]float64, len(df))
	for t, d := range df {
		// BM25 IDF with the +1 inside the log so it is always >= 0: a term in
		// every doc (d==n) yields a small positive weight, never negative.
		idf[t] = math.Log(1 + (float64(n)-float64(d)+0.5)/(float64(d)+0.5))
	}
	avg := 0.0
	if n > 0 {
		avg = totalLen / float64(n)
	}
	return corpusModel{idf: idf, avgLen: avg}
}

// bm25Score scores one document bag against the stemmed query terms.
func bm25Score(bag map[string]float64, docLen float64, m corpusModel, qStems []string) float64 {
	if m.avgLen == 0 {
		return 0
	}
	norm := 1 - bm25B + bm25B*docLen/m.avgLen
	score := 0.0
	for _, qt := range qStems {
		tf := bag[qt]
		if tf == 0 {
			continue
		}
		idf := m.idf[qt]
		if idf <= 0 {
			continue
		}
		score += idf * (tf * (bm25K1 + 1)) / (tf + bm25K1*norm)
	}
	return score
}

// rankHybrid is the retriever behind rankCards: build the corpus IDF model over
// the candidate cards, score each with BM25 + the IDF-scaled exact-name/tag
// boost, keep only positive hits, and order by score with cardLess as the
// deterministic tiebreak. An empty (or fully stop-word) query yields no hits,
// exactly as the flat scorer did (the q=="" surfaces sort the full set upstream
// rather than routing through here).
func rankHybrid(cards []FeatureCard, q string) []FeatureCard {
	qStems := stemAll(tokens(q))
	if len(qStems) == 0 {
		return nil
	}
	bags := make([]map[string]float64, len(cards))
	docLens := make([]float64, len(cards))
	for i, c := range cards {
		bag := cardTerms(c)
		bags[i] = bag
		l := 0.0
		for _, w := range bag {
			l += w
		}
		docLens[i] = l
	}
	model := buildCorpusModel(bags)

	type hit struct {
		card  FeatureCard
		score float64
	}
	var hits []hit
	for i, c := range cards {
		s := bm25Score(bags[i], docLens[i], model, qStems)
		nameTag := nameTagStemSet(c)
		for _, qt := range qStems {
			if nameTag[qt] {
				if idf := model.idf[qt]; idf > 0 {
					s += exactNameTagBoost * idf
				}
			}
		}
		if s > 0 {
			hits = append(hits, hit{c, s})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return cardLess(hits[i].card, hits[j].card)
	})
	out := make([]FeatureCard, len(hits))
	for i, h := range hits {
		out[i] = h.card
	}
	return out
}
