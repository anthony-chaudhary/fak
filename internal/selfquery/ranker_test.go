package selfquery

import (
	"sort"
	"testing"
)

// ranker_test.go is the #3235 witness: it proves the hybrid BM25 + stemming
// retriever (ranker.go) fixes the two failure modes of the flat lexical scorer
// (score) and does not regress recall — the evidence the deferral epic (#3229)
// leans on, since a tool the model cannot re-find from its intent becomes an
// invisible capability once cold schemas are deferred (#3231/#3232).

// rankerFixture is a compact stand-in for the real catalog: a live plane whose
// cards all carry the ubiquitous live/mcp/tool boilerplate tags, plus memory
// and dev cards. It is deliberately shaped like the real corpus (see
// selfquery.go devCards/toolCards) so the IDF and length-norm effects the
// production ranker relies on are exercised, without coupling the test to the
// live catalog's exact contents.
func rankerFixture() []FeatureCard {
	L := func(name, summary string, extra ...string) FeatureCard {
		return FeatureCard{Kind: "live-tool", Name: name, Summary: summary,
			Tags: append([]string{"live", "mcp", "tool"}, extra...), Source: "gateway.tools"}
	}
	return []FeatureCard{
		L("dos_commit_audit", "does a commit's claim match what its diff actually did", "commit", "audit", "diff"),
		L("dos_arbitrate", "may a worker take this lane right now", "lane", "lease", "arbitrate"),
		L("dos_verify", "did this plan phase actually ship", "verify", "ship", "phase"),
		L("dos_citation_resolve", "does this cited legal case exist and quote match", "citation", "legal", "case"),
		L("fak_tools_search", "search tool schemas on demand with progressive disclosure", "search", "schema"),
		{Kind: "memory-driver", Name: "memory-driver:compact", Summary: "compact the resident context window",
			Tags: []string{"memory", "driver", "compact", "context", "hygiene"}, Source: "memq"},
		{Kind: "memory-driver", Name: "memory-driver:clean", Summary: "clean stale cells out of memory",
			Tags: []string{"memory", "driver", "clean", "context", "hygiene"}, Source: "memq"},
		{Kind: "memory-driver", Name: "memory-driver:recall", Summary: "recall the relevant memories for a task",
			Tags: []string{"memory", "driver", "recall"}, Source: "memq"},
		{Kind: "cli-verb", Name: "fak index lane", Summary: "resolve a path to its owning lane and suggested commit stamp",
			Tags: []string{"dev", "index", "lane", "commit", "stamp"}, Source: "devindex"},
		{Kind: "dev-leaf", Name: "leaf:docs", Summary: "the documentation tree",
			Tags: []string{"dev", "leaf", "lane", "docs", "commit", "stamp"}, Source: "devindex"},
	}
}

func TestStemTokenCollapsesMorphologicalVariants(t *testing.T) {
	// Families the crude one-pass suffix stripper is DOCUMENTED to collapse:
	// plurals and the -s/-ing/-ion/-tion verb<->noun families on consonant
	// stems. It intentionally does NOT handle y<->i or silent-e alternations
	// (verify/verifies, resolve/resolves) — those are beyond a single-pass
	// stripper, and over-stemming them risks collisions; the ranker leans on
	// co-occurring terms for those, not the stem.
	groups := [][]string{
		{"compact", "compacts", "compacting", "compaction"},
		{"tool", "tools"},
		{"audit", "audits", "auditing"},
		{"search", "searching", "searches"},
		{"citation", "citations"},
	}
	for _, g := range groups {
		want := stemToken(g[0])
		for _, v := range g[1:] {
			if got := stemToken(v); got != want {
				t.Errorf("stemToken(%q)=%q, want %q (same stem as %q)", v, got, want, g[0])
			}
		}
	}
	// Short tokens must be left intact (remainder-guard), so we never over-stem
	// a discriminating short term into a collision.
	for _, keep := range []string{"index", "lane", "live", "mcp"} {
		if got := stemToken(keep); got != keep {
			t.Errorf("stemToken(%q)=%q, want it left intact", keep, got)
		}
	}
}

// hybridTopScore returns the highest hybrid score any card earns for q, using
// the same scoring the production ranker uses (in-package access).
func hybridTopScore(cards []FeatureCard, q string) float64 {
	qStems := stemAll(tokens(q))
	bags := make([]map[string]float64, len(cards))
	docLens := make([]float64, len(cards))
	for i, c := range cards {
		bags[i] = cardTerms(c)
		for _, w := range bags[i] {
			docLens[i] += w
		}
	}
	model := buildCorpusModel(bags)
	top := 0.0
	for i, c := range cards {
		s := bm25Score(bags[i], docLens[i], model, qStems)
		nt := nameTagStemSet(c)
		for _, qt := range qStems {
			if nt[qt] && model.idf[qt] > 0 {
				s += exactNameTagBoost * model.idf[qt]
			}
		}
		if s > top {
			top = s
		}
	}
	return top
}

// lexicalTopScore is the same, for the retained flat baseline (score).
func lexicalTopScore(cards []FeatureCard, q string) int {
	toks := tokens(q)
	top := 0
	for _, c := range cards {
		if s := score(c, toks); s > top {
			top = s
		}
	}
	return top
}

// TestIDFDefusesUbiquitousTags is the headline witness: a query made ENTIRELY of
// the boilerplate tags every live card carries must not score highly, because
// IDF crushes terms that appear in nearly every card — so it can no longer drag
// the whole live plane forward. The flat baseline, by contrast, hands that same
// query a large score (a fixed +12 per exact tag), which is exactly the drag.
func TestIDFDefusesUbiquitousTags(t *testing.T) {
	cards := rankerFixture()
	const ubiquitous = "live mcp tool"
	const discriminating = "citation legal case"

	hUbiq := hybridTopScore(cards, ubiquitous)
	hDisc := hybridTopScore(cards, discriminating)
	if !(hUbiq < hDisc) {
		t.Fatalf("hybrid: ubiquitous-tag query top score %.4f should be well below a discriminating query's %.4f", hUbiq, hDisc)
	}

	lUbiq := lexicalTopScore(cards, ubiquitous)
	if lUbiq < 30 {
		t.Fatalf("baseline sanity: expected the flat scorer to inflate the ubiquitous query (>=30), got %d", lUbiq)
	}
	t.Logf("ubiquitous-tag top score: hybrid=%.4f (defused) vs lexical=%d (drag); discriminating hybrid top=%.4f",
		hUbiq, lUbiq, hDisc)
}

// The labeled intent set this A/B grades on now lives in eval.go as
// DefaultIntentFixture (#3162), so the relative A/B below and the absolute
// recall@K / MRR floors in eval_test.go cannot drift apart. The intents
// deliberately include morphological variants ("compacting") and tag-shaped
// noise ("tool", "mcp") so the stemming + IDF rungs are what decide.

// rankLexicalBaseline reproduces the pre-#3235 rankCards using the retained flat
// score, so the A/B compares two full rankers over identical inputs.
func rankLexicalBaseline(cards []FeatureCard, q string) []FeatureCard {
	toks := tokens(q)
	type hit struct {
		card  FeatureCard
		score int
	}
	var hits []hit
	for _, c := range cards {
		if s := score(c, toks); s > 0 {
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

// recallAt delegates to the shared eval harness (#3162) so this A/B and the
// recorded floors grade with exactly the same metric definition.
func recallAt(rank func([]FeatureCard, string) []FeatureCard, cards []FeatureCard, k int) float64 {
	return EvalRetrieval(rank, cards, DefaultIntentFixture, k).RecallAtK
}

// TestHybridRankerImprovesRecallAtK is the before/after recall witness (the
// number that goes in the PR body). It grades the hybrid ranker against the
// retained flat baseline over the intent fixture; the hybrid must not regress
// mean recall@5 and must strictly beat it on at least the stemming intent that
// the flat substring match cannot reach.
func TestHybridRankerImprovesRecallAtK(t *testing.T) {
	cards := rankerFixture()
	const k = 5
	base := recallAt(rankLexicalBaseline, cards, k)
	hybrid := recallAt(rankCards, cards, k)
	t.Logf("recall@%d: lexical-baseline=%.3f  hybrid=%.3f  (delta=%+.3f)", k, base, hybrid, hybrid-base)
	if hybrid < base {
		t.Fatalf("hybrid recall@%d regressed: hybrid=%.3f < baseline=%.3f", k, hybrid, base)
	}

	if !(hybrid > base) {
		t.Fatalf("expected the sparse morphological intents to give the hybrid a strict recall@%d edge: hybrid=%.3f base=%.3f", k, hybrid, base)
	}

	// The bare "compaction" intent is the concrete case the flat substring
	// scorer misses (not a substring of "compact") and the hybrid's stemming
	// rung catches: assert it directly so the improvement is witnessed, not just
	// averaged away.
	got := rankCards(cards, "compaction")
	if len(got) == 0 || got[0].Name != "memory-driver:compact" {
		t.Fatalf("stemming intent top = %v, want memory-driver:compact", firstNameOf(got))
	}
	baseGot := rankLexicalBaseline(cards, "compaction")
	if inTopK(baseGot, "memory-driver:compact", k) {
		t.Fatalf("baseline unexpectedly found the bare-stem intent; fixture no longer demonstrates the delta")
	}
}

func TestRankHybridDeterministic(t *testing.T) {
	cards := rankerFixture()
	for _, q := range []string{"commit stamp for a path", "live mcp tool", "compacting context"} {
		a := firstNamesOf(rankCards(cards, q))
		b := firstNamesOf(rankCards(cards, q))
		if len(a) != len(b) {
			t.Fatalf("q=%q non-deterministic length: %d vs %d", q, len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("q=%q non-deterministic order at %d: %v vs %v", q, i, a, b)
			}
		}
	}
}

func firstNameOf(cards []FeatureCard) string {
	if len(cards) == 0 {
		return "<none>"
	}
	return cards[0].Name
}

func firstNamesOf(cards []FeatureCard) []string {
	out := make([]string, len(cards))
	for i, c := range cards {
		out[i] = c.Name
	}
	return out
}

func inTopK(cards []FeatureCard, name string, k int) bool {
	for i, c := range cards {
		if i >= k {
			return false
		}
		if c.Name == name {
			return true
		}
	}
	return false
}
