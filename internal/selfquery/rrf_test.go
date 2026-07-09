package selfquery

import (
	"math"
	"testing"
)

func rrfCard(name, summary string, tags ...string) FeatureCard {
	return FeatureCard{
		Kind:      "dev-leaf",
		Name:      name,
		Summary:   summary,
		Tags:      tags,
		Source:    "devindex",
		DetailRef: "internal/" + name,
	}
}

// TestRRFReproducesCodesearchWeighting pins the fusion arithmetic: a card ranked
// first (rank 0) by two arms at k=20 scores 1/(20+0+1) + 1/(20+0+1) = 2/21 — the
// exact reciprocal-rank weighting codesearch's fusion uses.
func TestRRFReproducesCodesearchWeighting(t *testing.T) {
	top := rrfCard("logvault", "rotate compress store logs", "logging")
	other := rrfCard("metrics", "counters gauges", "telemetry")

	// Same card at rank 0 in both arms; a filler card fills rank 1 in each.
	listA := []FeatureCard{top, other}
	listB := []FeatureCard{top, other}
	fused := fuseRanked([][]FeatureCard{listA, listB}, []float64{20, 20})

	if len(fused) != 2 {
		t.Fatalf("fused %d rows, want 2", len(fused))
	}
	if fused[0].Card.Name != "logvault" {
		t.Fatalf("top fused card = %q, want logvault", fused[0].Card.Name)
	}
	want := 2.0 / 21.0
	if math.Abs(fused[0].Score-want) > 1e-9 {
		t.Errorf("rank-0-in-two-arms score = %v, want %v (1/21 + 1/21)", fused[0].Score, want)
	}
	// The runner-up sat at rank 1 in both arms: 2 * 1/(20+1+1) = 2/22.
	wantOther := 2.0 / 22.0
	if math.Abs(fused[1].Score-wantOther) > 1e-9 {
		t.Errorf("rank-1-in-two-arms score = %v, want %v", fused[1].Score, wantOther)
	}
}

// TestHybridRRFBeatsBM25Alone is the ship-alone claim for #3436: fusing the simhash
// arm into BM25 never lowers recall on a held-out intent set and strictly raises it
// — because the semantic arm surfaces a card the exact-lexical arm misses.
func TestHybridRRFBeatsBM25Alone(t *testing.T) {
	corpus := []FeatureCard{
		rrfCard("authentication", "verify identity and issue tokens", "identity"),
		rrfCard("logvault", "rotate compress and store logs", "logging"),
		rrfCard("dispatch", "route waves to accounts", "routing"),
		rrfCard("metrics", "counters gauges histograms", "telemetry"),
	}
	// Held-out (query -> the one relevant card). "auth" shares NO stemmed token
	// with "authentication" (Porter stems it to "authent"), so BM25 cannot retrieve
	// it — but the char-ngram arm can (shared trigrams "aut","uth").
	intents := []struct{ query, want string }{
		{"auth", "authentication"},
		{"logging", "logvault"},
		{"route waves", "dispatch"},
		{"gauges", "metrics"},
	}
	const k = 3

	inTopK := func(ranked []FeatureCard, name string) bool {
		for i, c := range ranked {
			if i >= k {
				break
			}
			if c.Name == name {
				return true
			}
		}
		return false
	}

	bm25Recall, rrfRecall := 0, 0
	var authBM25, authRRF bool
	for _, it := range intents {
		bm25 := rankHybrid(corpus, it.query)
		rrf := HybridRRF(corpus, it.query)
		bHit := inTopK(bm25, it.want)
		rHit := inTopK(rrf, it.want)
		if rHit {
			rrfRecall++
		}
		if bHit {
			bm25Recall++
		}
		// RRF must never be worse than BM25 alone on any single intent.
		if bHit && !rHit {
			t.Errorf("RRF regressed on %q: BM25 found %q but RRF did not", it.query, it.want)
		}
		if it.query == "auth" {
			authBM25, authRRF = bHit, rHit
		}
	}

	if authBM25 {
		t.Error("expected BM25 alone to MISS 'auth' -> authentication (no shared stem)")
	}
	if !authRRF {
		t.Error("expected RRF to RECOVER 'auth' -> authentication via the semantic arm")
	}
	if rrfRecall <= bm25Recall {
		t.Errorf("RRF recall %d/%d did not beat BM25 recall %d/%d", rrfRecall, len(intents), bm25Recall, len(intents))
	}
}

func TestClassifyQueryAndAdaptK(t *testing.T) {
	cases := []struct {
		q     string
		class QueryClass
	}{
		{"cardLess", ClassIdentifier},
		{"a_b_c", ClassIdentifier},
		{"how do I rotate logs", ClassSemantic},
		{"foo(.*)", ClassStructural},
		{"pkg.Method", ClassStructural},
		{"123abc", ClassSemantic}, // starts with a digit -> not an identifier token
		{"", ClassSemantic},
	}
	for _, c := range cases {
		if got := ClassifyQuery(c.q); got != c.class {
			t.Errorf("ClassifyQuery(%q) = %d, want %d", c.q, got, c.class)
		}
	}

	// A smaller k gives an arm more weight: identifier trusts fts, semantic is
	// balanced. Assert the exact adapt_rrf_k table.
	if fts, vec := AdaptRRFK(ClassIdentifier); fts != 12 || vec != 28 {
		t.Errorf("identifier k = (%v,%v), want (12,28)", fts, vec)
	}
	if fts, vec := AdaptRRFK(ClassStructural); fts != 15 || vec != 25 {
		t.Errorf("structural k = (%v,%v), want (15,25)", fts, vec)
	}
	if fts, vec := AdaptRRFK(ClassSemantic); fts != 20 || vec != 20 {
		t.Errorf("semantic k = (%v,%v), want (20,20)", fts, vec)
	}
}
