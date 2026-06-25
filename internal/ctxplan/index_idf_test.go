package ctxplan

import (
	"context"
	"testing"
)

// TestDocFreqAndIDF checks the per-token posting statistic and its IDF weight: a
// token in many spans is less selective (lower idf) than a token in one.
func TestDocFreqAndIDF(t *testing.T) {
	st := NewMemStore()
	// "common" appears in every span; "rare" in exactly one.
	for i := 0; i < 9; i++ {
		st.Add("Bash", DurabilityTurn, []byte("common log line "+itoaTest(i)), false)
	}
	st.Add("WebSearch", DurabilityDurable, []byte("common rare runbook"), false)
	ctx := context.Background()
	spans, _ := st.Spans(ctx)
	ix := BuildIndex(spans)

	if df := ix.docFreq("common"); df != 10 {
		t.Fatalf("docFreq(common) = %d, want 10", df)
	}
	if df := ix.docFreq("rare"); df != 1 {
		t.Fatalf("docFreq(rare) = %d, want 1", df)
	}
	if df := ix.docFreq("absent"); df != 0 {
		t.Fatalf("docFreq(absent) = %d, want 0", df)
	}
	// A rarer token must weigh MORE than a common one, and a matched token always
	// contributes (idf >= 1).
	if ix.idf("rare") <= ix.idf("common") {
		t.Fatalf("idf(rare)=%v must exceed idf(common)=%v", ix.idf("rare"), ix.idf("common"))
	}
	if ix.idf("common") < 1 {
		t.Fatalf("idf(common)=%v must be >= 1 (a match always contributes)", ix.idf("common"))
	}
}

// TestProbeRanksBySelectivity is the #564 property: when the candidate cap forces a
// choice between a span matched only by a COMMON intent token and one matched by a
// RARE intent token, the rare (more selective) span is kept. Before IDF weighting,
// the relevance tier was unordered and the cap dropped by recency alone, so a span
// surfaced by a noise-frequency token could crowd out a discriminating match.
func TestProbeRanksBySelectivity(t *testing.T) {
	ctx := context.Background()
	st := NewMemStore()
	// Many spans share the common token "status"; only ONE carries the rare token
	// "kerberos". Both are OLD (added first), so recency cannot save either — the cap
	// must keep the selective one on IDF, not on position.
	for i := 0; i < 12; i++ {
		st.Add("Bash", DurabilityTurn, []byte("status check "+itoaTest(i)), false)
	}
	st.Add("WebSearch", DurabilityTurn, []byte("kerberos status ticket renewal"), false) // the selective span
	rareID := "span:12"
	// A block of recent, unrelated spans pushes everything above out of the recency tail.
	for i := 0; i < 20; i++ {
		st.Add("Read", DurabilityTurn, []byte("unrelated recent note "+itoaTest(i)), false)
	}
	spans, _ := st.Spans(ctx)
	ix := BuildIndex(spans)

	// The forecast intent names BOTH a common and a rare token. With a cap small
	// enough that not every "status" match fits, the kerberos span must survive.
	f := Forecast{Intents: []string{"status kerberos"}}
	probe := ix.Probe(f, ProbeOptions{RecencyWindow: 2, MaxCandidates: 5})
	got := probeIDset(probe)
	if !got[rareID] {
		t.Fatalf("the selective (kerberos) span %s must survive the cap on IDF; probed=%v", rareID, idsOf(probe))
	}

	// Stronger: across the relevance tier, the kerberos span should rank at or above
	// every common-only "status" span. Build the relevance-only ordering and assert
	// the rare span is not crowded out by common-only matches.
	relevanceOnly := ix.Probe(f, ProbeOptions{RecencyWindow: 0, MaxCandidates: 1})
	if len(relevanceOnly) != 1 || relevanceOnly[0].ID != rareID {
		t.Fatalf("top-1 relevance candidate must be the most selective span %s, got %v", rareID, idsOf(relevanceOnly))
	}
}

func idsOf(spans []Span) []string {
	out := make([]string, len(spans))
	for i, s := range spans {
		out[i] = s.ID
	}
	return out
}

// TestProbeStillDeterministicWithIDF guards the determinism contract the IDF
// refinement must not break: same (index, forecast, options) => same span set.
func TestProbeStillDeterministicWithIDF(t *testing.T) {
	ctx := context.Background()
	st := goodPlusNoiseStore(50)
	spans, _ := st.Spans(ctx)
	f := Forecast{Intents: []string{"auth token rotation", "revoke"}}
	a := idsOf(BuildIndex(spans).Probe(f, ProbeOptions{MaxCandidates: 8}))
	b := idsOf(BuildIndex(spans).Probe(f, ProbeOptions{MaxCandidates: 8}))
	if len(a) != len(b) {
		t.Fatalf("non-deterministic probe size: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic probe order at %d: %q vs %q", i, a[i], b[i])
		}
	}
}
