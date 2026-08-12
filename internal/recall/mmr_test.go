package recall

import "testing"

func journalIndex(rows ...JournalRow) *JournalIndex {
	idx := NewJournalIndex()
	for _, row := range rows {
		idx.Add(row)
	}
	return idx
}

func TestRecallMMRDiversifiesWithinWitnessedTierDeterministically(t *testing.T) {
	t.Setenv(mmrEnv, "true")
	t.Setenv(mmrLambdaEnv, "0.5")
	idx := journalIndex(
		JournalRow{Text: "cache lease expired on gateway alpha", Provenance: ProvWitnessed, Seq: 3},
		JournalRow{Text: "cache lease expired on gateway beta", Provenance: ProvWitnessed, Seq: 2},
		JournalRow{Text: "cache eviction pressure reduced latency", Provenance: ProvWitnessed, Seq: 1},
	)
	first := idx.Recall("cache lease eviction", 2)
	second := idx.Recall("cache lease eviction", 2)
	if len(first) != 2 || first[0].Text != "cache lease expired on gateway alpha" || first[1].Text != "cache eviction pressure reduced latency" {
		t.Fatalf("MMR did not suppress redundant second result: %#v", first)
	}
	if first[0].Text != second[0].Text || first[1].Text != second[1].Text {
		t.Fatalf("MMR order is nondeterministic: %#v vs %#v", first, second)
	}
}

func TestRecallMMRNeverPromotesClaimAboveWitness(t *testing.T) {
	t.Setenv(mmrEnv, "true")
	t.Setenv(mmrLambdaEnv, "0")
	idx := journalIndex(
		JournalRow{Text: "gateway cache lease result", Provenance: ProvWitnessed, Seq: 2},
		JournalRow{Text: "gateway cache lease claim", Provenance: ProvUnverified, Seq: 3},
	)
	hits := idx.Recall("gateway cache lease", 2)
	if len(hits) != 2 || hits[0].Provenance != ProvWitnessed || hits[1].Provenance != ProvUnverified {
		t.Fatalf("MMR crossed provenance boundary: %#v", hits)
	}
}

func TestRecallMMRIsOffByDefault(t *testing.T) {
	t.Setenv(mmrEnv, "")
	idx := journalIndex(
		JournalRow{Text: "cache lease expired on gateway alpha", Provenance: ProvWitnessed, Seq: 3},
		JournalRow{Text: "cache lease expired on gateway beta", Provenance: ProvWitnessed, Seq: 2},
		JournalRow{Text: "cache eviction pressure reduced latency", Provenance: ProvWitnessed, Seq: 1},
	)
	hits := idx.Recall("cache lease eviction", 2)
	if len(hits) != 2 || hits[1].Text != "cache lease expired on gateway beta" {
		t.Fatalf("default ordering changed while gate is off: %#v", hits)
	}
}
