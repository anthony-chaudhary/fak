package recall

import "testing"

// Issue #2840 done condition, made a test: a recall query returns rows tagged with
// provenance (witnessed/kept/unverified), ranked so witnessed rows outrank un-witnessed
// claims of EQUAL recency, and an un-witnessed claim is NEVER surfaced as fact.
func TestJournalIndexRankingWitnessedOverClaimAtEqualRecency(t *testing.T) {
	idx := NewJournalIndex()
	// Same query match, same recency (Seq): the ONLY difference is provenance.
	idx.Add(JournalRow{Seq: 10, Text: "gateway retry budget exhausted", Provenance: ProvUnverified})
	idx.Add(JournalRow{Seq: 10, Text: "gateway retry budget honored", Provenance: ProvWitnessed})

	hits := idx.Recall("gateway retry budget", 10)
	if len(hits) != 2 {
		t.Fatalf("want 2 hits, got %d: %+v", len(hits), hits)
	}
	if hits[0].Provenance != ProvWitnessed {
		t.Fatalf("witnessed row must outrank the claim at equal recency; got first=%s", hits[0].Provenance)
	}
	if !hits[0].AsFact {
		t.Fatalf("witnessed hit must be surfaceable as fact")
	}
	if hits[1].Provenance != ProvUnverified || hits[1].AsFact {
		t.Fatalf("un-verified claim must rank last and NEVER be surfaced as fact; got %s AsFact=%v", hits[1].Provenance, hits[1].AsFact)
	}
}

// Trustworthiness beats recency: a MORE-RECENT un-verified claim still ranks below an
// older witnessed row. This is the "rank by trustworthiness, not just recency" property
// and the stated confusion-risk guard — provenance is the primary key, recency the
// within-tier tie-break, not the other way round.
func TestJournalIndexProvenanceOutranksRecency(t *testing.T) {
	idx := NewJournalIndex()
	idx.Add(JournalRow{Seq: 5, Text: "deploy rollout completed", Provenance: ProvWitnessed})
	idx.Add(JournalRow{Seq: 99, Text: "deploy rollout completed", Provenance: ProvUnverified}) // far more recent

	hits := idx.Recall("deploy rollout", 10)
	if len(hits) != 2 {
		t.Fatalf("want 2 hits, got %d", len(hits))
	}
	if hits[0].Provenance != ProvWitnessed || hits[0].Seq != 5 {
		t.Fatalf("older witnessed row must outrank the newer un-verified claim; got first Seq=%d %s", hits[0].Seq, hits[0].Provenance)
	}
}

// Within one provenance tier, recency-demotion still applies: the more recent row wins.
func TestJournalIndexRecencyWithinTier(t *testing.T) {
	idx := NewJournalIndex()
	idx.Add(JournalRow{Seq: 1, Text: "cache hit ratio measured", Provenance: ProvWitnessed})
	idx.Add(JournalRow{Seq: 2, Text: "cache hit ratio measured", Provenance: ProvWitnessed})

	hits := idx.Recall("cache hit ratio", 10)
	if len(hits) != 2 {
		t.Fatalf("want 2 hits, got %d", len(hits))
	}
	if hits[0].Seq != 2 {
		t.Fatalf("within a tier the more recent row ranks first; got Seq=%d first", hits[0].Seq)
	}
}

// RecallFacts refuses to surface an un-witnessed claim at all: it returns only
// fact-surfaceable rows (witnessed/kept), dropping the claim entirely.
func TestJournalIndexRecallFactsDropsUnverified(t *testing.T) {
	idx := NewJournalIndex()
	idx.Add(JournalRow{Seq: 3, Text: "token budget policy kept", Provenance: ProvKept})
	idx.Add(JournalRow{Seq: 4, Text: "token budget policy claimed", Provenance: ProvUnverified})
	idx.Add(JournalRow{Seq: 5, Text: "token budget policy witnessed", Provenance: ProvWitnessed})

	facts := idx.RecallFacts("token budget policy", 10)
	if len(facts) != 2 {
		t.Fatalf("RecallFacts must drop the un-verified claim, keeping witnessed+kept; got %d: %+v", len(facts), facts)
	}
	for _, h := range facts {
		if h.Provenance == ProvUnverified || !h.AsFact {
			t.Fatalf("RecallFacts surfaced a non-fact row: %s AsFact=%v", h.Provenance, h.AsFact)
		}
	}
	// witnessed outranks kept.
	if facts[0].Provenance != ProvWitnessed {
		t.Fatalf("witnessed must outrank kept; got first=%s", facts[0].Provenance)
	}
}

// Relevance is a hard candidacy gate: a row with no query-token overlap is never
// returned, so provenance re-ranks within the relevant set and never widens it — even a
// witnessed row that does not match the query stays out.
func TestJournalIndexRelevanceGate(t *testing.T) {
	idx := NewJournalIndex()
	idx.Add(JournalRow{Seq: 1, Text: "witnessed but unrelated topic", Provenance: ProvWitnessed})
	idx.Add(JournalRow{Seq: 2, Text: "scheduler fairness verdict", Provenance: ProvUnverified})

	hits := idx.Recall("scheduler fairness", 10)
	if len(hits) != 1 {
		t.Fatalf("only the relevant row should return; got %d: %+v", len(hits), hits)
	}
	if hits[0].Seq != 2 {
		t.Fatalf("an off-topic witnessed row must not be resurrected by provenance; got Seq=%d", hits[0].Seq)
	}
}

// The journal-vocabulary classifier folds the real journal.Row status strings into the
// provenance tiers, fail-closed to unverified.
func TestProvenanceFromJournal(t *testing.T) {
	cases := []struct {
		name    string
		verdict string
		taint   string
		capKind string
		witness bool
		want    Provenance
	}{
		{"live witness", "ALLOW", "tainted", "", true, ProvWitnessed},
		{"witness verdict", "WITNESS", "tainted", "", false, ProvWitnessed},
		{"trusted taint", "ALLOW", "trusted", "", false, ProvWitnessed},
		{"kept skill", "ALLOW", "tainted", "skill", false, ProvKept},
		{"bare claim", "ALLOW", "tainted", "", false, ProvUnverified},
		{"empty fails closed", "", "", "", false, ProvUnverified},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ProvenanceFromJournal(c.verdict, c.taint, c.capKind, c.witness)
			if got != c.want {
				t.Fatalf("ProvenanceFromJournal(%q,%q,%q,%v) = %s, want %s", c.verdict, c.taint, c.capKind, c.witness, got, c.want)
			}
			if c.want == ProvUnverified && got.Fact() {
				t.Fatalf("an un-verified claim must never report Fact()")
			}
		})
	}
}
