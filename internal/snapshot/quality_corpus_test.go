package snapshot

import "testing"

func fixtureCorpus() QualityCorpus {
	c := QualityCorpus{ID: "executive-dogfood", Revision: "r1", Provenance: "scrubbed-prod-2026-07", FailureClass: "unsupported-fact", Split: "held-out", ContaminationNotes: "no training overlap known", Owner: "quality-team", Tier: "release", CostSeconds: 2, Cases: []string{"case-b", "case-a"}}
	c.Digest = CorpusDigest(c)
	return c
}
func TestQualityCorpusRegistryRoundTripAndMutation(t *testing.T) {
	c := fixtureCorpus()
	r := QualityCorpusRegistry{Schema: QualityCorpusRegistrySchema, Corpora: []QualityCorpus{c}}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	got, _, err := r.Lookup(c.ID, c.Revision)
	if err != nil || got.Digest != c.Digest {
		t.Fatalf("lookup: %+v %v", got, err)
	}
	r.Corpora[0].Cases[0] = "mutated"
	if err := r.Validate(); err == nil {
		t.Fatal("mutation retained digest and passed")
	}
}
func TestQualityCorpusRegistryRefusesUnregisteredAndMissingEvidence(t *testing.T) {
	r := QualityCorpusRegistry{Schema: QualityCorpusRegistrySchema, Corpora: []QualityCorpus{fixtureCorpus()}}
	_, replay, err := r.Lookup("unknown", "r1")
	if err == nil || replay.FirstDivergence == "" || !replay.Scrubbed {
		t.Fatalf("unregistered passed: %+v %v", replay, err)
	}
	bad := fixtureCorpus()
	bad.Owner = ""
	bad.Digest = CorpusDigest(bad)
	rr := QualityCorpusRegistry{Schema: QualityCorpusRegistrySchema, Corpora: []QualityCorpus{bad}}
	if err := rr.Validate(); err == nil {
		t.Fatal("missing owner passed")
	}
}
