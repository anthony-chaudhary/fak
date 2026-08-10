package main

import (
	"path/filepath"
	"testing"
)

func TestProvenanceFoldSelfcheck(t *testing.T) {
	p := filepath.Join(t.TempDir(), "fold.json")
	if err := runProvenanceFoldSelfcheck(p); err != nil {
		t.Fatal(err)
	}
	if err := verifyProvenanceFoldArtifact(p); err != nil {
		t.Fatal(err)
	}
}
func TestFoldDeterministicAcrossOrderAndShape(t *testing.T) {
	facts := makeFoldFacts(257)
	a, _ := buildFold(facts, 8, false)
	for i, j := 0, len(facts)-1; i < j; i, j = i+1, j-1 {
		facts[i], facts[j] = facts[j], facts[i]
	}
	b, _ := buildFold(facts, 17, false)
	if a.ResultHash != b.ResultHash {
		t.Fatalf("result drift: %s != %s", a.ResultHash, b.ResultHash)
	}
}
func TestFoldPreservesMinorityAndUncertainty(t *testing.T) {
	r, _ := buildFold(makeFoldFacts(1000), 8, false)
	if !hasCitation(r.Root, "issue-00777") || !hasCitation(r.Root, "issue-00778") {
		t.Fatal("minority or abstention citation lost")
	}
	if r.Root.StatusCounts["abstain"] != 1 {
		t.Fatal("uncertainty count lost")
	}
}
func TestFoldStableTopKTie(t *testing.T) {
	a := foldSummary{TopK: []foldCandidate{{"b", 9}}}
	b := foldSummary{TopK: []foldCandidate{{"a", 9}}}
	r := mergeSummaries([]foldSummary{a, b})
	if len(r.TopK) != 2 || r.TopK[0].SourceID != "a" {
		t.Fatalf("unstable tie: %+v", r.TopK)
	}
}
func TestVerifyFoldRejectsUnsafeClaims(t *testing.T) {
	base := provenanceFoldReport{Schema: provenanceFoldSchema, Sources: 1000, Baseline: foldRun{FanIn: 8, MaxInput: 8, MaxOutputCitations: 10, ResultHash: "x", Root: foldSummary{Coverage: 1000}}, ReorderedResultHash: "x", AlternateFanInResultHash: "x", RecomputedNodes: 5, ExpectedAncestorPath: 5, UnaffectedNodesReused: 995, CitationsResolved: true, MinorityOutlierPreserved: true, UncertaintyPreserved: true, NegativeAuditPassed: true}
	cases := []struct {
		name string
		f    func(*provenanceFoldReport)
	}{{"overflow", func(r *provenanceFoldReport) { r.Baseline.MaxInput = 9 }}, {"order-drift", func(r *provenanceFoldReport) { r.ReorderedResultHash = "y" }}, {"wide-invalidation", func(r *provenanceFoldReport) { r.RecomputedNodes = 6 }}, {"citation-loss", func(r *provenanceFoldReport) { r.CitationsResolved = false }}, {"dissent-loss", func(r *provenanceFoldReport) { r.MinorityOutlierPreserved = false }}, {"coverage-loss", func(r *provenanceFoldReport) { r.Baseline.Root.Coverage = 999 }}}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := base
			c.f(&r)
			if verifyFoldReport(r) == nil {
				t.Fatal("unsafe fold accepted")
			}
		})
	}
}
