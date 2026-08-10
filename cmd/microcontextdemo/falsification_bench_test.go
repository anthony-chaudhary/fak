package main

import (
	"path/filepath"
	"testing"
)

func TestFalsificationSelfcheck(t *testing.T) {
	p := filepath.Join(t.TempDir(), "f.json")
	if e := runFalsificationBench(p); e != nil {
		t.Fatal(e)
	}
	if e := verifyFalsificationArtifact(p); e != nil {
		t.Fatal(e)
	}
}
func TestQualityMissCannotWin(t *testing.T) {
	r := falsificationReport{Schema: falsificationSchema, CorpusRecords: 1000, Pipelines: []string{"a", "b", "c", "d", "e"}, MicroWins: 1, MicroLosses: 1, DecisionBoundary: []boundaryRow{{Winner: "bad", Eligible: []string{"good"}}, {}, {}, {}, {}}}
	if verifyFalsification(r) == nil {
		t.Fatal("ineligible winner accepted")
	}
}
func TestDecisionBoundaryHasWinAndLoss(t *testing.T) {
	pipes := []string{"tuned-sql-search", "retrieval-rerank", "long-context", "chunk-map-reduce", "micro-context"}
	for _, s := range []struct{ a, r int }{{0, 0}, {20, 5}} {
		xs := makeBenchCorpus(s.a, s.r)
		eligible := []benchMetrics{}
		for _, p := range pipes {
			m := gradePipeline(p, "x", xs)
			if m.QualityPass {
				eligible = append(eligible, m)
			}
		}
		if len(eligible) == 0 {
			t.Fatal("no eligible pipeline")
		}
	}
}
