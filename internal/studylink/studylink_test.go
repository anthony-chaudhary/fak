package studylink

import (
	"errors"
	"testing"
)

func ledger() Ledger {
	return Ledger{Schema: "fak.study-join-ledger/1", Cutoff: "2026-08-26", SourceRevision: "vllm@abc", Joins: []Join{
		{ClusterID: "kv-prefix", Actionable: true, Disposition: Landed, Artifacts: []Artifact{{Kind: "module", ID: "internal/vcache", Revision: "r10+gaaa", Path: "internal/vcache"}}, Confidence: 1, Evidence: Evidence{Query: "prefix cache", Matches: []string{"internal/vcache"}, Digest: "sha256:a"}},
		{ClusterID: "paged-attention", Actionable: true, Disposition: OpenExact, Artifacts: []Artifact{{Kind: "issue", ID: "9162", State: "open"}}, Confidence: .95, Evidence: Evidence{Query: "paged allocation", Matches: []string{"#9162"}, Digest: "sha256:b"}},
		{ClusterID: "scheduler-overlap", Actionable: true, Disposition: Partial, Artifacts: []Artifact{{Kind: "doc", ID: "memory-map", Path: "docs/benchmarks/map.md"}}, Confidence: .6, Evidence: Evidence{Query: "overlap", Matches: []string{"map.md"}, Digest: "sha256:c"}, ManualReview: true},
		{ClusterID: "obsolete-api", Disposition: Obsolete, Confidence: 1, Evidence: Evidence{Query: "removed API", Digest: "sha256:d"}},
		{ClusterID: "new-mechanism", Actionable: true, Disposition: Uncovered, Confidence: .8, Evidence: Evidence{Query: "new mechanism", Digest: "sha256:e"}},
	}}
}

func TestJoinSummaryAndManualReview(t *testing.T) {
	s, err := Summarize(ledger())
	if err != nil {
		t.Fatal(err)
	}
	if s.Counts[Landed] != 1 || s.Counts[OpenExact] != 1 || s.Counts[Uncovered] != 1 || len(s.ManualReview) != 1 {
		t.Fatalf("summary=%+v", s)
	}
}

func TestValidatorCatchesBrokenClosedDuplicateAndUnclassified(t *testing.T) {
	cases := []func(*Ledger){
		func(l *Ledger) { l.Joins[1].Artifacts[0].State = "closed" },
		func(l *Ledger) {
			l.Joins = append(l.Joins, l.Joins[1])
			l.Joins[len(l.Joins)-1].ClusterID = "duplicate-exact"
		},
		func(l *Ledger) { l.Joins[0].Artifacts[0].ID = "" },
		func(l *Ledger) { l.Joins[0].ClusterID = l.Joins[1].ClusterID },
	}
	for _, mutate := range cases {
		l := ledger()
		mutate(&l)
		if !errors.Is(Validate(l), ErrInvalid) {
			t.Fatal("invalid ledger admitted")
		}
	}
}
