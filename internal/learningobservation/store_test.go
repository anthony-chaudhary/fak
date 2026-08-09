package learningobservation

import (
	"errors"
	"path/filepath"
	"testing"
)

func TestAddStableIdempotentAndConflict(t *testing.T) {
	s := &Store{Schema: Schema}
	first, created, err := s.Add(KindObservation, " trajectory://run/7 ", " tool failed\nthen recovered ", "")
	if err != nil || !created {
		t.Fatalf("first add: created=%v err=%v", created, err)
	}
	second, created, err := s.Add(KindObservation, "trajectory://run/7", "tool failed then recovered", "")
	if err != nil || created || second.ID != first.ID {
		t.Fatalf("duplicate: record=%+v created=%v err=%v", second, created, err)
	}
	_, _, err = s.Add(KindObservation, "trajectory://run/7", "different", "")
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("conflict err=%v", err)
	}
}

func TestAllRelationsAndOutcomesPersist(t *testing.T) {
	s := &Store{Schema: Schema}
	ids := make([]string, 8)
	kinds := []Kind{KindObservation, KindObservation, KindCandidate, KindCandidate, KindWitness, KindWitness, KindVerdict, KindVerdict}
	outcomes := []Outcome{"", "", "", "", "", "", OutcomeKept, OutcomeRejected}
	for i := range ids {
		r, _, err := s.Add(kinds[i], "source-"+string(rune('a'+i)), "content-"+string(rune('a'+i)), outcomes[i])
		if err != nil {
			t.Fatal(err)
		}
		ids[i] = r.ID
	}
	edges := []Edge{
		{ids[0], ObservedFrom, ids[1]}, {ids[0], Supports, ids[2]}, {ids[1], Contradicts, ids[3]},
		{ids[0], Proposes, ids[4]}, {ids[2], TestedBy, ids[5]}, {ids[4], KeptAs, ids[6]}, {ids[5], RejectedAs, ids[7]},
	}
	for _, edge := range edges {
		if _, err := s.Link(edge.From, edge.Relation, edge.To); err != nil {
			t.Fatalf("link %+v: %v", edge, err)
		}
	}
	path := filepath.Join(t.TempDir(), "store.json")
	if err := s.Save(path); err != nil {
		t.Fatal(err)
	}
	got, err := Load(path)
	if err != nil || len(got.Edges) != len(Relations) {
		t.Fatalf("load: edges=%d err=%v", len(got.Edges), err)
	}
}

func TestLinkDeniesUnknownDanglingAndCycles(t *testing.T) {
	s := &Store{Schema: Schema}
	a, _, _ := s.Add(KindCandidate, "a", "a", "")
	b, _, _ := s.Add(KindWitness, "b", "b", "")
	if _, err := s.Link(a.ID, "invented", b.ID); !errors.Is(err, ErrUnknownRelation) {
		t.Fatalf("unknown relation err=%v", err)
	}
	if _, err := s.Link(a.ID, TestedBy, "lo_missing"); !errors.Is(err, ErrDanglingID) {
		t.Fatalf("dangling err=%v", err)
	}
	if _, err := s.Link(a.ID, TestedBy, b.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Link(b.ID, Supports, a.ID); !errors.Is(err, ErrCycle) {
		t.Fatalf("cycle err=%v", err)
	}
}

func TestTraceContainsOnlyReachableLineage(t *testing.T) {
	s := &Store{Schema: Schema}
	candidate, _, _ := s.Add(KindCandidate, "candidate", "candidate", "")
	witness, _, _ := s.Add(KindWitness, "witness", "witness", "")
	verdict, _, _ := s.Add(KindVerdict, "verdict", "verdict", OutcomeKept)
	unrelated, _, _ := s.Add(KindWitness, "unrelated", "unrelated", "")
	_, _ = s.Link(candidate.ID, TestedBy, witness.ID)
	_, _ = s.Link(witness.ID, KeptAs, verdict.ID)
	records, edges, err := s.Trace(candidate.ID)
	if err != nil || len(records) != 3 || len(edges) != 2 {
		t.Fatalf("trace records=%d edges=%d err=%v", len(records), len(edges), err)
	}
	for _, record := range records {
		if record.ID == unrelated.ID {
			t.Fatal("trace invented unrelated edge")
		}
	}
}
