package learningobservation

import (
	"errors"
	"fmt"
	"path/filepath"
	"testing"
)

// BenchmarkLearningObservation measures the complete lifecycle of record creation,
// idempotency verification, lineage linking, cycle defense, and graph tracing.
func BenchmarkLearningObservation(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := &Store{Schema: Schema}

		obs, created, err := s.Add(KindObservation, "trajectory://run/bench", "tool execution failed then recovered", "")
		if err != nil || !created {
			b.Fatalf("add observation: %v", err)
		}

		cand, created, err := s.Add(KindCandidate, "candidate://rule/retry", "apply exponential backoff on rate limit", "")
		if err != nil || !created {
			b.Fatalf("add candidate: %v", err)
		}

		wit, created, err := s.Add(KindWitness, "witness://eval/timeout", "zero rate limit errors observed across 100 runs", "")
		if err != nil || !created {
			b.Fatalf("add witness: %v", err)
		}

		verd, created, err := s.Add(KindVerdict, "verdict://decision/promote", "rule promoted to active policy", OutcomeKept)
		if err != nil || !created {
			b.Fatalf("add verdict: %v", err)
		}

		dup, created, err := s.Add(KindObservation, "trajectory://run/bench", "tool execution failed then recovered", "")
		if err != nil || created || dup.ID != obs.ID {
			b.Fatalf("duplicate add failed: created=%v err=%v", created, err)
		}

		if _, err := s.Link(cand.ID, ObservedFrom, obs.ID); err != nil {
			b.Fatalf("link observed-from: %v", err)
		}
		if _, err := s.Link(cand.ID, TestedBy, wit.ID); err != nil {
			b.Fatalf("link tested-by: %v", err)
		}
		if _, err := s.Link(wit.ID, KeptAs, verd.ID); err != nil {
			b.Fatalf("link kept-as: %v", err)
		}

		if _, err := s.Link(verd.ID, Supports, cand.ID); !errors.Is(err, ErrCycle) {
			b.Fatalf("expected cycle error, got: %v", err)
		}

		records, edges, err := s.Trace(cand.ID)
		if err != nil || len(records) != 4 || len(edges) != 3 {
			b.Fatalf("trace: records=%d edges=%d err=%v", len(records), len(edges), err)
		}
	}
}

// BenchmarkStoreAdd measures record insertion and deduplication lookup across various sizes.
func BenchmarkStoreAdd(b *testing.B) {
	b.Run("UniqueRecords", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := &Store{Schema: Schema}
			for j := 0; j < 32; j++ {
				source := fmt.Sprintf("src://run/%d", j)
				content := fmt.Sprintf("content description %d", j)
				if _, _, err := s.Add(KindObservation, source, content, ""); err != nil {
					b.Fatalf("add record: %v", err)
				}
			}
		}
	})

	b.Run("DeduplicationHits", func(b *testing.B) {
		s := &Store{Schema: Schema}
		for j := 0; j < 64; j++ {
			source := fmt.Sprintf("src://run/%d", j)
			content := fmt.Sprintf("content description %d", j)
			if _, _, err := s.Add(KindObservation, source, content, ""); err != nil {
				b.Fatalf("add record: %v", err)
			}
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx := i % 64
			source := fmt.Sprintf("src://run/%d", idx)
			content := fmt.Sprintf("content description %d", idx)
			rec, created, err := s.Add(KindObservation, source, content, "")
			if err != nil || created || rec.ID == "" {
				b.Fatalf("dedup add failed: created=%v err=%v", created, err)
			}
		}
	})
}

// BenchmarkStoreLink measures edge insertion and cycle-prevention reachability checks.
func BenchmarkStoreLink(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := &Store{Schema: Schema}
		ids := make([]string, 16)
		for j := 0; j < 16; j++ {
			r, _, err := s.Add(KindObservation, fmt.Sprintf("src://%d", j), fmt.Sprintf("node %d", j), "")
			if err != nil {
				b.Fatal(err)
			}
			ids[j] = r.ID
		}
		for j := 0; j < 15; j++ {
			if _, err := s.Link(ids[j], Supports, ids[j+1]); err != nil {
				b.Fatalf("link edge %d: %v", j, err)
			}
		}
	}
}

// BenchmarkStoreTrace measures lineage reachability traversal over branching candidate trees.
func BenchmarkStoreTrace(b *testing.B) {
	s := &Store{Schema: Schema}
	cand, _, err := s.Add(KindCandidate, "cand://root", "root candidate", "")
	if err != nil {
		b.Fatal(err)
	}

	for w := 0; w < 10; w++ {
		wit, _, err := s.Add(KindWitness, fmt.Sprintf("wit://%d", w), fmt.Sprintf("witness %d", w), "")
		if err != nil {
			b.Fatal(err)
		}
		if _, err := s.Link(cand.ID, TestedBy, wit.ID); err != nil {
			b.Fatal(err)
		}

		verd, _, err := s.Add(KindVerdict, fmt.Sprintf("verd://%d", w), fmt.Sprintf("verdict %d", w), OutcomeKept)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := s.Link(wit.ID, KeptAs, verd.ID); err != nil {
			b.Fatal(err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		records, edges, err := s.Trace(cand.ID)
		if err != nil || len(records) != 21 || len(edges) != 20 {
			b.Fatalf("trace failed: records=%d edges=%d err=%v", len(records), len(edges), err)
		}
	}
}

// BenchmarkStoreSaveLoad measures atomic persistence to disk and roundtrip deserialization.
func BenchmarkStoreSaveLoad(b *testing.B) {
	s := &Store{Schema: Schema}
	cand, _, _ := s.Add(KindCandidate, "cand://1", "candidate", "")
	obs, _, _ := s.Add(KindObservation, "obs://1", "observation", "")
	wit, _, _ := s.Add(KindWitness, "wit://1", "witness", "")
	verd, _, _ := s.Add(KindVerdict, "verd://1", "verdict", OutcomeKept)
	_, _ = s.Link(cand.ID, ObservedFrom, obs.ID)
	_, _ = s.Link(cand.ID, TestedBy, wit.ID)
	_, _ = s.Link(wit.ID, KeptAs, verd.ID)

	dir := b.TempDir()
	path := filepath.Join(dir, "store.json")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := s.Save(path); err != nil {
			b.Fatalf("save: %v", err)
		}
		loaded, err := Load(path)
		if err != nil || len(loaded.Records) != 4 || len(loaded.Edges) != 3 {
			b.Fatalf("load failed: records=%d edges=%d err=%v", len(loaded.Records), len(loaded.Edges), err)
		}
	}
}
