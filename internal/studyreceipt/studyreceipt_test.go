package studyreceipt_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studyreceipt"
)

func validFixture() studyreceipt.Receipt {
	start := time.Date(2026, 9, 3, 10, 0, 0, 0, time.UTC)
	end := start.Add(15 * time.Minute)
	return studyreceipt.Receipt{
		Schema:      studyreceipt.Schema,
		ID:          "rcpt-study-001",
		StudyID:     "study-perf-inference",
		Track:       "native-eval",
		Participant: "worker-s0",
		Environment: "darwin-arm64",
		StartedAt:   start.Format(time.RFC3339Nano),
		CompletedAt: end.Format(time.RFC3339Nano),
		ElapsedSec:  900.0,
		Outcome:     studyreceipt.OutcomeSuccess,
		Artifacts:   []string{"dist/proof.json"},
		Sources: []studyreceipt.SourceRef{
			{
				Name:     "upstream-reference",
				URL:      "https://github.com/anthony-chaudhary/fak",
				Revision: "r10+g12345678",
				Digest:   "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
			},
		},
		Observations: []studyreceipt.Observation{
			{
				ID:         "obs-1",
				RecordedAt: end.Format(time.RFC3339Nano),
				Metric:     "tokens_per_sec",
				Value:      "142.5",
				Witness:    "cmd/fak bench",
			},
		},
		Decisions: []studyreceipt.Decision{
			{
				ID:          "dec-1",
				Candidate:   "qwen3.8-quant-kernel",
				Disposition: "DEFAULT",
				Rationale:   "exceeded reference throughput within envelope",
				Evidence:    []string{"obs-1"},
			},
		},
	}
}

func TestValidate_ValidReceipt(t *testing.T) {
	r := validFixture()
	if err := studyreceipt.Validate(r); err != nil {
		t.Fatalf("expected valid receipt, got error: %v", err)
	}

	d, err := studyreceipt.Digest(r)
	if err != nil {
		t.Fatalf("unexpected digest error: %v", err)
	}
	if !strings.HasPrefix(d, "sha256:") || len(d) != 71 {
		t.Fatalf("unexpected digest format: %s", d)
	}

	r.Digest = d
	if err := studyreceipt.Validate(r); err != nil {
		t.Fatalf("expected receipt with correct digest to validate, got: %v", err)
	}

	// Corrupt digest
	r.Digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
	if err := studyreceipt.Validate(r); err == nil {
		t.Fatal("expected corrupted receipt to fail validation")
	}
}

func TestValidate_InvalidFields(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(r *studyreceipt.Receipt)
	}{
		{
			name:   "bad schema",
			mutate: func(r *studyreceipt.Receipt) { r.Schema = "wrong/1" },
		},
		{
			name:   "bad ID",
			mutate: func(r *studyreceipt.Receipt) { r.ID = "INVALID ID WITH SPACES" },
		},
		{
			name:   "bad study ID",
			mutate: func(r *studyreceipt.Receipt) { r.StudyID = "!" },
		},
		{
			name:   "missing track",
			mutate: func(r *studyreceipt.Receipt) { r.Track = "" },
		},
		{
			name:   "missing participant",
			mutate: func(r *studyreceipt.Receipt) { r.Participant = "" },
		},
		{
			name:   "missing environment",
			mutate: func(r *studyreceipt.Receipt) { r.Environment = "   " },
		},
		{
			name:   "completed before started",
			mutate: func(r *studyreceipt.Receipt) { r.CompletedAt = "2026-01-01T00:00:00Z" },
		},
		{
			name:   "negative elapsed seconds",
			mutate: func(r *studyreceipt.Receipt) { r.ElapsedSec = -1 },
		},
		{
			name:   "bad outcome",
			mutate: func(r *studyreceipt.Receipt) { r.Outcome = "unknown" },
		},
		{
			name:   "empty sources",
			mutate: func(r *studyreceipt.Receipt) { r.Sources = nil },
		},
		{
			name: "source bad digest",
			mutate: func(r *studyreceipt.Receipt) {
				r.Sources[0].Digest = "not-a-sha256-digest"
			},
		},
		{
			name: "observation missing metric",
			mutate: func(r *studyreceipt.Receipt) {
				r.Observations[0].Metric = ""
			},
		},
		{
			name: "decision missing evidence",
			mutate: func(r *studyreceipt.Receipt) {
				r.Decisions[0].Evidence = nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := validFixture()
			tt.mutate(&r)
			if err := studyreceipt.Validate(r); err == nil {
				t.Errorf("test %q expected error, got nil", tt.name)
			}
		})
	}
}

func TestStore_PutGetList(t *testing.T) {
	dir := t.TempDir()
	store, err := studyreceipt.OpenStore(dir)
	if err != nil {
		t.Fatalf("OpenStore failed: %v", err)
	}

	r1 := validFixture()
	r1.ID = "rcpt-001"
	r1.StudyID = "study-a"

	r2 := validFixture()
	r2.ID = "rcpt-002"
	r2.StudyID = "study-b"

	sealed1, err := store.Put(r1)
	if err != nil {
		t.Fatalf("store.Put r1 failed: %v", err)
	}
	if sealed1.Digest == "" {
		t.Fatal("expected non-empty digest after Put")
	}

	sealed2, err := store.Put(r2)
	if err != nil {
		t.Fatalf("store.Put r2 failed: %v", err)
	}
	if sealed2.Digest == "" {
		t.Fatal("expected non-empty digest for r2")
	}

	// Test duplicate insertion
	if _, err := store.Put(r1); err == nil {
		t.Fatal("expected duplicate Put to fail")
	}

	// Test Get
	got1, err := store.Get("rcpt-001")
	if err != nil {
		t.Fatalf("store.Get rcpt-001 failed: %v", err)
	}
	if got1.Digest != sealed1.Digest {
		t.Fatalf("expected digest %s, got %s", sealed1.Digest, got1.Digest)
	}

	// Test invalid ID in Get
	if _, err := store.Get("bad/id"); err == nil {
		t.Fatal("expected invalid ID get to fail")
	}

	// Test List all
	all, err := store.List("")
	if err != nil {
		t.Fatalf("store.List failed: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("expected 2 receipts in store, got %d", len(all))
	}
	if all[0].ID != "rcpt-001" || all[1].ID != "rcpt-002" {
		t.Fatalf("unexpected order: %v, %v", all[0].ID, all[1].ID)
	}

	// Test List filtered by StudyID
	filtered, err := store.List("study-a")
	if err != nil {
		t.Fatalf("store.List study-a failed: %v", err)
	}
	if len(filtered) != 1 || filtered[0].ID != "rcpt-001" {
		t.Fatalf("unexpected filtered results: %+v", filtered)
	}

	// Test corrupted file detection during Get
	path1 := filepath.Join(dir, "rcpt-001.json")
	if err := os.WriteFile(path1, []byte("invalid json"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Get("rcpt-001"); err == nil {
		t.Fatal("expected Get on corrupted file to fail")
	}
}

func BenchmarkValidate(b *testing.B) {
	r := validFixture()
	d, err := studyreceipt.Digest(r)
	if err != nil {
		b.Fatalf("digest fixture failed: %v", err)
	}
	r.Digest = d
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := studyreceipt.Validate(r); err != nil {
			b.Fatalf("validate failed: %v", err)
		}
	}
}

func BenchmarkDigest(b *testing.B) {
	r := validFixture()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := studyreceipt.Digest(r); err != nil {
			b.Fatalf("digest failed: %v", err)
		}
	}
}

func BenchmarkStorePutGet(b *testing.B) {
	dir := b.TempDir()
	store, err := studyreceipt.OpenStore(dir)
	if err != nil {
		b.Fatalf("open store failed: %v", err)
	}
	r := validFixture()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.ID = "rcpt-bench-001"
		if _, err := store.Put(r); err != nil {
			b.Fatalf("put failed: %v", err)
		}
		if _, err := store.Get("rcpt-bench-001"); err != nil {
			b.Fatalf("get failed: %v", err)
		}
		// Clean up for the next iteration.
		_ = os.Remove(filepath.Join(dir, "rcpt-bench-001.json"))
	}
}
