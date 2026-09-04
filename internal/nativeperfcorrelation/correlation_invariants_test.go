package nativeperfcorrelation

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
)

func TestContractRetentionAndEvictionInvariant(t *testing.T) {
	for _, cap := range []int{1, 3, 5} {
		t.Run(fmt.Sprintf("capacity_%d", cap), func(t *testing.T) {
			index, err := NewIndex(cap)
			if err != nil {
				t.Fatalf("NewIndex(%d) error = %v", cap, err)
			}

			keys := make([]string, cap+1)
			for i := 0; i <= cap; i++ {
				rec, err := index.Add(testInput(fmt.Sprintf("req-%d", i), fmt.Sprintf("run-%d", i)))
				if err != nil {
					t.Fatalf("Add(%d) error = %v", i, err)
				}
				keys[i] = rec.Key
			}

			// Invariant: The oldest insertion (keys[0]) must have been evicted.
			if _, err := index.Lookup(keys[0]); !errors.Is(err, ErrNotFound) {
				t.Fatalf("evicted key %s lookup = %v, want ErrNotFound", keys[0], err)
			}

			// Invariant: Exactly the newest cap records remain.
			snapshot := index.Snapshot()
			if len(snapshot) != cap {
				t.Fatalf("snapshot length = %d, want capacity %d", len(snapshot), cap)
			}
			for i := 0; i < cap; i++ {
				expectedKey := keys[i+1]
				if snapshot[i].Key != expectedKey {
					t.Fatalf("snapshot[%d] = %s, want %s", i, snapshot[i].Key, expectedKey)
				}
				if _, err := index.Lookup(expectedKey); err != nil {
					t.Fatalf("lookup(%s) error = %v", expectedKey, err)
				}
			}
		})
	}
}

func TestContractIdempotencyInvariant(t *testing.T) {
	index, err := NewIndex(2)
	if err != nil {
		t.Fatal(err)
	}

	inputA := testInput("req-a", "run-a")
	rec1, err := index.Add(inputA)
	if err != nil {
		t.Fatal(err)
	}

	// Re-insert identical record.
	rec2, err := index.Add(inputA)
	if err != nil {
		t.Fatalf("re-inserting identical record error = %v", err)
	}
	if rec1.Key != rec2.Key {
		t.Fatalf("keys differ on idempotent insert: %s vs %s", rec1.Key, rec2.Key)
	}

	// Insert input B.
	inputB := testInput("req-b", "run-b")
	recB, err := index.Add(inputB)
	if err != nil {
		t.Fatal(err)
	}

	// Re-insert input A again: should NOT evict anything because it already exists.
	rec3, err := index.Add(inputA)
	if err != nil {
		t.Fatal(err)
	}
	if rec3.Key != rec1.Key {
		t.Fatalf("keys differ: %s vs %s", rec3.Key, rec1.Key)
	}

	snapshot := index.Snapshot()
	if len(snapshot) != 2 {
		t.Fatalf("snapshot length = %d, want 2", len(snapshot))
	}
	if snapshot[0].Key != rec1.Key || snapshot[1].Key != recB.Key {
		t.Fatalf("unexpected order after idempotent insertion: %+v", snapshot)
	}
}

func TestContractKeyCollisionFailClosedInvariant(t *testing.T) {
	constantKey := "npc1_bbbbbbbbbbbbbbbb"
	index, err := NewIndex(3, WithKeyFunc(func(Record) string {
		return constantKey
	}))
	if err != nil {
		t.Fatal(err)
	}

	input1 := testInput("req-1", "run-1")
	if _, err := index.Add(input1); err != nil {
		t.Fatalf("first Add error = %v", err)
	}

	// Distinct input producing the same key must be rejected with ErrCollision.
	input2 := testInput("req-2", "run-2")
	if _, err := index.Add(input2); !errors.Is(err, ErrCollision) {
		t.Fatalf("colliding Add error = %v, want ErrCollision", err)
	}

	// Verify original record is unchanged.
	rec, err := index.Lookup(constantKey)
	if err != nil {
		t.Fatalf("Lookup error = %v", err)
	}
	if rec.RequestFingerprint != fingerprint(input1.RequestID) {
		t.Fatalf("record corrupted after collision attempt: %+v", rec)
	}
}

func TestContractScrubbingFingerprintInvariant(t *testing.T) {
	input := testInput("user-secret-request-9999", "tenant-run-8888")
	input.ReceiptID = "internal-receipt-7777"
	input.TraceID = "internal-trace-6666"
	input.ProfileID = "internal-profile-5555"

	rec, err := scrub(input)
	if err != nil {
		t.Fatalf("scrub error = %v", err)
	}

	rawValues := []string{input.RequestID, input.RunID, input.ReceiptID, input.TraceID, input.ProfileID}
	encoded, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)

	for _, raw := range rawValues {
		if strings.Contains(text, raw) {
			t.Fatalf("raw identifier %q leaked into scrubbed JSON: %s", raw, text)
		}
	}

	// Invariant: fingerprints must match "sha256:<64 hex>"
	fingerprints := []string{
		rec.RequestFingerprint,
		rec.RunFingerprint,
		rec.ReceiptFingerprint,
		rec.TraceFingerprint,
		rec.ProfileFingerprint,
	}
	for _, fp := range fingerprints {
		if !strings.HasPrefix(fp, "sha256:") || len(fp) != len("sha256:")+64 {
			t.Fatalf("invalid fingerprint format: %q", fp)
		}
	}
}

func TestContractEngineValidationFailClosedInvariant(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Input)
		wantErr string
	}{
		{
			name: "engine name not fak-native",
			mutate: func(in *Input) {
				in.Engine.Name = "external-engine"
			},
			wantErr: "engine must be \"fak-native\"",
		},
		{
			name: "engine backend contains URL scheme",
			mutate: func(in *Input) {
				in.Engine.Backend = "https://example.com/api"
			},
			wantErr: "endpoint or credential syntax",
		},
		{
			name: "engine model contains credentials",
			mutate: func(in *Input) {
				in.Engine.Model = "user:pass@model"
			},
			wantErr: "not a scrubbed bounded identity",
		},
		{
			name: "engine quantization contains path separators",
			mutate: func(in *Input) {
				in.Engine.Quantization = "q4_k_m\\exploit"
			},
			wantErr: "not a scrubbed bounded identity",
		},
		{
			name: "engine backend empty",
			mutate: func(in *Input) {
				in.Engine.Backend = ""
			},
			wantErr: "not a scrubbed bounded identity",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			input := testInput("req", "run")
			tc.mutate(&input)
			_, err := scrub(input)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("scrub() error = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}

func TestContractModuleRevValidationInvariant(t *testing.T) {
	tests := []struct {
		name      string
		moduleRev string
		valid     bool
	}{
		{"valid module rev", "internal/nativeperfcorrelation@r1+g0123456", true},
		{"valid complex path", "pkg/sub/mod@r42+gabcdef0123456789", true},
		{"missing rev", "internal/nativeperfcorrelation@g0123456", false},
		{"missing commit", "internal/nativeperfcorrelation@r1", false},
		{"missing at symbol", "internal/nativeperfcorrelation-r1+g0123456", false},
		{"zero revision", "internal/nativeperfcorrelation@r0+g0123456", false},
		{"short commit sha", "internal/nativeperfcorrelation@r1+g01234", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			input := testInput("req", "run")
			input.ModuleAtRev = tc.moduleRev
			_, err := scrub(input)
			if tc.valid && err != nil {
				t.Fatalf("unexpected error for valid moduleRev: %v", err)
			}
			if !tc.valid && err == nil {
				t.Fatalf("expected error for invalid moduleRev: %s", tc.moduleRev)
			}
		})
	}
}

func TestContractArtifactValidationFailClosedInvariant(t *testing.T) {
	validArtifacts := func() []Artifact {
		return []Artifact{
			{Kind: ArtifactReceipt, Locator: "artifacts/receipt.json", SHA256: digest("receipt")},
			{Kind: ArtifactTrace, Locator: "artifacts/trace.json", SHA256: digest("trace")},
			{Kind: ArtifactProfile, Locator: "artifacts/profile.pb.gz", SHA256: digest("profile")},
		}
	}

	tests := []struct {
		name      string
		artifacts []Artifact
		wantErr   string
	}{
		{
			name: "too few artifacts",
			artifacts: []Artifact{
				{Kind: ArtifactReceipt, Locator: "artifacts/receipt.json", SHA256: digest("receipt")},
			},
			wantErr: "requires receipt, trace, and profile",
		},
		{
			name: "duplicate kind",
			artifacts: []Artifact{
				{Kind: ArtifactReceipt, Locator: "artifacts/receipt-1.json", SHA256: digest("r1")},
				{Kind: ArtifactReceipt, Locator: "artifacts/receipt-2.json", SHA256: digest("r2")},
				{Kind: ArtifactTrace, Locator: "artifacts/trace.json", SHA256: digest("t")},
			},
			wantErr: "duplicate receipt artifact",
		},
		{
			name: "unknown kind",
			artifacts: []Artifact{
				{Kind: "snapshot", Locator: "artifacts/snapshot.bin", SHA256: digest("s")},
				{Kind: ArtifactTrace, Locator: "artifacts/trace.json", SHA256: digest("t")},
				{Kind: ArtifactProfile, Locator: "artifacts/profile.pb.gz", SHA256: digest("p")},
			},
			wantErr: "unknown artifact kind",
		},
		{
			name: "locator not under artifacts/",
			artifacts: []Artifact{
				{Kind: ArtifactReceipt, Locator: "data/receipt.json", SHA256: digest("receipt")},
				{Kind: ArtifactTrace, Locator: "artifacts/trace.json", SHA256: digest("trace")},
				{Kind: ArtifactProfile, Locator: "artifacts/profile.pb.gz", SHA256: digest("profile")},
			},
			wantErr: "must be rooted under artifacts/",
		},
		{
			name: "locator contains directory traversal",
			artifacts: []Artifact{
				{Kind: ArtifactReceipt, Locator: "artifacts/../private/receipt.json", SHA256: digest("receipt")},
				{Kind: ArtifactTrace, Locator: "artifacts/trace.json", SHA256: digest("trace")},
				{Kind: ArtifactProfile, Locator: "artifacts/profile.pb.gz", SHA256: digest("profile")},
			},
			wantErr: "bounded relative",
		},
		{
			name: "invalid sha256 format",
			artifacts: []Artifact{
				{Kind: ArtifactReceipt, Locator: "artifacts/receipt.json", SHA256: "not-a-valid-sha"},
				{Kind: ArtifactTrace, Locator: "artifacts/trace.json", SHA256: digest("trace")},
				{Kind: ArtifactProfile, Locator: "artifacts/profile.pb.gz", SHA256: digest("profile")},
			},
			wantErr: "must be lowercase SHA-256",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			input := testInput("req", "run")
			input.Artifacts = tc.artifacts
			_, err := scrub(input)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("scrub() error = %v, want substring %q", err, tc.wantErr)
			}
		})
	}

	// Verify valid artifacts sort deterministically by kind
	input := testInput("req", "run")
	input.Artifacts = validArtifacts()
	rec, err := scrub(input)
	if err != nil {
		t.Fatalf("valid scrub error = %v", err)
	}
	if rec.Artifacts[0].Kind != ArtifactProfile || rec.Artifacts[1].Kind != ArtifactReceipt || rec.Artifacts[2].Kind != ArtifactTrace {
		t.Fatalf("artifacts not sorted deterministically by kind: %+v", rec.Artifacts)
	}
}

func TestContractVerifyArtifactBoundsAndTamperGuard(t *testing.T) {
	index, err := NewIndex(1)
	if err != nil {
		t.Fatal(err)
	}
	rec, err := index.Add(testInput("req", "run"))
	if err != nil {
		t.Fatal(err)
	}

	fsys := fstest.MapFS{
		"artifacts/receipt.json":  &fstest.MapFile{Data: []byte("receipt")},
		"artifacts/trace.json":    &fstest.MapFile{Data: []byte("tampered-trace-content")},
		"artifacts/profile.pb.gz": &fstest.MapFile{Data: []byte("profile-payload-exceeding-limit")},
	}

	// Guard: Valid artifact verifies without error
	if err := index.VerifyArtifact(rec.Key, ArtifactReceipt, fsys, 1024); err != nil {
		t.Fatalf("valid verification error = %v", err)
	}

	// Guard: Tampered content returns ErrDigestMismatch
	if err := index.VerifyArtifact(rec.Key, ArtifactTrace, fsys, 1024); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("tampered verify error = %v, want ErrDigestMismatch", err)
	}

	// Guard: Payload exceeding maxBytes returns ErrArtifactTooLarge
	if err := index.VerifyArtifact(rec.Key, ArtifactProfile, fsys, 4); !errors.Is(err, ErrArtifactTooLarge) {
		t.Fatalf("oversized verify error = %v, want ErrArtifactTooLarge", err)
	}

	// Guard: Missing artifact from filesystem returns ErrArtifactMissing
	delete(fsys, "artifacts/receipt.json")
	if err := index.VerifyArtifact(rec.Key, ArtifactReceipt, fsys, 1024); !errors.Is(err, ErrArtifactMissing) {
		t.Fatalf("missing file verify error = %v, want ErrArtifactMissing", err)
	}

	// Guard: Non-existent correlation key returns ErrNotFound
	if err := index.VerifyArtifact("npc1_nonexistentkey999", ArtifactReceipt, fsys, 1024); !errors.Is(err, ErrNotFound) {
		t.Fatalf("non-existent key verify error = %v, want ErrNotFound", err)
	}
}

func TestContractConcurrentSafetyInvariant(t *testing.T) {
	index, err := NewIndex(50)
	if err != nil {
		t.Fatal(err)
	}

	const workers = 16
	const iterations = 50
	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				input := testInput(fmt.Sprintf("w-%d-req-%d", workerID, i), fmt.Sprintf("w-%d-run-%d", workerID, i))
				rec, err := index.Add(input)
				if err != nil {
					t.Errorf("worker %d Add error = %v", workerID, err)
					return
				}

				if _, err := index.Lookup(rec.Key); err != nil && !errors.Is(err, ErrNotFound) {
					t.Errorf("worker %d Lookup error = %v", workerID, err)
					return
				}

				_ = index.Snapshot()
			}
		}(w)
	}

	wg.Wait()
	snapshot := index.Snapshot()
	if len(snapshot) > 50 {
		t.Fatalf("index exceeded capacity under concurrency: len = %d", len(snapshot))
	}
}

func TestContractSnapshotImmutabilityInvariant(t *testing.T) {
	index, err := NewIndex(5)
	if err != nil {
		t.Fatal(err)
	}

	rec, err := index.Add(testInput("req-immut", "run-immut"))
	if err != nil {
		t.Fatal(err)
	}

	// Mutate the returned record's Artifacts slice
	rec.Artifacts[0].Locator = "artifacts/mutated.json"

	// Fetch fresh copy via Lookup and Snapshot
	lookedUp, err := index.Lookup(rec.Key)
	if err != nil {
		t.Fatal(err)
	}
	if lookedUp.Artifacts[0].Locator == "artifacts/mutated.json" {
		t.Fatal("internal cache mutated through Lookup record")
	}

	snapshot := index.Snapshot()
	if snapshot[0].Artifacts[0].Locator == "artifacts/mutated.json" {
		t.Fatal("internal cache mutated through Snapshot record")
	}
}

func TestContractNewIndexPreconditions(t *testing.T) {
	if _, err := NewIndex(0); err == nil {
		t.Fatal("NewIndex(0) succeeded, want error")
	}
	if _, err := NewIndex(-5); err == nil {
		t.Fatal("NewIndex(-5) succeeded, want error")
	}
}
