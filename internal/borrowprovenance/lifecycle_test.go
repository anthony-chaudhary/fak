package borrowprovenance

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestLifecyclePinSerializeVerify(t *testing.T) {
	sourceContent := []byte("// borrowed kernel excerpt\nfunc FastMath() int { return 42 }\n")
	record, err := Pin(
		"https://github.com/example/mathlib",
		"v1.2.3",
		"pkg/math/fast.go",
		"MIT",
		"extracted FastMath function without modifications",
		sourceContent,
	)
	if err != nil {
		t.Fatalf("Pin failed unexpectedly: %v", err)
	}

	// Invariant check: generated checksum matches digest of source bytes.
	expectedDigest := Digest(sourceContent)
	if record.SourceSHA256 != expectedDigest {
		t.Fatalf("SourceSHA256 mismatch: got %s, want %s", record.SourceSHA256, expectedDigest)
	}

	// Serialize record to JSON and reconstruct it.
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var restored Record
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if err := restored.Validate(); err != nil {
		t.Fatalf("restored record failed validation: %v", err)
	}

	// Verify exact match.
	verif, err := Verify(restored, sourceContent)
	if err != nil {
		t.Fatalf("Verify failed unexpectedly: %v", err)
	}
	if !verif.Match {
		t.Fatalf("expected verification match, got mismatch: %+v", verif)
	}
	if verif.ExpectedSHA256 != verif.ActualSHA256 {
		t.Fatalf("expected matching hashes, got expected=%s actual=%s", verif.ExpectedSHA256, verif.ActualSHA256)
	}

	// Verify drift detection when source is altered.
	mutatedContent := append(bytes.Clone(sourceContent), byte('\n'))
	driftVerif, err := Verify(restored, mutatedContent)
	if err != nil {
		t.Fatalf("Verify with mutated content failed with error: %v", err)
	}
	if driftVerif.Match {
		t.Fatalf("expected drift detection, but verification passed: %+v", driftVerif)
	}
	if driftVerif.ExpectedSHA256 == driftVerif.ActualSHA256 {
		t.Fatalf("expected distinct hashes on drift, got identical: %s", driftVerif.ExpectedSHA256)
	}
}

func TestMultiBorrowTracking(t *testing.T) {
	sources := map[string][]byte{
		"header.h": []byte("#define CONST_A 100\n"),
		"impl.c":   []byte("int run() { return CONST_A; }\n"),
		"util.c":   []byte("void noop() {}\n"),
	}

	records := make(map[string]Record)
	for path, content := range sources {
		rec, err := Pin(
			"https://git.example.com/upstream/c-core",
			"commit-sha-abcdef0123456789",
			path,
			"Apache-2.0",
			"vendored into internal subsystem",
			content,
		)
		if err != nil {
			t.Fatalf("failed to pin %s: %v", path, err)
		}
		records[path] = rec
	}

	// Track and verify all records.
	for path, content := range sources {
		rec := records[path]
		verif, err := Verify(rec, content)
		if err != nil {
			t.Fatalf("Verify failed on %s: %v", path, err)
		}
		if !verif.Match {
			t.Fatalf("verification match failed on %s", path)
		}
	}

	// Introduce drift in one item and ensure isolated failure.
	tamperedPath := "impl.c"
	tamperedContent := []byte("int run() { return 0; }\n")
	tamperedVerif, err := Verify(records[tamperedPath], tamperedContent)
	if err != nil {
		t.Fatalf("Verify returned unexpected error on tampered content: %v", err)
	}
	if tamperedVerif.Match {
		t.Fatalf("tampered item %s should have reported mismatch", tamperedPath)
	}
}

func TestValidationBoundsAndFailClosed(t *testing.T) {
	validSHA := Digest([]byte("sample"))

	tests := []struct {
		name    string
		record  Record
		wantErr bool
	}{
		{
			name: "valid complete record",
			record: Record{
				Schema:         Schema,
				SourceURL:      "https://example.org/repo",
				SourceRef:      "main",
				SourcePath:     "path/to/file",
				SourceSHA256:   validSHA,
				License:        "BSD-3-Clause",
				Transformation: "none",
			},
			wantErr: false,
		},
		{
			name: "valid minimal record without optional fields",
			record: Record{
				Schema:       Schema,
				SourceURL:    "https://example.org/repo",
				SourceRef:    "v1.0",
				SourceSHA256: validSHA,
			},
			wantErr: false,
		},
		{
			name: "invalid schema string",
			record: Record{
				Schema:       "fak/borrow-provenance/v2",
				SourceURL:    "https://example.org/repo",
				SourceRef:    "main",
				SourceSHA256: validSHA,
			},
			wantErr: true,
		},
		{
			name: "empty schema",
			record: Record{
				Schema:       "",
				SourceURL:    "https://example.org/repo",
				SourceRef:    "main",
				SourceSHA256: validSHA,
			},
			wantErr: true,
		},
		{
			name: "empty source url",
			record: Record{
				Schema:       Schema,
				SourceURL:    "   ",
				SourceRef:    "main",
				SourceSHA256: validSHA,
			},
			wantErr: true,
		},
		{
			name: "empty source ref",
			record: Record{
				Schema:       Schema,
				SourceURL:    "https://example.org/repo",
				SourceRef:    "\t\n ",
				SourceSHA256: validSHA,
			},
			wantErr: true,
		},
		{
			name: "hash length too short",
			record: Record{
				Schema:       Schema,
				SourceURL:    "https://example.org/repo",
				SourceRef:    "main",
				SourceSHA256: strings.Repeat("a", 63),
			},
			wantErr: true,
		},
		{
			name: "hash length too long",
			record: Record{
				Schema:       Schema,
				SourceURL:    "https://example.org/repo",
				SourceRef:    "main",
				SourceSHA256: strings.Repeat("a", 65),
			},
			wantErr: true,
		},
		{
			name: "hash contains non-hex characters",
			record: Record{
				Schema:       Schema,
				SourceURL:    "https://example.org/repo",
				SourceRef:    "main",
				SourceSHA256: strings.Repeat("z", 64),
			},
			wantErr: true,
		},
		{
			name: "hash contains uppercase characters",
			record: Record{
				Schema:       Schema,
				SourceURL:    "https://example.org/repo",
				SourceRef:    "main",
				SourceSHA256: strings.ToUpper(validSHA),
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.record.Validate()
			if tc.wantErr && err == nil {
				t.Errorf("Validate() expected error for case %q, got nil", tc.name)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("Validate() unexpected error for case %q: %v", tc.name, err)
			}
		})
	}
}

func TestVerifyRejectsInvalidRecord(t *testing.T) {
	invalidRecord := Record{
		Schema:       "bad-schema",
		SourceURL:    "https://example.org",
		SourceRef:    "main",
		SourceSHA256: Digest([]byte("data")),
	}

	_, err := Verify(invalidRecord, []byte("data"))
	if err == nil {
		t.Fatal("Verify should have failed validation on invalid record, but returned nil")
	}
}

func TestPinFailClosedOnInvalidInput(t *testing.T) {
	data := []byte("payload")
	if _, err := Pin("", "ref", "path", "lic", "trans", data); err == nil {
		t.Error("Pin with empty sourceURL should fail")
	}
	if _, err := Pin("   ", "ref", "path", "lic", "trans", data); err == nil {
		t.Error("Pin with whitespace sourceURL should fail")
	}
	if _, err := Pin("https://example.org", "", "path", "lic", "trans", data); err == nil {
		t.Error("Pin with empty sourceRef should fail")
	}
	if _, err := Pin("https://example.org", "\n\t", "path", "lic", "trans", data); err == nil {
		t.Error("Pin with whitespace sourceRef should fail")
	}
}

func TestDigestWellKnownValues(t *testing.T) {
	// Empty slice SHA-256 standard test vector.
	emptyDigest := Digest([]byte{})
	const wantEmpty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	if emptyDigest != wantEmpty {
		t.Fatalf("Digest([]byte{}) = %s, want %s", emptyDigest, wantEmpty)
	}

	// Determinism check.
	d1 := Digest([]byte("determinism check payload"))
	d2 := Digest([]byte("determinism check payload"))
	if d1 != d2 {
		t.Fatalf("Digest not deterministic: %s != %s", d1, d2)
	}
}

func BenchmarkBorrowProvenance(b *testing.B) {
	sourceBytes := []byte("func SampleAlgorithm() int { return 1024 }\n")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		record, err := Pin(
			"https://example.com/algorithm",
			"release-v1.0",
			"algo.go",
			"Apache-2.0",
			"vendored function",
			sourceBytes,
		)
		if err != nil {
			b.Fatal(err)
		}
		verif, err := Verify(record, sourceBytes)
		if err != nil {
			b.Fatal(err)
		}
		if !verif.Match {
			b.Fatal("verification failed during benchmark")
		}
	}
}

func BenchmarkDigest(b *testing.B) {
	payload := bytes.Repeat([]byte("deterministic benchmark payload line\n"), 64)
	b.SetBytes(int64(len(payload)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Digest(payload)
	}
}

func BenchmarkVerify(b *testing.B) {
	sourceBytes := bytes.Repeat([]byte("package benchmark\n"), 32)
	record, err := Pin("https://example.com/repo", "v1", "bench.go", "MIT", "", sourceBytes)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		verif, err := Verify(record, sourceBytes)
		if err != nil {
			b.Fatal(err)
		}
		if !verif.Match {
			b.Fatal("mismatch in benchmark")
		}
	}
}
