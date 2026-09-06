package resultstier

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func setupBenchmarkDir(b *testing.B) string {
	b.Helper()
	dir := b.TempDir()

	files := map[string][]byte{
		"INDEX.md":                     []byte("# Test Index\nSummary of results.\n"),
		"perf.json":                    []byte(`{"latency_ms": 42.5, "throughput": 120.3}`),
		"run_manifest.json":            []byte(`{"run_id": "r1", "timestamp": "2026-09-06T00:00:00Z"}`),
		"eval_summary.json":            []byte(`{"pass": 98, "fail": 2}`),
		"README.md":                    []byte("# Readme\nDocumentation of run.\n"),
		"results.sha256":               []byte("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  INDEX.md\n"),
		"queries.sql":                  []byte("SELECT * FROM metrics;\n"),
		"driver.sh":                    []byte("#!/bin/bash\necho 'running'\n"),
		"predictions_epoch1.json":      []byte(strings.Repeat("prediction: sample output token tensor ", 20)),
		"logs/build.log":               []byte(strings.Repeat("INFO: step completed successfully\n", 30)),
		"logs/pmon-gpu0.txt":           []byte("utilization.gpu [0 %], memory.used [1200 MiB]\n"),
		"nested/times-001.json":        []byte(`[10.2, 10.4, 10.1, 10.5, 10.3]`),
		"nested/data/measurements.csv": []byte("step,loss,lr\n1,0.5,1e-4\n2,0.4,1e-4\n3,0.3,1e-4\n"),
		"artifacts/render.png":         []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"),
		"unknown_artifact.bin":         []byte("\x00\x01\x02\x03\x04\x05"),
		"models/network.onnx":          []byte("ONNX model binary placeholder bytes"),
		".gitignore":                   []byte("*.tmp\n*.bak\n"),
		".DS_Store":                    []byte("macOS metadata"),
	}

	for relPath, content := range files {
		fullPath := filepath.Join(dir, filepath.FromSlash(relPath))
		if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
			b.Fatalf("failed to create directory for %s: %v", relPath, err)
		}
		if err := os.WriteFile(fullPath, content, 0644); err != nil {
			b.Fatalf("failed to write %s: %v", relPath, err)
		}
	}

	gitDir := filepath.Join(dir, ".git")
	if err := os.MkdirAll(gitDir, 0755); err != nil {
		b.Fatalf("failed to create .git: %v", err)
	}
	if err := os.WriteFile(filepath.Join(gitDir, "HEAD"), []byte("ref: refs/heads/main\n"), 0644); err != nil {
		b.Fatalf("failed to write .git/HEAD: %v", err)
	}

	return dir
}

// BenchmarkTierOf measures rule evaluation latency across diverse artifact path patterns.
func BenchmarkTierOf(b *testing.B) {
	cases := []struct {
		name string
		path string
	}{
		{"Claim_Direct", "INDEX.md"},
		{"Claim_Pattern", "nested/eval_summary.json"},
		{"Payload_Direct", "prefill.json"},
		{"Payload_Pattern", "artifacts/run1/predictions_epoch1.json"},
		{"Payload_Extension", "deep/logs/build.log"},
		{"Unknown_Extension", "data/model.onnx"},
		{"Unknown_Root", "random.bin"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tier, _ := TierOf(tc.path)
				_ = tier
			}
		})
	}
}

// BenchmarkPayloadIndex_Lookup measures exact and normalized path lookup performance in PayloadIndex.
func BenchmarkPayloadIndex_Lookup(b *testing.B) {
	entries := make([]PayloadEntry, 100)
	validSHA := strings.Repeat("a", 64)
	for i := 0; i < len(entries); i++ {
		entries[i] = PayloadEntry{
			Path:   fmt.Sprintf("artifacts/run_%d/predictions.json", i),
			Bytes:  1024,
			SHA256: validSHA,
		}
	}
	idx := PayloadIndex{
		Schema:   PayloadIndexSchema,
		StoreURI: "s3://results/store",
		Entries:  entries,
	}

	b.Run("ExactHit", func(b *testing.B) {
		target := "artifacts/run_50/predictions.json"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			entry, ok := idx.Lookup(target)
			_ = entry
			_ = ok
		}
	})

	b.Run("NormalizedHit", func(b *testing.B) {
		target := "./artifacts/run_50/predictions.json"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			entry, ok := idx.Lookup(target)
			_ = entry
			_ = ok
		}
	})

	b.Run("Miss", func(b *testing.B) {
		target := "artifacts/run_999/not_found.json"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			entry, ok := idx.Lookup(target)
			_ = entry
			_ = ok
		}
	})
}

// BenchmarkClassify measures artifact classification throughput over varied path batch sizes.
func BenchmarkClassify(b *testing.B) {
	sizes := []int{10, 100, 1000}
	basePool := []string{
		"INDEX.md",
		"artifacts/run_manifest.json",
		"logs/build.log",
		"predictions_epoch1.json",
		"benchmarks/perf.json",
		"unknown_artifact.bin",
		"eval_summary.json",
		"pmon-gpu0.txt",
		"deep/nested/path/times-001.json",
		"data/model.onnx",
	}

	for _, size := range sizes {
		paths := make([]string, size)
		for i := 0; i < size; i++ {
			paths[i] = fmt.Sprintf("sub_%d/%s", i%5, basePool[i%len(basePool)])
		}
		b.Run(fmt.Sprintf("paths=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res := Classify(paths)
				_ = res
			}
		})
	}
}

// BenchmarkCensus_Metrics measures calculation and serialization cost of storage Census.
func BenchmarkCensus_Metrics(b *testing.B) {
	c := Census{
		ClaimFiles:   25,
		ClaimBytes:   1024 * 100,
		PayloadFiles: 75,
		PayloadBytes: 1024 * 1024 * 50,
		UnknownFiles: 5,
		UnknownBytes: 1024 * 10,
	}

	b.Run("Calculations", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			totF := c.TotalFiles()
			totB := c.TotalBytes()
			share := c.PayloadShare()
			shrink := c.Shrink()
			_ = totF
			_ = totB
			_ = share
			_ = shrink
		}
	})

	b.Run("String", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := c.String()
			_ = s
		}
	})
}

// BenchmarkPayloadEntry_VolumeKey measures validation and content-addressed key generation.
func BenchmarkPayloadEntry_VolumeKey(b *testing.B) {
	entry := PayloadEntry{
		Path:   "predictions_epoch1.json",
		Bytes:  4096,
		SHA256: strings.Repeat("f", 64),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		key, err := entry.VolumeKey()
		if err != nil {
			b.Fatal(err)
		}
		_ = key
	}
}

// BenchmarkCanMigrate measures migration gate enforcement checks.
func BenchmarkCanMigrate(b *testing.B) {
	validSHA := strings.Repeat("c", 64)
	idx := PayloadIndex{
		Schema:   PayloadIndexSchema,
		StoreURI: "s3://results/store",
		Entries: []PayloadEntry{
			{Path: "logs/build.log", Bytes: 1024, SHA256: validSHA},
			{Path: "data/predictions.json", Bytes: 2048, SHA256: validSHA},
		},
	}

	b.Run("Eligible", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := CanMigrate(idx, "logs/build.log"); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("IneligibleClaim", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = CanMigrate(idx, "perf.json")
		}
	})
}

// BenchmarkMintPayloadIndex measures full directory traversal, classification, payload hashing, and census minting.
func BenchmarkMintPayloadIndex(b *testing.B) {
	dir := setupBenchmarkDir(b)
	storeURI := "s3://results-archive/store"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, census, err := MintPayloadIndex(dir, storeURI)
		if err != nil {
			b.Fatalf("MintPayloadIndex failed: %v", err)
		}
		if len(idx.Entries) != 6 {
			b.Fatalf("expected 6 payload entries, got %d", len(idx.Entries))
		}
		if census.ClaimFiles != 8 {
			b.Fatalf("expected 8 claim files, got %d", census.ClaimFiles)
		}
	}
}

// BenchmarkVerifyPayloadIndex measures filesystem verification, stat checks, and SHA-256 integrity validation.
func BenchmarkVerifyPayloadIndex(b *testing.B) {
	dir := setupBenchmarkDir(b)
	storeURI := "s3://results-archive/store"

	idx, _, err := MintPayloadIndex(dir, storeURI)
	if err != nil {
		b.Fatalf("MintPayloadIndex failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		discrepancies, err := VerifyPayloadIndex(dir, idx)
		if err != nil {
			b.Fatalf("VerifyPayloadIndex failed: %v", err)
		}
		if len(discrepancies) != 0 {
			b.Fatalf("unexpected discrepancies: %v", discrepancies)
		}
	}
}
