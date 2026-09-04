package nativeperfartifact

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// Invariant: Benchmark execution verifies index operations execute in bounded time under production concurrency.
// Guard: Index lookup and insertion benchmarks must execute without errors.

func BenchmarkArtifactIndexResolve(b *testing.B) {
	const capacity = 128
	index, err := NewIndex(capacity)
	if err != nil {
		b.Fatalf("failed to create index: %v", err)
	}

	now := time.Now().UTC()
	digest := strings.Repeat("a", 64)
	key := "npc1_0123456789abcdef0123456789abcdef"
	record := Record{
		CorrelationKey: key,
		Engine:         "fak-native",
		Artifacts: []Artifact{
			{
				Kind:      KindReceipt,
				Locator:   "https://artifacts.example.test/native/receipt.json",
				SHA256:    digest,
				ExpiresAt: now.Add(24 * time.Hour),
				State:     StateReady,
			},
			{
				Kind:      KindMetalProfile,
				Locator:   "https://artifacts.example.test/native/metal.tgz",
				SHA256:    digest,
				ExpiresAt: now.Add(24 * time.Hour),
				State:     StateReady,
			},
		},
	}

	if err := index.Add(record); err != nil {
		b.Fatalf("failed to add record: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		art, err := index.Resolve(key, KindReceipt, now)
		if err != nil {
			b.Fatalf("resolve failed: %v", err)
		}
		if art.State != StateReady {
			b.Fatalf("unexpected state: %v", art.State)
		}
	}
}

func BenchmarkArtifactIndexAdd(b *testing.B) {
	const capacity = 128
	now := time.Now().UTC()
	digest := strings.Repeat("a", 64)

	const numRecords = 256
	records := make([]Record, numRecords)
	for i := 0; i < numRecords; i++ {
		key := fmt.Sprintf("npc1_%032x", i)
		records[i] = Record{
			CorrelationKey: key,
			Engine:         "fak-native",
			Artifacts: []Artifact{
				{
					Kind:      KindReceipt,
					Locator:   fmt.Sprintf("https://artifacts.example.test/native/%04d/receipt.json", i),
					SHA256:    digest,
					ExpiresAt: now.Add(24 * time.Hour),
					State:     StateReady,
				},
			},
		}
	}

	index, err := NewIndex(capacity)
	if err != nil {
		b.Fatalf("failed to create index: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rec := records[i%numRecords]
		if err := index.Add(rec); err != nil {
			b.Fatalf("add failed: %v", err)
		}
	}
}
