package cdb

import (
	"context"
	"path/filepath"
	"testing"
)

// BenchmarkCDB exercises page table lookup and session core image inspection in a loop.
func BenchmarkCDB(b *testing.B) {
	ctx := context.Background()
	fixturePath := filepath.Join("..", "..", "testdata", "cdb", "session.jsonl")
	rec, st, err := IngestSession(ctx, fixturePath, "cdb-bench-fixture")
	if err != nil {
		b.Fatalf("ingest fixture: %v", err)
	}
	if st.Pages == 0 {
		b.Fatalf("ingest recorded 0 pages")
	}

	dir := b.TempDir()
	if err := rec.Persist(dir); err != nil {
		b.Fatalf("persist image: %v", err)
	}
	im, err := Attach(dir)
	if err != nil {
		b.Fatalf("attach image: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		frames := im.Backtrace()
		if len(frames) == 0 {
			b.Fatal("backtrace returned 0 frames")
		}
		for step := 0; step < st.Pages; step++ {
			if _, ok := im.PageCacheEntry(step); !ok {
				b.Fatalf("page cache entry missing for step %d", step)
			}
		}
		_ = im.Info()
	}
}
