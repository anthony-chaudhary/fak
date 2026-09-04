package blob

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// BenchmarkDigest measures sha256 content addressing across representative
// payload sizes: inline threshold (64B), small CAS blob (1KB), medium
// buffer (64KB), and large chunk (1MB).
func BenchmarkDigest(b *testing.B) {
	sizes := []int{64, 1024, 64 * 1024, 1024 * 1024}
	for _, size := range sizes {
		name := fmt.Sprintf("%dB", size)
		if size >= 1024*1024 {
			name = fmt.Sprintf("%dMB", size/(1024*1024))
		} else if size >= 1024 {
			name = fmt.Sprintf("%dKB", size/1024)
		}
		data := makeBytes(size, 42)
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = Digest(data)
			}
		})
	}
}

// BenchmarkPreparePut_Inline measures the zero-CAS split prologue for payloads
// at or below InlineMax (<=256 bytes) that ride directly inline on the Ref.
func BenchmarkPreparePut_Inline(b *testing.B) {
	payload := makeBytes(64, 17)
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, inline := PreparePut(payload)
		if !inline || r.Kind != abi.RefInline {
			b.Fatalf("expected inline ref")
		}
	}
}

// BenchmarkPreparePut_Blob measures the split prologue for payloads exceeding
// InlineMax (>256 bytes) that require external CAS storage.
func BenchmarkPreparePut_Blob(b *testing.B) {
	payload := makeBytes(1024, 19)
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, inline := PreparePut(payload)
		if inline || r.Kind != abi.RefBlob {
			b.Fatalf("expected blob ref")
		}
	}
}

// BenchmarkPut_Inline measures the hot-path store Put for small payloads
// that bypass CAS locking and map mutations entirely.
func BenchmarkPut_Inline(b *testing.B) {
	ctx := context.Background()
	s := New()
	payload := makeBytes(64, 23)
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Put(ctx, payload); err != nil {
			b.Fatalf("Put inline: %v", err)
		}
	}
}

// BenchmarkPut_BlobDedup measures the content-addressed deduplication path:
// repeated Put of an already-resident payload (>InlineMax) verifying the
// hash, RWMutex acquisition, map hit, and counter increment.
func BenchmarkPut_BlobDedup(b *testing.B) {
	ctx := context.Background()
	s := New()
	payload := makeBytes(1024, 29)
	if _, err := s.Put(ctx, payload); err != nil {
		b.Fatalf("seed Put: %v", err)
	}

	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Put(ctx, payload); err != nil {
			b.Fatalf("Put dedup: %v", err)
		}
	}
}

// BenchmarkPut_BlobNew measures ingestion of distinct CAS blobs, exercising
// digest calculation, write lock acquisition, map insertion, and LRU tracking.
func BenchmarkPut_BlobNew(b *testing.B) {
	ctx := context.Background()
	const size = 512
	s := newStore(0)
	b.SetBytes(int64(size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := distinctPayload(i, size)
		if _, err := s.Put(ctx, p); err != nil {
			b.Fatalf("Put new: %v", err)
		}
	}
}

// BenchmarkPut_WithEviction measures steady-state Put throughput when the CAS
// is at capacity and each new distinct insertion evicts an unpinned entry.
func BenchmarkPut_WithEviction(b *testing.B) {
	ctx := context.Background()
	const size = 512
	const capacity = 128
	const maxBytes = int64(capacity * size)
	s := NewWithBudget(maxBytes)

	// Pre-fill to reach the eviction threshold
	for i := 0; i < capacity; i++ {
		if _, err := s.Put(ctx, distinctPayload(i, size)); err != nil {
			b.Fatalf("pre-fill Put: %v", err)
		}
	}

	b.SetBytes(int64(size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := distinctPayload(capacity+i, size)
		if _, err := s.Put(ctx, p); err != nil {
			b.Fatalf("Put with eviction: %v", err)
		}
	}
}

// BenchmarkResolve_Inline measures resolving inline payloads from RefInline.
func BenchmarkResolve_Inline(b *testing.B) {
	ctx := context.Background()
	s := New()
	payload := makeBytes(64, 31)
	ref, err := s.Put(ctx, payload)
	if err != nil {
		b.Fatalf("setup: %v", err)
	}

	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := s.Resolve(ctx, ref)
		if err != nil || len(res) != len(payload) {
			b.Fatalf("Resolve inline failed: %v", err)
		}
	}
}

// BenchmarkResolve_Blob measures resolving resident CAS blobs under RLock,
// atomic resolv counter increment, and memory defensive copying.
func BenchmarkResolve_Blob(b *testing.B) {
	ctx := context.Background()
	s := New()
	payload := makeBytes(1024, 37)
	ref, err := s.Put(ctx, payload)
	if err != nil {
		b.Fatalf("setup: %v", err)
	}

	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := s.Resolve(ctx, ref)
		if err != nil || len(res) != len(payload) {
			b.Fatalf("Resolve blob failed: %v", err)
		}
	}
}

// BenchmarkResolve_BlobParallel measures concurrent read throughput on a shared
// CAS blob across goroutines (production workload for K-arm replay and parallel workers).
func BenchmarkResolve_BlobParallel(b *testing.B) {
	ctx := context.Background()
	s := New()
	payload := makeBytes(1024, 41)
	ref, err := s.Put(ctx, payload)
	if err != nil {
		b.Fatalf("setup: %v", err)
	}

	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res, err := s.Resolve(ctx, ref)
			if err != nil || len(res) != len(payload) {
				b.Fatalf("parallel Resolve failed: %v", err)
			}
		}
	})
}

// BenchmarkPageOut measures migrating an inline Ref to a CAS-backed RefBlob
// (the context-MMU cold/quarantine page-out primitive).
func BenchmarkPageOut(b *testing.B) {
	ctx := context.Background()
	s := New()
	inlineRef := abi.Ref{
		Kind:   abi.RefInline,
		Inline: makeBytes(128, 43),
		Len:    128,
		Taint:  abi.TaintQuarantined,
		Scope:  abi.ScopeFleet,
	}

	b.SetBytes(128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle, err := s.PageOut(ctx, inlineRef)
		if err != nil || handle.Kind != abi.RefBlob {
			b.Fatalf("PageOut failed: %v", err)
		}
	}
}

// BenchmarkPageIn measures re-materializing a paged-out handle Ref into an inline Ref.
func BenchmarkPageIn(b *testing.B) {
	ctx := context.Background()
	s := New()
	inlineRef := abi.Ref{
		Kind:   abi.RefInline,
		Inline: makeBytes(128, 47),
		Len:    128,
		Taint:  abi.TaintQuarantined,
		Scope:  abi.ScopeFleet,
	}
	handle, err := s.PageOut(ctx, inlineRef)
	if err != nil {
		b.Fatalf("setup PageOut: %v", err)
	}

	b.SetBytes(128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		in, err := s.PageIn(ctx, handle)
		if err != nil || in.Kind != abi.RefInline {
			b.Fatalf("PageIn failed: %v", err)
		}
	}
}

// BenchmarkPinUnpin measures the latency of pin refcounting and LRU element
// protection/restoration for active working set retention.
func BenchmarkPinUnpin(b *testing.B) {
	ctx := context.Background()
	s := newStore(0)
	ref, err := s.Put(ctx, makeBytes(1024, 53))
	if err != nil {
		b.Fatalf("setup: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Pin(ref.Digest)
		s.Unpin(ref.Digest)
	}
}
