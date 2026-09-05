package blobfs

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

var (
	benchSinkRef   abi.Ref
	benchSinkBytes []byte
)

func benchDistinctPayload(seq, n int) []byte {
	b := make([]byte, n)
	for j := range b {
		b[j] = byte(seq*17 + j*31)
	}
	for k := 0; k < 8 && k < n; k++ {
		b[k] = byte(uint(seq) >> (8 * uint(k)))
	}
	return b
}

// BenchmarkStore exercises core blobfs operations under representative production workloads.
func BenchmarkStore(b *testing.B) {
	b.Run("Put_Inline", BenchmarkStore_Put_Inline)
	b.Run("Put_BlobDedup", BenchmarkStore_Put_BlobDedup)
	b.Run("Put_BlobNew", BenchmarkStore_Put_BlobNew)
	b.Run("PutAsync", BenchmarkStore_PutAsync)
	b.Run("Resolve_Inline", BenchmarkStore_Resolve_Inline)
	b.Run("Resolve_Blob", BenchmarkStore_Resolve_Blob)
	b.Run("Resolve_BlobParallel", BenchmarkStore_Resolve_BlobParallel)
	b.Run("PageOut", BenchmarkStore_PageOut)
	b.Run("PageIn", BenchmarkStore_PageIn)
	b.Run("PinUnpin", BenchmarkStore_PinUnpin)
	b.Run("Scan", BenchmarkStore_Scan)
}

// BenchmarkStore_Put_Inline measures the hot-path store Put for small payloads
// (<= InlineMax) that stay inline on the Ref and never touch disk.
func BenchmarkStore_Put_Inline(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	data := payload(64, 'i')
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := s.Put(ctx, data)
		if err != nil || r.Kind != abi.RefInline {
			b.Fatalf("Put inline: %v", err)
		}
		benchSinkRef = r
	}
}

// BenchmarkStore_Put_BlobDedup measures content-addressed deduplication:
// repeated Put of an already-resident payload (> InlineMax) verifying hash,
// mutex acquisition, in-memory index hit, and metric updates without rewriting to disk.
func BenchmarkStore_Put_BlobDedup(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	data := payload(1024, 'd')
	if _, err := s.Put(ctx, data); err != nil {
		b.Fatalf("seed Put: %v", err)
	}
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := s.Put(ctx, data)
		if err != nil || r.Kind != abi.RefBlob {
			b.Fatalf("Put dedup: %v", err)
		}
		benchSinkRef = r
	}
}

// BenchmarkStore_Put_BlobNew measures ingestion of distinct CAS blobs:
// digest computation, temp file creation, payload write, fsync, atomic rename, and index update.
func BenchmarkStore_Put_BlobNew(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	const size = 1024
	b.SetBytes(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := benchDistinctPayload(i, size)
		r, err := s.Put(ctx, p)
		if err != nil || r.Kind != abi.RefBlob {
			b.Fatalf("Put new: %v", err)
		}
		benchSinkRef = r
	}
}

// BenchmarkStore_PutAsync measures non-blocking asynchronous durable writes:
// synchronous buffer copy, inflight coalesce check, and bounded channel enqueue.
func BenchmarkStore_PutAsync(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	defer s.Close()
	const size = 1024
	b.SetBytes(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := benchDistinctPayload(i, size)
		r, err := s.PutAsync(ctx, p)
		if err != nil || r.Kind != abi.RefBlob {
			b.Fatalf("PutAsync: %v", err)
		}
		benchSinkRef = r
	}
	b.StopTimer()
	if err := s.Flush(); err != nil {
		b.Fatalf("Flush: %v", err)
	}
}

// BenchmarkStore_Resolve_Inline measures resolving inline payloads from RefInline.
func BenchmarkStore_Resolve_Inline(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	data := payload(64, 'r')
	r, err := s.Put(ctx, data)
	if err != nil {
		b.Fatalf("Put: %v", err)
	}
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := s.Resolve(ctx, r)
		if err != nil || len(got) != len(data) {
			b.Fatalf("Resolve inline: %v", err)
		}
		benchSinkBytes = got
	}
}

// BenchmarkStore_Resolve_Blob measures resolving durable on-disk CAS blobs:
// path lookup, file read, and sha256 checksum verification.
func BenchmarkStore_Resolve_Blob(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	data := payload(4096, 'b')
	r, err := s.Put(ctx, data)
	if err != nil {
		b.Fatalf("Put: %v", err)
	}
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := s.Resolve(ctx, r)
		if err != nil || len(got) != len(data) {
			b.Fatalf("Resolve blob: %v", err)
		}
		benchSinkBytes = got
	}
}

// BenchmarkStore_Resolve_BlobParallel measures concurrent read throughput on a shared
// CAS blob across multiple goroutines.
func BenchmarkStore_Resolve_BlobParallel(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	data := payload(4096, 'p')
	r, err := s.Put(ctx, data)
	if err != nil {
		b.Fatalf("Put: %v", err)
	}
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			got, err := s.Resolve(ctx, r)
			if err != nil || len(got) != len(data) {
				b.Fatalf("parallel Resolve blob: %v", err)
			}
		}
	})
}

// BenchmarkStore_PageOut measures migrating an inline Ref to an on-disk CAS-backed RefBlob.
func BenchmarkStore_PageOut(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	inlineRef := abi.Ref{
		Kind:   abi.RefInline,
		Inline: payload(128, 'q'),
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
			b.Fatalf("PageOut: %v", err)
		}
		benchSinkRef = handle
	}
}

// BenchmarkStore_PageIn measures re-materializing a paged-out handle Ref into an inline Ref.
func BenchmarkStore_PageIn(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	inlineRef := abi.Ref{
		Kind:   abi.RefInline,
		Inline: payload(128, 'k'),
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
			b.Fatalf("PageIn: %v", err)
		}
		benchSinkRef = in
	}
}

// BenchmarkStore_PinUnpin measures the latency of pin refcounting and LRU element
// protection and restoration.
func BenchmarkStore_PinUnpin(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	ref, err := s.Put(ctx, payload(1024, 'u'))
	if err != nil {
		b.Fatalf("setup Put: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Pin(ref.Digest)
		s.Unpin(ref.Digest)
	}
}

// BenchmarkStore_Scan measures startup recovery and directory tree scanning over
// pre-existing on-disk blobs.
func BenchmarkStore_Scan(b *testing.B) {
	ctx := context.Background()
	dir := b.TempDir()
	s, err := New(dir)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	const count = 32
	for i := 0; i < count; i++ {
		if _, err := s.Put(ctx, benchDistinctPayload(i, 512)); err != nil {
			b.Fatalf("seed Put: %v", err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reopened, err := New(dir)
		if err != nil {
			b.Fatalf("reopen New: %v", err)
		}
		if cnt, _, _ := reopened.Resident(); cnt != count {
			b.Fatalf("reopened resident count = %d, want %d", cnt, count)
		}
	}
}
