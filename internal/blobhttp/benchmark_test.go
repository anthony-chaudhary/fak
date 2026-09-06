package blobhttp

import (
	"context"
	"net/http/httptest"
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

// BenchmarkPutResolveRoundTrip measures the complete Put + Resolve cycle for a
// representative 4KB payload over HTTP, exercising digest calculation, HEAD probe,
// upload/dedup, and download with read-time digest verification.
func BenchmarkPutResolveRoundTrip(b *testing.B) {
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL, WithClient(srv.Client()))

	data := payload(4096, 'r')
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ref, err := s.Put(ctx, data)
		if err != nil {
			b.Fatalf("Put: %v", err)
		}
		got, err := s.Resolve(ctx, ref)
		if err != nil || len(got) != len(data) {
			b.Fatalf("Resolve: %v", err)
		}
		benchSinkBytes = got
	}
}

// BenchmarkInlineSmallPayload measures the hot-path store operations for small
// payloads (<= InlineMax) that ride inline on the Ref, bypassing HTTP calls entirely.
func BenchmarkInlineSmallPayload(b *testing.B) {
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL, WithClient(srv.Client()))

	data := payload(64, 'i')
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ref, err := s.Put(ctx, data)
		if err != nil || ref.Kind != abi.RefInline {
			b.Fatalf("Put inline: %v", err)
		}
		got, err := s.Resolve(ctx, ref)
		if err != nil || len(got) != len(data) {
			b.Fatalf("Resolve inline: %v", err)
		}
		benchSinkBytes = got
	}
}

// BenchmarkContentDedupViaHead measures content-addressed deduplication:
// repeated Put of an already-resident payload (> InlineMax) verifying digest
// computation, HEAD existence probe, atomic stats updates, and zero re-upload.
func BenchmarkContentDedupViaHead(b *testing.B) {
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL, WithClient(srv.Client()))

	data := payload(4096, 'd')
	if _, err := s.Put(ctx, data); err != nil {
		b.Fatalf("seed Put: %v", err)
	}

	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ref, err := s.Put(ctx, data)
		if err != nil || ref.Kind != abi.RefBlob {
			b.Fatalf("Put dedup: %v", err)
		}
		benchSinkRef = ref
	}
}

// BenchmarkPutNewBlob measures cold ingestion of distinct CAS blobs over HTTP:
// digest calculation, HEAD existence probe (miss), HTTP PUT request serialization,
// remote upload, and status verification.
func BenchmarkPutNewBlob(b *testing.B) {
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL, WithClient(srv.Client()))

	const size = 1024
	b.SetBytes(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := benchDistinctPayload(i, size)
		ref, err := s.Put(ctx, p)
		if err != nil || ref.Kind != abi.RefBlob {
			b.Fatalf("Put new: %v", err)
		}
		benchSinkRef = ref
	}
}

// BenchmarkResolve_Blob measures resolving a remote CAS blob over HTTP:
// GET request, response streaming, and client-side sha256 checksum verification.
func BenchmarkResolve_Blob(b *testing.B) {
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL, WithClient(srv.Client()))

	data := payload(4096, 'b')
	ref, err := s.Put(ctx, data)
	if err != nil {
		b.Fatalf("seed Put: %v", err)
	}

	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := s.Resolve(ctx, ref)
		if err != nil || len(got) != len(data) {
			b.Fatalf("Resolve blob: %v", err)
		}
		benchSinkBytes = got
	}
}

// BenchmarkPageOutPageIn measures the lifecycle of context-MMU cold storage:
// unconditionally committing a small payload to the remote store via PageOut,
// and fetching and re-inlining it back via PageIn.
func BenchmarkPageOutPageIn(b *testing.B) {
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL, WithClient(srv.Client()))

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
		in, err := s.PageIn(ctx, handle)
		if err != nil || in.Kind != abi.RefInline {
			b.Fatalf("PageIn: %v", err)
		}
		benchSinkRef = in
	}
}
