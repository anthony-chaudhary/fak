package refutil

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type testBackend struct {
	resolver abi.Resolver
}

func (b testBackend) Resolver() abi.Resolver { return b.resolver }
func (b testBackend) Caps() []abi.Capability { return nil }

type testResolver struct {
	payload []byte
	err     error
}

func (r testResolver) Resolve(ctx context.Context, ref abi.Ref) ([]byte, error) {
	if r.err != nil {
		return nil, r.err
	}
	return r.payload, nil
}

func (r testResolver) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	return abi.Ref{Kind: abi.RefBlob, Len: int64(len(b))}, nil
}

func TestBytesReturnsInlinePayload(t *testing.T) {
	want := []byte("payload")
	got := Bytes(context.Background(), abi.Ref{Kind: abi.RefInline, Inline: want})
	if string(got) != string(want) {
		t.Fatalf("Bytes(inline) = %q, want %q", got, want)
	}
}

func TestBytesFailsClosedWithoutResolver(t *testing.T) {
	if got := Bytes(context.Background(), abi.Ref{Kind: abi.RefBlob, Handle: 99}); got != nil {
		t.Fatalf("Bytes(unresolved) = %q, want nil", got)
	}
}

func TestBytesMaterializesThroughActiveResolver(t *testing.T) {
	want := []byte("external-blob-content")
	abi.RegisterRegionBackend(testBackend{resolver: testResolver{payload: want}})
	t.Cleanup(func() { abi.RegisterRegionBackend(nil) })

	got := Bytes(context.Background(), abi.Ref{Kind: abi.RefBlob, Handle: 123})
	if string(got) != string(want) {
		t.Fatalf("Bytes(resolved) = %q, want %q", got, want)
	}
}

func TestBytesFailsClosedOnResolverError(t *testing.T) {
	abi.RegisterRegionBackend(testBackend{resolver: testResolver{err: errors.New("storage error")}})
	t.Cleanup(func() { abi.RegisterRegionBackend(nil) })

	if got := Bytes(context.Background(), abi.Ref{Kind: abi.RefBlob, Handle: 123}); got != nil {
		t.Fatalf("Bytes(error) = %q, want nil", got)
	}
}

func BenchmarkBytesInline(b *testing.B) {
	ctx := context.Background()
	sizes := []int{32, 256}
	for _, size := range sizes {
		name := fmt.Sprintf("%dB", size)
		data := make([]byte, size)
		ref := abi.Ref{Kind: abi.RefInline, Inline: data, Len: int64(size)}
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res := Bytes(ctx, ref)
				if len(res) != size {
					b.Fatalf("unexpected len: got %d want %d", len(res), size)
				}
			}
		})
	}
}

func BenchmarkBytesUnresolved(b *testing.B) {
	ctx := context.Background()
	ref := abi.Ref{Kind: abi.RefBlob, Handle: 999}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res := Bytes(ctx, ref); res != nil {
			b.Fatalf("expected nil for unresolved ref, got %v", res)
		}
	}
}

func BenchmarkBytesResolved(b *testing.B) {
	ctx := context.Background()
	payload := make([]byte, 1024)
	abi.RegisterRegionBackend(testBackend{resolver: testResolver{payload: payload}})
	b.Cleanup(func() { abi.RegisterRegionBackend(nil) })

	ref := abi.Ref{Kind: abi.RefBlob, Handle: 456}
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Bytes(ctx, ref)
		if len(res) != len(payload) {
			b.Fatalf("unexpected len: got %d want %d", len(res), len(payload))
		}
	}
}

func BenchmarkBytesParallel(b *testing.B) {
	ctx := context.Background()
	payload := make([]byte, 128)
	ref := abi.Ref{Kind: abi.RefInline, Inline: payload, Len: int64(len(payload))}
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res := Bytes(ctx, ref)
			if len(res) != len(payload) {
				b.Fatalf("unexpected len: got %d want %d", len(res), len(payload))
			}
		}
	})
}

func TestBenchmarksSanity(t *testing.T) {
	if res := testing.Benchmark(BenchmarkBytesInline); res.N <= 0 {
		t.Fatalf("expected BenchmarkBytesInline iterations > 0, got %d", res.N)
	}
	if res := testing.Benchmark(BenchmarkBytesUnresolved); res.N <= 0 {
		t.Fatalf("expected BenchmarkBytesUnresolved iterations > 0, got %d", res.N)
	}
	if res := testing.Benchmark(BenchmarkBytesResolved); res.N <= 0 {
		t.Fatalf("expected BenchmarkBytesResolved iterations > 0, got %d", res.N)
	}
	if res := testing.Benchmark(BenchmarkBytesParallel); res.N <= 0 {
		t.Fatalf("expected BenchmarkBytesParallel iterations > 0, got %d", res.N)
	}
}
