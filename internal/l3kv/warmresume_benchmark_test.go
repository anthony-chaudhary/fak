package l3kv

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type benchSerialBackend struct{}

func (*benchSerialBackend) Len() int                { return 0 }
func (*benchSerialBackend) Prefill([]int) []float32 { return nil }
func (*benchSerialBackend) Evict(int, int) int      { return 0 }
func (*benchSerialBackend) ModelID() string         { return "bench-serial" }
func (*benchSerialBackend) StageSpan(_ context.Context, digest string, _ int, n int) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: n}, nil
}
func (*benchSerialBackend) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: 1}, nil
}

type benchBatchBackend struct{ benchSerialBackend }

func (*benchBatchBackend) StageSpans(_ context.Context, reqs []abi.KVResidencyRequest) []abi.KVResidency {
	out := make([]abi.KVResidency, len(reqs))
	for i, req := range reqs {
		out[i] = abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: req.Digest, Positions: req.Positions}
	}
	return out
}
func (*benchBatchBackend) RestoreSpans(_ context.Context, reqs []abi.KVResidencyRequest) []abi.KVResidency {
	out := make([]abi.KVResidency, len(reqs))
	for i, req := range reqs {
		out[i] = abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: req.Digest, Positions: req.Positions}
	}
	return out
}

func benchmarkSpans(n int) []Span {
	spans := make([]Span, n)
	for i := range spans {
		spans[i] = Span{Digest: fmt.Sprintf("%064x", i+1), From: i, Positions: 1}
	}
	return spans
}

func benchmarkManifest(spans []Span) PrefixManifest {
	m := PrefixManifest{TotalPositions: len(spans), Spans: make([]ManifestSpan, len(spans))}
	for i, sp := range spans {
		m.Spans[i] = ManifestSpan{Digest: sp.Digest, Positions: sp.Positions}
	}
	return m
}

func BenchmarkPersistPrefixSerialAdapter(b *testing.B) { benchmarkPersistPrefix(b, false) }
func BenchmarkPersistPrefixNativeBatch(b *testing.B)   { benchmarkPersistPrefix(b, true) }

func benchmarkPersistPrefix(b *testing.B, batch bool) {
	for _, n := range []int{1, 8, 64} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			spans := benchmarkSpans(n)
			var backend abi.KVBackend = &benchSerialBackend{}
			if batch {
				backend = &benchBatchBackend{}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := PersistPrefix(context.Background(), backend, spans); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkRestorePrefixSerialAdapter(b *testing.B) { benchmarkRestorePrefix(b, false) }
func BenchmarkRestorePrefixNativeBatch(b *testing.B)   { benchmarkRestorePrefix(b, true) }

func benchmarkRestorePrefix(b *testing.B, batch bool) {
	for _, n := range []int{1, 8, 64} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			m := benchmarkManifest(benchmarkSpans(n))
			var backend abi.KVBackend = &benchSerialBackend{}
			if batch {
				backend = &benchBatchBackend{}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = RestorePrefix(context.Background(), backend, m)
			}
		})
	}
}
