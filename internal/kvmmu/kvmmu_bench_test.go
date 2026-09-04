package kvmmu

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type benchKVBackend struct {
	length int
	model  string
}

func newBenchKVBackend() *benchKVBackend {
	return &benchKVBackend{model: "bench-kv-model"}
}

func (b *benchKVBackend) Len() int {
	return b.length
}

func (b *benchKVBackend) Prefill(ids []int) []float32 {
	b.length += len(ids)
	return []float32{0.1, 0.2, 0.3}
}

func (b *benchKVBackend) Evict(from, n int) int {
	if n > b.length {
		n = b.length
	}
	b.length -= n
	return n
}

func (b *benchKVBackend) ModelID() string {
	return b.model
}

func (b *benchKVBackend) StageSpan(ctx context.Context, digest string, from, n int) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: n}, nil
}

func (b *benchKVBackend) RestoreSpan(ctx context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest}, nil
}

func (b *benchKVBackend) ReScoreSpans(probe []int, spans [][2]int) ([]float64, error) {
	scores := make([]float64, len(spans))
	if len(spans) == 0 {
		return scores, nil
	}
	uniform := 1.0 / float64(len(spans))
	for i := range scores {
		scores[i] = uniform
	}
	return scores, nil
}

type benchAllowGate struct{}

func (benchAllowGate) Admit(_ context.Context, _ *abi.ToolCall, _ *abi.Result) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "bench-allow-gate"}
}

// BenchmarkKVMMUAlloc measures memory allocations and runtime latency during Context creation,
// segment registration, admission gating, TTL expiration, and segment eviction workflows.
func BenchmarkKVMMUAlloc(b *testing.B) {
	b.ReportAllocs()
	ctx := context.Background()
	promptTokens := []int{101, 102, 103, 104, 105, 106, 107, 108}
	resultTokens := []int{201, 202, 203, 204}
	resultBody := []byte(`{"status":"success","data":"payload"}`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backend := newBenchKVBackend()
		c := NewBackendWithGate(backend, benchAllowGate{})

		c.Append("system-1", "system", promptTokens)
		_, _, _ = c.AdmitResult(ctx, "tool-call-1", "search", resultTokens, resultBody)
		c.SetTTL("tool-call-1", 100)
		_ = c.Expire(200)
		_, _ = c.Quarantine("system-1")
	}
}

// BenchmarkAttentionScore measures execution throughput and allocations across attention attribution,
// attention mass aggregation, decaying EMA turn transitions, and probe candidate re-scoring.
func BenchmarkAttentionScore(b *testing.B) {
	b.ReportAllocs()
	backend := newBenchKVBackend()
	c := NewBackendWithGate(backend, benchAllowGate{})

	spanTokens := make([]int, 32)
	for j := 0; j < 32; j++ {
		spanTokens[j] = j
	}
	for s := 0; s < 8; s++ {
		c.Append(fmt.Sprintf("span-%d", s), "tool", spanTokens)
	}

	keyPositions := make([]int, 256)
	weights := make([]float32, 256)
	for k := 0; k < 256; k++ {
		keyPositions[k] = k
		weights[k] = 1.0 / 256.0
	}

	candidateIDs := []string{"span-0", "span-1", "span-2", "span-3", "span-4", "span-5", "span-6", "span-7"}
	probe := []int{10, 20, 30}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = c.AttributeRow(keyPositions, weights)
		_ = c.AttendedMass()
		c.CloseTurn(0.9)
		_, _ = c.ReScore(probe, candidateIDs)
	}
}

// BenchmarkEvictColdest measures execution throughput and allocations when evaluating
// attention scores and evicting coldest spans under budget constraints.
func BenchmarkEvictColdest(b *testing.B) {
	b.ReportAllocs()
	spanTokens := make([]int, 16)
	for j := 0; j < 16; j++ {
		spanTokens[j] = j
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backend := newBenchKVBackend()
		c := NewBackendWithGate(backend, benchAllowGate{})
		for s := 0; s < 8; s++ {
			c.Append(fmt.Sprintf("span-%d", s), "tool", spanTokens)
		}
		keyPositions := make([]int, 128)
		weights := make([]float32, 128)
		for k := 0; k < 128; k++ {
			keyPositions[k] = k
			weights[k] = float32(k) / 128.0
		}
		_ = c.AttributeRow(keyPositions, weights)
		c.CloseTurn(0.8)
		_ = c.EvictColdest(32)
		_ = c.LastRetainedMass()
	}
}

// BenchmarkSessionAttentionReport measures performance when folding historical attention
// trajectories and computing session-integrated signal-to-noise reports.
func BenchmarkSessionAttentionReport(b *testing.B) {
	b.ReportAllocs()
	acc := NewAttentionAccumulator(0.9, 64)
	spans := []string{"span-0", "span-1", "span-2", "span-3", "span-4"}
	cost := map[string]int{
		"span-0": 16,
		"span-1": 32,
		"span-2": 48,
		"span-3": 64,
		"span-4": 128,
	}

	for turn := 1; turn <= 10; turn++ {
		m := make(map[string]float64, len(spans))
		for idx, id := range spans {
			m[id] = float64(turn * (idx + 1))
		}
		acc.Observe(m)
	}

	curve := make([]TurnSN, 10)
	for t := 0; t < 10; t++ {
		curve[t] = TurnSN{
			Turn:     t + 1,
			Ratio:    0.85 - float64(t)*0.02,
			Cost:     288,
			CacheHit: 0.60 + float64(t)*0.03,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = BuildSessionAttentionReport(acc, curve, cost, 3, 50.0)
	}
}
