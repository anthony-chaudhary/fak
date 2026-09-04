package decodemigrate

import (
	"testing"
)

func newBenchmarkState(tb testing.TB, numTokens, payloadBytes int) *DecodeState {
	tb.Helper()
	tokens := make([]int64, numTokens)
	for i := range tokens {
		tokens[i] = int64(1000 + i)
	}
	meta := KVBufferMetadata{
		NumLayers:   32,
		NumHeads:    16,
		HeadDim:     128,
		BlockSize:   16,
		TotalTokens: numTokens,
	}
	payload := make([]byte, payloadBytes)
	copy(payload[:4], []byte(tagV1))

	state, err := NewDecodeState(VersionV1Legacy, "qwen-38b", "seq-bench", tokens, meta, payload)
	if err != nil {
		tb.Fatalf("NewDecodeState failed: %v", err)
	}
	return state
}

// BenchmarkDecodeMigrate measures round-trip multi-hop migration throughput in a b.N loop.
func BenchmarkDecodeMigrate(b *testing.B) {
	engine := NewMigrationRunner()
	state := newBenchmarkState(b, 512, 8192)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v4, err := engine.Migrate(state, VersionV4Compressed)
		if err != nil {
			b.Fatalf("Migrate failed: %v", err)
		}
		if _, err := engine.Migrate(v4, VersionV1Legacy); err != nil {
			b.Fatalf("Reverse migrate failed: %v", err)
		}
	}
}

// BenchmarkDecodeMigrateSingleHop measures single-hop forward migration throughput in a b.N loop.
func BenchmarkDecodeMigrateSingleHop(b *testing.B) {
	engine := NewMigrationRunner()
	state := newBenchmarkState(b, 512, 8192)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := engine.Migrate(state, VersionV2Paged); err != nil {
			b.Fatalf("Migrate failed: %v", err)
		}
	}
}

// BenchmarkDecodeMigrateRoute measures BFS migration route calculation throughput in a b.N loop.
func BenchmarkDecodeMigrateRoute(b *testing.B) {
	engine := NewMigrationRunner()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		route, err := engine.Route(VersionV1Legacy, VersionV4Compressed)
		if err != nil || route.TotalSteps() != 3 {
			b.Fatalf("Route failed: %v", err)
		}
	}
}

// BenchmarkDecodeStateValidate measures state integrity and checksum verification throughput in a b.N loop.
func BenchmarkDecodeStateValidate(b *testing.B) {
	state := newBenchmarkState(b, 512, 8192)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := state.Validate(); err != nil {
			b.Fatalf("Validate failed: %v", err)
		}
	}
}

// BenchmarkDecodeStateComputeChecksum measures SHA-256 state hashing throughput in a b.N loop.
func BenchmarkDecodeStateComputeChecksum(b *testing.B) {
	state := newBenchmarkState(b, 512, 8192)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		digest := state.ComputeChecksum()
		if digest[0] == 0 && digest[1] == 0 && digest[2] == 0 && digest[3] == 0 {
			b.Fatal("unexpected zero digest prefix")
		}
	}
}

// BenchmarkDecodeStateClone measures deep copy throughput of decode states in a b.N loop.
func BenchmarkDecodeStateClone(b *testing.B) {
	state := newBenchmarkState(b, 512, 8192)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cp := state.Clone()
		if cp == nil || len(cp.Tokens) != len(state.Tokens) {
			b.Fatal("Clone failed")
		}
	}
}

// BenchmarkDecodeMigratePayloadScales measures migration performance across varying payload sizes.
func BenchmarkDecodeMigratePayloadScales(b *testing.B) {
	engine := NewMigrationRunner()
	sizes := []struct {
		name string
		kb   int
	}{
		{"4KB", 4},
		{"64KB", 64},
		{"256KB", 256},
	}

	for _, tc := range sizes {
		b.Run(tc.name, func(b *testing.B) {
			state := newBenchmarkState(b, 512, tc.kb*1024)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				v2, err := engine.Migrate(state, VersionV2Paged)
				if err != nil {
					b.Fatalf("Migrate failed: %v", err)
				}
				if _, err := engine.Migrate(v2, VersionV1Legacy); err != nil {
					b.Fatalf("Reverse migrate failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkDecodeMigrateParallel measures concurrent migration throughput across goroutines.
func BenchmarkDecodeMigrateParallel(b *testing.B) {
	engine := NewMigrationRunner()
	state := newBenchmarkState(b, 256, 4096)

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v2, err := engine.Migrate(state, VersionV2Paged)
			if err != nil {
				b.Fatalf("Migrate failed: %v", err)
			}
			if _, err := engine.Migrate(v2, VersionV1Legacy); err != nil {
				b.Fatalf("Reverse migrate failed: %v", err)
			}
		}
	})
}

// TestBenchmarkExecution verifies that BenchmarkDecodeMigrate executes cleanly.
func TestBenchmarkExecution(t *testing.T) {
	res := testing.Benchmark(BenchmarkDecodeMigrate)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
