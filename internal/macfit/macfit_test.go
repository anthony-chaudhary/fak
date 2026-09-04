package macfit

import "testing"

func TestQwen25SevenBQ4On36GiBWorkedExample(t *testing.T) {
	const gib = uint64(1 << 30)
	got, err := Calculate(Input{
		MemoryBytes: 36 * gib, ReserveBytes: 6 * gib, WeightBytes: 9 * gib / 2,
		ContextTokens: 32768, Layers: 28, KVHeads: 4, HeadDim: 128,
		KVBytesPerElement: 2, SharedPrefixTokens: 8192, TailCapTokens: 8192,
	})
	if err != nil {
		t.Fatal(err)
	}
	// KV/token = 2(K,V)*28*4*128*2 = 57,344 B.
	// Pool = 25.5 GiB; full 32K KV = 1.75 GiB => floor(25.5/1.75)=14.
	// Shared 8K = 0.4375 GiB once; private capped 8K = 0.4375 GiB/agent
	// => floor((25.5-0.4375)/0.4375)=57.
	if got.KVBytesPerToken != 57344 || got.OffAgentsThatFit != 14 || got.OnAgentsThatFit != 57 || got.ExtraAgents != 43 {
		t.Fatalf("worked example mismatch: %+v", got)
	}
	if got.Provenance != "modeled" || !got.CrossoverFound || got.CrossoverContextTokens != 8193 {
		t.Fatalf("missing model/crossover labeling: %+v", got)
	}
}

func TestCalculateRejectsImpossibleBudget(t *testing.T) {
	_, err := Calculate(Input{MemoryBytes: 10, ReserveBytes: 8, WeightBytes: 3, ContextTokens: 1, Layers: 1, KVHeads: 1, HeadDim: 1, KVBytesPerElement: 2, TailCapTokens: 1})
	if err == nil {
		t.Fatal("expected impossible budget refusal")
	}
}

func BenchmarkCalculate(b *testing.B) {
	const gib = uint64(1 << 30)
	in := Input{
		MemoryBytes:        36 * gib,
		ReserveBytes:       6 * gib,
		WeightBytes:        9 * gib / 2,
		ContextTokens:      32768,
		Layers:             28,
		KVHeads:            4,
		HeadDim:            128,
		KVBytesPerElement:  2,
		SharedPrefixTokens: 8192,
		TailCapTokens:      8192,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Calculate(in)
		if err != nil || res.OnAgentsThatFit == 0 {
			b.Fatalf("Calculate failed: %v", err)
		}
	}
}
