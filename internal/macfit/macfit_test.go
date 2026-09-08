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

func TestTurnkeyTierSelectionAndHeadroomGuarantee(t *testing.T) {
	cases := []struct {
		name      string
		memoryGiB uint64
		wantTier  string
		wantQuant string
	}{
		{name: "16GB MacBook Air", memoryGiB: 16, wantTier: "7B", wantQuant: "Q4_K_M"},
		{name: "24GB MacBook Pro", memoryGiB: 24, wantTier: "7B", wantQuant: "Q4_K_M"},
		{name: "36GB MacBook Pro", memoryGiB: 36, wantTier: "27B", wantQuant: "Q4_K_M"},
		{name: "48GB MacBook Pro", memoryGiB: 48, wantTier: "27B", wantQuant: "Q4_K_M"},
		{name: "64GB Mac Studio", memoryGiB: 64, wantTier: "70B", wantQuant: "Q4_K_M"},
		{name: "128GB Mac Studio", memoryGiB: 128, wantTier: "70B", wantQuant: "Q4_K_M"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			memBytes := tc.memoryGiB * GiB
			tier := SelectModelTier(memBytes)
			if tier.Name != tc.wantTier {
				t.Fatalf("tier name = %q, want %q", tier.Name, tc.wantTier)
			}
			if tier.QuantTier != tc.wantQuant {
				t.Fatalf("quant tier = %q, want %q", tier.QuantTier, tc.wantQuant)
			}

			plan, err := ConfigureTurnkey(memBytes)
			if err != nil {
				t.Fatalf("ConfigureTurnkey(%d GiB): %v", tc.memoryGiB, err)
			}
			if plan.Tier.Name != tc.wantTier {
				t.Fatalf("plan tier = %q, want %q", plan.Tier.Name, tc.wantTier)
			}
			if plan.HeadroomRatio < 0.20 {
				t.Fatalf("plan headroom ratio = %.3f, want >= 0.20 (20%% guarantee)", plan.HeadroomRatio)
			}
			if plan.ContextBudgetTokens == 0 {
				t.Fatal("plan context budget tokens must be > 0")
			}
			allocated := plan.Tier.WeightBytes + (plan.ContextBudgetTokens * plan.KVBytesPerToken)
			if allocated+plan.HeadroomBytes != memBytes {
				t.Fatalf("allocated (%d) + headroom (%d) != total memory (%d)", allocated, plan.HeadroomBytes, memBytes)
			}
			// Verify allocated memory does not exceed 80% of total memory (guaranteeing >= 20% headroom)
			maxAllocated := (memBytes * 80) / 100
			if allocated > maxAllocated {
				t.Fatalf("allocated %d bytes exceeds 80%% limit %d bytes (headroom violated)", allocated, maxAllocated)
			}
		})
	}
}

func TestDetectUnifiedMemoryOverride(t *testing.T) {
	t.Setenv("FAK_UP_MEMORY_BYTES", "38654705664") // 36 GiB
	got, err := DetectUnifiedMemory()
	if err != nil {
		t.Fatalf("DetectUnifiedMemory: %v", err)
	}
	if got != 38654705664 {
		t.Fatalf("DetectUnifiedMemory() = %d, want 38654705664", got)
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
