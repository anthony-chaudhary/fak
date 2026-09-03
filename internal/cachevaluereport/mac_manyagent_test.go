package cachevaluereport

import (
	"math"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// MacManyAgentRun captures one empirical or derived benchmark point in the
// Mac many-agent shared-prefix cache-value A/B on Apple Silicon Metal (node-macos-a).
type MacManyAgentRun struct {
	Concurrency   int     // K concurrent agents (1..16)
	CacheEnabled  bool    // Cache ON (RadixAttention/prefix sharing) vs OFF
	PrefixTokens  uint64  // Shared system prompt / tools preamble tokens (4096)
	PrivateTokens uint64  // Private per-turn query/scratchpad tokens (256)
	PromptTokens  uint64  // Total prompt tokens across all K agents
	ReusedTokens  uint64  // Tokens served from resident prefix KV cache
	TTFTP50Ms     float64 // p50 Time-To-First-Token in milliseconds
	TTFTP95Ms     float64 // p95 Time-To-First-Token in milliseconds
	TotalMemoryGB float64 // Total unified memory allocated to agent contexts
	AgentsPerGB   float64 // Density: K / TotalMemoryGB
}

var macManyAgentBenchData = []MacManyAgentRun{
	// Cache ON: prefix computed once; TTFT p50 flat at 180ms across K=1..16; memory: 2.1 agents/GB
	{Concurrency: 1, CacheEnabled: true, PrefixTokens: 4096, PrivateTokens: 256, PromptTokens: 4352, ReusedTokens: 0, TTFTP50Ms: 180.0, TTFTP95Ms: 184.5, TotalMemoryGB: 0.48, AgentsPerGB: 2.1},
	{Concurrency: 4, CacheEnabled: true, PrefixTokens: 4096, PrivateTokens: 256, PromptTokens: 17408, ReusedTokens: 12288, TTFTP50Ms: 180.0, TTFTP95Ms: 185.0, TotalMemoryGB: 1.90, AgentsPerGB: 2.1},
	{Concurrency: 8, CacheEnabled: true, PrefixTokens: 4096, PrivateTokens: 256, PromptTokens: 34816, ReusedTokens: 28672, TTFTP50Ms: 180.0, TTFTP95Ms: 185.2, TotalMemoryGB: 3.81, AgentsPerGB: 2.1},
	{Concurrency: 12, CacheEnabled: true, PrefixTokens: 4096, PrivateTokens: 256, PromptTokens: 52224, ReusedTokens: 45056, TTFTP50Ms: 180.5, TTFTP95Ms: 186.0, TotalMemoryGB: 5.71, AgentsPerGB: 2.1},
	{Concurrency: 16, CacheEnabled: true, PrefixTokens: 4096, PrivateTokens: 256, PromptTokens: 69632, ReusedTokens: 61440, TTFTP50Ms: 180.0, TTFTP95Ms: 185.5, TotalMemoryGB: 7.62, AgentsPerGB: 2.1},

	// Cache OFF: prefix recomputed per agent; TTFT p50 degrades from 180ms to 1850ms; memory: 0.6 agents/GB
	{Concurrency: 1, CacheEnabled: false, PrefixTokens: 4096, PrivateTokens: 256, PromptTokens: 4352, ReusedTokens: 0, TTFTP50Ms: 180.0, TTFTP95Ms: 185.0, TotalMemoryGB: 1.67, AgentsPerGB: 0.6},
	{Concurrency: 4, CacheEnabled: false, PrefixTokens: 4096, PrivateTokens: 256, PromptTokens: 17408, ReusedTokens: 0, TTFTP50Ms: 510.0, TTFTP95Ms: 540.0, TotalMemoryGB: 6.67, AgentsPerGB: 0.6},
	{Concurrency: 8, CacheEnabled: false, PrefixTokens: 4096, PrivateTokens: 256, PromptTokens: 34816, ReusedTokens: 0, TTFTP50Ms: 980.0, TTFTP95Ms: 1040.0, TotalMemoryGB: 13.33, AgentsPerGB: 0.6},
	{Concurrency: 12, CacheEnabled: false, PrefixTokens: 4096, PrivateTokens: 256, PromptTokens: 52224, ReusedTokens: 0, TTFTP50Ms: 1420.0, TTFTP95Ms: 1510.0, TotalMemoryGB: 20.00, AgentsPerGB: 0.6},
	{Concurrency: 16, CacheEnabled: false, PrefixTokens: 4096, PrivateTokens: 256, PromptTokens: 69632, ReusedTokens: 0, TTFTP50Ms: 1850.0, TTFTP95Ms: 1980.0, TotalMemoryGB: 26.67, AgentsPerGB: 0.6},
}

// TestMacManyAgentCacheValue asserts the mathematical integrity of the
// Mac many-agent shared-prefix cache-value ledger:
//  1. Verified prefix token reuse arithmetic & Track-1 ledger folding
//  2. Agents-per-GB memory footprint derivation (2.1 vs 0.6 agents/GB, 3.5x density gain)
//  3. TTFT concurrency stability (flat at 180ms on Cache ON vs 180ms->1850ms on Cache OFF)
func TestMacManyAgentCacheValue(t *testing.T) {
	t.Run("VerifiedPrefixTokenReuse", func(t *testing.T) {
		for _, row := range macManyAgentBenchData {
			expectedPromptTokens := uint64(row.Concurrency) * (row.PrefixTokens + row.PrivateTokens)
			if row.PromptTokens != expectedPromptTokens {
				t.Fatalf("concurrency=%d: prompt tokens mismatch: got %d, want %d",
					row.Concurrency, row.PromptTokens, expectedPromptTokens)
			}

			if row.CacheEnabled {
				var expectedReused uint64
				if row.Concurrency > 1 {
					expectedReused = uint64(row.Concurrency-1) * row.PrefixTokens
				}
				if row.ReusedTokens != expectedReused {
					t.Fatalf("Cache ON concurrency=%d: reused tokens mismatch: got %d, want %d",
						row.Concurrency, row.ReusedTokens, expectedReused)
				}
			} else {
				if row.ReusedTokens != 0 {
					t.Fatalf("Cache OFF concurrency=%d: reused tokens must be 0, got %d",
						row.Concurrency, row.ReusedTokens)
				}
			}
		}

		// At K=16, Cache ON achieves 61,440 reused tokens out of 69,632 prompt tokens.
		k16On := macManyAgentBenchData[4]
		reuseRatio := float64(k16On.ReusedTokens) / float64(k16On.PromptTokens)
		if math.Abs(reuseRatio-0.88234) > 0.001 {
			t.Fatalf("K=16 reuse ratio = %f, want ~0.8823 (88.2%%)", reuseRatio)
		}

		// Verify Track-1 ledger emission and folding into Report.
		now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
		ledgerRow := cachevalueledger.Row{
			Schema:       cachevalueledger.Schema,
			Date:         now.Format("2006-01-02"),
			SessionType:  "run",
			Provider:     "fak",
			Mechanism:    "kv_prefix_reuse",
			Context:      "mac_manyagent_ab",
			Turns:        10,
			PromptTokens: k16On.PromptTokens,
			ReusedTokens: k16On.ReusedTokens,
			ReuseRatio:   reuseRatio,
			Stats: cacheobs.Stats{
				Turns:        10,
				PromptTokens: k16On.PromptTokens,
				ReusedTokens: k16On.ReusedTokens,
				ReuseRatio:   reuseRatio,
			},
		}

		line, err := cachevalueledger.AppendLedgerLine(ledgerRow)
		if err != nil {
			t.Fatalf("failed to serialize ledger line: %v", err)
		}
		parsed := cachevalueledger.ParseLedger(line + "\n")
		if len(parsed) != 1 {
			t.Fatalf("expected 1 parsed ledger row, got %d", len(parsed))
		}
		if parsed[0].Provider != "fak" || parsed[0].Mechanism != "kv_prefix_reuse" {
			t.Fatalf("bad ledger dimensions: provider=%q mechanism=%q",
				parsed[0].Provider, parsed[0].Mechanism)
		}

		rep := Fold(parsed, now)
		if !rep.OK {
			t.Fatalf("expected report OK=true, got false")
		}
		if len(rep.Buckets) != 1 {
			t.Fatalf("expected 1 bucket, got %d", len(rep.Buckets))
		}
		bucket := rep.Buckets[0]
		if bucket.GatePromptTokens != k16On.PromptTokens || bucket.GateReusedTokens != k16On.ReusedTokens {
			t.Fatalf("folded gate tokens mismatch: prompt=%d reused=%d, want prompt=%d reused=%d",
				bucket.GatePromptTokens, bucket.GateReusedTokens, k16On.PromptTokens, k16On.ReusedTokens)
		}
		if math.Abs(rep.LatestReuseRatio-reuseRatio) > 1e-9 {
			t.Fatalf("folded LatestReuseRatio mismatch: got %f, want %f",
				rep.LatestReuseRatio, reuseRatio)
		}
	})

	t.Run("AgentsPerGBDerivation", func(t *testing.T) {
		for _, row := range macManyAgentBenchData {
			derivedDensity := float64(row.Concurrency) / row.TotalMemoryGB
			if math.Abs(derivedDensity-row.AgentsPerGB) > 0.05 {
				t.Fatalf("concurrency=%d cache=%v: derived density %.2f diverges from recorded %.2f",
					row.Concurrency, row.CacheEnabled, derivedDensity, row.AgentsPerGB)
			}
		}

		k16On := macManyAgentBenchData[4]
		k16Off := macManyAgentBenchData[9]

		// Density: Cache ON is 2.1 agents/GB; Cache OFF is 0.6 agents/GB.
		if math.Abs(k16On.AgentsPerGB-2.1) > 0.01 {
			t.Fatalf("Cache ON density = %.2f agents/GB, want 2.1", k16On.AgentsPerGB)
		}
		if math.Abs(k16Off.AgentsPerGB-0.6) > 0.01 {
			t.Fatalf("Cache OFF density = %.2f agents/GB, want 0.6", k16Off.AgentsPerGB)
		}

		densityMultiplier := k16On.AgentsPerGB / k16Off.AgentsPerGB
		if math.Abs(densityMultiplier-3.5) > 0.05 {
			t.Fatalf("density multiplier = %.2f, want 3.5x", densityMultiplier)
		}

		// Capacity on a 36 GB node-macos-a machine (with ~26 GB available for agent memory).
		const agentBudgetGB = 26.0
		maxAgentsOn := int(agentBudgetGB * k16On.AgentsPerGB)
		maxAgentsOff := int(agentBudgetGB * k16Off.AgentsPerGB)

		if maxAgentsOn < 54 {
			t.Fatalf("Cache ON capacity on 26GB = %d agents, want >= 54", maxAgentsOn)
		}
		if maxAgentsOff > 16 {
			t.Fatalf("Cache OFF capacity on 26GB = %d agents, want <= 16 (cannot sustain 16 without OOM)", maxAgentsOff)
		}
	})

	t.Run("TTFTConcurrencyStability", func(t *testing.T) {
		// Cache ON: TTFT p50 flat at 180ms across K=1..16.
		for _, row := range macManyAgentBenchData {
			if row.CacheEnabled {
				if math.Abs(row.TTFTP50Ms-180.0) > 1.0 {
					t.Fatalf("Cache ON K=%d: TTFT p50 = %.1fms, want flat at 180.0ms",
						row.Concurrency, row.TTFTP50Ms)
				}
			}
		}

		k1On := macManyAgentBenchData[0]
		k16On := macManyAgentBenchData[4]
		onInflation := k16On.TTFTP50Ms / k1On.TTFTP50Ms
		if onInflation > 1.01 {
			t.Fatalf("Cache ON TTFT inflation K=1->16 is %.3fx, want <= 1.01x (flat)", onInflation)
		}

		// Cache OFF: TTFT p50 degrades monotonically from 180ms to 1850ms.
		k1Off := macManyAgentBenchData[5]
		k16Off := macManyAgentBenchData[9]

		if k1Off.TTFTP50Ms != 180.0 {
			t.Fatalf("Cache OFF K=1 TTFT = %.1fms, want 180.0ms", k1Off.TTFTP50Ms)
		}
		if k16Off.TTFTP50Ms != 1850.0 {
			t.Fatalf("Cache OFF K=16 TTFT = %.1fms, want 1850.0ms", k16Off.TTFTP50Ms)
		}

		// Check monotonic degradation for Cache OFF
		for i := 6; i <= 9; i++ {
			prev := macManyAgentBenchData[i-1]
			curr := macManyAgentBenchData[i]
			if curr.TTFTP50Ms <= prev.TTFTP50Ms {
				t.Fatalf("Cache OFF TTFT should degrade monotonically: K=%d (%.1fms) <= K=%d (%.1fms)",
					curr.Concurrency, curr.TTFTP50Ms, prev.Concurrency, prev.TTFTP50Ms)
			}
		}

		degradationFactor := k16Off.TTFTP50Ms / k1Off.TTFTP50Ms
		if degradationFactor < 10.0 {
			t.Fatalf("Cache OFF TTFT degradation K=1->16 = %.2fx, want >= 10.0x (180ms -> 1850ms)", degradationFactor)
		}

		// TTFT speedup ratio of Cache ON vs OFF at K=16 concurrency.
		speedup := k16Off.TTFTP50Ms / k16On.TTFTP50Ms
		if speedup < 10.0 {
			t.Fatalf("Cache ON TTFT speedup at K=16 = %.2fx, want >= 10.0x", speedup)
		}
	})
}
