package turnbench

import (
	"context"
	"strings"
	"testing"
)

// TestFanoutDeterministic: a fixed (profile, N, subTurns, trials, seed) yields the
// identical cell — the reproducibility discipline the whole sweep rests on.
func TestFanoutDeterministic(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	a := RunFanoutCell(ctx, FanoutResearch, 8, 4, 8, 0xF1EE, cm)
	b := RunFanoutCell(ctx, FanoutResearch, 8, 4, 8, 0xF1EE, cm)
	if a.SharedSaved != b.SharedSaved || a.IsolatedSaved != b.IsolatedSaved || a.CrossUplift != b.CrossUplift {
		t.Fatalf("dedup distributions not reproducible:\n a=%+v\n b=%+v", a, b)
	}
	if a.PrefixTokensSaved != b.PrefixTokensSaved || a.TokenMultReuse != b.TokenMultReuse || a.ParallelSpeedup != b.ParallelSpeedup {
		t.Fatalf("projection not reproducible:\n a=%+v\n b=%+v", a, b)
	}
}

// TestFanoutNoShareZeroUplift: the anti-inflation control. With no shared goal reads,
// no plan, no writes, there is nothing for cross-agent dedup to serve, so the uplift
// MUST be exactly 0 at every N — a non-zero value would be a harness bug.
func TestFanoutNoShareZeroUplift(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	for _, N := range []int{1, 2, 8, 32, 64} {
		c := RunFanoutCell(ctx, FanoutNoShare, N, 4, 6, 1, cm)
		if c.CrossUplift.Min != 0 || c.CrossUplift.Max != 0 {
			t.Fatalf("no-share N=%d: cross_uplift not identically 0 (min=%d max=%d p50=%d) — fan-out inflated the count",
				N, c.CrossUplift.Min, c.CrossUplift.Max, c.CrossUplift.P50)
		}
	}
}

// TestFanoutSingleAgentNoUplift: N=1 is the budget-controlled single-agent control —
// one worker doing the whole goal (its own plan + execution). Because the ISOLATED arm
// gives each sub-agent its OWN plan warm-up, the SHARED and ISOLATED arms are
// byte-identical at N=1 (one plan, one sub-agent, one epoch), so the measured
// cross-agent uplift is exactly 0, the prefix-reuse geometry saves nothing, and the
// prefix-cache tax-clawed-back is 0 (caching a prefix with no other readers cannot help
// a lone agent). cross_uplift here is the partner of the no-share zero-uplift control.
func TestFanoutSingleAgentNoUplift(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	c := RunFanoutCell(ctx, FanoutResearch, 1, 6, 8, 7, cm)
	if c.CrossUplift.Min != 0 || c.CrossUplift.Max != 0 {
		t.Fatalf("N=1: cross_uplift not identically 0 (min=%d max=%d) — a lone agent has no sibling to share with", c.CrossUplift.Min, c.CrossUplift.Max)
	}
	if c.PrefixTokensSaved != 0 {
		t.Fatalf("N=1: prefix_tokens_saved=%d, want 0 (nothing to reuse across one agent)", c.PrefixTokensSaved)
	}
	if c.TaxClawedBack != 0 {
		t.Fatalf("N=1: tax_clawed_back=%.4f, want 0 (caching a prefix with no readers does not help one agent)", c.TaxClawedBack)
	}
	if c.DedupTokensSaved != 0 {
		t.Fatalf("N=1: dedup_tokens_saved=%d, want 0 (no siblings, no measured dedup)", c.DedupTokensSaved)
	}
}

// TestFanoutResearchPositiveUplift: the headline. A research goal fanned out to a
// real number of sub-agents deletes turns that the same sub-agents run apart cannot —
// the cross-agent dedup the fan-out structure buys is strictly positive and grows
// with N.
func TestFanoutResearchPositiveUplift(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	small := RunFanoutCell(ctx, FanoutResearch, 4, 4, 8, 3, cm)
	big := RunFanoutCell(ctx, FanoutResearch, 32, 4, 8, 3, cm)
	if small.CrossUplift.P50 <= 0 {
		t.Fatalf("N=4: cross_uplift p50=%d, want > 0 (fan-out should share the goal's reads)", small.CrossUplift.P50)
	}
	if big.CrossUplift.P50 <= small.CrossUplift.P50 {
		t.Fatalf("uplift did not grow with N: N=4 p50=%d vs N=32 p50=%d", small.CrossUplift.P50, big.CrossUplift.P50)
	}
	// shared must dominate isolated (more agents in one epoch => more tier-2 hits).
	if big.SharedSaved.P50 <= big.IsolatedSaved.P50 {
		t.Fatalf("N=32: shared p50=%d not > isolated p50=%d", big.SharedSaved.P50, big.IsolatedSaved.P50)
	}
}

// TestFanoutPrefixGeometry: the prefix-reuse saving is exact — (N−1)·PrefixTokens
// prefill the kernel does not redo because the master-goal prefix is materialized once
// and cloned (NewBatchFromPrefix). cmd/fanbench's TestPrefixReuseFanoutWitness grounds
// the clone's bit-identity in the real model.
func TestFanoutPrefixGeometry(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	for _, N := range []int{1, 2, 16, 256, 1024} {
		c := RunFanoutCell(ctx, FanoutResearch, N, 2, 2, 11, cm)
		if c.PrefixTokens != cm.PrefixTokens {
			t.Fatalf("N=%d: prefix_tokens=%d, want %d", N, c.PrefixTokens, cm.PrefixTokens)
		}
		if want := (N - 1) * cm.PrefixTokens; c.PrefixTokensSaved != want {
			t.Fatalf("N=%d: prefix_tokens_saved=%d, want (N-1)*P=%d", N, c.PrefixTokensSaved, want)
		}
	}
}

// TestFanoutSaturationAndClawback exercises the MODELED projection across the N curve
// without feeding it stochastic measured dedup. The token tax clawed back remains
// positive once siblings amortize the cached prefix, parallel speedup rises then
// saturates (the fold's coordination cost grows with N), and reuse never costs more
// than naive for N>1.
func TestFanoutSaturationAndClawback(t *testing.T) {
	cm := DefaultFanoutCostModel()
	Ns := []int{1, 2, 8, 64, 512, 1024}
	var prevTax, prevSpeedup float64
	for i, N := range Ns {
		pr := cm.project(N, 4, 0) // dedup=0: exercise the pure-modeled prefix-cache curve in isolation
		if N > 1 {
			if pr.tokenMultReuse > pr.tokenMultNaive {
				t.Fatalf("N=%d: reuse multiplier %.3f > naive %.3f (reuse must never cost more)", N, pr.tokenMultReuse, pr.tokenMultNaive)
			}
			if pr.taxClawedBack <= 0 {
				t.Fatalf("N=%d: tax_clawed_back=%.4f, want > 0", N, pr.taxClawedBack)
			}
			if pr.netTokensSaved <= 0 {
				t.Fatalf("N=%d: net_tokens_saved=%d, want > 0 (the prefix-cache lever should save for N>1)", N, pr.netTokensSaved)
			}
		}
		// At N=1 the prefix-cache lever is a small NET LOSS — you pay the 1.25× cache
		// write with no other reader to amortize it — which is the honest single-agent
		// truth (caching does not help a lone agent). Assert that, don't hide it.
		if N == 1 && pr.netTokensSaved >= 0 {
			t.Fatalf("N=1: net_tokens_saved=%d, want < 0 (cache-write overhead with no readers)", pr.netTokensSaved)
		}
		if i > 0 {
			// Pure-modeled curves (no measured noise): tax clawed back rises monotonically,
			// parallel speedup rises then saturates (fold cost grows with N) — neither falls.
			if pr.taxClawedBack < prevTax-1e-9 {
				t.Fatalf("tax clawed back fell with N (%.4f -> %.4f at N=%d)", prevTax, pr.taxClawedBack, N)
			}
			if pr.parallelSpeedup < prevSpeedup-1e-9 {
				t.Fatalf("parallel speedup fell with N (%.3f -> %.3f at N=%d)", prevSpeedup, pr.parallelSpeedup, N)
			}
		}
		prevTax, prevSpeedup = pr.taxClawedBack, pr.parallelSpeedup
	}
}

// TestFanoutSweepArtifacts: the sweep emits one cell per (subTurns, N) and a CSV with a
// header + one row per cell, in the fleetbench shape.
func TestFanoutSweepArtifacts(t *testing.T) {
	ctx := context.Background()
	cm := DefaultFanoutCostModel()
	agents := []int{1, 4, 16}
	subTurns := []int{2, 4}
	prefixes := []int{1024, 8192}
	sw := RunFanoutPrefixSweep(ctx, FanoutResearch, agents, subTurns, prefixes, 4, 5, cm, nil)
	if len(sw.Cells) != len(agents)*len(subTurns)*len(prefixes) {
		t.Fatalf("cells=%d, want %d", len(sw.Cells), len(agents)*len(subTurns)*len(prefixes))
	}
	if sw.AppVersion == "" {
		t.Fatal("fanout sweep app_version is empty")
	}
	if sw.Profile.Version != BenchmarkConceptVersion {
		t.Fatalf("fanout profile version=%q, want %q", sw.Profile.Version, BenchmarkConceptVersion)
	}
	if sw.Cost.Version != FanoutCostModelVersion {
		t.Fatalf("fanout cost model version=%q, want %q", sw.Cost.Version, FanoutCostModelVersion)
	}
	if len(sw.PrefixGrid) != len(prefixes) {
		t.Fatalf("prefix grid=%v, want %v", sw.PrefixGrid, prefixes)
	}
	seenPrefixes := map[int]bool{}
	for _, c := range sw.Cells {
		seenPrefixes[c.PrefixTokens] = true
		if c.PrefixTokensSaved != (c.Agents-1)*c.PrefixTokens {
			t.Fatalf("cell %+v: prefix_tokens_saved not tied to cell prefix", c)
		}
	}
	for _, p := range prefixes {
		if !seenPrefixes[p] {
			t.Fatalf("prefix %d missing from cells", p)
		}
	}
	if len(sw.JSON()) == 0 {
		t.Fatal("empty JSON artifact")
	}
	csv := string(sw.CSV())
	lines := strings.Count(strings.TrimRight(csv, "\n"), "\n") + 1
	if lines != len(sw.Cells)+1 { // +1 header
		t.Fatalf("CSV rows=%d, want %d (header + %d cells)", lines, len(sw.Cells)+1, len(sw.Cells))
	}
	if !strings.HasPrefix(csv, "agents,sub_turns,prefix_tokens,calls,") {
		t.Fatalf("CSV header unexpected: %q", strings.SplitN(csv, "\n", 2)[0])
	}
}
