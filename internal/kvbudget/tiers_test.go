package kvbudget

import (
	"sync"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestTierPoolAegisInkling16GB(t *testing.T) {
	const aegis384k = 384 * 1024 // 393,216 tokens

	plan, err := AutotuneTiersFromTokenBudget(aegis384k, ProfileUltracode)
	if err != nil {
		t.Fatalf("AutotuneTiersFromTokenBudget: %v", err)
	}
	if plan.TotalTokenCapacity != aegis384k {
		t.Fatalf("TotalTokenCapacity = %d, want %d", plan.TotalTokenCapacity, aegis384k)
	}
	if len(plan.Specs) != 3 {
		t.Fatalf("len(Specs) = %d, want 3", len(plan.Specs))
	}

	// Verify exact 1x128k + 4x32k + 32x4k fit.
	want := []struct {
		name   string
		level  int
		maxCtx int
		min    int
	}{
		{TierMaster, 1, 131072, 1},
		{TierWorker, 2, 32768, 4},
		{TierMicro, 3, 4096, 32},
	}
	totalReserved := 0
	for i, w := range want {
		got := plan.Specs[i]
		if got.Name != w.name || got.Level != w.level || got.MaxContext != w.maxCtx || got.MinSlots != w.min {
			t.Fatalf("spec[%d] = %+v, want name=%s level=%d maxCtx=%d min=%d",
				i, got, w.name, w.level, w.maxCtx, w.min)
		}
		totalReserved += got.MinSlots * got.MaxContext
	}
	if totalReserved != aegis384k {
		t.Fatalf("totalReserved = %d, want exact fit %d", totalReserved, aegis384k)
	}

	pool, err := NewTierPool(plan.TotalTokenCapacity, plan.Specs)
	if err != nil {
		t.Fatalf("NewTierPool: %v", err)
	}

	// Verify Classify picks smallest tier >= tokens.
	if tier, ok := pool.Classify(2048); !ok || tier != TierMicro {
		t.Fatalf("Classify(2048) = (%q, %v), want (%q, true)", tier, ok, TierMicro)
	}
	if tier, ok := pool.Classify(16384); !ok || tier != TierWorker {
		t.Fatalf("Classify(16384) = (%q, %v), want (%q, true)", tier, ok, TierWorker)
	}
	if tier, ok := pool.Classify(65536); !ok || tier != TierMaster {
		t.Fatalf("Classify(65536) = (%q, %v), want (%q, true)", tier, ok, TierMaster)
	}
	if _, ok := pool.Classify(262144); ok {
		t.Fatal("Classify(262144) = true, want false")
	}

	// 1. Co-admission: 1x128k + 4x32k + 32x4k all fit simultaneously.
	l1, v1 := pool.Acquire(TierMaster, 100000)
	if !v1.Admitted || v1.Borrowed || v1.CapTokens != 131072 {
		t.Fatalf("L1 acquire = %+v, want admitted without borrow", v1)
	}
	var workers []TierLease
	for i := 0; i < 4; i++ {
		lw, vw := pool.Acquire(TierWorker, 24000)
		if !vw.Admitted || vw.Borrowed {
			t.Fatalf("L2 acquire %d = %+v, want admitted from floor", i, vw)
		}
		workers = append(workers, lw)
	}
	var micros []TierLease
	for i := 0; i < 32; i++ {
		lm, vm := pool.Acquire(TierMicro, 3000)
		if !vm.Admitted || vm.Borrowed {
			t.Fatalf("L3 acquire %d = %+v, want admitted from floor", i, vm)
		}
		micros = append(micros, lm)
	}

	snap := pool.Snapshot()
	if snap.FreeTokens != 0 {
		t.Fatalf("FreeTokens = %d, want 0 when full", snap.FreeTokens)
	}

	// 2. Refusal when full; release 2 micro slots (8192 tokens) and verify
	// worker refusal reports micro as available lower tier.
	if _, vFull := pool.Acquire(TierMicro, 1024); vFull.Admitted {
		t.Fatalf("expected refusal when pool is 100%% full, got %+v", vFull)
	}
	pool.Release(micros[0])
	pool.Release(micros[1])

	_, vRefused := pool.Acquire(TierWorker, 32768)
	if vRefused.Admitted {
		t.Fatalf("expected worker refusal with only 8192 tokens free")
	}
	if vRefused.AdviceCode != "TIER_EXHAUSTED_DOWNGRADE_AVAILABLE" {
		t.Fatalf("AdviceCode = %q, want TIER_EXHAUSTED_DOWNGRADE_AVAILABLE", vRefused.AdviceCode)
	}
	var microAvail int
	for _, ts := range vRefused.AvailableTiers {
		if ts.Name == TierMicro {
			microAvail = ts.AvailableNow
		}
	}
	if microAvail != 2 {
		t.Fatalf("AvailableTiers[micro].AvailableNow = %d, want 2", microAvail)
	}

	// 3. Elastic borrowing when L1 is idle: release L1 (128k) and borrow 4 extra
	// 32k worker slots from L1's idle capacity.
	pool.Release(l1)
	var borrowedWorkers []TierLease
	for i := 0; i < 4; i++ {
		lb, vb := pool.Acquire(TierWorker, 32768)
		if !vb.Admitted || !vb.Borrowed {
			t.Fatalf("elastic L2 borrow %d = %+v, want admitted+borrowed", i, vb)
		}
		borrowedWorkers = append(borrowedWorkers, lb)
	}

	// Clean up remaining leases.
	for _, lw := range workers {
		pool.Release(lw)
	}
	for _, lb := range borrowedWorkers {
		pool.Release(lb)
	}
	for _, lm := range micros[2:] {
		pool.Release(lm)
	}
	if got := pool.Snapshot().FreeTokens; got != aegis384k {
		t.Fatalf("FreeTokens after full release = %d, want %d", got, aegis384k)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAutotuneTiersEdgeAndDatacenter(t *testing.T) {
	// Edge 16GB (384k tokens), 24GB/48GB workstation budgets, and 206 GiB
	// datacenter (GLM52DSA @ F16, both raw 206 GiB and usable 165 GiB).
	tokenCases := []struct {
		name    string
		tokens  int
		profile TierWorkloadProfile
	}{
		{"edge-16gb-384k", 384 * 1024, ProfileUltracode},
		{"workstation-24gb", 768 * 1024, ProfileBalanced},
		{"workstation-48gb", 1536 * 1024, ProfileMicroBurst},
		{"small-64k", 64 * 1024, ProfileUltracode},
	}

	for _, tc := range tokenCases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := AutotuneTiersFromTokenBudget(tc.tokens, tc.profile)
			if err != nil {
				t.Fatalf("AutotuneTiersFromTokenBudget: %v", err)
			}
			assertConservation(t, plan)
			if _, err := NewTierPool(plan.TotalTokenCapacity, plan.Specs); err != nil {
				t.Fatalf("NewTierPool: %v", err)
			}
		})
	}

	dcCases := []struct {
		name        string
		usableBytes uint64
		profile     TierWorkloadProfile
	}{
		{"datacenter-206GiB-raw", uint64(FreeVRAMGiB * GiB), ProfileDatacenterMultiTenant},
		{"datacenter-165GiB-usable", uint64(UsableVRAMGiB * GiB), ProfileDatacenterMultiTenant},
	}

	for _, tc := range dcCases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := AutotuneTiers(tc.usableBytes, GLM52DSA, F16, tc.profile)
			if err != nil {
				t.Fatalf("AutotuneTiers: %v", err)
			}
			if plan.TotalTokenCapacity < 1_000_000 {
				t.Fatalf("expected >1M tokens for GLM52DSA datacenter budget, got %d", plan.TotalTokenCapacity)
			}
			assertConservation(t, plan)
			if _, err := NewTierPool(plan.TotalTokenCapacity, plan.Specs); err != nil {
				t.Fatalf("NewTierPool: %v", err)
			}
		})
	}
}

func assertConservation(t *testing.T, plan TierPlan) {
	t.Helper()
	var sum int64
	for _, sp := range plan.Specs {
		if sp.MaxContext <= 0 {
			t.Fatalf("tier %s has non-positive MaxContext %d", sp.Name, sp.MaxContext)
		}
		if sp.MinSlots < 0 {
			t.Fatalf("tier %s has negative MinSlots %d", sp.Name, sp.MinSlots)
		}
		sum += int64(sp.MinSlots) * int64(sp.MaxContext)
	}
	if sum > int64(plan.TotalTokenCapacity) {
		t.Fatalf("conservation violated: sum(MinSlots*MaxContext)=%d > TotalTokenCapacity=%d", sum, plan.TotalTokenCapacity)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestTierPoolRebalanceAndRelease(t *testing.T) {
	const budget = 384 * 1024
	plan, err := AutotuneTiersFromTokenBudget(budget, ProfileUltracode)
	if err != nil {
		t.Fatalf("AutotuneTiersFromTokenBudget: %v", err)
	}
	pool, err := NewTierPool(plan.TotalTokenCapacity, plan.Specs)
	if err != nil {
		t.Fatalf("NewTierPool: %v", err)
	}

	// Double-release must be idempotent and never leak or underflow ActiveSlots.
	lease, v := pool.Acquire(TierWorker, 32768)
	if !v.Admitted {
		t.Fatalf("Acquire failed: %+v", v)
	}
	pool.Release(lease)
	pool.Release(lease)

	snap := pool.Snapshot()
	if snap.FreeTokens != budget {
		t.Fatalf("FreeTokens = %d after double release, want %d", snap.FreeTokens, budget)
	}
	for _, ts := range snap.Tiers {
		if ts.ActiveSlots != 0 {
			t.Fatalf("tier %s ActiveSlots = %d, want 0", ts.Name, ts.ActiveSlots)
		}
	}

	// Small demand delta within hysteresis should not shift MinSlots.
	pool.Rebalance(map[string]int{
		TierMaster: 1,
		TierWorker: 4,
		TierMicro:  33, // +1 is below hysteresis
	})
	for _, ts := range pool.Snapshot().Tiers {
		if ts.Name == TierMicro && ts.MinSlots != 32 {
			t.Fatalf("micro MinSlots shifted to %d on sub-hysteresis demand", ts.MinSlots)
		}
	}

	// Bursty micro demand with idle master/worker should shift MinSlots toward micro
	// while maintaining conservation invariant.
	pool.Rebalance(map[string]int{
		TierMaster: 0,
		TierWorker: 2,
		TierMicro:  64,
	})
	snapAfter := pool.Snapshot()
	var microMin, reservedTokens int
	for _, ts := range snapAfter.Tiers {
		reservedTokens += ts.MinSlots * ts.MaxContext
		if ts.Name == TierMicro {
			microMin = ts.MinSlots
		}
	}
	if microMin <= 32 {
		t.Fatalf("expected micro MinSlots > 32 after burst rebalance, got %d", microMin)
	}
	if reservedTokens > budget {
		t.Fatalf("conservation violated after rebalance: %d > %d", reservedTokens, budget)
	}

	// Concurrency safety check under parallel acquire/release.
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				l, verdict := pool.Acquire(TierMicro, 4096)
				if verdict.Admitted {
					pool.Release(l)
				}
			}
		}()
	}
	wg.Wait()
	if got := pool.Snapshot().FreeTokens; got != budget {
		t.Fatalf("FreeTokens after concurrent acquire/release = %d, want %d", got, budget)
	}
}
