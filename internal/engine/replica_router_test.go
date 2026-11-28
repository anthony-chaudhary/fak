package engine

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=25ms lane=default

func test14B4ReplicaPlan(t *testing.T, slotsPerNode int, ctxTokens int) SingleNodePlacementPlan {
	t.Helper()
	plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
		Box: SingleNodeBoxSpec{
			GPUCount:        4,
			PerGPUVRAMBytes: 24 * bytesPerGiB,
			Interconnect:    InterconnectPCIe,
		},
		Model: SingleNodeModelSpec{
			ModelID:              "Qwen3.8-14B-Q4",
			TotalWeightBytes:     9 * bytesPerGiB,
			AttnLayers:           40,
			NumAttentionHeads:    40,
			NumKVHeads:           8,
			KVBytesPerToken:      64 * 1024,
			RuntimeOverheadBytes: 1536 * bytesPerMiB,
		},
		Workload: SingleNodeWorkloadSpec{
			ConcurrentSlots:    slotsPerNode,
			MinSlotsPerReplica: slotsPerNode / 4,
			ContextTokens:      ctxTokens,
		},
	})
	if !plan.Admitted || plan.DP != 4 || plan.TP != 1 {
		t.Fatalf("expected admitted DP=4 TP=1 plan, got %+v", plan)
	}
	return plan
}

func TestContinuousBatcherTryAdmitAndSubmitBoundedFailClosed(t *testing.T) {
	t.Parallel()

	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 2
	cfg.MaxKVCacheBytes = 10_000
	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher: %v", err)
	}
	t.Cleanup(func() { _ = cb.Close() })

	// Single request > MaxKVCacheBytes must fail closed immediately without touching uidSeq or queue.
	if _, err := cb.TryAdmit(&SubagentRequest{
		SessionID:       "too-big",
		PromptTokens:    []int{1, 2, 3, 4, 5},
		TargetTokens:    6,
		KVBytesPerToken: 1_000, // 11 * 1000 = 11000 > 10000
	}); !errors.Is(err, ErrBatcherKVOverBudget) {
		t.Fatalf("oversize TryAdmit err = %v, want ErrBatcherKVOverBudget", err)
	}
	if cb.WaitingQueueLength() != 0 {
		t.Fatalf("WaitingQueueLength = %d, want 0", cb.WaitingQueueLength())
	}

	// Fill 1 slot (4000 bytes), then verify second 7000-byte request fails KV budget.
	if _, err := cb.TryAdmit(&SubagentRequest{
		SessionID: "r1", PromptTokens: []int{1, 2}, TargetTokens: 2, KVBytesPerToken: 1_000,
	}); err != nil {
		t.Fatalf("TryAdmit(r1): %v", err)
	}
	if _, err := cb.TryAdmit(&SubagentRequest{
		SessionID: "r2", PromptTokens: []int{1, 2, 3, 4}, TargetTokens: 3, KVBytesPerToken: 1_000,
	}); !errors.Is(err, ErrBatcherKVOverBudget) {
		t.Fatalf("over-budget TryAdmit(r2) err = %v, want ErrBatcherKVOverBudget", err)
	}

	// Fill slot 2 (4000 bytes -> 8000/10000), then slot-full TryAdmit must return ErrBatcherSlotsFull.
	if _, err := cb.TryAdmit(&SubagentRequest{
		SessionID: "r3", PromptTokens: []int{1, 2}, TargetTokens: 2, KVBytesPerToken: 1_000,
	}); err != nil {
		t.Fatalf("TryAdmit(r3): %v", err)
	}
	if _, err := cb.TryAdmit(&SubagentRequest{
		SessionID: "r4", PromptTokens: []int{1}, TargetTokens: 1, KVBytesPerToken: 500,
	}); !errors.Is(err, ErrBatcherSlotsFull) {
		t.Fatalf("full slots TryAdmit(r4) err = %v, want ErrBatcherSlotsFull", err)
	}

	// SubmitBounded with maxWaiting=1 queues r4 (1000 bytes -> total 9000 <= 10000) and refuses r5.
	if _, err := cb.SubmitBounded(&SubagentRequest{
		SessionID: "r4", PromptTokens: []int{1}, TargetTokens: 1, KVBytesPerToken: 500,
	}, 1); err != nil {
		t.Fatalf("SubmitBounded(r4): %v", err)
	}
	if _, err := cb.SubmitBounded(&SubagentRequest{
		SessionID: "r5", PromptTokens: []int{1}, TargetTokens: 1, KVBytesPerToken: 500,
	}, 1); !errors.Is(err, ErrBatcherSlotsFull) {
		t.Fatalf("SubmitBounded(r5) past queue limit err = %v, want ErrBatcherSlotsFull", err)
	}
	snap := cb.LoadSnapshot()
	if snap.ActiveSlots != 2 || snap.WaitingCount != 1 || snap.CurrentKVCacheBytes != 8_000 || snap.QueuedKVCacheBytes != 1_000 {
		t.Fatalf("unexpected LoadSnapshot: %+v", snap)
	}
}

func TestMultiReplicaRouterRefusesInvalidPlans(t *testing.T) {
	t.Parallel()

	if _, err := NewMultiReplicaRouter(SingleNodePlacementPlan{Status: PlacementRefused}, MultiReplicaConfig{}); !errors.Is(err, ErrPlacementNotAdmitted) {
		t.Fatalf("refused plan err = %v, want ErrPlacementNotAdmitted", err)
	}

	tooManySlots := test14B4ReplicaPlan(t, 16, 4096)
	tooManySlots.Replicas[0].MaxSlots = MaxSlots + 1
	if _, err := NewMultiReplicaRouter(tooManySlots, MultiReplicaConfig{}); !errors.Is(err, ErrPlacementExceedsBatcherSlots) {
		t.Fatalf("excessive max_slots err = %v, want ErrPlacementExceedsBatcherSlots", err)
	}
}

func TestMultiReplicaRouterPrefixAwareRoutingBeatsRoundRobin(t *testing.T) {
	t.Parallel()

	plan := test14B4ReplicaPlan(t, 32, 4096)
	router, err := NewMultiReplicaRouter(plan, MultiReplicaConfig{
		Skew: SkewPolicy{AbsLoad: 8, RelLoad: 2.0},
	})
	if err != nil {
		t.Fatalf("NewMultiReplicaRouter: %v", err)
	}
	t.Cleanup(func() { _ = router.Close() })

	// 4 distinct 128-token system prefixes.
	makePrompt := func(prefixID int, suffix int) []int {
		p := make([]int, 132)
		for i := 0; i < 128; i++ {
			p[i] = prefixID*10_000 + i + 1
		}
		for i := 128; i < 132; i++ {
			p[i] = suffix*100 + i
		}
		return p
	}

	ctx := context.Background()
	prefixHomes := make(map[int]int)
	// Cold wave: 4 requests with 4 distinct prefixes must spread evenly across 4 replicas.
	for pID := 1; pID <= 4; pID++ {
		rec, err := router.Admit(&SubagentRequest{
			PromptTokens: makePrompt(pID, 1),
			TargetTokens: 1,
		})
		if err != nil {
			t.Fatalf("cold Admit prefix %d: %v", pID, err)
		}
		if rec.RouteReason != ReplicaRouteColdLeastLoad {
			t.Fatalf("cold Admit prefix %d reason = %s, want %s", pID, rec.RouteReason, ReplicaRouteColdLeastLoad)
		}
		prefixHomes[pID] = rec.ReplicaIndex
	}
	if len(map[int]bool{
		prefixHomes[1]: true,
		prefixHomes[2]: true,
		prefixHomes[3]: true,
		prefixHomes[4]: true,
	}) != 4 {
		t.Fatalf("cold prefixes did not spread across all 4 replicas: %v", prefixHomes)
	}

	// Drain wave so slots stay clear.
	if _, err := router.StepPhaseAll(ctx); err != nil {
		t.Fatalf("StepPhaseAll: %v", err)
	}

	// Warm workload: 20 follow-up requests across the 4 prefixes (with different SessionIDs).
	// Compare prefix-aware hit rate against cache-blind round-robin on the exact same stream.
	rrReplicas := make([]*tokenPrefixLRU, 4)
	for i := range rrReplicas {
		rrReplicas[i] = newTokenPrefixLRU(64)
		rrReplicas[i].admit(makePrompt(i+1, 1)) // mirror the cold wave
	}

	caHits, rrHits := 0, 0
	const warmRequests = 20
	for i := 0; i < warmRequests; i++ {
		pID := (i % 4) + 1
		prompt := makePrompt(pID, i+10)

		// Cache-blind round-robin baseline (shift by 1 to avoid accidental lockstep).
		rrTarget := (i + 1) % 4
		if rrReplicas[rrTarget].longestOverlap(prompt) >= 128 {
			rrHits++
		}
		rrReplicas[rrTarget].admit(prompt)

		// Prefix-aware MultiReplicaRouter.
		rec, err := router.Admit(&SubagentRequest{
			PromptTokens: prompt,
			TargetTokens: 1,
		})
		if err != nil {
			t.Fatalf("warm Admit %d: %v", i, err)
		}
		if rec.MatchedPrefixTokens >= 128 && rec.ReplicaIndex == prefixHomes[pID] {
			caHits++
		}
		if _, err := router.StepPhaseAll(ctx); err != nil {
			t.Fatalf("StepPhaseAll %d: %v", i, err)
		}
	}

	if caHits != warmRequests {
		t.Fatalf("prefix-aware hits = %d/%d, want 100%% warm prefix affinity", caHits, warmRequests)
	}
	if caHits <= rrHits {
		t.Fatalf("prefix-aware hits (%d) did not exceed round-robin baseline (%d)", caHits, rrHits)
	}
	stats := router.Stats()
	if stats.ColdRoutes != 4 || stats.PrefixHitRoutes != warmRequests {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestMultiReplicaRouterSkewFallbackAndSlotFailover(t *testing.T) {
	t.Parallel()

	// 4 replicas, MaxSlots=4 each (16 total slots), Skew triggers at gap >= 3 && >= 1.5x.
	plan := test14B4ReplicaPlan(t, 16, 4096)
	router, err := NewMultiReplicaRouter(plan, MultiReplicaConfig{
		Skew: SkewPolicy{AbsLoad: 3, RelLoad: 1.5},
	})
	if err != nil {
		t.Fatalf("NewMultiReplicaRouter: %v", err)
	}
	t.Cleanup(func() { _ = router.Close() })

	shared := []int{10, 20, 30, 40, 50, 60, 70, 80}

	// Req 0 lands cold on replica 0.
	r0, err := router.Admit(&SubagentRequest{PromptTokens: shared, TargetTokens: 4})
	if err != nil || r0.ReplicaIndex != 0 || r0.RouteReason != ReplicaRouteColdLeastLoad {
		t.Fatalf("req 0 = %+v (err=%v)", r0, err)
	}
	// Req 1 & 2 hit replica 0 (loads become 2, then 3 on replica 0 vs 0 on others).
	for k := 1; k <= 2; k++ {
		rk, err := router.Admit(&SubagentRequest{PromptTokens: shared, TargetTokens: 4})
		if err != nil || rk.ReplicaIndex != 0 || rk.RouteReason != ReplicaRoutePrefixHit {
			t.Fatalf("req %d = %+v (err=%v), want prefix_hit on replica 0", k, rk, err)
		}
	}
	// Req 3: replica 0 has load 3 vs minLoad 0 (gap=3 >= AbsLoad=3 and 3 >= 1.5*1) -> skew fallback sheds to replica 1!
	r3, err := router.Admit(&SubagentRequest{PromptTokens: shared, TargetTokens: 4})
	if err != nil || r3.ReplicaIndex == 0 || r3.RouteReason != ReplicaRouteSkewFallback {
		t.Fatalf("req 3 = %+v (err=%v), want skew_fallback off replica 0", r3, err)
	}

	// Fill remaining 12 slots (4 already used, 16 total capacity -> 12 more succeed).
	for i := 0; i < 12; i++ {
		if _, err := router.Admit(&SubagentRequest{PromptTokens: []int{1000 + i}, TargetTokens: 4}); err != nil {
			t.Fatalf("fill slot %d failed: %v", i, err)
		}
	}

	// 17th request MUST fail closed with ErrReplicaSlotsExhausted.
	if _, err := router.Admit(&SubagentRequest{PromptTokens: []int{9999}, TargetTokens: 4}); !errors.Is(err, ErrReplicaSlotsExhausted) {
		t.Fatalf("17th request err = %v, want ErrReplicaSlotsExhausted", err)
	}

	// Cancel one request and verify admission immediately recovers.
	if err := router.Cancel(r0.SessionID); err != nil {
		t.Fatalf("Cancel(%s): %v", r0.SessionID, err)
	}
	recovered, err := router.Admit(&SubagentRequest{PromptTokens: []int{8888}, TargetTokens: 2})
	if err != nil || recovered.ReplicaIndex != r0.ReplicaIndex {
		t.Fatalf("post-cancel Admit = %+v (err=%v), want replica %d", recovered, err, r0.ReplicaIndex)
	}
}

func TestMultiReplicaRouterHybridDPTPFailClosedKVAndVRAM(t *testing.T) {
	t.Parallel()

	// 8x 24 GiB NVLink with 70B Q4 -> DP=2, TP=4 ([0,1,2,3] and [4,5,6,7]).
	plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
		Box: SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 24 * bytesPerGiB, Interconnect: InterconnectNVLink},
		Model: SingleNodeModelSpec{
			ModelID:                      "Llama-70B-Q4",
			TotalWeightBytes:             42 * bytesPerGiB,
			AttnLayers:                   80,
			NumAttentionHeads:            64,
			NumKVHeads:                   8,
			KVBytesPerToken:              1024 * 1024, // 1 MiB/token
			RuntimeOverheadBytes:         2 * bytesPerGiB,
			CollectiveBufferBytesPerRank: 512 * bytesPerMiB,
		},
		Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096},
	})
	if !plan.Admitted || plan.Strategy != PlacementHybridDPTP || plan.DP != 2 || plan.TP != 4 {
		t.Fatalf("expected DP=2 TP=4 plan, got %+v", plan)
	}

	// Force tight KV cap per replica to exercise fail-closed KV admission deterministically.
	plan.Replicas[0].MaxKVCacheBytes = 10_000
	plan.Replicas[1].MaxKVCacheBytes = 10_000

	// Probe GPU 0 as tight (only 2000 bytes free -> blocks replica 0), GPU 4..7 roomy.
	backends := map[int]compute.Backend{
		0: fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: 2_000, probe: true},
		1: fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: 10 * bytesPerGiB, probe: true},
		2: fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: 10 * bytesPerGiB, probe: true},
		3: fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: 10 * bytesPerGiB, probe: true},
		4: fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: 10 * bytesPerGiB, probe: true},
		5: fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: 10 * bytesPerGiB, probe: true},
		6: fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: 10 * bytesPerGiB, probe: true},
		7: fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: 10 * bytesPerGiB, probe: true},
	}

	router, err := NewMultiReplicaRouter(plan, MultiReplicaConfig{
		DeviceBackends:     backends,
		TargetVRAMPressure: 0.99,
	})
	if err != nil {
		t.Fatalf("NewMultiReplicaRouter: %v", err)
	}
	t.Cleanup(func() { _ = router.Close() })

	// 1. Oversize KV request (> 10_000) fails closed across all replicas.
	if _, err := router.Admit(&SubagentRequest{
		PromptTokens: []int{1, 2, 3, 4, 5, 6}, TargetTokens: 5, KVBytesPerToken: 1_000,
	}); !errors.Is(err, ErrReplicaKVCapacityExceeded) {
		t.Fatalf("oversize KV err = %v, want ErrReplicaKVCapacityExceeded", err)
	}

	// 2. Request needing 3_000 KV per rank (12_000 total? Wait: 9_000 total / 4 KVShards = 2250/rank > 2000 free on GPU 0!)
	// Must skip replica 0 (GPU 0 low free VRAM) and land on TP=4 replica 1 (GPUs [4,5,6,7]).
	rec, err := router.Admit(&SubagentRequest{
		PromptTokens: []int{1, 2, 3, 4, 5}, TargetTokens: 4, KVBytesPerToken: 1_000, // 9_000 total, 2_250/rank
	})
	if err != nil {
		t.Fatalf("Admit onto replica 1: %v", err)
	}
	if rec.ReplicaIndex != 1 || rec.TPDegree != 4 || len(rec.GPUIDs) != 4 {
		t.Fatalf("expected TP=4 replica 1, got %+v", rec)
	}

	// 3. Second request (2_000 bytes) fits replica 1's KV (9_000+2_000 > 10_000) NO, it exceeds 10_000 on replica 1
	// AND replica 0 has GPU 0 with 2_000 < ceil(2000/4)=500? Wait: 2_000/4 = 500 <= 2_000 free!
	// Wait: let's test an 8_500 request which exceeds replica 1's remaining KV (1_000) AND exceeds GPU 0's 2_000 free (ceil(8500/4)=2125 > 2000)!
	if _, err := router.Admit(&SubagentRequest{
		PromptTokens: []int{1, 2, 3, 4, 5}, TargetTokens: 4, KVBytesPerToken: 1_000,
	}); err == nil {
		t.Fatal("expected over-budget refusal when replica 1 KV and replica 0 VRAM are both exceeded")
	}

	// 4. Verify live VRAM refusal directly when all replicas fail live GPU free-byte check.
	tightBackends := map[int]compute.Backend{
		0: fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: 100, probe: true},
		4: fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: 100, probe: true},
	}
	vramRouter, err := NewMultiReplicaRouter(plan, MultiReplicaConfig{
		DeviceBackends:     tightBackends,
		TargetVRAMPressure: 0.99,
	})
	if err != nil {
		t.Fatalf("NewMultiReplicaRouter(tight): %v", err)
	}
	t.Cleanup(func() { _ = vramRouter.Close() })
	if _, err := vramRouter.Admit(&SubagentRequest{
		PromptTokens: []int{1, 2}, TargetTokens: 2, KVBytesPerToken: 1_000,
	}); !errors.Is(err, ErrReplicaVRAMExceeded) {
		t.Fatalf("tight VRAM probe err = %v, want ErrReplicaVRAMExceeded", err)
	}
}

func TestSingleNodeServingEngineLifecycleIntegration(t *testing.T) {
	t.Parallel()

	plan := test14B4ReplicaPlan(t, 8, 4096)
	for i := range plan.Replicas {
		plan.Replicas[i].MaxKVCacheBytes = 10_000
	}
	eng, err := NewSingleNodeServingEngine(plan, MultiReplicaConfig{})
	if err != nil {
		t.Fatalf("NewSingleNodeServingEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	if !eng.WeightBearing() {
		t.Fatal("SingleNodeServingEngine must self-declare WeightBearing()==true")
	}
	if !abi.CapsHaveLifecycle(eng.Caps()) {
		t.Fatalf("missing lifecycle cap in %v", eng.Caps())
	}

	ctx := context.Background()
	sharedPrefix := []int{101, 102, 103, 104, 105, 106}
	call1Args, _ := json.Marshal(clusterAdmitArgs{
		batcherAdmitArgs: batcherAdmitArgs{
			PromptTokens: sharedPrefix,
			TargetTokens: 3,
		},
		KVBytesPerToken: 100,
	})
	req1, err := eng.Admit(ctx, &abi.ToolCall{
		Tool: "generate",
		Args: abi.Ref{Kind: abi.RefInline, Inline: call1Args},
	})
	if err != nil {
		t.Fatalf("Admit 1: %v", err)
	}
	tokCount := 0
	for range req1.Tokens() {
		tokCount++
	}
	res1, err := req1.Result()
	if err != nil {
		t.Fatalf("Result 1: %v", err)
	}
	if tokCount != 3 || res1.Status != abi.StatusOK {
		t.Fatalf("tokCount=%d res=%+v", tokCount, res1)
	}
	if res1.Meta["replica_id"] != "replica-0" || res1.Meta["tp_degree"] != "1" || res1.Meta["route_reason"] != string(ReplicaRouteColdLeastLoad) {
		t.Fatalf("unexpected route meta: %+v", res1.Meta)
	}

	// Second call sharing prefix via Complete() should hit replica-0 with matched_prefix_tokens=6.
	call2Args, _ := json.Marshal(clusterAdmitArgs{
		batcherAdmitArgs: batcherAdmitArgs{
			PromptTokens: append(append([]int(nil), sharedPrefix...), 201, 202),
			TargetTokens: 2,
		},
		KVBytesPerToken: 100,
	})
	res2, err := eng.Complete(ctx, &abi.ToolCall{
		Tool: "generate",
		Args: abi.Ref{Kind: abi.RefInline, Inline: call2Args},
	})
	if err != nil {
		t.Fatalf("Complete 2: %v", err)
	}
	if res2.Meta["replica_id"] != "replica-0" || res2.Meta["matched_prefix_tokens"] != "6" || res2.Meta["route_reason"] != string(ReplicaRoutePrefixHit) {
		t.Fatalf("expected prefix_hit on replica-0 with 6 matched tokens, got meta=%+v", res2.Meta)
	}

	// Over-budget call must fail closed through Admit().
	overArgs, _ := json.Marshal(clusterAdmitArgs{
		batcherAdmitArgs: batcherAdmitArgs{
			PromptTokens: sharedPrefix,
			TargetTokens: 5,
		},
		KVBytesPerToken: 1_000, // 11 * 1000 = 11000 > 10000
	})
	if _, err := eng.Admit(ctx, &abi.ToolCall{
		Tool: "generate",
		Args: abi.Ref{Kind: abi.RefInline, Inline: overArgs},
	}); !errors.Is(err, ErrReplicaKVCapacityExceeded) {
		t.Fatalf("over-budget Admit err = %v, want ErrReplicaKVCapacityExceeded", err)
	}
}

func TestRouteWorkersByPrefixWithPrefixResidencyIndex(t *testing.T) {
	t.Parallel()

	idx := NewPrefixResidencyIndex()
	rec := NewCacheEventRecorder()

	// Feed vLLM worker-0 and SGLang worker-1 via their real engine residency producers.
	vllmSrc := NewVLLMJSONKVEventSource(io.NopCloser(strings.NewReader(
		`{"ts":1.0,"worker_id":"vllm-0","model_id":"qwen","events":[{"type":"BlockStored","block_hashes":["b1","b2","b3"],"block_size":16,"medium":"GPU"}]}` + "\n",
	)))
	vllmEng := NewVLLMEngine(VLLMConfig{WorkerID: "vllm-0", Model: "qwen", Residency: idx, CacheRecorder: rec, KVEvents: vllmSrc})
	if err := vllmEng.RunKVEventSubscription(context.Background()); err != nil {
		t.Fatalf("RunKVEventSubscription: %v", err)
	}
	RecordSGLangRadixSnapshot("sglang-1", "qwen", idx, SGLangRadixSnapshot{
		WorkerID: "sglang-1",
		Resident: []SGLangResidentPrefix{{Digest: "b1", Tokens: 16}},
	})

	workers := []PrefixWorkerCandidate{
		{WorkerID: "vllm-0", ReplicaIndex: 0, TPDegree: 2, GPUIDs: []int{0, 1}, ActiveSlots: 1, MaxSlots: 8, KVBytesUsed: 1 << 30, MaxKVBytes: 16 << 30, Healthy: true},
		{WorkerID: "sglang-1", ReplicaIndex: 1, TPDegree: 2, GPUIDs: []int{2, 3}, ActiveSlots: 1, MaxSlots: 8, KVBytesUsed: 1 << 30, MaxKVBytes: 16 << 30, Healthy: true},
	}

	// Longest contiguous prefix match (3 blocks on vllm-0 vs 1 block on sglang-1).
	dec, err := RouteWorkersByPrefix(idx, workers, []string{"b1", "b2", "b3", "b4"}, 1<<30, SkewPolicy{AbsLoad: 4, RelLoad: 1.5})
	if err != nil {
		t.Fatalf("RouteWorkersByPrefix: %v", err)
	}
	if dec.WorkerID != "vllm-0" || dec.MatchedBlocks != 3 || dec.Reason != ReplicaRoutePrefixHit || dec.TPDegree != 2 {
		t.Fatalf("unexpected worker route decision: %+v", dec)
	}

	// Skew fallback when vllm-0 gets overloaded relative to sglang-1.
	workers[0].ActiveSlots = 6
	skewDec, err := RouteWorkersByPrefix(idx, workers, []string{"b1", "b2", "b3", "b4"}, 1<<30, SkewPolicy{AbsLoad: 4, RelLoad: 1.5})
	if err != nil {
		t.Fatalf("RouteWorkersByPrefix skew: %v", err)
	}
	if skewDec.WorkerID != "sglang-1" || skewDec.Reason != ReplicaRouteSkewFallback {
		t.Fatalf("expected skew fallback to sglang-1, got %+v", skewDec)
	}

	// Fail closed when all workers lack KV headroom.
	if _, err := RouteWorkersByPrefix(idx, workers, []string{"b1"}, 20<<30, SkewPolicy{}); !errors.Is(err, ErrReplicaKVCapacityExceeded) {
		t.Fatalf("oversize worker KV err = %v, want ErrReplicaKVCapacityExceeded", err)
	}
	// Fail closed when all workers are full on slots.
	for i := range workers {
		workers[i].ActiveSlots = workers[i].MaxSlots
	}
	if _, err := RouteWorkersByPrefix(idx, workers, []string{"b1"}, 1<<30, SkewPolicy{}); !errors.Is(err, ErrReplicaSlotsExhausted) {
		t.Fatalf("full worker slots err = %v, want ErrReplicaSlotsExhausted", err)
	}
}
