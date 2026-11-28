package engine

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=15ms lane=default

func TestSingleNodeTopologyVocabularyClosures(t *testing.T) {
	t.Parallel()

	for _, ic := range NodeInterconnects() {
		if !ic.Valid() {
			t.Fatalf("interconnect %q should be valid", ic)
		}
	}
	if NodeInterconnect("").Valid() || NodeInterconnect("rdma").Valid() {
		t.Fatal("unset or out-of-scope interconnect reported Valid()==true")
	}

	for _, st := range PlacementStrategies() {
		if !st.Valid() {
			t.Fatalf("strategy %q should be valid", st)
		}
	}
	if PlacementStrategy("").Valid() || PlacementStrategy("pp_ep").Valid() {
		t.Fatal("unset or out-of-scope strategy reported Valid()==true")
	}

	for _, obj := range PlacementObjectives() {
		if !obj.Valid() {
			t.Fatalf("objective %q should be valid", obj)
		}
	}
	if PlacementObjective("").Valid() || PlacementObjective("cost").Valid() {
		t.Fatal("unset or invalid objective reported Valid()==true")
	}

	for _, ps := range PlacementStatuses() {
		if !ps.Valid() {
			t.Fatalf("status %q should be valid", ps)
		}
	}
	if PlacementStatus("").Valid() || PlacementStatus("pending").Valid() {
		t.Fatal("unset or invalid status reported Valid()==true")
	}

	if PlacementRefusalNone.Refused() || !PlacementRefusalNone.Valid() {
		t.Fatal("PlacementRefusalNone must be Valid() and !Refused()")
	}
	for _, r := range PlacementRefusalReasons() {
		if !r.Valid() {
			t.Fatalf("refusal reason %q should be valid", r)
		}
		if r != PlacementRefusalNone && !r.Refused() {
			t.Fatalf("refusal reason %q must report Refused()==true", r)
		}
	}
	if PlacementRefusalReason("bogus").Valid() {
		t.Fatal("bogus refusal reason reported Valid()==true")
	}

	for _, rr := range ReplicaRouteReasons() {
		if !rr.Valid() {
			t.Fatalf("route reason %q should be valid", rr)
		}
	}
	if ReplicaRouteReason("").Valid() || ReplicaRouteReason("round_robin").Valid() {
		t.Fatal("unset or invalid route reason reported Valid()==true")
	}
}

func TestClassifySingleNodeBoxPCIeNVLinkUMAAndOutOfScope(t *testing.T) {
	t.Parallel()

	t.Run("Discrete1GPU_PCIe", func(t *testing.T) {
		t.Parallel()
		prof, reason, detail := ClassifySingleNodeBox(SingleNodeBoxSpec{
			GPUCount:        1,
			PerGPUVRAMBytes: 16 * bytesPerGiB,
		})
		if reason.Refused() {
			t.Fatalf("unexpected refusal %s: %s", reason, detail)
		}
		if prof.Interconnect != InterconnectPCIe || prof.GPUCount != 1 || prof.P2PCapable || prof.UnifiedMemory {
			t.Fatalf("unexpected 1-GPU profile: %+v", prof)
		}
	})

	t.Run("MultiGPU_PCIe_4x24GiB", func(t *testing.T) {
		t.Parallel()
		prof, reason, detail := ClassifySingleNodeBox(SingleNodeBoxSpec{
			GPUCount:        4,
			PerGPUVRAMBytes: 24 * bytesPerGiB,
		})
		if reason.Refused() {
			t.Fatalf("unexpected refusal %s: %s", reason, detail)
		}
		if prof.Interconnect != InterconnectPCIe || !prof.P2PCapable || !prof.GenericLinkRates {
			t.Fatalf("unexpected 4xPCIe profile: %+v", prof)
		}
	})

	t.Run("MultiGPU_NVLink_InferredFromLinkBandwidth", func(t *testing.T) {
		t.Parallel()
		prof, reason, detail := ClassifySingleNodeBox(SingleNodeBoxSpec{
			GPUCount:         8,
			PerGPUVRAMBytes:  80 * bytesPerGiB,
			LinkBandwidthGBs: 450,
		})
		if reason.Refused() {
			t.Fatalf("unexpected refusal %s: %s", reason, detail)
		}
		if prof.Interconnect != InterconnectNVLink || !prof.P2PCapable || prof.TotalVRAMBytes != 640*bytesPerGiB {
			t.Fatalf("unexpected 8xNVLink profile: %+v", prof)
		}
	})

	t.Run("SingleNode_UMA_StrixHalo128GiB", func(t *testing.T) {
		t.Parallel()
		prof, reason, detail := ClassifySingleNodeBox(SingleNodeBoxSpec{
			GPUCount:        1,
			PerGPUVRAMBytes: 128 * bytesPerGiB,
			UnifiedMemory:   true,
			HostDRAMBytes:   120 * bytesPerGiB,
		})
		if reason.Refused() {
			t.Fatalf("unexpected refusal %s: %s", reason, detail)
		}
		if prof.Interconnect != InterconnectUMA || !prof.UnifiedMemory || prof.P2PCapable {
			t.Fatalf("unexpected UMA profile: %+v", prof)
		}
		if prof.UsableVRAMBytes[0] != 120*bytesPerGiB {
			t.Fatalf("UMA usable bytes = %d, want host-capped %d", prof.UsableVRAMBytes[0], 120*bytesPerGiB)
		}
	})

	outOfScope := []struct {
		name string
		spec SingleNodeBoxSpec
		want PlacementRefusalReason
	}{
		{"multi_node", SingleNodeBoxSpec{NodeCount: 2, GPUCount: 8, PerGPUVRAMBytes: 80 * bytesPerGiB}, PlacementRefusalMultiNodeOutOfScope},
		{"cross_node_rdma", SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 80 * bytesPerGiB, CrossNodeFabric: "rdma"}, PlacementRefusalMultiNodeOutOfScope},
		{"cross_node_ib", SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 80 * bytesPerGiB, CrossNodeFabric: "infiniband"}, PlacementRefusalMultiNodeOutOfScope},
		{"gb200_nvl72", SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 141 * bytesPerGiB, CrossNodeFabric: "gb200-nvl72"}, PlacementRefusalMultiNodeOutOfScope},
		{"zero_gpus", SingleNodeBoxSpec{GPUCount: 0, PerGPUVRAMBytes: 24 * bytesPerGiB}, PlacementRefusalGPUCountOutOfRange},
		{"nine_gpus", SingleNodeBoxSpec{GPUCount: 9, PerGPUVRAMBytes: 80 * bytesPerGiB}, PlacementRefusalGPUCountOutOfRange},
		{"mlperf_288_gpus", SingleNodeBoxSpec{GPUCount: 288, PerGPUVRAMBytes: 80 * bytesPerGiB}, PlacementRefusalGPUCountOutOfRange},
		{"vram_unknown", SingleNodeBoxSpec{GPUCount: 4, PerGPUVRAMBytes: 0}, PlacementRefusalVRAMUnknown},
		{"vram_below_12gib", SingleNodeBoxSpec{GPUCount: 2, PerGPUVRAMBytes: 8 * bytesPerGiB}, PlacementRefusalVRAMOutOfRange},
		{"vram_above_141gib", SingleNodeBoxSpec{GPUCount: 4, PerGPUVRAMBytes: 192 * bytesPerGiB}, PlacementRefusalVRAMOutOfRange},
		{"uma_multi_gpu_contradiction", SingleNodeBoxSpec{GPUCount: 2, PerGPUVRAMBytes: 64 * bytesPerGiB, UnifiedMemory: true}, PlacementRefusalTopologyMismatch},
		{"nvlink_1gpu_contradiction", SingleNodeBoxSpec{GPUCount: 1, PerGPUVRAMBytes: 80 * bytesPerGiB, Interconnect: InterconnectNVLink}, PlacementRefusalTopologyMismatch},
		{"pcie_nvlink_contradiction", SingleNodeBoxSpec{GPUCount: 4, PerGPUVRAMBytes: 80 * bytesPerGiB, Interconnect: InterconnectPCIe, NVLinkActive: true}, PlacementRefusalTopologyMismatch},
	}
	for _, tc := range outOfScope {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, reason, _ := ClassifySingleNodeBox(tc.spec)
			if reason != tc.want {
				t.Fatalf("got refusal %q, want %q", reason, tc.want)
			}
		})
	}
}

func TestPlanSingleNodePlacement1To8GPUsDPvsTP(t *testing.T) {
	t.Parallel()

	model14BQ4 := SingleNodeModelSpec{
		ModelID:              "Qwen3.8-14B-Q4",
		TotalWeightBytes:     9 * bytesPerGiB,
		ActiveParams:         14_700_000_000,
		HiddenSize:           5120,
		AttnLayers:           40,
		NumAttentionHeads:    40,
		NumKVHeads:           8,
		KVBytesPerToken:      64 * 1024,
		RuntimeOverheadBytes: 1536 * bytesPerMiB,
	}
	model32BQ4 := SingleNodeModelSpec{
		ModelID:                      "Qwen3.8-32B-Q4",
		TotalWeightBytes:             20 * bytesPerGiB,
		ActiveParams:                 32_000_000_000,
		HiddenSize:                   5120,
		AttnLayers:                   64,
		NumAttentionHeads:            40,
		NumKVHeads:                   8,
		KVBytesPerToken:              128 * 1024,
		RuntimeOverheadBytes:         2 * bytesPerGiB,
		CollectiveBufferBytesPerRank: 256 * bytesPerMiB,
	}
	model70BQ4 := SingleNodeModelSpec{
		ModelID:                      "Llama-70B-Q4",
		TotalWeightBytes:             42 * bytesPerGiB,
		ActiveParams:                 70_000_000_000,
		HiddenSize:                   8192,
		AttnLayers:                   80,
		NumAttentionHeads:            64,
		NumKVHeads:                   8,
		KVBytesPerToken:              64 * 1024,
		RuntimeOverheadBytes:         2 * bytesPerGiB,
		CollectiveBufferBytesPerRank: 512 * bytesPerMiB,
	}
	model235BFP8 := SingleNodeModelSpec{
		ModelID:                      "Qwen3-235B-FP8",
		TotalWeightBytes:             235 * bytesPerGiB,
		ActiveParams:                 22_000_000_000,
		HiddenSize:                   4096,
		AttnLayers:                   94,
		NumAttentionHeads:            64,
		NumKVHeads:                   4,
		KVBytesPerToken:              256 * 1024,
		RuntimeOverheadBytes:         4 * bytesPerGiB,
		CollectiveBufferBytesPerRank: 512 * bytesPerMiB,
	}

	t.Run("1x16GiB_PCIe_14B_SingleReplica", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 1, PerGPUVRAMBytes: 16 * bytesPerGiB},
			Model:    model14BQ4,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096},
		})
		if !plan.Admitted || plan.Strategy != PlacementDataParallel || plan.DP != 1 || plan.TP != 1 {
			t.Fatalf("unexpected plan: %+v", plan)
		}
		if len(plan.Replicas) != 1 || !reflect.DeepEqual(plan.Replicas[0].GPUIDs, []int{0}) {
			t.Fatalf("replicas = %+v", plan.Replicas)
		}
	})

	t.Run("4x24GiB_PCIe_14B_DP4Replicas", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 4, PerGPUVRAMBytes: 24 * bytesPerGiB, Interconnect: InterconnectPCIe},
			Model:    model14BQ4,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 16, ContextTokens: 8192},
		})
		if !plan.Admitted || plan.Strategy != PlacementDataParallel || plan.DP != 4 || plan.TP != 1 {
			t.Fatalf("expected DP=4 TP=1, got status=%s refusal=%s dp=%d tp=%d", plan.Status, plan.Refusal, plan.DP, plan.TP)
		}
		if len(plan.Replicas) != 4 {
			t.Fatalf("replicas = %d, want 4", len(plan.Replicas))
		}
		for i, rp := range plan.Replicas {
			if !reflect.DeepEqual(rp.GPUIDs, []int{i}) || rp.MaxSlots != 4 {
				t.Fatalf("replica %d = %+v, want GPU [%d] MaxSlots=4", i, rp, i)
			}
		}
	})

	t.Run("8x24GiB_NVLink_14B_DP8Replicas", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 24 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model:    model14BQ4,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 32, ContextTokens: 8192},
		})
		if !plan.Admitted || plan.Strategy != PlacementDataParallel || plan.DP != 8 || plan.TP != 1 {
			t.Fatalf("expected DP=8 TP=1, got %+v", plan)
		}
	})

	t.Run("8x24GiB_NVLink_32B_HighKV_HybridDP4xTP2", func(t *testing.T) {
		t.Parallel()
		// 1 replica needs 20G weights + 2G overhead + (4 slots * 8192 * 128KiB = 4G KV) = 26G > 24G -> TP=1 fails.
		// TP=2 gives DP=4, 8 slots/replica -> 10G weights + 2.25G overhead + 4G KV/rank = 16.25G <= 24G.
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 24 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model:    model32BQ4,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 32, ContextTokens: 8192},
		})
		if !plan.Admitted || plan.Strategy != PlacementHybridDPTP || plan.DP != 4 || plan.TP != 2 {
			t.Fatalf("expected Hybrid DP=4 TP=2, got status=%s refusal=%s detail=%s dp=%d tp=%d", plan.Status, plan.Refusal, plan.RefusalDetail, plan.DP, plan.TP)
		}
		if len(plan.Replicas) != 4 || plan.Replicas[0].TPPlan == nil || len(plan.Replicas[0].TPPlan.Shards) != 2 {
			t.Fatalf("expected 4 TP=2 replicas with valid TPPlan, got %+v", plan.Replicas)
		}
		if !reflect.DeepEqual(plan.Replicas[1].GPUIDs, []int{2, 3}) {
			t.Fatalf("replica 1 GPUs = %v, want [2, 3]", plan.Replicas[1].GPUIDs)
		}
	})

	t.Run("4x24GiB_NVLink_70B_IntraNodeTP4", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 4, PerGPUVRAMBytes: 24 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model:    model70BQ4,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 8, ContextTokens: 8192},
		})
		if !plan.Admitted || plan.Strategy != PlacementTensorParallel || plan.DP != 1 || plan.TP != 4 {
			t.Fatalf("expected TP=4, got status=%s refusal=%s dp=%d tp=%d", plan.Status, plan.Refusal, plan.DP, plan.TP)
		}
		if !reflect.DeepEqual(plan.Replicas[0].GPUIDs, []int{0, 1, 2, 3}) {
			t.Fatalf("GPUs = %v, want [0 1 2 3]", plan.Replicas[0].GPUIDs)
		}
	})

	t.Run("8x24GiB_NVLink_70B_HybridDP2xTP4_vs_LatencyTP8", func(t *testing.T) {
		t.Parallel()
		box := SingleNodeBoxSpec{
			GPUCount:           8,
			PerGPUVRAMBytes:    24 * bytesPerGiB,
			Interconnect:       InterconnectNVLink,
			MemoryBandwidthGBs: 1000,
			ComputeTFLOPs:      300,
		}
		tpPlan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      box,
			Model:    model70BQ4,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 16, ContextTokens: 8192, Objective: ObjectiveThroughput},
		})
		if !tpPlan.Admitted || tpPlan.Strategy != PlacementHybridDPTP || tpPlan.DP != 2 || tpPlan.TP != 4 {
			t.Fatalf("throughput objective: expected DP=2 TP=4, got %s dp=%d tp=%d (%s)", tpPlan.Strategy, tpPlan.DP, tpPlan.TP, tpPlan.RefusalDetail)
		}

		latPlan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      box,
			Model:    model70BQ4,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 1, ContextTokens: 4096, Objective: ObjectiveLatency},
		})
		if !latPlan.Admitted || latPlan.Strategy != PlacementTensorParallel || latPlan.DP != 1 || latPlan.TP != 8 {
			t.Fatalf("latency objective on NVLink: expected DP=1 TP=8, got %s dp=%d tp=%d (%s)", latPlan.Strategy, latPlan.DP, latPlan.TP, latPlan.RefusalDetail)
		}
	})

	t.Run("8x80GiB_and_8x141GiB_235B_HybridAndTP8_GQAKVSharding", func(t *testing.T) {
		t.Parallel()
		// 8x80 GiB with 16 slots * 8192 * 256KiB = 32 GiB KV at DP=1, or 16 GiB KV at DP=2.
		// At TP=4: weights=58.75 GiB, overhead=4.5 GiB, NumKVHeads=4 -> kvShards=4 -> KV/rank = 4 GiB -> 67.25 GiB <= 80 GiB -> DP=2, TP=4.
		plan80 := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 80 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model:    model235BFP8,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 16, ContextTokens: 8192},
		})
		if !plan80.Admitted || plan80.DP != 2 || plan80.TP != 4 {
			t.Fatalf("8x80GiB 235B: expected DP=2 TP=4, got status=%s refusal=%s dp=%d tp=%d", plan80.Status, plan80.Refusal, plan80.DP, plan80.TP)
		}

		// With 32768 context and 20 slots, TP=4 needs 58.75 + 4.5 + (10*8GiB/4 = 20GiB) = 83.25 GiB > 80 GiB -> steps to TP=8.
		// At TP=8 with NumKVHeads=4: KVShards MUST clamp to 4 (not 8!), yielding 29.375 + 40 + 4.5 = 73.875 GiB <= 80 GiB.
		plan80Heavy := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 80 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model:    model235BFP8,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 20, ContextTokens: 32768},
		})
		if !plan80Heavy.Admitted || plan80Heavy.DP != 1 || plan80Heavy.TP != 8 {
			t.Fatalf("8x80GiB 235B heavy KV: expected DP=1 TP=8, got status=%s refusal=%s dp=%d tp=%d", plan80Heavy.Status, plan80Heavy.Refusal, plan80Heavy.DP, plan80Heavy.TP)
		}
		if plan80Heavy.Replicas[0].KVShards != 4 {
			t.Fatalf("GQA NumKVHeads=4 at TP=8 must clamp KVShards to 4, got %d", plan80Heavy.Replicas[0].KVShards)
		}

		// With 32 slots at 32768 context, GQA NumKVHeads=4 requires 64 GiB KV/rank (+ 29.375 + 4.5 = 97.875 GiB > 80 GiB),
		// so fail-closed VRAM check MUST refuse instead of dividing KV by 8.
		plan80GQAOver := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 80 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model:    model235BFP8,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 32, ContextTokens: 32768},
		})
		if plan80GQAOver.Admitted || plan80GQAOver.Refusal != PlacementRefusalInsufficientVRAM {
			t.Fatalf("8x80GiB 235B 32-slot GQA-4 KV over budget must fail closed, got %+v", plan80GQAOver)
		}

		// On 8x141 GiB (H200), TP=2 fits (117.5 + 4.5 + 8 = 130 <= 141) -> Hybrid DP=4, TP=2.
		plan141 := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 141 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model:    model235BFP8,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 32, ContextTokens: 8192},
		})
		if !plan141.Admitted || plan141.Strategy != PlacementHybridDPTP || plan141.DP != 4 || plan141.TP != 2 {
			t.Fatalf("8x141GiB 235B: expected DP=4 TP=2, got status=%s refusal=%s dp=%d tp=%d", plan141.Status, plan141.Refusal, plan141.DP, plan141.TP)
		}
	})

	t.Run("1x128GiB_UMA_70B_SingleUnifiedPool", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 1, PerGPUVRAMBytes: 128 * bytesPerGiB, UnifiedMemory: true},
			Model:    model70BQ4,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 8, ContextTokens: 8192},
		})
		if !plan.Admitted || plan.Box.Interconnect != InterconnectUMA || plan.DP != 1 || plan.TP != 1 {
			t.Fatalf("expected 1x UMA DP=1 TP=1, got %+v", plan)
		}
	})

	t.Run("HeterogeneousFreeVRAM_SkipsBusyGPU", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box: SingleNodeBoxSpec{
				Interconnect: InterconnectPCIe,
				GPUs: []SingleNodeDevice{
					{Index: 0, TotalVRAMBytes: 24 * bytesPerGiB, FreeVRAMBytes: 24 * bytesPerGiB, FreeVRAMKnown: true},
					{Index: 1, TotalVRAMBytes: 24 * bytesPerGiB, FreeVRAMBytes: 6 * bytesPerGiB, FreeVRAMKnown: true},
					{Index: 2, TotalVRAMBytes: 24 * bytesPerGiB, FreeVRAMBytes: 24 * bytesPerGiB, FreeVRAMKnown: true},
					{Index: 3, TotalVRAMBytes: 24 * bytesPerGiB, FreeVRAMBytes: 24 * bytesPerGiB, FreeVRAMKnown: true},
				},
			},
			Model:    model14BQ4,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 9, ContextTokens: 4096},
		})
		if !plan.Admitted || plan.DP != 3 || plan.TP != 1 {
			t.Fatalf("expected DP=3 skipping busy GPU 1, got %+v", plan)
		}
		if !reflect.DeepEqual(plan.UnusedGPUIDs, []int{1}) {
			t.Fatalf("UnusedGPUIDs = %v, want [1]", plan.UnusedGPUIDs)
		}
	})
}

func TestPlanSingleNodePlacementFailClosedRefusals(t *testing.T) {
	t.Parallel()

	baseModel := SingleNodeModelSpec{
		TotalWeightBytes:     42 * bytesPerGiB,
		ActiveParams:         70_000_000_000,
		HiddenSize:           8192,
		AttnLayers:           80,
		NumAttentionHeads:    64,
		NumKVHeads:           8,
		KVBytesPerToken:      64 * 1024,
		RuntimeOverheadBytes: 2 * bytesPerGiB,
	}

	t.Run("InsufficientVRAM_235B_On_4x24GiB", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box: SingleNodeBoxSpec{GPUCount: 4, PerGPUVRAMBytes: 24 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model: SingleNodeModelSpec{
				TotalWeightBytes:     235 * bytesPerGiB,
				KVBytesPerToken:      256 * 1024,
				RuntimeOverheadBytes: 4 * bytesPerGiB,
			},
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096},
		})
		if plan.Admitted || plan.Status != PlacementRefused || plan.Refusal != PlacementRefusalInsufficientVRAM || len(plan.Replicas) != 0 {
			t.Fatalf("expected fail-closed InsufficientVRAM with nil Replicas, got %+v", plan)
		}
	})

	t.Run("OverheadUnbudgeted", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box: SingleNodeBoxSpec{GPUCount: 4, PerGPUVRAMBytes: 24 * bytesPerGiB},
			Model: SingleNodeModelSpec{
				TotalWeightBytes: 9 * bytesPerGiB,
				KVBytesPerToken:  64 * 1024,
			},
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096},
		})
		if plan.Admitted || plan.Refusal != PlacementRefusalOverheadUnbudgeted {
			t.Fatalf("got %s (%v), want %s", plan.Refusal, plan.Admitted, PlacementRefusalOverheadUnbudgeted)
		}
	})

	t.Run("KVGeometryUnknown", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box: SingleNodeBoxSpec{GPUCount: 1, PerGPUVRAMBytes: 24 * bytesPerGiB},
			Model: SingleNodeModelSpec{
				TotalWeightBytes:     9 * bytesPerGiB,
				RuntimeOverheadBytes: 1 * bytesPerGiB,
			},
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096},
		})
		if plan.Admitted || plan.Refusal != PlacementRefusalKVGeometryUnknown {
			t.Fatalf("got %s, want %s", plan.Refusal, PlacementRefusalKVGeometryUnknown)
		}
	})

	t.Run("ConcurrentSlotsExceedReplicaCapacity", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 4, PerGPUVRAMBytes: 24 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model:    baseModel,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 48, MaxSlotsPerReplica: 32, ContextTokens: 2048},
		})
		if plan.Admitted || plan.Refusal != PlacementRefusalSlotsExceedReplicaCapacity {
			t.Fatalf("got %s (%s), want %s", plan.Refusal, plan.RefusalDetail, PlacementRefusalSlotsExceedReplicaCapacity)
		}
	})

	t.Run("PCIeNoP2P_RefusesTP", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box: SingleNodeBoxSpec{
				GPUCount:        4,
				PerGPUVRAMBytes: 24 * bytesPerGiB,
				Interconnect:    InterconnectPCIe,
				P2PUnavailable:  true,
			},
			Model:    baseModel,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096},
		})
		if plan.Admitted || plan.Refusal != PlacementRefusalP2PUnavailable {
			t.Fatalf("got %s, want %s", plan.Refusal, PlacementRefusalP2PUnavailable)
		}
	})

	t.Run("RequireNVLinkForTP_RefusesPCIeTP", func(t *testing.T) {
		t.Parallel()
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box: SingleNodeBoxSpec{
				GPUCount:           4,
				PerGPUVRAMBytes:    24 * bytesPerGiB,
				Interconnect:       InterconnectPCIe,
				RequireNVLinkForTP: true,
			},
			Model:    baseModel,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096},
		})
		if plan.Admitted || plan.Refusal != PlacementRefusalTPRequiresNVLink {
			t.Fatalf("got %s, want %s", plan.Refusal, PlacementRefusalTPRequiresNVLink)
		}
	})

	t.Run("ModelNotDivisibleByTP", func(t *testing.T) {
		t.Parallel()
		oddHeads := baseModel
		oddHeads.NumAttentionHeads = 28
		oddHeads.NumKVHeads = 4
		oddHeads.HiddenSize = 3584
		plan := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 8, PerGPUVRAMBytes: 24 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model:    oddHeads,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096},
			PinnedTP: 8,
		})
		if plan.Admitted || plan.Refusal != PlacementRefusalModelNotDivisibleByTP {
			t.Fatalf("got %s, want %s", plan.Refusal, PlacementRefusalModelNotDivisibleByTP)
		}
	})

	t.Run("SLORateUnknownAndSLOViolated", func(t *testing.T) {
		t.Parallel()
		missingBW := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box:      SingleNodeBoxSpec{GPUCount: 4, PerGPUVRAMBytes: 24 * bytesPerGiB, Interconnect: InterconnectNVLink},
			Model:    baseModel,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096, MaxTPOTMS: 20},
		})
		if missingBW.Admitted || missingBW.Refusal != PlacementRefusalRateUnknownForSLO {
			t.Fatalf("missing BW for SLO: got %s, want %s", missingBW.Refusal, PlacementRefusalRateUnknownForSLO)
		}

		// PCIe TP=4 with tight TPOT SLO=15ms violates due to PCIe AllReduce latency across 80 layers.
		slowPCIe := PlanSingleNodePlacement(SingleNodePlacementRequest{
			Box: SingleNodeBoxSpec{
				GPUCount:           4,
				PerGPUVRAMBytes:    24 * bytesPerGiB,
				Interconnect:       InterconnectPCIe,
				MemoryBandwidthGBs: 900,
			},
			Model:    baseModel,
			Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096, MaxTPOTMS: 12.0},
		})
		if slowPCIe.Admitted || slowPCIe.Refusal != PlacementRefusalSLOViolated {
			t.Fatalf("slow PCIe TP SLO: got %s (%s), want %s", slowPCIe.Refusal, slowPCIe.RefusalDetail, PlacementRefusalSLOViolated)
		}
	})
}

func TestProbeSingleNodeBoxIntegration(t *testing.T) {
	t.Parallel()

	// Non-probing reference backend must fail closed with VRAMUnknown.
	unknownSpec := ProbeSingleNodeBox([]compute.Backend{compute.Default(), compute.Default()}, nil, SingleNodeBoxSpec{
		Interconnect: InterconnectPCIe,
	})
	refused := PlanSingleNodePlacement(SingleNodePlacementRequest{
		Box: unknownSpec,
		Model: SingleNodeModelSpec{
			TotalWeightBytes:     9 * bytesPerGiB,
			KVBytesPerToken:      64 * 1024,
			RuntimeOverheadBytes: 1 * bytesPerGiB,
		},
		Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 4, ContextTokens: 4096},
	})
	if refused.Admitted || refused.Refusal != PlacementRefusalVRAMUnknown {
		t.Fatalf("non-probing backend must fail closed with VRAMUnknown, got %+v", refused)
	}

	// Probing backends (4x 24 GiB with 2 GiB already resident) classify and admit DP=4.
	b := fakeCapBackend{Backend: compute.Default(), total: 24 * bytesPerGiB, free: compute.FreeUnknown, probe: true}
	probed := ProbeSingleNodeBox(
		[]compute.Backend{b, b, b, b},
		[]int64{2 * bytesPerGiB, 2 * bytesPerGiB, 2 * bytesPerGiB, 2 * bytesPerGiB},
		SingleNodeBoxSpec{Interconnect: InterconnectPCIe},
	)
	admitted := PlanSingleNodePlacement(SingleNodePlacementRequest{
		Box: probed,
		Model: SingleNodeModelSpec{
			TotalWeightBytes:     9 * bytesPerGiB,
			KVBytesPerToken:      64 * 1024,
			RuntimeOverheadBytes: 1 * bytesPerGiB,
		},
		Workload: SingleNodeWorkloadSpec{ConcurrentSlots: 8, ContextTokens: 4096},
	})
	if !admitted.Admitted || admitted.DP != 4 || admitted.TP != 1 {
		t.Fatalf("probed 4x24GiB box: expected DP=4 TP=1, got %+v", admitted)
	}
	if admitted.Replicas[0].UsableMinRankBytes != 22*bytesPerGiB {
		t.Fatalf("usable VRAM after 2 GiB resident = %d, want 22 GiB", admitted.Replicas[0].UsableMinRankBytes)
	}
}
