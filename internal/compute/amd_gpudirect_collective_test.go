package compute

import (
	"math"
	"testing"
)

func setupTestCollective(t *testing.T, ranks int) (*AMDGPUDirectCollective, *AMDGPUDirectHAL) {
	t.Helper()
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{
		EnableLargeBARCheck:    true,
		EnforceACSZeroRedirect: true,
		PreferXGMI:             true,
	})

	rankToNode := make(map[int]int, ranks)
	for r := 0; r < ranks; r++ {
		rankToNode[r] = r
		peerLinks := make([]PeerLink, 0)
		for p := 0; p < ranks; p++ {
			if p != r {
				fabric := FabricXGMI
				if (r == 0 && p == 3) || (r == 3 && p == 0) {
					// Cross-node RDMA
					fabric = FabricPCIeSwitch
				}
				peerLinks = append(peerLinks, PeerLink{
					TargetNodeID:     p,
					Fabric:           fabric,
					BandwidthGBps:    896.0,
					LatencyNanos:     210,
					DirectP2PCapable: true,
					Coherent:         fabric == FabricXGMI,
				})
			}
		}
		_ = hal.RegisterNode(AMDDeviceNode{
			NodeID:         r,
			TotalVRAMBytes: 64 * 1024 * 1024 * 1024,
			BAR1SizeBytes:  64 * 1024 * 1024 * 1024,
			IsLargeBAR:     true,
			DMABUFCapable:  true,
			Peers:          peerLinks,
		})
	}

	coll, err := NewAMDGPUDirectCollective(Pick("cpu-ref"), hal, rankToNode)
	if err != nil {
		t.Fatalf("failed to create collective: %v", err)
	}
	return coll, hal
}

func TestAMDGPUDirectCollective_AllReduceSumRankOrder(t *testing.T) {
	coll, _ := setupTestCollective(t, 4)
	const n = 16

	for _, ranks := range []int{1, 2, 4} {
		parts := make([]Tensor, ranks)
		raw := make([][]float32, ranks)
		for r := 0; r < ranks; r++ {
			raw[r] = make([]float32, n)
			for i := 0; i < n; i++ {
				raw[r][i] = float32(r*100 + i + 1)
			}
			parts[r] = NewF32(coll.Backend, []int{n}, raw[r])
		}

		// Reference sum
		want := make([]float32, n)
		for r := 0; r < ranks; r++ {
			for i := 0; i < n; i++ {
				want[i] += raw[r][i]
			}
		}

		out, err := coll.AllReduceSum(parts)
		if err != nil {
			t.Fatalf("AllReduceSum ranks=%d failed: %v", ranks, err)
		}

		got := coll.Read(out)
		if len(got) != n {
			t.Fatalf("AllReduceSum ranks=%d len=%d, want %d", ranks, len(got), n)
		}
		for i := 0; i < n; i++ {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("AllReduceSum ranks=%d i=%d got %f want %f", ranks, i, got[i], want[i])
			}
		}
	}
}

func TestAMDGPUDirectCollective_AllGatherAndReduceScatterIdentity(t *testing.T) {
	coll, _ := setupTestCollective(t, 4)
	const ranks = 4
	const n = 32 // multiple of 4

	parts := make([]Tensor, ranks)
	for r := 0; r < ranks; r++ {
		data := make([]float32, n)
		for i := 0; i < n; i++ {
			data[i] = float32(r*10 + i)
		}
		parts[r] = NewF32(coll.Backend, []int{n}, data)
	}

	// 1. Compute direct AllReduceSum
	allReduced, err := coll.AllReduceSum(parts)
	if err != nil {
		t.Fatalf("AllReduceSum failed: %v", err)
	}
	expectedReduced := coll.Read(allReduced)

	// 2. Compute ReduceScatter
	shards, err := coll.ReduceScatter(parts)
	if err != nil {
		t.Fatalf("ReduceScatter failed: %v", err)
	}
	if len(shards) != ranks {
		t.Fatalf("expected %d shards, got %d", ranks, len(shards))
	}

	// 3. Compute AllGather over shards
	gathered, err := coll.AllGather(shards)
	if err != nil {
		t.Fatalf("AllGather failed: %v", err)
	}
	gatheredReduced := coll.Read(gathered)

	// Invariant: AllReduceSum ≡ AllGather ∘ ReduceScatter bit-exact!
	if len(gatheredReduced) != len(expectedReduced) {
		t.Fatalf("length mismatch: got %d, want %d", len(gatheredReduced), len(expectedReduced))
	}
	for i := 0; i < len(expectedReduced); i++ {
		if math.Float32bits(gatheredReduced[i]) != math.Float32bits(expectedReduced[i]) {
			t.Fatalf("AllGather(ReduceScatter) != AllReduceSum at index %d: %f != %f", i, gatheredReduced[i], expectedReduced[i])
		}
	}
}

func TestAMDGPUDirectCollective_AllToAllInvolution(t *testing.T) {
	coll, _ := setupTestCollective(t, 4)
	const ranks = 4
	const n = 16 // 4 per shard

	parts := make([]Tensor, ranks)
	for r := 0; r < ranks; r++ {
		data := make([]float32, n)
		for i := 0; i < n; i++ {
			data[i] = float32(r*1000 + i)
		}
		parts[r] = NewF32(coll.Backend, []int{n}, data)
	}

	// First transpose
	transposed, err := coll.AllToAll(parts)
	if err != nil {
		t.Fatalf("first AllToAll failed: %v", err)
	}

	// Second transpose (involution should return original data)
	restored, err := coll.AllToAll(transposed)
	if err != nil {
		t.Fatalf("second AllToAll failed: %v", err)
	}

	for r := 0; r < ranks; r++ {
		orig := coll.Read(parts[r])
		rec := coll.Read(restored[r])
		for i := 0; i < n; i++ {
			if math.Float32bits(orig[i]) != math.Float32bits(rec[i]) {
				t.Fatalf("AllToAll involution failed for rank %d at index %d: got %f, want %f", r, i, rec[i], orig[i])
			}
		}
	}
}

func TestAMDGPUDirectCollective_ZeroCopyInvariant(t *testing.T) {
	coll, _ := setupTestCollective(t, 2)
	if coll.StagingCopyCount() != 0 {
		t.Errorf("expected 0 staging copies, got %d", coll.StagingCopyCount())
	}

	p0 := NewF32(coll.Backend, []int{4}, []float32{1, 2, 3, 4})
	p1 := NewF32(coll.Backend, []int{4}, []float32{5, 6, 7, 8})

	_, _ = coll.AllReduceSum([]Tensor{p0, p1})

	stats := coll.Stats()
	if stats.TotalCollectives != 1 {
		t.Errorf("expected 1 collective run, got %d", stats.TotalCollectives)
	}
	if stats.StagingCopyCount != 0 {
		t.Errorf("expected 0 staging copies, got %d", stats.StagingCopyCount)
	}
	if stats.ZeroCopyBytes == 0 {
		t.Errorf("expected >0 zero-copy bytes moved")
	}
}
