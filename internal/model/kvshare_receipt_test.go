package model

import "testing"

func TestPagedKVForkMeasuredProvesZeroCloneAndCOW(t *testing.T) {
	pool := NewPagedKVPool(pagedTestCfg(), 16)
	parent := pagedFillSeq(pool, 2048)
	before := pool.PhysicalBlocks()
	child, receipt := parent.ForkMeasured(64)
	if receipt.Engine != "fak-native" || receipt.Validation != "shared-zero-clone" {
		t.Fatalf("receipt = %+v", receipt)
	}
	if receipt.ForkCloneBytes != 0 || receipt.PhysicalBytesAfterFork != receipt.PhysicalBytesBefore {
		t.Fatalf("fork cloned physical KV: %+v", receipt)
	}
	if receipt.BytesAvoided != receipt.LogicalPrefixBytes || receipt.BytesAvoidedPerAccepted <= 0 {
		t.Fatalf("reuse accounting = %+v", receipt)
	}
	if pool.PhysicalBlocks() != before {
		t.Fatalf("physical blocks after fork = %d, want %d", pool.PhysicalBlocks(), before)
	}

	// The first divergent write copies only the shared tail block, not the prefix.
	K := make([][]float32, pool.nLayers)
	V := make([][]float32, pool.nLayers)
	for i := range K {
		K[i] = make([]float32, pool.stride)
		V[i] = make([]float32, pool.stride)
	}
	child.Append(K, V)
	if got := pool.PhysicalBlocks() - before; got != 1 {
		t.Fatalf("divergent write allocated %d blocks, want one tail COW block", got)
	}
	if parent.Len() != 2048 || child.Len() != 2049 {
		t.Fatalf("COW changed parent/child lengths: %d/%d", parent.Len(), child.Len())
	}
}

func TestPagedKVForkMeasuredAccountsFragmentation(t *testing.T) {
	pool := NewPagedKVPool(pagedTestCfg(), 16)
	parent := pagedFillSeq(pool, 17)
	child, receipt := parent.ForkMeasured(1)
	defer child.Free()
	if receipt.SharedBlocks != 2 || receipt.FragmentationRatio != 15.0/17.0 {
		t.Fatalf("fragmentation receipt = %+v", receipt)
	}
}
