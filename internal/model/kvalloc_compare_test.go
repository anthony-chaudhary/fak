package model

import "testing"

func TestCompareKVAllocationCosts(t *testing.T) {
	const tokens, admitted, rowBytes, blockTokens = 1000, 1024, 4096, 16
	contiguous, err := CompareKVAllocation(KVAllocationContiguous, tokens, admitted, rowBytes, blockTokens)
	if err != nil {
		t.Fatal(err)
	}
	paged, err := CompareKVAllocation(KVAllocationPaged, tokens, admitted, rowBytes, blockTokens)
	if err != nil {
		t.Fatal(err)
	}
	virtual, err := CompareKVAllocation(KVAllocationVirtual, tokens, admitted, rowBytes, blockTokens)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range []KVAllocationReceipt{contiguous, paged, virtual} {
		if r.Engine != "fak-native" || r.AcceptedTokens != tokens || r.LiveBytes != tokens*rowBytes || r.BytesPerAcceptedToken <= 0 {
			t.Fatalf("receipt=%+v", r)
		}
	}
	if contiguous.GrowthCopyBytes != 0 || paged.GrowthCopyBytes != 0 {
		t.Fatal("fixed and paged layouts reported growth copies")
	}
	if virtual.GrowthCopyBytes <= 0 {
		t.Fatal("virtual growth failed to account relocation copies")
	}
	if paged.BlockTouches != 63 {
		t.Fatalf("paged block touches=%d", paged.BlockTouches)
	}
	if contiguous.FragmentationBytes != 24*rowBytes || paged.FragmentationBytes != 24*rowBytes {
		t.Fatalf("slack mismatch contiguous=%d paged=%d", contiguous.FragmentationBytes, paged.FragmentationBytes)
	}
}

func TestCompareKVAllocationReversalShapes(t *testing.T) {
	// Tight admission makes paging avoid arena over-reservation; exact power-of-two
	// growth makes virtual allocation pay relocation bytes despite zero final slack.
	paged, _ := CompareKVAllocation(KVAllocationPaged, 17, 17, 1024, 16)
	arena, _ := CompareKVAllocation(KVAllocationContiguous, 17, 1024, 1024, 16)
	virtual, _ := CompareKVAllocation(KVAllocationVirtual, 1024, 1024, 1024, 16)
	if paged.PhysicalBytes >= arena.PhysicalBytes {
		t.Fatalf("paged=%d arena=%d", paged.PhysicalBytes, arena.PhysicalBytes)
	}
	if virtual.GrowthCopyBytes == 0 || virtual.FragmentationBytes != 0 {
		t.Fatalf("virtual=%+v", virtual)
	}
	if _, err := CompareKVAllocation(KVAllocationVirtual, 18, 17, 1024, 16); err == nil {
		t.Fatal("accepted tokens beyond admission accepted")
	}
}

func BenchmarkCompareKVAllocation(b *testing.B) {
	for _, mode := range []KVAllocationMode{KVAllocationContiguous, KVAllocationPaged, KVAllocationVirtual} {
		b.Run(string(mode), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := CompareKVAllocation(mode, 32768, 65536, 4096, 16); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
