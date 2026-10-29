package compute

import (
	"reflect"
	"testing"
)

func TestPagedKVBlocksShareAndRecycle(t *testing.T) {
	var a pagedKVBlocks
	if _, ok := a.alloc(); ok {
		t.Fatal("alloc on an empty arena must report full")
	}
	a.grow(2)
	x, _ := a.alloc()
	y, _ := a.alloc()
	if _, ok := a.alloc(); ok || x == y || a.live != 2 {
		t.Fatalf("two-block arena: x=%d y=%d live=%d", x, y, a.live)
	}
	a.retain(x) // a clone maps x
	if !a.shared(x) || a.shared(y) {
		t.Fatal("retained block must read shared, the other private")
	}
	if a.release(x) || !a.release(x) || a.live != 1 {
		t.Fatalf("x must free only on its last release; live=%d", a.live)
	}
	z, ok := a.alloc()
	if !ok || z != x {
		t.Fatalf("freed id must be reused: got %d ok=%v", z, ok)
	}
	a.grow(4)
	if len(a.free) != 2 || a.refs[y] != 1 {
		t.Fatal("grow must keep live ids and add only new free ids")
	}
}

func TestPagedKVBlocksRejectsDeadBlock(t *testing.T) {
	var a pagedKVBlocks
	a.grow(1)
	id, _ := a.alloc()
	a.release(id)
	defer func() {
		if recover() == nil {
			t.Fatal("releasing a free block must panic")
		}
	}()
	a.release(id)
}

func TestPagedKVRunsNeverCrossAPage(t *testing.T) {
	got := pagedKVRuns(14, 20, 16)
	want := []pagedKVRun{{blk: 0, off: 14, src: 0, n: 2}, {blk: 1, off: 0, src: 2, n: 16}, {blk: 2, off: 0, src: 18, n: 2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("runs = %+v, want %+v", got, want)
	}
	if pagedKVGrowBlocks(0, 3) != 64 || pagedKVGrowBlocks(100, 101) != 150 || pagedKVGrowBlocks(100, 400) != 400 {
		t.Fatal("arena growth: 64 floor, 1.5x, or the need")
	}
}

func TestCUDAPagedKVEligibility(t *testing.T) {
	cfg := KVConfig{NumLayers: 2, NumKVHeads: 4, HeadDim: 128}
	if !CUDAPagedKVEligible(cfg) {
		t.Fatal("f32 hd128 must be eligible")
	}
	for _, bad := range []KVConfig{
		{NumLayers: 2, NumKVHeads: 4, HeadDim: 72},
		{NumLayers: 2, NumKVHeads: 4, HeadDim: 128, Precision: KVPrecisionF16},
		{NumLayers: 0, NumKVHeads: 4, HeadDim: 128},
	} {
		if CUDAPagedKVEligible(bad) {
			t.Fatalf("%+v must stay contiguous", bad)
		}
	}
	t.Setenv("FAK_CUDA_PAGED_KV", "")
	if CUDAPagedKVEnabled() {
		t.Fatal("paged KV must default off")
	}
	t.Setenv("FAK_CUDA_PAGED_KV", "on")
	if !CUDAPagedKVEnabled() {
		t.Fatal("FAK_CUDA_PAGED_KV=on must opt in")
	}
}
