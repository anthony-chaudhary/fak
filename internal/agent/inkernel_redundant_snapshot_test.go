package agent

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

type snapshotCloneCountingBackend struct {
	*countingBackend
	cloneCalls atomic.Int64
}

func (b *snapshotCloneCountingBackend) CloneTensor(t compute.Tensor) (compute.Tensor, error) {
	b.cloneCalls.Add(1)
	return b.countingBackend.CloneTensor(t)
}

func (b *snapshotCloneCountingBackend) resetSnapshotCloneCalls() {
	b.cloneCalls.Store(0)
}

func (b *snapshotCloneCountingBackend) snapshotCloneCalls() int64 {
	return b.cloneCalls.Load()
}

func TestInKernelExactDeviceL1CachedLogitsSkipsRedundantSnapshotClone(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")

	cfg := tinyHybridCfg()
	backend := &snapshotCloneCountingBackend{countingBackend: &countingBackend{
		Backend:      compute.Default(),
		deviceMemory: true,
	}}
	planner := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "exact-device-l1-clone-count", false, backend, false)
	planner.quant = false
	ids := synthIDs(cfg.VocabSize, 9, 9527)
	run := func() (gen []int, cacheable, matched int, tier radixkv.SnapshotTier) {
		_, _, cacheable, matched, tier, _, _, _, err := planner.generateReusedContextWithBias(
			context.Background(), ids, 1, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
				gen = append(gen, id)
				return false
			},
		)
		if err != nil {
			t.Fatalf("generateReusedContextWithBias: %v", err)
		}
		return gen, cacheable, matched, tier
	}

	primeTokens, primeCacheable, primeMatched, primeTier := run()
	if primeCacheable != 0 || primeMatched != 0 || primeTier != radixkv.SnapshotTierMiss {
		t.Fatalf("cold request cacheable=%d matched=%d tier=%s, want 0/0/miss", primeCacheable, primeMatched, primeTier)
	}
	oneSnapshotClone := backend.snapshotCloneCalls()
	if oneSnapshotClone == 0 {
		t.Fatal("cold admission performed no backend tensor clones; test cannot observe snapshot depth")
	}

	backend.resetSnapshotCloneCalls()
	replayTokens, cacheable, matched, tier := run()
	if tier != radixkv.SnapshotTierDeviceL1 || cacheable != len(ids) || matched != len(ids) {
		t.Fatalf("exact replay cacheable=%d matched=%d tier=%s, want %d/%d/device-l1", cacheable, matched, tier, len(ids), len(ids))
	}
	if !eqInts(replayTokens, primeTokens) {
		t.Fatalf("exact replay changed generated token: prime=%v replay=%v", primeTokens, replayTokens)
	}
	if got := backend.snapshotCloneCalls(); got != oneSnapshotClone {
		t.Fatalf("exact Device-L1 cached-logits hit cloned %d backend tensors, want lookup-side clone only (%d); redundant post-restore snapshot admission ran", got, oneSnapshotClone)
	}
}
