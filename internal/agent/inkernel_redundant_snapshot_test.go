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
	// Recurrent-state admission is copy-on-write: an unshared owner is retained
	// by reference, so a cold request snapshots with zero eager tensor clones.
	// Snapshot depth is therefore observed as the COW baseline of 0; the
	// invariant under test is that an exact replay adds no clone on top of it.
	coldSnapshotClones := backend.snapshotCloneCalls()
	if coldSnapshotClones != 0 {
		t.Fatalf("cold admission eagerly cloned %d backend tensors, want 0 (copy-on-write snapshot)", coldSnapshotClones)
	}

	backend.resetSnapshotCloneCalls()
	replayTokens, cacheable, matched, tier := run()
	if tier != radixkv.SnapshotTierDeviceL1 || cacheable != len(ids) || matched != len(ids) {
		t.Fatalf("exact replay cacheable=%d matched=%d tier=%s, want %d/%d/device-l1", cacheable, matched, tier, len(ids), len(ids))
	}
	if !eqInts(replayTokens, primeTokens) {
		t.Fatalf("exact replay changed generated token: prime=%v replay=%v", primeTokens, replayTokens)
	}
	if got := backend.snapshotCloneCalls(); got != 0 {
		t.Fatalf("exact Device-L1 cached-logits hit cloned %d backend tensors, want 0; redundant post-restore snapshot admission ran", got)
	}
}

func TestInKernelSameTenantExactDeviceL1CachedLogitsSkipsRedundantSnapshotClone(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")

	cfg := tinyHybridCfg()
	backend := &snapshotCloneCountingBackend{countingBackend: &countingBackend{
		Backend:      compute.Default(),
		deviceMemory: true,
	}}
	planner := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "same-tenant-exact-device-l1-clone-count", false, backend, false)
	planner.quant = false
	ids := synthIDs(cfg.VocabSize, 9, 9527)

	prefixCacheIdentityContext := func(ctx context.Context, tenant, agent string) context.Context {
		return WithPrefixCacheIdentity(ctx, tenant, agent)
	}

	run := func(ctx context.Context) (gen []int, cacheable, matched int, tier radixkv.SnapshotTier) {
		_, _, cacheable, matched, tier, _, _, _, err := planner.generateReusedContextWithBias(
			ctx, ids, 1, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
				gen = append(gen, id)
				return false
			},
		)
		if err != nil {
			t.Fatalf("generateReusedContextWithBias: %v", err)
		}
		return gen, cacheable, matched, tier
	}

	ctx1 := prefixCacheIdentityContext(context.Background(), "tenant-alpha", "agent-1")
	primeTokens, primeCacheable, primeMatched, primeTier := run(ctx1)
	if primeCacheable != 0 || primeMatched != 0 || primeTier != radixkv.SnapshotTierMiss {
		t.Fatalf("cold request cacheable=%d matched=%d tier=%s, want 0/0/miss", primeCacheable, primeMatched, primeTier)
	}
	// Copy-on-write admission: the unshared owner is shared by reference, so a
	// cold same-tenant request also snapshots with zero eager tensor clones.
	coldSnapshotClones := backend.snapshotCloneCalls()
	if coldSnapshotClones != 0 {
		t.Fatalf("cold admission eagerly cloned %d backend tensors, want 0 (copy-on-write snapshot)", coldSnapshotClones)
	}

	backend.resetSnapshotCloneCalls()
	ctx2 := prefixCacheIdentityContext(context.Background(), "tenant-alpha", "agent-2")
	replayTokens, cacheable, matched, tier := run(ctx2)
	if tier != radixkv.SnapshotTierDeviceL1 || cacheable != len(ids) || matched != len(ids) {
		t.Fatalf("exact replay cacheable=%d matched=%d tier=%s, want %d/%d/device-l1", cacheable, matched, tier, len(ids), len(ids))
	}
	if !eqInts(replayTokens, primeTokens) {
		t.Fatalf("exact replay changed generated token: prime=%v replay=%v", primeTokens, replayTokens)
	}
	if got := backend.snapshotCloneCalls(); got != 0 {
		t.Fatalf("same-tenant exact Device-L1 cached-logits hit cloned %d backend tensors, want 0; redundant post-restore snapshot admission ran", got)
	}

	// Verify the snapshot remains resident in the scoped tree at ScopeTenant.
	snap, _, matchedTokens, scope, lookupTier, err := planner.scopedTree.LookupSnapshotTieredContext(
		context.Background(),
		radixkv.CacheIdentity{Tenant: "tenant-alpha", Agent: "agent-2"},
		ids,
	)
	if err != nil {
		t.Fatalf("LookupSnapshotTieredContext: %v", err)
	}
	if snap == nil {
		t.Fatal("expected resident snapshot in scopedTree, got nil")
	}
	snap.Close()
	if matchedTokens != len(ids) {
		t.Fatalf("resident snapshot matched=%d, want %d", matchedTokens, len(ids))
	}
	if scope != radixkv.ScopeTenant {
		t.Fatalf("resident snapshot scope=%v, want %v", scope, radixkv.ScopeTenant)
	}
	if lookupTier != radixkv.SnapshotTierDeviceL1 {
		t.Fatalf("resident snapshot tier=%s, want %s", lookupTier, radixkv.SnapshotTierDeviceL1)
	}
}
