package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

type plannerPrefixCancellationBackend struct {
	*countingBackend
	stores     []*plannerPrefixCancellationKV
	afterClone func()
}

func (b *plannerPrefixCancellationBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}

func (b *plannerPrefixCancellationBackend) wrap(k compute.KVStore) *plannerPrefixCancellationKV {
	out := &plannerPrefixCancellationKV{KVStore: k, backend: b}
	b.stores = append(b.stores, out)
	return out
}

func (b *plannerPrefixCancellationBackend) NewKV(c compute.KVConfig) compute.KVStore {
	return b.wrap(b.Backend.NewKV(c))
}

type plannerPrefixCancellationKV struct {
	compute.KVStore
	backend *plannerPrefixCancellationBackend
	frees   int
}

func (k *plannerPrefixCancellationKV) Clone() compute.KVStore {
	out := k.backend.wrap(k.KVStore.Clone())
	if k.backend.afterClone != nil {
		k.backend.afterClone()
	}
	return out
}

func (k *plannerPrefixCancellationKV) Free() { k.frees++; k.KVStore.Free() }

func (k *plannerPrefixCancellationKV) KVConfig() compute.KVConfig {
	return k.KVStore.(compute.KVGeometry).KVConfig()
}

// fak-test:runtime medium est=1s lane=default
func TestInKernelPrefixLookupCancellationStopsConsumers(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	for _, scoped := range []bool{false, true} {
		entry := "unscoped"
		if scoped {
			entry = "scoped"
		}
		for _, consumer := range []string{"ordinary", "speculative"} {
			t.Run(entry+"/"+consumer, func(t *testing.T) {
				be := &plannerPrefixCancellationBackend{countingBackend: &countingBackend{Backend: compute.Default(), deviceMemory: true}}
				p := NewInKernelPlanner(model.NewSynthetic(tinyHybridCfg()), nil, "prefix-cancel", false, be, false)
				p.quant = false
				p.tree = radixkv.New(0)
				p.scopedTree = radixkv.WrapScopedWithLocker(p.tree, &p.mu)
				if !inKernelPlannerPrefixReuseSupported(p.m, p.backend) {
					t.Fatal("fixture architecture refuses backend prefix reuse")
				}
				ids := []int{3, 7, 11, 13}
				s, err := p.m.NewBackendSessionChecked(be)
				if err != nil {
					t.Fatal(err)
				}
				s.Prefill(ids)
				snap, err := s.PrefixSnapshot()
				s.Close()
				if err != nil {
					t.Fatal(err)
				}
				owner := radixkv.CacheIdentity{Tenant: "planner-cancel"}
				ctx := context.Background()
				if scoped {
					ctx = WithPrefixCacheIdentity(ctx, owner.Tenant, "")
					err = p.scopedTree.AdmitPrivateSnapshot(owner, ids, snap, nil)
				} else {
					root, _ := p.tree.Lookup(nil)
					leaf, insertErr := p.tree.InsertSnapshot(root, ids, snap, nil)
					p.tree.Done(leaf)
					err = insertErr
				}
				if err != nil {
					snap.Close()
					t.Fatal(err)
				}
				defer snap.Close()
				before := p.tree.Stats()
				owners := len(be.stores)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()
				cloneAcquired := false
				be.afterClone = func() { cloneAcquired = true; cancel() }
				emitted, gen, matched := 0, 0, 0
				emit := func(int) bool { emitted++; return false }
				if consumer == "ordinary" {
					gen, _, _, matched, _, _, _, _, err = p.generateReusedContextWithBias(ctx, ids, 2, 0, 0, 0, nil, 0, 0, map[int]bool{}, emit)
				} else {
					var result inKernelGenerateResult
					result, err = p.generateReusedSpeculative(ctx, ids, 2, 0, 0, 0, nil, 0, 0, map[int]bool{}, emit)
					gen, matched = result.gen, result.matched
				}
				be.afterClone = nil
				if !cloneAcquired || !errors.Is(ctx.Err(), context.Canceled) {
					t.Fatal("consumer never reached clone cancellation seam")
				}
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("consumer cancellation err=%v", err)
				}
				if gen != 0 || matched != 0 || emitted != 0 {
					t.Fatalf("canceled generation gen/matched/emitted=%d/%d/%d", gen, matched, emitted)
				}
				if len(be.stores)-owners != 1 {
					t.Fatalf("temporary owners=%d want 1; consumer fell through to session allocation", len(be.stores)-owners)
				}
				if be.stores[owners].frees != 1 {
					t.Fatalf("temporary owner frees=%d", be.stores[owners].frees)
				}
				after := p.tree.Stats()
				if after.L1Hits != before.L1Hits || after.L1HitTokens != before.L1HitTokens || after.DeviceSnapshotBytes != before.DeviceSnapshotBytes {
					t.Fatal("canceled consumer published successful cache reuse")
				}
				if scoped {
					if freed := p.tree.EvictPrefixNS("private/tenant/"+owner.Tenant, ids); freed == 0 {
						t.Fatal("consumer retained source lease")
					}
				} else if freed := p.tree.EvictPrefix(ids); freed == 0 {
					t.Fatal("consumer retained source lease")
				}
				for _, kv := range be.stores {
					if kv.frees != 1 {
						t.Fatalf("final owner frees=%d", kv.frees)
					}
				}
			})
		}
	}
}
