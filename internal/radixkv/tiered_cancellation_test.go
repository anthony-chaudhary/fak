package radixkv

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type tieredCancellationBackend struct {
	compute.Backend
	stores       []*tieredCancellationKV
	afterAcquire func()
}

func (b *tieredCancellationBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}

func (b *tieredCancellationBackend) wrap(k compute.KVStore) compute.KVStore {
	out := &tieredCancellationKV{KVStore: k, backend: b}
	b.stores = append(b.stores, out)
	if b.afterAcquire != nil {
		b.afterAcquire()
	}
	return out
}

func (b *tieredCancellationBackend) NewKV(c compute.KVConfig) compute.KVStore {
	return b.wrap(b.Backend.NewKV(c))
}

type tieredCancellationKV struct {
	compute.KVStore
	backend *tieredCancellationBackend
	frees   int
}

func (k *tieredCancellationKV) Clone() compute.KVStore { return k.backend.wrap(k.KVStore.Clone()) }
func (k *tieredCancellationKV) Free()                  { k.frees++; k.KVStore.Free() }
func (k *tieredCancellationKV) KVConfig() compute.KVConfig {
	return k.KVStore.(compute.KVGeometry).KVConfig()
}
func (k *tieredCancellationKV) SnapshotToHost() (compute.KVHostSnapshot, error) {
	return compute.SnapshotKVToHost(k.KVStore)
}

// fak-test:runtime medium est=1s lane=default
func TestTieredSnapshotCancellationOwnership(t *testing.T) {
	for _, entry := range []string{"unscoped", "scoped", "tenant-over-promoted-fleet", "fleet-selected-snapshot"} {
		scoped := entry != "unscoped"
		for _, tier := range []SnapshotTier{SnapshotTierDeviceL1, SnapshotTierHostL2} {
			for _, phase := range []string{"before", "during", "success"} {
				t.Run(entry+"/"+string(tier)+"/"+phase, func(t *testing.T) {
					m := model.NewSynthetic(remoteL3TestConfig())
					be := &tieredCancellationBackend{Backend: compute.Default()}
					ids := []int{3, 7, 11, 13}
					session, err := m.NewBackendSessionChecked(be)
					if err != nil {
						t.Fatal(err)
					}
					session.Prefill(ids)
					source, err := session.PrefixSnapshot()
					session.Close()
					if err != nil {
						t.Fatal(err)
					}
					tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)
					cache := WrapScoped(tree)
					owner := CacheIdentity{Tenant: "cancel-tenant", Agent: "cancel-agent"}
					wantScope := ScopeTenant
					if entry == "fleet-selected-snapshot" {
						owner = CacheIdentity{Tenant: "different-tenant", Agent: "different-agent"}
						wantScope = ScopeFleet
						// Promote carries legacy KV only; this fixture admits an existing complete fleet record.
						root, matched := tree.LookupNS("shared/fleet", ids)
						var leaf *node
						leaf, err = tree.InsertSnapshot(root, ids[matched:], source, nil)
						tree.Done(leaf)
					} else if scoped {
						err = cache.AdmitPrivateSnapshot(owner, ids, source, nil)
					} else {
						root, _ := tree.Lookup(nil)
						var leaf *node
						leaf, err = tree.InsertSnapshot(root, ids, source, nil)
						tree.Done(leaf)
					}
					if err != nil {
						source.Close()
						t.Fatal(err)
					}
					if entry == "tenant-over-promoted-fleet" {
						if err := cache.Promote(ScopeTenant, owner, ids); err != nil {
							t.Fatal(err)
						}
						defer cache.RevokeFleet(ids)
					}
					_, candidates := tree.PressuredSnapshotCandidates()
					if len(candidates) != 1 {
						t.Fatalf("candidates=%d", len(candidates))
					}
					digest := candidates[0].Digest
					n := tree.findSnapshotByDigest(digest)
					defer tree.releaseSnapshotPayload(n)
					if tier == SnapshotTierHostL2 {
						if stage := tree.StageSnapshotToHost(digest); stage.Outcome != SnapshotTransferOK {
							t.Fatalf("stage outcome=%s", stage.Outcome)
						}
						if tree.EvictHotSnapshot(digest) != len(ids) {
							t.Fatal("hot owner retained")
						}
					}
					before := tree.Stats()
					hot, host := n.snapshot, n.hostSnapshot
					lookup := func(ctx context.Context) (*model.PrefixSnapshot, int, SnapshotTier, error) {
						if scoped {
							snap, _, matched, scope, gotTier, err := cache.LookupSnapshotTieredContext(ctx, owner, ids)
							if snap != nil && scope != wantScope {
								snap.Close()
								t.Fatalf("snapshot selection scope=%d want=%d", scope, wantScope)
							}
							return snap, matched, gotTier, err
						}
						lease, snap, matched, gotTier, err := tree.LookupSnapshotTieredContext(ctx, ids)
						tree.Done(lease)
						return snap, matched, gotTier, err
					}
					ctx, cancel := context.WithCancel(context.Background())
					defer cancel()
					if phase == "before" {
						cancel()
					}
					if phase == "during" {
						be.afterAcquire = cancel
					}
					owners := len(be.stores)
					snap, matched, gotTier, err := lookup(ctx)
					be.afterAcquire = nil
					if snap != nil {
						defer snap.Close()
					}
					if phase != "success" {
						if !errors.Is(err, context.Canceled) || snap != nil {
							t.Fatalf("canceled lookup snapshot_present=%t err=%v", snap != nil, err)
						}
						wantOwners := 0
						if phase == "during" {
							wantOwners = 1
						}
						if len(be.stores)-owners != wantOwners {
							t.Fatalf("new owners=%d want=%d", len(be.stores)-owners, wantOwners)
						}
						for _, kv := range be.stores[owners:] {
							if kv.frees != 1 {
								t.Fatalf("canceled owner frees=%d", kv.frees)
							}
						}
						after := tree.Stats()
						if after.L1Hits != before.L1Hits || after.L2Hits != before.L2Hits || after.L1HitTokens != before.L1HitTokens || after.L2HitTokens != before.L2HitTokens || after.L2RestoreBytes != before.L2RestoreBytes {
							t.Fatal("canceled request counted as successful reuse")
						}
					} else if err != nil || snap == nil || matched != len(ids) || gotTier != tier {
						t.Fatalf("success snapshot=%v matched=%d tier=%s err=%v", snap, matched, gotTier, err)
					}
					if n.refs != 0 {
						t.Fatalf("unreleased leases=%d", n.refs)
					}
					after := tree.Stats()
					if n.snapshot != hot || n.hostSnapshot != host || after.DeviceSnapshotBytes != before.DeviceSnapshotBytes || after.HostSnapshotBytes != before.HostSnapshotBytes {
						t.Fatal("lookup changed cached source ownership or residency")
					}
					if phase != "success" {
						snap, matched, gotTier, err = lookup(context.Background())
						if err != nil || snap == nil || matched != len(ids) || gotTier != tier {
							t.Fatalf("retry snapshot=%v matched=%d tier=%s err=%v", snap, matched, gotTier, err)
						}
						defer snap.Close()
					}
					wantOwners := 1
					if phase == "during" {
						wantOwners = 2
					}
					if len(be.stores)-owners != wantOwners {
						t.Fatalf("successful lookup owner count=%d want=%d", len(be.stores)-owners, wantOwners)
					}
					warm, err := m.NewBackendSessionChecked(be)
					if err != nil {
						t.Fatal(err)
					}
					defer warm.Close()
					if err := snap.Restore(warm); err != nil {
						t.Fatal(err)
					}
					snap.Close()
					cold, err := m.NewBackendSessionChecked(be)
					if err != nil {
						t.Fatal(err)
					}
					defer cold.Close()
					cold.Prefill(ids)
					got, want := warm.Step(17), cold.Step(17)
					if m.Cfg.VocabSize <= 0 || len(got) != m.Cfg.VocabSize || len(want) != m.Cfg.VocabSize {
						t.Fatalf("logit lengths=%d/%d want=%d", len(got), len(want), m.Cfg.VocabSize)
					}
					for i := range got {
						if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) {
							t.Fatalf("nonfinite continuation logit at %d", i)
						}
					}
					if diff := maxAbsDiff(got, want); diff != 0 {
						t.Fatalf("continuation drift=%g", diff)
					}
					warm.Close()
					cold.Close()
					tree.releaseSnapshotPayload(n)
					for _, kv := range be.stores {
						if kv.frees != 1 {
							t.Fatalf("final owner frees=%d", kv.frees)
						}
					}
					stats := tree.Stats()
					wantL1, wantL2 := 0, 0
					if tier == SnapshotTierDeviceL1 {
						wantL1 = 1
					} else {
						wantL2 = 1
					}
					if stats.L1Hits-before.L1Hits != wantL1 || stats.L2Hits-before.L2Hits != wantL2 {
						t.Fatalf("successful reuse hit delta=%d/%d want=%d/%d", stats.L1Hits-before.L1Hits, stats.L2Hits-before.L2Hits, wantL1, wantL2)
					}
				})
			}
		}
	}
}
