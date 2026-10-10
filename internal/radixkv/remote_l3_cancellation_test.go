package radixkv

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type cancellationSnapshotStore struct {
	memorySnapshotStore
	afterGet func()
	gets     int
}

func (s *cancellationSnapshotStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	s.gets++
	data, found, err := s.memorySnapshotStore.Get(ctx, key)
	if s.afterGet != nil {
		s.afterGet()
	}
	return data, found, err
}

type cancellationRestoreBackend struct {
	compute.Backend
	afterNewKV func()
	stores     []*cancellationRestoreKV
}

func (b *cancellationRestoreBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	kv := &cancellationRestoreKV{KVStore: b.Backend.NewKV(cfg)}
	b.stores = append(b.stores, kv)
	if b.afterNewKV != nil {
		b.afterNewKV()
	}
	return kv
}

type cancellationRestoreKV struct {
	compute.KVStore
	frees int
}

func (k *cancellationRestoreKV) Free() { k.frees++; k.KVStore.Free() }

// fak-test:runtime medium est=1s lane=default
// Estimated only, not measured: bounded synthetic model and ownership cases.
func TestRemoteSnapshotCancellationBeforePublication(t *testing.T) {
	for _, entry := range []string{"lookup", "page-in"} {
		for _, phase := range []string{"before-read", "after-read", "during-restore", "success"} {
			t.Run(entry+"/"+phase, func(t *testing.T) {
				cfg := remoteL3TestConfig()
				m := model.NewSynthetic(cfg)
				source := &deviceCapsBackend{Backend: compute.Default()}
				store := &cancellationSnapshotStore{}
				tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 0, EvictionLRU)
				if err := tree.ConfigureRemoteSnapshotStore(store, "cancel-test", source, m.Cfg); err != nil {
					t.Fatal(err)
				}
				ids := []int{3, 7, 11, 13}
				digest := insertRemoteL3Snapshot(t, tree, m, source, ids)
				if got := tree.StageSnapshotToRemote(context.Background(), digest); got.Outcome != SnapshotTransferOK {
					t.Fatalf("stage: %+v", got)
				}
				if tree.EvictHotSnapshot(digest) != len(ids) {
					t.Fatal("could not remove hot owner")
				}
				n := tree.findSnapshotByDigest(digest)
				defer tree.releaseSnapshotPayload(n)
				ref, remoteBytes := n.remoteSnapshot, tree.remoteSnapshotBytes
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				target := &cancellationRestoreBackend{Backend: compute.Default()}
				tree.remoteBackend = target
				switch phase {
				case "before-read":
					cancel()
				case "after-read":
					store.afterGet = cancel
				case "during-restore":
					target.afterNewKV = cancel
				}
				var snap *model.PrefixSnapshot
				if entry == "lookup" {
					lease, got, _, _, err := tree.LookupSnapshotTieredContext(ctx, ids)
					tree.Done(lease)
					snap = got
					if phase == "success" {
						if err != nil || snap == nil {
							t.Fatalf("successful lookup snapshot=%v err=%v", snap, err)
						}
					} else if !errors.Is(err, context.Canceled) || snap != nil {
						t.Fatalf("canceled lookup snapshot=%v err=%v", snap, err)
					}
				} else {
					got := tree.RestoreSnapshotFromRemote(ctx, digest)
					if phase == "success" {
						if got.Outcome != SnapshotTransferOK || n.snapshot == nil {
							t.Fatalf("successful page-in: %+v", got)
						}
					} else if got.Outcome != SnapshotTransferFault {
						t.Fatalf("canceled page-in: %+v", got)
					}
				}
				if phase != "success" {
					if n.snapshot != nil || tree.snapshotBytes != 0 || tree.l3Hits != 0 || tree.l3HitTokens != 0 {
						t.Fatal("cancellation published restored ownership or a hit")
					}
					if n.remoteSnapshot != ref || tree.remoteSnapshotBytes != remoteBytes {
						t.Fatal("cancellation discarded reusable remote image")
					}
				} else if tree.l3Hits != 1 {
					t.Fatalf("success hits=%d", tree.l3Hits)
				}
				wantReads, wantStores := 1, 0
				if phase == "before-read" {
					wantReads = 0
				}
				if phase == "during-restore" || phase == "success" {
					wantStores = 1
				}
				if store.gets != wantReads || len(target.stores) != wantStores {
					t.Fatalf("reads/stores=%d/%d, want %d/%d", store.gets, len(target.stores), wantReads, wantStores)
				}
				if phase == "during-restore" && target.stores[0].frees != 1 {
					t.Fatalf("canceled owner frees=%d", target.stores[0].frees)
				}
				stats := tree.Stats()
				if stats.L3BreakerConsecutiveFaults != 0 || stats.L3BreakerTotalFaults != 0 {
					t.Fatal("caller cancellation penalized remote breaker")
				}
				if snap != nil {
					snap.Close()
				}
				tree.releaseSnapshotPayload(n)
				for _, kv := range target.stores {
					if kv.frees != 1 {
						t.Fatalf("final owner frees=%d, want 1", kv.frees)
					}
				}
			})
		}
	}
}
