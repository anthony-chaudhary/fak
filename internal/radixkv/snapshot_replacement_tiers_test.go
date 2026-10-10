package radixkv

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Count the physical KV owner releases while retaining the real CPU snapshot,
// host staging, and restore implementations. No device execution is required.
type replacementTierBackend struct {
	compute.Backend
	last *replacementTierKV
}

func (b *replacementTierBackend) wrap(kv compute.KVStore) *replacementTierKV {
	out := &replacementTierKV{KVStore: kv, backend: b}
	b.last = out
	return out
}
func (b *replacementTierBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	return b.wrap(b.Backend.NewKV(cfg))
}

type replacementTierKV struct {
	compute.KVStore
	backend *replacementTierBackend
	frees   int
}

func (k *replacementTierKV) Clone() compute.KVStore { return k.backend.wrap(k.KVStore.Clone()) }
func (k *replacementTierKV) KVConfig() compute.KVConfig {
	return k.KVStore.(compute.KVGeometry).KVConfig()
}
func (k *replacementTierKV) SnapshotToHost() (compute.KVHostSnapshot, error) {
	return compute.SnapshotKVToHost(k.KVStore)
}
func (k *replacementTierKV) Free() { k.frees++; k.KVStore.Free() }

// fak-test:runtime medium est=1s lane=default
// Estimated only, not measured: bounded synthetic model and ownership cases.
func TestSnapshotReplacementRetiresPreviousStagedTiers(t *testing.T) {
	for _, mode := range []string{"replace-hot", "replace-host-only", "same-owner", "reject"} {
		t.Run(mode, func(t *testing.T) {
			cfg := remoteL3TestConfig()
			m := model.NewSynthetic(cfg)
			be := &replacementTierBackend{Backend: compute.Default()}
			ids := []int{3, 7, 11, 13}
			makeSnapshot := func() (*model.PrefixSnapshot, *replacementTierKV) {
				s, err := m.NewBackendSessionChecked(be)
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				s.Prefill(ids)
				snap, err := s.PrefixSnapshot()
				if err != nil {
					t.Fatal(err)
				}
				return snap, be.last
			}
			tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)
			store := &memorySnapshotStore{}
			if err := tree.ConfigureRemoteSnapshotStore(store, "replacement-test", be, cfg); err != nil {
				t.Fatal(err)
			}
			old, oldKV := makeSnapshot()
			root, _ := tree.Lookup(nil)
			n, err := tree.InsertSnapshot(root, ids, old, []float32{1})
			if err != nil {
				old.Close()
				t.Fatal(err)
			}
			defer tree.releaseSnapshotPayload(n)
			oldHandle, _ := n.RecordHandle()
			digest := tree.snapshotDigest("", n)
			if got := tree.StageSnapshotToHost(digest); got.Outcome != SnapshotTransferOK {
				t.Fatalf("host stage: %+v", got)
			}
			if got := tree.StageSnapshotToRemote(context.Background(), digest); got.Outcome != SnapshotTransferOK {
				t.Fatalf("remote stage: %+v", got)
			}
			oldHost, oldRemote := n.hostSnapshot, n.remoteSnapshot
			hostBytes, remoteBytes := tree.hostSnapshotBytes, tree.remoteSnapshotBytes
			if hostBytes <= 0 || remoteBytes <= 0 {
				t.Fatal("fixture did not own both staged tiers")
			}
			if mode == "replace-host-only" {
				if tree.EvictHotSnapshot(digest) != len(ids) {
					t.Fatal("could not evict old hot owner")
				}
			}
			next, nextKV := old, oldKV
			if mode != "same-owner" {
				next, nextKV = makeSnapshot()
			}
			defer next.Close()
			if mode == "reject" {
				tree.maxSnapshotBytes = 1
			}
			got, err := tree.InsertSnapshot(n, nil, next, []float32{2})
			if mode == "reject" {
				if !errors.Is(err, ErrSnapshotByteBudget) {
					t.Fatalf("rejection = %v", err)
				}
				if n.snapshot != old || n.hostSnapshot != oldHost || n.remoteSnapshot != oldRemote || !tree.ValidateRecord(oldHandle) {
					t.Fatal("failed admission changed old ownership or incarnation")
				}
				if tree.hostSnapshotBytes != hostBytes || tree.remoteSnapshotBytes != remoteBytes || oldKV.frees != 0 || nextKV.frees != 0 {
					t.Fatal("failed admission changed accounting or freed a retained owner")
				}
				return
			}
			if err != nil || got != n || n.snapshot != next {
				t.Fatalf("replacement node=%p err=%v", got, err)
			}
			if n.hostSnapshot != nil || n.remoteSnapshot != nil || tree.hostSnapshotBytes != 0 || tree.remoteSnapshotBytes != 0 {
				t.Fatal("replacement retained prior staged tiers or their charges")
			}
			if oldHost.ResidentBytes() != 0 {
				t.Fatal("retired host image was not closed")
			}
			if tree.ValidateRecord(oldHandle) {
				t.Fatal("replacement kept prior record generation")
			}
			if nextKV.frees != 0 || next.Cache == nil {
				t.Fatal("replacement closed its new live owner")
			}
			if mode != "same-owner" && oldKV.frees != 1 {
				t.Fatalf("old KV frees=%d, want 1", oldKV.frees)
			}
			if tree.snapshotBytes != snapshotResidentBytes(next, n.cachedLogits) {
				t.Fatal("new hot accounting mismatch")
			}
			// Until the new incarnation is staged, the old remote object must not
			// authorize dropping its only live owner or satisfy a remote page-in.
			if tree.EvictHotSnapshot(digest) != 0 {
				t.Fatal("old staged copy authorized new-owner eviction")
			}
			if got := tree.RestoreSnapshotFromRemote(context.Background(), digest); got.Outcome != SnapshotTransferMiss {
				t.Fatalf("old remote object remained reachable: %+v", got)
			}
			if got := tree.StageSnapshotToHost(digest); got.Outcome != SnapshotTransferOK || n.hostSnapshot == oldHost {
				t.Fatalf("new host stage: %+v", got)
			}
			if got := tree.StageSnapshotToRemote(context.Background(), digest); got.Outcome != SnapshotTransferOK {
				t.Fatalf("new remote stage: %+v", got)
			}
			tree.releaseSnapshotPayload(n)
			tree.releaseSnapshotPayload(n)
			if nextKV.frees != 1 || tree.snapshotBytes != 0 || tree.hostSnapshotBytes != 0 || tree.remoteSnapshotBytes != 0 {
				t.Fatalf("teardown frees=%d accounting=%d/%d/%d", nextKV.frees, tree.snapshotBytes, tree.hostSnapshotBytes, tree.remoteSnapshotBytes)
			}
		})
	}
}
