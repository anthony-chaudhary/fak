package radixkv

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type splitTransferStore struct {
	mu          sync.Mutex
	values      map[string][]byte
	block       string
	entered     chan struct{}
	release     chan struct{}
	enteredOnce sync.Once
	releaseOnce sync.Once
	panicCall   bool
}

func (s *splitTransferStore) pause(kind string) {
	if s.block != kind {
		return
	}
	s.enteredOnce.Do(func() { close(s.entered) })
	<-s.release // deliberately ignores cancellation: ownership must wait for return
	if s.panicCall {
		panic("injected store callback")
	}
}
func (s *splitTransferStore) unblock() { s.releaseOnce.Do(func() { close(s.release) }) }
func (s *splitTransferStore) Put(_ context.Context, key string, value []byte) error {
	s.pause("put")
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.values == nil {
		s.values = map[string][]byte{}
	}
	s.values[key] = append([]byte(nil), value...)
	return nil
}
func (s *splitTransferStore) Get(_ context.Context, key string) ([]byte, bool, error) {
	s.pause("get")
	s.mu.Lock()
	defer s.mu.Unlock()
	value, ok := s.values[key]
	return append([]byte(nil), value...), ok, nil
}
func waitSplitTransfer(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(3 * time.Second):
		t.Fatalf("timeout: %s", what)
	}
}

// fak-test:runtime medium est=2s lane=default
// Estimated only. Barriers prove progress/ownership, not wall-time speedup.
func TestRemoteSnapshotSplitPhaseLifecycle(t *testing.T) {
	for _, direction := range []string{"put", "get"} {
		for _, change := range []string{"none", "cancel", "reconfigure", "detach", "replace", "teardown", "panic"} {
			t.Run(direction+"/"+change, func(t *testing.T) {
				cfg := remoteL3TestConfig()
				m := model.NewSynthetic(cfg)
				source := &deviceCapsBackend{Backend: compute.Default()}
				target := &cancellationRestoreBackend{Backend: compute.Default()}
				store := &splitTransferStore{entered: make(chan struct{}), release: make(chan struct{})}
				tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 0, EvictionLRU)
				if err := tree.ConfigureRemoteSnapshotStore(store, "split-transfer", target, cfg); err != nil {
					t.Fatal(err)
				}
				ids := []int{3, 7, 11, 13}
				digest := insertRemoteL3Snapshot(t, tree, m, source, ids)
				n := tree.findSnapshotByDigest(digest)
				if direction == "get" {
					if got := tree.StageSnapshotToRemote(context.Background(), digest); got.Outcome != SnapshotTransferOK {
						t.Fatalf("prepare remote: %+v", got)
					}
					if tree.EvictHotSnapshot(digest) != len(ids) {
						t.Fatal("prepare cold owner")
					}
				}
				before := tree.RemoteSnapshotTransfers()
				originalRef := n.remoteSnapshot
				store.block = direction
				store.panicCall = change == "panic"
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				var mu sync.Mutex
				result := make(chan SnapshotTransfer, 1)
				done := make(chan struct{})
				go func() {
					defer close(done)
					mu.Lock()
					defer mu.Unlock()
					if direction == "put" {
						result <- tree.StageSnapshotToRemoteLocked(ctx, digest, &mu)
					} else {
						result <- tree.RestoreSnapshotFromRemoteLocked(ctx, digest, &mu)
					}
				}()
				defer func() { store.unblock(); waitSplitTransfer(t, done, "job retirement") }()
				waitSplitTransfer(t, store.entered, "store entered")
				if !mu.TryLock() {
					t.Fatal("remote callback retained the external tree mutex")
				}
				// Inspect the live real transaction while its store callback is blocked.
				active := tree.RemoteSnapshotTransfers()
				if active.Active != 1 || active.Started != before.Started+1 || active.Completed != before.Completed {
					mu.Unlock()
					t.Fatalf("pending ownership %+v before %+v", active, before)
				}
				if n.refs != 1 {
					mu.Unlock()
					t.Fatalf("source lease refs=%d", n.refs)
				}
				if direction == "get" {
					probeBefore := tree.RemoteL3Breaker().Stats().ProbesAttempted
					snap, found, err := tree.restoreSnapshotFromRemoteWithLock(context.Background(), "", n, &remoteTransferLock{locker: &mu})
					if err != nil || found || snap != nil || n.remoteSnapshot != originalRef || tree.RemoteL3Breaker().Stats().ProbesAttempted != probeBefore {
						mu.Unlock()
						t.Fatal("busy work acquired a probe, mutated the ref, or returned a hit")
					}
				}
				var replacement *model.PrefixSnapshot
				switch change {
				case "cancel":
					cancel()
				case "reconfigure":
					if err := tree.ConfigureRemoteSnapshotStore(store, "split-transfer", target, cfg); err != nil {
						mu.Unlock()
						t.Fatal(err)
					}
				case "detach":
					if err := tree.DetachRemoteSnapshotStore(context.Background()); err != nil {
						mu.Unlock()
						t.Fatal(err)
					}
				case "replace":
					session, err := m.NewBackendSessionChecked(source)
					if err != nil {
						mu.Unlock()
						t.Fatal(err)
					}
					session.Prefill(ids)
					replacement, err = session.PrefixSnapshot()
					session.Close()
					if err != nil {
						mu.Unlock()
						t.Fatal(err)
					}
					boundary, matched := tree.Lookup(ids)
					leaf, err := tree.InsertSnapshot(boundary, ids[matched:], replacement, nil)
					tree.Done(leaf)
					if err != nil {
						replacement.Close()
						mu.Unlock()
						t.Fatal(err)
					}
				case "teardown":
					tree.closeSubtreeSnapshots(n)
				}
				mu.Unlock()
				// Cancellation/detach does not pretend a still-running callback retired.
				select {
				case <-done:
					t.Fatal("job retired before the store returned")
				default:
				}
				store.unblock()
				waitSplitTransfer(t, done, "completed job")
				got := <-result
				mu.Lock()
				if change == "none" {
					if got.Outcome != SnapshotTransferOK {
						mu.Unlock()
						t.Fatalf("success %+v", got)
					}
					if direction == "put" && n.remoteSnapshot == nil {
						mu.Unlock()
						t.Fatal("successful put not published")
					}
					if direction == "get" && n.snapshot == nil {
						mu.Unlock()
						t.Fatal("successful page-in not published")
					}
				} else {
					wantOutcome := SnapshotTransferFault
					if direction == "get" && (change == "reconfigure" || change == "detach" || change == "replace" || change == "teardown") {
						wantOutcome = SnapshotTransferMiss
					}
					if got.Outcome != wantOutcome {
						mu.Unlock()
						t.Fatalf("refused job %+v want %s", got, wantOutcome)
					}
					if direction == "put" && n.remoteSnapshot != nil {
						mu.Unlock()
						t.Fatal("failed/stale put published a local reference")
					}
					if direction == "get" && change != "replace" && n.snapshot != nil {
						mu.Unlock()
						t.Fatal("failed/stale read published a hot owner")
					}
					if change == "replace" && n.snapshot != replacement {
						mu.Unlock()
						t.Fatal("late job damaged replacement owner")
					}
				}
				after := tree.RemoteSnapshotTransfers()
				if after.Active != 0 || after.Started != before.Started+1 || after.Completed != before.Completed+1 || n.refs != 0 {
					mu.Unlock()
					t.Fatalf("unbalanced ownership after=%+v refs=%d", after, n.refs)
				}
				if direction == "get" && change != "none" && tree.l3Hits != 0 {
					mu.Unlock()
					t.Fatal("failed read counted a hit")
				}
				tree.releaseSnapshotPayload(n)
				mu.Unlock()
				for _, kv := range target.stores {
					if kv.frees != 1 {
						t.Fatalf("destination freed %d times", kv.frees)
					}
				}
			})
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRemoteSnapshotSplitPhaseConfigurationAndLockRefusals(t *testing.T) {
	cfg := remoteL3TestConfig()
	tree := New(0)
	store := &memorySnapshotStore{}
	backend := compute.Default()
	if got := tree.StageSnapshotToRemoteLocked(context.Background(), "x", nil); got.Outcome != SnapshotTransferFault {
		t.Fatal(got)
	}
	if _, _, _, _, err := tree.LookupSnapshotTieredContextLocked(context.Background(), nil, nil); !errors.Is(err, ErrRemoteTransferLock) {
		t.Fatal(err)
	}
	tree.remoteConfigGeneration = ^uint64(0)
	if err := tree.ConfigureRemoteSnapshotStore(store, "overflow", backend, cfg); !errors.Is(err, ErrRemoteConfigExhausted) {
		t.Fatal(err)
	}
	if tree.remoteSnapshotStore != nil {
		t.Fatal("overflow configuration mutated store")
	}
	if err := tree.DetachRemoteSnapshotStore(context.Background()); !errors.Is(err, ErrRemoteConfigExhausted) {
		t.Fatal(err)
	}
}

// fak-test:runtime medium est=1s lane=default
// Policy eviction can detach the returned lookup boundary while a different,
// still-valid ancestor owns the remote transfer. Both leases must be balanced.
func TestRemoteSnapshotSplitPhaseRefreshesEvictedBoundary(t *testing.T) {
	for _, ancestor := range []bool{false, true} {
		name := "direct-stale"
		if ancestor {
			name = "valid-ancestor-detached-boundary"
		}
		t.Run(name, func(t *testing.T) {
			cfg := remoteL3TestConfig()
			m := model.NewSynthetic(cfg)
			source := &deviceCapsBackend{Backend: compute.Default()}
			target := &cancellationRestoreBackend{Backend: compute.Default()}
			store := &splitTransferStore{entered: make(chan struct{}), release: make(chan struct{})}
			tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 0, EvictionLRU)
			if err := tree.ConfigureRemoteSnapshotStore(store, "refresh-boundary", target, cfg); err != nil {
				t.Fatal(err)
			}
			ids := []int{3, 7, 11, 13}
			prefix := ids
			if ancestor {
				prefix = ids[:2]
			}
			digest := insertRemoteL3Snapshot(t, tree, m, source, prefix)
			owner := tree.findSnapshotByDigest(digest)
			if ancestor {
				boundary, matched := tree.Lookup(ids)
				leaf := tree.InsertWithLogits(boundary, ids[matched:], nil, nil)
				tree.Done(leaf)
			}
			old, _ := tree.Lookup(ids)
			tree.Done(old)
			if old == nil || old.parent == nil {
				t.Fatal("missing policy-eviction boundary")
			}
			if got := tree.StageSnapshotToRemote(context.Background(), digest); got.Outcome != SnapshotTransferOK {
				t.Fatal(got)
			}
			if tree.EvictHotSnapshot(digest) != len(prefix) {
				t.Fatal("could not evict hot prefix")
			}
			store.block = "get"
			var mu sync.Mutex
			type lookupResult struct {
				boundary *node
				snapshot *model.PrefixSnapshot
				matched  int
				tier     SnapshotTier
				err      error
			}
			results := make(chan lookupResult, 1)
			done := make(chan struct{})
			go func() {
				defer close(done)
				mu.Lock()
				defer mu.Unlock()
				var result lookupResult
				result.boundary, result.snapshot, result.matched, result.tier, result.err = tree.LookupSnapshotTieredContextLocked(context.Background(), ids, &mu)
				results <- result
			}()
			defer func() { store.unblock(); waitSplitTransfer(t, done, "lookup retirement") }()
			waitSplitTransfer(t, store.entered, "lookup store entered")
			mu.Lock()
			if old.refs < 1 {
				mu.Unlock()
				t.Fatal("lookup boundary lease was not retained")
			}
			if tree.EvictNode(old) == 0 || tree.nodeAttached(old) {
				mu.Unlock()
				t.Fatal("policy eviction did not detach old boundary")
			}
			mu.Unlock()
			store.unblock()
			waitSplitTransfer(t, done, "lookup completed")
			result := <-results
			mu.Lock()
			if result.err != nil || result.boundary == old || !tree.nodeAttached(result.boundary) || old.refs != 0 {
				mu.Unlock()
				t.Fatalf("stale boundary returned or lease leaked: result=%+v oldrefs=%d", result, old.refs)
			}
			if ancestor {
				if result.snapshot == nil || result.matched != len(prefix) || result.tier != SnapshotTierRemoteL3 {
					mu.Unlock()
					t.Fatalf("valid ancestor result lost: %+v", result)
				}
			} else if result.snapshot != nil || result.matched != 0 || result.tier != SnapshotTierMiss {
				mu.Unlock()
				t.Fatalf("stale direct result published: %+v", result)
			}
			if result.snapshot != nil {
				result.snapshot.Close()
			}
			tree.Done(result.boundary)
			stats := tree.RemoteSnapshotTransfers()
			if stats.Active != 0 || stats.Started != stats.Completed || owner.refs != 0 {
				mu.Unlock()
				t.Fatalf("job/boundary ownership not retired: %+v refs=%d", stats, owner.refs)
			}
			tree.releaseSnapshotPayload(owner)
			mu.Unlock()
			for _, kv := range target.stores {
				if kv.frees != 1 {
					t.Fatalf("destination freed %d times", kv.frees)
				}
			}
		})
	}
}
