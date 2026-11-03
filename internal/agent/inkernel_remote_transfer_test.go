package agent

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

type plannerSplitTransferStore struct {
	mu          sync.Mutex
	values      map[string][]byte
	blockKey    string
	entered     chan struct{}
	release     chan struct{}
	enterOnce   sync.Once
	releaseOnce sync.Once
}

func (s *plannerSplitTransferStore) Put(_ context.Context, key string, value []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.values == nil {
		s.values = map[string][]byte{}
	}
	s.values[key] = append([]byte(nil), value...)
	return nil
}
func (s *plannerSplitTransferStore) Get(_ context.Context, key string) ([]byte, bool, error) {
	if key == s.blockKey {
		s.enterOnce.Do(func() { close(s.entered) })
		<-s.release
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.values[key]
	return append([]byte(nil), v...), ok, nil
}
func (s *plannerSplitTransferStore) unblock() { s.releaseOnce.Do(func() { close(s.release) }) }

type plannerSplitTransferResult struct {
	tokens  []int
	matched int
	tier    radixkv.SnapshotTier
	err     error
}

// fak-test:runtime medium est=2s lane=default
// This actual planner route witness is synthetic CPU-backed, not a device or
// measured-throughput claim. It covers non-speculative generation; unscoped
// speculative lookup deliberately retains its serialized generic API.
// The store barrier stays held until a separate hot
// request completes; no sleep/setup interval is credited as an improvement.
func TestInKernelRemoteTransferAllowsUnrelatedHotRequest(t *testing.T) {
	for _, scoped := range []bool{false, true} {
		name := "unscoped"
		if scoped {
			name = "scoped"
		}
		t.Run(name, func(t *testing.T) {
			t.Setenv("FAK_INKERNEL_RADIX", "on")
			t.Setenv("FAK_INKERNEL_RADIX_HOST_L2_BYTES", "0")
			t.Setenv("FAK_INKERNEL_RADIX_SSD_BYTES", "0")
			cfg := tinyHybridCfg()
			backend := &countingBackend{Backend: compute.Default(), deviceMemory: true}
			planner := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "split-transfer-planner", false, backend, false)
			planner.quant = false
			if planner.tree == nil {
				t.Fatal("normal constructor did not install the prefix tree")
			}
			store := &plannerSplitTransferStore{entered: make(chan struct{}), release: make(chan struct{})}
			if err := planner.ConfigureKVPrefixRemote(store); err != nil {
				t.Fatal(err)
			}
			base := context.Background()
			if scoped {
				base = WithPrefixCacheIdentity(base, "transfer-tenant", "transfer-agent")
			}
			ctx, cancel := context.WithTimeout(base, 10*time.Second)
			defer cancel()
			idsA := []int{3, 7, 11, 13}
			idsB := []int{5, 9, 15, 17}
			run := func(ids []int) plannerSplitTransferResult {
				var result plannerSplitTransferResult
				_, _, _, result.matched, result.tier, _, _, _, result.err = planner.generateReusedContextWithBias(
					ctx, ids, 1, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool { result.tokens = append(result.tokens, id); return false },
				)
				return result
			}
			coldA := run(idsA)
			if coldA.err != nil {
				t.Fatal(coldA.err)
			}
			_, candidates := planner.KVPrefixPressuredCandidates()
			if len(candidates) != 1 || candidates[0].Tokens != len(idsA) {
				t.Fatalf("actual admission candidates=%+v", candidates)
			}
			digest := candidates[0].SpanDigest
			if got := planner.StageKVPrefixToHost(ctx, digest); got.Outcome != radixkv.SnapshotTransferOK {
				t.Fatalf("actual pressure stage %+v", got)
			}
			if planner.EvictHotKVPrefix(digest) != len(idsA) {
				t.Fatal("remote image did not preserve hot eviction")
			}
			coldB := run(idsB)
			if coldB.err != nil {
				t.Fatal(coldB.err)
			}
			store.blockKey = digest
			remoteDone := make(chan plannerSplitTransferResult, 1)
			hotDone := make(chan plannerSplitTransferResult, 1)
			go func() { remoteDone <- run(idsA) }()
			joinedRemote := false
			joinedHot := false
			startedHot := false
			defer func() {
				store.unblock()
				if !joinedRemote {
					select {
					case <-remoteDone:
					case <-time.After(3 * time.Second):
						t.Error("remote request did not retire")
					}
				}
				if startedHot && !joinedHot {
					select {
					case <-hotDone:
					case <-time.After(3 * time.Second):
						t.Error("hot request did not retire")
					}
				}
			}()
			select {
			case <-store.entered:
			case <-time.After(3 * time.Second):
				t.Fatal("normal request did not reach remote Get")
			}
			startedHot = true
			go func() { hotDone <- run(idsB) }()
			var hot plannerSplitTransferResult
			select {
			case hot = <-hotDone:
				joinedHot = true
			case <-time.After(3 * time.Second):
				t.Fatal("unrelated hot request blocked behind remote I/O")
			}
			if hot.err != nil || hot.matched != len(idsB) || hot.tier != radixkv.SnapshotTierDeviceL1 || !eqInts(hot.tokens, coldB.tokens) {
				t.Fatalf("hot result=%+v cold=%+v", hot, coldB)
			}
			planner.mu.Lock()
			pending := planner.tree.Stats().L3Transfers
			planner.mu.Unlock()
			if pending.Active != 1 {
				t.Fatalf("pending remote ownership=%+v", pending)
			}
			store.unblock()
			var remote plannerSplitTransferResult
			select {
			case remote = <-remoteDone:
				joinedRemote = true
			case <-time.After(3 * time.Second):
				t.Fatal("remote request did not finish")
			}
			if remote.err != nil || remote.matched != len(idsA) || remote.tier != radixkv.SnapshotTierRemoteL3 || !eqInts(remote.tokens, coldA.tokens) {
				t.Fatalf("remote result=%+v cold=%+v", remote, coldA)
			}
			planner.mu.Lock()
			final := planner.tree.Stats().L3Transfers
			planner.mu.Unlock()
			if final.Active != 0 || final.Started != final.Completed {
				t.Fatalf("transaction ownership not retired: %+v", final)
			}
		})
	}
}
