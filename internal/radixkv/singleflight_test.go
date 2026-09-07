package radixkv

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestPrefixFlightGroupCoalescing verifies that N concurrent subagents requesting
// the same prompt prefix execute exactly ONE prefill pass (Leader), while all other
// subagents (Followers) block on the completion broadcast and reuse the computed KV cache.
func TestPrefixFlightGroupCoalescing(t *testing.T) {
	m := newSyntheticTiny()
	tree := New(0)
	g := NewPrefixFlightGroup(tree)

	const numSubagents = 20
	prefix := seq(1, 16) // shared prompt prefix (e.g. system prompt + tools)

	var prefillCount atomic.Int64
	var startGate sync.WaitGroup
	startGate.Add(1)

	type subagentResult struct {
		kv       *model.KVCache
		logits   []float32
		isLeader bool
		err      error
		duration time.Duration
	}

	results := make([]subagentResult, numSubagents)
	var wg sync.WaitGroup

	startTime := time.Now()
	for i := 0; i < numSubagents; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			startGate.Wait() // align arrivals for maximum concurrency

			t0 := time.Now()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			kv, logits, isLeader, err := g.Coalesce(ctx, prefix, func(pctx context.Context) (*model.KVCache, []float32, error) {
				prefillCount.Add(1)
				// Simulate GEMM compute latency
				time.Sleep(30 * time.Millisecond)

				sess := m.NewSession()
				l := sess.Prefill(prefix)
				return sess.Cache, l, nil
			})

			results[idx] = subagentResult{
				kv:       kv,
				logits:   logits,
				isLeader: isLeader,
				err:      err,
				duration: time.Since(t0),
			}
		}(i)
	}

	// Release all subagents simultaneously
	startGate.Done()
	wg.Wait()
	totalElapsed := time.Since(startTime)

	// Invariant 1: Exactly 1 GEMM prefill executed
	if count := prefillCount.Load(); count != 1 {
		t.Fatalf("expected exactly 1 prefill computation, got %d", count)
	}

	// Invariant 2: Exactly 1 Leader and N-1 Followers coalesced
	if leaders := g.Leaders(); leaders != 1 {
		t.Errorf("expected 1 leader, got %d", leaders)
	}
	if coalesced := g.Coalesced(); coalesced != numSubagents-1 {
		t.Errorf("expected %d coalesced followers, got %d", numSubagents-1, coalesced)
	}

	// Invariant 3: All subagents received bit-identical valid outputs
	var leaderLogits []float32
	var leaderIdx int
	for i, r := range results {
		if r.err != nil {
			t.Fatalf("subagent %d failed: %v", i, r.err)
		}
		if r.kv == nil || r.kv.Len() != len(prefix) {
			t.Fatalf("subagent %d got invalid kv length: %v", i, r.kv)
		}
		if r.isLeader {
			leaderIdx = i
			leaderLogits = r.logits
		}
	}

	for i, r := range results {
		if i == leaderIdx {
			continue
		}
		if d := maxAbsDiff(r.logits, leaderLogits); d > 1e-6 {
			t.Errorf("follower %d logits differ from leader: max|Δ|=%.3e", i, d)
		}
	}

	t.Logf("coalescing pass: %d subagents completed in %v; leader idx=%d, coalesced=%d",
		numSubagents, totalElapsed, leaderIdx, g.Coalesced())
}

// TestPrefixFlightGroupZeroCopySuffixForking verifies that upon completion of the shared
// prefix prefill, all subagents fork their private suffix leaf branches off the shared
// physical prefix node with zero duplicate prefix VRAM allocation.
func TestPrefixFlightGroupZeroCopySuffixForking(t *testing.T) {
	m := newSyntheticTiny()
	tree := New(0)
	g := NewPrefixFlightGroup(tree)

	const numSubagents = 10
	prefix := seq(1, 12) // common preamble

	var prefixPrefills atomic.Int64
	var startGate sync.WaitGroup
	startGate.Add(1)

	type forkResult struct {
		leaf     *node
		isLeader bool
		err      error
	}

	results := make([]forkResult, numSubagents)
	var wg sync.WaitGroup

	for i := 0; i < numSubagents; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			startGate.Wait()

			ctx := context.Background()
			suffix := []int{20 + idx*2, 21 + idx*2} // unique suffix per subagent (< vocab 64)

			leaf, isLeader, err := g.ForkSuffix(
				ctx,
				"",
				prefix,
				suffix,
				// Prefix prefill func (called only by leader)
				func(pctx context.Context) (*model.KVCache, []float32, error) {
					prefixPrefills.Add(1)
					time.Sleep(20 * time.Millisecond)
					sess := m.NewSession()
					l := sess.Prefill(prefix)
					return sess.Cache, l, nil
				},
				// Suffix prefill func (called by each subagent on the warm prefix node)
				func(sctx context.Context, prefixNode *node) (*model.KVCache, []float32, error) {
					sess := m.SessionFromPrefix(prefixNode.KV())
					l := sess.Prefill(suffix)
					return sess.Cache, l, nil
				},
			)

			results[idx] = forkResult{leaf: leaf, isLeader: isLeader, err: err}
		}(i)
	}

	startGate.Done()
	wg.Wait()

	// Invariant 1: Shared prefix was prefilled exactly once
	if count := prefixPrefills.Load(); count != 1 {
		t.Fatalf("expected 1 prefix prefill, got %d", count)
	}

	// Invariant 2: All subagent leaves share the SAME prefix node as parent
	var sharedPrefixNode *node
	for i, r := range results {
		if r.err != nil {
			t.Fatalf("subagent %d failed: %v", i, r.err)
		}
		if r.leaf == nil {
			t.Fatalf("subagent %d returned nil leaf", i)
		}
		parent := r.leaf.parent
		if parent == nil {
			t.Fatalf("subagent %d leaf has nil parent", i)
		}
		if sharedPrefixNode == nil {
			sharedPrefixNode = parent
		} else if parent != sharedPrefixNode {
			t.Fatalf("subagent %d leaf attached to different parent node %p != %p", i, parent, sharedPrefixNode)
		}
	}

	// Invariant 3: Shared prefix node has len(children) == numSubagents
	if numChildren := len(sharedPrefixNode.children); numChildren != numSubagents {
		t.Fatalf("shared prefix node has %d children, want %d", numChildren, numSubagents)
	}

	// Invariant 4: Shared prefix node length equals prefix length
	if sharedPrefixNode.plen != len(prefix) {
		t.Fatalf("shared prefix plen=%d, want %d", sharedPrefixNode.plen, len(prefix))
	}

	// Invariant 5: Each leaf length equals prefix + suffix
	for i, r := range results {
		if r.leaf.plen != len(prefix)+2 {
			t.Errorf("leaf %d plen=%d, want %d", i, r.leaf.plen, len(prefix)+2)
		}
		// Clean up leaf leases
		g.Done(r.leaf)
	}

	t.Logf("zero-copy suffix forking: %d private leaves attached to single shared prefix node %p",
		numSubagents, sharedPrefixNode)
}

// TestPrefixFlightGroupErrorHandling verifies that when a leader's prefill fails,
// all waiting followers wake up immediately and receive the error without hanging,
// the in-flight node is cleaned up, and subsequent requests can retry and succeed.
func TestPrefixFlightGroupErrorHandling(t *testing.T) {
	tree := New(0)
	g := NewPrefixFlightGroup(tree)

	prefix := seq(10, 8)
	simulatedErr := errors.New("simulated hardware out-of-memory error")

	const numSubagents = 10
	var startGate sync.WaitGroup
	startGate.Add(1)

	var wg sync.WaitGroup
	errs := make([]error, numSubagents)

	for i := 0; i < numSubagents; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			startGate.Wait()

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			_, _, _, err := g.Coalesce(ctx, prefix, func(pctx context.Context) (*model.KVCache, []float32, error) {
				time.Sleep(25 * time.Millisecond)
				return nil, nil, simulatedErr
			})
			errs[idx] = err
		}(i)
	}

	startGate.Done()
	wg.Wait()

	// All subagents (leader and followers) must have received simulatedErr
	for i, err := range errs {
		if err == nil {
			t.Fatalf("subagent %d did not receive error", i)
		}
		if !errors.Is(err, simulatedErr) && err.Error() != simulatedErr.Error() {
			t.Errorf("subagent %d got unexpected error: %v", i, err)
		}
	}

	// Tree state must be clean: in-flight entry was removed
	if inf := g.InFlight(); inf != 0 {
		t.Fatalf("in-flight map has %d entries, want 0", inf)
	}

	// Retry: Next request must succeed cleanly
	m := newSyntheticTiny()
	kv, _, isLeader, err := g.Coalesce(context.Background(), prefix, func(pctx context.Context) (*model.KVCache, []float32, error) {
		sess := m.NewSession()
		sess.Prefill(prefix)
		return sess.Cache, nil, nil
	})
	if err != nil {
		t.Fatalf("subsequent retry failed: %v", err)
	}
	if !isLeader {
		t.Errorf("retry should be leader, got leader=false")
	}
	if kv == nil || kv.Len() != len(prefix) {
		t.Fatalf("retry got invalid kv: %v", kv)
	}
}

// TestPrefixFlightGroupPanicRecovery verifies that if a leader panics during prefill,
// waiting followers do not panic, do not hang, and receive a descriptive error.
func TestPrefixFlightGroupPanicRecovery(t *testing.T) {
	tree := New(0)
	g := NewPrefixFlightGroup(tree)

	prefix := seq(20, 8)
	const numFollowers = 5

	var startGate sync.WaitGroup
	startGate.Add(1)

	var leaderPanicked atomic.Bool
	followerErrs := make([]error, numFollowers)

	var wg sync.WaitGroup

	// Leader goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				leaderPanicked.Store(true)
			}
		}()
		startGate.Wait()

		_, _, _, _ = g.Coalesce(context.Background(), prefix, func(pctx context.Context) (*model.KVCache, []float32, error) {
			time.Sleep(20 * time.Millisecond)
			panic("simulated kernel panic in prefill GEMM")
		})
	}()

	// Follower goroutines
	for i := 0; i < numFollowers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			startGate.Wait()
			// Short sleep to ensure leader registers first
			time.Sleep(5 * time.Millisecond)

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			_, _, _, err := g.Coalesce(ctx, prefix, func(pctx context.Context) (*model.KVCache, []float32, error) {
				return nil, nil, errors.New("follower should not execute prefill")
			})
			followerErrs[idx] = err
		}(i)
	}

	startGate.Done()
	wg.Wait()

	if !leaderPanicked.Load() {
		t.Fatalf("leader did not panic as expected")
	}

	// Followers must not have hung or panicked; they must have received the abandoned/panic error
	for i, err := range followerErrs {
		if err == nil {
			t.Fatalf("follower %d did not receive error after leader panic", i)
		}
		t.Logf("follower %d correctly received error after leader panic: %v", i, err)
	}
}

// TestPrefixFlightGroupFollowerCancellation verifies that if a follower's context
// expires or is cancelled while waiting on the leader, the follower returns ctx.Err()
// immediately without waiting for the leader or causing refcount leaks.
func TestPrefixFlightGroupFollowerCancellation(t *testing.T) {
	m := newSyntheticTiny()
	tree := New(0)
	g := NewPrefixFlightGroup(tree)

	prefix := seq(30, 8)
	var startGate sync.WaitGroup
	startGate.Add(1)

	var leaderDone sync.WaitGroup
	leaderDone.Add(1)

	// Leader with 200ms sleep
	go func() {
		defer leaderDone.Done()
		startGate.Wait()
		_, _, _, _ = g.Coalesce(context.Background(), prefix, func(pctx context.Context) (*model.KVCache, []float32, error) {
			time.Sleep(150 * time.Millisecond)
			sess := m.NewSession()
			sess.Prefill(prefix)
			return sess.Cache, nil, nil
		})
	}()

	// Follower with 20ms timeout
	followerDone := make(chan error, 1)
	go func() {
		startGate.Wait()
		time.Sleep(5 * time.Millisecond) // ensure leader registered

		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()

		t0 := time.Now()
		_, _, _, err := g.Coalesce(ctx, prefix, func(pctx context.Context) (*model.KVCache, []float32, error) {
			return nil, nil, errors.New("follower prefill should not be called")
		})
		elapsed := time.Since(t0)

		if elapsed > 100*time.Millisecond {
			followerDone <- fmt.Errorf("follower took %v, should have aborted near 20ms", elapsed)
			return
		}
		followerDone <- err
	}()

	startGate.Done()
	err := <-followerDone
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		t.Fatalf("expected deadline exceeded, got: %v", err)
	}

	leaderDone.Wait()

	// Verify leader finished cleanly and tree is consistent
	node, matched := tree.Lookup(prefix)
	if matched != len(prefix) {
		t.Fatalf("expected prefix to be cached in tree, matched=%d", matched)
	}
	tree.Done(node)
}

// TestNodeStateLifecycleAndReadyBroadcast tests direct node state transitions
// and channel synchronization primitives on the Tree and node types.
func TestNodeStateLifecycleAndReadyBroadcast(t *testing.T) {
	tree := New(0)
	prefix := seq(40, 6)

	// 1. RegisterFlight creates node in NodeComputingPrefill
	n, isLeader := tree.RegisterFlight(prefix)
	if !isLeader {
		t.Fatalf("first registration should be leader")
	}
	if !n.IsComputing() || n.State() != NodeComputingPrefill {
		t.Fatalf("expected NodeComputingPrefill state, got %v", n.State())
	}
	if n.IsWarm() {
		t.Fatalf("node should not be warm while computing")
	}

	// 2. Second registration on same prefix is a follower
	n2, isLeader2 := tree.RegisterFlight(prefix)
	if isLeader2 {
		t.Fatalf("second registration should be follower")
	}
	if n2 != n {
		t.Fatalf("follower got different node %p != %p", n2, n)
	}

	// 3. WaitReady blocks until CommitFlight
	waiterDone := make(chan error, 1)
	go func() {
		waiterDone <- n.WaitReady(context.Background())
	}()

	select {
	case <-waiterDone:
		t.Fatalf("WaitReady returned before CommitFlight")
	case <-time.After(15 * time.Millisecond):
		// Expected to still be blocked
	}

	// 4. Commit flight
	m := newSyntheticTiny()
	sess := m.NewSession()
	logits := sess.Prefill(prefix)
	tree.CommitFlight(n, sess.Cache, logits)

	select {
	case err := <-waiterDone:
		if err != nil {
			t.Fatalf("WaitReady returned error: %v", err)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatalf("WaitReady did not unblock after CommitFlight")
	}

	if !n.IsWarm() || n.State() != NodeWarm {
		t.Fatalf("expected NodeWarm state after commit, got %v", n.State())
	}
	if n.IsComputing() {
		t.Fatalf("node should not be computing after commit")
	}

	tree.Done(n)
	tree.Done(n2)
}

// TestPrefixFlightGroupRace runs high-concurrency requests across overlapping
// and disjoint prefixes under the race detector.
func TestPrefixFlightGroupRace(t *testing.T) {
	m := newSyntheticTiny()
	tree := New(1000)
	g := NewPrefixFlightGroup(tree)

	const numWorkers = 25
	const numIterations = 10

	prefixes := [][]int{
		seq(1, 8),
		seq(1, 12),
		seq(10, 8),
		seq(20, 10),
	}

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for iter := 0; iter < numIterations; iter++ {
				prefix := prefixes[(workerID+iter)%len(prefixes)]
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)

				node, isLeader, err := g.CoalescePrefix(ctx, "", prefix, func(pctx context.Context) (*model.KVCache, []float32, error) {
					sess := m.NewSession()
					l := sess.Prefill(prefix)
					return sess.Cache, l, nil
				})

				if err == nil && node != nil {
					// Read state
					_ = node.State()
					_ = node.KV()
					_ = node.Logits()
					_ = isLeader
					g.Done(node)
				}
				cancel()
			}
		}(w)
	}

	wg.Wait()
	t.Logf("race test completed: leaders=%d, coalesced=%d", g.Leaders(), g.Coalesced())
}

// TestPrefixFlightGroupStandalone verifies singleflight behavior when tree is nil.
func TestPrefixFlightGroupStandalone(t *testing.T) {
	m := newSyntheticTiny()
	g := NewPrefixFlightGroup(nil) // standalone mode

	prefix := seq(50, 8)
	const numSubagents = 8

	var prefillCount atomic.Int64
	var startGate sync.WaitGroup
	startGate.Add(1)

	var wg sync.WaitGroup
	for i := 0; i < numSubagents; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			startGate.Wait()

			kv, _, _, err := g.Coalesce(context.Background(), prefix, func(ctx context.Context) (*model.KVCache, []float32, error) {
				prefillCount.Add(1)
				time.Sleep(20 * time.Millisecond)
				sess := m.NewSession()
				sess.Prefill(prefix)
				return sess.Cache, nil, nil
			})
			if err != nil {
				t.Errorf("coalesce error: %v", err)
			}
			if kv == nil || kv.Len() != len(prefix) {
				t.Errorf("invalid kv len: %v", kv)
			}
		}()
	}

	startGate.Done()
	wg.Wait()

	if count := prefillCount.Load(); count != 1 {
		t.Fatalf("expected 1 prefill in standalone mode, got %d", count)
	}
	if coalesced := g.Coalesced(); coalesced != numSubagents-1 {
		t.Fatalf("expected %d coalesced, got %d", numSubagents-1, coalesced)
	}
}
