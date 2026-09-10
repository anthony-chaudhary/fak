package radixkv

import (
	"context"
	"errors"
	"fmt"
	"runtime"
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

type sharedPrefixFlightResult struct {
	kv      *model.KVCache
	logits  []float32
	matched int
	leader  bool
	err     error
}

func sharedPrefixTestIDs(n int, seed uint64) []int {
	ids := make([]int, n)
	state := seed + 1
	for i := range ids {
		state = state*6364136223846793005 + 1442695040888963407
		ids[i] = 1 + int(state%61)
	}
	return ids
}

func sharedPrefixTestPrompt(common []int, suffixLen, marker int, seed uint64) []int {
	suffix := sharedPrefixTestIDs(suffixLen, seed)
	suffix[0] = marker
	return append(append([]int(nil), common...), suffix...)
}

func awaitSharedPrefixCondition(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		runtime.Gosched()
	}
}

func receiveSharedPrefixResult(t *testing.T, ch <-chan sharedPrefixFlightResult) sharedPrefixFlightResult {
	t.Helper()
	select {
	case result := <-ch:
		return result
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for prefix-flight result")
		return sharedPrefixFlightResult{}
	}
}

// TestPrefixFlightGroupSharedPrefixContract is the behavioral contract for
// transient CPU prefill reuse. Distinct prompts may share only valuable work,
// and every waiter receives independently owned state.
func TestPrefixFlightGroupSharedPrefixContract(t *testing.T) {
	t.Run("distinct siblings clone a valuable prefix", func(t *testing.T) {
		m := newSyntheticTiny()
		g := NewPrefixFlightGroup(nil)
		common := sharedPrefixTestIDs(128, 1)
		prompts := [][]int{
			sharedPrefixTestPrompt(common, 80, 7, 2),
			sharedPrefixTestPrompt(common, 90, 8, 3),
			sharedPrefixTestPrompt(common, 100, 9, 4),
		}

		entered := make(chan struct{})
		release := make(chan struct{})
		leaderResult := make(chan sharedPrefixFlightResult, 1)
		var callbacks atomic.Int64
		var publishedBytes atomic.Int64
		leaderSession := m.NewSession()
		go func() {
			kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", prompts[0], 64, func(context.Context) (*model.KVCache, []float32, error) {
				callbacks.Add(1)
				close(entered)
				<-release
				logits := leaderSession.Prefill(prompts[0])
				publishedBytes.Store(leaderSession.Cache.ClonePayloadBytes())
				return leaderSession.Cache, logits, nil
			})
			if err == nil {
				leaderSession.Step(5)
				leaderSession.Cache.Truncate(1)
			}
			leaderResult <- sharedPrefixFlightResult{kv: kv, logits: logits, matched: matched, leader: leader, err: err}
		}()
		<-entered

		followers := make([]chan sharedPrefixFlightResult, 2)
		for i := range followers {
			followers[i] = make(chan sharedPrefixFlightResult, 1)
			prompt := prompts[i+1]
			go func(out chan<- sharedPrefixFlightResult) {
				kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", prompt, 64, func(context.Context) (*model.KVCache, []float32, error) {
					callbacks.Add(1)
					s := m.NewSession()
					return s.Cache, s.Prefill(prompt), nil
				})
				out <- sharedPrefixFlightResult{kv: kv, logits: logits, matched: matched, leader: leader, err: err}
			}(followers[i])
		}
		awaitSharedPrefixCondition(t, "two followers to join", func() bool { return g.Coalesced() == 2 })
		close(release)

		leader := receiveSharedPrefixResult(t, leaderResult)
		if leader.err != nil || !leader.leader || leader.matched != 0 {
			t.Fatalf("leader result=%+v, want leader with matched=0", leader)
		}
		got := make([]sharedPrefixFlightResult, 2)
		for i := range got {
			got[i] = receiveSharedPrefixResult(t, followers[i])
			if got[i].err != nil || got[i].leader || got[i].matched != len(common) {
				t.Fatalf("follower %d result=%+v, want matched=%d follower", i, got[i], len(common))
			}
			if got[i].kv == nil || got[i].kv.Len() != len(common) {
				t.Fatalf("follower %d KV len=%v, want %d", i, got[i].kv, len(common))
			}
			if got[i].logits != nil {
				t.Fatalf("follower %d received partial-prefix logits", i)
			}
			lineageSession := m.SessionFromPrefix(got[i].kv)
			if _, err := lineageSession.VerifyTokenLineage(common); err != nil {
				t.Fatalf("follower %d tail removal did not preserve token lineage: %v", i, err)
			}
		}
		if callbacks.Load() != 1 {
			t.Fatalf("prefill callbacks=%d, want 1", callbacks.Load())
		}
		bytesPerClone := publishedBytes.Load()
		if bytesPerClone <= 0 || g.ClonePayloadBytes() != 3*bytesPerClone {
			t.Fatalf("clone payload bytes=%d, want publication plus two follower clones=%d", g.ClonePayloadBytes(), 3*bytesPerClone)
		}
		got[0].kv.Truncate(64)
		if got[1].kv.Len() != len(common) {
			t.Fatalf("follower KV aliases sibling: len=%d want %d", got[1].kv.Len(), len(common))
		}
	})

	t.Run("shorter exact follower never receives longer-leader logits", func(t *testing.T) {
		m := newSyntheticTiny()
		g := NewPrefixFlightGroup(nil)
		short := sharedPrefixTestIDs(128, 5)
		long := sharedPrefixTestPrompt(short, 80, 7, 6)
		entered, release := make(chan struct{}), make(chan struct{})
		leaderDone := make(chan sharedPrefixFlightResult, 1)
		go func() {
			s := m.NewSession()
			kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", long, 64, func(context.Context) (*model.KVCache, []float32, error) {
				close(entered)
				<-release
				return s.Cache, s.Prefill(long), nil
			})
			leaderDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
		}()
		<-entered
		followerDone := make(chan sharedPrefixFlightResult, 1)
		go func() {
			kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", short, 64, func(context.Context) (*model.KVCache, []float32, error) {
				return nil, nil, errors.New("shorter follower ran callback")
			})
			followerDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
		}()
		awaitSharedPrefixCondition(t, "shorter exact follower to join", func() bool { return g.Coalesced() == 1 })
		close(release)
		if result := receiveSharedPrefixResult(t, leaderDone); result.err != nil {
			t.Fatal(result.err)
		}
		result := receiveSharedPrefixResult(t, followerDone)
		if result.err != nil || result.leader || result.matched != len(short) || result.kv == nil || result.kv.Len() != len(short) {
			t.Fatalf("shorter follower result leader=%v matched=%d kv=%v err=%v", result.leader, result.matched, result.kv, result.err)
		}
		if result.logits != nil {
			t.Fatal("shorter exact follower received logits for the longer leader prompt")
		}
		if _, err := m.SessionFromPrefix(result.kv).VerifyTokenLineage(short); err != nil {
			t.Fatalf("shorter follower lineage: %v", err)
		}
	})

	t.Run("recurrent partial tail falls back cold", func(t *testing.T) {
		cfg := model.Config{
			HiddenSize: 32, NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
			IntermediateSize: 64, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: -1,
			LayerTypes:          []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
			LinearConvKernelDim: 3, LinearKeyHeadDim: 8, LinearNumKeyHeads: 2,
			LinearValueHeadDim: 8, LinearNumValueHeads: 4, AttnOutputGate: true,
			FullAttentionInterval: 4, NormGain1p: true, TieWordEmbeddings: true,
		}
		m := model.NewSynthetic(cfg)
		g := NewPrefixFlightGroup(nil)
		common := sharedPrefixTestIDs(128, 7)
		long := sharedPrefixTestPrompt(common, 80, 7, 8)
		follower := sharedPrefixTestPrompt(common, 90, 8, 9)
		entered, release := make(chan struct{}), make(chan struct{})
		leaderDone := make(chan sharedPrefixFlightResult, 1)
		go func() {
			s := m.NewSession()
			kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", long, 64, func(context.Context) (*model.KVCache, []float32, error) {
				close(entered)
				<-release
				return s.Cache, s.Prefill(long), nil
			})
			leaderDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
		}()
		<-entered
		followerDone := make(chan sharedPrefixFlightResult, 1)
		go func() {
			kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", follower, 64, func(context.Context) (*model.KVCache, []float32, error) {
				return nil, nil, errors.New("joined recurrent follower ran callback")
			})
			followerDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
		}()
		awaitSharedPrefixCondition(t, "recurrent follower to join", func() bool { return g.Coalesced() == 1 })
		close(release)
		if result := receiveSharedPrefixResult(t, leaderDone); result.err != nil {
			t.Fatal(result.err)
		}
		result := receiveSharedPrefixResult(t, followerDone)
		if result.err != nil || result.leader || result.matched != 0 || result.kv != nil || result.logits != nil {
			t.Fatalf("unsupported recurrent partial restore did not fall back cold: leader=%v matched=%d kv=%v logits=%d err=%v", result.leader, result.matched, result.kv, len(result.logits), result.err)
		}
	})

	t.Run("admission floor and saved-work ratio", func(t *testing.T) {
		for _, tc := range []struct {
			name       string
			commonLen  int
			leaderTail int
		}{
			{name: "below floor", commonLen: 63, leaderTail: 32},
			{name: "leader tail exceeds common", commonLen: 128, leaderTail: 129},
		} {
			t.Run(tc.name, func(t *testing.T) {
				m := newSyntheticTiny()
				g := NewPrefixFlightGroup(nil)
				common := sharedPrefixTestIDs(tc.commonLen, 10)
				a := sharedPrefixTestPrompt(common, tc.leaderTail, 7, 11)
				b := sharedPrefixTestPrompt(common, 70, 8, 12)
				entered, release := make(chan struct{}), make(chan struct{})
				leaderDone := make(chan sharedPrefixFlightResult, 1)
				go func() {
					kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", a, 64, func(context.Context) (*model.KVCache, []float32, error) {
						close(entered)
						<-release
						s := m.NewSession()
						return s.Cache, s.Prefill(a), nil
					})
					leaderDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
				}()
				<-entered
				s := m.NewSession()
				kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", b, 64, func(context.Context) (*model.KVCache, []float32, error) {
					return s.Cache, s.Prefill(b), nil
				})
				if err != nil || !leader || matched != 0 || kv == nil || logits == nil {
					t.Fatalf("low-value request unexpectedly followed: leader=%v matched=%d err=%v", leader, matched, err)
				}
				if g.Coalesced() != 0 {
					t.Fatalf("low-value coalesced=%d, want 0", g.Coalesced())
				}
				close(release)
				if result := receiveSharedPrefixResult(t, leaderDone); result.err != nil {
					t.Fatal(result.err)
				}
			})
		}
	})

	t.Run("rank saved work before raw common length", func(t *testing.T) {
		m := newSyntheticTiny()
		g := NewPrefixFlightGroup(nil)
		base := sharedPrefixTestIDs(120, 20)
		extra := sharedPrefixTestIDs(8, 21)
		longCommon := append(append([]int(nil), base...), extra...)
		longLeader := sharedPrefixTestPrompt(longCommon, 128, 7, 22)
		shortLeader := append([]int(nil), base...)
		follower := sharedPrefixTestPrompt(longCommon, 72, 9, 23)
		releaseLong, releaseShort := make(chan struct{}), make(chan struct{})
		enteredLong, enteredShort := make(chan struct{}), make(chan struct{})
		done := make(chan sharedPrefixFlightResult, 2)
		start := func(tokens []int, entered chan struct{}, release chan struct{}) {
			go func() {
				kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", tokens, 64, func(context.Context) (*model.KVCache, []float32, error) {
					close(entered)
					<-release
					s := m.NewSession()
					return s.Cache, s.Prefill(tokens), nil
				})
				done <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
			}()
		}
		start(longLeader, enteredLong, releaseLong)
		<-enteredLong
		start(shortLeader, enteredShort, releaseShort)
		<-enteredShort
		followerDone := make(chan sharedPrefixFlightResult, 1)
		go func() {
			s := m.NewSession()
			kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", follower, 64, func(context.Context) (*model.KVCache, []float32, error) {
				return s.Cache, s.Prefill(follower), nil
			})
			followerDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
		}()
		awaitSharedPrefixCondition(t, "ranked follower to join", func() bool { return g.Coalesced() == 1 })
		close(releaseShort)
		result := receiveSharedPrefixResult(t, followerDone)
		if result.err != nil || result.leader || result.matched != len(base) {
			t.Fatalf("ranked result leader=%v matched=%d err=%v, want shorter 120-token flight with greater saved-work surplus", result.leader, result.matched, result.err)
		}
		close(releaseLong)
		for range 2 {
			if result := receiveSharedPrefixResult(t, done); result.err != nil {
				t.Fatal(result.err)
			}
		}
	})

	t.Run("namespace cancellation and failed leader cleanup", func(t *testing.T) {
		m := newSyntheticTiny()
		common := sharedPrefixTestIDs(128, 30)
		a := sharedPrefixTestPrompt(common, 80, 7, 31)
		b := sharedPrefixTestPrompt(common, 90, 8, 32)

		t.Run("namespace isolation", func(t *testing.T) {
			g := NewPrefixFlightGroup(nil)
			entered, release := make(chan struct{}), make(chan struct{})
			done := make(chan sharedPrefixFlightResult, 1)
			go func() {
				kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", a, 64, func(context.Context) (*model.KVCache, []float32, error) {
					close(entered)
					<-release
					s := m.NewSession()
					return s.Cache, s.Prefill(a), nil
				})
				done <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
			}()
			<-entered
			s := m.NewSession()
			_, _, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/b", b, 64, func(context.Context) (*model.KVCache, []float32, error) {
				return s.Cache, s.Prefill(b), nil
			})
			if err != nil || !leader || matched != 0 || g.Coalesced() != 0 {
				t.Fatalf("cross-namespace request joined: leader=%v matched=%d coalesced=%d err=%v", leader, matched, g.Coalesced(), err)
			}
			close(release)
			if result := receiveSharedPrefixResult(t, done); result.err != nil {
				t.Fatal(result.err)
			}
		})

		t.Run("follower cancellation", func(t *testing.T) {
			g := NewPrefixFlightGroup(nil)
			entered, release := make(chan struct{}), make(chan struct{})
			leaderDone := make(chan sharedPrefixFlightResult, 1)
			go func() {
				kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", a, 64, func(context.Context) (*model.KVCache, []float32, error) {
					close(entered)
					<-release
					s := m.NewSession()
					return s.Cache, s.Prefill(a), nil
				})
				leaderDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
			}()
			<-entered
			ctx, cancel := context.WithCancel(context.Background())
			followerDone := make(chan sharedPrefixFlightResult, 1)
			go func() {
				kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(ctx, "tenant/a", b, 64, func(context.Context) (*model.KVCache, []float32, error) {
					return nil, nil, errors.New("cancelled follower ran callback")
				})
				followerDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
			}()
			awaitSharedPrefixCondition(t, "cancellable follower to join", func() bool { return g.Coalesced() == 1 })
			cancel()
			if result := receiveSharedPrefixResult(t, followerDone); !errors.Is(result.err, context.Canceled) {
				t.Fatalf("follower cancellation err=%v", result.err)
			}
			close(release)
			if result := receiveSharedPrefixResult(t, leaderDone); result.err != nil {
				t.Fatal(result.err)
			}
			if g.InFlight() != 0 {
				t.Fatalf("in-flight entries=%d after cancellation", g.InFlight())
			}
		})

		t.Run("cancel before call and ready race avoid clone", func(t *testing.T) {
			g := NewPrefixFlightGroup(nil)
			cancelled, cancel := context.WithCancel(context.Background())
			cancel()
			var callbacks atomic.Int64
			if kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(cancelled, "tenant/a", a, 64, func(context.Context) (*model.KVCache, []float32, error) {
				callbacks.Add(1)
				return nil, nil, nil
			}); !errors.Is(err, context.Canceled) || kv != nil || logits != nil || matched != 0 || leader || callbacks.Load() != 0 || g.InFlight() != 0 {
				t.Fatalf("pre-cancel result leader=%v matched=%d kv=%v logits=%d callbacks=%d in_flight=%d err=%v", leader, matched, kv, len(logits), callbacks.Load(), g.InFlight(), err)
			}

			entered, release := make(chan struct{}), make(chan struct{})
			leaderDone := make(chan sharedPrefixFlightResult, 1)
			go func() {
				s := m.NewSession()
				kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", a, 64, func(context.Context) (*model.KVCache, []float32, error) {
					close(entered)
					<-release
					logits := s.Prefill(a)
					return s.Cache, logits, nil
				})
				leaderDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
			}()
			<-entered
			ctx, cancelRace := context.WithCancel(context.Background())
			followerDone := make(chan sharedPrefixFlightResult, 1)
			go func() {
				kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(ctx, "tenant/a", b, 64, func(context.Context) (*model.KVCache, []float32, error) {
					return nil, nil, errors.New("cancelled ready-race follower ran callback")
				})
				followerDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
			}()
			awaitSharedPrefixCondition(t, "ready-race follower to join", func() bool { return g.Coalesced() == 1 })
			cancelRace()
			close(release)
			if result := receiveSharedPrefixResult(t, followerDone); !errors.Is(result.err, context.Canceled) {
				t.Fatalf("ready-race follower err=%v", result.err)
			}
			if result := receiveSharedPrefixResult(t, leaderDone); result.err != nil {
				t.Fatal(result.err)
			}
			if got := g.ClonePayloadBytes(); got != 0 {
				t.Fatalf("all followers cancelled before completion but flight cloned %d payload bytes", got)
			}
		})

		for _, failure := range []struct {
			name string
			run  func() error
		}{
			{name: "error", run: func() error { return errors.New("leader failed") }},
			{name: "cancelled", run: func() error { return context.Canceled }},
		} {
			t.Run("leader "+failure.name+" permits one retry", func(t *testing.T) {
				g := NewPrefixFlightGroup(nil)
				entered, release := make(chan struct{}), make(chan struct{})
				leaderDone := make(chan sharedPrefixFlightResult, 1)
				go func() {
					kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", a, 64, func(context.Context) (*model.KVCache, []float32, error) {
						close(entered)
						<-release
						return nil, nil, failure.run()
					})
					leaderDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
				}()
				<-entered
				var retries atomic.Int64
				followerDone := make(chan sharedPrefixFlightResult, 1)
				go func() {
					kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", b, 64, func(context.Context) (*model.KVCache, []float32, error) {
						retries.Add(1)
						s := m.NewSession()
						return s.Cache, s.Prefill(b), nil
					})
					followerDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
				}()
				awaitSharedPrefixCondition(t, "retrying follower to join", func() bool { return g.Coalesced() == 1 })
				close(release)
				if result := receiveSharedPrefixResult(t, leaderDone); result.err == nil {
					t.Fatal("failed leader returned nil error")
				}
				result := receiveSharedPrefixResult(t, followerDone)
				if result.err != nil || !result.leader || result.matched != 0 || retries.Load() != 1 {
					t.Fatalf("retry result leader=%v matched=%d retries=%d err=%v", result.leader, result.matched, retries.Load(), result.err)
				}
				if g.InFlight() != 0 {
					t.Fatalf("in-flight entries=%d after retry", g.InFlight())
				}
			})
		}

		t.Run("leader panic drains flight", func(t *testing.T) {
			g := NewPrefixFlightGroup(nil)
			entered, release := make(chan struct{}), make(chan struct{})
			leaderRecovered := make(chan any, 1)
			go func() {
				defer func() { leaderRecovered <- recover() }()
				_, _, _, _, _ = g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", a, 64, func(context.Context) (*model.KVCache, []float32, error) {
					close(entered)
					<-release
					panic("leader panic")
				})
			}()
			<-entered
			followerDone := make(chan sharedPrefixFlightResult, 1)
			go func() {
				s := m.NewSession()
				kv, logits, matched, leader, err := g.CoalesceSharedPrefixNS(context.Background(), "tenant/a", b, 64, func(context.Context) (*model.KVCache, []float32, error) {
					return s.Cache, s.Prefill(b), nil
				})
				followerDone <- sharedPrefixFlightResult{kv, logits, matched, leader, err}
			}()
			awaitSharedPrefixCondition(t, "panic follower to join", func() bool { return g.Coalesced() == 1 })
			close(release)
			if recovered := <-leaderRecovered; recovered == nil {
				t.Fatal("leader panic was not propagated")
			}
			result := receiveSharedPrefixResult(t, followerDone)
			if result.err != nil || !result.leader || result.matched != 0 {
				t.Fatalf("post-panic retry leader=%v matched=%d err=%v", result.leader, result.matched, result.err)
			}
			if g.InFlight() != 0 {
				t.Fatalf("in-flight entries=%d after panic", g.InFlight())
			}
		})
	})
}
