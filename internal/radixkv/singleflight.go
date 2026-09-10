package radixkv

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func closeReady(ch chan struct{}) {
	if ch == nil {
		return
	}
	select {
	case <-ch:
	default:
		close(ch)
	}
}

// In-flight promise errors.
var (
	// ErrFlightAbandoned is returned to followers when a leader fails or panics
	// without committing its prefill result.
	ErrFlightAbandoned = errors.New("radixkv: in-flight prefill abandoned")
	// ErrFlightFailed is returned when a leader explicitly returns an execution error.
	ErrFlightFailed = errors.New("radixkv: in-flight prefill failed")
)

// PrefillFunc computes the KV cache and optional logits for a given prompt prefix.
type PrefillFunc func(ctx context.Context) (*model.KVCache, []float32, error)

// flightPromise represents an in-flight prefix prefill promise being computed by a leader.
type flightPromise struct {
	node      *node
	ready     chan struct{}
	kv        *model.KVCache
	logits    []float32
	err       error
	followers int
	waiters   int
	namespace string
	tokens    []int
	sequence  uint64
}

// PrefixFlightGroup coordinates concurrent exact-prefix and value-qualified
// shared-prefix prefills. Tree-backed exact-prefix methods retain node leases;
// standalone shared-prefix followers receive independently owned KV clones and
// account for their copy cost. Leader failure or panic wakes every follower.
type PrefixFlightGroup struct {
	mu         sync.Mutex
	treeMu     sync.Mutex
	tree       *Tree
	lock       sync.Locker
	flights    map[string]*flightPromise
	shared     map[uint64]*flightPromise
	nextSeq    uint64
	coalesced  atomic.Int64
	leaders    atomic.Int64
	cloneBytes atomic.Int64
}

// NewPrefixFlightGroup creates a new prefix singleflight group attached to the given tree.
func NewPrefixFlightGroup(tree *Tree) *PrefixFlightGroup {
	return &PrefixFlightGroup{
		tree:    tree,
		flights: make(map[string]*flightPromise),
	}
}

// NewPrefixFlightGroupWithLocker creates a new prefix singleflight group with an external
// locker to serialize tree operations with other access paths (e.g. ScopedTree).
func NewPrefixFlightGroupWithLocker(tree *Tree, lock sync.Locker) *PrefixFlightGroup {
	return &PrefixFlightGroup{
		tree:    tree,
		lock:    lock,
		flights: make(map[string]*flightPromise),
	}
}

func (g *PrefixFlightGroup) treeLock() {
	if g.lock != nil {
		g.lock.Lock()
	} else {
		g.treeMu.Lock()
	}
}

func (g *PrefixFlightGroup) treeUnlock() {
	if g.lock != nil {
		g.lock.Unlock()
	} else {
		g.treeMu.Unlock()
	}
}

// prefixFlightKey encodes namespace and tokens into a binary collision-free key.
func prefixFlightKey(ns string, tokens []int) string {
	buf := make([]byte, 4+len(ns)+1+len(tokens)*4)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(len(ns)))
	copy(buf[4:4+len(ns)], ns)
	offset := 4 + len(ns)
	buf[offset] = ':'
	offset++
	for _, tok := range tokens {
		binary.LittleEndian.PutUint32(buf[offset:], uint32(tok))
		offset += 4
	}
	return string(buf)
}

// Coalesce executes prefill for prefix under the default namespace, or reuses an existing
// or in-flight computation. Followers receive an independently cloneable KV cache.
// leader reports whether this caller executed fn.
func (g *PrefixFlightGroup) Coalesce(ctx context.Context, prefix []int, fn PrefillFunc) (*model.KVCache, []float32, bool, error) {
	return g.CoalesceNS(ctx, "", prefix, fn)
}

// CoalesceNS executes prefill for prefix under namespace ns, or reuses an existing
// or in-flight computation. Followers receive an independently cloneable KV cache.
// leader reports whether this caller executed fn.
func (g *PrefixFlightGroup) CoalesceNS(ctx context.Context, ns string, prefix []int, fn PrefillFunc) (*model.KVCache, []float32, bool, error) {
	if len(prefix) == 0 {
		return nil, nil, false, nil
	}

	if g.tree == nil {
		key := prefixFlightKey(ns, prefix)
		g.mu.Lock()
		if g.flights == nil {
			g.flights = make(map[string]*flightPromise)
		}
		if inflight, ok := g.flights[key]; ok {
			inflight.followers++
			g.coalesced.Add(1)
			ready := inflight.ready
			g.mu.Unlock()

			select {
			case <-ctx.Done():
				return nil, nil, false, ctx.Err()
			case <-ready:
				if inflight.err != nil {
					return nil, nil, false, inflight.err
				}
				kv := inflight.kv
				if kv != nil {
					kv = kv.Clone()
				}
				return kv, inflight.logits, false, nil
			}
		}

		f := &flightPromise{
			ready: make(chan struct{}),
			err:   ErrFlightAbandoned,
		}
		g.flights[key] = f
		g.leaders.Add(1)
		g.mu.Unlock()

		var (
			kv       *model.KVCache
			logits   []float32
			execErr  error
			panicked = true
		)
		defer func() {
			if panicked {
				r := recover()
				err := fmt.Errorf("radixkv: leader prefill panicked: %v", r)
				g.mu.Lock()
				f.err = err
				delete(g.flights, key)
				closeReady(f.ready)
				g.mu.Unlock()
				panic(r)
			}
		}()

		kv, logits, execErr = fn(ctx)
		panicked = false

		g.mu.Lock()
		delete(g.flights, key)
		f.err = execErr
		f.kv = kv
		f.logits = logits
		closeReady(f.ready)
		g.mu.Unlock()

		if execErr != nil {
			return nil, nil, true, execErr
		}
		return kv, logits, true, nil
	}

	node, leader, err := g.CoalescePrefix(ctx, ns, prefix, fn)
	if err != nil {
		return nil, nil, leader, err
	}
	if node == nil {
		return nil, nil, leader, nil
	}
	kv := node.CloneKV()
	logits := node.Logits()
	g.Done(node)
	return kv, logits, leader, nil
}

// CoalesceSharedPrefixNS coalesces a cold prefill with a concurrent request in
// the same namespace when the shared work justifies waiting for the leader's
// divergent suffix. Unlike CoalesceNS, prompts need not be identical.
//
// A follower joins only when the common prefix is at least minShared tokens and
// the leader-only suffix is no longer than that common prefix. Among eligible
// leaders it prefers the greatest common-minus-divergent surplus, then the
// longer common prefix, then the earlier registration. The leader receives the
// live result returned by fn with matched == 0. A follower receives an
// independently owned KV clone truncated to matched tokens. Partial matches
// never return logits; exact matches receive a logits copy.
func (g *PrefixFlightGroup) CoalesceSharedPrefixNS(ctx context.Context, ns string, tokens []int, minShared int, fn PrefillFunc) (*model.KVCache, []float32, int, bool, error) {
	if len(tokens) == 0 {
		return nil, nil, 0, false, nil
	}
	if minShared < 1 {
		minShared = 1
	}

	retries := 0
	for {
		if err := ctx.Err(); err != nil {
			return nil, nil, 0, false, err
		}
		g.mu.Lock()
		if g.shared == nil {
			g.shared = make(map[uint64]*flightPromise)
		}
		candidate, matched := g.bestSharedPrefixFlightLocked(ns, tokens, minShared)
		if candidate == nil {
			g.nextSeq++
			candidate = &flightPromise{
				ready:     make(chan struct{}),
				err:       ErrFlightAbandoned,
				namespace: ns,
				tokens:    append([]int(nil), tokens...),
				sequence:  g.nextSeq,
			}
			g.shared[candidate.sequence] = candidate
			g.leaders.Add(1)
			g.mu.Unlock()

			kv, logits, err := g.runSharedPrefixLeader(ctx, candidate, fn)
			return kv, logits, 0, true, err
		}

		candidate.followers++
		candidate.waiters++
		g.coalesced.Add(1)
		ready := candidate.ready
		g.mu.Unlock()

		cancelled := false
		select {
		case <-ctx.Done():
			cancelled = true
		case <-ready:
		}
		g.releaseSharedPrefixWaiter(candidate)
		if cancelled {
			return nil, nil, 0, false, ctx.Err()
		}
		if err := ctx.Err(); err != nil {
			return nil, nil, 0, false, err
		}

		if candidate.err != nil {
			if retries == 0 && ctx.Err() == nil {
				retries++
				continue
			}
			return nil, nil, 0, false, candidate.err
		}
		if candidate.kv == nil || candidate.kv.Len() < matched {
			return nil, nil, 0, false, fmt.Errorf("%w: incomplete shared-prefix KV", ErrFlightFailed)
		}
		// A recurrent cache cannot discard the leader-only suffix safely. Fail
		// open to the caller's cold path without claiming a realized match.
		if matched < candidate.kv.Len() && candidate.kv.CanEvict() != nil {
			return nil, nil, 0, false, nil
		}

		kv := candidate.kv.Clone()
		g.cloneBytes.Add(kv.ClonePayloadBytes())
		if matched < kv.Len() {
			want := kv.Len() - matched
			removed, evictErr := kv.TryEvict(matched, want)
			if evictErr != nil || removed != want {
				return nil, nil, 0, false, nil
			}
		}
		var logits []float32
		if matched == len(tokens) && matched == len(candidate.tokens) {
			logits = append([]float32(nil), candidate.logits...)
		}
		return kv, logits, matched, false, nil
	}
}

func (g *PrefixFlightGroup) bestSharedPrefixFlightLocked(ns string, tokens []int, minShared int) (*flightPromise, int) {
	var best *flightPromise
	bestMatched, bestSurplus := 0, 0
	for _, candidate := range g.shared {
		if candidate == nil || candidate.namespace != ns {
			continue
		}
		matched := commonTokenPrefix(tokens, candidate.tokens)
		leaderSuffix := len(candidate.tokens) - matched
		if matched < minShared || leaderSuffix > matched {
			continue
		}
		surplus := matched - leaderSuffix
		if best == nil || surplus > bestSurplus ||
			(surplus == bestSurplus && matched > bestMatched) ||
			(surplus == bestSurplus && matched == bestMatched && candidate.sequence < best.sequence) {
			best, bestMatched, bestSurplus = candidate, matched, surplus
		}
	}
	return best, bestMatched
}

func commonTokenPrefix(a, b []int) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func (g *PrefixFlightGroup) runSharedPrefixLeader(ctx context.Context, f *flightPromise, fn PrefillFunc) (kv *model.KVCache, logits []float32, err error) {
	completed := false
	defer func() {
		if completed {
			return
		}
		r := recover()
		panicErr := fmt.Errorf("radixkv: leader prefill panicked: %v", r)
		g.finishSharedPrefixFlight(f, nil, nil, panicErr)
		panic(r)
	}()

	kv, logits, err = fn(ctx)
	if err == nil && (kv == nil || kv.Len() != len(f.tokens)) {
		err = fmt.Errorf("%w: leader returned incomplete KV", ErrFlightFailed)
	}
	if err != nil {
		g.finishSharedPrefixFlight(f, nil, nil, err)
		completed = true
		return nil, nil, err
	}

	// Avoid the publication clone for an uncontended cold request. Once a
	// follower exists, clone before waking it or returning control to leader
	// decode, so later mutation of the live session cannot affect the handoff.
	g.mu.Lock()
	waiters := f.waiters
	if waiters == 0 {
		if current := g.shared[f.sequence]; current == f {
			delete(g.shared, f.sequence)
		}
		f.kv = nil
		f.logits = nil
		f.err = nil
		closeReady(f.ready)
		g.mu.Unlock()
		completed = true
		return kv, logits, nil
	}
	g.mu.Unlock()
	var publishedKV *model.KVCache
	var publishedLogits []float32
	publishedKV = kv.Clone()
	g.cloneBytes.Add(publishedKV.ClonePayloadBytes())
	publishedLogits = append([]float32(nil), logits...)
	g.finishSharedPrefixFlight(f, publishedKV, publishedLogits, nil)
	completed = true
	return kv, logits, nil
}

func (g *PrefixFlightGroup) finishSharedPrefixFlight(f *flightPromise, kv *model.KVCache, logits []float32, err error) {
	g.mu.Lock()
	if current := g.shared[f.sequence]; current == f {
		delete(g.shared, f.sequence)
	}
	f.kv = kv
	f.logits = logits
	f.err = err
	closeReady(f.ready)
	g.mu.Unlock()
}

func (g *PrefixFlightGroup) releaseSharedPrefixWaiter(f *flightPromise) {
	g.mu.Lock()
	f.waiters--
	if f.waiters < 0 {
		g.mu.Unlock()
		panic("radixkv: shared-prefix waiter released more than once")
	}
	g.mu.Unlock()
}

// CoalescePrefix looks up prefix in the attached tree under namespace ns.
// If resident in NodeWarm state, it returns the node immediately.
// If an in-flight computation is active, it joins as a follower and waits on completion.
// If not resident and not in flight, this caller becomes Leader, attaches an in-flight
// NodeComputingPrefill leaf, executes fn, and commits the result to the tree.
// The returned node has an active lease (refs > 0); the caller must eventually call g.Done(node).
func (g *PrefixFlightGroup) CoalescePrefix(ctx context.Context, ns string, prefix []int, fn PrefillFunc) (*node, bool, error) {
	if len(prefix) == 0 {
		return nil, false, nil
	}

	key := prefixFlightKey(ns, prefix)

	g.mu.Lock()
	if g.flights == nil {
		g.flights = make(map[string]*flightPromise)
	}

	// 1. Check if the prefix is already warm in the tree.
	if g.tree != nil {
		g.treeLock()
		boundary, matched := g.tree.LookupNS(ns, prefix)
		g.treeUnlock()
		if matched == len(prefix) && boundary.IsWarm() {
			g.mu.Unlock()
			return boundary, false, nil
		}
		if boundary != nil {
			g.treeLock()
			g.tree.Done(boundary)
			g.treeUnlock()
		}
	}

	// 2. Check if an execution is already in flight.
	if inflight, ok := g.flights[key]; ok {
		inflight.followers++
		g.coalesced.Add(1)
		ready := inflight.ready
		n := inflight.node
		if n != nil {
			g.treeLock()
			n.refs++
			g.treeUnlock()
		}
		g.mu.Unlock()

		select {
		case <-ctx.Done():
			if n != nil {
				g.treeLock()
				g.tree.Done(n)
				g.treeUnlock()
			}
			return nil, false, ctx.Err()
		case <-ready:
			if inflight.err != nil {
				if n != nil {
					g.treeLock()
					g.tree.Done(n)
					g.treeUnlock()
				}
				return nil, false, inflight.err
			}
			return n, false, nil
		}
	}

	// 3. Leader registration.
	f := &flightPromise{
		ready: make(chan struct{}),
		err:   ErrFlightAbandoned,
	}
	g.flights[key] = f
	g.leaders.Add(1)

	var computingNode *node
	if g.tree != nil {
		g.treeLock()
		root := g.tree.rootFor(ns)
		boundary, matched := g.tree.boundaryFor(root, prefix)
		suffix := prefix[matched:]
		if len(suffix) > 0 {
			computingNode = g.tree.attachComputingLeaf(boundary, suffix, f.ready)
		} else {
			boundary.SetState(NodeComputingPrefill)
			boundary.ready = f.ready
			boundary.refs++
			computingNode = boundary
		}
		g.treeUnlock()
		f.node = computingNode
	}
	g.mu.Unlock()

	// 4. Leader execution with panic safety.
	var (
		kv       *model.KVCache
		logits   []float32
		execErr  error
		panicked = true
	)

	defer func() {
		if panicked {
			r := recover()
			err := fmt.Errorf("radixkv: leader prefill panicked: %v", r)
			g.mu.Lock()
			f.err = err
			delete(g.flights, key)
			if f.node != nil && g.tree != nil {
				g.treeLock()
				g.tree.AbortFlight(f.node, err)
				g.treeUnlock()
			}
			closeReady(f.ready)
			g.mu.Unlock()
			panic(r)
		}
	}()

	kv, logits, execErr = fn(ctx)
	panicked = false

	g.mu.Lock()
	delete(g.flights, key)
	if execErr != nil {
		f.err = execErr
		if f.node != nil && g.tree != nil {
			g.treeLock()
			g.tree.AbortFlight(f.node, execErr)
			g.treeUnlock()
		}
	} else {
		f.err = nil
		f.kv = kv
		f.logits = logits
		if f.node != nil && g.tree != nil {
			g.treeLock()
			g.tree.CommitFlight(f.node, kv, logits)
			g.treeUnlock()
		}
	}
	closeReady(f.ready)
	g.mu.Unlock()

	if execErr != nil {
		return nil, true, execErr
	}
	return f.node, true, nil
}

// ForkSuffix coalesces the prefix prefill among concurrent subagents, calls suffixFn
// with the warm prefix node to compute the suffix KV cache, and attaches a private suffix
// leaf branch under the shared prefix node.
//
// All subagents share the single physical prefix node in the radix tree, achieving zero
// duplicate GEMM prefill and zero duplicate prefix VRAM allocation.
// The caller must call g.Done(leaf) when finished with the returned leaf.
func (g *PrefixFlightGroup) ForkSuffix(
	ctx context.Context,
	ns string,
	prefix []int,
	suffix []int,
	prefixFn PrefillFunc,
	suffixFn func(ctx context.Context, prefixNode *node) (*model.KVCache, []float32, error),
) (leaf *node, leader bool, err error) {
	pNode, isLeader, err := g.CoalescePrefix(ctx, ns, prefix, prefixFn)
	if err != nil {
		return nil, isLeader, err
	}

	if len(suffix) == 0 {
		return pNode, isLeader, nil
	}
	defer g.Done(pNode)

	fullKV, logits, err := suffixFn(ctx, pNode)
	if err != nil {
		return nil, isLeader, err
	}

	g.mu.Lock()
	g.treeLock()
	leaf = g.tree.InsertWithLogits(pNode, suffix, fullKV, logits)
	g.treeUnlock()
	g.mu.Unlock()

	return leaf, isLeader, nil
}

// Done releases an active lease on a node returned by CoalescePrefix or ForkSuffix.
func (g *PrefixFlightGroup) Done(n *node) {
	if n == nil || g.tree == nil {
		return
	}
	g.treeLock()
	g.tree.Done(n)
	g.treeUnlock()
}

// Coalesced reports the total number of followers that joined an in-flight prefill
// rather than duplicating GEMM computation.
func (g *PrefixFlightGroup) Coalesced() int64 {
	return g.coalesced.Load()
}

// Leaders reports the total number of subagents that executed as prefill leaders.
func (g *PrefixFlightGroup) Leaders() int64 {
	return g.leaders.Load()
}

// ClonePayloadBytes reports the cumulative KV payload bytes copied for shared-
// prefix publication and follower handoff over this group's lifetime.
func (g *PrefixFlightGroup) ClonePayloadBytes() int64 {
	return g.cloneBytes.Load()
}

// InFlight returns the number of prefix prefill operations currently active.
func (g *PrefixFlightGroup) InFlight() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.flights) + len(g.shared)
}

// Tree returns the underlying radix tree, or nil if none was configured.
func (g *PrefixFlightGroup) Tree() *Tree {
	return g.tree
}

// attachComputingLeaf builds an in-flight promise leaf node representing a prefix
// currently undergoing prefill. The node is marked with NodeComputingPrefill state
// and its ready channel is signaled when prefill completes.
func (t *Tree) attachComputingLeaf(boundary *node, suffix []int, ready chan struct{}) *node {
	if boundary == nil || len(suffix) == 0 {
		return boundary
	}
	s := append([]int(nil), suffix...)
	leaf := &node{
		key:       s,
		parent:    boundary,
		children:  map[int]*node{},
		plen:      boundary.plen + len(s),
		state:     uint32(NodeComputingPrefill),
		ready:     ready,
		lastUsed:  t.clock,
		refs:      1, // leased while computing
		regimeKey: boundary.regimeKey,
	}
	boundary.children[s[0]] = leaf
	t.tokens += len(s)
	return leaf
}

// removeComputingNode detaches a failed or abandoned in-flight computing node
// from the tree and adjusts token accounting.
func (t *Tree) removeComputingNode(n *node) {
	if n == nil || n.parent == nil {
		return
	}
	if len(n.key) > 0 {
		first := n.key[0]
		if n.parent.children[first] == n {
			delete(n.parent.children, first)
			t.tokens -= len(n.key)
			if t.tokens < 0 {
				t.tokens = 0
			}
		}
	}
}

// RegisterFlightNS attempts to register an in-flight prefill promise for the given prefix under ns.
// If the prefix is already warm in the tree, returns (boundary, false) with boundary leased.
// If an in-flight computation is already registered for this prefix, returns (boundary, false) with boundary leased.
// If the prefix is not yet cached or computing, attaches a new node in NodeComputingPrefill state and returns (leaf, true).
// The caller must serialize calls to RegisterFlightNS (or use PrefixFlightGroup).
func (t *Tree) RegisterFlightNS(ns string, tokens []int) (*node, bool) {
	if len(tokens) == 0 {
		return nil, false
	}
	root := t.rootFor(ns)
	boundary, matched := t.boundaryFor(root, tokens)
	if matched == len(tokens) {
		boundary.refs++
		if boundary.IsComputing() {
			return boundary, false
		}
		return boundary, false
	}
	suffix := tokens[matched:]
	ready := make(chan struct{})
	leaf := t.attachComputingLeaf(boundary, suffix, ready)
	return leaf, true
}

// RegisterFlight is RegisterFlightNS on the default ("") namespace.
func (t *Tree) RegisterFlight(tokens []int) (*node, bool) {
	return t.RegisterFlightNS("", tokens)
}

// CommitFlight marks a computing node as NodeWarm, sets its KV cache and logits,
// and broadcasts completion to all waiting followers by closing ready.
func (t *Tree) CommitFlight(n *node, kv *model.KVCache, logits []float32) {
	if n == nil {
		return
	}
	n.kv = kv
	n.logits = copyCPULogits(logits)
	n.SetState(NodeWarm)
	if n.ready != nil {
		select {
		case <-n.ready:
		default:
			close(n.ready)
		}
	}
	t.evictToBudget()
}

// AbortFlight marks a computing node as NodeFailed with the provided error,
// detaches it from the tree, and broadcasts completion to waiting followers.
func (t *Tree) AbortFlight(n *node, err error) {
	if n == nil {
		return
	}
	n.flightErr = err
	n.SetState(NodeFailed)
	t.removeComputingNode(n)
	if n.ready != nil {
		select {
		case <-n.ready:
		default:
			close(n.ready)
		}
	}
}
