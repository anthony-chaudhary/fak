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
}

// PrefixFlightGroup coalesces concurrent prefill computations for identical
// prompt prefixes across multiple subagents in a fleet (#11633).
//
// When an orchestrator launches parallel subagents (e.g. explore, tester, general, worker),
// 80–95% of their prompt context (system prompt, repo map, instructions, type definitions)
// is identical. Independent prefills for each subagent trigger redundant GEMM compute blasts,
// saturating memory bus bandwidth and GPU/host memory.
//
// PrefixFlightGroup solves this with a Leader/Follower pattern:
//  1. The first subagent to request an uncached prefix registers the in-flight promise (Leader)
//     and executes the prefill pass once.
//  2. Concurrent subagents requesting the same prefix (Followers) block on the completion
//     signal (<-ready) instead of issuing duplicate GEMM prefills.
//  3. Zero-Copy Suffix Forking: Upon completion broadcast, all subagents attach to the shared
//     physical prefix node and fork their private suffix leaf branches with zero duplicate VRAM.
//  4. Fault Tolerance: Leader panics, context cancellations, or failures broadcast immediately
//     so followers never hang.
type PrefixFlightGroup struct {
	mu        sync.Mutex
	treeMu    sync.Mutex
	tree      *Tree
	lock      sync.Locker
	flights   map[string]*flightPromise
	coalesced atomic.Int64
	leaders   atomic.Int64
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
	kv := node.KV()
	if !leader && kv != nil {
		kv = kv.Clone()
	}
	logits := node.Logits()
	g.Done(node)
	return kv, logits, leader, nil
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

// InFlight returns the number of prefix prefill operations currently active.
func (g *PrefixFlightGroup) InFlight() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.flights)
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
