// eviction.go — bump-topology KV eviction with allocation-free recency tracking.
//
// ADAPT port of incoai/splash's bump-topology KV-eviction policy (Apache-2.0),
// pinned at 6c6002d42aabdddb9e31b96c9bba9041e2fba29f:
//
//	runtime/engine/KvCache.hpp:143    resident-leaf eligibility + candidate scan
//	runtime/engine/KvCache.cpp        reindex / inherit / retain-release / drop-adopt
//	runtime/engine/RecencyOrder.hpp:13 intrusive last-used order
//	runtime/engine/CacheRecency.hpp:6  shared monotonic access clock
//
// License record: THIRD_PARTY_NOTICES.md ("Splash bump-topology KV eviction").
package kvmmu

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
)

// RecencyClock is a shared monotonic access-order stamp. Every entry that
// enters or refreshes its position in a recency order takes the next stamp, so
// distinct stamps order distinct touches; the clock saturates at MaxUint64.
type RecencyClock struct {
	value atomic.Uint64
}

// NewRecencyClock returns a clock whose first stamp is 1.
func NewRecencyClock() *RecencyClock { return &RecencyClock{} }

// Next returns the next strictly increasing stamp.
func (c *RecencyClock) Next() uint64 {
	for {
		v := c.value.Load()
		if v == math.MaxUint64 {
			return v
		}
		if c.value.CompareAndSwap(v, v+1) {
			return v + 1
		}
	}
}

// EvictionCandidate names one evictable block in last-used order.
type EvictionCandidate struct {
	ID       int64
	LastUsed uint64
}

// RecencyNode is an intrusive node in a RecencyOrder. The owning entry
// allocates its node once, when published, and keeps it while unlinked, so
// linking and unlinking under memory pressure never allocate.
type RecencyNode struct {
	key      [2]uint64 // {lastUsed, id}
	priority uint64
	left     *RecencyNode
	right    *RecencyNode
	owner    *RecencyOrder
}

// NewRecencyNode allocates an order node. This is the only operation in the
// eviction path that allocates.
func NewRecencyNode() *RecencyNode { return &RecencyNode{} }

// Linked reports whether the node currently sits in an order.
func (n *RecencyNode) Linked() bool { return n.owner != nil }

// Unlink removes the node from its order. Panics if the node is not linked.
func (n *RecencyNode) Unlink() {
	o := n.owner
	if o == nil {
		panic("kvmmu: unlink of unlinked recency node")
	}
	root, ok := deleteByKey(o.root, n.key)
	if !ok {
		panic("kvmmu: recency order lost its own node")
	}
	o.root = root
	o.size--
	n.owner = nil
	n.left, n.right = nil, nil
}

// RecencyOrder keeps members in last-used order, oldest first, keyed by
// (lastUsed, id). It is a treap over preallocated nodes: insert, delete,
// oldest, and successor scans are pointer surgery only. Not goroutine-safe;
// guard externally.
type RecencyOrder struct {
	root *RecencyNode
	size int
}

// NewRecencyOrder returns an empty order.
func NewRecencyOrder() *RecencyOrder { return &RecencyOrder{} }

// Link inserts node into the order at position (lastUsed, id). Ids must be
// positive. Panics if the node is already linked.
func (o *RecencyOrder) Link(n *RecencyNode, lastUsed uint64, id int64) {
	if n.Linked() {
		panic("kvmmu: link of already-linked recency node")
	}
	n.key = [2]uint64{lastUsed, uint64(id)}
	n.priority = splitmix64(n.key[0] ^ (n.key[1] + 0x9E3779B97F4A7C15))
	n.owner = o
	left, right := split(o.root, n.key)
	o.root = merge(merge(left, n), right)
	o.size++
}

// Oldest reports the least-recently-used member, or false when empty.
func (o *RecencyOrder) Oldest() (EvictionCandidate, bool) {
	n := o.root
	if n == nil {
		return EvictionCandidate{}, false
	}
	for n.left != nil {
		n = n.left
	}
	return candidateOf(n), true
}

// Next reports the member strictly after the given position in the order, for
// a scan that skips members.
func (o *RecencyOrder) Next(after EvictionCandidate) (EvictionCandidate, bool) {
	key := [2]uint64{after.LastUsed, uint64(after.ID)}
	var succ *RecencyNode
	n := o.root
	for n != nil {
		if lessKey(key, n.key) {
			succ = n
			n = n.left
		} else {
			n = n.right
		}
	}
	if succ == nil {
		return EvictionCandidate{}, false
	}
	return candidateOf(succ), true
}

// NewestID reports the most-recently-used member's id, or 0 when empty.
func (o *RecencyOrder) NewestID() int64 {
	n := o.root
	if n == nil {
		return 0
	}
	for n.right != nil {
		n = n.right
	}
	return int64(n.key[1])
}

// Len reports the number of members.
func (o *RecencyOrder) Len() int { return o.size }

func candidateOf(n *RecencyNode) EvictionCandidate {
	return EvictionCandidate{ID: int64(n.key[1]), LastUsed: n.key[0]}
}

func lessKey(a, b [2]uint64) bool {
	return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1])
}

// split divides root into left (keys < k) and right (keys >= k).
func split(root *RecencyNode, k [2]uint64) (left, right *RecencyNode) {
	if root == nil {
		return nil, nil
	}
	if lessKey(k, root.key) {
		l, r := split(root.left, k)
		root.left = r
		return l, root
	}
	l, r := split(root.right, k)
	root.right = l
	return root, r
}

// merge joins two trees whose keys all compare consistently.
func merge(a, b *RecencyNode) *RecencyNode {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	if a.priority > b.priority {
		a.right = merge(a.right, b)
		return a
	}
	b.left = merge(a, b.left)
	return b
}

// deleteByKey removes the node holding key from root, reporting whether it was
// present.
func deleteByKey(root *RecencyNode, key [2]uint64) (*RecencyNode, bool) {
	if root == nil {
		return nil, false
	}
	switch {
	case lessKey(key, root.key):
		left, ok := deleteByKey(root.left, key)
		root.left = left
		return root, ok
	case lessKey(root.key, key):
		right, ok := deleteByKey(root.right, key)
		root.right = right
		return root, ok
	default:
		found := root
		root = merge(found.left, found.right)
		found.left, found.right = nil, nil
		return root, true
	}
}

// splitmix64 derives a deterministic treap priority from a key, so the order's
// shape depends only on the keys, not on insertion history.
func splitmix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
	x = (x ^ (x >> 27)) * 0x94D049BB133111EB
	return x ^ (x >> 31)
}

// block is the eviction-relevant state of one KV cache block: its position in
// the topology tree, who references it, and its recency stamp.
type block struct {
	id               int64
	parent           int64
	children         int
	resident         bool
	residentChildren int
	users            int
	transferring     bool
	lastUsed         uint64
	node             RecencyNode
}

// Evictor tracks a tree of KV blocks and selects victims in bump-topology
// order: only resident leaves (no resident children, no users, no in-flight
// transfer) are evictable, kept in an intrusive last-used order so eviction
// never allocates. When a leaf leaves the resident tier, its parent inherits
// the leaf's recency stamp (the bump): a parent that becomes a leaf was used
// no later than its child and must not jump ahead of colder chains.
//
// Page release itself stays with the refcount layer: DropPage only moves the
// block out of the resident tier; the caller releases the physical page.
// All methods are safe for concurrent use.
type Evictor struct {
	mu        sync.Mutex
	clock     *RecencyClock
	blocks    map[int64]*block
	ramLeaves RecencyOrder
}

// NewEvictor returns an empty topology bound to clock for recency stamps.
func NewEvictor(clock *RecencyClock) *Evictor {
	return &Evictor{clock: clock, blocks: make(map[int64]*block)}
}

// Publish adds a block to the topology. Parent 0 marks a root; a resident
// child requires a resident parent. The block receives a fresh recency stamp
// and enters the leaf order when eligible. Panics on a duplicate id, an
// unknown parent, or a non-positive id.
func (e *Evictor) Publish(id, parent int64, resident bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if id <= 0 {
		panic(fmt.Sprintf("kvmmu: block id %d must be positive", id))
	}
	if _, exists := e.blocks[id]; exists {
		panic(fmt.Sprintf("kvmmu: duplicate KV block id %d", id))
	}
	var p *block
	if parent != 0 {
		p = e.mustBlock(parent)
		if resident && !p.resident {
			panic(fmt.Sprintf("kvmmu: resident block %d published under non-resident parent %d", id, parent))
		}
	}
	b := &block{id: id, parent: parent, resident: resident}
	b.lastUsed = e.clock.Next()
	e.blocks[id] = b
	if p != nil {
		p.children++
		if resident {
			p.residentChildren++
		}
		e.reindex(p)
	}
	e.reindex(b)
}

// Retain records that a request now uses the block and refreshes its recency.
func (e *Evictor) Retain(id int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	b := e.mustBlock(id)
	b.users++
	b.lastUsed = e.clock.Next()
	e.reindex(b)
}

// Release drops one user. A released leaf becomes the newest member (it was
// just touched); a parent keeps the recency its children pass on. Panics if
// the block has no active users.
func (e *Evictor) Release(id int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	b := e.mustBlock(id)
	if b.users == 0 {
		panic(fmt.Sprintf("kvmmu: release on block %d with no active users", id))
	}
	b.users--
	if b.users == 0 && b.children == 0 {
		b.lastUsed = e.clock.Next()
	}
	e.reindex(b)
}

// Touch refreshes a block's recency without changing its user count.
func (e *Evictor) Touch(id int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	b := e.mustBlock(id)
	b.lastUsed = e.clock.Next()
	e.reindex(b)
}

// SetTransferring marks a block as moving between tiers; a transferring block
// is neither evicted nor replaced until the transfer clears. Panics if the
// state did not change.
func (e *Evictor) SetTransferring(id int64, transferring bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	b := e.mustBlock(id)
	if b.transferring == transferring {
		panic(fmt.Sprintf("kvmmu: transfer state of block %d did not change", id))
	}
	b.transferring = transferring
	e.reindex(b)
}

// DropPage moves a resident leaf out of the resident tier: the block stays in
// the topology as disk-only and its parent bumps to the leaf's recency.
// Panics if the block is not an unused resident leaf.
func (e *Evictor) DropPage(id int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	b := e.mustBlock(id)
	if !b.resident || b.residentChildren > 0 || b.users > 0 {
		panic(fmt.Sprintf("kvmmu: cannot drop the page of block %d that is still in use", id))
	}
	b.resident = false
	e.reindex(b)
	if b.parent != 0 {
		p := e.mustBlock(b.parent)
		p.residentChildren--
		e.inherit(p, b.lastUsed)
	}
}

// AdoptPage brings a disk-only block back into the resident tier; its parent
// must be resident. The block keeps its existing recency stamp: its content
// is old regardless of the restore.
func (e *Evictor) AdoptPage(id int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	b := e.mustBlock(id)
	if b.resident {
		panic(fmt.Sprintf("kvmmu: block %d already has a page", id))
	}
	if b.parent != 0 {
		p := e.mustBlock(b.parent)
		if !p.resident {
			panic(fmt.Sprintf("kvmmu: cannot adopt a page for block %d under non-resident parent %d", id, b.parent))
		}
	}
	b.resident = true
	e.reindex(b)
	if b.parent != 0 {
		p := e.mustBlock(b.parent)
		p.residentChildren++
		e.reindex(p)
	}
}

// Erase removes a block from the topology entirely. Only a block without
// children, users, or an in-flight transfer may be erased; its parent bumps
// to the erased block's recency.
func (e *Evictor) Erase(id int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	b := e.mustBlock(id)
	if b.children > 0 || b.users > 0 || b.transferring {
		panic(fmt.Sprintf("kvmmu: cannot erase referenced KV block %d", id))
	}
	if b.node.Linked() {
		b.node.Unlink()
	}
	delete(e.blocks, b.id)
	if b.parent != 0 {
		p := e.mustBlock(b.parent)
		p.children--
		if b.resident {
			p.residentChildren--
		}
		e.inherit(p, b.lastUsed)
	}
}

// ResidentLeaf reports whether the block's page can go: resident, without
// resident children or users, and not in transfer.
func (e *Evictor) ResidentLeaf(id int64) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	b := e.mustBlock(id)
	return b.resident && b.residentChildren == 0 && b.users == 0 && !b.transferring
}

// EvictionCandidate returns the oldest resident leaf that no request uses.
// Pass the previous candidate's ID to continue the scan without a lookup;
// zero starts at the head of the order.
func (e *Evictor) EvictionCandidate(after int64) (EvictionCandidate, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if after == 0 {
		return e.ramLeaves.Oldest()
	}
	b := e.mustBlock(after)
	return e.ramLeaves.Next(EvictionCandidate{ID: after, LastUsed: b.lastUsed})
}

// LastUsed reports the block's current recency stamp.
func (e *Evictor) LastUsed(id int64) uint64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.mustBlock(id).lastUsed
}

// Blocks reports the number of blocks tracked, resident or not.
func (e *Evictor) Blocks() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.blocks)
}

// reindex places the block in the order its state calls for: resident leaves
// only. A block a request uses, one in transfer, or one with resident
// children is in no order.
func (e *Evictor) reindex(b *block) {
	if b.node.Linked() {
		b.node.Unlink()
	}
	if b.resident && b.residentChildren == 0 && b.users == 0 && !b.transferring {
		e.ramLeaves.Link(&b.node, b.lastUsed, b.id)
	}
}

// inherit bumps a parent that becomes a leaf when a child leaves its tier: it
// was used no later than the child and must not jump ahead of colder chains.
func (e *Evictor) inherit(p *block, childLastUsed uint64) {
	var leaf bool
	if p.resident {
		leaf = p.residentChildren == 0
	} else {
		leaf = p.children == 0
	}
	if leaf && p.users == 0 && childLastUsed > p.lastUsed {
		p.lastUsed = childLastUsed
	}
	e.reindex(p)
}

func (e *Evictor) mustBlock(id int64) *block {
	b, ok := e.blocks[id]
	if !ok {
		panic(fmt.Sprintf("kvmmu: unknown KV block %d", id))
	}
	return b
}
