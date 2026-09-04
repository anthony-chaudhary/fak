package cluster

import (
	"encoding/binary"
	"math"
	"sort"
	"sync"
)

// Node represents a cluster member.
type Node struct {
	ID      string
	Addr    string
	IsLocal bool
	Alive   bool
	Weight  float64 // default 1.0
}

// Ring implements rendezvous (highest random weight) hashing.
// Clients compute hash(key, nodeID) for all nodes and pick the top-N.
type Ring struct {
	mu    sync.RWMutex
	nodes []*Node
}

// NewRing creates a new rendezvous hash ring.
func NewRing() *Ring {
	return &Ring{}
}

// AddNode adds a node to the ring.
func (r *Ring) AddNode(n *Node) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if n.Weight == 0 {
		n.Weight = 1.0
	}
	// Replace if already exists
	for i, existing := range r.nodes {
		if existing.ID == n.ID {
			r.nodes[i] = n
			return
		}
	}
	r.nodes = append(r.nodes, n)
}

// RemoveNode removes a node from the ring.
func (r *Ring) RemoveNode(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i, n := range r.nodes {
		if n.ID == nodeID {
			r.nodes = append(r.nodes[:i], r.nodes[i+1:]...)
			return
		}
	}
}

// GetNodes returns the top-N responsible nodes for a key, in priority order.
// Only alive nodes are included.
func (r *Ring) GetNodes(key []byte, n int) []*Node {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if n <= 0 {
		n = 1
	}

	type scored struct {
		node  *Node
		score float64
	}
	var scores []scored

	for _, node := range r.nodes {
		if !node.Alive {
			continue
		}
		s := rendezvousScore(key, node.ID, node.Weight)
		scores = append(scores, scored{node: node, score: s})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	if n > len(scores) {
		n = len(scores)
	}

	result := make([]*Node, n)
	for i := 0; i < n; i++ {
		result[i] = scores[i].node
	}
	return result
}

// GetPrimary returns the single primary node for a key.
func (r *Ring) GetPrimary(key []byte) *Node {
	nodes := r.GetNodes(key, 1)
	if len(nodes) == 0 {
		return nil
	}
	return nodes[0]
}

// Nodes returns all current nodes.
func (r *Ring) Nodes() []*Node {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]*Node, len(r.nodes))
	copy(result, r.nodes)
	return result
}

// rendezvousScore computes the hash-based weight for (key, nodeID).
// Uses a mixing function to produce a uniform pseudo-random score.
func rendezvousScore(key []byte, nodeID string, weight float64) float64 {
	// Combine key and nodeID using FNV-1a
	h := uint64(14695981039346656037)
	for _, b := range key {
		h ^= uint64(b)
		h *= 1099511628211
	}
	for _, b := range []byte(nodeID) {
		h ^= uint64(b)
		h *= 1099511628211
	}
	// Convert to uniform [0,1) and apply weight
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], h)
	// Convert hash to float in [0, 1)
	f := float64(h) / float64(math.MaxUint64)
	// Weight: -log(f) / weight gives weighted rendezvous
	if f <= 0 {
		f = 1e-18
	}
	return weight / (-math.Log(f))
}
