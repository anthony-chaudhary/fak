package cluster

import (
	"log"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/l3server/client"
)

// Replicator handles async replication to peer nodes.
type Replicator struct {
	ring     *Ring
	localID  string
	replicas int // number of replicas (including primary)
	queue    chan replicateOp
	quit     chan struct{}
	mu       sync.Mutex
	pool     map[string]*client.Client
}

type replicateOp struct {
	Key   []byte
	Value []byte
	TTLMs int64
	Op    string // "set", "delete"
}

// NewReplicator creates an async replicator.
func NewReplicator(ring *Ring, localID string, replicas int) *Replicator {
	if replicas < 1 {
		replicas = 1
	}
	return &Replicator{
		ring:     ring,
		localID:  localID,
		replicas: replicas,
		queue:    make(chan replicateOp, 10000),
		quit:     make(chan struct{}),
		pool:     make(map[string]*client.Client),
	}
}

// Start begins processing the replication queue.
func (r *Replicator) Start() {
	go r.processLoop()
}

// Stop halts replication and closes all pooled connections.
func (r *Replicator) Stop() {
	close(r.quit)
	r.mu.Lock()
	defer r.mu.Unlock()
	for addr, c := range r.pool {
		c.Close()
		delete(r.pool, addr)
	}
}

// ReplicateSet queues an async SET replication.
func (r *Replicator) ReplicateSet(key, value []byte, ttlMs int64) {
	select {
	case r.queue <- replicateOp{Key: key, Value: value, TTLMs: ttlMs, Op: "set"}:
	default:
		// Queue full â€” drop (fire-and-forget)
	}
}

// ReplicateDelete queues an async DELETE replication.
func (r *Replicator) ReplicateDelete(key []byte) {
	select {
	case r.queue <- replicateOp{Key: key, Op: "delete"}:
	default:
	}
}

// QueueDepth returns the current number of pending replication operations.
func (r *Replicator) QueueDepth() int { return len(r.queue) }

// QueueCap returns the capacity of the replication queue.
func (r *Replicator) QueueCap() int { return cap(r.queue) }

func (r *Replicator) processLoop() {
	for {
		select {
		case <-r.quit:
			return
		case op := <-r.queue:
			r.replicate(op)
		}
	}
}

// getConn returns a pooled connection to addr, creating one lazily.
func (r *Replicator) getConn(addr string) (*client.Client, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if c, ok := r.pool[addr]; ok {
		return c, nil
	}
	c, err := client.New(addr)
	if err != nil {
		return nil, err
	}
	r.pool[addr] = c
	return c, nil
}

// removeConn closes and removes a pooled connection (called on error).
func (r *Replicator) removeConn(addr string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if c, ok := r.pool[addr]; ok {
		c.Close()
		delete(r.pool, addr)
	}
}

func (r *Replicator) replicate(op replicateOp) {
	nodes := r.ring.GetNodes(op.Key, r.replicas)

	for _, node := range nodes {
		if node.ID == r.localID || node.IsLocal {
			continue // skip self
		}

		c, err := r.getConn(node.Addr)
		if err != nil {
			log.Printf("[replication] connect to %s failed: %v", node.Addr, err)
			continue
		}

		switch op.Op {
		case "set":
			if err := c.SetReplicated(op.Key, op.Value, op.TTLMs); err != nil {
				log.Printf("[replication] SET to %s failed: %v", node.Addr, err)
				r.removeConn(node.Addr)
			}
		case "delete":
			if err := c.DeleteReplicated(op.Key); err != nil {
				log.Printf("[replication] DELETE to %s failed: %v", node.Addr, err)
				r.removeConn(node.Addr)
			}
		}
	}
}
