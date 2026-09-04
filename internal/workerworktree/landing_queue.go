package workerworktree

import (
	"sync"
)

// LandingQueue provides a serialized landing coordinator backing landing operations
// per repository root to prevent thundering-herd livelocks under high concurrency (#11235).
type LandingQueue struct {
	mu    sync.Mutex
	locks map[string]*sync.Mutex
}

// NewLandingQueue creates an initialized LandingQueue.
func NewLandingQueue() *LandingQueue {
	return &LandingQueue{
		locks: make(map[string]*sync.Mutex),
	}
}

// DefaultLandingQueue is the default serialized landing coordinator.
var DefaultLandingQueue = NewLandingQueue()

func (q *LandingQueue) repoMutex(root string) *sync.Mutex {
	key := canonicalComparisonPath(root)
	q.mu.Lock()
	defer q.mu.Unlock()
	m, ok := q.locks[key]
	if !ok {
		m = &sync.Mutex{}
		q.locks[key] = m
	}
	return m
}

// Lock acquires the landing lock for root and returns an unlock function.
func (q *LandingQueue) Lock(root string) func() {
	m := q.repoMutex(root)
	m.Lock()
	return func() {
		m.Unlock()
	}
}

// Coordinate runs op exclusively under the landing coordinator for root.
func (q *LandingQueue) Coordinate(root string, op func() Result) Result {
	unlock := q.Lock(root)
	defer unlock()
	return op()
}

// Enqueue runs op exclusively under the landing coordinator for root.
func (q *LandingQueue) Enqueue(root string, op func() (Result, error)) (Result, error) {
	unlock := q.Lock(root)
	defer unlock()
	return op()
}
