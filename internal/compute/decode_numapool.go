package compute

import (
	"errors"
	"fmt"
	"sync"
)

type numaWorkerTask struct {
	nodeID int
	lo, hi int
	body   func(nodeID, lo, hi int)
	wg     *sync.WaitGroup
}

type numaPoolWorker struct {
	placement DecodeWorkerPlacement
	taskCh    chan numaWorkerTask
	stopCh    chan struct{}
}

// NUMADecodePool manages a set of pinned decode workers distributed across NUMA nodes
// according to a DecodeNUMASchedule.
type NUMADecodePool struct {
	mu        sync.Mutex
	schedule  DecodeNUMASchedule
	workers   []*numaPoolWorker
	workersWg sync.WaitGroup
	closed    bool
}

// NewNUMADecodePool creates and starts a persistent worker pool pinned according to the
// provided schedule.
func NewNUMADecodePool(schedule DecodeNUMASchedule) (*NUMADecodePool, error) {
	if !schedule.Eligible {
		return nil, fmt.Errorf("compute: schedule ineligible: %s", schedule.Reason)
	}
	if len(schedule.Placements) == 0 {
		return nil, errors.New("compute: schedule carries no placements")
	}

	p := &NUMADecodePool{
		schedule: schedule,
		workers:  make([]*numaPoolWorker, len(schedule.Placements)),
	}

	var readyWg sync.WaitGroup
	readyWg.Add(len(schedule.Placements))
	p.workersWg.Add(len(schedule.Placements))

	for i, placement := range schedule.Placements {
		w := &numaPoolWorker{
			placement: placement,
			taskCh:    make(chan numaWorkerTask, 1),
			stopCh:    make(chan struct{}),
		}
		p.workers[i] = w

		go func(worker *numaPoolWorker) {
			defer p.workersWg.Done()
			unpin, _ := PinCurrentThreadToCPUs(worker.placement.CPUs)
			defer unpin()
			readyWg.Done()

			for {
				select {
				case <-worker.stopCh:
					return
				case task, ok := <-worker.taskCh:
					if !ok {
						return
					}
					if task.body != nil && task.lo < task.hi {
						task.body(task.nodeID, task.lo, task.hi)
					}
					if task.wg != nil {
						task.wg.Done()
					}
				}
			}
		}(w)
	}

	readyWg.Wait()
	return p, nil
}

// Schedule returns the schedule this pool was initialized with.
func (p *NUMADecodePool) Schedule() DecodeNUMASchedule {
	if p == nil {
		return DecodeNUMASchedule{}
	}
	return p.schedule
}

// Dispatch executes body over n output rows across the pool's workers.
// Row intervals [lo, hi) are statically determined from worker placement,
// without cross-socket cursor locks or work stealing.
func (p *NUMADecodePool) Dispatch(n int, body func(nodeID, lo, hi int)) error {
	if p == nil {
		return errors.New("compute: nil NUMADecodePool")
	}
	if n <= 0 {
		return nil
	}
	if body == nil {
		return errors.New("compute: nil dispatch body")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return errors.New("compute: dispatch on closed NUMADecodePool")
	}

	numWorkers := len(p.workers)
	var jobWg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		lo, hi := workerRowBounds(w, numWorkers, n)
		if lo >= hi {
			continue
		}
		jobWg.Add(1)
		p.workers[w].taskCh <- numaWorkerTask{
			nodeID: p.workers[w].placement.NodeID,
			lo:     lo,
			hi:     hi,
			body:   body,
			wg:     &jobWg,
		}
	}

	jobWg.Wait()
	return nil
}

// Close shuts down all worker goroutines and releases their thread bindings.
func (p *NUMADecodePool) Close() error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	for _, w := range p.workers {
		close(w.stopCh)
	}
	p.mu.Unlock()

	p.workersWg.Wait()
	return nil
}

func workerRowBounds(workerIdx, numWorkers, n int) (int, int) {
	if workerIdx < 0 || workerIdx >= numWorkers || n <= 0 {
		return 0, 0
	}
	base := n / numWorkers
	extra := n % numWorkers
	var lo, hi int
	if workerIdx < extra {
		lo = workerIdx * (base + 1)
		hi = lo + (base + 1)
	} else {
		lo = extra*(base+1) + (workerIdx-extra)*base
		hi = lo + base
	}
	if lo > n {
		lo = n
	}
	if hi > n {
		hi = n
	}
	return lo, hi
}
