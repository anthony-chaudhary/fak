package engine

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// OverlapConfig holds concurrency configuration.
type OverlapConfig struct {
	InFlightDepth int `json:"in_flight_depth"`
}

// InFlightTask defines an executable task.
type InFlightTask[T any] struct {
	ID        string
	DependsOn []string
	Execute   func(ctx context.Context) (T, error)
}

// OverlapResult records the outcome of a finished task.
type OverlapResult[T any] struct {
	ID        string
	Value     T
	Err       error
	Duration  time.Duration
	Committed bool
}

type taskRecord[T any] struct {
	id        string
	done      chan struct{}
	committed bool
	finished  bool
	result    *OverlapResult[T]
}

// OverlapRunner coordinates concurrent task execution.
type OverlapRunner[T any] struct {
	depth    int
	inFlight atomic.Int64
	sem      chan struct{}
	closeCh  chan struct{}
	closed   bool

	mu       sync.Mutex
	tasks    map[string]*taskRecord[T]
	results  []*OverlapResult[T]
	buffered []*OverlapResult[T]
	yielded  map[string]bool
	wg       sync.WaitGroup
}

// NewOverlapRunner creates a scheduler with bounded depth.
func NewOverlapRunner[T any](depth int) *OverlapRunner[T] {
	if depth < 0 {
		depth = 0
	}
	var sem chan struct{}
	if depth > 0 {
		sem = make(chan struct{}, depth)
	}
	return &OverlapRunner[T]{
		depth:   depth,
		sem:     sem,
		closeCh: make(chan struct{}),
		tasks:   make(map[string]*taskRecord[T]),
		yielded: make(map[string]bool),
	}
}

// NewOverlapRunnerWithConfig constructs a scheduler using an OverlapConfig.
func NewOverlapRunnerWithConfig[T any](cfg OverlapConfig) *OverlapRunner[T] {
	return NewOverlapRunner[T](cfg.InFlightDepth)
}

// Depth returns configured in-flight depth.
func (s *OverlapRunner[T]) Depth() int {
	return s.depth
}

// InFlightCount returns current active in-flight count.
func (s *OverlapRunner[T]) InFlightCount() int64 {
	return s.inFlight.Load()
}

// Submit enqueues or executes a task.
func (s *OverlapRunner[T]) Submit(ctx context.Context, task InFlightTask[T]) (*OverlapResult[T], error) {
	taskID := task.ID
	if taskID == "" {
		taskID = fmt.Sprintf("task-%d", time.Now().UnixNano())
	}
	task.ID = taskID

	if s.depth <= 0 {
		s.mu.Lock()
		if s.closed {
			s.mu.Unlock()
			return nil, errors.New("scheduler closed")
		}
		if ctx.Err() != nil {
			s.mu.Unlock()
			return nil, ctx.Err()
		}

		var waitChans []<-chan struct{}
		for _, depID := range task.DependsOn {
			if dep, ok := s.tasks[depID]; ok && !dep.committed {
				waitChans = append(waitChans, dep.done)
			}
		}

		rec := &taskRecord[T]{
			id:   taskID,
			done: make(chan struct{}),
		}
		s.tasks[taskID] = rec
		s.mu.Unlock()

		abortTask := func() {
			s.mu.Lock()
			delete(s.tasks, taskID)
			rec.finished = true
			close(rec.done)
			s.mu.Unlock()
		}

		for _, ch := range waitChans {
			select {
			case <-ch:
			case <-ctx.Done():
				abortTask()
				return nil, ctx.Err()
			case <-s.closeCh:
				abortTask()
				return nil, errors.New("scheduler closed")
			}
		}

		s.inFlight.Add(1)
		start := time.Now()
		val, execErr := task.Execute(ctx)
		dur := time.Since(start)
		s.inFlight.Add(-1)

		res := &OverlapResult[T]{
			ID:        taskID,
			Value:     val,
			Err:       execErr,
			Duration:  dur,
			Committed: true,
		}

		s.mu.Lock()
		rec.committed = true
		rec.finished = true
		rec.result = res
		close(rec.done)
		s.results = append(s.results, res)
		s.yielded[taskID] = true
		s.mu.Unlock()

		return res, execErr
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, errors.New("scheduler closed")
	}
	if ctx.Err() != nil {
		s.mu.Unlock()
		return nil, ctx.Err()
	}

	var waitChans []<-chan struct{}
	for _, depID := range task.DependsOn {
		if dep, ok := s.tasks[depID]; ok && !dep.committed {
			waitChans = append(waitChans, dep.done)
		}
	}

	rec := &taskRecord[T]{
		id:   taskID,
		done: make(chan struct{}),
	}
	s.tasks[taskID] = rec
	s.mu.Unlock()

	abortTask := func() {
		s.mu.Lock()
		delete(s.tasks, taskID)
		rec.finished = true
		close(rec.done)
		s.mu.Unlock()
	}

	for _, ch := range waitChans {
		select {
		case <-ch:
		case <-ctx.Done():
			abortTask()
			return nil, ctx.Err()
		case <-s.closeCh:
			abortTask()
			return nil, errors.New("scheduler closed")
		}
	}

	select {
	case s.sem <- struct{}{}:
	case <-ctx.Done():
		abortTask()
		return nil, ctx.Err()
	case <-s.closeCh:
		abortTask()
		return nil, errors.New("scheduler closed")
	}

	s.inFlight.Add(1)
	s.wg.Add(1)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				s.mu.Lock()
				rec.committed = true
				res := &OverlapResult[T]{
					ID:        taskID,
					Err:       fmt.Errorf("task panicked: %v", r),
					Committed: true,
				}
				rec.result = res
				s.results = append(s.results, res)
				s.buffered = append(s.buffered, res)
				s.mu.Unlock()
			}
			s.inFlight.Add(-1)
			<-s.sem
			s.wg.Done()
			s.mu.Lock()
			rec.finished = true
			close(rec.done)
			s.mu.Unlock()
		}()

		start := time.Now()
		val, execErr := task.Execute(ctx)
		dur := time.Since(start)

		res := &OverlapResult[T]{
			ID:        taskID,
			Value:     val,
			Err:       execErr,
			Duration:  dur,
			Committed: true,
		}

		s.mu.Lock()
		rec.committed = true
		rec.result = res
		s.results = append(s.results, res)
		s.buffered = append(s.buffered, res)
		s.mu.Unlock()
	}()

	s.mu.Lock()
	var yielded *OverlapResult[T]
	if len(s.buffered) >= s.depth {
		yielded = s.buffered[0]
		s.buffered = s.buffered[1:]
		s.yielded[yielded.ID] = true
	}
	s.mu.Unlock()

	return yielded, nil
}

// Drain awaits pending in-flight tasks and collects remaining results.
func (s *OverlapRunner[T]) Drain(ctx context.Context) ([]*OverlapResult[T], error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-s.closeCh:
		return nil, errors.New("scheduler closed")
	default:
	}

	for {
		var pending []<-chan struct{}
		s.mu.Lock()
		for _, rec := range s.tasks {
			if !rec.finished {
				pending = append(pending, rec.done)
			}
		}
		s.mu.Unlock()

		if len(pending) == 0 {
			break
		}

		for _, ch := range pending {
			select {
			case <-ch:
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-s.closeCh:
				return nil, errors.New("scheduler closed")
			}
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var remaining []*OverlapResult[T]
	for _, r := range s.results {
		if !s.yielded[r.ID] {
			remaining = append(remaining, r)
			s.yielded[r.ID] = true
		}
	}
	s.buffered = nil
	return remaining, nil
}

// Results returns all committed results in order.
func (s *OverlapRunner[T]) Results() []*OverlapResult[T] {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*OverlapResult[T], len(s.results))
	copy(out, s.results)
	return out
}

// Close marks scheduler closed.
func (s *OverlapRunner[T]) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	close(s.closeCh)
}
