// Adapted from ai-dynamo/aiperf, src/aiperf/common/loop_scheduler.py:38 at
// timers: a scheduled job is pending until its deadline fires, then running
// until it returns, and CancelAll disposes of both states atomically.

package main

import (
	"context"
	"sync"
	"time"
)

type loopJob struct {
	timer *time.Timer
}

// LoopScheduler tracks scheduled work in two states (pending, running).
type LoopScheduler struct {
	mu      sync.Mutex
	ctx     context.Context
	cancel  context.CancelFunc
	pending map[*loopJob]struct{}
	running int
	wg      sync.WaitGroup
	closed  bool
}

// NewLoopScheduler returns an empty scheduler.
func NewLoopScheduler() *LoopScheduler {
	ctx, cancel := context.WithCancel(context.Background())
	return &LoopScheduler{ctx: ctx, cancel: cancel, pending: map[*loopJob]struct{}{}}
}

// ScheduleAt runs fn(ctx) at deadline (immediately when already past). It
// reports false when the scheduler has been cancelled.
func (s *LoopScheduler) ScheduleAt(deadline time.Time, fn func(ctx context.Context)) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return false
	}
	job := &loopJob{}
	s.wg.Add(1)
	job.timer = time.AfterFunc(time.Until(deadline), func() {
		s.mu.Lock()
		if _, ok := s.pending[job]; !ok {
			s.mu.Unlock()
			return
		}
		delete(s.pending, job)
		s.running++
		ctx := s.ctx
		s.mu.Unlock()
		defer func() {
			s.mu.Lock()
			s.running--
			s.mu.Unlock()
			s.wg.Done()
		}()
		fn(ctx)
	})
	s.pending[job] = struct{}{}
	return true
}

// Counts reports the pending and running job counts.
func (s *LoopScheduler) Counts() (pending, running int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.pending), s.running
}

// CancelAll refuses new work, drops every pending job without running it, and
// cancels the context handed to running jobs. It returns how many pending jobs
// were dropped.
func (s *LoopScheduler) CancelAll() int {
	s.mu.Lock()
	s.closed = true
	dropped := 0
	for job := range s.pending {
		if job.timer.Stop() {
			s.wg.Done()
		}
		dropped++
	}
	s.pending = map[*loopJob]struct{}{}
	s.mu.Unlock()
	s.cancel()
	return dropped
}

// Wait blocks until every running job has returned.
func (s *LoopScheduler) Wait() { s.wg.Wait() }
