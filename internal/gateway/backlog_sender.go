package gateway

import "sync"

// a non-blocking send over a bounded transport with an unbounded FIFO backlog that is
// drained in order, so a full transport never blocks the caller and never reorders
// messages. Component and host oracle only: not wired into a serve path.

// BacklogSender sends over a bounded channel without blocking and keeps overflow
// in order in an unbounded FIFO.
type BacklogSender[T any] struct {
	mu      sync.Mutex
	out     chan T
	backlog []T
}

// NewBacklogSender wraps out (the bounded transport).
func NewBacklogSender[T any](out chan T) *BacklogSender[T] {
	return &BacklogSender[T]{out: out}
}

// Send never blocks. If earlier messages are still backlogged, v queues behind them
// so ordering holds; otherwise it goes straight to the transport when there is room.
func (s *BacklogSender[T]) Send(v T) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.backlog) > 0 {
		s.backlog = append(s.backlog, v)
		s.drainLocked()
		return
	}
	select {
	case s.out <- v:
	default:
		s.backlog = append(s.backlog, v)
	}
}

// Drain moves as much backlog into the transport as it will accept, in FIFO order.
func (s *BacklogSender[T]) Drain() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.drainLocked()
}

func (s *BacklogSender[T]) drainLocked() int {
	moved := 0
	for len(s.backlog) > 0 {
		select {
		case s.out <- s.backlog[0]:
			s.backlog = s.backlog[1:]
			moved++
		default:
			return moved
		}
	}
	s.backlog = nil
	return moved
}

// Pending reports the backlog length.
func (s *BacklogSender[T]) Pending() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.backlog)
}
