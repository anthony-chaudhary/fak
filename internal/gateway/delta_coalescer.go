package gateway

import (
	"sync"
)

// DeltaChunk represents a text chunk and associated token count emitted during generation.
type DeltaChunk struct {
	Text   string
	Tokens int
}

// CoalescingSender buffers and coalesces streamed deltas when the consumer experiences
// network backpressure (#10725, borrowed from vLLM output_processor).
// It decouples the model forward/generation iteration from client network I/O: when a client's
// HTTP SSE connection cannot keep up with generation, pending text deltas and tokens are
// coalesced in a bounded intermediate queue via in-place aggregation rather than stalling
// the model generation loop or queueing unboundedly.
type CoalescingSender struct {
	mu         sync.Mutex
	pending    DeltaChunk
	hasPending bool
	closed     bool
	notify     chan struct{}
	done       chan struct{}
}

// NewCoalescingSender creates a coalescing sender that drains deltas to deliverFn in a background consumer goroutine.
func NewCoalescingSender(deliverFn func(chunk DeltaChunk) error) *CoalescingSender {
	s := &CoalescingSender{
		notify: make(chan struct{}, 1),
		done:   make(chan struct{}),
	}
	go s.drainConsumer(deliverFn)
	return s
}

// Send submits a delta chunk from the fast generator loop.
// If the consumer is busy delivering a previous chunk, consecutive incoming deltas are coalesced
// in-place without blocking the generator loop.
func (s *CoalescingSender) Send(text string, tokens int) {
	if s == nil || (text == "" && tokens == 0) {
		return
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	if !s.hasPending {
		s.pending = DeltaChunk{Text: text, Tokens: tokens}
		s.hasPending = true
	} else {
		// Coalesce in-place
		s.pending.Text += text
		s.pending.Tokens += tokens
	}
	s.mu.Unlock()

	// Notify drain loop without blocking
	select {
	case s.notify <- struct{}{}:
	default:
	}
}

// Close signals that generation is complete and waits for all pending coalesced deltas to be delivered.
func (s *CoalescingSender) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	s.mu.Unlock()

	select {
	case s.notify <- struct{}{}:
	default:
	}
	<-s.done
}

func (s *CoalescingSender) drainConsumer(deliverFn func(chunk DeltaChunk) error) {
	defer close(s.done)
	for {
		<-s.notify

		for {
			s.mu.Lock()
			if !s.hasPending {
				isClosed := s.closed
				s.mu.Unlock()
				if isClosed {
					return
				}
				break
			}
			chunk := s.pending
			s.pending = DeltaChunk{}
			s.hasPending = false
			s.mu.Unlock()

			if deliverFn != nil {
				_ = deliverFn(chunk)
			}
		}
	}
}
