// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vLLM at e5196e2959a42908313510f55fa21f213a48802e (Apache-2.0).
// License and notice: THIRD_PARTY_NOTICES.md ("vLLM").
package gateway

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
)

// DeltaChunk represents a text chunk and associated token count emitted during generation.
type DeltaChunk struct {
	Text    string
	Tokens  int
	held    bool
	barrier bool // serialized protocol event: never merge through this boundary
}

// These are per-request transport buffers, not model/context budgets. Even a
// tiny/empty-fragment producer has a finite event budget. There is no queue wait
// on the generator path: a client exceeding either bound fails its own stream.
const (
	streamPendingBytes  = 1 << 20
	streamPendingEvents = 1024
)

var (
	errStreamBackpressure = errors.New("downstream stream pending output limit exceeded")
	errStreamDelivery     = errors.New("downstream stream delivery failed")
)

// CoalescingSender decouples generation from delivery (#10725, inspired by vLLM
// output_processor). Both budgets include the chunk currently being delivered;
// a one-slot queue alone would not bound an ever-growing pending string.
// Delivery must be interruptible by ctx (the live HTTP caller uses write deadlines).
// Overflow and delivery failures terminate the stream explicitly, never as success.
type CoalescingSender struct {
	mu             sync.Mutex
	pending        []DeltaChunk
	bufferedBytes  int
	bufferedEvents int
	closed         bool
	firstError     error
	preserveChunks bool
	ctx            context.Context
	onError        func()
	notify         chan struct{}
	done           chan struct{}
}

// NewCoalescingSender merges adjacent pending text deltas. Callers that require
// per-fragment wire boundaries use the request-local preserving mode instead.
func NewCoalescingSender(deliverFn func(DeltaChunk) error) *CoalescingSender {
	return newCoalescingSender(context.Background(), false, deliverFn, nil)
}

func newCoalescingSender(ctx context.Context, preserveChunks bool, deliverFn func(DeltaChunk) error, onError func()) *CoalescingSender {
	s := &CoalescingSender{
		ctx: ctx, preserveChunks: preserveChunks, onError: onError,
		notify: make(chan struct{}, 1), done: make(chan struct{}),
	}
	go s.drainConsumer(deliverFn)
	return s
}

// Send never waits for network I/O. An accepted fragment is delivered in order
// unless this stream explicitly fails; no successful completion silently drops it.
func (s *CoalescingSender) Send(text string, tokens int) error {
	return s.send(DeltaChunk{Text: text, Tokens: tokens})
}

func (s *CoalescingSender) activity() error { return s.send(DeltaChunk{held: true}) }

// barrier queues an already-serialized protocol event in the same bounded order
// as text. send clones its backing bytes; the caller retains no mutable loan.
func (s *CoalescingSender) barrier(frame string) error {
	return s.send(DeltaChunk{Text: frame, barrier: true})
}

func (s *CoalescingSender) send(chunk DeltaChunk) error {
	if s == nil {
		return nil
	}
	if err := s.ctx.Err(); err != nil {
		s.fail(err)
		return err
	}
	s.mu.Lock()
	if s.closed {
		err := s.firstError
		if err == nil {
			err = io.ErrClosedPipe
		}
		s.mu.Unlock()
		return err
	}
	if !chunk.held && !chunk.barrier && chunk.Text == "" && chunk.Tokens == 0 {
		s.mu.Unlock()
		return nil
	}
	last := len(s.pending) - 1
	// Held progress has no payload. Adjacent notifications can be coalesced,
	// but never across a content fragment or with a content fragment.
	if chunk.held && last >= 0 && s.pending[last].held {
		s.mu.Unlock()
		return nil
	}
	merge := !s.preserveChunks && !chunk.held && !chunk.barrier && last >= 0 && !s.pending[last].held && !s.pending[last].barrier
	if len(chunk.Text) > streamPendingBytes-s.bufferedBytes || (!merge && s.bufferedEvents >= streamPendingEvents) {
		s.mu.Unlock()
		s.fail(errStreamBackpressure)
		return errStreamBackpressure
	}
	s.bufferedBytes += len(chunk.Text)
	if merge {
		s.pending[last].Text += chunk.Text
		s.pending[last].Tokens += chunk.Tokens
	} else {
		// Do not retain a larger producer backing string through a small slice.
		chunk.Text = strings.Clone(chunk.Text)
		s.pending = append(s.pending, chunk)
		s.bufferedEvents++
	}
	s.mu.Unlock()
	s.wake()
	return nil
}

func (s *CoalescingSender) wake() {
	select {
	case s.notify <- struct{}{}:
	default:
	}
}

func (s *CoalescingSender) fail(err error) {
	s.mu.Lock()
	first := s.firstError == nil
	if first {
		s.firstError = err
	}
	s.closed = true
	s.pending = nil
	s.mu.Unlock()
	s.wake()
	if first && s.onError != nil {
		s.onError()
	}
}

// Close joins the consumer, including concurrent/repeated callers, and returns
// its first error. The delivery callback's I/O bound also bounds a final drain.
func (s *CoalescingSender) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	s.wake()
	<-s.done
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.firstError
}

func (s *CoalescingSender) drainConsumer(deliverFn func(DeltaChunk) error) {
	defer func() {
		// The async boundary must not turn an ordinary HTTP writer panic into a
		// process-wide panic, or expose arbitrary panic contents to the client.
		if recover() != nil {
			s.fail(errStreamDelivery)
		}
		close(s.done)
	}()
	for {
		select {
		case <-s.notify:
		case <-s.ctx.Done():
			s.fail(s.ctx.Err())
			return
		}
		for {
			if err := s.ctx.Err(); err != nil {
				s.fail(err)
				return
			}
			s.mu.Lock()
			if s.firstError != nil || len(s.pending) == 0 {
				closed := s.closed
				s.mu.Unlock()
				if closed {
					return
				}
				break
			}
			chunk := s.pending[0]
			s.pending[0] = DeltaChunk{}
			s.pending = s.pending[1:]
			if len(s.pending) == 0 {
				s.pending = nil
			}
			s.mu.Unlock()
			if deliverFn != nil {
				if err := deliverFn(chunk); err != nil {
					s.fail(err)
					return
				}
			}
			s.mu.Lock()
			s.bufferedBytes -= len(chunk.Text)
			s.bufferedEvents--
			s.mu.Unlock()
		}
	}
}
