package stepspan

import (
	"fmt"
	"io"
	"sync"
	"sync/atomic"
)

type Sink interface {
	WriteSpan(Span) error
	Close() error
}

// Emitter never performs sink I/O on the caller's path. Close drains accepted spans.
type Emitter struct {
	queue     chan Span
	stop      chan struct{}
	done      chan struct{}
	gate      sync.RWMutex
	closed    bool
	closeOnce sync.Once
	sink      Sink
	dropped   [9]atomic.Uint64
	failures  atomic.Uint64
	err       error
}

var planes = [...]Plane{Ops, Agent, Planner, Router, Gateway, Sched, Engine, Kernel, "unknown"}

func planeIndex(p Plane) int {
	for i, v := range planes[:8] {
		if p == v {
			return i
		}
	}
	return 8
}
func NewEmitter(capacity int, sink Sink) (*Emitter, error) {
	if capacity <= 0 || sink == nil {
		return nil, ErrInvalidSpan
	}
	e := &Emitter{queue: make(chan Span, capacity), stop: make(chan struct{}), done: make(chan struct{}), sink: sink}
	go e.run()
	return e, nil
}

// Emit returns false when the bounded queue is full, closing, or contended.
// Callers retain ownership of their slices after Emit returns.
func (e *Emitter) Emit(s Span) bool {
	i := planeIndex(s.Plane)
	if !e.gate.TryRLock() {
		e.dropped[i].Add(1)
		return false
	}
	defer e.gate.RUnlock()
	if e.closed {
		e.dropped[i].Add(1)
		return false
	}
	s.Links = append([]Link(nil), s.Links...)
	s.IDs.SeqIDs = append([]string(nil), s.IDs.SeqIDs...)
	select {
	case e.queue <- s:
		return true
	default:
		e.dropped[i].Add(1)
		return false
	}
}
func (e *Emitter) run() {
	defer close(e.done)
	write := func(s Span) {
		if err := e.sink.WriteSpan(s); err != nil {
			e.failures.Add(1)
			if e.err == nil {
				e.err = err
			}
		}
	}
	for {
		select {
		case s := <-e.queue:
			write(s)
		case <-e.stop:
			for {
				select {
				case s := <-e.queue:
					write(s)
				default:
					if err := e.sink.Close(); e.err == nil {
						e.err = err
					}
					return
				}
			}
		}
	}
}
func (e *Emitter) Close() error {
	e.closeOnce.Do(func() { e.gate.Lock(); e.closed = true; close(e.stop); e.gate.Unlock() })
	<-e.done
	return e.err
}
func (e *Emitter) Dropped(p Plane) uint64 { return e.dropped[planeIndex(p)].Load() }
func (e *Emitter) SinkFailures() uint64   { return e.failures.Load() }
func (e *Emitter) WritePrometheus(w io.Writer) error {
	if _, err := fmt.Fprintln(w, "# HELP fak_stepspan_dropped_total Spans rejected by the bounded emitter.\n# TYPE fak_stepspan_dropped_total counter"); err != nil {
		return err
	}
	for i, p := range planes {
		if _, err := fmt.Fprintf(w, "fak_stepspan_dropped_total{plane=%q} %d\n", p, e.dropped[i].Load()); err != nil {
			return err
		}
	}
	return nil
}
