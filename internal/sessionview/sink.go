package sessionview

import (
	"errors"
	"sync"
	"sync/atomic"
)

var (
	// ErrSinkClosed indicates the sink was closed and can no longer receive rows.
	ErrSinkClosed = errors.New("sink is closed")
	// ErrChannelFull indicates a non-blocking channel sink could not accept the row.
	ErrChannelFull = errors.New("channel sink buffer full")
)

// ViewSink receives materialized rows directly as they are processed by a MaterializedView.
// Row-level consumers like loggers, metric collectors, or audit sidecars implement this interface.
type ViewSink interface {
	ConsumeRow(row Row) error
}

// SinkFunc is an adapter allowing a plain function to act as a ViewSink.
type SinkFunc func(row Row) error

// ConsumeRow calls the underlying function.
func (f SinkFunc) ConsumeRow(row Row) error {
	return f(row)
}

// ViewSinkFunc is an alias for SinkFunc for naming compatibility.
type ViewSinkFunc = SinkFunc

// ChannelSink forwards consumed rows into a Go channel.
type ChannelSink struct {
	ch       chan Row
	dropFull bool
	closed   atomic.Bool
}

// NewChannelSink creates a ChannelSink with the specified buffer capacity.
// If dropFull is true, attempts to send to a full channel drop the row silently;
// otherwise ErrChannelFull is returned.
func NewChannelSink(bufferSize int, dropFull bool) *ChannelSink {
	if bufferSize < 0 {
		bufferSize = 0
	}
	return &ChannelSink{
		ch:       make(chan Row, bufferSize),
		dropFull: dropFull,
	}
}

// ConsumeRow enqueues the row onto the channel.
func (s *ChannelSink) ConsumeRow(row Row) error {
	if s.closed.Load() {
		return ErrSinkClosed
	}
	if s.dropFull {
		select {
		case s.ch <- row:
			return nil
		default:
			return nil
		}
	}
	select {
	case s.ch <- row:
		return nil
	default:
		return ErrChannelFull
	}
}

// Channel returns the receive-only channel.
func (s *ChannelSink) Channel() <-chan Row {
	return s.ch
}

// Close marks the sink closed and closes the underlying channel.
func (s *ChannelSink) Close() {
	if s.closed.CompareAndSwap(false, true) {
		close(s.ch)
	}
}

// SliceSink safely accumulates all received rows in-memory for testing, auditing,
// or inspection.
type SliceSink struct {
	mu   sync.Mutex
	rows []Row
}

// NewSliceSink creates an empty SliceSink.
func NewSliceSink() *SliceSink {
	return &SliceSink{
		rows: make([]Row, 0),
	}
}

// ConsumeRow records the row in the in-memory slice.
func (s *SliceSink) ConsumeRow(row Row) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rows = append(s.rows, row)
	return nil
}

// Rows returns a shallow copy of all accumulated rows.
func (s *SliceSink) Rows() []Row {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Row, len(s.rows))
	copy(out, s.rows)
	return out
}

// Count returns the number of rows received so far.
func (s *SliceSink) Count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.rows)
}

// Clear resets the accumulated rows slice.
func (s *SliceSink) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rows = make([]Row, 0)
}
