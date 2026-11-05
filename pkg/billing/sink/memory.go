package sink

import (
	"sync"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

// Memory is an in-memory Backend: it keeps every appended record in a
// growable slice for tests and reference use. It is not durable; it
// exists so sinks can be exercised without a file or network.
type Memory struct {
	mu     sync.Mutex
	recs   []billing.UsageRecord
	closed bool
}

// NewMemory returns an empty Memory backend.
func NewMemory() *Memory { return &Memory{} }

// Append stores the batch in arrival order. It returns ErrClosed once
// Close has been called.
func (m *Memory) Append(batch []billing.UsageRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return ErrClosed
	}
	m.recs = append(m.recs, batch...)
	return nil
}

// Close marks the backend closed; later Appends fail with ErrClosed.
// It is idempotent.
func (m *Memory) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

// Records returns a copy of every record appended so far, in order.
func (m *Memory) Records() []billing.UsageRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]billing.UsageRecord, len(m.recs))
	copy(out, m.recs)
	return out
}
