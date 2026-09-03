package control

import (
	"sync"
	"time"
)

// Standard audit event names.
const (
	EventSystemConfigAutomaticRollback = "SYSTEM_CONFIG_AUTOMATIC_ROLLBACK"
	EventSystemConfigApplied           = "SYSTEM_CONFIG_APPLIED"
	EventSystemConfigCommittedLKG      = "SYSTEM_CONFIG_COMMITTED_LKG"
	EventSystemConfigRollbackManual    = "SYSTEM_CONFIG_ROLLBACK_MANUAL"
)

// AuditEvent represents a verifiable configuration change or rollback event.
type AuditEvent struct {
	Seq       uint64        `json:"seq"`
	Event     string        `json:"event"`
	Timestamp time.Time     `json:"timestamp"`
	FromEpoch uint64        `json:"from_epoch"`
	ToEpoch   uint64        `json:"to_epoch"`
	Trigger   string        `json:"trigger,omitempty"`
	Detail    string        `json:"detail,omitempty"`
	Config    ServingConfig `json:"config"`
}

// EventStream maintains a thread-safe, bounded in-memory journal of configuration audit events.
type EventStream struct {
	mu          sync.RWMutex
	capacity    int
	seq         uint64
	nextSubID   uint64
	events      []AuditEvent
	subscribers map[uint64]func(AuditEvent)
}

// NewEventStream creates an EventStream with the given buffer capacity.
func NewEventStream(capacity int) *EventStream {
	if capacity <= 0 {
		capacity = 1024
	}
	return &EventStream{
		capacity:    capacity,
		events:      make([]AuditEvent, 0, capacity),
		subscribers: make(map[uint64]func(AuditEvent)),
	}
}

// Append records a new audit event into the stream and notifies active subscribers.
func (es *EventStream) Append(ev AuditEvent) AuditEvent {
	es.mu.Lock()
	es.seq++
	ev.Seq = es.seq
	if ev.Timestamp.IsZero() {
		ev.Timestamp = time.Now().UTC()
	}

	if len(es.events) >= es.capacity {
		copy(es.events, es.events[1:])
		es.events[len(es.events)-1] = ev
	} else {
		es.events = append(es.events, ev)
	}

	subs := make([]func(AuditEvent), 0, len(es.subscribers))
	for _, fn := range es.subscribers {
		subs = append(subs, fn)
	}
	es.mu.Unlock()

	for _, sub := range subs {
		sub(ev)
	}
	return ev
}

// Snapshot returns a copy of all recorded events in chronological order.
func (es *EventStream) Snapshot() []AuditEvent {
	es.mu.RLock()
	defer es.mu.RUnlock()
	out := make([]AuditEvent, len(es.events))
	copy(out, es.events)
	return out
}

// Latest returns the most recently emitted audit event, or nil if none.
func (es *EventStream) Latest() *AuditEvent {
	es.mu.RLock()
	defer es.mu.RUnlock()
	if len(es.events) == 0 {
		return nil
	}
	ev := es.events[len(es.events)-1]
	return &ev
}

// Subscribe registers a listener for newly appended events.
func (es *EventStream) Subscribe(fn func(AuditEvent)) func() {
	es.mu.Lock()
	es.nextSubID++
	id := es.nextSubID
	es.subscribers[id] = fn
	es.mu.Unlock()

	return func() {
		es.mu.Lock()
		defer es.mu.Unlock()
		delete(es.subscribers, id)
	}
}
