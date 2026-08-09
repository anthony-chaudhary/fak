package accountobs

import (
	"net/http"
	"time"
)

// Harvester accumulates provider observations for the current process and
// persists their coalesced headers so another process can resume from them.
// Persistence is deliberately best-effort at the caller boundary: observing a
// provider response must never fail the response path.
type Harvester struct {
	Tracker *Tracker
	Store   Store
	Key     string
	Now     func() time.Time
}

// NewHarvester constructs one process-level accumulator backed by store.
func NewHarvester(store Store, key string) *Harvester {
	return &Harvester{Tracker: New(), Store: store, Key: key, Now: time.Now}
}

// Observe feeds both the in-process accumulator and the durable per-seat row.
// It returns a persistence error for diagnostics; callers on hot paths should
// intentionally fail open.
func (h *Harvester) Observe(status int, header http.Header) error {
	if h == nil {
		return nil
	}
	if h.Tracker == nil {
		h.Tracker = New()
	}
	h.Tracker.Observe(status, header)
	now := time.Now()
	if h.Now != nil {
		now = h.Now()
	}
	return h.Store.Observe(h.Key, status, header, now)
}

// Snapshot returns the current process's accumulated observation.
func (h *Harvester) Snapshot() Snapshot {
	if h == nil || h.Tracker == nil {
		return Snapshot{}
	}
	return h.Tracker.Snapshot()
}
