// Package flowcredit implements receiver-granted cumulative credit accounting
// for cross-node backpressure, pacing sender transmissions to consumer buffer capacity.
package flowcredit

import "sync"

// Lane identifies an isolated credit channel between receiver, sender, and traffic class.
type Lane struct {
	Receiver string
	Sender   string
	Class    string
}

// Snapshot reports point-in-time credit accounting values for a single lane.
type Snapshot struct {
	Granted   uint64
	Reserved  uint64
	Available uint64
	LastSeq   uint64
}

// Ledger tracks cumulative receiver grants and reservations per flow lane concurrently.
type Ledger struct {
	mu    sync.Mutex
	lanes map[Lane]*laneState
}

type laneState struct {
	granted  uint64
	reserved uint64
	lastSeq  uint64
}

// NewLedger returns an empty credit ledger with zero initial credit across all lanes.
func NewLedger() *Ledger {
	return &Ledger{lanes: make(map[Lane]*laneState)}
}

// Grant applies a cumulative receiver credit advertisement if seq exceeds the watermark.
func (g *Ledger) Grant(l Lane, seq, cumulative uint64) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	s := g.lane(l)
	if seq <= s.lastSeq {
		return false
	}
	s.lastSeq = seq
	if cumulative <= s.granted {
		return false
	}
	s.granted = cumulative
	return true
}

// TryReserve attempts to deduct n credit units from the available window.
func (g *Ledger) TryReserve(l Lane, n uint64) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	s := g.lane(l)
	if n > s.granted-s.reserved {
		return false
	}
	s.reserved += n
	return true
}

// Rollback releases up to n previously reserved units back to available credit.
func (g *Ledger) Rollback(l Lane, n uint64) uint64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	s := g.lane(l)
	if n > s.reserved {
		n = s.reserved
	}
	s.reserved -= n
	return n
}

// View inspects the point-in-time credit allocations and watermark for a lane.
func (g *Ledger) View(l Lane) Snapshot {
	g.mu.Lock()
	defer g.mu.Unlock()
	s := g.lane(l)
	return Snapshot{
		Granted:   s.granted,
		Reserved:  s.reserved,
		Available: s.granted - s.reserved,
		LastSeq:   s.lastSeq,
	}
}

func (g *Ledger) lane(l Lane) *laneState {
	if g.lanes == nil {
		g.lanes = make(map[Lane]*laneState)
	}
	s, ok := g.lanes[l]
	if !ok {
		s = &laneState{}
		g.lanes[l] = s
	}
	return s
}
