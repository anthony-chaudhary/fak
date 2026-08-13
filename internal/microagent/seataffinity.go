package microagent

import (
	"errors"
	"hash/fnv"
	"sync"
	"sync/atomic"
)

var ErrNoSeatAvailable = errors.New("microagent: no provider seat available")

type Seat struct {
	ID        string
	Scheduler *Scheduler
}

type SeatLease struct {
	SeatID  string
	release func()
	once    sync.Once
}

func (l *SeatLease) Release() {
	if l != nil {
		l.once.Do(l.release)
	}
}

// SeatPool chooses among independently bounded provider seats. A non-empty
// affinity key prefers one stable seat; a busy preferred seat falls through to
// another available seat. Empty affinity rotates the first probe for ordinary
// capacity-based admission.
type SeatPool struct {
	seats []Seat
	next  atomic.Uint64
}

func NewSeatPool(seats []Seat) (*SeatPool, error) {
	if len(seats) == 0 {
		return nil, errors.New("microagent: seat pool requires at least one seat")
	}
	seen := make(map[string]struct{}, len(seats))
	copySeats := append([]Seat(nil), seats...)
	for _, seat := range copySeats {
		if seat.ID == "" || seat.Scheduler == nil {
			return nil, errors.New("microagent: seat requires non-empty ID and scheduler")
		}
		if _, ok := seen[seat.ID]; ok {
			return nil, errors.New("microagent: duplicate seat ID")
		}
		seen[seat.ID] = struct{}{}
	}
	return &SeatPool{seats: copySeats}, nil
}

func (p *SeatPool) TryAcquire(affinityKey string) (*SeatLease, error) {
	if p == nil || len(p.seats) == 0 {
		return nil, ErrNoSeatAvailable
	}
	start := p.start(affinityKey)
	for offset := 0; offset < len(p.seats); offset++ {
		seat := p.seats[(start+offset)%len(p.seats)]
		if release, ok := seat.Scheduler.TryAcquire(); ok {
			return &SeatLease{SeatID: seat.ID, release: release}, nil
		}
	}
	return nil, ErrNoSeatAvailable
}

func (p *SeatPool) start(key string) int {
	if key == "" {
		return int((p.next.Add(1) - 1) % uint64(len(p.seats)))
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(key))
	return int(h.Sum64() % uint64(len(p.seats)))
}
