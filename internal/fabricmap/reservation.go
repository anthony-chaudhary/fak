package fabricmap

import (
	"container/heap"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

var (
	ErrInsufficientCapacity = errors.New("fabricmap: insufficient reservable capacity")
	ErrDuplicateReservation = errors.New("fabricmap: reservation ID already exists")
)

type ReservationRequest struct {
	ID                      string
	Route                   Request
	BandwidthBytesPerSecond uint64
	Now                     time.Time
	TTL                     time.Duration
}

type Reservation struct {
	ID             string
	Request        ReservationRequest
	Route          Route
	ExpiresAt      time.Time
	AdmissionOrder uint64
	resources      map[string]uint64
}

type ResourceCapacity struct {
	CapacityBytesPerSecond          uint64
	ReservedBandwidthBytesPerSecond uint64
}

type CapacitySnapshot struct {
	At        time.Time
	Resources map[string]ResourceCapacity
}

type resourceState struct{ capacity, reserved uint64 }

type Allocator struct {
	mu                             sync.Mutex
	cond                           *sync.Cond
	graph                          Graph
	resources                      map[string]*resourceState
	reservations                   map[string]Reservation
	nextTicket, serving, nextOrder uint64
}

func NewAllocator(graph Graph) (*Allocator, error) {
	if err := graph.Validate(); err != nil {
		return nil, err
	}
	a := &Allocator{graph: graph, resources: make(map[string]*resourceState), reservations: make(map[string]Reservation)}
	a.cond = sync.NewCond(&a.mu)
	for _, link := range graph.Links {
		if link.ReservableBandwidthBytesPerSecond == 0 {
			continue
		}
		id := resourceID(link)
		if existing, ok := a.resources[id]; ok {
			if existing.capacity != link.ReservableBandwidthBytesPerSecond {
				return nil, fmt.Errorf("fabricmap: shared resource %q has conflicting capacities %d and %d", id, existing.capacity, link.ReservableBandwidthBytesPerSecond)
			}
			continue
		}
		a.resources[id] = &resourceState{capacity: link.ReservableBandwidthBytesPerSecond}
	}
	return a, nil
}

// Reserve assigns a FIFO ticket, then selects and debits a route under one lock.
func (a *Allocator) Reserve(request ReservationRequest) (Reservation, error) {
	a.mu.Lock()
	ticket := a.nextTicket
	a.nextTicket++
	for ticket != a.serving {
		a.cond.Wait()
	}
	defer func() { a.serving++; a.cond.Broadcast(); a.mu.Unlock() }()
	if request.ID == "" || request.BandwidthBytesPerSecond == 0 || request.TTL <= 0 || request.Now.IsZero() {
		return Reservation{}, errors.New("fabricmap: reservation requires ID, bandwidth, explicit Now, and positive TTL")
	}
	a.expireLocked(request.Now)
	if _, exists := a.reservations[request.ID]; exists {
		return Reservation{}, fmt.Errorf("%w: %s", ErrDuplicateReservation, request.ID)
	}
	if _, err := a.graph.Plan(request.Route); err != nil {
		return Reservation{}, err
	}
	route, debits, err := a.selectReservableRouteLocked(request.Route, request.BandwidthBytesPerSecond)
	if err != nil {
		return Reservation{}, err
	}
	for id, amount := range debits {
		a.resources[id].reserved += amount
	}
	a.nextOrder++
	reservation := Reservation{ID: request.ID, Request: request, Route: route, ExpiresAt: request.Now.Add(request.TTL), AdmissionOrder: a.nextOrder, resources: cloneDebits(debits)}
	a.reservations[request.ID] = reservation
	return reservation, nil
}

func (a *Allocator) Release(id string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	reservation, ok := a.reservations[id]
	if !ok {
		return false
	}
	a.releaseLocked(reservation)
	return true
}

func (a *Allocator) Expire(now time.Time) []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.expireLocked(now)
}

func (a *Allocator) Capacity(now time.Time) CapacitySnapshot {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.expireLocked(now)
	out := CapacitySnapshot{At: now, Resources: make(map[string]ResourceCapacity, len(a.resources))}
	for id, state := range a.resources {
		out.Resources[id] = ResourceCapacity{CapacityBytesPerSecond: state.capacity, ReservedBandwidthBytesPerSecond: state.reserved}
	}
	return out
}

func (a *Allocator) releaseLocked(reservation Reservation) {
	for id, amount := range reservation.resources {
		a.resources[id].reserved -= amount
	}
	delete(a.reservations, reservation.ID)
}

func (a *Allocator) expireLocked(now time.Time) []string {
	var expired []Reservation
	for _, reservation := range a.reservations {
		if !reservation.ExpiresAt.After(now) {
			expired = append(expired, reservation)
		}
	}
	sort.Slice(expired, func(i, j int) bool {
		if !expired[i].ExpiresAt.Equal(expired[j].ExpiresAt) {
			return expired[i].ExpiresAt.Before(expired[j].ExpiresAt)
		}
		if expired[i].AdmissionOrder != expired[j].AdmissionOrder {
			return expired[i].AdmissionOrder < expired[j].AdmissionOrder
		}
		return expired[i].ID < expired[j].ID
	})
	ids := make([]string, 0, len(expired))
	for _, reservation := range expired {
		ids = append(ids, reservation.ID)
		a.releaseLocked(reservation)
	}
	return ids
}

func resourceID(link Link) string {
	if link.SharedResourceID != "" {
		return link.SharedResourceID
	}
	return link.ID
}
func cloneDebits(in map[string]uint64) map[string]uint64 {
	out := make(map[string]uint64, len(in))
	for id, amount := range in {
		out[id] = amount
	}
	return out
}

func cloneVisited(in map[string]bool) map[string]bool {
	out := make(map[string]bool, len(in))
	for id, seen := range in {
		out[id] = seen
	}
	return out
}

type flowState struct {
	visited    map[string]bool
	node       string
	links      []Link
	debits     map[string]uint64
	bottleneck uint64
	score
}
type flowQueue []*flowState

func (q flowQueue) Len() int           { return len(q) }
func (q flowQueue) Less(i, j int) bool { return q[i].score.less(q[j].score) }
func (q flowQueue) Swap(i, j int)      { q[i], q[j] = q[j], q[i] }
func (q *flowQueue) Push(x any)        { *q = append(*q, x.(*flowState)) }
func (q *flowQueue) Pop() any          { old := *q; n := len(old); x := old[n-1]; *q = old[:n-1]; return x }

func (a *Allocator) selectReservableRouteLocked(request Request, bandwidth uint64) (Route, map[string]uint64, error) {
	q := &flowQueue{}
	heap.Init(q)
	heap.Push(q, &flowState{node: request.From, bottleneck: ^uint64(0), debits: map[string]uint64{}, visited: map[string]bool{request.From: true}})
	for q.Len() > 0 {
		current := heap.Pop(q).(*flowState)
		if current.node == request.To {
			bottleneck := current.bottleneck
			if bottleneck == ^uint64(0) {
				bottleneck = 0
			}
			return Route{From: request.From, To: request.To, Links: current.links, TotalCost: current.cost, TotalLatencyNanos: current.latency, BottleneckBandwidthBytesPerSecond: bottleneck}, current.debits, nil
		}
		for _, link := range a.graph.Links {
			if link.From != current.node || current.visited[link.To] || !reservationLinkAllowed(link, request) || link.ReservableBandwidthBytesPerSecond == 0 {
				continue
			}
			debits := cloneDebits(current.debits)
			id := resourceID(link)
			if _, alreadyDebited := debits[id]; !alreadyDebited {
				debits[id] = bandwidth
			}
			state := a.resources[id]
			if state == nil || debits[id] > state.capacity-state.reserved {
				continue
			}
			links := appendCopy(current.links, link)
			nextScore := score{cost: current.cost + effectiveCost(link), latency: current.latency + link.LatencyNanos, hops: len(links), key: current.key + "\x00" + link.ID}
			visited := cloneVisited(current.visited)
			visited[link.To] = true
			heap.Push(q, &flowState{node: link.To, links: links, debits: debits, visited: visited, bottleneck: minBandwidth(current.bottleneck, link.BandwidthBytesPerSecond), score: nextScore})
		}
	}
	return Route{}, nil, ErrInsufficientCapacity
}

func reservationLinkAllowed(link Link, req Request) bool { return eligible(link, req) }
