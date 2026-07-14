package main

import (
	"context"
	"path/filepath"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// dispatchDiscoveryRegistry coalesces discovery by endpoint. Every subscriber gets
// the latest snapshot (coalesced to one unread value) and every subsequent update
// on its lossless event feed. The last Close cancels the sole upstream watch.
type dispatchDiscoveryRegistry struct {
	mu      sync.Mutex
	sources map[string]*dispatchDiscoverySource
}

type dispatchDiscoveryOpen func(context.Context) (*runsSnapshot, <-chan *runsSnapshot)

type dispatchDiscoverySource struct {
	registry *dispatchDiscoveryRegistry
	key      string
	cancel   context.CancelFunc
	refs     int
	latest   *runsSnapshot
	nextID   int
	subs     map[int]*dispatchDiscoverySubscriber
}

type dispatchDiscoverySubscriber struct {
	ctx       context.Context
	cancel    context.CancelFunc
	snapshots chan *runsSnapshot
	events    chan *runsSnapshot
	inbox     chan *runsSnapshot
}

type dispatchDiscoverySubscription struct {
	Snapshots <-chan *runsSnapshot
	Events    <-chan *runsSnapshot
	close     func()
	once      sync.Once
}

func (s *dispatchDiscoverySubscription) Close() {
	if s != nil {
		s.once.Do(s.close)
	}
}

func (r *dispatchDiscoveryRegistry) Subscribe(key string, open dispatchDiscoveryOpen) *dispatchDiscoverySubscription {
	r.mu.Lock()
	if r.sources == nil {
		r.sources = make(map[string]*dispatchDiscoverySource)
	}
	source := r.sources[key]
	var initial *runsSnapshot
	var updates <-chan *runsSnapshot
	if source == nil {
		ctx, cancel := context.WithCancel(context.Background())
		initial, updates = open(ctx)
		source = &dispatchDiscoverySource{registry: r, key: key, cancel: cancel, latest: initial, subs: make(map[int]*dispatchDiscoverySubscriber)}
		r.sources[key] = source
		go source.forward(ctx, updates)
	}
	ctx, cancel := context.WithCancel(context.Background())
	id := source.nextID
	source.nextID++
	sub := &dispatchDiscoverySubscriber{
		ctx: ctx, cancel: cancel,
		snapshots: make(chan *runsSnapshot, 1),
		events:    make(chan *runsSnapshot),
		inbox:     make(chan *runsSnapshot),
	}
	go sub.forwardEvents()
	source.subs[id] = sub
	source.refs++
	if source.latest != nil {
		sub.snapshots <- source.latest
	}
	r.mu.Unlock()
	return &dispatchDiscoverySubscription{
		Snapshots: sub.snapshots,
		Events:    sub.events,
		close: func() {
			r.unsubscribe(source, id)
		},
	}
}

func (r *dispatchDiscoveryRegistry) unsubscribe(source *dispatchDiscoverySource, id int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sources[source.key] != source {
		return
	}
	sub, ok := source.subs[id]
	if !ok {
		return
	}
	delete(source.subs, id)
	source.refs--
	sub.cancel()
	if source.refs == 0 {
		delete(r.sources, source.key)
		source.cancel()
	}
}

func (s *dispatchDiscoverySource) forward(ctx context.Context, updates <-chan *runsSnapshot) {
	for {
		select {
		case <-ctx.Done():
			return
		case snapshot, ok := <-updates:
			if !ok {
				return
			}
			s.registry.publish(s, snapshot)
		}
	}
}

func (r *dispatchDiscoveryRegistry) publish(source *dispatchDiscoverySource, snapshot *runsSnapshot) {
	r.mu.Lock()
	if r.sources[source.key] != source {
		r.mu.Unlock()
		return
	}
	source.latest = snapshot
	subs := make([]*dispatchDiscoverySubscriber, 0, len(source.subs))
	for _, sub := range source.subs {
		select {
		case <-sub.snapshots:
		default:
		}
		sub.snapshots <- snapshot
		subs = append(subs, sub)
	}
	r.mu.Unlock()
	for _, sub := range subs {
		select {
		case sub.inbox <- snapshot:
		case <-sub.ctx.Done():
		}
	}
}

// forwardEvents is an unbounded FIFO between the upstream watch and one consumer.
// A slow subscriber therefore cannot drop events or stall delivery to its peers.
func (s *dispatchDiscoverySubscriber) forwardEvents() {
	var queue []*runsSnapshot
	for {
		var out chan *runsSnapshot
		var next *runsSnapshot
		if len(queue) > 0 {
			out = s.events
			next = queue[0]
		}
		select {
		case <-s.ctx.Done():
			return
		case snapshot := <-s.inbox:
			queue = append(queue, snapshot)
		case out <- next:
			queue = queue[1:]
		}
	}
}

func subscribeDispatchWaveDiscovery(root string, n int) []*dispatchDiscoverySubscription {
	registry := &dispatchDiscoveryRegistry{}
	subs := make([]*dispatchDiscoverySubscription, n)
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	open := func(ctx context.Context) (*runsSnapshot, <-chan *runsSnapshot) {
		updates := make(chan *runsSnapshot)
		go func() {
			<-ctx.Done()
			close(updates)
		}()
		return scanRunsSnapshot(runsDir, time.Now()), updates
	}
	for i := range subs {
		subs[i] = registry.Subscribe(runsDir, open)
	}
	return subs
}

func closeDispatchDiscoverySubscriptions(subs []*dispatchDiscoverySubscription) {
	for _, sub := range subs {
		sub.Close()
	}
}
