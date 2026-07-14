package main

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestDispatchDiscoveryRegistrySharesWatchAndLastDropCloses(t *testing.T) {
	var opened atomic.Int32
	var closed atomic.Int32
	updates := make(chan *runsSnapshot)
	initial := &runsSnapshot{now: time.Unix(1, 0)}
	open := func(ctx context.Context) (*runsSnapshot, <-chan *runsSnapshot) {
		opened.Add(1)
		go func() {
			<-ctx.Done()
			closed.Add(1)
		}()
		return initial, updates
	}

	registry := &dispatchDiscoveryRegistry{}
	const n = 8
	subs := make([]*dispatchDiscoverySubscription, n)
	for i := range subs {
		subs[i] = registry.Subscribe("control-plane-a", open)
		if got := <-subs[i].Snapshots; got != initial {
			t.Fatalf("subscriber %d initial snapshot = %p, want %p", i, got, initial)
		}
	}
	if got := opened.Load(); got != 1 {
		t.Fatalf("upstream watches opened = %d, want 1", got)
	}

	first := &runsSnapshot{now: time.Unix(2, 0)}
	second := &runsSnapshot{now: time.Unix(3, 0)}
	for _, update := range []*runsSnapshot{first, second} {
		events := make(chan *runsSnapshot, n)
		for _, sub := range subs {
			go func(sub *dispatchDiscoverySubscription) { events <- <-sub.Events }(sub)
		}
		updates <- update
		for i := 0; i < n; i++ {
			if got := <-events; got != update {
				t.Fatalf("event = %p, want %p", got, update)
			}
		}
	}
	for i, sub := range subs {
		if got := <-sub.Snapshots; got != second {
			t.Fatalf("subscriber %d coalesced snapshot = %p, want latest %p", i, got, second)
		}
	}

	for i := 0; i < n-1; i++ {
		subs[i].Close()
	}
	if got := closed.Load(); got != 0 {
		t.Fatalf("upstream closed before last drop: %d", got)
	}
	subs[n-1].Close()
	deadline := time.Now().Add(time.Second)
	for closed.Load() != 1 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := closed.Load(); got != 1 {
		t.Fatalf("upstream closes = %d after last drop, want 1", got)
	}
}

func TestSubscribeDispatchWaveDiscoveryScansOnceForNDeciders(t *testing.T) {
	root := t.TempDir()
	origGlob := fsGlob
	t.Cleanup(func() { fsGlob = origGlob })
	var globs atomic.Int32
	fsGlob = func(pattern string) ([]string, error) {
		globs.Add(1)
		return origGlob(pattern)
	}

	subs := subscribeDispatchWaveDiscovery(root, 6)
	defer closeDispatchDiscoverySubscriptions(subs)
	for i, sub := range subs {
		if snapshot := <-sub.Snapshots; snapshot == nil {
			t.Fatalf("subscriber %d received nil snapshot", i)
		}
	}
	// scanRunsSnapshot has two glob views (.log and .witness); N independent deciders
	// would pay 2*N. The coalesced source pays exactly one scan.
	if got := globs.Load(); got != 2 {
		t.Fatalf("discovery glob calls = %d, want 2 (one scan shared across N deciders)", got)
	}
}
