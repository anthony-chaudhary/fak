package fabricmap

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestReserveAtomicallyChoosesCapacityAwareRouteAndRollsBack(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "src"}, {ID: "mid"}, {ID: "alt"}, {ID: "dst"}}, Links: []Link{
		{ID: "fast-a", From: "src", To: "mid", Transport: "fabric", Cost: 1, ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "nic-fast"},
		{ID: "fast-b", From: "mid", To: "dst", Transport: "fabric", Cost: 1, ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "pcie-fast"},
		{ID: "slow-a", From: "src", To: "alt", Transport: "fabric", Cost: 3, ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "nic-slow"},
		{ID: "slow-b", From: "alt", To: "dst", Transport: "fabric", Cost: 3, ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "pcie-slow"},
	}}
	a, err := NewAllocator(graph)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	first, err := a.Reserve(ReservationRequest{ID: "first", Route: Request{From: "src", To: "dst"}, BandwidthBytesPerSecond: 8, Now: now, TTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(first.Route); got != "fast-a,fast-b" {
		t.Fatalf("first route = %s", got)
	}
	second, err := a.Reserve(ReservationRequest{ID: "second", Route: Request{From: "src", To: "dst"}, BandwidthBytesPerSecond: 8, Now: now, TTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(second.Route); got != "slow-a,slow-b" {
		t.Fatalf("second route = %s", got)
	}

	before := a.Capacity(now)
	_, err = a.Reserve(ReservationRequest{ID: "refused", Route: Request{From: "src", To: "dst"}, BandwidthBytesPerSecond: 3, Now: now, TTL: time.Minute})
	if !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("error = %v, want insufficient capacity", err)
	}
	after := a.Capacity(now)
	if fmt.Sprint(before.Resources) != fmt.Sprint(after.Resources) {
		t.Fatalf("partial refusal leaked capacity: before=%v after=%v", before.Resources, after.Resources)
	}
}

func TestSharedResourceDirectionalConcurrencyIsAtomicAndAsymmetric(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "disk"}, {ID: "gpu"}, {ID: "host"}}, Links: []Link{
		{ID: "upload", From: "disk", To: "gpu", Transport: "fabric", Cost: 1, ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "pcie-switch"},
		{ID: "download-a", From: "gpu", To: "host", Transport: "fabric", Cost: 1, ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "pcie-switch"},
		{ID: "download-b", From: "host", To: "disk", Transport: "fabric", Cost: 1, ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "storage-bus"},
	}}
	a, err := NewAllocator(graph)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(200, 0)
	start := make(chan struct{})
	type result struct {
		r   Reservation
		err error
	}
	results := make(chan result, 2)
	reserve := func(id, from, to string) {
		<-start
		r, err := a.Reserve(ReservationRequest{ID: id, Route: Request{From: from, To: to}, BandwidthBytesPerSecond: 6, Now: now, TTL: time.Minute})
		results <- result{r, err}
	}
	go reserve("forward", "disk", "gpu")
	go reserve("reverse", "gpu", "disk")
	close(start)
	got := []result{<-results, <-results}
	successes := 0
	for _, result := range got {
		if result.err == nil {
			successes++
			continue
		}
		if !errors.Is(result.err, ErrInsufficientCapacity) {
			t.Fatalf("unexpected error: %v", result.err)
		}
	}
	if successes != 1 {
		t.Fatalf("successes = %d, want exactly one atomic winner: %#v", successes, got)
	}
	for _, result := range got {
		if result.err != nil {
			continue
		}
		ids := linkIDs(result.r.Route)
		if result.r.Request.Route.From == "disk" && ids != "upload" {
			t.Fatalf("forward route invented direction: %s", ids)
		}
		if result.r.Request.Route.From == "gpu" && ids != "download-a,download-b" {
			t.Fatalf("reverse route lost asymmetry: %s", ids)
		}
	}
}

func TestReleaseExpiryAndAdmissionOrderAreDeterministic(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "a"}, {ID: "b"}}, Links: []Link{{ID: "a-b", From: "a", To: "b", Transport: "fabric", ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "nic"}}}
	a, err := NewAllocator(graph)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(300, 0)
	first, err := a.Reserve(ReservationRequest{ID: "first", Route: Request{From: "a", To: "b"}, BandwidthBytesPerSecond: 10, Now: now, TTL: 2 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	if first.AdmissionOrder != 1 {
		t.Fatalf("order = %d", first.AdmissionOrder)
	}
	if a.Release("first") != true || a.Release("first") != false {
		t.Fatal("release is not deterministic/idempotent")
	}
	second, err := a.Reserve(ReservationRequest{ID: "second", Route: Request{From: "a", To: "b"}, BandwidthBytesPerSecond: 10, Now: now, TTL: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	if second.AdmissionOrder != 2 {
		t.Fatalf("order = %d", second.AdmissionOrder)
	}
	expired := a.Expire(now.Add(time.Second))
	if fmt.Sprint(expired) != "[second]" {
		t.Fatalf("expired = %v", expired)
	}
	if _, err := a.Reserve(ReservationRequest{ID: "third", Route: Request{From: "a", To: "b"}, BandwidthBytesPerSecond: 10, Now: now.Add(time.Second), TTL: time.Second}); err != nil {
		t.Fatal(err)
	}
}

func TestReservationRaceWitnessNeverOversubscribes(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "a"}, {ID: "b"}}, Links: []Link{{ID: "a-b", From: "a", To: "b", Transport: "fabric", ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "nic"}}}
	a, err := NewAllocator(graph)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(400, 0)
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := fmt.Sprintf("r-%03d", i)
			r, err := a.Reserve(ReservationRequest{ID: id, Route: Request{From: "a", To: "b"}, BandwidthBytesPerSecond: 1, Now: now, TTL: time.Minute})
			if err == nil {
				a.Release(r.ID)
			} else if !errors.Is(err, ErrInsufficientCapacity) {
				t.Errorf("reserve: %v", err)
			}
		}(i)
	}
	wg.Wait()
	snapshot := a.Capacity(now)
	if got := snapshot.Resources["nic"].ReservedBandwidthBytesPerSecond; got != 0 {
		t.Fatalf("reserved = %d", got)
	}
}

func TestSharedResourceIsDebitedOnceAcrossMultipleRouteHops(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "a"}, {ID: "b"}, {ID: "c"}}, Links: []Link{
		{ID: "a-b", From: "a", To: "b", Transport: "fabric", ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "switch"},
		{ID: "b-c", From: "b", To: "c", Transport: "fabric", ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "switch"},
	}}
	a, err := NewAllocator(graph)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(500, 0)
	if _, err := a.Reserve(ReservationRequest{ID: "flow", Route: Request{From: "a", To: "c"}, BandwidthBytesPerSecond: 6, Now: now, TTL: time.Minute}); err != nil {
		t.Fatal(err)
	}
	if got := a.Capacity(now).Resources["switch"].ReservedBandwidthBytesPerSecond; got != 6 {
		t.Fatalf("shared switch reserved = %d, want one 6-byte/s flow debit", got)
	}
	if _, err := a.Reserve(ReservationRequest{ID: "blocked", Route: Request{From: "a", To: "c"}, BandwidthBytesPerSecond: 5, Now: now, TTL: time.Minute}); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("second flow error = %v, want shared-resource refusal", err)
	}
}

func TestAllocatorRejectsConflictingSharedResourceCapacity(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "a"}, {ID: "b"}}, Links: []Link{
		{ID: "forward", From: "a", To: "b", Transport: "fabric", ReservableBandwidthBytesPerSecond: 10, SharedResourceID: "nic"},
		{ID: "reverse", From: "b", To: "a", Transport: "fabric", ReservableBandwidthBytesPerSecond: 9, SharedResourceID: "nic"},
	}}
	if _, err := NewAllocator(graph); err == nil {
		t.Fatal("conflicting shared-resource capacities accepted")
	}
}
