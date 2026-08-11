package fabricmap

import (
	"context"
	"errors"
	"testing"
	"time"
)

type staticProvider struct {
	snapshot Snapshot
	err      error
}

func (p staticProvider) Snapshot(context.Context) (Snapshot, error) { return p.snapshot, p.err }

func TestRefreshComposesHeterogeneousProvidersInArbitraryDirections(t *testing.T) {
	now := time.Date(2026, 8, 10, 12, 0, 0, 0, time.UTC)
	storage := staticProvider{snapshot: Snapshot{Provider: "storage", Generation: 1, ObservedAt: now, ValidUntil: now.Add(time.Minute), Endpoints: []Endpoint{{ID: "L3", Kind: "ssd"}, {ID: "nic", Kind: "fabric"}}, Links: []Link{{ID: "ssd-nic", From: "L3", To: "nic", Transport: "rdma", CPUPath: "bypass"}}}}
	compute := staticProvider{snapshot: Snapshot{Provider: "compute", Generation: 7, ObservedAt: now, ValidUntil: now.Add(time.Minute), Endpoints: []Endpoint{{ID: "nic", Kind: "fabric"}, {ID: "L1", Kind: "gpu-memory"}}, Links: []Link{{ID: "nic-gpu", From: "nic", To: "L1", Transport: "gpudirect", CPUPath: "bypass"}, {ID: "gpu-nic", From: "L1", To: "nic", Transport: "gpudirect", CPUPath: "bypass"}}}}
	var set SnapshotSet
	graph, err := set.Refresh(context.Background(), now, compute, storage)
	if err != nil {
		t.Fatal(err)
	}
	forward, err := graph.Plan(Request{From: "L3", To: "L1", AllowedCPUPaths: []string{"bypass"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(forward); got != "ssd-nic,nic-gpu" {
		t.Fatalf("L3 -> L1 = %s", got)
	}
	if _, err := graph.Plan(Request{From: "L1", To: "L3"}); !errors.Is(err, ErrNoRoute) {
		t.Fatalf("reverse error = %v, want no route without nic -> SSD link", err)
	}
}

func TestSnapshotSetRejectsConflictingStableIdentities(t *testing.T) {
	now := time.Now().UTC()
	var set SnapshotSet
	if err := set.Update(Snapshot{Provider: "a", Generation: 1, ObservedAt: now, Endpoints: []Endpoint{{ID: "shared", Kind: "nic"}}}, now); err != nil {
		t.Fatal(err)
	}
	if err := set.Update(Snapshot{Provider: "b", Generation: 1, ObservedAt: now, Endpoints: []Endpoint{{ID: "shared", Kind: "gpu"}}}, now); err != nil {
		t.Fatal(err)
	}
	if _, err := set.Graph(now); err == nil {
		t.Fatal("expected conflicting endpoint failure")
	}
}

func TestExpiredCapabilitiesAreRemovedBeforePlanning(t *testing.T) {
	now := time.Date(2026, 8, 10, 12, 0, 0, 0, time.UTC)
	var set SnapshotSet
	if err := set.Update(Snapshot{Provider: "fabric", Generation: 1, ObservedAt: now, ValidUntil: now.Add(time.Second), Endpoints: []Endpoint{{ID: "a"}, {ID: "b"}}, Links: []Link{{ID: "a-b", From: "a", To: "b", Transport: "test"}}}, now); err != nil {
		t.Fatal(err)
	}
	graph, err := set.Graph(now.Add(2 * time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if len(graph.Endpoints) != 0 || len(graph.Links) != 0 {
		t.Fatalf("expired graph = %+v", graph)
	}
}

func TestRefreshIsAtomicOnProviderFailure(t *testing.T) {
	now := time.Now().UTC()
	var set SnapshotSet
	good := staticProvider{snapshot: Snapshot{Provider: "good", Generation: 1, ObservedAt: now, Endpoints: []Endpoint{{ID: "old"}}}}
	if _, err := set.Refresh(context.Background(), now, good); err != nil {
		t.Fatal(err)
	}
	newer := staticProvider{snapshot: Snapshot{Provider: "good", Generation: 2, ObservedAt: now, Endpoints: []Endpoint{{ID: "new"}}}}
	bad := staticProvider{err: errors.New("discovery down")}
	if _, err := set.Refresh(context.Background(), now, newer, bad); err == nil {
		t.Fatal("expected refresh failure")
	}
	graph, err := set.Graph(now)
	if err != nil {
		t.Fatal(err)
	}
	if len(graph.Endpoints) != 1 || graph.Endpoints[0].ID != "old" {
		t.Fatalf("failed refresh mutated set: %+v", graph)
	}
}

func TestGenerationCannotChangeContentOrGoBackward(t *testing.T) {
	now := time.Now().UTC()
	var set SnapshotSet
	base := Snapshot{Provider: "p", Generation: 2, ObservedAt: now, Endpoints: []Endpoint{{ID: "a"}}}
	if err := set.Update(base, now); err != nil {
		t.Fatal(err)
	}
	changed := base
	changed.Endpoints = []Endpoint{{ID: "b"}}
	if err := set.Update(changed, now); err == nil {
		t.Fatal("same generation changed content")
	}
	older := base
	older.Generation = 1
	if err := set.Update(older, now); err == nil {
		t.Fatal("older generation accepted")
	}
}
