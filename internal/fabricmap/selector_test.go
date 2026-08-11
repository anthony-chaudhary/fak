package fabricmap

import (
	"errors"
	"testing"
)

func contextualGraph() Graph {
	return Graph{Endpoints: []Endpoint{
		{ID: "ssd-west", Kind: "storage", Labels: map[string]string{"tier": "L3", "region": "west", "tenant": "a"}},
		{ID: "ssd-east", Kind: "storage", Labels: map[string]string{"tier": "L3", "region": "east", "tenant": "a"}},
		{ID: "gpu-west", Kind: "compute", Labels: map[string]string{"tier": "L1", "region": "west", "tenant": "a"}},
		{ID: "gpu-east", Kind: "compute", Labels: map[string]string{"tier": "L1", "region": "east", "tenant": "a"}},
		{ID: "host-west", Kind: "memory", Labels: map[string]string{"tier": "L2", "region": "west", "tenant": "a"}},
	}, Links: []Link{
		{ID: "west-direct", From: "ssd-west", To: "gpu-west", Transport: "gds", Cost: 1},
		{ID: "east-direct", From: "ssd-east", To: "gpu-east", Transport: "gds", Cost: 2},
		{ID: "gpu-host", From: "gpu-west", To: "host-west", Transport: "pcie", Cost: 1},
		{ID: "host-ssd", From: "host-west", To: "ssd-west", Transport: "nvme", Cost: 1},
		{ID: "ssd-host", From: "ssd-west", To: "host-west", Transport: "nvme", Cost: 1},
	}}

}
func TestSelectRouteSelectsManyToManyByMetadataAndRoute(t *testing.T) {
	mapping, err := contextualGraph().SelectRoute(SelectionRequest{From: EndpointSelector{Kind: "storage", Labels: map[string]string{"tenant": "a"}}, To: EndpointSelector{Kind: "compute", Labels: map[string]string{"tenant": "a"}}})
	if err != nil {
		t.Fatal(err)
	}
	if mapping.Source.ID != "ssd-west" || mapping.Destination.ID != "gpu-west" || linkIDs(mapping.Route) != "west-direct" {
		t.Fatalf("mapping = %+v", mapping)
	}
}

func TestSelectRouteSupportsReverseAndLateralHierarchyLabels(t *testing.T) {
	graph := contextualGraph()
	reverse, err := graph.SelectRoute(SelectionRequest{From: EndpointSelector{Labels: map[string]string{"tier": "L1", "region": "west"}}, To: EndpointSelector{Labels: map[string]string{"tier": "L3", "region": "west"}}})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(reverse.Route); got != "gpu-host,host-ssd" {
		t.Fatalf("L1 -> L3 = %s", got)
	}
	lateral, err := graph.SelectRoute(SelectionRequest{From: EndpointSelector{Labels: map[string]string{"tier": "L3", "region": "west"}}, To: EndpointSelector{Labels: map[string]string{"tier": "L2", "region": "west"}}})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(lateral.Route); got != "ssd-host" {
		t.Fatalf("L3 -> L2 = %s", got)
	}
}

func TestSelectRouteDoesNotInferMissingDirection(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "L3", Labels: map[string]string{"tier": "L3"}}, {ID: "L1", Labels: map[string]string{"tier": "L1"}}}, Links: []Link{{ID: "only-up", From: "L3", To: "L1", Transport: "x"}}}
	_, err := graph.SelectRoute(SelectionRequest{From: EndpointSelector{Labels: map[string]string{"tier": "L1"}}, To: EndpointSelector{Labels: map[string]string{"tier": "L3"}}})
	if !errors.Is(err, ErrNoRoute) {
		t.Fatalf("error = %v", err)
	}
}

func TestSelectRouteDeterministicallyReturnsConcretePair(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "a-source", Kind: "s"}, {ID: "b-source", Kind: "s"}, {ID: "a-target", Kind: "t"}, {ID: "b-target", Kind: "t"}}, Links: []Link{{ID: "route-a", From: "a-source", To: "a-target", Transport: "x", Cost: 1}, {ID: "route-b", From: "b-source", To: "b-target", Transport: "x", Cost: 1}}}
	first, err := graph.SelectRoute(SelectionRequest{From: EndpointSelector{Kind: "s"}, To: EndpointSelector{Kind: "t"}})
	if err != nil {
		t.Fatal(err)
	}
	if first.Source.ID != "a-source" || first.Destination.ID != "a-target" {
		t.Fatalf("mapping = %+v", first)
	}
	for i := 0; i < 20; i++ {
		again, err := graph.SelectRoute(SelectionRequest{From: EndpointSelector{Kind: "s"}, To: EndpointSelector{Kind: "t"}})
		if err != nil || again.Source.ID != first.Source.ID || again.Destination.ID != first.Destination.ID {
			t.Fatalf("iteration %d = %+v, %v", i, again, err)
		}
	}
}

func TestSelectRouteReportsWhichSelectorHasNoMatch(t *testing.T) {
	_, err := contextualGraph().SelectRoute(SelectionRequest{From: EndpointSelector{Kind: "missing"}, To: EndpointSelector{Kind: "compute"}})
	if !errors.Is(err, ErrNoEndpointMatch) {
		t.Fatalf("error = %v", err)
	}
}
