package fabricmap

import (
	"errors"
	"testing"
)

func TestPlanSupportsArbitraryDirectionsAndAsymmetricLinks(t *testing.T) {
	graph := Graph{
		Endpoints: []Endpoint{{ID: "L1", Kind: "gpu-memory"}, {ID: "L2", Kind: "host-memory"}, {ID: "L3", Kind: "ssd"}},
		Links: []Link{
			{ID: "ssd-gpu-direct", From: "L3", To: "L1", Transport: "gds", Cost: 1, CPUPath: "bypass", BandwidthBytesPerSecond: 10, Labels: map[string]string{"dma": "gpu-direct"}},
			{ID: "gpu-host", From: "L1", To: "L2", Transport: "pcie", Cost: 1, CPUPath: "copy", BandwidthBytesPerSecond: 8},
			{ID: "host-ssd", From: "L2", To: "L3", Transport: "nvme", Cost: 1, CPUPath: "copy", BandwidthBytesPerSecond: 6},
			{ID: "ssd-host", From: "L3", To: "L2", Transport: "nvme", Cost: 9, CPUPath: "copy", BandwidthBytesPerSecond: 6},
		},
	}
	direct, err := graph.Plan(Request{From: "L3", To: "L1", AllowedCPUPaths: []string{"bypass"}, RequiredLinkLabels: map[string]string{"dma": "gpu-direct"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(direct); got != "ssd-gpu-direct" {
		t.Fatalf("L3 -> L1 route = %s", got)
	}

	reverse, err := graph.Plan(Request{From: "L1", To: "L3"})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(reverse); got != "gpu-host,host-ssd" {
		t.Fatalf("L1 -> L3 route = %s", got)
	}
	if reverse.BottleneckBandwidthBytesPerSecond != 6 {
		t.Fatalf("bottleneck = %d", reverse.BottleneckBandwidthBytesPerSecond)
	}

	lateral, err := graph.Plan(Request{From: "L3", To: "L2"})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(lateral); got != "ssd-gpu-direct,gpu-host" {
		t.Fatalf("L3 -> L2 route = %s; labels were incorrectly treated as hierarchy", got)
	}
}

func TestPlanDoesNotInventReverseCapability(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "disk"}, {ID: "gpu"}}, Links: []Link{{ID: "upload", From: "disk", To: "gpu", Transport: "rdma"}}}
	_, err := graph.Plan(Request{From: "gpu", To: "disk"})
	if !errors.Is(err, ErrNoRoute) {
		t.Fatalf("error = %v, want ErrNoRoute", err)
	}
}

func TestPlanComposesOpenEndedCapabilities(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "object-store"}, {ID: "nic"}, {ID: "accelerator"}}, Links: []Link{
		{ID: "store-nic", From: "object-store", To: "nic", Transport: "future-fabric", Labels: map[string]string{"coherent": "yes"}},
		{ID: "nic-device", From: "nic", To: "accelerator", Transport: "future-fabric", Labels: map[string]string{"coherent": "yes"}},
	}}
	route, err := graph.Plan(Request{From: "object-store", To: "accelerator", AllowedTransports: []string{"future-fabric"}, RequiredLinkLabels: map[string]string{"coherent": "yes"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(route); got != "store-nic,nic-device" {
		t.Fatalf("route = %s", got)
	}
}

func linkIDs(route Route) string {
	out := ""
	for i, link := range route.Links {
		if i > 0 {
			out += ","
		}
		out += link.ID
	}
	return out
}
