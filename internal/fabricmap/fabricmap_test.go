package fabricmap

import (
	"encoding/json"
	"errors"
	"math"
	"strings"
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

func TestPlanEstimatedReadyTimeChangesWithByteSize(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "source"}, {ID: "sink"}}, Links: []Link{
		{ID: "low-latency-slow", From: "source", To: "sink", Transport: "slow", Cost: 1, LatencyNanos: 1, BandwidthBytesPerSecond: 1_000},
		{ID: "higher-latency-fast", From: "source", To: "sink", Transport: "fast", Cost: 2, LatencyNanos: 2_000_000, BandwidthBytesPerSecond: 1_000_000_000},
	}}

	compat, err := graph.Plan(Request{From: "source", To: "sink", Bytes: 1_000_000_000})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(compat); got != "low-latency-slow" || compat.Objective != RouteObjectiveStaticCost {
		t.Fatalf("compatibility route = %s objective=%q, want static low-cost route", got, compat.Objective)
	}

	small, err := graph.Plan(Request{From: "source", To: "sink", Bytes: 1, Objective: RouteObjectiveEstimatedReadyTime})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(small); got != "low-latency-slow" {
		t.Fatalf("small route = %s, want low-latency-slow", got)
	}
	large, err := graph.Plan(Request{From: "source", To: "sink", Bytes: 1_000_000_000, Objective: RouteObjectiveEstimatedReadyTime})
	if err != nil {
		t.Fatal(err)
	}
	if got := linkIDs(large); got != "higher-latency-fast" {
		t.Fatalf("large route = %s, want higher-latency-fast", got)
	}
	if large.EstimatedReadyTimeNanos != 1_002_000_000 {
		t.Fatalf("large estimated ready time = %d, want 1002000000", large.EstimatedReadyTimeNanos)
	}
}

func TestPlanEstimatedReadyTimeSumsRoundedHopTimes(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "a"}, {ID: "b"}, {ID: "c"}}, Links: []Link{
		{ID: "ab", From: "a", To: "b", Transport: "one", LatencyNanos: 5, BandwidthBytesPerSecond: 4},
		{ID: "bc", From: "b", To: "c", Transport: "two", LatencyNanos: 7, BandwidthBytesPerSecond: 6},
	}}
	route, err := graph.Plan(Request{From: "a", To: "c", Bytes: 10, Objective: RouteObjectiveEstimatedReadyTime})
	if err != nil {
		t.Fatal(err)
	}
	if route.EstimatedReadyTimeNanos != 4_166_666_679 {
		t.Fatalf("estimated ready time = %d, want 4166666679", route.EstimatedReadyTimeNanos)
	}
}

func TestPlanEstimatedReadyTimeFailsClosed(t *testing.T) {
	base := Graph{Endpoints: []Endpoint{{ID: "a"}, {ID: "b"}}}
	tests := []struct {
		name string
		req  Request
		link Link
		want error
	}{
		{name: "missing bytes", req: Request{From: "a", To: "b", Objective: RouteObjectiveEstimatedReadyTime}, link: Link{ID: "ab", From: "a", To: "b", Transport: "x", BandwidthBytesPerSecond: 1}, want: ErrReadyTimeBytesRequired},
		{name: "unknown bandwidth", req: Request{From: "a", To: "b", Bytes: 1, Objective: RouteObjectiveEstimatedReadyTime}, link: Link{ID: "ab", From: "a", To: "b", Transport: "x"}, want: ErrReadyTimeBandwidth},
		{name: "overflow", req: Request{From: "a", To: "b", Bytes: math.MaxUint64, Objective: RouteObjectiveEstimatedReadyTime}, link: Link{ID: "ab", From: "a", To: "b", Transport: "x", BandwidthBytesPerSecond: 1}, want: ErrReadyTimeOverflow},
		{name: "unknown objective", req: Request{From: "a", To: "b", Bytes: 1, Objective: "measured_latency"}, link: Link{ID: "ab", From: "a", To: "b", Transport: "x", BandwidthBytesPerSecond: 1}, want: ErrUnknownRouteObjective},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			graph := base
			graph.Links = []Link{tc.link}
			_, err := graph.Plan(tc.req)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestPlanReadyTimeRouteJSONNamesEstimateInputs(t *testing.T) {
	graph := Graph{Endpoints: []Endpoint{{ID: "a"}, {ID: "b"}}, Links: []Link{{ID: "ab", From: "a", To: "b", Transport: "x", LatencyNanos: 3, BandwidthBytesPerSecond: 1_000_000_000}}}
	route, err := graph.Plan(Request{From: "a", To: "b", Bytes: 7, Objective: RouteObjectiveEstimatedReadyTime})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(route)
	if err != nil {
		t.Fatal(err)
	}
	got := string(raw)
	for _, field := range []string{`"bytes":7`, `"objective":"estimated_ready_time"`, `"estimated_ready_time_nanos":10`} {
		if !strings.Contains(got, field) {
			t.Fatalf("route JSON = %s, want %s", got, field)
		}
	}
	if strings.Contains(got, "measured") {
		t.Fatalf("route JSON labels an estimate as a measurement: %s", got)
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
