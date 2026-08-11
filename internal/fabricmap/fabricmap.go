package fabricmap

import (
	"container/heap"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Endpoint is an addressable place that can source or receive data. Kind and
// Labels are descriptive metadata, never an ordering rule: an endpoint named
// "L3" may be either side of a transfer to one named "L1".
type Endpoint struct {
	ID     string            `json:"id"`
	Kind   string            `json:"kind,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`
}

// Link is one directed transfer capability. Reverse traffic requires a second
// link and may use a different transport, cost, bandwidth, or CPU path.
type Link struct {
	ID                                string            `json:"id"`
	From                              string            `json:"from"`
	To                                string            `json:"to"`
	Transport                         string            `json:"transport"`
	Cost                              uint64            `json:"cost,omitempty"`
	LatencyNanos                      uint64            `json:"latency_nanos,omitempty"`
	BandwidthBytesPerSecond           uint64            `json:"bandwidth_bytes_per_second,omitempty"`
	ReservableBandwidthBytesPerSecond uint64            `json:"reservable_bandwidth_bytes_per_second,omitempty"`
	SharedResourceID                  string            `json:"shared_resource_id,omitempty"`
	CPUPath                           string            `json:"cpu_path,omitempty"`
	Labels                            map[string]string `json:"labels,omitempty"`
}

// Graph is a composable inventory. It deliberately has no built-in storage or
// memory hierarchy: endpoints and directed links are the complete topology.
type Graph struct {
	Endpoints []Endpoint `json:"endpoints"`
	Links     []Link     `json:"links"`
}

// Request describes one directional mapping. RequiredLinkLabels supports
// capabilities not known to this package (encryption, coherency, GPUDirect,
// persistence, a future transport, and so on).
type Request struct {
	From                       string            `json:"from"`
	To                         string            `json:"to"`
	Bytes                      uint64            `json:"bytes,omitempty"`
	MinBandwidthBytesPerSecond uint64            `json:"min_bandwidth_bytes_per_second,omitempty"`
	MaxLinkLatencyNanos        uint64            `json:"max_link_latency_nanos,omitempty"`
	AllowedCPUPaths            []string          `json:"allowed_cpu_paths,omitempty"`
	AllowedTransports          []string          `json:"allowed_transports,omitempty"`
	RequiredLinkLabels         map[string]string `json:"required_link_labels,omitempty"`
}

// Route is an ordered, directional transfer plan.
type Route struct {
	From                              string `json:"from"`
	To                                string `json:"to"`
	Links                             []Link `json:"links"`
	TotalCost                         uint64 `json:"total_cost"`
	TotalLatencyNanos                 uint64 `json:"total_latency_nanos"`
	BottleneckBandwidthBytesPerSecond uint64 `json:"bottleneck_bandwidth_bytes_per_second,omitempty"`
}

func (g Graph) Validate() error {
	ids := make(map[string]struct{}, len(g.Endpoints))
	for i, endpoint := range g.Endpoints {
		endpoint.ID = strings.TrimSpace(endpoint.ID)
		if endpoint.ID == "" {
			return fmt.Errorf("endpoint %d: id is required", i)
		}
		if _, exists := ids[endpoint.ID]; exists {
			return fmt.Errorf("endpoint %q: duplicate id", endpoint.ID)
		}
		ids[endpoint.ID] = struct{}{}
	}
	linkIDs := make(map[string]struct{}, len(g.Links))
	for i, link := range g.Links {
		if strings.TrimSpace(link.ID) == "" {
			return fmt.Errorf("link %d: id is required", i)
		}
		if _, exists := linkIDs[link.ID]; exists {
			return fmt.Errorf("link %q: duplicate id", link.ID)
		}
		linkIDs[link.ID] = struct{}{}
		if _, ok := ids[link.From]; !ok {
			return fmt.Errorf("link %q: unknown from endpoint %q", link.ID, link.From)
		}
		if _, ok := ids[link.To]; !ok {
			return fmt.Errorf("link %q: unknown to endpoint %q", link.ID, link.To)
		}
		if link.From == link.To {
			return fmt.Errorf("link %q: self-link is not a transfer capability", link.ID)
		}
		if strings.TrimSpace(link.Transport) == "" {
			return fmt.Errorf("link %q: transport is required", link.ID)
		}
	}
	return nil
}

var ErrNoRoute = errors.New("no route satisfies the directional mapping")

// Plan chooses the lowest-cost eligible route. Ties are resolved by latency,
// hop count, then link IDs, making manifests reproducible. Each hop is checked
// independently because a multi-hop route may compose unlike technologies.
func (g Graph) Plan(req Request) (Route, error) {
	if err := g.Validate(); err != nil {
		return Route{}, err
	}
	if req.From == "" || req.To == "" {
		return Route{}, errors.New("request from and to are required")
	}
	endpoint := make(map[string]bool, len(g.Endpoints))
	adjacency := make(map[string][]Link)
	for _, node := range g.Endpoints {
		endpoint[node.ID] = true
	}
	if !endpoint[req.From] {
		return Route{}, fmt.Errorf("unknown source endpoint %q", req.From)
	}
	if !endpoint[req.To] {
		return Route{}, fmt.Errorf("unknown destination endpoint %q", req.To)
	}
	if req.From == req.To {
		return Route{From: req.From, To: req.To, Links: []Link{}}, nil
	}
	for _, link := range g.Links {
		if eligible(link, req) {
			adjacency[link.From] = append(adjacency[link.From], link)
		}
	}
	for from := range adjacency {
		sort.Slice(adjacency[from], func(i, j int) bool { return adjacency[from][i].ID < adjacency[from][j].ID })
	}
	q := priorityQueue{&candidate{node: req.From, bottleneck: ^uint64(0)}}
	heap.Init(&q)
	best := map[string]score{req.From: {}}
	for q.Len() > 0 {
		current := heap.Pop(&q).(*candidate)
		if known, ok := best[current.node]; ok && known.less(current.score) {
			continue
		}
		if current.node == req.To {
			bottleneck := current.bottleneck
			if len(current.links) == 0 {
				bottleneck = 0
			}
			return Route{From: req.From, To: req.To, Links: current.links, TotalCost: current.cost, TotalLatencyNanos: current.latency, BottleneckBandwidthBytesPerSecond: bottleneck}, nil
		}
		for _, link := range adjacency[current.node] {
			next := &candidate{node: link.To, links: appendCopy(current.links, link), bottleneck: minBandwidth(current.bottleneck, link.BandwidthBytesPerSecond)}
			next.cost = current.cost + effectiveCost(link)
			next.latency = current.latency + link.LatencyNanos
			next.hops = current.hops + 1
			next.key = current.key + "\x00" + link.ID
			old, seen := best[next.node]
			if !seen || next.score.less(old) {
				best[next.node] = next.score
				heap.Push(&q, next)
			}
		}
	}
	return Route{}, fmt.Errorf("%w: %s -> %s", ErrNoRoute, req.From, req.To)
}

func eligible(link Link, req Request) bool {
	if req.MinBandwidthBytesPerSecond > 0 && link.BandwidthBytesPerSecond < req.MinBandwidthBytesPerSecond {
		return false
	}
	if req.MaxLinkLatencyNanos > 0 && link.LatencyNanos > req.MaxLinkLatencyNanos {
		return false
	}
	if !allowed(link.CPUPath, req.AllowedCPUPaths) || !allowed(link.Transport, req.AllowedTransports) {
		return false
	}
	for key, value := range req.RequiredLinkLabels {
		if link.Labels[key] != value {
			return false
		}
	}
	return true
}
func allowed(value string, set []string) bool {
	if len(set) == 0 {
		return true
	}
	for _, candidate := range set {
		if value == candidate {
			return true
		}
	}
	return false
}
func effectiveCost(link Link) uint64 {
	if link.Cost == 0 {
		return 1
	}
	return link.Cost
}
func minBandwidth(a, b uint64) uint64 {
	if a == ^uint64(0) {
		return b
	}
	if a == 0 || b == 0 {
		return 0
	}
	if a < b {
		return a
	}
	return b
}
func appendCopy(links []Link, link Link) []Link {
	out := make([]Link, len(links), len(links)+1)
	copy(out, links)
	return append(out, link)
}

type score struct {
	cost, latency uint64
	hops          int
	key           string
}

func (s score) less(other score) bool {
	if s.cost != other.cost {
		return s.cost < other.cost
	}
	if s.latency != other.latency {
		return s.latency < other.latency
	}
	if s.hops != other.hops {
		return s.hops < other.hops
	}
	return s.key < other.key
}

type candidate struct {
	node       string
	links      []Link
	bottleneck uint64
	score
}
type priorityQueue []*candidate

func (q priorityQueue) Len() int           { return len(q) }
func (q priorityQueue) Less(i, j int) bool { return q[i].score.less(q[j].score) }
func (q priorityQueue) Swap(i, j int)      { q[i], q[j] = q[j], q[i] }
func (q *priorityQueue) Push(x any)        { *q = append(*q, x.(*candidate)) }
func (q *priorityQueue) Pop() any          { old := *q; n := len(old); x := old[n-1]; *q = old[:n-1]; return x }
