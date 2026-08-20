package fabricmap

import (
	"container/heap"
	"errors"
	"fmt"
	"math/bits"
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

// RouteObjective selects the primary route ordering. The zero value preserves
// the original static-cost behavior.
type RouteObjective string

const (
	RouteObjectiveStaticCost         RouteObjective = "static_cost"
	RouteObjectiveEstimatedReadyTime RouteObjective = "estimated_ready_time"
)

// Request describes one directional mapping. RequiredLinkLabels supports
// capabilities not known to this package (encryption, coherency, GPUDirect,
// persistence, a future transport, and so on).
type Request struct {
	From                       string            `json:"from"`
	To                         string            `json:"to"`
	Bytes                      uint64            `json:"bytes,omitempty"`
	Objective                  RouteObjective    `json:"objective,omitempty"`
	MinBandwidthBytesPerSecond uint64            `json:"min_bandwidth_bytes_per_second,omitempty"`
	MaxLinkLatencyNanos        uint64            `json:"max_link_latency_nanos,omitempty"`
	AllowedCPUPaths            []string          `json:"allowed_cpu_paths,omitempty"`
	AllowedTransports          []string          `json:"allowed_transports,omitempty"`
	RequiredLinkLabels         map[string]string `json:"required_link_labels,omitempty"`
}

// Route is an ordered, directional transfer plan.
type Route struct {
	From                              string         `json:"from"`
	To                                string         `json:"to"`
	Links                             []Link         `json:"links"`
	Bytes                             uint64         `json:"bytes"`
	Objective                         RouteObjective `json:"objective"`
	TotalCost                         uint64         `json:"total_cost"`
	TotalLatencyNanos                 uint64         `json:"total_latency_nanos"`
	EstimatedReadyTimeNanos           uint64         `json:"estimated_ready_time_nanos,omitempty"`
	BottleneckBandwidthBytesPerSecond uint64         `json:"bottleneck_bandwidth_bytes_per_second,omitempty"`
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

var (
	ErrNoRoute                = errors.New("no route satisfies the directional mapping")
	ErrUnknownRouteObjective  = errors.New("fabricmap: unknown route objective")
	ErrReadyTimeBytesRequired = errors.New("fabricmap: estimated-ready-time objective requires nonzero bytes")
	ErrReadyTimeBandwidth     = errors.New("fabricmap: estimated-ready-time objective requires known nonzero bandwidth")
	ErrReadyTimeOverflow      = errors.New("fabricmap: estimated-ready-time estimate overflows uint64 nanoseconds")
)

// Plan chooses the eligible route under the request's explicit objective. The
// zero-value objective preserves cost, latency, hop, then link-ID ordering.
// Each hop is checked independently because routes may compose unlike technologies.
func (g Graph) Plan(req Request) (Route, error) {
	if err := g.Validate(); err != nil {
		return Route{}, err
	}
	objective, err := normalizeRouteObjective(req.Objective)
	if err != nil {
		return Route{}, err
	}
	if objective == RouteObjectiveEstimatedReadyTime && req.Bytes == 0 {
		return Route{}, ErrReadyTimeBytesRequired
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
		return Route{From: req.From, To: req.To, Links: []Link{}, Bytes: req.Bytes, Objective: objective}, nil
	}
	for _, link := range g.Links {
		if eligible(link, req) {
			adjacency[link.From] = append(adjacency[link.From], link)
		}
	}
	for from := range adjacency {
		sort.Slice(adjacency[from], func(i, j int) bool { return adjacency[from][i].ID < adjacency[from][j].ID })
	}
	q := priorityQueue{&candidate{node: req.From, bottleneck: ^uint64(0), score: score{objective: objective}}}
	heap.Init(&q)
	best := map[string]score{req.From: {objective: objective}}
	var estimateErr error
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
			return routeFromCandidate(req, objective, current.links, current.score, bottleneck), nil
		}
		for _, link := range adjacency[current.node] {
			next := &candidate{node: link.To, links: appendCopy(current.links, link), bottleneck: minBandwidth(current.bottleneck, link.BandwidthBytesPerSecond)}
			nextScore, err := extendScore(current.score, link, req.Bytes)
			if err != nil {
				if estimateErr == nil || errors.Is(err, ErrReadyTimeOverflow) {
					estimateErr = err
				}
				continue
			}
			next.score = nextScore
			next.hops = current.hops + 1
			next.key = current.key + "\x00" + link.ID
			old, seen := best[next.node]
			if !seen || next.score.less(old) {
				best[next.node] = next.score
				heap.Push(&q, next)
			}
		}
	}
	if estimateErr != nil {
		return Route{}, estimateErr
	}
	return Route{}, fmt.Errorf("%w: %s -> %s", ErrNoRoute, req.From, req.To)
}

func normalizeRouteObjective(objective RouteObjective) (RouteObjective, error) {
	switch objective {
	case "", RouteObjectiveStaticCost:
		return RouteObjectiveStaticCost, nil
	case RouteObjectiveEstimatedReadyTime:
		return objective, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrUnknownRouteObjective, objective)
	}
}

func routeFromCandidate(req Request, objective RouteObjective, links []Link, routeScore score, bottleneck uint64) Route {
	return Route{
		From:                              req.From,
		To:                                req.To,
		Links:                             links,
		Bytes:                             req.Bytes,
		Objective:                         objective,
		TotalCost:                         routeScore.cost,
		TotalLatencyNanos:                 routeScore.latency,
		EstimatedReadyTimeNanos:           routeScore.readyTime,
		BottleneckBandwidthBytesPerSecond: bottleneck,
	}
}

func extendScore(current score, link Link, byteCount uint64) (score, error) {
	next := score{
		objective: current.objective,
		cost:      current.cost + effectiveCost(link),
		latency:   current.latency + link.LatencyNanos,
	}
	if current.objective != RouteObjectiveEstimatedReadyTime {
		return next, nil
	}
	if link.BandwidthBytesPerSecond == 0 {
		return score{}, fmt.Errorf("%w: link %q", ErrReadyTimeBandwidth, link.ID)
	}
	if next.cost < current.cost || next.latency < current.latency {
		return score{}, fmt.Errorf("%w: link %q cumulative metric", ErrReadyTimeOverflow, link.ID)
	}
	serialization, err := serializationNanos(byteCount, link.BandwidthBytesPerSecond)
	if err != nil {
		return score{}, fmt.Errorf("%w: link %q", err, link.ID)
	}
	hopReady := link.LatencyNanos + serialization
	if hopReady < link.LatencyNanos || current.readyTime > ^uint64(0)-hopReady {
		return score{}, fmt.Errorf("%w: link %q cumulative ready time", ErrReadyTimeOverflow, link.ID)
	}
	next.readyTime = current.readyTime + hopReady
	return next, nil
}

// serializationNanos returns ceil(bytes*1e9/bandwidth) without narrowing the
// 128-bit intermediate product.
func serializationNanos(byteCount, bandwidth uint64) (uint64, error) {
	if bandwidth == 0 {
		return 0, ErrReadyTimeBandwidth
	}
	hi, lo := bits.Mul64(byteCount, 1_000_000_000)
	if hi >= bandwidth {
		return 0, ErrReadyTimeOverflow
	}
	quotient, remainder := bits.Div64(hi, lo, bandwidth)
	if remainder != 0 {
		if quotient == ^uint64(0) {
			return 0, ErrReadyTimeOverflow
		}
		quotient++
	}
	return quotient, nil
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
	objective                RouteObjective
	cost, latency, readyTime uint64
	hops                     int
	key                      string
}

func (s score) less(other score) bool {
	if s.objective == RouteObjectiveEstimatedReadyTime && s.readyTime != other.readyTime {
		return s.readyTime < other.readyTime
	}
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
