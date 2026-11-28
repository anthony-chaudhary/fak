package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// KV-aware fleet routing — the cross-instance analogue of fak's on-instance
// cache-aware scheduling (the FCFS 62.1% -> 86.7% radix-hit recovery folded into
// the kv-hit-rate parity row). The on-instance scheduler reorders the queue of
// ONE kernel so a request runs while its prefix KV is resident; this routes a
// request ACROSS a fleet of kernels to the instance that already holds its prefix
// KV, so the prefill is reused instead of recomputed cold on whichever worker a
// cache-blind balancer happened to pick.
//
// It is the fleet-level request-to-cached-block router the `kv-cache-aware-routing`
// industry-scorecard gap names: fak had cache-aware SCHEDULING within one instance
// and a coherence bus, but no cross-GPU residency router — so a request could land
// on a worker that must re-prefill a prefix another worker already holds. The
// router below carries the same residency signal across instances. A cold prefix
// lands on the least-loaded instance and is kept balanced, the overlap-minus-load
// placement the Dynamo Smart TierPolicy / SGLang cache-aware router approximate.
//
// What is measured here is the routing DECISION hit-rate — does the router send a
// request to an instance that already holds its prefix? — the same kind of number
// as the on-instance radix hit-rate. The downstream wall-clock TTFT/throughput win
// those hits produce on real GPUs (Baseten/Dynamo's +62% output tok/s, GORGO's
// 2.5x TTFT) is host-gated and not claimed here.

// residentSet is one instance's bounded set of resident prefix-KV blocks: an LRU
// keyed by prefix identity. capacity is the number of DISTINCT prefixes whose KV
// fits in that instance's budget; admitting past capacity evicts the LRU prefix.
type residentSet struct {
	capacity int
	order    []string // least-recently-used first, most-recently-used last
	held     map[string]struct{}
	segments map[string][]string
}

func newResidentSet(capacity int) *residentSet {
	if capacity < 1 {
		capacity = 1
	}
	return &residentSet{
		capacity: capacity,
		held:     make(map[string]struct{}, capacity),
		segments: make(map[string][]string, capacity),
	}
}

func (s *residentSet) holds(key string) bool {
	_, ok := s.held[key]
	return ok
}

// touch marks an already-resident key most-recently-used (a hit re-prefills nothing).
func (s *residentSet) touch(key string) {
	for i, k := range s.order {
		if k == key {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
	s.order = append(s.order, key)
}

// admit brings a cold prefix resident, evicting the LRU prefix if at capacity.
func (s *residentSet) admit(key string) {
	if s.holds(key) {
		s.touch(key)
		return
	}
	if len(s.order) >= s.capacity {
		evict := s.order[0]
		s.order = s.order[1:]
		delete(s.held, evict)
		delete(s.segments, evict)
	}
	s.order = append(s.order, key)
	s.held[key] = struct{}{}
}

func (s *residentSet) admitPrefix(key string, prefix []string) {
	s.admit(key)
	if len(prefix) > 0 {
		if s.segments == nil {
			s.segments = make(map[string][]string, s.capacity)
		}
		s.segments[key] = append([]string(nil), prefix...)
	}
}

func (s *residentSet) overlap(prefix []string) int {
	if len(prefix) == 0 || len(s.held) == 0 {
		return 0
	}
	best := 0
	for k := range s.held {
		seg := s.segments[k]
		if len(seg) == 0 {
			seg = []string{k}
		}
		if l := strmatch.CommonSlicePrefixLen(seg, prefix); l > best {
			best = l
		}
	}
	return best
}

// FleetRouter places a request (identified by the shareable prefix it would
// re-prefill cold) on one instance of a fixed fleet and reports whether that
// instance already held the prefix KV — a cross-instance hit, no re-prefill.
type FleetRouter interface {
	Route(prefixKey string) (instance int, hit bool)
}

// CacheBlindRouter is the baseline: round-robin placement that ignores residency
// (what the existing ReplicaDispatch.pick does). It still records whether the chosen
// instance happened to hold the prefix, so its hit-rate is directly comparable to
// the KV-aware router's on the same request stream.
type CacheBlindRouter struct {
	instances []*residentSet
	next      int
}

// newResidentInstances builds a clamped (n>=1) slice of n resident sets, each able to keep
// capPerInstance distinct prefixes resident — the shared fleet-construction step of the
// cache-blind and KV-aware routers.
func newResidentInstances(n, capPerInstance int) []*residentSet {
	if n < 1 {
		n = 1
	}
	insts := make([]*residentSet, n)
	for i := range insts {
		insts[i] = newResidentSet(capPerInstance)
	}
	return insts
}

// NewCacheBlindRouter builds a round-robin fleet of n instances, each able to keep
// capPerInstance distinct prefixes resident.
func NewCacheBlindRouter(n, capPerInstance int) *CacheBlindRouter {
	return &CacheBlindRouter{instances: newResidentInstances(n, capPerInstance)}
}

func (r *CacheBlindRouter) Route(prefixKey string) (int, bool) {
	i := r.next % len(r.instances)
	r.next++
	hit := r.instances[i].holds(prefixKey)
	r.instances[i].admit(prefixKey)
	return i, hit
}

// FleetCacheRouter is the KV-aware router: it routes to an instance that already
// holds the prefix KV (locality), and lands a cold prefix on the least-loaded
// instance so prefix homes stay balanced (the overlap-minus-load placement). It
// is the cross-instance composition of the on-instance residency signal — the
// fleet-level request-to-cached-block router the kv-cache-aware-routing gap names.
type FleetCacheRouter struct {
	instances []*residentSet
	load      []int

	// tolerance arms the bounded load-imbalance slack on the sticky path (the
	// microsoft/vidur slack-tolerant sticky scheduler borrow; see routeSlack).
	// Zero keeps the rigid sticky behavior the committed hit-rate witness pins;
	// a positive value sheds a sticky prefix whose home instance's load exceeds
	// tolerance x the fleet's minimum load.
	tolerance float64

	// home is the sticky record the slack path consults: prefix key -> the
	// instance the policy currently pins it to. Nil unless the slack is armed
	// (NewSlackTolerantFleetCacheRouter); the rigid path stays record-free.
	home map[string]int

	// idx keeps load as a bucket map so the slack path reads the fleet minimum
	// in O(1) and tie-breaks only the tied class (the ai-dynamo/aiperf
	// sticky-router borrow; see loadIndex). Nil until first slack-path use;
	// rigid routers never build it.
	idx *loadIndex
}

// NewFleetCacheRouter builds a KV-aware fleet of n instances, each able to keep
// capPerInstance distinct prefixes resident.
func NewFleetCacheRouter(n, capPerInstance int) *FleetCacheRouter {
	insts := newResidentInstances(n, capPerInstance)
	return &FleetCacheRouter{instances: insts, load: make([]int, len(insts))}
}

// NewSlackTolerantFleetCacheRouter builds a KV-aware fleet of n instances with the
// bounded load-imbalance slack armed: a prefix's home instance keeps its traffic
// while its load stays within tolerance x the fleet's minimum load, and sheds to
// the least-loaded instance once it exceeds the bound. Adapted from
// microsoft/vidur's TolerantStickyLOPUncachedGlobalScheduler (source:
// vidur/scheduler/global_scheduler/tolerant_sticky_lop_uncached_global_scheduler.py
// code). The slack only bites above one replica: with a single instance the home
// IS the minimum, so the bound can never be exceeded. tolerance <= 0 falls back
// to the rigid sticky router.
func NewSlackTolerantFleetCacheRouter(n, capPerInstance int, tolerance float64) *FleetCacheRouter {
	r := NewFleetCacheRouter(n, capPerInstance)
	if tolerance > 0 {
		r.tolerance = tolerance
		r.home = make(map[string]int)
	}
	return r
}

func (r *FleetCacheRouter) Route(prefixKey string) (int, bool) {
	if r.tolerance > 0 && len(r.instances) > 1 {
		return r.routeSlack(prefixKey)
	}
	for i, s := range r.instances {
		if s.holds(prefixKey) {
			s.touch(prefixKey)
			r.bumpLoad(i)
			return i, true
		}
	}
	pick := r.coldPlacement()
	r.instances[pick].admit(prefixKey)
	r.bumpLoad(pick)
	return pick, false
}

// routeSlack is the slack-tolerant sticky placement. The prefix's home instance
// (the sticky record, seeded by its last placement) keeps the turn while its load
// stays within the bounded slack — load[home] <= tolerance x min(load) — because
// each turn there reuses resident prefix KV instead of a cold prefill. Past the
// bound the turn sheds to the least-loaded OTHER instance (a cold placement that
// duplicates the prefix's residency there) and the sticky record migrates with
// it, so the session tracks the cool side of the fleet until loads rebalance. A
// home that evicted the prefix (LRU pressure) loses the record and re-homes cold.
// Deterministic: ties break to the lower instance index, unlike the upstream
// random tie-break, so the witness is reproducible.
func (r *FleetCacheRouter) routeSlack(prefixKey string) (int, bool) {
	if r.home == nil {
		r.home = make(map[string]int)
	}
	home, pinned := r.home[prefixKey]
	liveHome := pinned && r.instances[home].holds(prefixKey)
	if liveHome && float64(r.load[home]) <= r.tolerance*float64(r.minLoad()) {
		// Within the bounded slack: keep the sticky placement (prefix hit).
		r.instances[home].touch(prefixKey)
		r.bumpLoad(home)
		return home, true
	}
	// No live sticky record (or the home evicted the prefix, or the home's load
	// exceeded the slack): place on the least-loaded instance, skipping the
	// overloaded home so a shed actually moves the session, and migrate the
	// sticky record with it.
	exclude := -1
	if liveHome {
		exclude = home
	}
	pick := r.leastLoaded(exclude)
	hit := r.instances[pick].holds(prefixKey)
	r.instances[pick].admit(prefixKey)
	r.bumpLoad(pick)
	r.home[prefixKey] = pick
	return pick, hit
}

// loadIndex keeps the fleet load vector as a bucket map: buckets[l] holds the
// instance indices whose load is l, and keys lists the distinct loads in
// ascending order, so the least-loaded class is reached in O(1) (keys[0]) and
// a tie-break scans only the k-way tie instead of all n instances. Adapted
// from ai-dynamo/aiperf's sticky router, which keeps _workers_by_load (load ->
// worker set) plus a cached _min_load for exactly this reason (source:
// src/aiperf/credit/sticky_router.py:389-391,550-581
// copied code). Upstream's tie-break secondary tuple (active sessions, virtual
// sent credits, last-sent timestamp) has no per-instance counterpart in fak's
// fixed-fleet model, so the total order falls back to instance index — the
// deterministic tie-break the committed witness pins.
//
// The view is maintained incrementally: set moves one instance between buckets
// when its load changes by one. It carries a slice stamp (backing-array
// address, length, endpoints) so a wholesale reassignment of the load slice —
// the same-package test pattern of setting r.load between Route calls — is
// detected and the view rebuilt in O(n) before next use.
type loadIndex struct {
	buckets map[int][]int // load -> instance indices (order not significant)
	keys    []int         // distinct loads, ascending; keys[0] is the minimum
	ptr     *int          // &loads[0] at the last build
	n       int
	first   int
	last    int
}

func newLoadIndex(loads []int) *loadIndex {
	x := &loadIndex{buckets: make(map[int][]int, len(loads))}
	x.rebuild(loads)
	return x
}

// rebuild recomputes the view from loads in O(n): one pass into the buckets,
// then the distinct loads sorted ascending.
func (x *loadIndex) rebuild(loads []int) {
	b := make(map[int][]int, len(loads))
	for i, l := range loads {
		b[l] = append(b[l], i)
	}
	keys := make([]int, 0, len(b))
	for l := range b {
		keys = append(keys, l)
	}
	sort.Ints(keys)
	x.buckets = b
	x.keys = keys
	if len(loads) > 0 {
		x.ptr = &loads[0]
		x.first = loads[0]
		x.last = loads[len(loads)-1]
	} else {
		x.ptr = nil
	}
	x.n = len(loads)
}

// fresh reports whether the view still describes loads: same backing array,
// length, and endpoints. A wholesale reassignment fails at least one term and
// forces a rebuild; in-place single-step writes keep it current through set.
func (x *loadIndex) fresh(loads []int) bool {
	if x.n != len(loads) || x.n == 0 {
		return false
	}
	return x.ptr == &loads[0] && x.first == loads[0] && x.last == loads[x.n-1]
}

// set moves instance i from old load o to new load v, keeping buckets and keys
// current. O(k) in the source bucket size; the target bucket is appended
// unsorted because the tie-break scan does not rely on bucket order.
func (x *loadIndex) set(i, o, v int) {
	if o == v {
		return
	}
	src := x.buckets[o]
	for j := range src {
		if src[j] == i {
			x.buckets[o] = append(src[:j], src[j+1:]...)
			break
		}
	}
	if len(x.buckets[o]) == 0 {
		delete(x.buckets, o)
		x.dropKey(o)
	}
	if _, ok := x.buckets[v]; !ok {
		x.insertKey(v)
	}
	x.buckets[v] = append(x.buckets[v], i)
}

// dropKey removes load l from the sorted key list (no-op if absent).
func (x *loadIndex) dropKey(l int) {
	for j := range x.keys {
		if x.keys[j] == l {
			x.keys = append(x.keys[:j], x.keys[j+1:]...)
			return
		}
	}
}

// insertKey adds load l to the sorted key list in ascending position.
func (x *loadIndex) insertKey(l int) {
	j := len(x.keys)
	for k := range x.keys {
		if x.keys[k] > l {
			j = k
			break
		}
	}
	x.keys = append(x.keys, 0)
	copy(x.keys[j+1:], x.keys[j:])
	x.keys[j] = l
}

// ensureFresh (re)builds the load index when r.load was reassigned out-of-band
// (same-package tests set r.load wholesale between Route calls); production
// traffic only mutates in place through bumpLoad, which keeps it current.
func (r *FleetCacheRouter) ensureFresh() {
	if r.idx == nil {
		r.idx = newLoadIndex(r.load)
		return
	}
	if !r.idx.fresh(r.load) {
		r.idx.rebuild(r.load)
	}
}

// bumpLoad records one request landing on instance i. Every in-place write to
// r.load goes through here while the slack path is armed, so the load index
// stays current; wholesale reassignment of r.load is caught by the index's
// slice stamp instead. Rigid routers skip the index entirely — they never
// read it, so the pinned hit-rate harness pays nothing for the borrow.
func (r *FleetCacheRouter) bumpLoad(i int) {
	if r.tolerance <= 0 {
		r.load[i]++
		return
	}
	r.ensureFresh()
	old := r.load[i]
	r.load[i] = old + 1
	r.idx.set(i, old, old+1)
}

// minLoad is the fleet's minimum in-flight request count — the slack bound's
// denominator (the upstream min_load_imbalance read over the replica set).
// O(1): the load index caches the minimum (the ai-dynamo/aiperf sticky
// router's cached _min_load; see loadIndex).
func (r *FleetCacheRouter) minLoad() int {
	r.ensureFresh()
	return r.idx.keys[0]
}

// leastLoaded picks the instance with the fewest in-flight requests, skipping
// except (the sticky home being shed from, so a shed always moves the session);
// ties break to the lower index. except may be -1 for no exclusion. O(k) in
// the size of the minimum-load tie: the load index hands back the tied class
// directly (the ai-dynamo/aiperf sticky router's single-pass tie-break over
// its min-load worker set; see loadIndex). Only when except is the sole
// occupant of the minimum bucket does the scan advance to the next key.
func (r *FleetCacheRouter) leastLoaded(except int) int {
	r.ensureFresh()
	best := -1
	for _, l := range r.idx.keys {
		for _, i := range r.idx.buckets[l] {
			if i == except {
				continue
			}
			if best == -1 || i < best {
				best = i
			}
		}
		if best != -1 {
			return best
		}
	}
	return best
}

// coldPlacement homes a brand-new prefix on the instance with the most FREE cache
// (fewest resident distinct prefixes), so the fleet's aggregate KV is filled
// before anything is evicted — the overlap-minus-load rule with occupancy as the
// load term. Ties break to the lower request count, then the lower index, so the
// placement is deterministic.
func (r *FleetCacheRouter) coldPlacement() int {
	best := 0
	for i := 1; i < len(r.instances); i++ {
		bi, ci := len(r.instances[best].order), len(r.instances[i].order)
		switch {
		case ci < bi:
			best = i
		case ci == bi && r.load[i] < r.load[best]:
			best = i
		}
	}
	return best
}

// FleetRoutingStats is one routing policy's outcome over a request stream.
type FleetRoutingStats struct {
	Requests int     `json:"requests"`
	Hits     int     `json:"hits"`
	HitRate  float64 `json:"hit_rate"`
}

func runFleetPolicy(router FleetRouter, stream []string) FleetRoutingStats {
	hits := 0
	for _, key := range stream {
		if _, hit := router.Route(key); hit {
			hits++
		}
	}
	st := FleetRoutingStats{Requests: len(stream), Hits: hits}
	if len(stream) > 0 {
		st.HitRate = float64(hits) / float64(len(stream))
	}
	return st
}

// KVFleetWorkloadSpec describes the synthetic shared-prefix agent fleet the
// harness routes. It is deliberately in the realistic regime that motivates
// KV-aware routing: the aggregate distinct prefix working set (PrefixFamilies)
// EXCEEDS one fleet's resident capacity (Instances * CapPerInstance), so no
// router can hold everything and locality is what keeps the hot prefixes home.
// Traffic is Zipf-skewed (a few shared system-prompts / agent scaffolds dominate),
// as a real cross-agent fleet's is, and requests are interleaved one-per-family
// per round so each family's turns are spread across the stream (concurrent
// agents, not back-to-back bursts).
type KVFleetWorkloadSpec struct {
	Instances      int `json:"instances"`
	CapPerInstance int `json:"cap_per_instance_prefixes"`
	PrefixFamilies int `json:"prefix_families"`
	HotWeight      int `json:"zipf_hot_weight"`
}

// buildStream renders the spec into a deterministic request stream: family i
// (1-indexed) gets max(1, HotWeight/i) requests (a Zipf 1/i popularity), and the
// requests are dealt one-family-per-round so same-family turns are spread out.
func (w KVFleetWorkloadSpec) buildStream() []string {
	remaining := make([]int, w.PrefixFamilies)
	total := 0
	for i := range remaining {
		n := w.HotWeight / (i + 1)
		if n < 1 {
			n = 1
		}
		remaining[i] = n
		total += n
	}
	stream := make([]string, 0, total)
	for emitted := 0; emitted < total; {
		for i := range remaining {
			if remaining[i] > 0 {
				stream = append(stream, prefixKey(i))
				remaining[i]--
				emitted++
			}
		}
	}
	return stream
}

func prefixKey(family int) string {
	// A stable per-family prefix identity (the shared system-prompt / scaffold a
	// router would re-prefill cold). Content is irrelevant; identity is the key.
	return "prefix-family-" + strconv.Itoa(family)
}

// KVFleetCompetitor records the SOTA cross-replica cache-hit bar the fak number
// is read against — the same metric, so the comparison is apples-to-apples.
type KVFleetCompetitor struct {
	Name             string  `json:"name"`
	CrossReplicaHit  float64 `json:"cross_replica_cache_hit"`
	Replicas         int     `json:"replicas"`
	DownstreamEffect string  `json:"downstream_wallclock_effect"`
	Source           string  `json:"source"`
}

// KVFleetRoutingResult is the committed witness: both policies' hit-rates on the
// same stream, the lift, and the competitor bar. The cache_blind_round_robin arm
// is fak's own cache-blind baseline (what ReplicaDispatch does today), so the lift
// isolates KV-aware ROUTING the way the on-instance row isolated cache-aware
// SCHEDULING (FCFS 62.1% -> DFS 86.7%) on one held-constant kernel.
type KVFleetRoutingResult struct {
	Harness     string              `json:"harness"`
	Workload    KVFleetWorkloadSpec `json:"workload"`
	CacheBlind  FleetRoutingStats   `json:"cache_blind_round_robin"`
	KVAware     FleetRoutingStats   `json:"kv_aware_locality"`
	HitRateLift float64             `json:"hit_rate_lift_x"`
	Competitor  KVFleetCompetitor   `json:"competitor"`
}

// MeasureKVAwareFleetRouting runs the cache-blind and KV-aware routers over one
// shared-prefix fleet workload and returns the comparison. Pure and deterministic:
// the same spec always yields the same numbers, so the committed artifact stays
// reproducible.
func MeasureKVAwareFleetRouting(spec KVFleetWorkloadSpec) KVFleetRoutingResult {
	stream := spec.buildStream()
	blind := runFleetPolicy(NewCacheBlindRouter(spec.Instances, spec.CapPerInstance), stream)
	aware := runFleetPolicy(NewFleetCacheRouter(spec.Instances, spec.CapPerInstance), stream)
	lift := 0.0
	if blind.HitRate > 0 {
		lift = aware.HitRate / blind.HitRate
	}
	return KVFleetRoutingResult{
		Harness:     "TestKVAwareFleetRoutingHitRate",
		Workload:    spec,
		CacheBlind:  blind,
		KVAware:     aware,
		HitRateLift: lift,
		Competitor: KVFleetCompetitor{
			Name:             "Baseten on NVIDIA Dynamo (KV-aware router)",
			CrossReplicaHit:  0.89,
			Replicas:         4,
			DownstreamEffect: "-50% TTFT / +62% output tok/s / +61% RPS on Qwen3 480B ~50k-token inputs (host-gated; not measured here)",
			Source:           "https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/ (2025-01)",
		},
	}
}

// DefaultKVFleetWorkload is the harness's standing workload: 4 instances (matching
// Baseten's 4 replicas), 8 resident prefixes each (fleet capacity 32), against 28
// distinct prefix-families — the standard KV-aware-routing regime where the working
// set fits the FLEET (28 <= 32) but NOT one instance (28 > 8). So one cache-blind
// instance must thrash its 8 slots over all 28 families, while locality routing
// keeps each family resident on a stable home (7 per instance) and re-prefills
// nothing on a recurrence.
var DefaultKVFleetWorkload = KVFleetWorkloadSpec{
	Instances:      4,
	CapPerInstance: 8,
	PrefixFamilies: 28,
	HotWeight:      120,
}

// ---------------------------------------------------------------------------
// Live PickPolicy / AffinityPicker bridge for FleetCacheRouter -> ReplicaDispatch
// ---------------------------------------------------------------------------

// FleetCachePickStats reports live routing outcomes recorded by FleetCachePickPolicy.
type FleetCachePickStats struct {
	Requests    int     `json:"requests"`
	Hits        int     `json:"hits"`
	PartialHits int     `json:"partial_hits"`
	ColdMisses  int     `json:"cold_misses"`
	Sheds       int     `json:"sheds"`
	HitRate     float64 `json:"hit_rate"`
}

// FleetCachePickPolicy adapts FleetCacheRouter / NewSlackTolerantFleetCacheRouter
// onto the live ReplicaDispatch PickPolicy and AffinityPicker seams so multi-GPU
// local replicas (1-8 backends) share prefix-aware KV locality and bounded slack
// shedding instead of cache-blind placement. Safe for concurrent use.
type FleetCachePickPolicy struct {
	mu          sync.Mutex
	router      *FleetCacheRouter
	nameToSlot  map[string]int
	slotToName  []string
	affinityTTL time.Duration
	affinityMax int
	affinities  map[string]affinityTarget
	now         func() time.Time
	stats       FleetCachePickStats
}

var (
	_ PickPolicy     = (*FleetCachePickPolicy)(nil)
	_ AffinityPicker = (*FleetCachePickPolicy)(nil)
)

// NewFleetCachePickPolicy wraps router as a live PickPolicy and AffinityPicker.
// When router is nil a default 4-instance slack-tolerant router (tolerance=1.5) is built.
func NewFleetCachePickPolicy(router *FleetCacheRouter) *FleetCachePickPolicy {
	if router == nil {
		router = NewSlackTolerantFleetCacheRouter(4, 64, 1.5)
	}
	return &FleetCachePickPolicy{
		router:      router,
		nameToSlot:  make(map[string]int, len(router.instances)),
		affinityTTL: 5 * time.Minute,
		affinityMax: defaultAffinityMax,
		affinities:  make(map[string]affinityTarget),
	}
}

// NewSlackTolerantFleetPickPolicy builds a live slack-tolerant prefix-aware PickPolicy
// sized for instances local replicas with capPerInstance resident prefix capacity.
func NewSlackTolerantFleetPickPolicy(instances, capPerInstance int, tolerance float64) *FleetCachePickPolicy {
	return NewFleetCachePickPolicy(NewSlackTolerantFleetCacheRouter(instances, capPerInstance, tolerance))
}

// AsPickPolicy bridges this FleetCacheRouter onto ReplicaDispatch.WithPickPolicy.
func (r *FleetCacheRouter) AsPickPolicy() *FleetCachePickPolicy {
	return NewFleetCachePickPolicy(r)
}

// Router returns the underlying FleetCacheRouter.
func (p *FleetCachePickPolicy) Router() *FleetCacheRouter {
	if p == nil {
		return nil
	}
	return p.router
}

// WithClock overrides the clock used for session affinity expiration (testing).
func (p *FleetCachePickPolicy) WithClock(now func() time.Time) *FleetCachePickPolicy {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.now = now
	return p
}

// WithAffinityTTL sets the session affinity TTL window.
func (p *FleetCachePickPolicy) WithAffinityTTL(ttl time.Duration) *FleetCachePickPolicy {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.affinityTTL = ttl
	return p
}

// WithAffinityMax bounds the number of retained session affinity entries.
func (p *FleetCachePickPolicy) WithAffinityMax(max int) *FleetCachePickPolicy {
	p.mu.Lock()
	defer p.mu.Unlock()
	if max <= 0 {
		max = defaultAffinityMax
	}
	p.affinityMax = max
	return p
}

// Stats returns a point-in-time snapshot of live placement statistics.
func (p *FleetCachePickPolicy) Stats() FleetCachePickStats {
	if p == nil {
		return FleetCachePickStats{}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	st := p.stats
	if st.Requests > 0 {
		st.HitRate = float64(st.Hits) / float64(st.Requests)
	}
	return st
}

// Pick implements PickPolicy by delegating to PickWithAffinity with an empty key.
func (p *FleetCachePickPolicy) Pick(candidates []PlannerReplica, prefix []string, load func(name string) int) (PlannerReplica, bool) {
	return p.PickWithAffinity(candidates, prefix, "", load)
}

// PickWithAffinity implements AffinityPicker using prefix residency and bounded slack.
func (p *FleetCachePickPolicy) PickWithAffinity(candidates []PlannerReplica, prefix []string, affinityKey string, load func(name string) int) (PlannerReplica, bool) {
	if p == nil || p.router == nil || len(candidates) == 0 {
		return PlannerReplica{}, false
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	eligible, bySlot := p.bindCandidatesLocked(candidates)
	if len(eligible) == 0 {
		return PlannerReplica{}, false
	}
	if load != nil {
		for _, slot := range eligible {
			l := load(bySlot[slot].Name)
			if l < 0 {
				l = 0
			}
			p.router.load[slot] = l
		}
		if p.router.idx != nil {
			p.router.idx.rebuild(p.router.load)
		}
	}

	effMin := p.effectiveMinLoadLocked(eligible)
	now := p.clockLocked()
	ttl := p.affinityTTL
	if ttl <= 0 {
		ttl = 5 * time.Minute
	}
	trimmed := strings.TrimSpace(affinityKey)

	if trimmed != "" {
		if slot, ok := p.pickStickyAffinityLocked(trimmed, eligible, effMin, now); ok {
			p.recordChoiceLocked(slot, prefix, false, load == nil)
			p.setAffinityLocked(trimmed, bySlot[slot].Name, now.Add(ttl))
			return bySlot[slot], true
		}
	}

	chosenSlot, shed := p.pickPrefixLocked(eligible, prefix, effMin)
	p.recordChoiceLocked(chosenSlot, prefix, shed, load == nil)
	if trimmed != "" {
		p.setAffinityLocked(trimmed, bySlot[chosenSlot].Name, now.Add(ttl))
	}
	return bySlot[chosenSlot], true
}

func (p *FleetCachePickPolicy) clockLocked() time.Time {
	if p.now != nil {
		return p.now()
	}
	return time.Now()
}

func (p *FleetCachePickPolicy) bindCandidatesLocked(candidates []PlannerReplica) ([]int, map[int]PlannerReplica) {
	if p.nameToSlot == nil {
		p.nameToSlot = make(map[string]int, len(candidates))
	}
	eligible := make([]int, 0, len(candidates))
	bySlot := make(map[int]PlannerReplica, len(candidates))
	for _, c := range candidates {
		slot, ok := p.nameToSlot[c.Name]
		if !ok {
			if len(p.slotToName) < len(p.router.instances) {
				slot = len(p.slotToName)
			} else {
				capPer := 64
				if len(p.router.instances) > 0 {
					capPer = p.router.instances[0].capacity
				}
				slot = len(p.router.instances)
				p.router.instances = append(p.router.instances, newResidentSet(capPer))
				p.router.load = append(p.router.load, 0)
				if p.router.idx != nil {
					p.router.idx.rebuild(p.router.load)
				}
			}
			p.nameToSlot[c.Name] = slot
			p.slotToName = append(p.slotToName, c.Name)
		}
		if _, dup := bySlot[slot]; !dup {
			eligible = append(eligible, slot)
			bySlot[slot] = c
		}
	}
	return eligible, bySlot
}

func (p *FleetCachePickPolicy) effectiveMinLoadLocked(eligible []int) int {
	minL := p.router.load[eligible[0]]
	for _, s := range eligible[1:] {
		if p.router.load[s] < minL {
			minL = p.router.load[s]
		}
	}
	// Floor denominator at 1 so sequential/1-inflight requests do not shed off warm KV.
	return mathx.MaxInt(minL, 1)
}

func (p *FleetCachePickPolicy) pickStickyAffinityLocked(key string, eligible []int, effMin int, now time.Time) (int, bool) {
	entry, ok := p.affinities[key]
	if !ok {
		return -1, false
	}
	if now.After(entry.deadline) {
		delete(p.affinities, key)
		return -1, false
	}
	slot, bound := p.nameToSlot[entry.worker]
	if !bound || !containsSlot(eligible, slot) {
		return -1, false
	}
	if p.router.tolerance > 0 && len(eligible) > 1 && float64(p.router.load[slot]) > p.router.tolerance*float64(effMin) {
		return -1, false
	}
	return slot, true
}

func (p *FleetCachePickPolicy) pickPrefixLocked(eligible []int, prefix []string, effMin int) (chosen int, shed bool) {
	prefixKey := residencyKey(prefix)
	bestLoc, bestOv := p.bestLocalitySlotLocked(eligible, prefix, prefixKey)
	slackArmed := p.router.tolerance > 0 && len(eligible) > 1

	if !slackArmed {
		if bestLoc != -1 {
			return bestLoc, false
		}
		return p.coldEligibleSlotLocked(eligible), false
	}

	if bestLoc != -1 {
		if float64(p.router.load[bestLoc]) <= p.router.tolerance*float64(effMin) {
			return bestLoc, false
		}
		// Locality holder saturated past bounded slack: shed to least-loaded peer.
		_ = bestOv
		return p.leastLoadedEligibleLocked(eligible, bestLoc, prefix), true
	}
	return p.leastLoadedEligibleLocked(eligible, -1, prefix), false
}

func (p *FleetCachePickPolicy) bestLocalitySlotLocked(eligible []int, prefix []string, prefixKey string) (int, int) {
	if len(prefix) == 0 {
		return -1, 0
	}
	pinnedSlot, hasHome := -1, false
	if p.router.home != nil && prefixKey != "" {
		if h, ok := p.router.home[prefixKey]; ok && p.router.instances[h].holds(prefixKey) {
			pinnedSlot, hasHome = h, true
		}
	}
	bestSlot, bestOv := -1, 0
	for _, slot := range eligible {
		ov := p.router.instances[slot].overlap(prefix)
		if ov == 0 {
			continue
		}
		switch {
		case bestSlot == -1 || ov > bestOv:
			bestSlot, bestOv = slot, ov
		case ov == bestOv:
			loadSlot, loadBest := p.router.load[slot], p.router.load[bestSlot]
			if loadSlot < loadBest || (loadSlot == loadBest && hasHome && slot == pinnedSlot && bestSlot != pinnedSlot) {
				bestSlot = slot
			}
		}
	}
	return bestSlot, bestOv
}

func (p *FleetCachePickPolicy) coldEligibleSlotLocked(eligible []int) int {
	best := eligible[0]
	for _, slot := range eligible[1:] {
		occSlot, occBest := len(p.router.instances[slot].order), len(p.router.instances[best].order)
		if occSlot < occBest || (occSlot == occBest && p.router.load[slot] < p.router.load[best]) {
			best = slot
		}
	}
	return best
}

func (p *FleetCachePickPolicy) leastLoadedEligibleLocked(eligible []int, exclude int, prefix []string) int {
	best := -1
	bestOv := -1
	for _, slot := range eligible {
		if slot == exclude {
			continue
		}
		ov := p.router.instances[slot].overlap(prefix)
		if best == -1 {
			best, bestOv = slot, ov
			continue
		}
		ls, lb := p.router.load[slot], p.router.load[best]
		if ls < lb {
			best, bestOv = slot, ov
			continue
		}
		if ls == lb {
			if ov > bestOv {
				best, bestOv = slot, ov
			} else if ov == bestOv && len(p.router.instances[slot].order) < len(p.router.instances[best].order) {
				best, bestOv = slot, ov
			}
		}
	}
	if best == -1 {
		return eligible[0]
	}
	return best
}

func (p *FleetCachePickPolicy) recordChoiceLocked(slot int, prefix []string, shed, bumpLoad bool) {
	ov := p.router.instances[slot].overlap(prefix)
	p.stats.Requests++
	switch {
	case len(prefix) > 0 && ov == len(prefix):
		p.stats.Hits++
	case len(prefix) > 0 && ov > 0:
		p.stats.PartialHits++
	default:
		p.stats.ColdMisses++
	}
	if shed {
		p.stats.Sheds++
	}
	if len(prefix) > 0 {
		prefixKey := residencyKey(prefix)
		p.router.instances[slot].admitPrefix(prefixKey, prefix)
		if p.router.home != nil {
			p.router.home[prefixKey] = slot
		}
	}
	if bumpLoad {
		p.router.bumpLoad(slot)
	}
}

func (p *FleetCachePickPolicy) setAffinityLocked(key, worker string, deadline time.Time) {
	if p.affinities == nil {
		p.affinities = make(map[string]affinityTarget)
	}
	max := p.affinityMax
	if max <= 0 {
		max = defaultAffinityMax
	}
	evictExpiredOrBounded(p.affinities, p.clockLocked(), max)
	p.affinities[key] = affinityTarget{worker: worker, deadline: deadline}
}

func evictExpiredOrBounded(m map[string]affinityTarget, now time.Time, max int) {
	if len(m) < max {
		return
	}
	budget := max / 64
	if budget < 16 {
		budget = 16
	}
	if budget > 512 {
		budget = 512
	}
	fallback, visited := "", 0
	for k, v := range m {
		if visited >= budget {
			break
		}
		visited++
		if now.After(v.deadline) {
			delete(m, k)
			if len(m) < max {
				return
			}
			continue
		}
		if fallback == "" || k < fallback {
			fallback = k
		}
	}
	if fallback != "" {
		delete(m, fallback)
	}
}

func containsSlot(slots []int, target int) bool {
	for _, s := range slots {
		if s == target {
			return true
		}
	}
	return false
}

// FleetRouter returns the server's multi-replica dispatch router when configured.
func (s *Server) FleetRouter() *ReplicaDispatch {
	if s == nil {
		return nil
	}
	return s.fleetRouter
}

// UseFleetPickPolicy installs policy on the server's ReplicaDispatch and arms its
// membership after an initial probe if unarmed, enabling live prefix-aware routing
// across local replicas without waiting for background Serve loops.
func (s *Server) UseFleetPickPolicy(ctx context.Context, policy PickPolicy) bool {
	if s == nil || s.fleetRouter == nil {
		return false
	}
	s.fleetRouter.WithPickPolicy(policy)
	if s.fleetRouter.liveMembership() == nil && s.fleetRouter.FleetMembership() != nil {
		if ctx == nil {
			ctx = context.Background()
		}
		s.fleetRouter.FleetMembership().ProbeOnce(ctx)
		s.fleetRouter.WithMembership(s.fleetRouter.FleetMembership())
	}
	return true
}

// ---------------------------------------------------------------------------
// Gateway Per-Aspect Routing + Escalation Adapter (over internal/modelroute)
// ---------------------------------------------------------------------------

var (
	// ErrPerAspectNoRoster is returned when PerAspectRouter is constructed without a Roster.
	ErrPerAspectNoRoster = errors.New("gateway: per-aspect router requires a modelroute roster")
	// ErrPerAspectNoCandidates is returned when PerAspectRouter has no placement candidates.
	ErrPerAspectNoCandidates = errors.New("gateway: per-aspect router requires at least one placement candidate")
	// ErrPerAspectEscalationExhausted is returned fail-closed when EscalationBounds prevent further attempts.
	ErrPerAspectEscalationExhausted = errors.New("gateway: per-aspect escalation bounds exhausted")
)

const (
	// HeaderFakAspect selects the routing aspect on live HTTP requests (request|tool_call|query|step).
	HeaderFakAspect = "X-Fak-Aspect"
	// HeaderFakWorkItem identifies the multi-turn work item whose attempts/escalations are tracked.
	HeaderFakWorkItem = "X-Fak-Work-Item"
	// HeaderFakWorkClass explicitly declares modelroute.ClassLabel over HTTP.
	HeaderFakWorkClass = "X-Fak-Work-Class"
	// HeaderFakComplexity optionally sets Subject.Complexity (low|medium|high) over HTTP.
	HeaderFakComplexity = "X-Fak-Complexity"
	// HeaderFakTool optionally names the tool for tool_call/query subjects over HTTP.
	HeaderFakTool = "X-Fak-Tool"
	// HeaderFakSensitivity marks a tenant/sensitive payload that must remain on-box (ZoneDevice).
	HeaderFakSensitivity = "X-Fak-Sensitivity"

	// Response metadata headers emitted by PerAspectRouter.WrapHandler.
	HeaderFakRouteAspect     = "X-Fak-Route-Aspect"
	HeaderFakRouteModel      = "X-Fak-Route-Model"
	HeaderFakRouteZone       = "X-Fak-Route-Zone"
	HeaderFakRouteEscalated  = "X-Fak-Route-Escalated"
	HeaderFakEscalationAct   = "X-Fak-Escalation-Action"
	HeaderFakEscalationWhy   = "X-Fak-Escalation-Reason"
	HeaderFakPriorEscalation = "X-Fak-Prior-Escalations"
)

// PerAspectRouterConfig configures PerAspectRouter over a validated modelroute.Roster
// and bounded escalation policy for single-node 1-8 GPU local backends + frontier escalation.
type PerAspectRouterConfig struct {
	Roster                    *modelroute.Roster
	Manifest                  *modelroute.Manifest
	Candidates                []modelroute.Candidate
	Bounds                    modelroute.EscalationBounds
	DefaultClass              modelroute.WorkClass
	MinFailuresBeforeEscalate int
	MaxSameRungRetries        int
	Planners                  map[string]agent.Planner
	LocalFleetModels          []string
	AutoRetryInRequest        bool
	EvaluateHTTPAttempt       func(req AspectRouteRequest, dec AspectRouteDecision, status int, hdr http.Header, body []byte) modelroute.AttemptResult
}

// AspectRouteRequest describes one inbound routing subject (request, tool_call, query, or step)
// bound to an optional multi-turn work item identifier.
type AspectRouteRequest struct {
	WorkItemID string
	Subject    modelroute.Subject
	Serving    modelroute.ServingReport
	OnBoxOnly  bool
}

// AspectRouteDecision records the manifest match, classification, and zone placement for an attempt.
type AspectRouteDecision struct {
	Subject          modelroute.Subject
	ManifestDecision modelroute.Decision
	Classification   modelroute.Classification
	Placement        modelroute.Placement
	Escalated        bool
	PriorEscalations int
	Attempts         int
}

// WorkItemEscalationState is a read-only snapshot of one work item's attempt and escalation history.
type WorkItemEscalationState struct {
	WorkItemID           string
	Attempts             int
	UnderpoweredFailures int
	SameRungRetries      int
	PriorEscalations     int
	MinZone              modelroute.PlacementZone
	LastPlacement        modelroute.Placement
	LastVerdict          modelroute.EscalationVerdict
	Exhausted            bool
	ExhaustedReason      string
}

// AspectAttemptRecord captures one executed attempt in Execute / DispatchChat.
type AspectAttemptRecord struct {
	Attempt   int
	Decision  AspectRouteDecision
	Result    modelroute.AttemptResult
	Verdict   modelroute.EscalationVerdict
	Duration  time.Duration
	CallError error
}

// AspectExecutionResult is the complete multi-attempt audit trail for Execute.
type AspectExecutionResult struct {
	FinalDecision AspectRouteDecision
	FinalVerdict  modelroute.EscalationVerdict
	Attempts      []AspectAttemptRecord
}

// AspectChatResult pairs a completion with the per-aspect routing and escalation trail.
type AspectChatResult struct {
	Execution  AspectExecutionResult
	Completion *agent.Completion
}

// AspectAttemptFunc executes one routed attempt and returns its verified/failure outcome.
type AspectAttemptFunc func(ctx context.Context, decision AspectRouteDecision) (modelroute.AttemptResult, error)

type workItemRouteState struct {
	attempts             int
	underpoweredFailures int
	sameRungRetries      int
	priorEscalations     int
	minZone              modelroute.PlacementZone
	lastPlacement        modelroute.Placement
	lastVerdict          modelroute.EscalationVerdict
	exhausted            bool
	exhaustedReason      string
	onBoxOnly            bool
}

// PerAspectRouter bridges modelroute's per-aspect routing, Roster.PlaceSubject placement,
// and modelroute.AfterAttempt bounded escalation into the gateway. Single-turn tool/query/
// step/request work routes to local 3B-32B replicas by default, tracks prior attempts/failures
// per work item, and auto-escalates within EscalationBounds (failing closed when exhausted).
type PerAspectRouter struct {
	mu                        sync.Mutex
	roster                    *modelroute.Roster
	manifest                  *modelroute.Manifest
	candidates                []modelroute.Candidate
	bounds                    modelroute.EscalationBounds
	defaultClass              modelroute.WorkClass
	minFailuresBeforeEscalate int
	maxSameRungRetries        int
	planners                  map[string]agent.Planner
	httpPlanners              map[string]*agent.HTTPPlanner
	localFleetModels          map[string]bool
	autoRetryInRequest        bool
	evaluateHTTP              func(req AspectRouteRequest, dec AspectRouteDecision, status int, hdr http.Header, body []byte) modelroute.AttemptResult
	items                     map[string]*workItemRouteState
}

// NewPerAspectRouter validates cfg and returns a concurrency-safe PerAspectRouter.
func NewPerAspectRouter(cfg PerAspectRouterConfig) (*PerAspectRouter, error) {
	if cfg.Roster == nil {
		return nil, ErrPerAspectNoRoster
	}
	if err := cfg.Roster.Validate(); err != nil {
		return nil, fmt.Errorf("gateway: per-aspect router roster: %w", err)
	}
	if cfg.Manifest != nil {
		if err := cfg.Manifest.Validate(); err != nil {
			return nil, fmt.Errorf("gateway: per-aspect router manifest: %w", err)
		}
	}
	if len(cfg.Candidates) == 0 {
		return nil, ErrPerAspectNoCandidates
	}
	for _, cand := range cfg.Candidates {
		if strings.TrimSpace(cand.Model) == "" {
			return nil, errors.New("gateway: per-aspect router candidate has empty model id")
		}
		if _, err := cfg.Roster.Resolve(cand.Model); err != nil {
			return nil, fmt.Errorf("gateway: per-aspect router candidate %q: %w", cand.Model, err)
		}
	}
	defaultClass := cfg.DefaultClass
	if defaultClass == "" {
		defaultClass = modelroute.ClassRoutine
	}
	minFail := cfg.MinFailuresBeforeEscalate
	if minFail <= 0 {
		minFail = 1
	}
	maxSame := cfg.MaxSameRungRetries
	if maxSame <= 0 {
		maxSame = 1
	}
	planners := make(map[string]agent.Planner, len(cfg.Planners))
	for k, v := range cfg.Planners {
		if v != nil {
			planners[k] = v
		}
	}
	localFleet := make(map[string]bool, len(cfg.LocalFleetModels))
	for _, m := range cfg.LocalFleetModels {
		if trimmed := strings.TrimSpace(m); trimmed != "" {
			localFleet[trimmed] = true
		}
	}
	return &PerAspectRouter{
		roster:                    cfg.Roster,
		manifest:                  cfg.Manifest,
		candidates:                append([]modelroute.Candidate(nil), cfg.Candidates...),
		bounds:                    cfg.Bounds,
		defaultClass:              defaultClass,
		minFailuresBeforeEscalate: minFail,
		maxSameRungRetries:        maxSame,
		planners:                  planners,
		httpPlanners:              make(map[string]*agent.HTTPPlanner),
		localFleetModels:          localFleet,
		autoRetryInRequest:        cfg.AutoRetryInRequest,
		evaluateHTTP:              cfg.EvaluateHTTPAttempt,
		items:                     make(map[string]*workItemRouteState),
	}, nil
}

// State returns a snapshot of the tracked attempt and escalation state for workItemID.
func (r *PerAspectRouter) State(workItemID string) (WorkItemEscalationState, bool) {
	if r == nil || strings.TrimSpace(workItemID) == "" {
		return WorkItemEscalationState{}, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	st, ok := r.items[strings.TrimSpace(workItemID)]
	if !ok {
		return WorkItemEscalationState{}, false
	}
	return WorkItemEscalationState{
		WorkItemID:           strings.TrimSpace(workItemID),
		Attempts:             st.attempts,
		UnderpoweredFailures: st.underpoweredFailures,
		SameRungRetries:      st.sameRungRetries,
		PriorEscalations:     st.priorEscalations,
		MinZone:              st.minZone,
		LastPlacement:        st.lastPlacement,
		LastVerdict:          st.lastVerdict,
		Exhausted:            st.exhausted,
		ExhaustedReason:      st.exhaustedReason,
	}, true
}

// ResetWorkItem clears any tracked attempt or exhaustion state for workItemID.
func (r *PerAspectRouter) ResetWorkItem(workItemID string) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.items, strings.TrimSpace(workItemID))
}

// RouteSubject classifies req.Subject across AspectRequest/AspectToolCall/AspectQuery/AspectStep
// and places it through Roster.PlaceSubject (or PlaceSubjectWithServing), honoring prior
// escalations and fail-closed EscalationBounds for req.WorkItemID.
func (r *PerAspectRouter) RouteSubject(_ context.Context, req AspectRouteRequest) (AspectRouteDecision, error) {
	if r == nil || r.roster == nil {
		return AspectRouteDecision{}, ErrPerAspectNoRoster
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	itemKey := strings.TrimSpace(req.WorkItemID)
	var st *workItemRouteState
	if itemKey != "" {
		st = r.ensureItemLocked(itemKey)
		if st.exhausted {
			return AspectRouteDecision{}, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, st.exhaustedReason)
		}
	}

	onBoxOnly := req.OnBoxOnly || isSensitiveRoutingLabels(req.Subject.Labels) || (st != nil && st.onBoxOnly)
	if st != nil && onBoxOnly {
		st.onBoxOnly = true
	}
	bounds := r.effectiveBoundsLocked(onBoxOnly)

	subj, manifestDec := r.prepareSubjectLocked(req.Subject)
	minZone := modelroute.ZoneDevice
	maxZone := modelroute.PlacementZone("")
	if st != nil && st.minZone.Valid() {
		minZone = st.minZone
		if bounds.Ceiling.Valid() {
			maxZone = bounds.Ceiling
		}
	}
	if onBoxOnly {
		maxZone = modelroute.ZoneDevice
	}
	if maxZone.Valid() && minZone.Rank() > maxZone.Rank() {
		if st != nil {
			st.exhausted = true
			st.exhaustedReason = modelroute.ReasonAtCeiling
		}
		return AspectRouteDecision{}, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, modelroute.ReasonAtCeiling)
	}

	cands := r.filterCandidatesLocked(minZone, maxZone, manifestDec)
	if len(cands) == 0 {
		reason := modelroute.ReasonNoZoneCanServe
		if maxZone.Valid() {
			reason = modelroute.ReasonAtCeiling
		}
		if st != nil {
			st.exhausted = true
			st.exhaustedReason = reason
		}
		return AspectRouteDecision{}, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, reason)
	}

	placement, cls, err := r.placeWithServingLocked(subj, cands, req.Serving)
	if err != nil {
		return AspectRouteDecision{}, err
	}
	if onBoxOnly && placement.Target.Remote() {
		return AspectRouteDecision{}, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, modelroute.ReasonAtCeiling)
	}

	dec := AspectRouteDecision{
		Subject:          subj,
		ManifestDecision: manifestDec,
		Classification:   cls,
		Placement:        placement,
	}
	if st != nil {
		st.lastPlacement = placement
		if !st.minZone.Valid() {
			st.minZone = placement.Zone
		}
		dec.Escalated = st.priorEscalations > 0
		dec.PriorEscalations = st.priorEscalations
		dec.Attempts = st.attempts
	}
	return dec, nil
}

// AfterAttempt records one attempt outcome for workItemID and applies modelroute.AfterAttempt.
// When escalation is authorized within EscalationBounds, it returns the updated decision for
// the higher rung; when bounds or rules refuse further attempts, it marks the item exhausted
// and returns ErrPerAspectEscalationExhausted.
func (r *PerAspectRouter) AfterAttempt(_ context.Context, workItemID string, placement modelroute.Placement, res modelroute.AttemptResult) (modelroute.EscalationVerdict, *AspectRouteDecision, error) {
	if r == nil || r.roster == nil {
		return modelroute.EscalationVerdict{Action: modelroute.ActionStop, Reason: modelroute.ReasonUnplacedAttempt}, nil, ErrPerAspectNoRoster
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	itemKey := strings.TrimSpace(workItemID)
	var st *workItemRouteState
	priorEsc := 0
	sameRungRetries := 0
	onBoxOnly := false
	if itemKey != "" {
		st = r.ensureItemLocked(itemKey)
		st.attempts++
		priorEsc = st.priorEscalations
		sameRungRetries = st.sameRungRetries
		onBoxOnly = st.onBoxOnly
	}
	bounds := r.effectiveBoundsLocked(onBoxOnly)

	if res.Succeeded {
		if st != nil {
			st.underpoweredFailures = 0
			st.sameRungRetries = 0
		}
		verdict := modelroute.AfterAttempt(placement, res, bounds, priorEsc)
		if st != nil {
			st.lastVerdict = verdict
		}
		return verdict, nil, nil
	}

	if res.Fail == modelroute.FailUnderpowered {
		underFailures := 1
		if st != nil {
			st.underpoweredFailures++
			underFailures = st.underpoweredFailures
		}
		if underFailures < r.minFailuresBeforeEscalate {
			if sameRungRetries >= r.maxSameRungRetries {
				verdict := modelroute.EscalationVerdict{Action: modelroute.ActionStop, From: placement.Zone, Reason: modelroute.ReasonBudgetSpent}
				r.markExhaustedLocked(st, verdict)
				return verdict, nil, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, verdict.Reason)
			}
			if st != nil {
				st.sameRungRetries++
			}
			verdict := modelroute.EscalationVerdict{
				Action: modelroute.ActionRetrySameRung,
				From:   placement.Zone,
				Reason: "awaiting-multi-turn-failure-threshold",
			}
			if st != nil {
				st.lastVerdict = verdict
			}
			nextDec := AspectRouteDecision{
				Placement:        placement,
				Escalated:        priorEsc > 0,
				PriorEscalations: priorEsc,
			}
			if st != nil {
				nextDec.Attempts = st.attempts
			}
			return verdict, &nextDec, nil
		}
	}

	verdict := modelroute.AfterAttempt(placement, res, bounds, priorEsc)
	switch verdict.Action {
	case modelroute.ActionRetrySameRung:
		if sameRungRetries >= r.maxSameRungRetries {
			stop := modelroute.EscalationVerdict{Action: modelroute.ActionStop, From: placement.Zone, Reason: verdict.Reason}
			r.markExhaustedLocked(st, stop)
			return stop, nil, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, stop.Reason)
		}
		if st != nil {
			st.sameRungRetries++
			st.lastVerdict = verdict
		}
		nextDec := AspectRouteDecision{
			Placement:        placement,
			Escalated:        priorEsc > 0,
			PriorEscalations: priorEsc,
		}
		if st != nil {
			nextDec.Attempts = st.attempts
		}
		return verdict, &nextDec, nil

	case modelroute.ActionEscalate:
		maxZone := bounds.Ceiling
		if onBoxOnly {
			maxZone = modelroute.ZoneDevice
		}
		if !maxZone.Valid() || verdict.To.Rank() > maxZone.Rank() {
			stop := modelroute.EscalationVerdict{Action: modelroute.ActionStop, From: placement.Zone, Reason: modelroute.ReasonAtCeiling}
			r.markExhaustedLocked(st, stop)
			return stop, nil, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, stop.Reason)
		}
		cands := r.filterCandidatesLocked(verdict.To, maxZone, modelroute.Decision{})
		if len(cands) == 0 {
			stop := modelroute.EscalationVerdict{Action: modelroute.ActionStop, From: placement.Zone, Reason: modelroute.ReasonAtCeiling}
			r.markExhaustedLocked(st, stop)
			return stop, nil, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, stop.Reason)
		}
		// Escalate through PlaceSubject using routine floor once the escalation gate
		// itself authorized moving up from the exhausted rung.
		escSubj := modelroute.Subject{
			Aspect: modelroute.AspectStep,
			Labels: map[string]string{modelroute.ClassLabel: string(r.defaultClass)},
		}
		if onBoxOnly {
			escSubj.Labels["sensitivity"] = "tenant"
		}
		nextPlacement, cls, err := r.roster.PlaceSubject(escSubj, cands)
		if err != nil || nextPlacement.Zone.Rank() > maxZone.Rank() || (onBoxOnly && nextPlacement.Target.Remote()) {
			stop := modelroute.EscalationVerdict{Action: modelroute.ActionStop, From: placement.Zone, Reason: modelroute.ReasonAtCeiling}
			r.markExhaustedLocked(st, stop)
			return stop, nil, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, stop.Reason)
		}
		if st != nil {
			st.priorEscalations++
			st.underpoweredFailures = 0
			st.sameRungRetries = 0
			st.minZone = nextPlacement.Zone
			st.lastPlacement = nextPlacement
			st.lastVerdict = verdict
		}
		nextDec := &AspectRouteDecision{
			Subject:          escSubj,
			Classification:   cls,
			Placement:        nextPlacement,
			Escalated:        true,
			PriorEscalations: priorEsc + 1,
		}
		if st != nil {
			nextDec.Attempts = st.attempts
		}
		return verdict, nextDec, nil

	default:
		r.markExhaustedLocked(st, verdict)
		return verdict, nil, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, verdict.Reason)
	}
}

// Execute runs fn under PerAspectRouter's placement and bounded AfterAttempt escalation loop.
func (r *PerAspectRouter) Execute(ctx context.Context, req AspectRouteRequest, fn AspectAttemptFunc) (AspectExecutionResult, error) {
	if fn == nil {
		return AspectExecutionResult{}, errors.New("gateway: per-aspect execute requires attempt callback")
	}
	dec, err := r.RouteSubject(ctx, req)
	if err != nil {
		return AspectExecutionResult{}, err
	}
	var out AspectExecutionResult
	maxLoop := 1 + r.maxSameRungRetries + mathx.MaxInt(r.bounds.MaxAttempts, 0) + 2
	for i := 0; i < maxLoop; i++ {
		began := time.Now()
		res, callErr := fn(ctx, dec)
		verdict, nextDec, escErr := r.AfterAttempt(ctx, req.WorkItemID, dec.Placement, res)
		out.Attempts = append(out.Attempts, AspectAttemptRecord{
			Attempt:   i + 1,
			Decision:  dec,
			Result:    res,
			Verdict:   verdict,
			Duration:  time.Since(began),
			CallError: callErr,
		})
		out.FinalDecision = dec
		out.FinalVerdict = verdict
		if verdict.Action == modelroute.ActionAccept || verdict.Action == modelroute.ActionVerifyFirst {
			return out, nil
		}
		if escErr != nil || nextDec == nil {
			if escErr != nil {
				return out, escErr
			}
			return out, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, verdict.Reason)
		}
		nextDec.Subject = dec.Subject
		dec = *nextDec
	}
	return out, fmt.Errorf("%w: %s", ErrPerAspectEscalationExhausted, modelroute.ReasonBudgetSpent)
}

// DispatchChat routes a chat turn across local 3B-32B replicas (or configured *ReplicaDispatch
// with FleetCachePickPolicy) and auto-escalates via modelroute.AfterAttempt on failure within bounds.
func (r *PerAspectRouter) DispatchChat(ctx context.Context, req AspectRouteRequest, messages []agent.Message, tools []agent.ToolDef, eval func(*agent.Completion, error) modelroute.AttemptResult, opts ...agent.SampleOpt) (AspectChatResult, error) {
	var finalComp *agent.Completion
	execRes, err := r.Execute(ctx, req, func(callCtx context.Context, dec AspectRouteDecision) (modelroute.AttemptResult, error) {
		planner, wireModel, perr := r.plannerForPlacement(dec.Placement)
		if perr != nil {
			return modelroute.AttemptResult{Fail: modelroute.FailWorkItem}, perr
		}
		callOpts := append([]agent.SampleOpt(nil), opts...)
		if wireModel != "" {
			callOpts = append(callOpts, agent.WithModel(wireModel))
		}
		comp, cerr := planner.Complete(callCtx, messages, tools, callOpts...)
		if cerr == nil && comp != nil {
			finalComp = comp
		}
		if eval != nil {
			return eval(comp, cerr), cerr
		}
		return defaultCompletionAttemptResult(comp, cerr), cerr
	})
	return AspectChatResult{Execution: execRes, Completion: finalComp}, err
}

// WrapHandler returns an http.Handler that routes /v1/chat/completions, /v1/messages,
// and /v1/fak/syscall through PerAspectRouter before dispatching into srv.
func (r *PerAspectRouter) WrapHandler(srv *Server) http.Handler {
	inner := srv.Handler()
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if r == nil || srv == nil {
			inner.ServeHTTP(w, req)
			return
		}
		switch {
		case req.Method == http.MethodPost && (req.URL.Path == "/v1/chat/completions" || req.URL.Path == "/v1/messages"):
			r.serveAspectChatHTTP(w, req, srv, inner)
		case req.Method == http.MethodPost && req.URL.Path == "/v1/fak/syscall":
			srv.withAuth(http.HandlerFunc(func(aw http.ResponseWriter, ar *http.Request) {
				r.serveAspectSyscallHTTP(aw, ar, srv)
			})).ServeHTTP(w, req)
		default:
			inner.ServeHTTP(w, req)
		}
	})
}

func (r *PerAspectRouter) serveAspectChatHTTP(w http.ResponseWriter, req *http.Request, srv *Server, inner http.Handler) {
	rawBody, err := io.ReadAll(http.MaxBytesReader(w, req.Body, maxTranscriptBody))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	var chatReq ChatRequest
	if err := json.Unmarshal(rawBody, &chatReq); err != nil {
		// Let inner handler render canonical decode errors.
		fwd := req.Clone(req.Context())
		fwd.Body = io.NopCloser(bytes.NewReader(rawBody))
		inner.ServeHTTP(w, fwd)
		return
	}
	aspect := aspectFromHTTPRequest(req, chatReq)
	workItem := workItemFromHTTPRequest(req, "")
	routeReq := r.buildHTTPSubjectRequest(req, aspect, workItem, chatReq.Messages, chatReq.Tools)

	dec, err := r.RouteSubject(req.Context(), routeReq)
	if err != nil {
		r.writeExhaustedHTTP(w, workItem, aspect, err)
		return
	}

	maxAttempts := 1
	if r.autoRetryInRequest {
		maxAttempts = 1 + r.maxSameRungRetries + mathx.MaxInt(r.bounds.MaxAttempts, 0) + 1
	}
	for i := 0; i < maxAttempts; i++ {
		if srv.metrics != nil && r.manifest != nil {
			srv.metrics.observeRouteDecision(r.manifest.Version, dec.ManifestDecision, 0)
		}
		targetModel := dec.Placement.Model
		if dec.Placement.Zone == modelroute.ZoneDevice && r.isLocalFleetModel(dec.Placement.Model) && srv.FleetRouter() != nil {
			targetModel = srv.model
		}
		fwdBody := rewriteJSONModel(rawBody, targetModel)
		fwd := req.Clone(req.Context())
		fwd.Body = io.NopCloser(bytes.NewReader(fwdBody))
		fwd.ContentLength = int64(len(fwdBody))

		rec := httptest.NewRecorder()
		inner.ServeHTTP(rec, fwd)

		res := r.evalHTTP(routeReq, dec, rec.Code, rec.Header(), rec.Body.Bytes())
		verdict, nextDec, escErr := r.AfterAttempt(req.Context(), workItem, dec.Placement, res)
		if r.autoRetryInRequest && escErr == nil && nextDec != nil &&
			(verdict.Action == modelroute.ActionRetrySameRung || verdict.Action == modelroute.ActionEscalate) {
			nextDec.Subject = dec.Subject
			nextDec.ManifestDecision = dec.ManifestDecision
			dec = *nextDec
			continue
		}
		if r.autoRetryInRequest && escErr != nil && !res.Succeeded && verdict.Action == modelroute.ActionStop {
			r.writeStopHeaders(w, aspect, dec, verdict)
			writeErrCode(w, http.StatusServiceUnavailable, "escalation_bounds_exhausted", escErr.Error())
			return
		}
		copyResponseHeaders(w.Header(), rec.Header())
		r.writeStopHeaders(w, aspect, dec, verdict)
		w.WriteHeader(rec.Code)
		_, _ = w.Write(rec.Body.Bytes())
		return
	}
	r.writeExhaustedHTTP(w, workItem, aspect, ErrPerAspectEscalationExhausted)
}

func (r *PerAspectRouter) serveAspectSyscallHTTP(w http.ResponseWriter, req *http.Request, srv *Server) {
	scReq, ok := srv.decodeSyscall(w, req)
	if !ok {
		return
	}
	aspect := aspectFromHeaderOrDefault(req, modelroute.AspectToolCall)
	workItem := workItemFromHTTPRequest(req, scReq.TraceID)
	labels := map[string]string{
		"read_only": boolLabel(scReq.ReadOnly),
	}
	if p := principalFor(req, scReq.Principal); p != "" {
		labels["tenant"] = p
	}
	if sens := strings.TrimSpace(req.Header.Get(HeaderFakSensitivity)); sens != "" {
		labels["sensitivity"] = sens
	}
	if wc := strings.TrimSpace(req.Header.Get(HeaderFakWorkClass)); wc != "" {
		labels[modelroute.ClassLabel] = wc
	}
	routeReq := AspectRouteRequest{
		WorkItemID: workItem,
		OnBoxOnly:  isSensitiveRoutingLabels(labels),
		Subject: modelroute.Subject{
			Aspect:     aspect,
			Tool:       scReq.Tool,
			Complexity: parseComplexityHeader(req.Header.Get(HeaderFakComplexity)),
			Labels:     labels,
		},
	}

	dec, err := r.RouteSubject(req.Context(), routeReq)
	if err != nil {
		r.writeExhaustedHTTP(w, workItem, aspect, err)
		return
	}

	ctx := WithPrincipal(req.Context(), principalFor(req, scReq.Principal))
	maxAttempts := 1
	if r.autoRetryInRequest {
		maxAttempts = 1 + r.maxSameRungRetries + mathx.MaxInt(r.bounds.MaxAttempts, 0) + 1
	}
	for i := 0; i < maxAttempts; i++ {
		if srv.metrics != nil && r.manifest != nil {
			srv.metrics.observeRouteDecision(r.manifest.Version, dec.ManifestDecision, 0)
		}
		tc, berr := srv.buildCall(ctx, scReq.Tool, rawArgs(scReq.Arguments), scReq.ReadOnly, scReq.Witness, scReq.TraceID)
		if berr != nil {
			writeErr(w, http.StatusBadRequest, berr.Error())
			return
		}
		if dec.Placement.Target.Account != "" && nativeToolName(tc.Tool) == "" && !isExecCommandRequest(tc) {
			tc.Engine = dec.Placement.Target.EngineRoute()
		}
		if sens := labels["sensitivity"]; sens != "" {
			tc.Meta["sensitivity"] = sens
		}
		resEnv, wv := r.executeKernelSyscall(ctx, srv, tc)
		body, _ := json.Marshal(SyscallResponse{Verdict: wv, Result: resEnv, TraceID: tc.TraceID})
		status := http.StatusOK
		if wv.Kind == "DENY" {
			status = http.StatusForbidden
		}
		attemptRes := r.evalHTTP(routeReq, dec, status, nil, body)
		if !attemptRes.Succeeded && wv.Kind == "DENY" {
			attemptRes = modelroute.AttemptResult{Fail: modelroute.FailRefused}
		}
		verdict, nextDec, escErr := r.AfterAttempt(ctx, workItem, dec.Placement, attemptRes)
		if r.autoRetryInRequest && escErr == nil && nextDec != nil &&
			(verdict.Action == modelroute.ActionRetrySameRung || verdict.Action == modelroute.ActionEscalate) {
			nextDec.Subject = dec.Subject
			nextDec.ManifestDecision = dec.ManifestDecision
			dec = *nextDec
			continue
		}
		if r.autoRetryInRequest && escErr != nil && !attemptRes.Succeeded && verdict.Action == modelroute.ActionStop && wv.Kind != "DENY" {
			r.writeStopHeaders(w, aspect, dec, verdict)
			writeErrCode(w, http.StatusServiceUnavailable, "escalation_bounds_exhausted", escErr.Error())
			return
		}
		r.writeStopHeaders(w, aspect, dec, verdict)
		writeJSON(w, http.StatusOK, SyscallResponse{Verdict: wv, Result: resEnv, TraceID: tc.TraceID})
		return
	}
	r.writeExhaustedHTTP(w, workItem, aspect, ErrPerAspectEscalationExhausted)
}

func (r *PerAspectRouter) executeKernelSyscall(ctx context.Context, srv *Server, tc *abi.ToolCall) (*ResultEnvelope, WireVerdict) {
	abRes, v := srv.k.Syscall(ctx, tc)
	wv := renderVerdict(v, resultMeta(abRes))
	var env *ResultEnvelope
	if abRes != nil {
		env = &ResultEnvelope{
			Status:  statusName(abRes.Status),
			Content: string(resolveBytes(ctx, abRes.Payload)),
			Meta:    abRes.Meta,
		}
		if env.Meta == nil {
			env.Meta = map[string]string{}
		}
	}
	return env, wv
}

func (r *PerAspectRouter) ensureItemLocked(itemKey string) *workItemRouteState {
	if r.items == nil {
		r.items = make(map[string]*workItemRouteState)
	}
	st, ok := r.items[itemKey]
	if !ok {
		st = &workItemRouteState{}
		r.items[itemKey] = st
	}
	return st
}

func (r *PerAspectRouter) markExhaustedLocked(st *workItemRouteState, verdict modelroute.EscalationVerdict) {
	if st == nil {
		return
	}
	st.exhausted = true
	st.exhaustedReason = verdict.Reason
	st.lastVerdict = verdict
}

func (r *PerAspectRouter) effectiveBoundsLocked(onBoxOnly bool) modelroute.EscalationBounds {
	b := r.bounds
	if onBoxOnly {
		b.Ceiling = modelroute.ZoneDevice
	}
	return b
}

func (r *PerAspectRouter) prepareSubjectLocked(in modelroute.Subject) (modelroute.Subject, modelroute.Decision) {
	subj := in
	labels := make(map[string]string, len(in.Labels)+1)
	for k, v := range in.Labels {
		labels[k] = v
	}
	subj.Labels = labels
	if subj.Aspect == "" {
		if subj.Tool != "" {
			subj.Aspect = modelroute.AspectToolCall
		} else {
			subj.Aspect = modelroute.AspectRequest
		}
	}
	var manifestDec modelroute.Decision
	if r.manifest != nil {
		manifestDec = r.manifest.Route(subj)
	}
	if _, hasExplicit := subj.Labels[modelroute.ClassLabel]; !hasExplicit {
		if manifestDec.Matched {
			if cls, ok := r.workClassForManifestPlanLocked(manifestDec.Plan.Primary()); ok {
				subj.Labels[modelroute.ClassLabel] = string(cls)
			}
		}
	}
	if _, hasExplicit := subj.Labels[modelroute.ClassLabel]; !hasExplicit && subj.Aspect != modelroute.AspectScout {
		switch subj.Complexity {
		case modelroute.ComplexityHigh:
			subj.Labels[modelroute.ClassLabel] = string(modelroute.ClassNormalImpl)
		case modelroute.ComplexityMedium:
			subj.Labels[modelroute.ClassLabel] = string(modelroute.ClassNormalImpl)
		default:
			subj.Labels[modelroute.ClassLabel] = string(r.defaultClass)
		}
	}
	return subj, manifestDec
}

func (r *PerAspectRouter) workClassForManifestPlanLocked(modelID string) (modelroute.WorkClass, bool) {
	for _, c := range r.candidates {
		if c.Model != modelID {
			continue
		}
		switch c.Capability {
		case modelroute.TierT0:
			return modelroute.ClassUltraHard, true
		case modelroute.TierT1:
			return modelroute.ClassNormalImpl, true
		case modelroute.TierT2:
			return modelroute.ClassRoutine, true
		}
	}
	return "", false
}

func (r *PerAspectRouter) filterCandidatesLocked(minZone, maxZone modelroute.PlacementZone, dec modelroute.Decision) []modelroute.Candidate {
	preferred := ""
	if dec.Matched {
		preferred = dec.Plan.Primary()
	}
	var front, rest []modelroute.Candidate
	for _, c := range r.candidates {
		t, err := r.roster.Resolve(c.Model)
		if err != nil {
			continue
		}
		z := t.Zone()
		if minZone.Valid() && z.Rank() < minZone.Rank() {
			continue
		}
		if maxZone.Valid() && z.Rank() > maxZone.Rank() {
			continue
		}
		if preferred != "" && c.Model == preferred {
			front = append(front, c)
		} else {
			rest = append(rest, c)
		}
	}
	return append(front, rest...)
}

func (r *PerAspectRouter) placeWithServingLocked(subj modelroute.Subject, cands []modelroute.Candidate, serving modelroute.ServingReport) (modelroute.Placement, modelroute.Classification, error) {
	if len(serving.Covers) == 0 && len(serving.Models) == 0 {
		return r.roster.PlaceSubject(subj, cands)
	}
	return r.roster.PlaceSubjectWithServing(subj, cands, serving)
}

func (r *PerAspectRouter) plannerForPlacement(p modelroute.Placement) (agent.Planner, string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if pl, ok := r.planners[p.Model]; ok && pl != nil {
		return pl, p.Target.UpstreamModel, nil
	}
	cacheKey := p.Target.Account + "\x00" + p.Target.BaseURL + "\x00" + p.Target.UpstreamModel
	if hp, ok := r.httpPlanners[cacheKey]; ok {
		return hp, p.Target.UpstreamModel, nil
	}
	provider := string(p.Target.Kind)
	switch p.Target.Kind {
	case modelroute.KindLocal, modelroute.KindFleet, modelroute.KindDeepSeek, modelroute.KindOpenRouter:
		provider = string(agent.ProviderOpenAI)
	}
	cred := ""
	if p.Target.CredEnv != "" {
		cred = os.Getenv(p.Target.CredEnv)
		if strings.TrimSpace(cred) == "" {
			return nil, "", fmt.Errorf("gateway: per-aspect route account %q credential env %q is empty", p.Target.Account, p.Target.CredEnv)
		}
	}
	hp, err := agent.NewProviderHTTPPlanner(provider, p.Target.BaseURL, p.Target.UpstreamModel, cred)
	if err != nil {
		return nil, "", fmt.Errorf("gateway: per-aspect route planner %q: %w", p.Target.Account, err)
	}
	if authScheme, ok := agent.ParseAnthropicAuthScheme(p.Target.AuthScheme); ok {
		hp.AnthropicAuthScheme = authScheme
	}
	r.httpPlanners[cacheKey] = hp
	return hp, p.Target.UpstreamModel, nil
}

func (r *PerAspectRouter) isLocalFleetModel(model string) bool {
	return r.localFleetModels[model]
}

func (r *PerAspectRouter) evalHTTP(req AspectRouteRequest, dec AspectRouteDecision, status int, hdr http.Header, body []byte) modelroute.AttemptResult {
	if r.evaluateHTTP != nil {
		return r.evaluateHTTP(req, dec, status, hdr, body)
	}
	return defaultHTTPAttemptResult(status, body)
}

func defaultHTTPAttemptResult(status int, body []byte) modelroute.AttemptResult {
	switch {
	case status == http.StatusForbidden:
		return modelroute.AttemptResult{Fail: modelroute.FailRefused}
	case status == http.StatusBadRequest:
		return modelroute.AttemptResult{Fail: modelroute.FailWorkItem}
	case status == http.StatusUnprocessableEntity:
		return modelroute.AttemptResult{Fail: modelroute.FailUnderpowered}
	case status >= 500 || status == http.StatusTooManyRequests || status == http.StatusRequestTimeout:
		return modelroute.AttemptResult{Fail: modelroute.FailTransport}
	case status >= 200 && status < 300:
		if strings.Contains(string(body), `"status":"ERROR"`) {
			return modelroute.AttemptResult{Fail: modelroute.FailUnderpowered}
		}
		return modelroute.AttemptResult{Succeeded: true, Verify: modelroute.VerifyWitness}
	default:
		return modelroute.AttemptResult{Fail: modelroute.FailUnclassified}
	}
}

func defaultCompletionAttemptResult(comp *agent.Completion, err error) modelroute.AttemptResult {
	if err != nil {
		if IsHaltException(err) {
			return modelroute.AttemptResult{Fail: modelroute.FailRefused}
		}
		return modelroute.AttemptResult{Fail: modelroute.FailTransport}
	}
	if comp == nil || (strings.TrimSpace(comp.Message.Content) == "" && len(comp.Message.ToolCalls) == 0) {
		return modelroute.AttemptResult{Fail: modelroute.FailUnderpowered}
	}
	return modelroute.AttemptResult{Succeeded: true, Verify: modelroute.VerifyWitness}
}

func (r *PerAspectRouter) buildHTTPSubjectRequest(req *http.Request, aspect modelroute.Aspect, workItem string, messages []agent.Message, tools []agent.ToolDef) AspectRouteRequest {
	labels := map[string]string{}
	if p := principalFromContext(req.Context()); p != "" {
		labels["tenant"] = p
	} else if p := strings.TrimSpace(req.Header.Get("X-Fak-Principal")); p != "" {
		labels["tenant"] = p
	}
	if sens := strings.TrimSpace(req.Header.Get(HeaderFakSensitivity)); sens != "" {
		labels["sensitivity"] = sens
	}
	if wc := strings.TrimSpace(req.Header.Get(HeaderFakWorkClass)); wc != "" {
		labels[modelroute.ClassLabel] = wc
	}
	toolName := strings.TrimSpace(req.Header.Get(HeaderFakTool))
	if toolName == "" && len(tools) == 1 {
		toolName = tools[0].Function.Name
	}
	return AspectRouteRequest{
		WorkItemID: workItem,
		OnBoxOnly:  isSensitiveRoutingLabels(labels),
		Subject: modelroute.Subject{
			Aspect:       aspect,
			Tool:         toolName,
			PromptTokens: estimateMessageContentTokens(messages),
			Complexity:   parseComplexityHeader(req.Header.Get(HeaderFakComplexity)),
			Labels:       labels,
		},
	}
}

func (r *PerAspectRouter) writeExhaustedHTTP(w http.ResponseWriter, workItem string, aspect modelroute.Aspect, err error) {
	if st, ok := r.State(workItem); ok {
		w.Header().Set(HeaderFakRouteAspect, string(aspect))
		w.Header().Set(HeaderFakEscalationAct, string(modelroute.ActionStop))
		if st.ExhaustedReason != "" {
			w.Header().Set(HeaderFakEscalationWhy, st.ExhaustedReason)
		}
		w.Header().Set(HeaderFakPriorEscalation, strconv.Itoa(st.PriorEscalations))
	}
	writeErrCode(w, http.StatusServiceUnavailable, "escalation_bounds_exhausted", err.Error())
}

func (r *PerAspectRouter) writeStopHeaders(w http.ResponseWriter, aspect modelroute.Aspect, dec AspectRouteDecision, verdict modelroute.EscalationVerdict) {
	h := w.Header()
	h.Set(HeaderFakRouteAspect, string(aspect))
	h.Set(HeaderFakRouteModel, dec.Placement.Model)
	h.Set(HeaderFakRouteZone, string(dec.Placement.Zone))
	h.Set(HeaderFakRouteEscalated, boolLabel(dec.Escalated))
	h.Set(HeaderFakEscalationAct, string(verdict.Action))
	h.Set(HeaderFakEscalationWhy, verdict.Reason)
	h.Set(HeaderFakPriorEscalation, strconv.Itoa(dec.PriorEscalations))
}

func isSensitiveRoutingLabels(labels map[string]string) bool {
	if len(labels) == 0 {
		return false
	}
	s := strings.TrimSpace(strings.ToLower(labels["sensitivity"]))
	if s == "" {
		s = strings.TrimSpace(strings.ToLower(labels["data_sensitivity"]))
	}
	return s != "" && s != "public" && s != "none"
}

func aspectFromHTTPRequest(req *http.Request, chat ChatRequest) modelroute.Aspect {
	if raw := strings.TrimSpace(req.Header.Get(HeaderFakAspect)); raw != "" {
		return modelroute.Aspect(strings.ToLower(raw))
	}
	if len(chat.Tools) > 0 {
		return modelroute.AspectToolCall
	}
	for _, m := range chat.Messages {
		if m.Role == agent.RoleTool || len(m.ToolCalls) > 0 {
			return modelroute.AspectToolCall
		}
	}
	return modelroute.AspectRequest
}

func aspectFromHeaderOrDefault(req *http.Request, def modelroute.Aspect) modelroute.Aspect {
	if raw := strings.TrimSpace(req.Header.Get(HeaderFakAspect)); raw != "" {
		return modelroute.Aspect(strings.ToLower(raw))
	}
	return def
}

func workItemFromHTTPRequest(req *http.Request, fallback string) string {
	for _, h := range []string{HeaderFakWorkItem, "X-Session-ID", "X-Trace-Id"} {
		if v := strings.TrimSpace(req.Header.Get(h)); v != "" {
			return v
		}
	}
	return strings.TrimSpace(fallback)
}

func parseComplexityHeader(raw string) modelroute.Complexity {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case string(modelroute.ComplexityLow):
		return modelroute.ComplexityLow
	case string(modelroute.ComplexityMedium):
		return modelroute.ComplexityMedium
	case string(modelroute.ComplexityHigh):
		return modelroute.ComplexityHigh
	default:
		return modelroute.ComplexityAny
	}
}

func rewriteJSONModel(raw []byte, model string) []byte {
	if model == "" {
		return raw
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil || payload == nil {
		return raw
	}
	payload["model"] = model
	out, err := json.Marshal(payload)
	if err != nil {
		return raw
	}
	return out
}

func copyResponseHeaders(dst, src http.Header) {
	for k, vals := range src {
		for _, v := range vals {
			dst.Add(k, v)
		}
	}
}
