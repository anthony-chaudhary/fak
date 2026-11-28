package engine

// replica_router.go — Engine-level prefix-aware multi-replica/TP routing and
// fail-closed runtime admission over ContinuousBatcher replicas (and external
// PrefixResidencyIndex workers), cleanly layered inside internal/engine without
// importing internal/gateway.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/refutil"
	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

var (
	// ErrPlacementNotAdmitted refuses creating a router from an un-admitted or empty placement plan.
	ErrPlacementNotAdmitted = errors.New("replica_router: placement plan is not admitted")
	// ErrPlacementExceedsBatcherSlots refuses a replica whose MaxSlots exceeds ContinuousBatcher limits.
	ErrPlacementExceedsBatcherSlots = errors.New("replica_router: replica max_slots exceeds continuous_batcher limit")
	// ErrReplicaRouterClosed is returned after Close.
	ErrReplicaRouterClosed = errors.New("replica_router: router closed")
	// ErrNoAdmissibleReplica is returned when no replica passes admission.
	ErrNoAdmissibleReplica = errors.New("replica_router: no admissible replica")
	// ErrReplicaKVCapacityExceeded is returned when KV budgets refuse all replicas.
	ErrReplicaKVCapacityExceeded = errors.New("replica_router: KV capacity exceeded across all replicas")
	// ErrReplicaSlotsExhausted is returned when slot budgets are exhausted across all replicas.
	ErrReplicaSlotsExhausted = errors.New("replica_router: concurrent slot capacity exhausted across all replicas")
	// ErrReplicaVRAMExceeded is returned when live device VRAM probes fail admission.
	ErrReplicaVRAMExceeded = errors.New("replica_router: live device VRAM budget exceeded")
)

const (
	defaultPrefixLRUCapacity  = 64
	defaultSkewAbsLoad        = 4
	defaultSkewRelLoad        = 1.5
	defaultTargetVRAMPressure = 0.95
)

// ReplicaRouteReason is the closed vocabulary explaining why a request landed on a replica.
type ReplicaRouteReason string

const (
	ReplicaRoutePrefixHit     ReplicaRouteReason = "prefix_hit"
	ReplicaRouteAffinityHit   ReplicaRouteReason = "affinity_hit"
	ReplicaRouteColdLeastLoad ReplicaRouteReason = "cold_least_load"
	ReplicaRouteSkewFallback  ReplicaRouteReason = "skew_fallback"
)

// ReplicaRouteReasons returns the closed route-reason vocabulary in stable order.
func ReplicaRouteReasons() []ReplicaRouteReason {
	return []ReplicaRouteReason{
		ReplicaRoutePrefixHit,
		ReplicaRouteAffinityHit,
		ReplicaRouteColdLeastLoad,
		ReplicaRouteSkewFallback,
	}
}

// Valid reports whether r is a member of the closed route-reason vocabulary.
func (r ReplicaRouteReason) Valid() bool {
	switch r {
	case ReplicaRoutePrefixHit, ReplicaRouteAffinityHit, ReplicaRouteColdLeastLoad, ReplicaRouteSkewFallback:
		return true
	default:
		return false
	}
}

// SkewPolicy bounds how far ahead a hot replica may load before prefix locality
// sheds to the least-loaded admissible replica.
type SkewPolicy struct {
	AbsLoad int
	RelLoad float64
}

func (p SkewPolicy) normalized() SkewPolicy {
	if p.AbsLoad < 1 {
		p.AbsLoad = defaultSkewAbsLoad
	}
	if p.RelLoad < 1.0 {
		p.RelLoad = defaultSkewRelLoad
	}
	return p
}

func (p SkewPolicy) skewed(candidateLoad, minLoad, admissibleCount int) bool {
	if admissibleCount <= 1 {
		return false
	}
	minFloor := minLoad
	if minFloor < 1 {
		minFloor = 1
	}
	return (candidateLoad-minLoad) >= p.AbsLoad && float64(candidateLoad) >= p.RelLoad*float64(minFloor)
}

// MultiReplicaConfig configures a MultiReplicaRouter over an admitted SingleNodePlacementPlan.
type MultiReplicaConfig struct {
	// BaseBatcherConfig supplies shared model, KVBytesPerToken, LoopedBlock,
	// PrefillBudget, and roofline defaults for each replica's ContinuousBatcher.
	BaseBatcherConfig ContinuousBatcherConfig

	// Skew bounds locality herding. Zero defaults to AbsLoad=4, RelLoad=1.5.
	Skew SkewPolicy

	// PrefixCapacityPerReplica bounds per-replica tracked prompt prefixes (<=0 -> 64).
	PrefixCapacityPerReplica int

	// MaxWaitingPerReplica bounds waiting queue depth per replica. 0 (default)
	// enforces strict fail-closed immediate admission with zero waiting queue.
	MaxWaitingPerReplica int

	// DisableStickyAffinity turns off session/affinity-key stickiness.
	DisableStickyAffinity bool

	// DeviceBackends optionally provides live compute.Backend probes keyed by GPU index
	// for fail-closed live VRAM checks before admitting requests onto a replica's GPUs.
	DeviceBackends map[int]compute.Backend

	// TargetVRAMPressure is the live VRAM high-water fraction in (0, 1] enforced when
	// DeviceBackends are attached (<=0 -> 0.95).
	TargetVRAMPressure float64

	// ResidencyIndex optionally links the engine-level block PrefixResidencyIndex
	// (shared with vLLM/SGLang adapters) for block-digest prefix matching.
	ResidencyIndex *PrefixResidencyIndex
}

// ReplicaRouteReceipt records the routing decision and admission state for one request.
type ReplicaRouteReceipt struct {
	SessionID           string             `json:"session_id"`
	ReplicaIndex        int                `json:"replica_index"`
	ReplicaID           string             `json:"replica_id"`
	GPUIDs              []int              `json:"gpu_ids"`
	TPDegree            int                `json:"tp_degree"`
	Strategy            PlacementStrategy  `json:"strategy"`
	MatchedPrefixTokens int                `json:"matched_prefix_tokens"`
	MatchedPrefixBlocks int                `json:"matched_prefix_blocks,omitempty"`
	PrefixLength        int                `json:"prefix_length"`
	RouteReason         ReplicaRouteReason `json:"route_reason"`
	RequiredKVBytes     int64              `json:"required_kv_bytes"`
	ReplicaKVUsedBytes  int64              `json:"replica_kv_used_bytes"`
	ReplicaMaxKVBytes   int64              `json:"replica_max_kv_bytes"`
	ReplicaActiveSlots  int                `json:"replica_active_slots"`
	ReplicaMaxSlots     int                `json:"replica_max_slots"`
}

// ReplicaLoadStats captures live utilization and routing counters for one replica.
type ReplicaLoadStats struct {
	ReplicaIndex        int                 `json:"replica_index"`
	ReplicaID           string              `json:"replica_id"`
	GPUIDs              []int               `json:"gpu_ids"`
	TPDegree            int                 `json:"tp_degree"`
	Load                BatcherLoadSnapshot `json:"load"`
	TrackedPrefixes     int                 `json:"tracked_prefixes"`
	AdmittedRequests    int64               `json:"admitted_requests"`
	PrefixHitRoutes     int64               `json:"prefix_hit_routes"`
	PrefixMatchedTokens int64               `json:"prefix_matched_tokens"`
}

// MultiReplicaStats aggregates routing and replica utilization telemetry.
type MultiReplicaStats struct {
	Strategy        PlacementStrategy  `json:"strategy"`
	DP              int                `json:"dp"`
	TP              int                `json:"tp"`
	TotalAdmitted   int64              `json:"total_admitted"`
	TotalRefused    int64              `json:"total_refused"`
	PrefixHitRoutes int64              `json:"prefix_hit_routes"`
	ColdRoutes      int64              `json:"cold_routes"`
	SkewFallbacks   int64              `json:"skew_fallbacks"`
	AffinityHits    int64              `json:"affinity_hits"`
	Replicas        []ReplicaLoadStats `json:"replicas"`
}

// tokenPrefixLRU maintains a bounded LRU of resident prompt token sequences per replica.
type tokenPrefixLRU struct {
	capacity int
	entries  [][]int // oldest at 0, MRU at len-1
}

func newTokenPrefixLRU(capacity int) *tokenPrefixLRU {
	if capacity <= 0 {
		capacity = defaultPrefixLRUCapacity
	}
	return &tokenPrefixLRU{capacity: capacity}
}

func (l *tokenPrefixLRU) longestOverlap(prompt []int) int {
	if l == nil || len(prompt) == 0 {
		return 0
	}
	best := 0
	for _, held := range l.entries {
		if n := strmatch.CommonSlicePrefixLen(held, prompt); n > best {
			best = n
		}
	}
	return best
}

func (l *tokenPrefixLRU) admit(prompt []int) {
	if l == nil || len(prompt) == 0 {
		return
	}
	for i, held := range l.entries {
		if len(held) == len(prompt) && strmatch.CommonSlicePrefixLen(held, prompt) == len(prompt) {
			l.entries = append(l.entries[:i], l.entries[i+1:]...)
			l.entries = append(l.entries, held)
			return
		}
	}
	if len(l.entries) >= l.capacity {
		l.entries = l.entries[1:]
	}
	cp := append([]int(nil), prompt...)
	l.entries = append(l.entries, cp)
}

type managedReplica struct {
	placement       ReplicaPlacement
	batcher         *ContinuousBatcher
	prefixes        *tokenPrefixLRU
	admittedCount   int64
	prefixHits      int64
	prefixTokensSum int64
}

// MultiReplicaRouter routes requests across an admitted single-node DP/TP/hybrid
// placement plan with prefix-aware locality, bounded skew fallback, and fail-closed
// VRAM + KV + concurrent-slot admission.
type MultiReplicaRouter struct {
	mu               sync.Mutex
	plan             SingleNodePlacementPlan
	cfg              MultiReplicaConfig
	replicas         []*managedReplica
	sessionToReplica map[string]int
	affinityToRep    map[string]int
	seqCounter       atomic.Uint64
	closed           bool

	totalAdmitted int64
	totalRefused  int64
	prefixHits    int64
	coldRoutes    int64
	skewFallbacks int64
	affinityHits  int64
}

// NewMultiReplicaRouter constructs per-replica ContinuousBatchers from an admitted
// SingleNodePlacementPlan. Fails closed if the plan is not admitted or exceeds
// ContinuousBatcher slot limits.
func NewMultiReplicaRouter(plan SingleNodePlacementPlan, cfg MultiReplicaConfig) (*MultiReplicaRouter, error) {
	if !plan.Admitted || plan.Status != PlacementAdmitted || len(plan.Replicas) == 0 {
		return nil, ErrPlacementNotAdmitted
	}
	cfg.Skew = cfg.Skew.normalized()
	if cfg.PrefixCapacityPerReplica <= 0 {
		cfg.PrefixCapacityPerReplica = defaultPrefixLRUCapacity
	}
	if cfg.MaxWaitingPerReplica < 0 {
		cfg.MaxWaitingPerReplica = 0
	}
	if cfg.TargetVRAMPressure <= 0 || cfg.TargetVRAMPressure > 1 {
		cfg.TargetVRAMPressure = defaultTargetVRAMPressure
	}

	replicas := make([]*managedReplica, len(plan.Replicas))
	cleanup := func(upto int) {
		for i := 0; i < upto; i++ {
			if replicas[i] != nil && replicas[i].batcher != nil {
				_ = replicas[i].batcher.Close()
			}
		}
	}

	for i, rp := range plan.Replicas {
		if rp.MaxSlots < MinSlots || rp.MaxSlots > MaxSlots {
			cleanup(i)
			return nil, fmt.Errorf("%w: replica %d max_slots=%d outside [1, %d]", ErrPlacementExceedsBatcherSlots, i, rp.MaxSlots, MaxSlots)
		}
		if rp.MaxKVCacheBytes <= 0 {
			cleanup(i)
			return nil, fmt.Errorf("%w: replica %d has non-positive max_kv_cache_bytes %d", ErrPlacementNotAdmitted, i, rp.MaxKVCacheBytes)
		}
		bcfg := cfg.BaseBatcherConfig
		bcfg.MaxSlots = rp.MaxSlots
		bcfg.MaxKVCacheBytes = rp.MaxKVCacheBytes
		if rp.WeightsPerRankBytes > 0 {
			bcfg.ModelWeightsBytes = rp.WeightsPerRankBytes
		}
		cb, err := NewContinuousBatcher(bcfg)
		if err != nil {
			cleanup(i)
			return nil, err
		}
		replicas[i] = &managedReplica{
			placement: rp,
			batcher:   cb,
			prefixes:  newTokenPrefixLRU(cfg.PrefixCapacityPerReplica),
		}
	}

	return &MultiReplicaRouter{
		plan:             plan,
		cfg:              cfg,
		replicas:         replicas,
		sessionToReplica: make(map[string]int),
		affinityToRep:    make(map[string]int),
	}, nil
}

// Plan returns the admitted placement plan backing this router.
func (r *MultiReplicaRouter) Plan() SingleNodePlacementPlan {
	return r.plan
}

// ReplicaCount returns the number of DP replica groups managed by the router.
func (r *MultiReplicaRouter) ReplicaCount() int {
	return len(r.replicas)
}

// Batcher returns the underlying ContinuousBatcher for replica index idx, or nil.
func (r *MultiReplicaRouter) Batcher(idx int) *ContinuousBatcher {
	if idx < 0 || idx >= len(r.replicas) {
		return nil
	}
	return r.replicas[idx].batcher
}

// Admit routes req to an admissible replica using prefix locality + bounded skew,
// enforcing fail-closed VRAM, KV cache, and concurrent slot limits.
func (r *MultiReplicaRouter) Admit(req *SubagentRequest) (*ReplicaRouteReceipt, error) {
	if req == nil {
		return nil, ErrNilRequest
	}
	if req.TargetTokens <= 0 {
		if req.ExecutionDepth > 0 || req.RecurrentLoops > 0 {
			req.TargetTokens = req.effectiveDepth()
		} else {
			return nil, ErrInvalidTargetTokens
		}
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil, ErrReplicaRouterClosed
	}

	// Assign cluster-unique SessionID upfront so two replicas never mint colliding IDs.
	sessionID := req.SessionID
	if sessionID != "" {
		if idx, exists := r.sessionToReplica[sessionID]; exists && idx >= 0 && idx < len(r.replicas) {
			if slot, ok := r.replicas[idx].batcher.GetSlot(sessionID); ok && slot.State != SlotStateFinished {
				return nil, ErrSessionAlreadyExists
			}
		}
	} else {
		for {
			id := r.seqCounter.Add(1)
			cand := fmt.Sprintf("subagent-%d", id)
			if _, exists := r.sessionToReplica[cand]; !exists {
				sessionID = cand
				break
			}
		}
		req.SessionID = sessionID
	}

	blocks := parsePrefixBlocks(req.Metadata)
	affinityKey := affinityKeyForRequest(req, sessionID)

	excluded := make(map[int]bool)
	for {
		cands, refusalErr := r.admissibleCandidatesLocked(req, blocks, excluded)
		if len(cands) == 0 {
			r.totalRefused++
			return nil, refusalErr
		}

		chosen, reason := r.pickCandidateLocked(cands, affinityKey)
		rep := r.replicas[chosen.index]

		var submitErr error
		if r.cfg.MaxWaitingPerReplica == 0 {
			_, submitErr = rep.batcher.TryAdmit(req)
		} else {
			_, submitErr = rep.batcher.SubmitBounded(req, r.cfg.MaxWaitingPerReplica)
		}
		if submitErr != nil {
			if errors.Is(submitErr, ErrBatcherSlotsFull) || errors.Is(submitErr, ErrBatcherKVOverBudget) {
				excluded[chosen.index] = true
				continue
			}
			r.totalRefused++
			return nil, submitErr
		}

		rep.prefixes.admit(req.PromptTokens)
		if r.cfg.ResidencyIndex != nil && len(blocks) > 0 {
			rows := make([]PrefixResidency, 0, len(blocks))
			for _, b := range blocks {
				rows = append(rows, PrefixResidency{
					WorkerID: rep.placement.ReplicaID,
					Digest:   b,
					Medium:   "GPU",
				})
			}
			r.cfg.ResidencyIndex.Store(rep.placement.ReplicaID, rows...)
		}

		r.sessionToReplica[sessionID] = chosen.index
		if !r.cfg.DisableStickyAffinity && affinityKey != "" {
			r.affinityToRep[affinityKey] = chosen.index
		}

		rep.admittedCount++
		r.totalAdmitted++
		if chosen.matchedTokens > 0 || chosen.matchedBlocks > 0 {
			rep.prefixHits++
			rep.prefixTokensSum += int64(chosen.matchedTokens)
		}
		switch reason {
		case ReplicaRoutePrefixHit:
			r.prefixHits++
		case ReplicaRouteAffinityHit:
			r.affinityHits++
		case ReplicaRouteSkewFallback:
			r.skewFallbacks++
		case ReplicaRouteColdLeastLoad:
			r.coldRoutes++
		default:
		}

		afterSnap := rep.batcher.LoadSnapshot()
		return &ReplicaRouteReceipt{
			SessionID:           sessionID,
			ReplicaIndex:        chosen.index,
			ReplicaID:           rep.placement.ReplicaID,
			GPUIDs:              append([]int(nil), rep.placement.GPUIDs...),
			TPDegree:            rep.placement.TPDegree,
			Strategy:            r.plan.Strategy,
			MatchedPrefixTokens: chosen.matchedTokens,
			MatchedPrefixBlocks: chosen.matchedBlocks,
			PrefixLength:        len(req.PromptTokens),
			RouteReason:         reason,
			RequiredKVBytes:     chosen.reqKVBytes,
			ReplicaKVUsedBytes:  afterSnap.CurrentKVCacheBytes,
			ReplicaMaxKVBytes:   afterSnap.MaxKVCacheBytes,
			ReplicaActiveSlots:  afterSnap.ActiveSlots + afterSnap.PrefillingSlots + afterSnap.YieldedSlots,
			ReplicaMaxSlots:     afterSnap.MaxSlots,
		}, nil
	}
}

type scoredReplicaCandidate struct {
	index         int
	replicaID     string
	load          int
	kvRatio       float64
	reqKVBytes    int64
	matchedTokens int
	matchedBlocks int
}

func (r *MultiReplicaRouter) admissibleCandidatesLocked(req *SubagentRequest, blocks []string, excluded map[int]bool) ([]scoredReplicaCandidate, error) {
	var (
		cands       []scoredReplicaCandidate
		kvBlocked   int
		slotBlocked int
		vramBlocked int
	)

	for i, rep := range r.replicas {
		if excluded[i] {
			continue
		}
		snap := rep.batcher.LoadSnapshot()
		reqKV := rep.batcher.RequiredKVCacheBytes(req)

		if snap.MaxKVCacheBytes > 0 {
			usedAndQueued := snap.CurrentKVCacheBytes + snap.QueuedKVCacheBytes
			if reqKV > snap.MaxKVCacheBytes || usedAndQueued+reqKV > snap.MaxKVCacheBytes {
				kvBlocked++
				continue
			}
		}

		if r.cfg.MaxWaitingPerReplica == 0 {
			if snap.EmptySlots <= 0 || snap.WaitingCount > 0 {
				slotBlocked++
				continue
			}
		} else {
			if snap.EmptySlots <= 0 && snap.WaitingCount >= r.cfg.MaxWaitingPerReplica {
				slotBlocked++
				continue
			}
		}

		if !r.replicaLiveVRAMOK(rep, reqKV) {
			vramBlocked++
			continue
		}

		load := snap.ActiveSlots + snap.PrefillingSlots + snap.YieldedSlots + snap.WaitingCount
		kvRatio := 0.0
		if snap.MaxKVCacheBytes > 0 {
			kvRatio = float64(snap.CurrentKVCacheBytes+snap.QueuedKVCacheBytes) / float64(snap.MaxKVCacheBytes)
		}
		matchedTokens := rep.prefixes.longestOverlap(req.PromptTokens)
		matchedBlocks := 0
		if r.cfg.ResidencyIndex != nil && len(blocks) > 0 {
			matchedBlocks = contiguousBlockPrefixMatch(r.cfg.ResidencyIndex, rep.placement.ReplicaID, blocks)
		}

		cands = append(cands, scoredReplicaCandidate{
			index:         i,
			replicaID:     rep.placement.ReplicaID,
			load:          load,
			kvRatio:       kvRatio,
			reqKVBytes:    reqKV,
			matchedTokens: matchedTokens,
			matchedBlocks: matchedBlocks,
		})
	}

	if len(cands) > 0 {
		return cands, nil
	}
	if vramBlocked > 0 && kvBlocked == 0 && slotBlocked == 0 {
		return nil, ErrReplicaVRAMExceeded
	}
	if kvBlocked > 0 && slotBlocked == 0 {
		return nil, ErrReplicaKVCapacityExceeded
	}
	if slotBlocked > 0 && kvBlocked == 0 {
		return nil, ErrReplicaSlotsExhausted
	}
	if kvBlocked > 0 {
		return nil, ErrReplicaKVCapacityExceeded
	}
	return nil, ErrNoAdmissibleReplica
}

func (r *MultiReplicaRouter) replicaLiveVRAMOK(rep *managedReplica, reqKV int64) bool {
	if len(r.cfg.DeviceBackends) == 0 {
		return true
	}
	kvShards := int64(rep.placement.KVShards)
	if kvShards < 1 {
		kvShards = 1
	}
	reqKVPerRank := ceilDivInt64(reqKV, kvShards)
	for _, gpuID := range rep.placement.GPUIDs {
		b, ok := r.cfg.DeviceBackends[gpuID]
		if !ok || b == nil {
			continue
		}
		total, free, known := compute.DeviceMemoryInfo(b)
		if !known || total <= 0 {
			return false
		}
		if free >= 0 && free < reqKVPerRank {
			return false
		}
		resident := rep.placement.WeightsPerRankBytes + rep.placement.OverheadPerRankBytes + reqKVPerRank
		if pressure, _, pknown := DeviceHBMPressure(b, resident); pknown && pressure >= r.cfg.TargetVRAMPressure {
			return false
		}
	}
	return true
}

func (r *MultiReplicaRouter) pickCandidateLocked(cands []scoredReplicaCandidate, affinityKey string) (scoredReplicaCandidate, ReplicaRouteReason) {
	minLoad, maxLoad := cands[0].load, cands[0].load
	maxOverlap := 0
	for _, c := range cands {
		if c.load < minLoad {
			minLoad = c.load
		}
		if c.load > maxLoad {
			maxLoad = c.load
		}
		if ov := effectiveOverlapScore(c.matchedTokens, c.matchedBlocks); ov > maxOverlap {
			maxOverlap = ov
		}
	}

	// Sticky session affinity when home replica is admissible and not over skew bound.
	if !r.cfg.DisableStickyAffinity && affinityKey != "" {
		if homeIdx, ok := r.affinityToRep[affinityKey]; ok {
			for _, c := range cands {
				if c.index == homeIdx && !r.cfg.Skew.skewed(c.load, minLoad, len(cands)) {
					ov := effectiveOverlapScore(c.matchedTokens, c.matchedBlocks)
					if ov > 0 || maxOverlap == 0 {
						return c, ReplicaRouteAffinityHit
					}
				}
			}
		}
	}

	leastLoaded := pickLeastLoadedCandidate(cands)
	if maxOverlap == 0 {
		return leastLoaded, ReplicaRouteColdLeastLoad
	}
	if r.cfg.Skew.skewed(maxLoad, minLoad, len(cands)) {
		return leastLoaded, ReplicaRouteSkewFallback
	}

	// Prefix-aware locality choice: maximize overlap/(1+load), tie-break by overlap,
	// lower load, lower kvRatio, then lower replica index.
	best := cands[0]
	bestScore := prefixScore(best)
	for _, c := range cands[1:] {
		sc := prefixScore(c)
		switch {
		case sc > bestScore:
			best, bestScore = c, sc
		case sc == bestScore && effectiveOverlapScore(c.matchedTokens, c.matchedBlocks) > effectiveOverlapScore(best.matchedTokens, best.matchedBlocks):
			best = c
		case sc == bestScore && effectiveOverlapScore(c.matchedTokens, c.matchedBlocks) == effectiveOverlapScore(best.matchedTokens, best.matchedBlocks) && c.load < best.load:
			best = c
		case sc == bestScore && effectiveOverlapScore(c.matchedTokens, c.matchedBlocks) == effectiveOverlapScore(best.matchedTokens, best.matchedBlocks) && c.load == best.load && c.kvRatio < best.kvRatio:
			best = c
		case sc == bestScore && effectiveOverlapScore(c.matchedTokens, c.matchedBlocks) == effectiveOverlapScore(best.matchedTokens, best.matchedBlocks) && c.load == best.load && c.kvRatio == best.kvRatio && c.index < best.index:
			best = c
		}
	}
	if effectiveOverlapScore(best.matchedTokens, best.matchedBlocks) == 0 {
		return leastLoaded, ReplicaRouteColdLeastLoad
	}
	return best, ReplicaRoutePrefixHit
}

func effectiveOverlapScore(matchedTokens, matchedBlocks int) int {
	if matchedTokens > 0 {
		return matchedTokens
	}
	return matchedBlocks
}

func prefixScore(c scoredReplicaCandidate) float64 {
	ov := effectiveOverlapScore(c.matchedTokens, c.matchedBlocks)
	return float64(ov) / float64(1+c.load)
}

func pickLeastLoadedCandidate(cands []scoredReplicaCandidate) scoredReplicaCandidate {
	best := cands[0]
	for _, c := range cands[1:] {
		switch {
		case c.load < best.load:
			best = c
		case c.load == best.load && c.kvRatio < best.kvRatio:
			best = c
		case c.load == best.load && c.kvRatio == best.kvRatio && c.index < best.index:
			best = c
		}
	}
	return best
}

func affinityKeyForRequest(req *SubagentRequest, sessionID string) string {
	if req != nil && req.Metadata != nil {
		if k := strings.TrimSpace(req.Metadata["affinity_key"]); k != "" {
			return k
		}
	}
	return strings.TrimSpace(sessionID)
}

func parsePrefixBlocks(meta map[string]string) []string {
	if meta == nil {
		return nil
	}
	raw := strings.TrimSpace(meta["prefix_blocks"])
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if s := strings.TrimSpace(p); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func contiguousBlockPrefixMatch(idx *PrefixResidencyIndex, worker string, blocks []string) int {
	if idx == nil || worker == "" || len(blocks) == 0 {
		return 0
	}
	matched := 0
	for _, b := range blocks {
		if !idx.Has(worker, b) {
			break
		}
		matched++
	}
	return matched
}

// StepPhaseAll advances every replica one budgeted scheduler step in deterministic replica order.
func (r *MultiReplicaRouter) StepPhaseAll(ctx context.Context) ([]*BatchStepResult, error) {
	out := make([]*BatchStepResult, len(r.replicas))
	for i, rep := range r.replicas {
		res, err := rep.batcher.StepPhase(ctx)
		if err != nil {
			return out, err
		}
		out[i] = res
	}
	return out, nil
}

// StepAll advances every replica one decode-focused step in deterministic replica order.
func (r *MultiReplicaRouter) StepAll(ctx context.Context) ([]*BatchStepResult, error) {
	out := make([]*BatchStepResult, len(r.replicas))
	for i, rep := range r.replicas {
		res, err := rep.batcher.Step(ctx)
		if err != nil {
			return out, err
		}
		out[i] = res
	}
	return out, nil
}

// GetSlot resolves sessionID from the replica that owns it.
func (r *MultiReplicaRouter) GetSlot(sessionID string) (*Slot, int, bool) {
	r.mu.Lock()
	idx, ok := r.sessionToReplica[sessionID]
	r.mu.Unlock()
	if !ok || idx < 0 || idx >= len(r.replicas) {
		return nil, -1, false
	}
	slot, found := r.replicas[idx].batcher.GetSlot(sessionID)
	return slot, idx, found
}

// YieldSlot yields sessionID on its owning replica.
func (r *MultiReplicaRouter) YieldSlot(sessionID string) error {
	r.mu.Lock()
	idx, ok := r.sessionToReplica[sessionID]
	r.mu.Unlock()
	if !ok || idx < 0 || idx >= len(r.replicas) {
		return ErrSessionNotFound
	}
	return r.replicas[idx].batcher.YieldSlot(sessionID)
}

// ResumeSlot resumes sessionID on its owning replica.
func (r *MultiReplicaRouter) ResumeSlot(sessionID string) error {
	r.mu.Lock()
	idx, ok := r.sessionToReplica[sessionID]
	r.mu.Unlock()
	if !ok || idx < 0 || idx >= len(r.replicas) {
		return ErrSessionNotFound
	}
	return r.replicas[idx].batcher.ResumeSlot(sessionID)
}

// Cancel cancels sessionID on its owning replica.
func (r *MultiReplicaRouter) Cancel(sessionID string) error {
	r.mu.Lock()
	idx, ok := r.sessionToReplica[sessionID]
	r.mu.Unlock()
	if !ok || idx < 0 || idx >= len(r.replicas) {
		return ErrSessionNotFound
	}
	return r.replicas[idx].batcher.Cancel(sessionID)
}

func (r *MultiReplicaRouter) acknowledgeCompleted(sessionID string, seq uint64) {
	r.mu.Lock()
	idx, ok := r.sessionToReplica[sessionID]
	r.mu.Unlock()
	if !ok || idx < 0 || idx >= len(r.replicas) {
		return
	}
	cb := r.replicas[idx].batcher
	cb.mu.Lock()
	if slot := cb.completedMap[sessionID]; slot != nil && slot.SubmissionSeq == seq {
		delete(cb.completedMap, sessionID)
	}
	cb.mu.Unlock()
}

// Stats returns cluster-wide and per-replica load and routing telemetry.
func (r *MultiReplicaRouter) Stats() MultiReplicaStats {
	r.mu.Lock()
	defer r.mu.Unlock()
	stats := MultiReplicaStats{
		Strategy:        r.plan.Strategy,
		DP:              r.plan.DP,
		TP:              r.plan.TP,
		TotalAdmitted:   r.totalAdmitted,
		TotalRefused:    r.totalRefused,
		PrefixHitRoutes: r.prefixHits,
		ColdRoutes:      r.coldRoutes,
		SkewFallbacks:   r.skewFallbacks,
		AffinityHits:    r.affinityHits,
		Replicas:        make([]ReplicaLoadStats, len(r.replicas)),
	}
	for i, rep := range r.replicas {
		stats.Replicas[i] = ReplicaLoadStats{
			ReplicaIndex:        rep.placement.ReplicaIndex,
			ReplicaID:           rep.placement.ReplicaID,
			GPUIDs:              append([]int(nil), rep.placement.GPUIDs...),
			TPDegree:            rep.placement.TPDegree,
			Load:                rep.batcher.LoadSnapshot(),
			TrackedPrefixes:     len(rep.prefixes.entries),
			AdmittedRequests:    rep.admittedCount,
			PrefixHitRoutes:     rep.prefixHits,
			PrefixMatchedTokens: rep.prefixTokensSum,
		}
	}
	return stats
}

// Close closes every replica's ContinuousBatcher.
func (r *MultiReplicaRouter) Close() error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil
	}
	r.closed = true
	replicas := append([]*managedReplica(nil), r.replicas...)
	r.mu.Unlock()

	var firstErr error
	for _, rep := range replicas {
		if err := rep.batcher.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// PrefixWorkerCandidate describes an external (e.g. vLLM/SGLang) replica/TP group
// registered in a PrefixResidencyIndex for engine-level prefix-aware routing.
type PrefixWorkerCandidate struct {
	WorkerID     string
	ReplicaIndex int
	TPDegree     int
	GPUIDs       []int
	ActiveSlots  int
	MaxSlots     int
	KVBytesUsed  int64
	MaxKVBytes   int64
	Healthy      bool
}

// PrefixWorkerRouteDecision is the outcome of RouteWorkersByPrefix.
type PrefixWorkerRouteDecision struct {
	WorkerID      string
	ReplicaIndex  int
	TPDegree      int
	GPUIDs        []int
	MatchedBlocks int
	TotalBlocks   int
	Reason        ReplicaRouteReason
	WorkerLoad    int
}

// RouteWorkersByPrefix selects a prefix-aware target among external vLLM/SGLang workers
// tracked in idx, enforcing fail-closed slot and KV budget checks before scoring.
func RouteWorkersByPrefix(
	idx *PrefixResidencyIndex,
	candidates []PrefixWorkerCandidate,
	prefixBlocks []string,
	reqKVBytes int64,
	skew SkewPolicy,
) (PrefixWorkerRouteDecision, error) {
	if len(candidates) == 0 {
		return PrefixWorkerRouteDecision{}, ErrNoAdmissibleReplica
	}
	skew = skew.normalized()
	var (
		cands       []scoredReplicaCandidate
		kvBlocked   int
		slotBlocked int
	)
	for i, w := range candidates {
		if !w.Healthy || w.WorkerID == "" {
			continue
		}
		if w.MaxKVBytes <= 0 || reqKVBytes < 0 || reqKVBytes > w.MaxKVBytes || w.KVBytesUsed+reqKVBytes > w.MaxKVBytes {
			kvBlocked++
			continue
		}
		if w.MaxSlots <= 0 || w.ActiveSlots >= w.MaxSlots {
			slotBlocked++
			continue
		}
		matched := contiguousBlockPrefixMatch(idx, w.WorkerID, prefixBlocks)
		cands = append(cands, scoredReplicaCandidate{
			index:         i,
			replicaID:     w.WorkerID,
			load:          w.ActiveSlots,
			kvRatio:       float64(w.KVBytesUsed) / float64(w.MaxKVBytes),
			reqKVBytes:    reqKVBytes,
			matchedBlocks: matched,
		})
	}
	if len(cands) == 0 {
		if kvBlocked > 0 && slotBlocked == 0 {
			return PrefixWorkerRouteDecision{}, ErrReplicaKVCapacityExceeded
		}
		if slotBlocked > 0 && kvBlocked == 0 {
			return PrefixWorkerRouteDecision{}, ErrReplicaSlotsExhausted
		}
		if kvBlocked > 0 {
			return PrefixWorkerRouteDecision{}, ErrReplicaKVCapacityExceeded
		}
		return PrefixWorkerRouteDecision{}, ErrNoAdmissibleReplica
	}

	minLoad, maxLoad := cands[0].load, cands[0].load
	maxOverlap := 0
	for _, c := range cands {
		if c.load < minLoad {
			minLoad = c.load
		}
		if c.load > maxLoad {
			maxLoad = c.load
		}
		if c.matchedBlocks > maxOverlap {
			maxOverlap = c.matchedBlocks
		}
	}

	leastLoaded := pickLeastLoadedCandidate(cands)
	chosen := leastLoaded
	reason := ReplicaRouteColdLeastLoad
	if maxOverlap > 0 {
		if skew.skewed(maxLoad, minLoad, len(cands)) {
			reason = ReplicaRouteSkewFallback
		} else {
			best := cands[0]
			bestScore := prefixScore(best)
			for _, c := range cands[1:] {
				sc := prefixScore(c)
				if sc > bestScore ||
					(sc == bestScore && c.matchedBlocks > best.matchedBlocks) ||
					(sc == bestScore && c.matchedBlocks == best.matchedBlocks && c.load < best.load) ||
					(sc == bestScore && c.matchedBlocks == best.matchedBlocks && c.load == best.load && c.kvRatio < best.kvRatio) ||
					(sc == bestScore && c.matchedBlocks == best.matchedBlocks && c.load == best.load && c.kvRatio == best.kvRatio && c.index < best.index) {
					best, bestScore = c, sc
				}
			}
			chosen = best
			reason = ReplicaRoutePrefixHit
		}
	}

	orig := candidates[chosen.index]
	return PrefixWorkerRouteDecision{
		WorkerID:      orig.WorkerID,
		ReplicaIndex:  orig.ReplicaIndex,
		TPDegree:      orig.TPDegree,
		GPUIDs:        append([]int(nil), orig.GPUIDs...),
		MatchedBlocks: chosen.matchedBlocks,
		TotalBlocks:   len(prefixBlocks),
		Reason:        reason,
		WorkerLoad:    chosen.load,
	}, nil
}

// EngineIDSingleNodeCluster is the capability/engine identifier for SingleNodeServingEngine.
const EngineIDSingleNodeCluster = "single-node-cluster"

// SingleNodeServingEngine adapts MultiReplicaRouter onto the abi.LifecycleEngine
// seam so multi-replica/TP single-node pools execute through the same Admit/Complete
// lifecycle contract as BatchingEngine.
type SingleNodeServingEngine struct {
	router       *MultiReplicaRouter
	stepInterval time.Duration
	maxSteps     int

	mu       sync.Mutex
	requests map[string]*singleNodeEngineRequest

	ctx       context.Context
	cancel    context.CancelFunc
	done      chan struct{}
	wake      chan struct{}
	closeOnce sync.Once
	recorder  *enginestep.Recorder
}

type singleNodeEngineRequest struct {
	*batcherRequest
	route *ReplicaRouteReceipt
}

// NewSingleNodeServingEngine builds an abi.LifecycleEngine backed by MultiReplicaRouter.
func NewSingleNodeServingEngine(plan SingleNodePlacementPlan, cfg MultiReplicaConfig) (*SingleNodeServingEngine, error) {
	router, err := NewMultiReplicaRouter(plan, cfg)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	e := &SingleNodeServingEngine{
		router:       router,
		stepInterval: defaultBatchingStepInterval,
		maxSteps:     defaultBatchingMaxSteps,
		requests:     make(map[string]*singleNodeEngineRequest),
		ctx:          ctx,
		cancel:       cancel,
		done:         make(chan struct{}),
		wake:         make(chan struct{}, 1),
		recorder:     enginestep.Default,
	}
	go e.run()
	return e, nil
}

// Caps advertises the lifecycle seam and single-node multi-replica/TP routing capability.
func (e *SingleNodeServingEngine) Caps() []abi.Capability {
	return []abi.Capability{
		abi.EngineLifecycleCap,
		abi.Capability("engine.batcher"),
		abi.Capability("engine.single-node-cluster"),
	}
}

// WeightBearing declares that the single-node cluster engine executes model inference.
func (e *SingleNodeServingEngine) WeightBearing() bool { return true }

// Router exposes the underlying MultiReplicaRouter for stats and inspection.
func (e *SingleNodeServingEngine) Router() *MultiReplicaRouter { return e.router }

type clusterAdmitArgs struct {
	batcherAdmitArgs
	KVBytesPerToken int64    `json:"kv_bytes_per_token,omitempty"`
	AffinityKey     string   `json:"affinity_key,omitempty"`
	PrefixBlocks    []string `json:"prefix_blocks,omitempty"`
}

// Admit routes the incoming call across the placed replicas, failing closed on VRAM/KV/slot exhaustion.
func (e *SingleNodeServingEngine) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	if c == nil {
		return nil, ErrNilRequest
	}
	if e.ctx.Err() != nil {
		return nil, ErrReplicaRouterClosed
	}

	var args clusterAdmitArgs
	if raw := refutil.Bytes(ctx, c.Args); len(raw) > 0 {
		if err := json.Unmarshal(raw, &args); err != nil {
			return nil, fmt.Errorf("single-node-cluster: invalid args: %w", err)
		}
	}

	var meta map[string]string
	if args.AffinityKey != "" || len(args.PrefixBlocks) > 0 {
		meta = make(map[string]string, 2)
		if args.AffinityKey != "" {
			meta["affinity_key"] = args.AffinityKey
		}
		if len(args.PrefixBlocks) > 0 {
			meta["prefix_blocks"] = strings.Join(args.PrefixBlocks, ",")
		}
	}

	req := &SubagentRequest{
		SessionID:       args.SessionID,
		PromptTokens:    args.PromptTokens,
		TargetTokens:    args.TargetTokens,
		ChunkedPrefill:  args.ChunkedPrefill,
		ExecutionDepth:  args.Depth,
		KVBytesPerToken: args.KVBytesPerToken,
		Metadata:        meta,
	}
	receipt, err := e.router.Admit(req)
	if err != nil {
		return nil, err
	}

	rctx, rcancel := context.WithCancel(ctx)
	br := &batcherRequest{
		sessionID:     receipt.SessionID,
		submissionSeq: req.SubmissionSeq,
		tool:          c.Tool,
		engine:        EngineIDSingleNodeCluster,
		tokens:        make(chan abi.EngineToken, req.TargetTokens+16),
		done:          make(chan struct{}),
		ctx:           rctx,
		cancel:        rcancel,
		putCtx:        ctx,
		enqueuedAt:    time.Now(),
		recorderDone:  e.recorder.RequestStart(),
	}
	br.stopCancelWake = context.AfterFunc(rctx, e.signal)
	sr := &singleNodeEngineRequest{batcherRequest: br, route: receipt}

	e.mu.Lock()
	if e.ctx.Err() != nil {
		e.mu.Unlock()
		_ = e.router.Cancel(receipt.SessionID)
		br.stopCancelWake()
		rcancel()
		return nil, ErrReplicaRouterClosed
	}
	e.requests[receipt.SessionID] = sr
	e.mu.Unlock()

	e.signal()
	return sr, nil
}

// Complete drains Admit to a one-shot abi.Result.
func (e *SingleNodeServingEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return completeViaAdmit(ctx, e, c)
}

// Close shuts down the cluster step loop and all replica batchers.
func (e *SingleNodeServingEngine) Close() error {
	e.closeOnce.Do(func() { e.cancel() })
	<-e.done
	return e.router.Close()
}

func (e *SingleNodeServingEngine) run() {
	defer close(e.done)
	idle := 0
	for {
		if e.ctx.Err() != nil {
			e.finishAll(ErrReplicaRouterClosed)
			return
		}
		if e.outstanding() == 0 {
			select {
			case <-e.wake:
			case <-e.ctx.Done():
				e.finishAll(ErrReplicaRouterClosed)
				return
			}
			continue
		}

		stepResults, err := e.router.StepPhaseAll(e.ctx)
		if err != nil && (e.ctx.Err() != nil || errors.Is(err, ErrBatcherClosed)) {
			e.finishAll(ErrReplicaRouterClosed)
			return
		}
		e.pumpAll()

		progressed := false
		for _, res := range stepResults {
			if res != nil && (res.PrefillTokens > 0 || res.DecodeTokens > 0 ||
				len(res.PromotedSessionIDs) > 0 || len(res.RetiredSessionIDs) > 0) {
				progressed = true
				break
			}
		}
		if progressed {
			idle = 0
		} else {
			idle++
		}
		if e.maxSteps > 0 && idle >= e.maxSteps {
			select {
			case <-e.wake:
				idle = 0
			case <-e.ctx.Done():
				e.finishAll(ErrReplicaRouterClosed)
				return
			}
			continue
		}
		if e.stepInterval > 0 {
			select {
			case <-e.ctx.Done():
				e.finishAll(ErrReplicaRouterClosed)
				return
			case <-time.After(e.stepInterval):
			}
		}
	}
}

func (e *SingleNodeServingEngine) pumpAll() {
	e.mu.Lock()
	reqs := make([]*singleNodeEngineRequest, 0, len(e.requests))
	for _, sr := range e.requests {
		reqs = append(reqs, sr)
	}
	e.mu.Unlock()

	getSlot := func(id string) (*Slot, bool) {
		slot, _, ok := e.router.GetSlot(id)
		return slot, ok
	}
	for _, sr := range reqs {
		pumpSlotTokens(
			e.ctx,
			sr.batcherRequest,
			getSlot,
			func(br *batcherRequest, res *abi.Result) {
				e.finishRequest(sr, res)
			},
			func(br *batcherRequest) {
				e.cancelRequest(sr)
			},
		)
	}
}

func (e *SingleNodeServingEngine) finishRequest(sr *singleNodeEngineRequest, res *abi.Result) {
	if res != nil && sr.route != nil {
		if res.Meta == nil {
			res.Meta = make(map[string]string)
		}
		res.Meta["replica_index"] = strconv.Itoa(sr.route.ReplicaIndex)
		res.Meta["replica_id"] = sr.route.ReplicaID
		res.Meta["tp_degree"] = strconv.Itoa(sr.route.TPDegree)
		res.Meta["strategy"] = string(sr.route.Strategy)
		res.Meta["matched_prefix_tokens"] = strconv.Itoa(sr.route.MatchedPrefixTokens)
		res.Meta["route_reason"] = string(sr.route.RouteReason)
	}
	sr.finish(res, nil)
	e.removeRequest(sr)
}

func (e *SingleNodeServingEngine) cancelRequest(sr *singleNodeEngineRequest) {
	_ = e.router.Cancel(sr.sessionID)
	sr.finish(nil, context.Canceled)
	e.removeRequest(sr)
}

func (e *SingleNodeServingEngine) finishAll(err error) {
	e.mu.Lock()
	reqs := make([]*singleNodeEngineRequest, 0, len(e.requests))
	for _, sr := range e.requests {
		reqs = append(reqs, sr)
	}
	e.mu.Unlock()

	for _, sr := range reqs {
		if err == nil {
			e.finishRequest(sr, sr.assemble())
		} else {
			sr.finish(nil, err)
			e.removeRequest(sr)
		}
	}
}

func (e *SingleNodeServingEngine) removeRequest(sr *singleNodeEngineRequest) {
	e.mu.Lock()
	if e.requests[sr.sessionID] == sr {
		delete(e.requests, sr.sessionID)
	}
	e.mu.Unlock()
	e.router.acknowledgeCompleted(sr.sessionID, sr.submissionSeq)
}

func (e *SingleNodeServingEngine) outstanding() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.requests)
}

func (e *SingleNodeServingEngine) signal() {
	select {
	case e.wake <- struct{}{}:
	default:
	}
}

var (
	_ abi.LifecycleEngine = (*SingleNodeServingEngine)(nil)
	_ abi.EngineRequest   = (*singleNodeEngineRequest)(nil)
)
