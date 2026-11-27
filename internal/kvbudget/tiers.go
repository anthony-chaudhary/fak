package kvbudget

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
)

// TierClass identifies the canonical agentic hierarchy role for a KV slot tier.
type TierClass = string

const (
	TierMaster TierClass = "master"
	TierWorker TierClass = "worker"
	TierMicro  TierClass = "micro"
)

// TierWorkloadProfile selects the closed-form token budget split across L1/L2/L3.
type TierWorkloadProfile string

const (
	ProfileUltracode             TierWorkloadProfile = "ultracode"
	ProfileBalanced              TierWorkloadProfile = "balanced"
	ProfileMicroBurst            TierWorkloadProfile = "micro_burst"
	ProfileDatacenterMultiTenant TierWorkloadProfile = "datacenter_multi_tenant"
)

// TierSpec defines one KV-cache slot class in the multi-level hierarchy.
type TierSpec struct {
	Name                string `json:"name"`
	Level               int    `json:"level"` // 1=L1, 2=L2, 3=L3
	MaxContext          int    `json:"max_context"`
	MinSlots            int    `json:"min_slots"`
	MaxSlots            int    `json:"max_slots"` // 0 = bounded only by pool
	Priority            int    `json:"priority"`  // lower = higher priority
	Preemptible         bool   `json:"preemptible"`
	AllowBorrowFromIdle bool   `json:"allow_borrow_from_idle"`
}

// TierPlan is the deterministic output of AutotuneTiers / AutotuneTiersFromTokenBudget.
type TierPlan struct {
	TotalTokenCapacity int                 `json:"total_token_capacity"`
	Specs              []TierSpec          `json:"specs"`
	Profile            TierWorkloadProfile `json:"profile"`
}

// TierStatus describes the live capacity and utilization of one tier.
type TierStatus struct {
	Name             string `json:"name"`
	Level            int    `json:"level"`
	MaxContext       int    `json:"max_context"`
	MinSlots         int    `json:"min_slots"`
	ActiveSlots      int    `json:"active_slots"`
	AvailableNow     int    `json:"available_now"`
	ElasticAvailable int    `json:"elastic_available"`
	Preemptible      bool   `json:"preemptible"`
}

// FanoutRecommendation summarizes the safe concurrent fan-out shape for an
// agentic orchestrator reading /v1/fak/capacity/tiers or Snapshot().
type FanoutRecommendation struct {
	MasterMaxCtx int `json:"master_max_ctx"`
	WorkerSlots  int `json:"worker_slots"`
	WorkerMaxCtx int `json:"worker_max_ctx"`
	MicroSlots   int `json:"micro_slots"`
	MicroMaxCtx  int `json:"micro_max_ctx"`
}

// TierPoolSnapshot is a point-in-time read of a TierPool's state.
type TierPoolSnapshot struct {
	Enabled           bool                 `json:"enabled"`
	TotalTokens       int                  `json:"total_tokens"`
	FreeTokens        int                  `json:"free_tokens"`
	Tiers             []TierStatus         `json:"tiers"`
	RecommendedFanout FanoutRecommendation `json:"recommended_fanout"`
}

// TierLease represents an active slot allocation returned by TierPool.Acquire.
type TierLease struct {
	ID       uint64 `json:"id"`
	Tier     string `json:"tier"`
	Tokens   int    `json:"tokens"`
	Borrowed bool   `json:"borrowed"`
}

// TierVerdict is the structured admission outcome returned by TierPool.Acquire.
type TierVerdict struct {
	Admitted       bool         `json:"admitted"`
	Reason         string       `json:"reason"`
	Tier           string       `json:"tier"`
	CapTokens      int          `json:"cap_tokens"`
	Borrowed       bool         `json:"borrowed"`
	AdviceCode     string       `json:"advice_code,omitempty"`
	AvailableTiers []TierStatus `json:"available_tiers,omitempty"`
}

var (
	ErrInvalidTokenBudget = errors.New("kvbudget: total token capacity must be positive")
	ErrInvalidTierSpecs   = errors.New("kvbudget: invalid tier specs")
	ErrConservationFailed = errors.New("kvbudget: sum(MinSlots * MaxContext) exceeds total token capacity")
	ErrUnknownProfile     = errors.New("kvbudget: unknown tier workload profile")
)

const (
	defaultL1MaxCtx = 131072 // 128k
	defaultL2MaxCtx = 32768  // 32k
	defaultL3MaxCtx = 4096   // 4k
)

type profileWeights struct {
	l1 float64
	l2 float64
	l3 float64
}

func weightsForProfile(p TierWorkloadProfile) (profileWeights, error) {
	switch p {
	case "", ProfileUltracode:
		return profileWeights{l1: 1.0 / 3.0, l2: 1.0 / 3.0, l3: 1.0 / 3.0}, nil
	case ProfileBalanced:
		return profileWeights{l1: 0.25, l2: 0.50, l3: 0.25}, nil
	case ProfileMicroBurst:
		return profileWeights{l1: 0.20, l2: 0.30, l3: 0.50}, nil
	case ProfileDatacenterMultiTenant:
		return profileWeights{l1: 0.25, l2: 0.45, l3: 0.30}, nil
	default:
		return profileWeights{}, fmt.Errorf("%w: %q", ErrUnknownProfile, p)
	}
}

func largestPow2AtMost(n int) int {
	if n <= 0 {
		return 0
	}
	p := 1
	for p <= n/2 {
		p <<= 1
	}
	return p
}

// AutotuneTiers derives a deterministic L1/L2/L3 TierPlan from usable KV bytes,
// model cache Shape, and KV Quant, delegating to AutotuneTiersFromTokenBudget.
func AutotuneTiers(usableBytes uint64, shape Shape, quant Quant, profile TierWorkloadProfile) (TierPlan, error) {
	if usableBytes == 0 {
		return TierPlan{}, ErrInvalidTokenBudget
	}
	bpt := shape.KVBytesPerToken(quant)
	if bpt <= 0 || math.IsNaN(bpt) || math.IsInf(bpt, 0) {
		return TierPlan{}, fmt.Errorf("kvbudget: non-positive KV bytes per token (%.4f)", bpt)
	}
	tokensFloat := math.Floor(float64(usableBytes) / bpt)
	var totalTokens int
	if tokensFloat >= float64(math.MaxInt) {
		totalTokens = math.MaxInt
	} else {
		totalTokens = int(tokensFloat)
	}
	if totalTokens <= 0 {
		return TierPlan{}, ErrInvalidTokenBudget
	}
	return AutotuneTiersFromTokenBudget(totalTokens, profile)
}

// AutotuneTiersFromTokenBudget derives L1/L2/L3 TierSpecs from a total KV token
// budget using a closed-form proportional split that preserves conservation:
// sum(MinSlots_i * MaxContext_i) <= totalTokenCapacity.
func AutotuneTiersFromTokenBudget(totalTokenCapacity int, profile TierWorkloadProfile) (TierPlan, error) {
	if totalTokenCapacity <= 0 {
		return TierPlan{}, ErrInvalidTokenBudget
	}
	if profile == "" {
		profile = ProfileUltracode
	}
	w, err := weightsForProfile(profile)
	if err != nil {
		return TierPlan{}, err
	}

	l1Ctx, l2Ctx, l3Ctx := chooseTierContexts(totalTokenCapacity)

	// Allocate L1 slots from its target share, ensuring >= 1 slot whenever the
	// budget can hold at least one L1 + L2 + L3 minimum footprint.
	l1Slots := int(math.Floor(float64(totalTokenCapacity) * w.l1 / float64(l1Ctx)))
	if l1Slots == 0 && totalTokenCapacity >= l1Ctx+l2Ctx+l3Ctx {
		l1Slots = 1
	} else if l1Slots == 0 && totalTokenCapacity >= l1Ctx {
		l1Slots = 1
	}
	rem := totalTokenCapacity - l1Slots*l1Ctx

	// Allocate L2 slots from remaining budget according to w2/(w2+w3).
	l2Share := w.l2 / (w.l2 + w.l3)
	l2Budget := int(math.Floor(float64(rem) * l2Share))
	if profile == ProfileUltracode {
		// Exact closed-form 1/3 split on canonical 384k (393216) budget.
		targetL2 := int(math.Floor(float64(totalTokenCapacity) * w.l2))
		if targetL2 <= rem {
			l2Budget = targetL2
		}
	}
	l2Slots := l2Budget / l2Ctx
	if l2Slots == 0 && rem >= l2Ctx+l3Ctx {
		l2Slots = 1
	} else if l2Slots == 0 && rem >= l2Ctx {
		l2Slots = 1
	}
	rem -= l2Slots * l2Ctx

	// L3 absorbs remaining whole slots.
	l3Slots := rem / l3Ctx

	specs := []TierSpec{
		{
			Name:                TierMaster,
			Level:               1,
			MaxContext:          l1Ctx,
			MinSlots:            l1Slots,
			MaxSlots:            0,
			Priority:            1,
			Preemptible:         false,
			AllowBorrowFromIdle: true,
		},
		{
			Name:                TierWorker,
			Level:               2,
			MaxContext:          l2Ctx,
			MinSlots:            l2Slots,
			MaxSlots:            0,
			Priority:            2,
			Preemptible:         false,
			AllowBorrowFromIdle: true,
		},
		{
			Name:                TierMicro,
			Level:               3,
			MaxContext:          l3Ctx,
			MinSlots:            l3Slots,
			MaxSlots:            0,
			Priority:            3,
			Preemptible:         true,
			AllowBorrowFromIdle: true,
		},
	}

	return TierPlan{
		TotalTokenCapacity: totalTokenCapacity,
		Specs:              specs,
		Profile:            profile,
	}, nil
}

func chooseTierContexts(totalTokens int) (l1, l2, l3 int) {
	if totalTokens >= defaultL1MaxCtx+defaultL2MaxCtx+defaultL3MaxCtx {
		return defaultL1MaxCtx, defaultL2MaxCtx, defaultL3MaxCtx
	}
	l1 = largestPow2AtMost(totalTokens / 2)
	if l1 > defaultL1MaxCtx {
		l1 = defaultL1MaxCtx
	}
	if l1 < 1024 {
		l1 = max(1, totalTokens/2)
	}
	l2 = largestPow2AtMost(l1 / 4)
	if l2 > defaultL2MaxCtx {
		l2 = defaultL2MaxCtx
	}
	if l2 < 512 {
		l2 = max(1, l1/4)
	}
	l3 = largestPow2AtMost(l2 / 8)
	if l3 > defaultL3MaxCtx {
		l3 = defaultL3MaxCtx
	}
	if l3 < 256 {
		l3 = max(1, l2/4)
	}
	return l1, l2, l3
}

type tierState struct {
	spec        TierSpec
	baseMin     int
	activeSlots int
}

type activeLease struct {
	tier   string
	tokens int
}

// TierPool is a concurrency-safe dynamic allocator and elastic borrower over
// L1/L2/L3 KV-cache slot classes.
type TierPool struct {
	mu          sync.Mutex
	totalTokens int
	tiers       []*tierState
	byName      map[string]*tierState
	nextID      uint64
	active      map[uint64]activeLease
}

// NewTierPool constructs a TierPool after validating the conservation invariant
// sum(MinSlots_i * MaxContext_i) <= totalTokens.
func NewTierPool(totalTokens int, specs []TierSpec) (*TierPool, error) {
	if totalTokens <= 0 {
		return nil, ErrInvalidTokenBudget
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("%w: empty specs", ErrInvalidTierSpecs)
	}

	seen := make(map[string]struct{}, len(specs))
	states := make([]*tierState, 0, len(specs))
	byName := make(map[string]*tierState, len(specs))
	var minSum int64

	for _, sp := range specs {
		if sp.Name == "" {
			return nil, fmt.Errorf("%w: empty tier name", ErrInvalidTierSpecs)
		}
		if _, dup := seen[sp.Name]; dup {
			return nil, fmt.Errorf("%w: duplicate tier %q", ErrInvalidTierSpecs, sp.Name)
		}
		seen[sp.Name] = struct{}{}
		if sp.MaxContext <= 0 {
			return nil, fmt.Errorf("%w: tier %q MaxContext must be > 0", ErrInvalidTierSpecs, sp.Name)
		}
		if sp.MinSlots < 0 || sp.MaxSlots < 0 {
			return nil, fmt.Errorf("%w: tier %q negative slot bound", ErrInvalidTierSpecs, sp.Name)
		}
		if sp.MaxSlots > 0 && sp.MinSlots > sp.MaxSlots {
			return nil, fmt.Errorf("%w: tier %q MinSlots (%d) > MaxSlots (%d)", ErrInvalidTierSpecs, sp.Name, sp.MinSlots, sp.MaxSlots)
		}
		if sp.MinSlots > 0 && int64(sp.MaxContext) > math.MaxInt64/int64(sp.MinSlots) {
			return nil, fmt.Errorf("%w: tier %q MinSlots*MaxContext overflows int64", ErrInvalidTierSpecs, sp.Name)
		}
		slotToks := int64(sp.MinSlots) * int64(sp.MaxContext)
		if minSum > math.MaxInt64-slotToks {
			return nil, fmt.Errorf("%w: sum(MinSlots*MaxContext) overflows int64", ErrConservationFailed)
		}
		minSum += slotToks
		st := &tierState{spec: sp, baseMin: sp.MinSlots}
		states = append(states, st)
		byName[sp.Name] = st
	}

	if minSum > int64(totalTokens) {
		return nil, fmt.Errorf("%w: %d > %d", ErrConservationFailed, minSum, totalTokens)
	}

	sort.Slice(states, func(i, j int) bool {
		if states[i].spec.Level != states[j].spec.Level {
			return states[i].spec.Level < states[j].spec.Level
		}
		return states[i].spec.MaxContext > states[j].spec.MaxContext
	})

	return &TierPool{
		totalTokens: totalTokens,
		tiers:       states,
		byName:      byName,
		nextID:      1,
		active:      make(map[uint64]activeLease),
	}, nil
}

// Classify picks the smallest tier whose MaxContext >= tokens.
func (p *TierPool) Classify(tokens int) (string, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if tokens < 0 {
		return "", false
	}
	var best *tierState
	for _, st := range p.tiers {
		if st.spec.MaxContext >= tokens {
			if best == nil || st.spec.MaxContext < best.spec.MaxContext {
				best = st
			}
		}
	}
	if best == nil {
		return "", false
	}
	return best.spec.Name, true
}

func (p *TierPool) freeTokensLocked() int {
	used := 0
	for _, st := range p.tiers {
		used += st.activeSlots * st.spec.MaxContext
	}
	if used >= p.totalTokens {
		return 0
	}
	return p.totalTokens - used
}

// lockedOtherFloorsLocked returns the sum of unallocated floor tokens owned by
// tiers other than target that disallow idle borrowing.
func (p *TierPool) lockedOtherFloorsLocked(target string) int {
	locked := 0
	for _, st := range p.tiers {
		if st.spec.Name == target || st.spec.AllowBorrowFromIdle {
			continue
		}
		if st.spec.MinSlots > st.activeSlots {
			locked += (st.spec.MinSlots - st.activeSlots) * st.spec.MaxContext
		}
	}
	return locked
}

func (p *TierPool) availableTokensForLocked(st *tierState) int {
	free := p.freeTokensLocked()
	locked := p.lockedOtherFloorsLocked(st.spec.Name)
	if free <= locked {
		return 0
	}
	return free - locked
}

func (p *TierPool) tierStatusLocked(st *tierState) TierStatus {
	availTokens := p.availableTokensForLocked(st)
	availSlots := availTokens / st.spec.MaxContext
	if st.spec.MaxSlots > 0 {
		remCap := st.spec.MaxSlots - st.activeSlots
		if remCap < 0 {
			remCap = 0
		}
		if availSlots > remCap {
			availSlots = remCap
		}
	}
	ownRemaining := st.spec.MinSlots - st.activeSlots
	if ownRemaining < 0 {
		ownRemaining = 0
	}
	elastic := availSlots - ownRemaining
	if elastic < 0 {
		elastic = 0
	}
	return TierStatus{
		Name:             st.spec.Name,
		Level:            st.spec.Level,
		MaxContext:       st.spec.MaxContext,
		MinSlots:         st.spec.MinSlots,
		ActiveSlots:      st.activeSlots,
		AvailableNow:     availSlots,
		ElasticAvailable: elastic,
		Preemptible:      st.spec.Preemptible,
	}
}

func (p *TierPool) allStatusesLocked() []TierStatus {
	out := make([]TierStatus, 0, len(p.tiers))
	for _, st := range p.tiers {
		out = append(out, p.tierStatusLocked(st))
	}
	return out
}

// Acquire allocates one slot in tierName for a request of up to tokens.
// It uses the tier's own guaranteed floor first, then elastically borrows
// unreserved capacity from other tiers when permitted.
func (p *TierPool) Acquire(tierName string, tokens int) (TierLease, TierVerdict) {
	p.mu.Lock()
	defer p.mu.Unlock()

	st, ok := p.byName[tierName]
	if !ok {
		return TierLease{}, TierVerdict{
			Admitted:       false,
			Reason:         fmt.Sprintf("unknown tier %q", tierName),
			Tier:           tierName,
			AdviceCode:     "UNKNOWN_TIER",
			AvailableTiers: p.allStatusesLocked(),
		}
	}
	if tokens < 0 {
		return TierLease{}, TierVerdict{
			Admitted:       false,
			Reason:         "negative token request",
			Tier:           tierName,
			CapTokens:      st.spec.MaxContext,
			AdviceCode:     "INVALID_TOKENS",
			AvailableTiers: p.allStatusesLocked(),
		}
	}
	if tokens > st.spec.MaxContext {
		return TierLease{}, TierVerdict{
			Admitted:       false,
			Reason:         fmt.Sprintf("requested %d tokens exceeds tier %q cap %d", tokens, tierName, st.spec.MaxContext),
			Tier:           tierName,
			CapTokens:      st.spec.MaxContext,
			AdviceCode:     "EXCEEDS_TIER_MAX_CONTEXT",
			AvailableTiers: p.allStatusesLocked(),
		}
	}
	if st.spec.MaxSlots > 0 && st.activeSlots >= st.spec.MaxSlots {
		return TierLease{}, TierVerdict{
			Admitted:       false,
			Reason:         fmt.Sprintf("tier %q reached MaxSlots (%d)", tierName, st.spec.MaxSlots),
			Tier:           tierName,
			CapTokens:      st.spec.MaxContext,
			AdviceCode:     p.exhaustedAdviceCodeLocked(tierName),
			AvailableTiers: p.allStatusesLocked(),
		}
	}

	availTokens := p.availableTokensForLocked(st)
	if availTokens < st.spec.MaxContext {
		return TierLease{}, TierVerdict{
			Admitted:       false,
			Reason:         fmt.Sprintf("insufficient KV capacity for tier %q (%d needed, %d available)", tierName, st.spec.MaxContext, availTokens),
			Tier:           tierName,
			CapTokens:      st.spec.MaxContext,
			AdviceCode:     p.exhaustedAdviceCodeLocked(tierName),
			AvailableTiers: p.allStatusesLocked(),
		}
	}

	borrowed := st.activeSlots >= st.spec.MinSlots
	st.activeSlots++
	id := p.nextID
	p.nextID++
	p.active[id] = activeLease{
		tier:   tierName,
		tokens: st.spec.MaxContext,
	}

	reason := "admitted from guaranteed tier floor"
	if borrowed {
		reason = "admitted via elastic borrow from idle capacity"
	}

	lease := TierLease{
		ID:       id,
		Tier:     tierName,
		Tokens:   st.spec.MaxContext,
		Borrowed: borrowed,
	}
	verdict := TierVerdict{
		Admitted:  true,
		Reason:    reason,
		Tier:      tierName,
		CapTokens: st.spec.MaxContext,
		Borrowed:  borrowed,
	}
	return lease, verdict
}

func (p *TierPool) exhaustedAdviceCodeLocked(requestedTier string) string {
	for _, st := range p.tiers {
		if st.spec.Name == requestedTier {
			continue
		}
		if p.tierStatusLocked(st).AvailableNow > 0 {
			return "TIER_EXHAUSTED_DOWNGRADE_AVAILABLE"
		}
	}
	return "POOL_EXHAUSTED"
}

// Release frees an active TierLease. Calling Release multiple times with the
// same lease is idempotent and safe.
func (p *TierPool) Release(lease TierLease) {
	if lease.ID == 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	act, ok := p.active[lease.ID]
	if !ok {
		return
	}
	delete(p.active, lease.ID)
	if st, exists := p.byName[act.tier]; exists && st.activeSlots > 0 {
		st.activeSlots--
	}
}

// Rebalance shifts MinSlots toward high-demand tiers when observed demand
// exceeds a hysteresis threshold, while preserving all active slots and the
// conservation invariant sum(MinSlots * MaxContext) <= totalTokens.
func (p *TierPool) Rebalance(demand map[string]int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(demand) == 0 || len(p.tiers) == 0 {
		return
	}

	// Hysteresis: only act when a tier's demand exceeds its current MinSlots by
	// at least max(2, MinSlots/4) slots.
	const minExcess = 2

	for _, receiver := range p.tiers {
		d := demand[receiver.spec.Name]
		hysteresis := max(minExcess, receiver.spec.MinSlots/4)
		if d < receiver.spec.MinSlots+hysteresis {
			continue
		}
		if receiver.spec.MaxSlots > 0 && receiver.spec.MinSlots >= receiver.spec.MaxSlots {
			continue
		}

		// Reclaim idle MinSlots from under-demanded tiers.
		for _, donor := range p.tiers {
			if donor.spec.Name == receiver.spec.Name {
				continue
			}
			donorDemand := demand[donor.spec.Name]
			donorFloor := max(donor.activeSlots, donorDemand)
			if donor.spec.MinSlots <= donorFloor {
				continue
			}
			reclaimableSlots := donor.spec.MinSlots - donorFloor
			reclaimedTokens := reclaimableSlots * donor.spec.MaxContext
			addSlots := reclaimedTokens / receiver.spec.MaxContext
			if addSlots <= 0 {
				continue
			}

			needed := d - receiver.spec.MinSlots
			if addSlots > needed {
				addSlots = needed
			}
			if receiver.spec.MaxSlots > 0 && receiver.spec.MinSlots+addSlots > receiver.spec.MaxSlots {
				addSlots = receiver.spec.MaxSlots - receiver.spec.MinSlots
			}
			if addSlots <= 0 {
				continue
			}

			tokensNeeded := addSlots * receiver.spec.MaxContext
			donorSlotsTaken := ceilDiv(tokensNeeded, donor.spec.MaxContext)
			if donorSlotsTaken > reclaimableSlots {
				donorSlotsTaken = reclaimableSlots
				tokensNeeded = donorSlotsTaken * donor.spec.MaxContext
				addSlots = tokensNeeded / receiver.spec.MaxContext
			}
			if addSlots <= 0 || donorSlotsTaken <= 0 {
				continue
			}

			donor.spec.MinSlots -= donorSlotsTaken
			receiver.spec.MinSlots += addSlots
			if receiver.spec.MinSlots >= d {
				break
			}
		}
	}
}

// Snapshot returns a point-in-time view of pool capacity, per-tier status, and
// recommended agentic fan-out.
func (p *TierPool) Snapshot() TierPoolSnapshot {
	p.mu.Lock()
	defer p.mu.Unlock()

	statuses := p.allStatusesLocked()
	var rec FanoutRecommendation
	for _, st := range statuses {
		switch st.Name {
		case TierMaster:
			rec.MasterMaxCtx = st.MaxContext
		case TierWorker:
			rec.WorkerMaxCtx = st.MaxContext
			rec.WorkerSlots = min(st.MinSlots, st.AvailableNow)
			if rec.WorkerSlots == 0 && st.AvailableNow > 0 {
				rec.WorkerSlots = st.AvailableNow
			}
		case TierMicro:
			rec.MicroMaxCtx = st.MaxContext
			rec.MicroSlots = min(st.MinSlots, st.AvailableNow)
			if rec.MicroSlots == 0 && st.AvailableNow > 0 {
				rec.MicroSlots = st.AvailableNow
			}
		}
	}

	return TierPoolSnapshot{
		Enabled:           len(statuses) > 0 && p.totalTokens > 0,
		TotalTokens:       p.totalTokens,
		FreeTokens:        p.freeTokensLocked(),
		Tiers:             statuses,
		RecommendedFanout: rec,
	}
}
