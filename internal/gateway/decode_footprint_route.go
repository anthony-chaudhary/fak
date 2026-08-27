package gateway

import (
	"sync"
)

const (
	defaultDecodeFootprintJournalCapacity = 128
	maxDecodeFootprintJournalCapacity     = 4096
	maxDecodeFootprintOutputTokens        = 1 << 20
)

// DecodeFootprintConfig bounds the output projection admitted to live residency
// scoring. UnknownOutputTokens is deliberately non-zero: an omitted max-token hint
// must reserve a conservative amount instead of looking free. MaxOutputTokens is a
// hard accounting bound, not a generation cap; the planner still receives the
// caller's original sampling options unchanged.
type DecodeFootprintConfig struct {
	BlockTokens         int
	UnknownOutputTokens int
	MaxOutputTokens     int
	Scale               float64
	JournalCapacity     int
}

// DefaultDecodeFootprintConfig keeps the unknown projection useful but bounded.
func DefaultDecodeFootprintConfig() DecodeFootprintConfig {
	return DecodeFootprintConfig{
		BlockTokens:         16,
		UnknownOutputTokens: 256,
		MaxOutputTokens:     4096,
		Scale:               1,
		JournalCapacity:     defaultDecodeFootprintJournalCapacity,
	}
}

func normalizeDecodeFootprintConfig(cfg DecodeFootprintConfig) DecodeFootprintConfig {
	def := DefaultDecodeFootprintConfig()
	if cfg.BlockTokens <= 0 {
		cfg.BlockTokens = def.BlockTokens
	}
	if cfg.MaxOutputTokens <= 0 {
		cfg.MaxOutputTokens = def.MaxOutputTokens
	}
	if cfg.MaxOutputTokens > maxDecodeFootprintOutputTokens {
		cfg.MaxOutputTokens = maxDecodeFootprintOutputTokens
	}
	if cfg.UnknownOutputTokens <= 0 {
		cfg.UnknownOutputTokens = def.UnknownOutputTokens
	}
	if cfg.UnknownOutputTokens > cfg.MaxOutputTokens {
		cfg.UnknownOutputTokens = cfg.MaxOutputTokens
	}
	if cfg.Scale <= 0 {
		cfg.Scale = def.Scale
	}
	if cfg.Scale > 1 {
		cfg.Scale = 1
	}
	if cfg.JournalCapacity <= 0 {
		cfg.JournalCapacity = def.JournalCapacity
	}
	if cfg.JournalCapacity > maxDecodeFootprintJournalCapacity {
		cfg.JournalCapacity = maxDecodeFootprintJournalCapacity
	}
	return cfg
}

// DecodeFootprintScore is one candidate's live score contribution at selection
// time. BookedOutputBlocks is the anticipated decode footprint of older live
// routes on that worker; EffectiveLoad is residency + in-flight count + those
// blocks. The request being placed is not charged until after selection, so the
// same new-request term cannot bias one candidate over another.
type DecodeFootprintScore struct {
	Worker             string
	Overlap            int
	BaseLoad           int
	BookedOutputBlocks int
	EffectiveLoad      int
	Score              float64
}

// DecodeFootprintRouteDecision is the bounded audit record carried by a live
// reservation. It is updated in place when a retry retargets the same logical
// booking and when observed output reconciles the prediction.
type DecodeFootprintRouteDecision struct {
	ID                    uint64
	Worker                string
	InitialWorker         string
	Engine                string
	Model                 string
	RequestedOutputTokens int
	ExpectedOutputTokens  int
	OutputCapTokens       int
	BookedOutputBlocks    int
	UnknownDefault        bool
	Capped                bool
	InitialCandidates     []DecodeFootprintScore
	InitialSelectedScore  float64
	Candidates            []DecodeFootprintScore
	SelectedScore         float64
	ObservedOutputTokens  int
	PredictionErrorTokens int
	Reconciled            bool
	Retargets             int
	Released              bool
	ReleaseReason         string
	ReleaseCount          int
}

// DecodeFootprintStats meters the live lifecycle without a second accounting
// source. PredictionErrorTokens is signed (observed - expected); the absolute
// companion supports calibration without cancellation between over/under-predicts.
type DecodeFootprintStats struct {
	Routes                        uint64
	UnknownDefaults               uint64
	CappedRoutes                  uint64
	CappedOutputTokens            uint64
	Retargets                     uint64
	Reconciliations               uint64
	Releases                      uint64
	CancellationReleases          uint64
	StreamCompletionReleases      uint64
	ActiveBookings                int
	ActiveBookedOutputBlocks      int
	PredictionErrorTokens         int64
	AbsolutePredictionErrorTokens uint64
}

type decodeFootprintRouteState struct {
	cfg     DecodeFootprintConfig
	nextID  uint64
	booked  map[string]int
	journal []*DecodeFootprintRouteDecision
	stats   DecodeFootprintStats
}

func newDecodeFootprintRouteState(cfg DecodeFootprintConfig) decodeFootprintRouteState {
	return decodeFootprintRouteState{cfg: normalizeDecodeFootprintConfig(cfg), booked: make(map[string]int)}
}

// WithDecodeFootprintConfig tunes the bounded live projection. It is intended for
// startup configuration, before routes are admitted. Existing active bookings are
// retained when the limits change.
func (p *CacheAwarePolicy) WithDecodeFootprintConfig(cfg DecodeFootprintConfig) *CacheAwarePolicy {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.decode.cfg = normalizeDecodeFootprintConfig(cfg)
	if p.decode.booked == nil {
		p.decode.booked = make(map[string]int)
	}
	return p
}

// DecodeFootprintDecisions returns the bounded route journal, newest last.
func (p *CacheAwarePolicy) DecodeFootprintDecisions() []DecodeFootprintRouteDecision {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]DecodeFootprintRouteDecision, len(p.decode.journal))
	for i, decision := range p.decode.journal {
		out[i] = cloneDecodeFootprintDecision(*decision)
	}
	return out
}

// DecodeFootprintStats returns a point-in-time copy of the lifecycle meters.
func (p *CacheAwarePolicy) DecodeFootprintStats() DecodeFootprintStats {
	if p == nil {
		return DecodeFootprintStats{}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.decode.stats
}

// DecodeFootprintActiveByWorker exposes the current anticipated occupancy used by
// the scorer. The returned map is detached from policy state.
func (p *CacheAwarePolicy) DecodeFootprintActiveByWorker() map[string]int {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make(map[string]int, len(p.decode.booked))
	for worker, blocks := range p.decode.booked {
		out[worker] = blocks
	}
	return out
}

type decodeFootprintRouteRequest struct {
	ExpectedOutputTokens int
}

// decodeFootprintPickPolicy is an additive live-reservation extension. Other
// PickPolicy implementations keep the historical Pick contract unchanged.
type decodeFootprintPickPolicy interface {
	reserveDecodeFootprint([]PlannerReplica, []string, func(string) int, func(string) int, decodeFootprintRouteRequest) (PlannerReplica, *decodeFootprintReservation, bool)
	retargetDecodeFootprint(*decodeFootprintReservation, []PlannerReplica, []string, func(string) int, func(string) int) (PlannerReplica, bool)
}

type decodeFootprintReservation struct {
	policy    *CacheAwarePolicy
	decision  *DecodeFootprintRouteDecision
	reconcile sync.Once
	release   sync.Once
}

func (p *CacheAwarePolicy) reserveDecodeFootprint(candidates []PlannerReplica, prefix []string, load, fleetBooked func(string) int, req decodeFootprintRouteRequest) (PlannerReplica, *decodeFootprintReservation, bool) {
	if p == nil || len(candidates) == 0 {
		return PlannerReplica{}, nil, false
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	cfg := p.decode.cfg
	if cfg.BlockTokens <= 0 {
		cfg = normalizeDecodeFootprintConfig(cfg)
		p.decode.cfg = cfg
	}
	requested := req.ExpectedOutputTokens
	expected := requested
	unknown := expected <= 0
	if unknown {
		expected = cfg.UnknownOutputTokens
	}
	capped := expected > cfg.MaxOutputTokens
	if capped {
		expected = cfg.MaxOutputTokens
	}
	blocks := AnticipatedDecodeBlocks(DecodeFootprintInputs{
		ExpectedOutputTokens: expected,
		BlockTokens:          cfg.BlockTokens,
		Scale:                cfg.Scale,
	})

	names, byName := decodeReplicaNames(candidates)
	withBooked := p.decodeLoadLocked(load, fleetBooked)
	chosen := p.chooseWorkerLocked(names, prefix, withBooked)
	scores := p.decodeScoresLocked(names, prefix, load, fleetBooked)
	p.index.Observe(chosen, prefix)
	p.decode.booked[chosen] = saturatingNonnegativeAdd(p.decode.booked[chosen], blocks)
	p.decode.nextID++
	record := &DecodeFootprintRouteDecision{
		ID:                    p.decode.nextID,
		Worker:                chosen,
		InitialWorker:         chosen,
		Model:                 byName[chosen].Planner.Model(),
		RequestedOutputTokens: requested,
		ExpectedOutputTokens:  expected,
		OutputCapTokens:       cfg.MaxOutputTokens,
		BookedOutputBlocks:    blocks,
		UnknownDefault:        unknown,
		Capped:                capped,
		InitialCandidates:     append([]DecodeFootprintScore(nil), scores...),
		InitialSelectedScore:  selectedDecodeScore(scores, chosen),
		Candidates:            scores,
		SelectedScore:         selectedDecodeScore(scores, chosen),
	}
	p.decode.journal = append(p.decode.journal, record)
	if overflow := len(p.decode.journal) - cfg.JournalCapacity; overflow > 0 {
		copy(p.decode.journal, p.decode.journal[overflow:])
		p.decode.journal = p.decode.journal[:cfg.JournalCapacity]
	}
	p.decode.stats.Routes++
	if unknown {
		p.decode.stats.UnknownDefaults++
	}
	if capped {
		p.decode.stats.CappedRoutes++
		p.decode.stats.CappedOutputTokens += uint64(requested - expected)
	}
	p.decode.stats.ActiveBookings++
	p.decode.stats.ActiveBookedOutputBlocks = saturatingNonnegativeAdd(p.decode.stats.ActiveBookedOutputBlocks, blocks)
	return byName[chosen], &decodeFootprintReservation{policy: p, decision: record}, true
}

func (p *CacheAwarePolicy) retargetDecodeFootprint(booking *decodeFootprintReservation, candidates []PlannerReplica, prefix []string, load, fleetBooked func(string) int) (PlannerReplica, bool) {
	if p == nil || booking == nil || booking.policy != p || booking.decision == nil || len(candidates) == 0 {
		return PlannerReplica{}, false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	record := booking.decision
	if record.Released {
		return PlannerReplica{}, false
	}
	names, byName := decodeReplicaNames(candidates)
	withBooked := p.decodeLoadLocked(load, fleetBooked)
	chosen := p.chooseWorkerLocked(names, prefix, withBooked)
	scores := p.decodeScoresLocked(names, prefix, load, fleetBooked)
	p.removeDecodeBlocksLocked(record.Worker, record.BookedOutputBlocks)
	p.decode.booked[chosen] = saturatingNonnegativeAdd(p.decode.booked[chosen], record.BookedOutputBlocks)
	p.index.Observe(chosen, prefix)
	record.Worker = chosen
	record.Model = byName[chosen].Planner.Model()
	record.Candidates = scores
	record.SelectedScore = selectedDecodeScore(scores, chosen)
	record.Retargets++
	p.decode.stats.Retargets++
	return byName[chosen], true
}

func decodeReplicaNames(candidates []PlannerReplica) ([]string, map[string]PlannerReplica) {
	names := make([]string, len(candidates))
	byName := make(map[string]PlannerReplica, len(candidates))
	for i, candidate := range candidates {
		names[i] = candidate.Name
		byName[candidate.Name] = candidate
	}
	return names, byName
}

func (p *CacheAwarePolicy) decodeLoadLocked(load, fleetBooked func(string) int) func(string) int {
	return func(worker string) int {
		base := 0
		if load != nil {
			base = load(worker)
		}
		booked := p.decode.booked[worker]
		if fleetBooked != nil {
			booked = fleetBooked(worker)
		}
		return saturatingNonnegativeAdd(base, booked)
	}
}

func (p *CacheAwarePolicy) decodeScoresLocked(workers []string, prefix []string, load, fleetBooked func(string) int) []DecodeFootprintScore {
	out := make([]DecodeFootprintScore, 0, len(workers))
	for _, worker := range workers {
		base := p.effectiveLoad(worker, load)
		booked := p.decode.booked[worker]
		if fleetBooked != nil {
			booked = fleetBooked(worker)
		}
		effective := saturatingNonnegativeAdd(base, booked)
		overlap := p.index.Overlap(worker, prefix)
		credit := float64(overlap)
		if p.tierEnabled {
			credit = TierWeightedOverlapCredit(p.tierWeights, p.index.TierOverlap(worker, prefix))
		}
		out = append(out, DecodeFootprintScore{
			Worker:             worker,
			Overlap:            overlap,
			BaseLoad:           base,
			BookedOutputBlocks: booked,
			EffectiveLoad:      effective,
			Score:              ScoreWithTierWeightedOverlap(credit, effective),
		})
	}
	return out
}

func selectedDecodeScore(scores []DecodeFootprintScore, worker string) float64 {
	for _, score := range scores {
		if score.Worker == worker {
			return score.Score
		}
	}
	return 0
}

func (r *decodeFootprintReservation) setIdentity(engine EngineKind, model string) {
	if r == nil || r.policy == nil || r.decision == nil {
		return
	}
	r.policy.mu.Lock()
	defer r.policy.mu.Unlock()
	if engine == EngineNative {
		r.decision.Engine = TurnIngressEngine
	} else {
		r.decision.Engine = string(engine)
	}
	if model != "" {
		r.decision.Model = model
	}
}

func (r *decodeFootprintReservation) decisionSnapshot() (DecodeFootprintRouteDecision, bool) {
	if r == nil || r.policy == nil || r.decision == nil {
		return DecodeFootprintRouteDecision{}, false
	}
	r.policy.mu.Lock()
	defer r.policy.mu.Unlock()
	return cloneDecodeFootprintDecision(*r.decision), true
}

func (r *decodeFootprintReservation) bookedBlocks() int {
	decision, ok := r.decisionSnapshot()
	if !ok {
		return 0
	}
	return decision.BookedOutputBlocks
}

func (r *decodeFootprintReservation) reconcileObserved(tokens int) {
	if r == nil || r.policy == nil || r.decision == nil {
		return
	}
	r.reconcile.Do(func() {
		if tokens < 0 {
			tokens = 0
		}
		p := r.policy
		p.mu.Lock()
		defer p.mu.Unlock()
		errorTokens := tokens - r.decision.ExpectedOutputTokens
		r.decision.ObservedOutputTokens = tokens
		r.decision.PredictionErrorTokens = errorTokens
		r.decision.Reconciled = true
		p.decode.stats.Reconciliations++
		p.decode.stats.PredictionErrorTokens += int64(errorTokens)
		if errorTokens < 0 {
			errorTokens = -errorTokens
		}
		p.decode.stats.AbsolutePredictionErrorTokens += uint64(errorTokens)
	})
}

func (r *decodeFootprintReservation) releaseOnce(reason string) {
	if r == nil || r.policy == nil || r.decision == nil {
		return
	}
	r.release.Do(func() {
		p := r.policy
		p.mu.Lock()
		defer p.mu.Unlock()
		record := r.decision
		p.removeDecodeBlocksLocked(record.Worker, record.BookedOutputBlocks)
		record.Released = true
		record.ReleaseReason = reason
		record.ReleaseCount++
		p.decode.stats.Releases++
		if reason == "cancellation" {
			p.decode.stats.CancellationReleases++
		}
		if reason == "stream_completion" {
			p.decode.stats.StreamCompletionReleases++
		}
		if p.decode.stats.ActiveBookings > 0 {
			p.decode.stats.ActiveBookings--
		}
		p.decode.stats.ActiveBookedOutputBlocks -= record.BookedOutputBlocks
		if p.decode.stats.ActiveBookedOutputBlocks < 0 {
			p.decode.stats.ActiveBookedOutputBlocks = 0
		}
	})
}

func (p *CacheAwarePolicy) removeDecodeBlocksLocked(worker string, blocks int) {
	remaining := p.decode.booked[worker] - blocks
	if remaining <= 0 {
		delete(p.decode.booked, worker)
		return
	}
	p.decode.booked[worker] = remaining
}

func cloneDecodeFootprintDecision(in DecodeFootprintRouteDecision) DecodeFootprintRouteDecision {
	in.InitialCandidates = append([]DecodeFootprintScore(nil), in.InitialCandidates...)
	in.Candidates = append([]DecodeFootprintScore(nil), in.Candidates...)
	return in
}
