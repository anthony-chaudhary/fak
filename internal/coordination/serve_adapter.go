package coordination

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// EngineFakNative is the only execution engine admitted by ServeAdapter.
const EngineFakNative = "fak_native"

const (
	maxServeCandidates = 128
	maxServeIdentity   = 256
	maxServeCapacity   = 1 << 30
	maxServeQueue      = 1 << 24
	maxServeContext    = 1 << 40
)

type ServeProvenance string

const (
	ServeMeasured  ServeProvenance = "measured"
	ServeEstimated ServeProvenance = "estimated"
)

type ServeAction string

const (
	ServeAdmit       ServeAction = "admit"
	ServeDefer       ServeAction = "defer"
	ServeReroute     ServeAction = "reroute"
	ServeResizeBatch ServeAction = "resize_batch"
	ServePrewarm     ServeAction = "prewarm"
	ServeReject      ServeAction = "reject"
)

type ServeRejection string

const (
	ServeNotRejected ServeRejection = "none"
	ServeRetryable   ServeRejection = "retryable"
	ServeTerminal    ServeRejection = "terminal"
)

type ServeApplyStatus string

const (
	ServeApplied     ServeApplyStatus = "applied"
	ServeSuperseded  ServeApplyStatus = "superseded"
	ServeUnavailable ServeApplyStatus = "unavailable"
	ServeFailed      ServeApplyStatus = "failed"
)

// ServeSourceCandidate is private, content-free observer input. Identity is
// replaced by StableID before it leaves Observe.
type ServeSourceCandidate struct {
	Identity          []byte
	Engine            string
	Generation        uint64
	CapacitySnapshot  uint64
	Ready             bool
	QueueDepth        int
	QueueCapacity     int
	BatchSize         int
	BatchCapacity     int
	PrefillAvailable  int64
	DecodeAvailable   int64
	AdmissionMillis   int64
	Backpressure      bool
	CancellationReady bool
	CacheAffinity     uint64
	Provenance        ServeProvenance
	ObservedAt        time.Time
	ValidUntil        time.Time
}

type ServeSourceSnapshot struct {
	Generation uint64
	CapturedAt time.Time
	Candidates []ServeSourceCandidate
}

type ServeStateObserver interface {
	SnapshotServe(context.Context) (ServeSourceSnapshot, error)
}

// ServeCandidate is a bounded, content-free serving observation.
type ServeCandidate struct {
	StableID          string          `json:"stableId"`
	Engine            string          `json:"engine"`
	Generation        uint64          `json:"generation"`
	CapacitySnapshot  uint64          `json:"capacitySnapshot"`
	Ready             bool            `json:"ready"`
	QueueDepth        int             `json:"queueDepth"`
	QueueCapacity     int             `json:"queueCapacity"`
	BatchSize         int             `json:"batchSize"`
	BatchCapacity     int             `json:"batchCapacity"`
	PrefillAvailable  int64           `json:"prefillAvailable"`
	DecodeAvailable   int64           `json:"decodeAvailable"`
	AdmissionMillis   int64           `json:"admissionMillis"`
	Backpressure      bool            `json:"backpressure"`
	CancellationReady bool            `json:"cancellationReady"`
	CacheAffinity     uint64          `json:"cacheAffinity"`
	Provenance        ServeProvenance `json:"provenance"`
	ObservedAt        time.Time       `json:"observedAt"`
	ValidUntil        time.Time       `json:"validUntil"`
	Fresh             bool            `json:"fresh"`
}

type ServeObservation struct {
	Generation uint64           `json:"generation"`
	ObservedAt time.Time        `json:"observedAt"`
	Candidates []ServeCandidate `json:"candidates"`
}

// ServeContextCapture describes captured context without carrying its content.
type ServeContextCapture struct {
	Captured      bool   `json:"captured"`
	Generation    uint64 `json:"generation"`
	ReusableBytes int64  `json:"reusableBytes"`
	CacheKey      string `json:"cacheKey"`
}

type ServeProjectionRequirements struct {
	PrefillTokens int64 `json:"prefillTokens"`
	DecodeTokens  int64 `json:"decodeTokens"`
	BatchSize     int   `json:"batchSize"`
}

type ServeBudgetImpact struct {
	Tokens      int64         `json:"tokens"`
	CostMicros  int64         `json:"costMicros"`
	Delay       time.Duration `json:"delay"`
	Concurrency int           `json:"concurrency"`
}

type ServePlanRequest struct {
	IdempotencyKey string                      `json:"idempotencyKey"`
	Now            time.Time                   `json:"now"`
	Deadline       time.Time                   `json:"deadline"`
	Context        ServeContextCapture         `json:"context"`
	Intent         NeutralHarnessIntent        `json:"intent"`
	Pressure       HarnessPressure             `json:"pressure"`
	Requirements   ServeProjectionRequirements `json:"requirements"`
}

// ServeComputeCandidate is the typed alternative selected by planning.
type ServeComputeCandidate struct {
	StableID         string `json:"stableId"`
	Engine           string `json:"engine"`
	Generation       uint64 `json:"generation"`
	CapacitySnapshot uint64 `json:"capacitySnapshot"`
}

type ServeDecision struct {
	PlanID           string                 `json:"planId"`
	Action           ServeAction            `json:"action"`
	Reason           string                 `json:"reason"`
	Selected         *ServeComputeCandidate `json:"selected,omitempty"`
	HarnessPressure  HarnessPressure        `json:"harnessPressure"`
	Deadline         time.Time              `json:"deadline"`
	BudgetImpact     ServeBudgetImpact      `json:"budgetImpact"`
	Rejection        ServeRejection         `json:"rejection"`
	Observation      uint64                 `json:"observationGeneration"`
	CapacitySnapshot uint64                 `json:"capacitySnapshot"`
	IdempotencyKey   string                 `json:"idempotencyKey"`
	LegacyDelegated  bool                   `json:"legacyDelegated"`
}

type LegacyServeAdmission interface {
	PlanLegacyServe(context.Context, ServeObservation, ServePlanRequest) (ServeDecision, error)
}

type ServeActionExecutor interface {
	ApplyServe(context.Context, ServeDecision) error
}

type ServeApplyResult struct {
	Status   ServeApplyStatus `json:"status"`
	Decision ServeDecision    `json:"decision"`
	Error    string           `json:"error,omitempty"`
}

type ServeAdapter struct {
	enabled  bool
	observer ServeStateObserver
	legacy   LegacyServeAdmission
	executor ServeActionExecutor
	now      func() time.Time

	mu       sync.Mutex
	applied  map[string]ServeApplyResult
	inflight map[string]*serveInFlight
}

type serveInFlight struct {
	planID string
	done   chan struct{}
}

func NewServeAdapter(enabled bool, observer ServeStateObserver, legacy LegacyServeAdmission, executor ServeActionExecutor) *ServeAdapter {
	return &ServeAdapter{
		enabled: enabled, observer: observer, legacy: legacy, executor: executor, now: time.Now,
		applied: make(map[string]ServeApplyResult), inflight: make(map[string]*serveInFlight),
	}
}

func (a *ServeAdapter) Observe(ctx context.Context) (ServeObservation, error) {
	if a == nil || a.observer == nil {
		return ServeObservation{}, errors.New("serve observer is required")
	}
	snapshot, err := a.observer.SnapshotServe(ctx)
	if err != nil {
		return ServeObservation{}, fmt.Errorf("observe serve state: %w", err)
	}
	now := a.nowTime()
	if snapshot.Generation == 0 || snapshot.CapturedAt.IsZero() || snapshot.CapturedAt.After(now) {
		return ServeObservation{}, errors.New("invalid serve snapshot identity or time")
	}
	if len(snapshot.Candidates) == 0 || len(snapshot.Candidates) > maxServeCandidates {
		return ServeObservation{}, errors.New("serve candidate set is missing or unbounded")
	}
	out := ServeObservation{Generation: snapshot.Generation, ObservedAt: snapshot.CapturedAt, Candidates: make([]ServeCandidate, 0, len(snapshot.Candidates))}
	seen := make(map[string]struct{}, len(snapshot.Candidates))
	for _, source := range snapshot.Candidates {
		candidate, err := projectServeCandidate(source, now)
		if err != nil {
			return ServeObservation{}, err
		}
		if _, ok := seen[candidate.StableID]; ok {
			return ServeObservation{}, errors.New("duplicate serve candidate identity")
		}
		seen[candidate.StableID] = struct{}{}
		out.Candidates = append(out.Candidates, candidate)
	}
	sort.Slice(out.Candidates, func(i, j int) bool { return out.Candidates[i].StableID < out.Candidates[j].StableID })
	return out, nil
}

func projectServeCandidate(source ServeSourceCandidate, now time.Time) (ServeCandidate, error) {
	if len(source.Identity) == 0 || len(source.Identity) > maxServeIdentity || source.Generation == 0 || source.CapacitySnapshot == 0 {
		return ServeCandidate{}, errors.New("malformed serve candidate identity")
	}
	if source.Engine != EngineFakNative {
		return ServeCandidate{}, errors.New("serve candidate engine must be fak_native")
	}
	if source.Provenance != ServeMeasured && source.Provenance != ServeEstimated {
		return ServeCandidate{}, errors.New("unknown serve provenance")
	}
	if source.ObservedAt.IsZero() || source.ValidUntil.IsZero() || source.ValidUntil.Before(source.ObservedAt) || source.ObservedAt.After(now) || !now.Before(source.ValidUntil) {
		return ServeCandidate{}, errors.New("stale or invalid serve candidate")
	}
	if !validServeCapacities(source) {
		return ServeCandidate{}, errors.New("invalid or unbounded serve capacity")
	}
	sum := sha256.Sum256(source.Identity)
	return ServeCandidate{
		StableID: hex.EncodeToString(sum[:16]), Engine: source.Engine, Generation: source.Generation,
		CapacitySnapshot: source.CapacitySnapshot, Ready: source.Ready, QueueDepth: source.QueueDepth,
		QueueCapacity: source.QueueCapacity, BatchSize: source.BatchSize, BatchCapacity: source.BatchCapacity,
		PrefillAvailable: source.PrefillAvailable, DecodeAvailable: source.DecodeAvailable,
		AdmissionMillis: source.AdmissionMillis, Backpressure: source.Backpressure,
		CancellationReady: source.CancellationReady, CacheAffinity: source.CacheAffinity,
		Provenance: source.Provenance, ObservedAt: source.ObservedAt, ValidUntil: source.ValidUntil, Fresh: true,
	}, nil
}

func validServeCapacities(c ServeSourceCandidate) bool {
	return c.QueueCapacity > 0 && c.QueueCapacity <= maxServeQueue && c.QueueDepth >= 0 && c.QueueDepth <= c.QueueCapacity &&
		c.BatchCapacity > 0 && c.BatchCapacity <= maxServeQueue && c.BatchSize >= 0 && c.BatchSize <= c.BatchCapacity &&
		c.PrefillAvailable >= 0 && c.PrefillAvailable <= maxServeCapacity && c.DecodeAvailable >= 0 && c.DecodeAvailable <= maxServeCapacity &&
		c.AdmissionMillis >= 0 && c.AdmissionMillis <= int64((24*time.Hour)/time.Millisecond)
}

func (a *ServeAdapter) Plan(ctx context.Context, observation ServeObservation, request ServePlanRequest) (ServeDecision, error) {
	if err := validateServePlanInput(a.nowTime(), observation, request); err != nil {
		return ServeDecision{}, err
	}
	if !a.enabled {
		if a.legacy == nil {
			return ServeDecision{}, errors.New("coordination disabled but legacy serve admission is unavailable")
		}
		decision, err := a.legacy.PlanLegacyServe(ctx, observation, request)
		if err != nil {
			return ServeDecision{}, fmt.Errorf("legacy serve admission: %w", err)
		}
		decision.LegacyDelegated = true
		if err := validateServeDecision(decision); err != nil {
			return ServeDecision{}, fmt.Errorf("invalid legacy serve decision: %w", err)
		}
		return decision, nil
	}

	pressure := request.Pressure
	impact := ServeBudgetImpact{Tokens: request.Requirements.PrefillTokens + request.Requirements.DecodeTokens, Concurrency: request.Intent.Concurrency}
	base := ServeDecision{Deadline: request.Deadline, BudgetImpact: impact, Rejection: ServeNotRejected, Observation: observation.Generation, IdempotencyKey: request.IdempotencyKey, HarnessPressure: pressure}
	if pressure.Cancelled {
		base.Action, base.Reason, base.Rejection = ServeReject, "harness cancelled", ServeTerminal
		return finishServeDecision(base), nil
	}
	if pressure.Exhausted {
		base.Action, base.Reason, base.Rejection = ServeReject, "harness budget exhausted", ServeTerminal
		return finishServeDecision(base), nil
	}

	ready := make([]ServeCandidate, 0, len(observation.Candidates))
	for _, candidate := range observation.Candidates {
		if candidate.Ready && candidate.CancellationReady && candidate.PrefillAvailable >= request.Requirements.PrefillTokens && candidate.DecodeAvailable >= request.Requirements.DecodeTokens {
			ready = append(ready, candidate)
		}
	}
	if len(ready) == 0 {
		base.Action, base.Reason, base.Rejection = ServeReject, "no ready fak_native candidate with required evidence and capacity", ServeRetryable
		return finishServeDecision(base), nil
	}
	sort.SliceStable(ready, func(i, j int) bool {
		li, lj := serveLoad(ready[i]), serveLoad(ready[j])
		if li != lj {
			return li < lj
		}
		if ready[i].AdmissionMillis != ready[j].AdmissionMillis {
			return ready[i].AdmissionMillis < ready[j].AdmissionMillis
		}
		if ready[i].CacheAffinity != ready[j].CacheAffinity {
			return ready[i].CacheAffinity > ready[j].CacheAffinity
		}
		return ready[i].StableID < ready[j].StableID
	})
	selected := ready[0]
	base.Selected = &ServeComputeCandidate{StableID: selected.StableID, Engine: selected.Engine, Generation: selected.Generation, CapacitySnapshot: selected.CapacitySnapshot}
	base.CapacitySnapshot = selected.CapacitySnapshot
	base.BudgetImpact.Delay = time.Duration(selected.AdmissionMillis) * time.Millisecond

	queuePressure := selected.Backpressure || selected.QueueDepth*4 >= selected.QueueCapacity*3
	batchPressure := request.Requirements.BatchSize > selected.BatchCapacity-selected.BatchSize
	switch {
	case request.Deadline.Before(request.Now.Add(base.BudgetImpact.Delay)):
		base.Action, base.Reason, base.Rejection = ServeReject, "admission estimate exceeds deadline", ServeRetryable
	case queuePressure:
		base.Action, base.Reason = ServeDefer, "queue backpressure requires delayed fan-out"
		base.HarnessPressure.Concurrency = maxInt(1, request.Pressure.Concurrency/2)
	case batchPressure:
		base.Action, base.Reason = ServeResizeBatch, "batch capacity requires reduced projection"
		base.HarnessPressure.Concurrency = maxInt(1, selected.BatchCapacity-selected.BatchSize)
	case request.Context.Captured && request.Context.ReusableBytes > 0 && selected.CacheAffinity == 0:
		base.Action, base.Reason = ServePrewarm, "captured context requires cache prewarm"
	case selected.StableID != observation.Candidates[0].StableID:
		base.Action, base.Reason = ServeReroute, "selected alternative compute candidate"
	default:
		base.Action, base.Reason = ServeAdmit, "ready capacity satisfies neutral harness projection"
	}
	return finishServeDecision(base), nil
}

func serveLoad(c ServeCandidate) int64 {
	return int64(c.QueueDepth)*1_000_000/int64(c.QueueCapacity) + int64(c.BatchSize)*1_000_000/int64(c.BatchCapacity)
}

func validateServePlanInput(now time.Time, observation ServeObservation, request ServePlanRequest) error {
	if observation.Generation == 0 || observation.ObservedAt.IsZero() || len(observation.Candidates) == 0 || len(observation.Candidates) > maxServeCandidates {
		return errors.New("missing or invalid serve observation")
	}
	if strings.TrimSpace(request.IdempotencyKey) == "" || len(request.IdempotencyKey) > maxServeIdentity || request.Now.IsZero() || request.Deadline.IsZero() || request.Deadline.Before(request.Now) {
		return errors.New("malformed serve request identity or deadline")
	}
	if request.Now.After(now.Add(time.Second)) || now.Before(observation.ObservedAt) {
		return errors.New("invalid serve planning time")
	}
	if request.Requirements.PrefillTokens < 0 || request.Requirements.PrefillTokens > maxServeCapacity || request.Requirements.DecodeTokens < 0 || request.Requirements.DecodeTokens > maxServeCapacity || request.Requirements.BatchSize <= 0 || request.Requirements.BatchSize > maxServeQueue {
		return errors.New("invalid or unbounded serve requirements")
	}
	if request.Context.ReusableBytes < 0 || request.Context.ReusableBytes > maxServeContext || (request.Context.Captured && (request.Context.Generation == 0 || strings.TrimSpace(request.Context.CacheKey) == "")) {
		return errors.New("missing or invalid captured context evidence")
	}
	if strings.TrimSpace(request.Intent.WorkID) == "" || strings.TrimSpace(request.Intent.CorrelationID) == "" || request.Intent.Fanout < 0 || request.Intent.Concurrency < 0 {
		return errors.New("malformed neutral harness intent")
	}
	for _, candidate := range observation.Candidates {
		if candidate.Engine != EngineFakNative || !candidate.Fresh || !request.Now.Before(candidate.ValidUntil) || candidate.Generation == 0 || candidate.CapacitySnapshot == 0 || candidate.StableID == "" {
			return errors.New("stale, non-fak-native, or incomplete serve evidence")
		}
	}
	return nil
}

func finishServeDecision(decision ServeDecision) ServeDecision {
	wire, _ := json.Marshal(struct {
		Action      ServeAction            `json:"action"`
		Reason      string                 `json:"reason"`
		Selected    *ServeComputeCandidate `json:"selected"`
		Observation uint64                 `json:"observation"`
		Deadline    int64                  `json:"deadline"`
		Key         string                 `json:"key"`
	}{decision.Action, decision.Reason, decision.Selected, decision.Observation, decision.Deadline.UnixNano(), decision.IdempotencyKey})
	sum := sha256.Sum256(wire)
	decision.PlanID = hex.EncodeToString(sum[:16])
	return decision
}

func validateServeDecision(d ServeDecision) error {
	switch d.Action {
	case ServeAdmit, ServeDefer, ServeReroute, ServeResizeBatch, ServePrewarm, ServeReject:
	default:
		return errors.New("unknown serve action")
	}
	if d.PlanID == "" || d.IdempotencyKey == "" || d.Observation == 0 || d.Deadline.IsZero() {
		return errors.New("serve decision lacks required evidence")
	}
	if d.Action == ServeReject {
		if d.Rejection != ServeRetryable && d.Rejection != ServeTerminal {
			return errors.New("reject decision must be retryable or terminal")
		}
	} else if d.Rejection != ServeNotRejected {
		return errors.New("non-reject decision has rejection state")
	}
	if d.Selected != nil && (d.Selected.Engine != EngineFakNative || d.Selected.StableID == "" || d.Selected.Generation == 0 || d.Selected.CapacitySnapshot == 0) {
		return errors.New("invalid selected compute candidate")
	}
	return nil
}

func (a *ServeAdapter) Apply(ctx context.Context, decision ServeDecision) ServeApplyResult {
	if err := validateServeDecision(decision); err != nil {
		return ServeApplyResult{Status: ServeFailed, Decision: decision, Error: err.Error()}
	}

	a.mu.Lock()
	if prior, ok := a.applied[decision.IdempotencyKey]; ok {
		a.mu.Unlock()
		if prior.Decision.PlanID == decision.PlanID {
			return prior
		}
		return ServeApplyResult{Status: ServeSuperseded, Decision: decision, Error: "idempotency key already applied to another plan"}
	}
	if flight, ok := a.inflight[decision.IdempotencyKey]; ok {
		if flight.planID != decision.PlanID {
			a.mu.Unlock()
			return ServeApplyResult{Status: ServeSuperseded, Decision: decision, Error: "idempotency key is executing another plan"}
		}
		done := flight.done
		a.mu.Unlock()
		select {
		case <-done:
			a.mu.Lock()
			result := a.applied[decision.IdempotencyKey]
			a.mu.Unlock()
			return result
		case <-ctx.Done():
			return ServeApplyResult{Status: ServeFailed, Decision: decision, Error: ctx.Err().Error()}
		}
	}
	flight := &serveInFlight{planID: decision.PlanID, done: make(chan struct{})}
	a.inflight[decision.IdempotencyKey] = flight
	a.mu.Unlock()

	finish := func(result ServeApplyResult) ServeApplyResult {
		a.mu.Lock()
		a.applied[decision.IdempotencyKey] = result
		delete(a.inflight, decision.IdempotencyKey)
		close(flight.done)
		a.mu.Unlock()
		return result
	}
	if decision.Selected != nil {
		observation, err := a.Observe(ctx)
		if err != nil {
			return finish(ServeApplyResult{Status: ServeUnavailable, Decision: decision, Error: err.Error()})
		}
		if observation.Generation != decision.Observation {
			return finish(ServeApplyResult{Status: ServeSuperseded, Decision: decision, Error: "observation generation changed"})
		}
		found := false
		for _, candidate := range observation.Candidates {
			if candidate.StableID == decision.Selected.StableID {
				found = candidate.Ready && candidate.Generation == decision.Selected.Generation && candidate.CapacitySnapshot == decision.CapacitySnapshot
				break
			}
		}
		if !found {
			return finish(ServeApplyResult{Status: ServeUnavailable, Decision: decision, Error: "selected capacity snapshot is no longer available"})
		}
	}
	if a.executor != nil {
		if err := a.executor.ApplyServe(ctx, decision); err != nil {
			return finish(ServeApplyResult{Status: ServeFailed, Decision: decision, Error: err.Error()})
		}
	}
	return finish(ServeApplyResult{Status: ServeApplied, Decision: decision})
}

type ServeAdapterSelfCheck struct {
	Passed bool        `json:"passed"`
	Action ServeAction `json:"action"`
	Digest string      `json:"digest"`
	Error  string      `json:"error,omitempty"`
}

// SelfCheck is deterministic, content-free, and exercises context, placement,
// harness pressure, and the resulting serve action.
func (a *ServeAdapter) SelfCheck() ServeAdapterSelfCheck {
	t := time.Unix(1_700_000_000, 0).UTC()
	obs := ServeObservation{Generation: 7, ObservedAt: t, Candidates: []ServeCandidate{{StableID: "selfcheck", Engine: EngineFakNative, Generation: 3, CapacitySnapshot: 11, Ready: true, QueueDepth: 8, QueueCapacity: 10, BatchSize: 1, BatchCapacity: 8, PrefillAvailable: 4096, DecodeAvailable: 1024, AdmissionMillis: 25, Backpressure: true, CancellationReady: true, CacheAffinity: 9, Provenance: ServeMeasured, ObservedAt: t, ValidUntil: t.Add(time.Minute), Fresh: true}}}
	req := ServePlanRequest{IdempotencyKey: "serve-selfcheck", Now: t, Deadline: t.Add(time.Second), Context: ServeContextCapture{Captured: true, Generation: 2, ReusableBytes: 512, CacheKey: "captured"}, Intent: NeutralHarnessIntent{WorkID: "selfcheck", CorrelationID: "selfcheck", Fanout: 4, Concurrency: 4}, Pressure: HarnessPressure{Concurrency: 4}, Requirements: ServeProjectionRequirements{PrefillTokens: 128, DecodeTokens: 32, BatchSize: 1}}
	clone := NewServeAdapter(true, nil, nil, nil)
	clone.now = func() time.Time { return t }
	decision, err := clone.Plan(context.Background(), obs, req)
	if err != nil {
		return ServeAdapterSelfCheck{Error: err.Error()}
	}
	wire, _ := json.Marshal(decision)
	sum := sha256.Sum256(wire)
	return ServeAdapterSelfCheck{Passed: decision.Action == ServeDefer && decision.HarnessPressure.Concurrency == 2, Action: decision.Action, Digest: hex.EncodeToString(sum[:])}
}

func (a *ServeAdapter) nowTime() time.Time {
	if a != nil && a.now != nil {
		return a.now().UTC()
	}
	return time.Now().UTC()
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
