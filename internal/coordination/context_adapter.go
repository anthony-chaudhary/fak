package coordination

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

type ContextResourceKind string

const (
	ContextResourcePrefix ContextResourceKind = "prefix"
	ContextResourceKV     ContextResourceKind = "kv"
	ContextResourceObject ContextResourceKind = "object"
)

type ContextLocationKind string

const (
	ContextLocationLocal       ContextLocationKind = "local"
	ContextLocationRemote      ContextLocationKind = "remote"
	ContextLocationObjectStore ContextLocationKind = "object_store"
)

type ContextCompactionState string

const (
	ContextCompactionNone        ContextCompactionState = "none"
	ContextCompactionRecommended ContextCompactionState = "recommended"
	ContextCompactionRunning     ContextCompactionState = "running"
	ContextCompactionComplete    ContextCompactionState = "complete"
)

// ContextCostEstimate keeps both cost and measurement freshness. Fresh is
// always derived by Observe rather than trusted from the source.
type ContextCostEstimate struct {
	Duration    time.Duration `json:"duration"`
	Bytes       int64         `json:"bytes"`
	Uncertainty float64       `json:"uncertainty"`
	MeasuredAt  time.Time     `json:"measuredAt"`
	FreshUntil  time.Time     `json:"freshUntil"`
	Fresh       bool          `json:"fresh"`
}

// ContextSourceResidency is private input metadata. Identity fields may
// contain tenant keys; they are hashed and never copied to public output.
type ContextSourceResidency struct {
	Identity              []byte
	Kind                  ContextResourceKind
	Generation            uint64
	CurrentGeneration     uint64
	Bytes                 int64
	Tokens                int64
	LocationKind          ContextLocationKind
	LocationIdentity      []byte
	ResidentSince         time.Time
	FreshUntil            time.Time
	Warm                  bool
	EstimatedReuseHorizon time.Duration
	TransferCost          ContextCostEstimate
	RehydrationCost       ContextCostEstimate
	Compaction            ContextCompactionState
}

type ContextSourceSnapshot struct {
	Generation  uint64
	CapturedAt  time.Time
	Managed     bool
	Pressure    float64
	Residencies []ContextSourceResidency
}

// ContextStateObserver is deliberately read-only. Cache storage remains owned
// by the existing context/cache subsystem.
type ContextStateObserver interface {
	SnapshotContext(context.Context) (ContextSourceSnapshot, error)
}

type ContextLocation struct {
	Kind     ContextLocationKind `json:"kind"`
	StableID string              `json:"stableId"`
}

type ContextResidency struct {
	StableID              string                 `json:"stableId"`
	Kind                  ContextResourceKind    `json:"kind"`
	Generation            uint64                 `json:"generation"`
	Bytes                 int64                  `json:"bytes"`
	Tokens                int64                  `json:"tokens"`
	Location              ContextLocation        `json:"location"`
	Age                   time.Duration          `json:"age"`
	FreshFor              time.Duration          `json:"freshFor"`
	Current               bool                   `json:"current"`
	Warm                  bool                   `json:"warm"`
	EstimatedReuseHorizon time.Duration          `json:"estimatedReuseHorizon"`
	TransferCost          ContextCostEstimate    `json:"transferCost"`
	RehydrationCost       ContextCostEstimate    `json:"rehydrationCost"`
	Compaction            ContextCompactionState `json:"compaction"`
}

type ContextObservation struct {
	ID          string             `json:"id"`
	Generation  uint64             `json:"generation"`
	CapturedAt  time.Time          `json:"capturedAt"`
	Age         time.Duration      `json:"age"`
	Managed     bool               `json:"managed"`
	Pressure    float64            `json:"pressure"`
	Residencies []ContextResidency `json:"residencies"`
}

var (
	ErrContextObservation  = errors.New("context observation failed")
	ErrInvalidContextState = errors.New("invalid context state")
)

type ContextActionKind string

const (
	ContextActionPin      ContextActionKind = "pin"
	ContextActionPrefetch ContextActionKind = "prefetch"
	ContextActionTransfer ContextActionKind = "transfer"
	ContextActionCompact  ContextActionKind = "compact"
	ContextActionEvict    ContextActionKind = "evict"
	ContextActionNoOp     ContextActionKind = "no_op"
)

type ContextAction struct {
	ID            string            `json:"id"`
	Kind          ContextActionKind `json:"kind"`
	ResourceID    string            `json:"resourceId"`
	Generation    uint64            `json:"generation"`
	Bytes         int64             `json:"bytes"`
	Tokens        int64             `json:"tokens"`
	Source        ContextLocation   `json:"source"`
	Destination   ContextLocation   `json:"destination"`
	ObservationID string            `json:"observationId"`
	ValidUntil    time.Time         `json:"validUntil"`
}

type ContextPlanRequest struct {
	ResourceKind ContextResourceKind `json:"resourceKind"`
	Target       ContextLocation     `json:"target"`
	ReuseHorizon time.Duration       `json:"reuseHorizon"`
}

type ContextPlanReference struct {
	ObservationID string          `json:"observationId"`
	ResourceID    string          `json:"resourceId"`
	Generation    uint64          `json:"generation"`
	Location      ContextLocation `json:"location"`
	ValidUntil    time.Time       `json:"validUntil"`
}

// ContextPlan binds Build's decision to the exact observation generation.
type ContextPlan struct {
	ID               string               `json:"id"`
	Reference        ContextPlanReference `json:"reference"`
	ProjectedContext ContextState         `json:"projectedContext"`
	Coordination     Plan                 `json:"coordination"`
	Actions          []ContextAction      `json:"actions"`
	UsesFallback     bool                 `json:"usesFallback"`
}

type ContextActionStatus string

const (
	ContextActionApplied ContextActionStatus = "applied"
	ContextActionPartial ContextActionStatus = "partial"
	ContextActionFailed  ContextActionStatus = "failed"
	ContextActionSkipped ContextActionStatus = "skipped"
)

type ContextActionFailure string

const (
	ContextFailureNone        ContextActionFailure = ""
	ContextFailureInvalid     ContextActionFailure = "invalid_action"
	ContextFailureStale       ContextActionFailure = "stale_observation"
	ContextFailureUnavailable ContextActionFailure = "unavailable"
	ContextFailureRejected    ContextActionFailure = "rejected"
	ContextFailureInternal    ContextActionFailure = "internal"
)

type ContextActionResult struct {
	ActionID     string               `json:"actionId"`
	Status       ContextActionStatus  `json:"status"`
	Failure      ContextActionFailure `json:"failure,omitempty"`
	AppliedBytes int64                `json:"appliedBytes"`
	Replayed     bool                 `json:"replayed"`
}

// ContextActionApplier owns actual cache operations. The stable action ID is
// also suitable as a durable idempotency key for the underlying owner.
type ContextActionApplier interface {
	ApplyContextAction(context.Context, ContextAction) ContextActionResult
}

type ContextApplyStatus string

const (
	ContextApplyApplied  ContextApplyStatus = "applied"
	ContextApplyPartial  ContextApplyStatus = "partial"
	ContextApplyFailed   ContextApplyStatus = "failed"
	ContextApplyNoOp     ContextApplyStatus = "no_op"
	ContextApplyFallback ContextApplyStatus = "fallback"
)

type ContextApplyResult struct {
	PlanID       string                `json:"planId"`
	Status       ContextApplyStatus    `json:"status"`
	Outcomes     []ContextActionResult `json:"outcomes"`
	UsedFallback bool                  `json:"usedFallback"`
}

// ContextPolicyFallback delegates to the existing policy when coordination is
// disabled instead of duplicating that policy in the adapter.
type ContextPolicyFallback interface {
	ApplyExistingContextPolicy(context.Context, ContextPlan) ContextApplyResult
}

type ContextAdapterOptions struct {
	Disabled              bool
	MaximumObservationAge time.Duration
	Now                   func() time.Time
}

type ContextAdapter struct {
	observer ContextStateObserver
	applier  ContextActionApplier
	fallback ContextPolicyFallback
	disabled bool
	maxAge   time.Duration
	now      func() time.Time

	applyMu sync.Mutex
	applied map[string]ContextActionResult
	planMu  sync.Mutex
	plans   map[string]string
}

func NewContextAdapter(observer ContextStateObserver, applier ContextActionApplier, fallback ContextPolicyFallback, options ContextAdapterOptions) *ContextAdapter {
	maxAge := options.MaximumObservationAge
	if maxAge <= 0 {
		maxAge = time.Minute
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &ContextAdapter{observer: observer, applier: applier, fallback: fallback,
		disabled: options.Disabled, maxAge: maxAge, now: now, applied: make(map[string]ContextActionResult),
		plans: make(map[string]string)}
}

// Observe reports metadata only. Source errors are collapsed so error text
// cannot leak a prompt or private tenant key into a trace.
func (a *ContextAdapter) Observe(ctx context.Context) (ContextObservation, error) {
	if a == nil || a.observer == nil {
		return ContextObservation{}, ErrContextObservation
	}
	snapshot, err := a.observer.SnapshotContext(ctx)
	if err != nil {
		return ContextObservation{}, ErrContextObservation
	}
	now := a.now()
	if snapshot.CapturedAt.IsZero() || snapshot.CapturedAt.After(now) || !validPressure(snapshot.Pressure) {
		return ContextObservation{}, ErrInvalidContextState
	}
	observationCurrent := now.Sub(snapshot.CapturedAt) <= a.maxAge
	residencies := make([]ContextResidency, 0, len(snapshot.Residencies))
	for _, source := range snapshot.Residencies {
		if !validSourceResidency(source, snapshot.CapturedAt) {
			return ContextObservation{}, ErrInvalidContextState
		}
		freshFor := source.FreshUntil.Sub(now)
		if freshFor < 0 {
			freshFor = 0
		}
		residencies = append(residencies, ContextResidency{
			StableID:              stableCoordinationID("ctx", source.Identity),
			Kind:                  source.Kind,
			Generation:            source.Generation,
			Bytes:                 source.Bytes,
			Tokens:                source.Tokens,
			Location:              ContextLocation{Kind: source.LocationKind, StableID: stableCoordinationID("loc", source.LocationIdentity)},
			Age:                   now.Sub(source.ResidentSince),
			FreshFor:              freshFor,
			Current:               observationCurrent && source.Generation == source.CurrentGeneration && now.Before(source.FreshUntil),
			Warm:                  source.Warm,
			EstimatedReuseHorizon: source.EstimatedReuseHorizon,
			TransferCost:          deriveCostFreshness(source.TransferCost, now),
			RehydrationCost:       deriveCostFreshness(source.RehydrationCost, now),
			Compaction:            source.Compaction,
		})
	}
	sort.Slice(residencies, func(i, j int) bool {
		if residencies[i].StableID != residencies[j].StableID {
			return residencies[i].StableID < residencies[j].StableID
		}
		if residencies[i].Location.Kind != residencies[j].Location.Kind {
			return residencies[i].Location.Kind < residencies[j].Location.Kind
		}
		return residencies[i].Location.StableID < residencies[j].Location.StableID
	})
	observation := ContextObservation{Generation: snapshot.Generation, CapturedAt: snapshot.CapturedAt,
		Age: now.Sub(snapshot.CapturedAt), Managed: snapshot.Managed, Pressure: snapshot.Pressure, Residencies: residencies}
	observation.ID = makeObservationID(observation)
	return observation, nil
}

func validSourceResidency(source ContextSourceResidency, capturedAt time.Time) bool {
	return len(source.Identity) != 0 && len(source.LocationIdentity) != 0 && source.Bytes >= 0 && source.Tokens >= 0 &&
		source.EstimatedReuseHorizon >= 0 && !source.ResidentSince.IsZero() && !source.ResidentSince.After(capturedAt) &&
		!source.FreshUntil.IsZero() && validContextResourceKind(source.Kind) && validContextLocationKind(source.LocationKind) &&
		validContextCompaction(source.Compaction) && validContextCost(source.TransferCost) && validContextCost(source.RehydrationCost)
}

func validContextCost(cost ContextCostEstimate) bool {
	return cost.Duration >= 0 && cost.Bytes >= 0 && !math.IsNaN(cost.Uncertainty) && cost.Uncertainty >= 0 &&
		cost.Uncertainty <= 1 && !cost.MeasuredAt.IsZero() && !cost.FreshUntil.IsZero()
}

func deriveCostFreshness(cost ContextCostEstimate, now time.Time) ContextCostEstimate {
	cost.Fresh = !now.Before(cost.MeasuredAt) && now.Before(cost.FreshUntil)
	return cost
}

func validContextResourceKind(kind ContextResourceKind) bool {
	return kind == ContextResourcePrefix || kind == ContextResourceKV || kind == ContextResourceObject
}

func validContextLocationKind(kind ContextLocationKind) bool {
	return kind == ContextLocationLocal || kind == ContextLocationRemote || kind == ContextLocationObjectStore
}

func validContextCompaction(state ContextCompactionState) bool {
	return state == ContextCompactionNone || state == ContextCompactionRecommended || state == ContextCompactionRunning || state == ContextCompactionComplete
}

// Plan selects the lowest readiness-cost placement. Remote placements are
// considered only when their cost fits inside the estimated reuse horizon.
func (a *ContextAdapter) Plan(input Input, observation ContextObservation, request ContextPlanRequest) ContextPlan {
	if a == nil || a.disabled {
		plan := ContextPlan{Reference: ContextPlanReference{ObservationID: observation.ID},
			ProjectedContext: input.ContextState, Coordination: Build(input), UsesFallback: true}
		plan.ID = makeFallbackContextPlanID(plan.Reference)
		if a != nil {
			a.rememberContextPlan(plan)
		}
		return plan
	}
	if !validContextResourceKind(request.ResourceKind) {
		request.ResourceKind = ContextResourcePrefix
	}
	selected, found := selectContextResidency(observation, request)
	projected := input
	projected.ContextState = ContextState{Managed: observation.Managed, Pressure: observation.Pressure}
	if found && selected.Warm && (selected.Kind == ContextResourcePrefix || selected.Kind == ContextResourceKV) {
		projected.ContextState.ReusablePrefixBytes = boundedContextInt(selected.Bytes)
	}
	reference := ContextPlanReference{ObservationID: observation.ID}
	action := ContextAction{Kind: ContextActionNoOp, ObservationID: observation.ID}
	if found {
		reference = ContextPlanReference{ObservationID: observation.ID, ResourceID: selected.StableID,
			Generation: selected.Generation, Location: selected.Location,
			ValidUntil: observation.CapturedAt.Add(observation.Age + selected.FreshFor)}
		action = contextActionFor(observation, selected, request.Target, reference.ValidUntil)
	}
	action.ID = makeContextActionID(observation.ID, action)
	plan := ContextPlan{Reference: reference, ProjectedContext: projected.ContextState,
		Coordination: Build(projected), Actions: []ContextAction{action}}
	plan.ID = makeContextPlanID(reference, action)
	a.rememberContextPlan(plan)
	return plan
}

func selectContextResidency(observation ContextObservation, request ContextPlanRequest) (ContextResidency, bool) {
	bestCost := time.Duration(math.MaxInt64)
	var best ContextResidency
	found := false
	for _, residency := range observation.Residencies {
		if !residency.Current || residency.Kind != request.ResourceKind {
			continue
		}
		local := residency.Location == request.Target
		cost := contextReadinessCost(residency, local)
		horizon := request.ReuseHorizon
		if horizon <= 0 {
			horizon = residency.EstimatedReuseHorizon
		}
		if !local && (horizon <= 0 || cost >= horizon) {
			continue
		}
		if !found || cost < bestCost || (cost == bestCost && residency.Location.StableID < best.Location.StableID) {
			best, bestCost, found = residency, cost, true
		}
	}
	return best, found
}

func contextReadinessCost(residency ContextResidency, local bool) time.Duration {
	if local && residency.Warm {
		return 0
	}
	cost := time.Duration(0)
	if !local {
		if !residency.TransferCost.Fresh {
			return time.Duration(math.MaxInt64)
		}
		cost = uncertainDuration(residency.TransferCost)
	}
	if !residency.Warm {
		if !residency.RehydrationCost.Fresh {
			return time.Duration(math.MaxInt64)
		}
		rehydrate := uncertainDuration(residency.RehydrationCost)
		if rehydrate > time.Duration(math.MaxInt64)-cost {
			return time.Duration(math.MaxInt64)
		}
		cost += rehydrate
	}
	return cost
}

func uncertainDuration(cost ContextCostEstimate) time.Duration {
	value := float64(cost.Duration) * (1 + cost.Uncertainty)
	if value >= float64(math.MaxInt64) {
		return time.Duration(math.MaxInt64)
	}
	return time.Duration(value)
}

func contextActionFor(observation ContextObservation, residency ContextResidency, target ContextLocation, validUntil time.Time) ContextAction {
	kind := ContextActionPin
	if residency.Location != target {
		kind = ContextActionTransfer
	} else if !residency.Warm {
		kind = ContextActionPrefetch
	}
	return ContextAction{Kind: kind, ResourceID: residency.StableID, Generation: residency.Generation,
		Bytes: residency.Bytes, Tokens: residency.Tokens, Source: residency.Location, Destination: target,
		ObservationID: observation.ID, ValidUntil: validUntil}
}

// Apply accepts only the closed typed action vocabulary and memoizes outcomes
// by action ID, so repeated or concurrent application cannot repeat an effect.
func (a *ContextAdapter) Apply(ctx context.Context, plan ContextPlan) ContextApplyResult {
	if a == nil {
		return unavailableContextApply(plan.ID)
	}
	if !a.validProducedContextPlan(plan) {
		return invalidContextApply(plan.ID)
	}
	if a.disabled {
		if a.fallback == nil {
			return unavailableContextApply(plan.ID)
		}
		result := a.fallback.ApplyExistingContextPolicy(ctx, plan)
		result.PlanID, result.Status, result.UsedFallback = plan.ID, ContextApplyFallback, true
		return result
	}
	result := ContextApplyResult{PlanID: plan.ID}
	for _, action := range plan.Actions {
		result.Outcomes = append(result.Outcomes, a.applyContextAction(ctx, plan, action))
	}
	result.Status = aggregateContextApply(result.Outcomes)
	return result
}

func (a *ContextAdapter) applyContextAction(ctx context.Context, plan ContextPlan, action ContextAction) ContextActionResult {
	a.applyMu.Lock()
	defer a.applyMu.Unlock()
	if prior, ok := a.applied[action.ID]; ok {
		prior.Replayed = true
		return prior
	}
	result := ContextActionResult{ActionID: action.ID}
	switch {
	case !validBoundContextAction(action):
		result.Status, result.Failure = ContextActionFailed, ContextFailureInvalid
	case action.Kind == ContextActionNoOp:
		result.Status = ContextActionSkipped
	case action.ObservationID != plan.Reference.ObservationID || action.ResourceID != plan.Reference.ResourceID || action.Generation != plan.Reference.Generation:
		result.Status, result.Failure = ContextActionFailed, ContextFailureInvalid
	case action.ValidUntil.IsZero() || !a.now().Before(action.ValidUntil):
		result.Status, result.Failure = ContextActionFailed, ContextFailureStale
	case a.applier == nil:
		result.Status, result.Failure = ContextActionFailed, ContextFailureUnavailable
	default:
		result = a.applier.ApplyContextAction(ctx, action)
		result.ActionID = action.ID
		if !validContextActionResult(result, action.Bytes) {
			result = ContextActionResult{ActionID: action.ID, Status: ContextActionFailed, Failure: ContextFailureInternal}
		}
	}
	a.applied[action.ID] = result
	return result
}

func validBoundContextAction(action ContextAction) bool {
	if action.ID == "" || !validContextActionKind(action.Kind) || action.Bytes < 0 || action.Tokens < 0 ||
		action.ObservationID == "" || action.ID != makeContextActionID(action.ObservationID, action) {
		return false
	}
	if action.Kind == ContextActionNoOp {
		return action.ResourceID == "" && action.Generation == 0 && action.Source == (ContextLocation{}) &&
			action.Destination == (ContextLocation{}) && action.Bytes == 0 && action.Tokens == 0 && action.ValidUntil.IsZero()
	}
	return validStableCoordinationID(action.ResourceID, "ctx") && action.Generation > 0 &&
		validBoundContextLocation(action.Source) && validBoundContextLocation(action.Destination) && !action.ValidUntil.IsZero()
}

func validBoundContextLocation(location ContextLocation) bool {
	return validContextLocationKind(location.Kind) && validStableCoordinationID(location.StableID, "loc")
}

func validContextActionKind(kind ContextActionKind) bool {
	switch kind {
	case ContextActionPin, ContextActionPrefetch, ContextActionTransfer, ContextActionCompact, ContextActionEvict, ContextActionNoOp:
		return true
	default:
		return false
	}
}

func validContextActionResult(result ContextActionResult, total int64) bool {
	if result.AppliedBytes < 0 || result.AppliedBytes > total {
		return false
	}
	switch result.Status {
	case ContextActionApplied:
		return result.Failure == ContextFailureNone
	case ContextActionPartial, ContextActionFailed:
		return result.Failure != ContextFailureNone
	default:
		return false
	}
}

func aggregateContextApply(outcomes []ContextActionResult) ContextApplyStatus {
	if len(outcomes) == 0 {
		return ContextApplyNoOp
	}
	applied, partial, failed := false, false, false
	for _, result := range outcomes {
		switch result.Status {
		case ContextActionApplied:
			applied = true
		case ContextActionPartial:
			partial = true
		case ContextActionFailed:
			failed = true
		case ContextActionSkipped:
			// A planned no-op is neutral; it must not turn an otherwise
			// successful application into a partial or failed result.
		}
	}
	if partial || (applied && failed) {
		return ContextApplyPartial
	}
	if failed {
		return ContextApplyFailed
	}
	if applied {
		return ContextApplyApplied
	}
	return ContextApplyNoOp
}

func unavailableContextApply(planID string) ContextApplyResult {
	return ContextApplyResult{PlanID: planID, Status: ContextApplyFailed,
		Outcomes: []ContextActionResult{{Status: ContextActionFailed, Failure: ContextFailureUnavailable}}}
}

func invalidContextApply(planID string) ContextApplyResult {
	return ContextApplyResult{PlanID: planID, Status: ContextApplyFailed,
		Outcomes: []ContextActionResult{{Status: ContextActionFailed, Failure: ContextFailureInvalid}}}
}

func (a *ContextAdapter) rememberContextPlan(plan ContextPlan) {
	a.planMu.Lock()
	defer a.planMu.Unlock()
	a.plans[plan.ID] = contextPlanFingerprint(plan)
}

// validProducedContextPlan requires both a deterministic structural binding
// and a match in this adapter's plan registry. IDs are integrity references,
// not permission for callers to manufacture new cache operations.
func (a *ContextAdapter) validProducedContextPlan(plan ContextPlan) bool {
	if plan.ID == "" {
		return false
	}
	if plan.UsesFallback {
		if len(plan.Actions) != 0 || plan.ID != makeFallbackContextPlanID(plan.Reference) {
			return false
		}
	} else {
		if len(plan.Actions) != 1 || !validBoundContextAction(plan.Actions[0]) ||
			plan.ID != makeContextPlanID(plan.Reference, plan.Actions[0]) {
			return false
		}
		action := plan.Actions[0]
		if action.ObservationID != plan.Reference.ObservationID {
			return false
		}
		if action.Kind == ContextActionNoOp {
			if plan.Reference.ResourceID != "" || plan.Reference.Generation != 0 ||
				plan.Reference.Location != (ContextLocation{}) || !plan.Reference.ValidUntil.IsZero() {
				return false
			}
		} else if action.ResourceID != plan.Reference.ResourceID || action.Generation != plan.Reference.Generation ||
			action.Source != plan.Reference.Location || action.ValidUntil != plan.Reference.ValidUntil ||
			!validStableCoordinationID(plan.Reference.ResourceID, "ctx") || plan.Reference.Generation == 0 ||
			!validBoundContextLocation(plan.Reference.Location) {
			return false
		}
	}
	a.planMu.Lock()
	defer a.planMu.Unlock()
	want, ok := a.plans[plan.ID]
	return ok && want == contextPlanFingerprint(plan)
}

func contextPlanFingerprint(plan ContextPlan) string {
	copy := plan
	copy.ID = ""
	encoded, err := json.Marshal(copy)
	if err != nil {
		return ""
	}
	return stableCoordinationID("planfp", encoded)
}

type ContextTracePhase string

const (
	ContextTraceObserve ContextTracePhase = "observe"
	ContextTracePlan    ContextTracePhase = "plan"
	ContextTraceApply   ContextTracePhase = "apply"
)

// ContextTraceEvent has no free-form string supplied by a cache or request.
type ContextTraceEvent struct {
	Phase         ContextTracePhase    `json:"phase"`
	ObservationID string               `json:"observationId,omitempty"`
	PlanID        string               `json:"planId,omitempty"`
	ResourceID    string               `json:"resourceId,omitempty"`
	ActionID      string               `json:"actionId,omitempty"`
	Action        ContextActionKind    `json:"action,omitempty"`
	ApplyStatus   ContextApplyStatus   `json:"applyStatus,omitempty"`
	Failure       ContextActionFailure `json:"failure,omitempty"`
	Bytes         int64                `json:"bytes,omitempty"`
	Tokens        int64                `json:"tokens,omitempty"`
}

type ContextAdapterSelfCheck struct {
	Observation ContextObservation   `json:"observation"`
	Plan        ContextPlan          `json:"plan"`
	Apply       ContextApplyResult   `json:"apply"`
	Trace       []ContextTraceEvent  `json:"trace"`
	Failure     ContextActionFailure `json:"failure,omitempty"`
}

// SelfCheck captures observe -> plan reference -> apply without request text.
func (a *ContextAdapter) SelfCheck(ctx context.Context, input Input, request ContextPlanRequest) ContextAdapterSelfCheck {
	observation, err := a.Observe(ctx)
	if err != nil {
		return ContextAdapterSelfCheck{Failure: ContextFailureUnavailable,
			Trace: []ContextTraceEvent{{Phase: ContextTraceObserve, Failure: ContextFailureUnavailable}}}
	}
	plan := a.Plan(input, observation, request)
	apply := a.Apply(ctx, plan)
	trace := []ContextTraceEvent{
		{Phase: ContextTraceObserve, ObservationID: observation.ID, ResourceID: plan.Reference.ResourceID},
		{Phase: ContextTracePlan, ObservationID: observation.ID, PlanID: plan.ID, ResourceID: plan.Reference.ResourceID},
	}
	for i, outcome := range apply.Outcomes {
		event := ContextTraceEvent{Phase: ContextTraceApply, ObservationID: observation.ID, PlanID: plan.ID,
			ResourceID: plan.Reference.ResourceID, ActionID: outcome.ActionID, ApplyStatus: apply.Status, Failure: outcome.Failure}
		if i < len(plan.Actions) {
			event.Action, event.Bytes, event.Tokens = plan.Actions[i].Kind, plan.Actions[i].Bytes, plan.Actions[i].Tokens
		}
		trace = append(trace, event)
	}
	return ContextAdapterSelfCheck{Observation: observation, Plan: plan, Apply: apply, Trace: trace}
}

func stableCoordinationID(prefix string, identity []byte) string {
	sum := sha256.Sum256(identity)
	return prefix + "_" + hex.EncodeToString(sum[:16])
}

func makeObservationID(observation ContextObservation) string {
	copy := observation
	copy.ID = ""
	encoded, err := json.Marshal(copy)
	if err != nil {
		return ""
	}
	return stableCoordinationID("obs", encoded)
}

func makeContextActionID(observationID string, action ContextAction) string {
	return stableCoordinationID("act", []byte(strings.Join([]string{observationID, string(action.Kind), action.ResourceID,
		fmt.Sprint(action.Generation), fmt.Sprint(action.Bytes), fmt.Sprint(action.Tokens),
		string(action.Source.Kind), action.Source.StableID, string(action.Destination.Kind), action.Destination.StableID,
		action.ValidUntil.UTC().Format(time.RFC3339Nano)}, "\x00")))
}

func makeContextPlanID(reference ContextPlanReference, action ContextAction) string {
	return stableCoordinationID("cplan", []byte(strings.Join([]string{reference.ObservationID, reference.ResourceID,
		fmt.Sprint(reference.Generation), string(reference.Location.Kind), reference.Location.StableID,
		reference.ValidUntil.UTC().Format(time.RFC3339Nano), action.ID, string(action.Kind)}, "\x00")))
}

func makeFallbackContextPlanID(reference ContextPlanReference) string {
	return stableCoordinationID("cplan", []byte("fallback\x00"+reference.ObservationID))
}

func validStableCoordinationID(value, prefix string) bool {
	wantPrefix := prefix + "_"
	if !strings.HasPrefix(value, wantPrefix) || len(value) != len(wantPrefix)+32 {
		return false
	}
	_, err := hex.DecodeString(value[len(wantPrefix):])
	return err == nil
}

func boundedContextInt(value int64) int {
	max := int64(^uint(0) >> 1)
	if value > max {
		return int(max)
	}
	return int(value)
}
