package coordination

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

// PlacementCandidateKind is the closed #5416 placement ladder. It describes
// where compute runs, independently of the model selected for the work.
type PlacementCandidateKind string

const (
	PlacementCandidateDevice PlacementCandidateKind = "device"
	PlacementCandidateFleet  PlacementCandidateKind = "fleet"
	PlacementCandidateVendor PlacementCandidateKind = "vendor"
)

// PlacementDataBoundary is the furthest trust boundary crossed by a candidate.
// Ordering is deliberate: a request may admit candidates at or below its limit.
type PlacementDataBoundary string

const (
	PlacementBoundaryDevice       PlacementDataBoundary = "device"
	PlacementBoundaryOrganization PlacementDataBoundary = "organization"
	PlacementBoundaryExternal     PlacementDataBoundary = "external"
)

type PlacementProvenance string

const (
	PlacementMeasured  PlacementProvenance = "measured"
	PlacementEstimated PlacementProvenance = "estimated"
)

// PlacementMetric carries the source of a value and its measurement window.
// Fresh is derived by Observe and never trusted from a source snapshot.
type PlacementMetric struct {
	Value      float64             `json:"value"`
	Provenance PlacementProvenance `json:"provenance"`
	ObservedAt time.Time           `json:"observedAt"`
	FreshUntil time.Time           `json:"freshUntil"`
	Fresh      bool                `json:"fresh"`
}

// PlacementSourceCandidate is private observer input. Identity and tenant
// values are hashed before they enter an observation, plan, trace, or selfcheck.
type PlacementSourceCandidate struct {
	Identity        []byte
	Generation      uint64
	Kind            PlacementCandidateKind
	Region          string
	FailureDomain   string
	DataBoundary    PlacementDataBoundary
	EligibleTenants [][]byte
	Accelerators    []string
	Models          []string
	Available       bool
	Provenance      PlacementProvenance
	ObservedAt      time.Time
	FreshUntil      time.Time
	CacheLocality   PlacementMetric
	ServePressure   PlacementMetric
	Capacity        PlacementMetric
	QueueSeconds    PlacementMetric
	PricePerMTok    PlacementMetric
	CarbonGrams     PlacementMetric
}

type PlacementSourceSnapshot struct {
	Generation uint64
	CapturedAt time.Time
	Candidates []PlacementSourceCandidate
}

// PlacementStateObserver is read-only. Device inventory, the fleet scheduler,
// and vendor routers retain ownership of their source state.
type PlacementStateObserver interface {
	SnapshotPlacement(context.Context) (PlacementSourceSnapshot, error)
}

// PlacementCandidate is the content-free public projection of a source.
type PlacementCandidate struct {
	StableID          string                 `json:"stableId"`
	Generation        uint64                 `json:"generation"`
	Kind              PlacementCandidateKind `json:"kind"`
	Region            string                 `json:"region"`
	FailureDomain     string                 `json:"failureDomain"`
	DataBoundary      PlacementDataBoundary  `json:"dataBoundary"`
	EligibleTenantIDs []string               `json:"eligibleTenantIds"`
	Accelerators      []string               `json:"accelerators"`
	Models            []string               `json:"models"`
	Available         bool                   `json:"available"`
	Provenance        PlacementProvenance    `json:"provenance"`
	ObservedAt        time.Time              `json:"observedAt"`
	FreshUntil        time.Time              `json:"freshUntil"`
	Fresh             bool                   `json:"fresh"`
	CacheLocality     PlacementMetric        `json:"cacheLocality"`
	ServePressure     PlacementMetric        `json:"servePressure"`
	Capacity          PlacementMetric        `json:"capacity"`
	QueueSeconds      PlacementMetric        `json:"queueSeconds"`
	PricePerMTok      PlacementMetric        `json:"pricePerMTok"`
	CarbonGrams       PlacementMetric        `json:"carbonGrams"`
}

type PlacementObservation struct {
	ID         string               `json:"id"`
	Generation uint64               `json:"generation"`
	CapturedAt time.Time            `json:"capturedAt"`
	Age        time.Duration        `json:"age"`
	Fresh      bool                 `json:"fresh"`
	Candidates []PlacementCandidate `json:"candidates"`
}

var (
	ErrPlacementObservation  = errors.New("placement observation failed")
	ErrInvalidPlacementState = errors.New("invalid placement state")
)

// PlacementRankingPolicy makes every soft preference explicit. Scales turn
// queue, price, and carbon quantities into the same bounded [0,1] domain as
// locality, pressure, and spare capacity.
type PlacementRankingPolicy struct {
	LocalityWeight   float64       `json:"localityWeight"`
	PressureWeight   float64       `json:"pressureWeight"`
	CapacityWeight   float64       `json:"capacityWeight"`
	QueueWeight      float64       `json:"queueWeight"`
	PriceWeight      float64       `json:"priceWeight"`
	CarbonWeight     float64       `json:"carbonWeight"`
	ConfidenceWeight float64       `json:"confidenceWeight"`
	QueueScale       time.Duration `json:"queueScale"`
	PriceScale       float64       `json:"priceScale"`
	CarbonScale      float64       `json:"carbonScale"`
}

// PlacementRequest contains only placement facts. Private identities are
// deliberately excluded from JSON so a selfcheck cannot echo workload or
// tenant content.
type PlacementRequest struct {
	WorkloadIdentity    []byte                 `json:"-"`
	TenantIdentity      []byte                 `json:"-"`
	Region              string                 `json:"region"`
	Accelerator         string                 `json:"accelerator"`
	Model               string                 `json:"model"`
	MaximumDataBoundary PlacementDataBoundary  `json:"maximumDataBoundary"`
	OperatorPin         []byte                 `json:"-"`
	RankingPolicy       PlacementRankingPolicy `json:"rankingPolicy"`
}

type PlacementConstraint string

const (
	PlacementConstraintUnavailable  PlacementConstraint = "unavailable"
	PlacementConstraintStale        PlacementConstraint = "stale"
	PlacementConstraintRegion       PlacementConstraint = "region"
	PlacementConstraintTenant       PlacementConstraint = "tenant"
	PlacementConstraintAccelerator  PlacementConstraint = "accelerator"
	PlacementConstraintModel        PlacementConstraint = "model"
	PlacementConstraintDataBoundary PlacementConstraint = "data_boundary"
	PlacementConstraintCapacity     PlacementConstraint = "capacity"
)

type PlacementScore struct {
	LocalityPenalty   float64 `json:"localityPenalty"`
	PressurePenalty   float64 `json:"pressurePenalty"`
	CapacityPenalty   float64 `json:"capacityPenalty"`
	QueuePenalty      float64 `json:"queuePenalty"`
	PricePenalty      float64 `json:"pricePenalty"`
	CarbonPenalty     float64 `json:"carbonPenalty"`
	ConfidencePenalty float64 `json:"confidencePenalty"`
	Total             float64 `json:"total"`
}

// PlacementRanking preserves the exact soft inputs used to score a candidate.
// Constrained candidates have no PlacementRanking at all, proving that hard
// filtering ran before scoring.
type PlacementRanking struct {
	CacheLocality PlacementMetric `json:"cacheLocality"`
	ServePressure PlacementMetric `json:"servePressure"`
	Capacity      PlacementMetric `json:"capacity"`
	QueueSeconds  PlacementMetric `json:"queueSeconds"`
	PricePerMTok  PlacementMetric `json:"pricePerMTok"`
	CarbonGrams   PlacementMetric `json:"carbonGrams"`
	Score         PlacementScore  `json:"score"`
}

type PlacementCandidateVerdict struct {
	CandidateID string                 `json:"candidateId"`
	Kind        PlacementCandidateKind `json:"kind"`
	Eligible    bool                   `json:"eligible"`
	Constraints []PlacementConstraint  `json:"constraints"`
	Pinned      bool                   `json:"pinned"`
	Selected    bool                   `json:"selected"`
	Ranking     *PlacementRanking      `json:"ranking,omitempty"`
}

type PlacementSelection struct {
	CandidateID    string                 `json:"candidateId"`
	Generation     uint64                 `json:"generation"`
	Kind           PlacementCandidateKind `json:"kind"`
	Region         string                 `json:"region"`
	FailureDomain  string                 `json:"failureDomain"`
	DataBoundary   PlacementDataBoundary  `json:"dataBoundary"`
	Model          string                 `json:"model"`
	Accelerator    string                 `json:"accelerator"`
	OperatorPinned bool                   `json:"operatorPinned"`
	Score          PlacementScore         `json:"score"`
}

type PlacementLease struct {
	ID             string    `json:"id"`
	WorkloadID     string    `json:"workloadId"`
	IdempotencyKey string    `json:"idempotencyKey"`
	ValidUntil     time.Time `json:"validUntil"`
}

type PlacementUnavailableReason string

const (
	PlacementUnavailableInvalidRequest PlacementUnavailableReason = "invalid_request"
	PlacementUnavailableObservation    PlacementUnavailableReason = "invalid_observation"
	PlacementUnavailableNoCandidate    PlacementUnavailableReason = "no_eligible_candidate"
	PlacementUnavailablePinMissing     PlacementUnavailableReason = "operator_pin_unavailable"
	PlacementUnavailablePinConstrained PlacementUnavailableReason = "operator_pin_constrained"
)

type PlacementPlan struct {
	ID                string                      `json:"id"`
	RequestID         string                      `json:"requestId"`
	ObservationID     string                      `json:"observationId,omitempty"`
	WorkloadID        string                      `json:"workloadId"`
	RequestedModel    string                      `json:"requestedModel"`
	RankingPolicy     PlacementRankingPolicy      `json:"rankingPolicy"`
	Candidates        []PlacementCandidateVerdict `json:"candidates"`
	Selection         PlacementSelection          `json:"selection,omitempty"`
	Lease             PlacementLease              `json:"lease"`
	UsesFallback      bool                        `json:"usesFallback"`
	UnavailableReason PlacementUnavailableReason  `json:"unavailableReason,omitempty"`
}

type PlacementApplyStatus string

const (
	PlacementApplyAccepted    PlacementApplyStatus = "accepted"
	PlacementApplySuperseded  PlacementApplyStatus = "superseded"
	PlacementApplyUnavailable PlacementApplyStatus = "unavailable"
	PlacementApplyFailed      PlacementApplyStatus = "failed"
)

// PlacementApplyRequest is the injected router/scheduler seam. The underlying
// owner must persist IdempotencyKey if it needs replay protection across process
// restarts; the adapter also prevents duplicate calls within this process.
type PlacementApplyRequest struct {
	PlanID    string             `json:"planId"`
	Selection PlacementSelection `json:"selection"`
	Lease     PlacementLease     `json:"lease"`
}

type PlacementApplyResult struct {
	PlanID         string               `json:"planId"`
	CandidateID    string               `json:"candidateId,omitempty"`
	LeaseID        string               `json:"leaseId"`
	IdempotencyKey string               `json:"idempotencyKey"`
	Status         PlacementApplyStatus `json:"status"`
	Replayed       bool                 `json:"replayed"`
	UsedFallback   bool                 `json:"usedFallback"`
}

type PlacementApplier interface {
	ApplyPlacement(context.Context, PlacementApplyRequest) PlacementApplyResult
}

// PlacementPolicyFallback is the coordination-off seam. It delegates the
// original request to #5416's standalone routing policy instead of copying or
// subtly changing that policy inside this adapter.
type PlacementPolicyFallback interface {
	ApplyExistingPlacementPolicy(context.Context, PlacementRequest, PlacementLease) PlacementApplyResult
}

type PlacementAdapterOptions struct {
	Disabled              bool
	MaximumObservationAge time.Duration
	LeaseDuration         time.Duration
	Now                   func() time.Time
}

type PlacementAdapter struct {
	observer PlacementStateObserver
	applier  PlacementApplier
	fallback PlacementPolicyFallback
	disabled bool
	maxAge   time.Duration
	leaseTTL time.Duration
	now      func() time.Time

	stateMu          sync.Mutex
	observations     map[string]struct{}
	plans            map[string]string
	fallbackRequests map[string]PlacementRequest
	latestLease      map[string]string
	applyResults     map[string]PlacementApplyResult
}

func NewPlacementAdapter(observer PlacementStateObserver, applier PlacementApplier, fallback PlacementPolicyFallback, options PlacementAdapterOptions) *PlacementAdapter {
	maxAge := options.MaximumObservationAge
	if maxAge <= 0 {
		maxAge = time.Minute
	}
	leaseTTL := options.LeaseDuration
	if leaseTTL <= 0 {
		leaseTTL = 30 * time.Second
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &PlacementAdapter{
		observer: observer, applier: applier, fallback: fallback, disabled: options.Disabled,
		maxAge: maxAge, leaseTTL: leaseTTL, now: now, observations: make(map[string]struct{}),
		plans: make(map[string]string), fallbackRequests: make(map[string]PlacementRequest),
		latestLease: make(map[string]string), applyResults: make(map[string]PlacementApplyResult),
	}
}

// Observe projects private inventories into a stable, sorted metadata-only
// observation. Observer errors are collapsed so they cannot enter a trace.
func (a *PlacementAdapter) Observe(ctx context.Context) (PlacementObservation, error) {
	if a == nil || a.observer == nil {
		return PlacementObservation{}, ErrPlacementObservation
	}
	snapshot, err := a.observer.SnapshotPlacement(ctx)
	if err != nil {
		return PlacementObservation{}, ErrPlacementObservation
	}
	now := a.now().UTC()
	snapshot.CapturedAt = snapshot.CapturedAt.UTC()
	if snapshot.Generation == 0 || snapshot.CapturedAt.IsZero() || snapshot.CapturedAt.After(now) {
		return PlacementObservation{}, ErrInvalidPlacementState
	}
	observationFresh := now.Sub(snapshot.CapturedAt) <= a.maxAge
	candidates := make([]PlacementCandidate, 0, len(snapshot.Candidates))
	for _, source := range snapshot.Candidates {
		candidate, ok := projectPlacementCandidate(source, snapshot.CapturedAt, now, observationFresh)
		if !ok {
			return PlacementObservation{}, ErrInvalidPlacementState
		}
		candidates = append(candidates, candidate)
	}
	sort.Slice(candidates, func(i, j int) bool {
		if placementKindRank(candidates[i].Kind) != placementKindRank(candidates[j].Kind) {
			return placementKindRank(candidates[i].Kind) < placementKindRank(candidates[j].Kind)
		}
		return candidates[i].StableID < candidates[j].StableID
	})
	for i := 1; i < len(candidates); i++ {
		if candidates[i-1].StableID == candidates[i].StableID {
			return PlacementObservation{}, ErrInvalidPlacementState
		}
	}
	observation := PlacementObservation{
		Generation: snapshot.Generation, CapturedAt: snapshot.CapturedAt,
		Age: now.Sub(snapshot.CapturedAt), Fresh: observationFresh, Candidates: candidates,
	}
	observation.ID = makePlacementObservationID(observation)
	a.stateMu.Lock()
	a.observations[observation.ID] = struct{}{}
	a.stateMu.Unlock()
	return observation, nil
}

func projectPlacementCandidate(source PlacementSourceCandidate, capturedAt, now time.Time, observationFresh bool) (PlacementCandidate, bool) {
	if len(source.Identity) == 0 || source.Generation == 0 || !validPlacementKind(source.Kind) ||
		!validPlacementBoundary(source.DataBoundary) || !validPlacementProvenance(source.Provenance) ||
		!validPlacementLabel(source.Region) || !validPlacementLabel(source.FailureDomain) ||
		len(source.EligibleTenants) == 0 || len(source.Accelerators) == 0 || len(source.Models) == 0 {
		return PlacementCandidate{}, false
	}
	source.ObservedAt = source.ObservedAt.UTC()
	source.FreshUntil = source.FreshUntil.UTC()
	if !validPlacementWindow(source.ObservedAt, source.FreshUntil, capturedAt) {
		return PlacementCandidate{}, false
	}
	accelerators, ok := normalizedPlacementLabels(source.Accelerators)
	if !ok {
		return PlacementCandidate{}, false
	}
	models, ok := normalizedPlacementLabels(source.Models)
	if !ok {
		return PlacementCandidate{}, false
	}
	tenantIDs := make([]string, 0, len(source.EligibleTenants))
	for _, tenant := range source.EligibleTenants {
		if len(tenant) == 0 {
			return PlacementCandidate{}, false
		}
		tenantIDs = append(tenantIDs, stableCoordinationID("tenant", tenant))
	}
	sort.Strings(tenantIDs)
	if hasDuplicatePlacementLabel(tenantIDs) {
		return PlacementCandidate{}, false
	}
	metrics := []*PlacementMetric{&source.CacheLocality, &source.ServePressure, &source.Capacity,
		&source.QueueSeconds, &source.PricePerMTok, &source.CarbonGrams}
	for i, metric := range metrics {
		if !derivePlacementMetric(metric, capturedAt, now, i < 3) {
			return PlacementCandidate{}, false
		}
	}
	return PlacementCandidate{
		StableID: stableCoordinationID("candidate", source.Identity), Generation: source.Generation,
		Kind: source.Kind, Region: source.Region, FailureDomain: source.FailureDomain,
		DataBoundary: source.DataBoundary, EligibleTenantIDs: tenantIDs, Accelerators: accelerators,
		Models: models, Available: source.Available, Provenance: source.Provenance,
		ObservedAt: source.ObservedAt, FreshUntil: source.FreshUntil,
		Fresh:         observationFresh && !now.Before(source.ObservedAt) && now.Before(source.FreshUntil),
		CacheLocality: source.CacheLocality, ServePressure: source.ServePressure, Capacity: source.Capacity,
		QueueSeconds: source.QueueSeconds, PricePerMTok: source.PricePerMTok, CarbonGrams: source.CarbonGrams,
	}, true
}

func derivePlacementMetric(metric *PlacementMetric, capturedAt, now time.Time, bounded bool) bool {
	metric.ObservedAt = metric.ObservedAt.UTC()
	metric.FreshUntil = metric.FreshUntil.UTC()
	if !finitePlacementValue(metric.Value) || metric.Value < 0 || (bounded && metric.Value > 1) ||
		!validPlacementProvenance(metric.Provenance) || !validPlacementWindow(metric.ObservedAt, metric.FreshUntil, capturedAt) {
		return false
	}
	if metric.Value == 0 {
		metric.Value = 0 // normalize negative zero before hashing or scoring
	}
	metric.Fresh = !now.Before(metric.ObservedAt) && now.Before(metric.FreshUntil)
	return true
}

func validPlacementWindow(observedAt, freshUntil, capturedAt time.Time) bool {
	return !observedAt.IsZero() && !freshUntil.IsZero() && !observedAt.After(capturedAt) && freshUntil.After(observedAt)
}

func normalizedPlacementLabels(values []string) ([]string, bool) {
	copyValues := append([]string(nil), values...)
	for _, value := range copyValues {
		if !validPlacementLabel(value) {
			return nil, false
		}
	}
	sort.Strings(copyValues)
	if hasDuplicatePlacementLabel(copyValues) {
		return nil, false
	}
	return copyValues, true
}

func hasDuplicatePlacementLabel(values []string) bool {
	for i := 1; i < len(values); i++ {
		if values[i-1] == values[i] {
			return true
		}
	}
	return false
}

func validPlacementLabel(value string) bool {
	return value != "" && strings.TrimSpace(value) == value && !strings.ContainsAny(value, "\x00\r\n")
}

func validPlacementKind(kind PlacementCandidateKind) bool {
	return kind == PlacementCandidateDevice || kind == PlacementCandidateFleet || kind == PlacementCandidateVendor
}

func placementKindRank(kind PlacementCandidateKind) int {
	switch kind {
	case PlacementCandidateDevice:
		return 0
	case PlacementCandidateFleet:
		return 1
	case PlacementCandidateVendor:
		return 2
	default:
		return math.MaxInt
	}
}

func validPlacementBoundary(boundary PlacementDataBoundary) bool {
	return boundary == PlacementBoundaryDevice || boundary == PlacementBoundaryOrganization || boundary == PlacementBoundaryExternal
}

func placementBoundaryRank(boundary PlacementDataBoundary) int {
	switch boundary {
	case PlacementBoundaryDevice:
		return 0
	case PlacementBoundaryOrganization:
		return 1
	case PlacementBoundaryExternal:
		return 2
	default:
		return math.MaxInt
	}
}

func validPlacementProvenance(provenance PlacementProvenance) bool {
	return provenance == PlacementMeasured || provenance == PlacementEstimated
}

func finitePlacementValue(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// Plan removes every constraint failure before it computes any soft score.
// An eligible operator pin wins after scoring, while an ineligible pin makes
// the plan unavailable rather than silently bypassing the pin or a constraint.
func (a *PlacementAdapter) Plan(observation PlacementObservation, request PlacementRequest) PlacementPlan {
	policy := effectivePlacementRankingPolicy(request.RankingPolicy)
	requestID, workloadID := placementRequestIDs(request, policy)
	base := PlacementPlan{
		RequestID: requestID, WorkloadID: workloadID, RequestedModel: request.Model,
		RankingPolicy: policy, Candidates: []PlacementCandidateVerdict{},
	}
	requestValid := validatePlacementRequest(request, policy)
	if a == nil || !requestValid {
		base.UnavailableReason = PlacementUnavailableInvalidRequest
		return a.finishPlacementPlan(base, request, false)
	}
	if a.disabled {
		base.UsesFallback = true
		return a.finishPlacementPlan(base, request, true)
	}
	if !a.validProducedPlacementObservation(observation) {
		base.UnavailableReason = PlacementUnavailableObservation
		return a.finishPlacementPlan(base, request, false)
	}
	base.ObservationID = observation.ID
	base.Lease.ValidUntil = observation.CapturedAt.Add(a.maxAge)
	planningNow := a.now().UTC()
	observationCurrent := observation.Fresh && !planningNow.Before(observation.CapturedAt) &&
		planningNow.Sub(observation.CapturedAt) <= a.maxAge
	pinID := ""
	if len(request.OperatorPin) != 0 {
		pinID = stableCoordinationID("candidate", request.OperatorPin)
	}
	tenantID := stableCoordinationID("tenant", request.TenantIdentity)
	selected := -1
	pinFound, pinEligible := false, false
	for _, observedCandidate := range observation.Candidates {
		candidate := refreshPlacementCandidate(observedCandidate, planningNow, observationCurrent)
		constraints := placementConstraintFailures(candidate, request, tenantID)
		verdict := PlacementCandidateVerdict{
			CandidateID: candidate.StableID, Kind: candidate.Kind, Constraints: constraints,
			Eligible: len(constraints) == 0, Pinned: pinID != "" && candidate.StableID == pinID,
		}
		if verdict.Pinned {
			pinFound, pinEligible = true, verdict.Eligible
		}
		if verdict.Eligible {
			ranking := rankPlacementCandidate(candidate, policy)
			verdict.Ranking = &ranking
		}
		base.Candidates = append(base.Candidates, verdict)
	}
	if pinID != "" {
		if !pinFound {
			base.UnavailableReason = PlacementUnavailablePinMissing
		} else if !pinEligible {
			base.UnavailableReason = PlacementUnavailablePinConstrained
		} else {
			selected = placementVerdictIndex(base.Candidates, pinID)
		}
	} else {
		for i := range base.Candidates {
			if !base.Candidates[i].Eligible {
				continue
			}
			if selected < 0 || placementVerdictLess(base.Candidates[i], base.Candidates[selected]) {
				selected = i
			}
		}
		if selected < 0 {
			base.UnavailableReason = PlacementUnavailableNoCandidate
		}
	}
	if selected >= 0 {
		base.Candidates[selected].Selected = true
		candidate := refreshPlacementCandidate(observation.Candidates[selected], planningNow, observationCurrent)
		base.Selection = PlacementSelection{
			CandidateID: candidate.StableID, Generation: candidate.Generation, Kind: candidate.Kind,
			Region: candidate.Region, FailureDomain: candidate.FailureDomain, DataBoundary: candidate.DataBoundary,
			Model: request.Model, Accelerator: request.Accelerator, OperatorPinned: pinID != "",
			Score: base.Candidates[selected].Ranking.Score,
		}
		candidateDeadline := placementCandidateLeaseDeadline(candidate)
		if candidateDeadline.Before(base.Lease.ValidUntil) {
			base.Lease.ValidUntil = candidateDeadline
		}
	}
	return a.finishPlacementPlan(base, request, true)
}

func refreshPlacementCandidate(candidate PlacementCandidate, now time.Time, observationCurrent bool) PlacementCandidate {
	candidate.Fresh = candidate.Fresh && observationCurrent && now.Before(candidate.FreshUntil)
	metrics := []*PlacementMetric{&candidate.CacheLocality, &candidate.ServePressure, &candidate.Capacity,
		&candidate.QueueSeconds, &candidate.PricePerMTok, &candidate.CarbonGrams}
	for _, metric := range metrics {
		metric.Fresh = metric.Fresh && !now.Before(metric.ObservedAt) && now.Before(metric.FreshUntil)
	}
	return candidate
}

func placementConstraintFailures(candidate PlacementCandidate, request PlacementRequest, tenantID string) []PlacementConstraint {
	failures := make([]PlacementConstraint, 0, 8)
	if !candidate.Available {
		failures = append(failures, PlacementConstraintUnavailable)
	}
	if !candidate.Fresh {
		failures = append(failures, PlacementConstraintStale)
	}
	if candidate.Region != request.Region {
		failures = append(failures, PlacementConstraintRegion)
	}
	if !containsPlacementLabel(candidate.EligibleTenantIDs, tenantID) {
		failures = append(failures, PlacementConstraintTenant)
	}
	if !containsPlacementLabel(candidate.Accelerators, request.Accelerator) {
		failures = append(failures, PlacementConstraintAccelerator)
	}
	if !containsPlacementLabel(candidate.Models, request.Model) {
		failures = append(failures, PlacementConstraintModel)
	}
	if placementBoundaryRank(candidate.DataBoundary) > placementBoundaryRank(request.MaximumDataBoundary) {
		failures = append(failures, PlacementConstraintDataBoundary)
	}
	if !candidate.Capacity.Fresh || candidate.Capacity.Value <= 0 {
		failures = append(failures, PlacementConstraintCapacity)
	}
	return failures
}

func containsPlacementLabel(sortedValues []string, value string) bool {
	i := sort.SearchStrings(sortedValues, value)
	return i < len(sortedValues) && sortedValues[i] == value
}

func rankPlacementCandidate(candidate PlacementCandidate, policy PlacementRankingPolicy) PlacementRanking {
	score := PlacementScore{
		LocalityPenalty: policy.LocalityWeight * placementHigherIsBetterPenalty(candidate.CacheLocality),
		PressurePenalty: policy.PressureWeight * placementLowerIsBetterPenalty(candidate.ServePressure, 1),
		CapacityPenalty: policy.CapacityWeight * placementHigherIsBetterPenalty(candidate.Capacity),
		QueuePenalty:    policy.QueueWeight * placementLowerIsBetterPenalty(candidate.QueueSeconds, policy.QueueScale.Seconds()),
		PricePenalty:    policy.PriceWeight * placementLowerIsBetterPenalty(candidate.PricePerMTok, policy.PriceScale),
		CarbonPenalty:   policy.CarbonWeight * placementLowerIsBetterPenalty(candidate.CarbonGrams, policy.CarbonScale),
	}
	metrics := []PlacementMetric{candidate.CacheLocality, candidate.ServePressure, candidate.Capacity,
		candidate.QueueSeconds, candidate.PricePerMTok, candidate.CarbonGrams}
	confidence := 0.0
	for _, metric := range metrics {
		switch {
		case !metric.Fresh:
			confidence += 1
		case metric.Provenance == PlacementEstimated:
			confidence += .5
		}
	}
	score.ConfidencePenalty = policy.ConfidenceWeight * confidence / float64(len(metrics))
	score.Total = score.LocalityPenalty + score.PressurePenalty + score.CapacityPenalty + score.QueuePenalty +
		score.PricePenalty + score.CarbonPenalty + score.ConfidencePenalty
	return PlacementRanking{
		CacheLocality: candidate.CacheLocality, ServePressure: candidate.ServePressure, Capacity: candidate.Capacity,
		QueueSeconds: candidate.QueueSeconds, PricePerMTok: candidate.PricePerMTok, CarbonGrams: candidate.CarbonGrams,
		Score: score,
	}
}

func placementHigherIsBetterPenalty(metric PlacementMetric) float64 {
	if !metric.Fresh {
		return 1
	}
	return 1 - metric.Value
}

func placementLowerIsBetterPenalty(metric PlacementMetric, scale float64) float64 {
	if !metric.Fresh {
		return 1
	}
	return math.Min(metric.Value/scale, 1)
}

func placementVerdictLess(left, right PlacementCandidateVerdict) bool {
	if left.Ranking.Score.Total != right.Ranking.Score.Total {
		return left.Ranking.Score.Total < right.Ranking.Score.Total
	}
	if placementKindRank(left.Kind) != placementKindRank(right.Kind) {
		return placementKindRank(left.Kind) < placementKindRank(right.Kind)
	}
	return left.CandidateID < right.CandidateID
}

func placementVerdictIndex(verdicts []PlacementCandidateVerdict, candidateID string) int {
	for i := range verdicts {
		if verdicts[i].CandidateID == candidateID {
			return i
		}
	}
	return -1
}

func effectivePlacementRankingPolicy(policy PlacementRankingPolicy) PlacementRankingPolicy {
	allWeightsZero := policy.LocalityWeight == 0 && policy.PressureWeight == 0 && policy.CapacityWeight == 0 &&
		policy.QueueWeight == 0 && policy.PriceWeight == 0 && policy.CarbonWeight == 0 && policy.ConfidenceWeight == 0
	if allWeightsZero {
		policy.LocalityWeight = 3
		policy.PressureWeight = 3
		policy.CapacityWeight = 1
		policy.QueueWeight = 2
		policy.PriceWeight = 1
		policy.CarbonWeight = 1
		policy.ConfidenceWeight = 1
	}
	if policy.QueueScale == 0 {
		policy.QueueScale = time.Minute
	}
	if policy.PriceScale == 0 {
		policy.PriceScale = 10
	}
	if policy.CarbonScale == 0 {
		policy.CarbonScale = 100
	}
	return policy
}

func validPlacementRankingPolicy(policy PlacementRankingPolicy) bool {
	values := []float64{policy.LocalityWeight, policy.PressureWeight, policy.CapacityWeight, policy.QueueWeight,
		policy.PriceWeight, policy.CarbonWeight, policy.ConfidenceWeight, policy.PriceScale, policy.CarbonScale}
	for _, value := range values {
		if !finitePlacementValue(value) || value < 0 {
			return false
		}
	}
	return policy.QueueScale > 0 && policy.PriceScale > 0 && policy.CarbonScale > 0
}

func validatePlacementRequest(request PlacementRequest, policy PlacementRankingPolicy) bool {
	return len(request.WorkloadIdentity) != 0 && len(request.TenantIdentity) != 0 &&
		validPlacementLabel(request.Region) && validPlacementLabel(request.Accelerator) && validPlacementLabel(request.Model) &&
		validPlacementBoundary(request.MaximumDataBoundary) && validPlacementRankingPolicy(policy)
}

func (a *PlacementAdapter) finishPlacementPlan(plan PlacementPlan, request PlacementRequest, activate bool) PlacementPlan {
	if a == nil {
		return plan
	}
	now := a.now().UTC()
	validUntil := now.Add(a.leaseTTL)
	if !plan.Lease.ValidUntil.IsZero() && plan.Lease.ValidUntil.Before(validUntil) {
		validUntil = plan.Lease.ValidUntil
	}
	plan.Lease = PlacementLease{WorkloadID: plan.WorkloadID, ValidUntil: validUntil}
	plan.Lease.ID = makePlacementLeaseID(plan)
	plan.Lease.IdempotencyKey = stableCoordinationID("idempotency", []byte(plan.Lease.ID))
	plan.ID = makePlacementPlanID(plan)
	fingerprint := placementPlanFingerprint(plan)
	a.stateMu.Lock()
	a.plans[plan.ID] = fingerprint
	if plan.UsesFallback {
		a.fallbackRequests[plan.ID] = clonePlacementRequest(request)
	}
	if activate {
		a.latestLease[plan.WorkloadID] = plan.Lease.ID
	}
	a.stateMu.Unlock()
	return plan
}

func placementCandidateLeaseDeadline(candidate PlacementCandidate) time.Time {
	earliest := candidate.FreshUntil
	metrics := []PlacementMetric{candidate.CacheLocality, candidate.ServePressure, candidate.Capacity,
		candidate.QueueSeconds, candidate.PricePerMTok, candidate.CarbonGrams}
	for _, metric := range metrics {
		// A stale soft estimate was already scored conservatively. It cannot
		// retroactively expire a lease; only fresh inputs bound their use.
		if metric.Fresh && metric.FreshUntil.Before(earliest) {
			earliest = metric.FreshUntil
		}
	}
	return earliest
}

func clonePlacementRequest(request PlacementRequest) PlacementRequest {
	request.WorkloadIdentity = append([]byte(nil), request.WorkloadIdentity...)
	request.TenantIdentity = append([]byte(nil), request.TenantIdentity...)
	request.OperatorPin = append([]byte(nil), request.OperatorPin...)
	return request
}

func (a *PlacementAdapter) validProducedPlacementObservation(observation PlacementObservation) bool {
	if observation.ID == "" || observation.ID != makePlacementObservationID(observation) {
		return false
	}
	a.stateMu.Lock()
	defer a.stateMu.Unlock()
	_, ok := a.observations[observation.ID]
	return ok
}

// Apply holds the local lease/idempotency lock across the injected seam. A
// newer plan for the same workload supersedes an unapplied older lease.
func (a *PlacementAdapter) Apply(ctx context.Context, plan PlacementPlan) PlacementApplyResult {
	if a == nil {
		return placementApplyResult(plan, PlacementApplyUnavailable, false, false)
	}
	a.stateMu.Lock()
	defer a.stateMu.Unlock()
	if !a.validProducedPlacementPlanLocked(plan) {
		return placementApplyResult(plan, PlacementApplyFailed, false, false)
	}
	if prior, ok := a.applyResults[plan.Lease.IdempotencyKey]; ok {
		prior.Replayed = true
		return prior
	}
	if plan.UnavailableReason == PlacementUnavailableInvalidRequest || plan.UnavailableReason == PlacementUnavailableObservation {
		return a.rememberPlacementApplyLocked(plan, placementApplyResult(plan, PlacementApplyUnavailable, false, plan.UsesFallback))
	}
	if a.latestLease[plan.WorkloadID] != plan.Lease.ID || !a.now().UTC().Before(plan.Lease.ValidUntil) {
		return a.rememberPlacementApplyLocked(plan, placementApplyResult(plan, PlacementApplySuperseded, false, plan.UsesFallback))
	}
	var result PlacementApplyResult
	if plan.UsesFallback {
		request, ok := a.fallbackRequests[plan.ID]
		if !ok || a.fallback == nil {
			result = placementApplyResult(plan, PlacementApplyUnavailable, false, true)
		} else {
			result = a.fallback.ApplyExistingPlacementPolicy(ctx, clonePlacementRequest(request), plan.Lease)
			result = normalizePlacementApplyResult(plan, result, true)
		}
	} else if plan.Selection.CandidateID == "" {
		result = placementApplyResult(plan, PlacementApplyUnavailable, false, false)
	} else if a.applier == nil {
		result = placementApplyResult(plan, PlacementApplyUnavailable, false, false)
	} else {
		result = a.applier.ApplyPlacement(ctx, PlacementApplyRequest{PlanID: plan.ID, Selection: plan.Selection, Lease: plan.Lease})
		result = normalizePlacementApplyResult(plan, result, false)
	}
	return a.rememberPlacementApplyLocked(plan, result)
}

func normalizePlacementApplyResult(plan PlacementPlan, result PlacementApplyResult, fallback bool) PlacementApplyResult {
	status := result.Status
	if !validPlacementApplyStatus(status) {
		status = PlacementApplyFailed
	}
	return placementApplyResult(plan, status, false, fallback)
}

func placementApplyResult(plan PlacementPlan, status PlacementApplyStatus, replayed, fallback bool) PlacementApplyResult {
	return PlacementApplyResult{
		PlanID: plan.ID, CandidateID: plan.Selection.CandidateID, LeaseID: plan.Lease.ID,
		IdempotencyKey: plan.Lease.IdempotencyKey, Status: status, Replayed: replayed, UsedFallback: fallback,
	}
}

func (a *PlacementAdapter) rememberPlacementApplyLocked(plan PlacementPlan, result PlacementApplyResult) PlacementApplyResult {
	a.applyResults[plan.Lease.IdempotencyKey] = result
	return result
}

func validPlacementApplyStatus(status PlacementApplyStatus) bool {
	switch status {
	case PlacementApplyAccepted, PlacementApplySuperseded, PlacementApplyUnavailable, PlacementApplyFailed:
		return true
	default:
		return false
	}
}

func (a *PlacementAdapter) validProducedPlacementPlanLocked(plan PlacementPlan) bool {
	if plan.ID == "" || plan.ID != makePlacementPlanID(plan) || plan.Lease.ID != makePlacementLeaseID(plan) ||
		plan.Lease.IdempotencyKey != stableCoordinationID("idempotency", []byte(plan.Lease.ID)) ||
		plan.Lease.WorkloadID != plan.WorkloadID || !validStableCoordinationID(plan.RequestID, "request") ||
		!validStableCoordinationID(plan.WorkloadID, "workload") || !validStableCoordinationID(plan.Lease.ID, "lease") ||
		!validStableCoordinationID(plan.Lease.IdempotencyKey, "idempotency") {
		return false
	}
	want, ok := a.plans[plan.ID]
	return ok && want == placementPlanFingerprint(plan)
}

type PlacementTracePhase string

const (
	PlacementTraceObserve PlacementTracePhase = "observe"
	PlacementTracePlan    PlacementTracePhase = "plan"
	PlacementTraceApply   PlacementTracePhase = "apply"
)

// PlacementTraceEvent contains closed vocabulary and stable IDs only.
type PlacementTraceEvent struct {
	Phase          PlacementTracePhase    `json:"phase"`
	ObservationID  string                 `json:"observationId,omitempty"`
	PlanID         string                 `json:"planId,omitempty"`
	CandidateID    string                 `json:"candidateId,omitempty"`
	Kind           PlacementCandidateKind `json:"kind,omitempty"`
	CandidateCount int                    `json:"candidateCount,omitempty"`
	EligibleCount  int                    `json:"eligibleCount,omitempty"`
	SelectedScore  float64                `json:"selectedScore,omitempty"`
	ApplyStatus    PlacementApplyStatus   `json:"applyStatus,omitempty"`
}

type PlacementAdapterSelfCheck struct {
	Observation PlacementObservation  `json:"observation"`
	Plan        PlacementPlan         `json:"plan"`
	Apply       PlacementApplyResult  `json:"apply"`
	Trace       []PlacementTraceEvent `json:"trace"`
	Failure     PlacementApplyStatus  `json:"failure,omitempty"`
}

// SelfCheck captures observe -> filtered ranking -> selected apply without
// workload, tenant, prompt, response, or observer error content. Replay state is
// normalized out so an identical offline fixture produces byte-stable JSON.
func (a *PlacementAdapter) SelfCheck(ctx context.Context, request PlacementRequest) PlacementAdapterSelfCheck {
	observation, err := a.Observe(ctx)
	if err != nil {
		return PlacementAdapterSelfCheck{
			Failure: PlacementApplyUnavailable,
			Trace:   []PlacementTraceEvent{{Phase: PlacementTraceObserve, ApplyStatus: PlacementApplyUnavailable}},
		}
	}
	plan := a.Plan(observation, request)
	apply := a.Apply(ctx, plan)
	apply.Replayed = false
	eligible := 0
	for _, verdict := range plan.Candidates {
		if verdict.Eligible {
			eligible++
		}
	}
	trace := []PlacementTraceEvent{
		{Phase: PlacementTraceObserve, ObservationID: observation.ID, CandidateCount: len(observation.Candidates)},
		{Phase: PlacementTracePlan, ObservationID: observation.ID, PlanID: plan.ID, CandidateID: plan.Selection.CandidateID,
			Kind: plan.Selection.Kind, CandidateCount: len(plan.Candidates), EligibleCount: eligible, SelectedScore: plan.Selection.Score.Total},
		{Phase: PlacementTraceApply, ObservationID: observation.ID, PlanID: plan.ID,
			CandidateID: plan.Selection.CandidateID, Kind: plan.Selection.Kind, ApplyStatus: apply.Status},
	}
	check := PlacementAdapterSelfCheck{Observation: observation, Plan: plan, Apply: apply, Trace: trace}
	if apply.Status != PlacementApplyAccepted {
		check.Failure = apply.Status
	}
	return check
}

func placementRequestIDs(request PlacementRequest, policy PlacementRankingPolicy) (requestID, workloadID string) {
	workloadID = stableCoordinationID("workload", request.WorkloadIdentity)
	pinID := ""
	if len(request.OperatorPin) != 0 {
		pinID = stableCoordinationID("candidate", request.OperatorPin)
	}
	reference := struct {
		WorkloadID    string                 `json:"workloadId"`
		TenantID      string                 `json:"tenantId"`
		Region        string                 `json:"region"`
		Accelerator   string                 `json:"accelerator"`
		Model         string                 `json:"model"`
		Boundary      PlacementDataBoundary  `json:"boundary"`
		PinID         string                 `json:"pinId"`
		RankingPolicy PlacementRankingPolicy `json:"rankingPolicy"`
	}{
		WorkloadID: workloadID, TenantID: stableCoordinationID("tenant", request.TenantIdentity),
		Region: request.Region, Accelerator: request.Accelerator, Model: request.Model,
		Boundary: request.MaximumDataBoundary, PinID: pinID, RankingPolicy: policy,
	}
	encoded, _ := json.Marshal(reference)
	return stableCoordinationID("request", encoded), workloadID
}

func makePlacementObservationID(observation PlacementObservation) string {
	copyObservation := observation
	copyObservation.ID = ""
	encoded, err := json.Marshal(copyObservation)
	if err != nil {
		return ""
	}
	return stableCoordinationID("observation", encoded)
}

func makePlacementLeaseID(plan PlacementPlan) string {
	reference := struct {
		RequestID     string    `json:"requestId"`
		ObservationID string    `json:"observationId"`
		CandidateID   string    `json:"candidateId"`
		Generation    uint64    `json:"generation"`
		WorkloadID    string    `json:"workloadId"`
		ValidUntil    time.Time `json:"validUntil"`
		Fallback      bool      `json:"fallback"`
	}{
		RequestID: plan.RequestID, ObservationID: plan.ObservationID, CandidateID: plan.Selection.CandidateID,
		Generation: plan.Selection.Generation, WorkloadID: plan.WorkloadID,
		ValidUntil: plan.Lease.ValidUntil, Fallback: plan.UsesFallback,
	}
	encoded, _ := json.Marshal(reference)
	return stableCoordinationID("lease", encoded)
}

func makePlacementPlanID(plan PlacementPlan) string {
	copyPlan := plan
	copyPlan.ID = ""
	encoded, err := json.Marshal(copyPlan)
	if err != nil {
		return ""
	}
	return stableCoordinationID("placement", encoded)
}

func placementPlanFingerprint(plan PlacementPlan) string {
	encoded, err := json.Marshal(plan)
	if err != nil {
		return ""
	}
	return stableCoordinationID("planfp", encoded)
}
