package coordination

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"
)

type placementObserverFunc func(context.Context) (PlacementSourceSnapshot, error)

func (f placementObserverFunc) SnapshotPlacement(ctx context.Context) (PlacementSourceSnapshot, error) {
	return f(ctx)
}

type recordingPlacementApplier struct {
	status PlacementApplyStatus
	calls  []PlacementApplyRequest
}

func (a *recordingPlacementApplier) ApplyPlacement(_ context.Context, request PlacementApplyRequest) PlacementApplyResult {
	a.calls = append(a.calls, request)
	status := a.status
	if status == "" {
		status = PlacementApplyAccepted
	}
	return PlacementApplyResult{Status: status}
}

type recordingPlacementFallback struct {
	status  PlacementApplyStatus
	calls   int
	request PlacementRequest
	lease   PlacementLease
}

func (f *recordingPlacementFallback) ApplyExistingPlacementPolicy(_ context.Context, request PlacementRequest, lease PlacementLease) PlacementApplyResult {
	f.calls++
	f.request = clonePlacementRequest(request)
	f.lease = lease
	status := f.status
	if status == "" {
		status = PlacementApplyAccepted
	}
	return PlacementApplyResult{Status: status}
}

func TestPlacementAdapterObservesAllCandidateKindsWithProvenanceAndFreshness(t *testing.T) {
	now := placementTestNow()
	device := placementCandidateFixture("private-device", PlacementCandidateDevice, now)
	fleet := placementCandidateFixture("private-fleet", PlacementCandidateFleet, now)
	fleet.Provenance = PlacementEstimated
	fleet.PricePerMTok.Provenance = PlacementEstimated
	fleet.PricePerMTok.FreshUntil = now
	vendor := placementCandidateFixture("private-vendor", PlacementCandidateVendor, now)
	snapshot := placementSnapshotFixture(now, vendor, fleet, device)

	adapter := newPlacementFixtureAdapter(snapshot, now, nil, nil, PlacementAdapterOptions{})
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !observation.Fresh || observation.Age != time.Second {
		t.Fatalf("observation freshness = %t age = %s", observation.Fresh, observation.Age)
	}
	wantKinds := []PlacementCandidateKind{PlacementCandidateDevice, PlacementCandidateFleet, PlacementCandidateVendor}
	for i, want := range wantKinds {
		if observation.Candidates[i].Kind != want {
			t.Fatalf("candidate %d kind = %q, want %q", i, observation.Candidates[i].Kind, want)
		}
	}
	observedFleet := observation.Candidates[1]
	if observedFleet.Provenance != PlacementEstimated || observedFleet.PricePerMTok.Provenance != PlacementEstimated {
		t.Fatalf("provenance was not preserved: %+v", observedFleet)
	}
	if observedFleet.PricePerMTok.Fresh {
		t.Fatalf("stale metric reported fresh: %+v", observedFleet.PricePerMTok)
	}
	if !observedFleet.Fresh || !observedFleet.CacheLocality.Fresh {
		t.Fatalf("fresh candidate or metric reported stale: %+v", observedFleet)
	}

	reversed := placementSnapshotFixture(now, device, fleet, vendor)
	reversedObservation, err := newPlacementFixtureAdapter(reversed, now, nil, nil, PlacementAdapterOptions{}).Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if observation.ID != reversedObservation.ID {
		t.Fatalf("observation ID depends on source order: %q != %q", observation.ID, reversedObservation.ID)
	}

	encoded, err := json.Marshal(observation)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"private-device", "private-fleet", "private-vendor", "private-tenant"} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("observation leaked %q: %s", secret, encoded)
		}
	}
}

func TestPlacementAdapterRejectsInvalidStateAndCollapsesObserverErrors(t *testing.T) {
	now := placementTestNow()
	invalid := placementCandidateFixture("device", PlacementCandidateDevice, now)
	invalid.CacheLocality.Provenance = PlacementProvenance("untrusted")
	adapter := newPlacementFixtureAdapter(placementSnapshotFixture(now, invalid), now, nil, nil, PlacementAdapterOptions{})
	if _, err := adapter.Observe(context.Background()); !errors.Is(err, ErrInvalidPlacementState) {
		t.Fatalf("invalid provenance error = %v", err)
	}

	secretError := errors.New("private-tenant prompt content")
	failing := NewPlacementAdapter(placementObserverFunc(func(context.Context) (PlacementSourceSnapshot, error) {
		return PlacementSourceSnapshot{}, secretError
	}), nil, nil, PlacementAdapterOptions{Now: func() time.Time { return now }})
	_, err := failing.Observe(context.Background())
	if !errors.Is(err, ErrPlacementObservation) || strings.Contains(err.Error(), secretError.Error()) {
		t.Fatalf("observer error was not safely collapsed: %v", err)
	}
}

func TestPlacementAdapterRechecksFreshnessBeforePlanning(t *testing.T) {
	now := placementTestNow()
	clock := now
	candidate := placementCandidateFixture("device", PlacementCandidateDevice, now)
	snapshot := placementSnapshotFixture(now, candidate)
	adapter := NewPlacementAdapter(placementObserverFunc(func(context.Context) (PlacementSourceSnapshot, error) {
		return snapshot, nil
	}), nil, nil, PlacementAdapterOptions{
		MaximumObservationAge: 5 * time.Minute,
		Now:                   func() time.Time { return clock },
	})
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	clock = candidate.FreshUntil
	plan := adapter.Plan(observation, placementRequestFixture())
	if plan.UnavailableReason != PlacementUnavailableNoCandidate || plan.Selection.CandidateID != "" {
		t.Fatalf("expired observation candidate remained selectable: %+v", plan)
	}
	if len(plan.Candidates) != 1 || plan.Candidates[0].Eligible || plan.Candidates[0].Ranking != nil ||
		!reflect.DeepEqual(plan.Candidates[0].Constraints, []PlacementConstraint{PlacementConstraintStale, PlacementConstraintCapacity}) {
		t.Fatalf("freshness was not a hard pre-score constraint: %+v", plan.Candidates)
	}
}

func TestPlacementAdapterFiltersHardConstraintsBeforeScoring(t *testing.T) {
	now := placementTestNow()
	good := placementCandidateFixture("good", PlacementCandidateDevice, now)
	wrongRegion := placementCandidateFixture("wrong-region", PlacementCandidateFleet, now)
	wrongRegion.Region = "eu-central"
	wrongTenant := placementCandidateFixture("wrong-tenant", PlacementCandidateFleet, now)
	wrongTenant.EligibleTenants = [][]byte{[]byte("other-tenant")}
	wrongAccelerator := placementCandidateFixture("wrong-accelerator", PlacementCandidateFleet, now)
	wrongAccelerator.Accelerators = []string{"tpu"}
	wrongModel := placementCandidateFixture("wrong-model", PlacementCandidateVendor, now)
	wrongModel.Models = []string{"other-model"}
	wrongBoundary := placementCandidateFixture("wrong-boundary", PlacementCandidateVendor, now)
	wrongBoundary.DataBoundary = PlacementBoundaryExternal
	unavailable := placementCandidateFixture("unavailable", PlacementCandidateFleet, now)
	unavailable.Available = false
	stale := placementCandidateFixture("stale", PlacementCandidateFleet, now)
	stale.FreshUntil = now
	noCapacity := placementCandidateFixture("no-capacity", PlacementCandidateFleet, now)
	noCapacity.Capacity.Value = 0
	snapshot := placementSnapshotFixture(now, noCapacity, stale, unavailable, wrongBoundary, wrongModel, wrongAccelerator, wrongTenant, wrongRegion, good)
	adapter := newPlacementFixtureAdapter(snapshot, now, nil, nil, PlacementAdapterOptions{})
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	request := placementRequestFixture()
	plan := adapter.Plan(observation, request)
	if plan.UnavailableReason != "" || plan.Selection.CandidateID != stablePlacementID("candidate", good.Identity) {
		t.Fatalf("eligible candidate was not selected: %+v", plan)
	}

	want := map[string]PlacementConstraint{
		stablePlacementID("candidate", wrongRegion.Identity):      PlacementConstraintRegion,
		stablePlacementID("candidate", wrongTenant.Identity):      PlacementConstraintTenant,
		stablePlacementID("candidate", wrongAccelerator.Identity): PlacementConstraintAccelerator,
		stablePlacementID("candidate", wrongModel.Identity):       PlacementConstraintModel,
		stablePlacementID("candidate", wrongBoundary.Identity):    PlacementConstraintDataBoundary,
		stablePlacementID("candidate", unavailable.Identity):      PlacementConstraintUnavailable,
		stablePlacementID("candidate", stale.Identity):            PlacementConstraintStale,
		stablePlacementID("candidate", noCapacity.Identity):       PlacementConstraintCapacity,
	}
	for _, verdict := range plan.Candidates {
		constraint, constrained := want[verdict.CandidateID]
		if !constrained {
			if !verdict.Eligible || verdict.Ranking == nil || !verdict.Selected {
				t.Fatalf("eligible verdict lost ranking or selection: %+v", verdict)
			}
			continue
		}
		if verdict.Eligible || verdict.Ranking != nil || verdict.Selected {
			t.Fatalf("constrained candidate was scored or selected: %+v", verdict)
		}
		if !reflect.DeepEqual(verdict.Constraints, []PlacementConstraint{constraint}) {
			t.Fatalf("constraints for %s = %v, want %v", verdict.CandidateID, verdict.Constraints, constraint)
		}
	}
}

func TestPlacementAdapterCapacityQueuePriceAndCarbonPolicyCanChangeRanking(t *testing.T) {
	now := placementTestNow()
	tests := []struct {
		name   string
		policy PlacementRankingPolicy
		mutate func(*PlacementSourceCandidate, *PlacementSourceCandidate)
		input  func(PlacementRanking) float64
		want   float64
	}{
		{
			name: "capacity", policy: PlacementRankingPolicy{CapacityWeight: 1}, want: .9,
			mutate: func(a, b *PlacementSourceCandidate) { a.Capacity.Value, b.Capacity.Value = .9, .1 },
			input:  func(r PlacementRanking) float64 { return r.Capacity.Value },
		},
		{
			name: "queue", policy: PlacementRankingPolicy{QueueWeight: 1, QueueScale: time.Minute}, want: 1,
			mutate: func(a, b *PlacementSourceCandidate) { a.QueueSeconds.Value, b.QueueSeconds.Value = 1, 50 },
			input:  func(r PlacementRanking) float64 { return r.QueueSeconds.Value },
		},
		{
			name: "price", policy: PlacementRankingPolicy{PriceWeight: 1, PriceScale: 10}, want: .1,
			mutate: func(a, b *PlacementSourceCandidate) { a.PricePerMTok.Value, b.PricePerMTok.Value = .1, 9 },
			input:  func(r PlacementRanking) float64 { return r.PricePerMTok.Value },
		},
		{
			name: "carbon", policy: PlacementRankingPolicy{CarbonWeight: 1, CarbonScale: 100}, want: 1,
			mutate: func(a, b *PlacementSourceCandidate) { a.CarbonGrams.Value, b.CarbonGrams.Value = 1, 90 },
			input:  func(r PlacementRanking) float64 { return r.CarbonGrams.Value },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preferred := placementCandidateFixture("preferred", PlacementCandidateDevice, now)
			other := placementCandidateFixture("other", PlacementCandidateFleet, now)
			tt.mutate(&preferred, &other)
			adapter := newPlacementFixtureAdapter(placementSnapshotFixture(now, other, preferred), now, nil, nil, PlacementAdapterOptions{})
			observation, err := adapter.Observe(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			request := placementRequestFixture()
			request.RankingPolicy = tt.policy
			plan := adapter.Plan(observation, request)
			preferredID := stablePlacementID("candidate", preferred.Identity)
			if plan.Selection.CandidateID != preferredID {
				t.Fatalf("selection = %+v, want preferred candidate", plan.Selection)
			}
			for _, verdict := range plan.Candidates {
				if verdict.CandidateID == preferredID {
					if verdict.Ranking == nil || tt.input(*verdict.Ranking) != tt.want {
						t.Fatalf("ranking did not capture %s input: %+v", tt.name, verdict.Ranking)
					}
					return
				}
			}
			t.Fatalf("preferred candidate %q missing from plan", preferredID)
		})
	}
}

func TestPlacementAdapterRanksLocalityAndPressureDeterministically(t *testing.T) {
	now := placementTestNow()
	local := placementCandidateFixture("local", PlacementCandidateDevice, now)
	local.CacheLocality.Value = 1
	local.ServePressure.Value = .1
	remote := placementCandidateFixture("remote", PlacementCandidateFleet, now)
	remote.CacheLocality.Value = .1
	remote.ServePressure.Value = .9
	adapter := newPlacementFixtureAdapter(placementSnapshotFixture(now, remote, local), now, nil, nil, PlacementAdapterOptions{})
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	request := placementRequestFixture()
	request.RankingPolicy = PlacementRankingPolicy{
		LocalityWeight: 1,
		PressureWeight: 1,
		QueueScale:     time.Second,
		PriceScale:     1,
		CarbonScale:    1,
	}
	plan := adapter.Plan(observation, request)
	if plan.Selection.CandidateID != stablePlacementID("candidate", local.Identity) {
		t.Fatalf("local low-pressure candidate not selected: %+v", plan.Selection)
	}

	tieDevice := placementCandidateFixture("tie-device", PlacementCandidateDevice, now)
	tieFleet := placementCandidateFixture("tie-fleet", PlacementCandidateFleet, now)
	tieAdapter := newPlacementFixtureAdapter(placementSnapshotFixture(now, tieFleet, tieDevice), now, nil, nil, PlacementAdapterOptions{})
	tieObservation, err := tieAdapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	tiePlan := tieAdapter.Plan(tieObservation, request)
	if tiePlan.Selection.CandidateID != stablePlacementID("candidate", tieDevice.Identity) {
		t.Fatalf("kind tie-break is not deterministic: %+v", tiePlan.Selection)
	}
}

func TestPlacementAdapterPinOverridesSoftScoreButNotHardConstraints(t *testing.T) {
	now := placementTestNow()
	best := placementCandidateFixture("best", PlacementCandidateDevice, now)
	best.CacheLocality.Value = 1
	best.ServePressure.Value = 0
	pinned := placementCandidateFixture("pinned", PlacementCandidateFleet, now)
	pinned.CacheLocality.Value = 0
	pinned.ServePressure.Value = 1
	adapter := newPlacementFixtureAdapter(placementSnapshotFixture(now, best, pinned), now, nil, nil, PlacementAdapterOptions{})
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	request := placementRequestFixture()
	request.OperatorPin = append([]byte(nil), pinned.Identity...)
	plan := adapter.Plan(observation, request)
	if plan.Selection.CandidateID != stablePlacementID("candidate", pinned.Identity) || !plan.Selection.OperatorPinned {
		t.Fatalf("eligible pin did not override score: %+v", plan)
	}

	constrained := pinned
	constrained.DataBoundary = PlacementBoundaryExternal
	constrainedAdapter := newPlacementFixtureAdapter(placementSnapshotFixture(now, best, constrained), now, nil, nil, PlacementAdapterOptions{})
	constrainedObservation, err := constrainedAdapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	request.MaximumDataBoundary = PlacementBoundaryOrganization
	constrainedPlan := constrainedAdapter.Plan(constrainedObservation, request)
	if constrainedPlan.UnavailableReason != PlacementUnavailablePinConstrained || constrainedPlan.Selection.CandidateID != "" {
		t.Fatalf("hard-constrained pin bypassed policy: %+v", constrainedPlan)
	}
	pinnedID := stablePlacementID("candidate", pinned.Identity)
	for _, verdict := range constrainedPlan.Candidates {
		if verdict.CandidateID == pinnedID && (!verdict.Pinned || verdict.Eligible || verdict.Ranking != nil) {
			t.Fatalf("constrained pin was scored: %+v", verdict)
		}
	}
}

func TestPlacementAdapterDisabledDelegatesTo5416Fallback(t *testing.T) {
	now := placementTestNow()
	request := placementRequestFixture()
	fallback := &recordingPlacementFallback{}
	applier := &recordingPlacementApplier{}
	adapter := NewPlacementAdapter(nil, applier, fallback, PlacementAdapterOptions{
		Disabled: true,
		Now:      func() time.Time { return now },
	})
	plan := adapter.Plan(PlacementObservation{}, request)
	if !plan.UsesFallback || plan.ObservationID != "" || plan.Selection.CandidateID != "" {
		t.Fatalf("disabled plan did not preserve the fallback boundary: %+v", plan)
	}
	result := adapter.Apply(context.Background(), plan)
	if result.Status != PlacementApplyAccepted || !result.UsedFallback || fallback.calls != 1 {
		t.Fatalf("fallback was not applied: calls=%d result=%+v", fallback.calls, result)
	}
	if len(applier.calls) != 0 {
		t.Fatalf("coordination applier called while disabled: %d", len(applier.calls))
	}
	if !reflect.DeepEqual(fallback.request, request) || fallback.lease != plan.Lease {
		t.Fatalf("fallback did not receive the original request and lease: request=%+v lease=%+v", fallback.request, fallback.lease)
	}
}

func TestPlacementAdapterApplyUsesLeaseAndIdempotencyStatuses(t *testing.T) {
	now := placementTestNow()
	snapshot := placementSnapshotFixture(now, placementCandidateFixture("device", PlacementCandidateDevice, now))
	applier := &recordingPlacementApplier{}
	adapter := newPlacementFixtureAdapter(snapshot, now, applier, nil, PlacementAdapterOptions{})
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	plan := adapter.Plan(observation, placementRequestFixture())
	accepted := adapter.Apply(context.Background(), plan)
	replayed := adapter.Apply(context.Background(), plan)
	if accepted.Status != PlacementApplyAccepted || accepted.Replayed || replayed.Status != PlacementApplyAccepted || !replayed.Replayed {
		t.Fatalf("accepted/replayed results = %+v / %+v", accepted, replayed)
	}
	if len(applier.calls) != 1 || applier.calls[0].Lease != plan.Lease || applier.calls[0].PlanID != plan.ID {
		t.Fatalf("lease/idempotency seam mismatch: calls=%+v plan=%+v", applier.calls, plan)
	}

	supersededApplier := &recordingPlacementApplier{}
	supersededAdapter := newPlacementFixtureAdapter(snapshot, now, supersededApplier, nil, PlacementAdapterOptions{})
	supersededObservation, err := supersededAdapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	oldPlan := supersededAdapter.Plan(supersededObservation, placementRequestFixture())
	newRequest := placementRequestFixture()
	newRequest.OperatorPin = []byte("device")
	newPlan := supersededAdapter.Plan(supersededObservation, newRequest)
	if oldPlan.Lease.ID == newPlan.Lease.ID {
		t.Fatal("fixture did not create a newer distinct lease")
	}
	if result := supersededAdapter.Apply(context.Background(), oldPlan); result.Status != PlacementApplySuperseded {
		t.Fatalf("old lease status = %+v", result)
	}
	if result := supersededAdapter.Apply(context.Background(), newPlan); result.Status != PlacementApplyAccepted {
		t.Fatalf("new lease status = %+v", result)
	}

	clock := now
	expiredApplier := &recordingPlacementApplier{}
	expiredAdapter := NewPlacementAdapter(placementObserverFunc(func(context.Context) (PlacementSourceSnapshot, error) {
		return snapshot, nil
	}), expiredApplier, nil, PlacementAdapterOptions{
		LeaseDuration: time.Second,
		Now:           func() time.Time { return clock },
	})
	expiredObservation, err := expiredAdapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	expiredPlan := expiredAdapter.Plan(expiredObservation, placementRequestFixture())
	clock = expiredPlan.Lease.ValidUntil
	if result := expiredAdapter.Apply(context.Background(), expiredPlan); result.Status != PlacementApplySuperseded {
		t.Fatalf("expired lease status = %+v", result)
	}
	if len(expiredApplier.calls) != 0 {
		t.Fatalf("expired lease reached applier: %d", len(expiredApplier.calls))
	}

	unavailableAdapter := newPlacementFixtureAdapter(snapshot, now, nil, nil, PlacementAdapterOptions{})
	unavailableObservation, err := unavailableAdapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	unavailablePlan := unavailableAdapter.Plan(unavailableObservation, placementRequestFixture())
	if result := unavailableAdapter.Apply(context.Background(), unavailablePlan); result.Status != PlacementApplyUnavailable {
		t.Fatalf("missing sink status = %+v", result)
	}

	failedApplier := &recordingPlacementApplier{status: PlacementApplyStatus("not-a-status")}
	failedAdapter := newPlacementFixtureAdapter(snapshot, now, failedApplier, nil, PlacementAdapterOptions{})
	failedObservation, err := failedAdapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	failedPlan := failedAdapter.Plan(failedObservation, placementRequestFixture())
	if result := failedAdapter.Apply(context.Background(), failedPlan); result.Status != PlacementApplyFailed {
		t.Fatalf("invalid sink result did not fail closed: %+v", result)
	}
}

func TestPlacementAdapterSelfCheckIsStableConnectedAndContentFree(t *testing.T) {
	now := placementTestNow()
	candidate := placementCandidateFixture("candidate-private-content", PlacementCandidateDevice, now)
	snapshot := placementSnapshotFixture(now, candidate)
	applier := &recordingPlacementApplier{}
	adapter := newPlacementFixtureAdapter(snapshot, now, applier, nil, PlacementAdapterOptions{})
	request := placementRequestFixture()
	request.WorkloadIdentity = []byte("workload prompt private content")
	request.TenantIdentity = []byte("private-tenant")

	first := adapter.SelfCheck(context.Background(), request)
	second := adapter.SelfCheck(context.Background(), request)
	firstJSON, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	secondJSON, err := json.Marshal(second)
	if err != nil {
		t.Fatal(err)
	}
	if string(firstJSON) != string(secondJSON) {
		t.Fatalf("selfcheck is not byte-stable:\n%s\n%s", firstJSON, secondJSON)
	}
	if first.Failure != "" || len(first.Trace) != 3 || first.Apply.Status != PlacementApplyAccepted {
		t.Fatalf("selfcheck failed: %+v", first)
	}
	if first.Plan.ObservationID != first.Observation.ID || first.Apply.PlanID != first.Plan.ID ||
		first.Trace[0].Phase != PlacementTraceObserve || first.Trace[1].Phase != PlacementTracePlan || first.Trace[2].Phase != PlacementTraceApply {
		t.Fatalf("selfcheck references are disconnected: %+v", first)
	}
	if len(applier.calls) != 1 {
		t.Fatalf("stable selfcheck repeated an idempotent effect: %d", len(applier.calls))
	}
	for _, secret := range []string{"workload prompt private content", "private-tenant", "candidate-private-content"} {
		if strings.Contains(string(firstJSON), secret) {
			t.Fatalf("selfcheck leaked %q: %s", secret, firstJSON)
		}
	}
}

func placementTestNow() time.Time {
	return time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
}

func placementCandidateFixture(identity string, kind PlacementCandidateKind, now time.Time) PlacementSourceCandidate {
	metric := func(value float64) PlacementMetric {
		return PlacementMetric{
			Value: value, Provenance: PlacementMeasured,
			ObservedAt: now.Add(-10 * time.Second), FreshUntil: now.Add(2 * time.Minute),
		}
	}
	return PlacementSourceCandidate{
		Identity:        []byte(identity),
		Generation:      7,
		Kind:            kind,
		Region:          "us-west",
		FailureDomain:   "zone-a",
		DataBoundary:    PlacementBoundaryOrganization,
		EligibleTenants: [][]byte{[]byte("private-tenant")},
		Accelerators:    []string{"cuda"},
		Models:          []string{"qwen3.8"},
		Available:       true,
		Provenance:      PlacementMeasured,
		ObservedAt:      now.Add(-10 * time.Second),
		FreshUntil:      now.Add(2 * time.Minute),
		CacheLocality:   metric(.6),
		ServePressure:   metric(.3),
		Capacity:        metric(.7),
		QueueSeconds:    metric(2),
		PricePerMTok:    metric(1),
		CarbonGrams:     metric(5),
	}
}

func placementSnapshotFixture(now time.Time, candidates ...PlacementSourceCandidate) PlacementSourceSnapshot {
	return PlacementSourceSnapshot{Generation: 11, CapturedAt: now.Add(-time.Second), Candidates: candidates}
}

func placementRequestFixture() PlacementRequest {
	return PlacementRequest{
		WorkloadIdentity:    []byte("workload"),
		TenantIdentity:      []byte("private-tenant"),
		Region:              "us-west",
		Accelerator:         "cuda",
		Model:               "qwen3.8",
		MaximumDataBoundary: PlacementBoundaryOrganization,
	}
}

func newPlacementFixtureAdapter(snapshot PlacementSourceSnapshot, now time.Time, applier PlacementApplier, fallback PlacementPolicyFallback, options PlacementAdapterOptions) *PlacementAdapter {
	options.Now = func() time.Time { return now }
	return NewPlacementAdapter(placementObserverFunc(func(context.Context) (PlacementSourceSnapshot, error) {
		return snapshot, nil
	}), applier, fallback, options)
}
