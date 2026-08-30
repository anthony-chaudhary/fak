package coordination

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

type contextObserverFunc func(context.Context) (ContextSourceSnapshot, error)

func (f contextObserverFunc) SnapshotContext(ctx context.Context) (ContextSourceSnapshot, error) {
	return f(ctx)
}

type recordingContextApplier struct {
	calls  []ContextAction
	result ContextActionResult
}

func (a *recordingContextApplier) ApplyContextAction(_ context.Context, action ContextAction) ContextActionResult {
	a.calls = append(a.calls, action)
	result := a.result
	if result.Status == "" {
		result.Status = ContextActionApplied
		result.AppliedBytes = action.Bytes
	}
	return result
}

type recordingContextFallback struct {
	calls int
}

func (f *recordingContextFallback) ApplyExistingContextPolicy(_ context.Context, plan ContextPlan) ContextApplyResult {
	f.calls++
	return ContextApplyResult{PlanID: plan.ID, Status: ContextApplyApplied}
}

func TestStableCoordinationIDContract(t *testing.T) {
	for _, prefix := range []string{"ctx", "candidate"} {
		want := prefix + "_689f6a627384c7dcb2dcc1487e540223"
		if got := stableCoordinationID(prefix, []byte("identity")); got != want {
			t.Fatalf("stableCoordinationID(%q) = %q, want %q", prefix, got, want)
		}
		if !validStableCoordinationID(want, prefix) {
			t.Fatalf("validStableCoordinationID rejected %q", want)
		}
	}

	for _, value := range []string{
		"ctx_689f6a627384c7dcb2dcc1487e54022",
		"ctx_689f6a627384c7dcb2dcc1487e5402230",
		"ctx_689f6a627384c7dcb2dcc1487e54022z",
		"candidate_689f6a627384c7dcb2dcc1487e540223",
	} {
		if validStableCoordinationID(value, "ctx") {
			t.Fatalf("validStableCoordinationID accepted %q", value)
		}
	}
}

func TestContextAdapterWarmRemotePrefixWinsUntilShortHorizon(t *testing.T) {
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	snapshot := contextFixture(now)
	adapter := newFixtureAdapter(snapshot, now, &recordingContextApplier{}, nil, ContextAdapterOptions{})

	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatalf("Observe() error = %v", err)
	}
	local := locationOfKind(t, observation, ContextLocationLocal)
	input := validContextInput()

	longReuse := adapter.Plan(input, observation, ContextPlanRequest{
		ResourceKind: ContextResourcePrefix,
		Target:       local,
		ReuseHorizon: 30 * time.Second,
	})
	if got := longReuse.Actions[0].Kind; got != ContextActionTransfer {
		t.Fatalf("long-horizon action = %q, want %q", got, ContextActionTransfer)
	}
	if got := longReuse.Reference.Location.Kind; got != ContextLocationRemote {
		t.Fatalf("long-horizon source = %q, want remote", got)
	}
	if !longReuse.Coordination.ContextReuse || longReuse.ProjectedContext.ReusablePrefixBytes != 8192 {
		t.Fatalf("long-horizon context projection = %+v, plan reuse = %v", longReuse.ProjectedContext, longReuse.Coordination.ContextReuse)
	}
	if longReuse.Reference.ObservationID != observation.ID || longReuse.Actions[0].ObservationID != observation.ID {
		t.Fatal("plan and action do not reference their observation")
	}

	shortReuse := adapter.Plan(input, observation, ContextPlanRequest{
		ResourceKind: ContextResourcePrefix,
		Target:       local,
		ReuseHorizon: 3 * time.Second,
	})
	if got := shortReuse.Actions[0].Kind; got != ContextActionPrefetch {
		t.Fatalf("short-horizon action = %q, want %q", got, ContextActionPrefetch)
	}
	if got := shortReuse.Reference.Location.Kind; got != ContextLocationLocal {
		t.Fatalf("short-horizon source = %q, want local", got)
	}
	if shortReuse.Coordination.ContextReuse || shortReuse.ProjectedContext.ReusablePrefixBytes != 0 {
		t.Fatalf("cold local placement incorrectly reported reusable: %+v", shortReuse.ProjectedContext)
	}
}

func TestContextAdapterObservationIsStableReadOnlyAndContentFree(t *testing.T) {
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	snapshot := contextFixture(now)
	privateResource := string(snapshot.Residencies[0].Identity)
	privateLocation := string(snapshot.Residencies[0].LocationIdentity)
	observerCalls := 0
	applier := &recordingContextApplier{}
	observer := contextObserverFunc(func(context.Context) (ContextSourceSnapshot, error) {
		observerCalls++
		return snapshot, nil
	})
	adapter := NewContextAdapter(observer, applier, nil, ContextAdapterOptions{Now: func() time.Time { return now }})

	first, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	second, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID {
		t.Fatalf("stable observation ID changed: %q != %q", first.ID, second.ID)
	}
	if first.Residencies[0].StableID != first.Residencies[1].StableID {
		t.Fatalf("the same resource has placement-dependent IDs: %+v", first.Residencies)
	}
	if observerCalls != 2 || len(applier.calls) != 0 {
		t.Fatalf("Observe mutated state: observer calls %d, apply calls %d", observerCalls, len(applier.calls))
	}

	encoded, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)
	for _, secret := range []string{privateResource, privateLocation, "private-tenant-key", "prompt text"} {
		if strings.Contains(text, secret) {
			t.Fatalf("observation leaked private content %q: %s", secret, text)
		}
	}
	for _, residency := range first.Residencies {
		if residency.StableID == "" || residency.Location.StableID == "" || residency.Bytes != 8192 || residency.Tokens != 2048 {
			t.Fatalf("incomplete public residency metadata: %+v", residency)
		}
		if residency.Age != 10*time.Second || residency.EstimatedReuseHorizon != time.Minute {
			t.Fatalf("wrong age or reuse horizon: %+v", residency)
		}
	}
}

func TestContextAdapterDerivesCurrentFromGenerationAndFreshness(t *testing.T) {
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	base := contextFixture(now).Residencies[0]
	tests := []struct {
		name       string
		capturedAt time.Time
		mutate     func(*ContextSourceResidency)
		want       bool
	}{
		{name: "current", capturedAt: now.Add(-time.Second), want: true},
		{name: "old generation", capturedAt: now.Add(-time.Second), mutate: func(r *ContextSourceResidency) { r.Generation-- }},
		{name: "expired residency", capturedAt: now.Add(-time.Second), mutate: func(r *ContextSourceResidency) { r.FreshUntil = now }},
		{name: "stale observation", capturedAt: now.Add(-2 * time.Minute)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			record := base
			if test.mutate != nil {
				test.mutate(&record)
			}
			record.ResidentSince = test.capturedAt.Add(-time.Second)
			snapshot := ContextSourceSnapshot{Generation: 9, CapturedAt: test.capturedAt, Managed: true, Pressure: .2, Residencies: []ContextSourceResidency{record}}
			adapter := newFixtureAdapter(snapshot, now, nil, nil, ContextAdapterOptions{MaximumObservationAge: time.Minute})
			observation, err := adapter.Observe(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if got := observation.Residencies[0].Current; got != test.want {
				t.Fatalf("Current = %v, want %v", got, test.want)
			}
		})
	}
}

func TestContextAdapterRejectsStaleCostAndInvalidSourceWithoutLeakingError(t *testing.T) {
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	snapshot := contextFixture(now)
	// The warm remote copy has a cheaper numeric estimate, but it is stale.
	snapshot.Residencies[1].TransferCost.FreshUntil = now
	adapter := newFixtureAdapter(snapshot, now, nil, nil, ContextAdapterOptions{})
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	local := locationOfKind(t, observation, ContextLocationLocal)
	plan := adapter.Plan(validContextInput(), observation, ContextPlanRequest{ResourceKind: ContextResourcePrefix, Target: local, ReuseHorizon: time.Minute})
	if plan.Reference.Location.Kind != ContextLocationLocal {
		t.Fatalf("planner used a stale transfer estimate: %+v", plan.Reference)
	}

	secretError := errors.New("private-tenant-key prompt text")
	failing := NewContextAdapter(contextObserverFunc(func(context.Context) (ContextSourceSnapshot, error) {
		return ContextSourceSnapshot{}, secretError
	}), nil, nil, ContextAdapterOptions{Now: func() time.Time { return now }})
	_, err = failing.Observe(context.Background())
	if !errors.Is(err, ErrContextObservation) || strings.Contains(err.Error(), secretError.Error()) {
		t.Fatalf("source error was not safely collapsed: %v", err)
	}
}

func TestContextAdapterApplyIsTypedIdempotentAndReportsPartialFailure(t *testing.T) {
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	snapshot := contextFixture(now)
	applier := &recordingContextApplier{}
	adapter := newFixtureAdapter(snapshot, now, applier, nil, ContextAdapterOptions{})
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	plan := adapter.Plan(validContextInput(), observation, ContextPlanRequest{
		ResourceKind: ContextResourcePrefix,
		Target:       locationOfKind(t, observation, ContextLocationLocal),
		ReuseHorizon: time.Minute,
	})

	first := adapter.Apply(context.Background(), plan)
	second := adapter.Apply(context.Background(), plan)
	if first.Status != ContextApplyApplied || second.Status != ContextApplyApplied {
		t.Fatalf("apply statuses = %q, %q", first.Status, second.Status)
	}
	if len(applier.calls) != 1 || !second.Outcomes[0].Replayed {
		t.Fatalf("idempotence failure: sink calls %d, replay %+v", len(applier.calls), second.Outcomes)
	}

	partialApplier := &recordingContextApplier{result: ContextActionResult{
		Status: ContextActionPartial, Failure: ContextFailureUnavailable, AppliedBytes: 1024,
	}}
	partialAdapter := newFixtureAdapter(snapshot, now, partialApplier, nil, ContextAdapterOptions{})
	partialPlan := partialAdapter.Plan(validContextInput(), observation, ContextPlanRequest{
		ResourceKind: ContextResourcePrefix,
		Target:       locationOfKind(t, observation, ContextLocationLocal),
		ReuseHorizon: time.Minute,
	})
	partial := partialAdapter.Apply(context.Background(), partialPlan)
	if partial.Status != ContextApplyPartial || partial.Outcomes[0].Status != ContextActionPartial {
		t.Fatalf("partial result lost its type: %+v", partial)
	}

	invalidPlan := partialPlan
	invalidPlan.Actions = append([]ContextAction(nil), partialPlan.Actions...)
	invalidPlan.Actions[0].ID = "invalid-action-id"
	invalidPlan.Actions[0].Kind = ContextActionKind("delete_everything")
	invalid := partialAdapter.Apply(context.Background(), invalidPlan)
	if invalid.Status != ContextApplyFailed || invalid.Outcomes[0].Failure != ContextFailureInvalid || len(partialApplier.calls) != 1 {
		t.Fatalf("untyped action reached sink: result %+v, calls %d", invalid, len(partialApplier.calls))
	}
}

func TestAggregateContextApplyTreatsSkippedAsNeutral(t *testing.T) {
	skipped := ContextActionResult{}
	skipped.Status = ContextActionSkipped
	if got := aggregateContextApply([]ContextActionResult{skipped}); got != ContextApplyNoOp {
		t.Fatalf("skipped-only aggregate = %s, want %s", got, ContextApplyNoOp)
	}
	applied := ContextActionResult{}
	applied.Status = ContextActionApplied
	if got := aggregateContextApply([]ContextActionResult{applied, skipped}); got != ContextApplyApplied {
		t.Fatalf("applied+skipped aggregate = %s, want %s", got, ContextApplyApplied)
	}
}

func TestContextAdapterActionVocabularyIsClosedAndStalePlansFailClosed(t *testing.T) {
	for _, kind := range []ContextActionKind{
		ContextActionPin,
		ContextActionPrefetch,
		ContextActionTransfer,
		ContextActionCompact,
		ContextActionEvict,
		ContextActionNoOp,
	} {
		if !validContextActionKind(kind) {
			t.Errorf("declared action %q is not accepted", kind)
		}
	}
	if validContextActionKind(ContextActionKind("delete")) {
		t.Fatal("action outside the closed vocabulary was accepted")
	}

	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	clock := now
	snapshot := contextFixture(now)
	applier := &recordingContextApplier{}
	adapter := NewContextAdapter(contextObserverFunc(func(context.Context) (ContextSourceSnapshot, error) {
		return snapshot, nil
	}), applier, nil, ContextAdapterOptions{Now: func() time.Time { return clock }})
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	plan := adapter.Plan(validContextInput(), observation, ContextPlanRequest{
		ResourceKind: ContextResourcePrefix,
		Target:       locationOfKind(t, observation, ContextLocationLocal),
		ReuseHorizon: time.Minute,
	})
	clock = now.Add(10 * time.Minute)
	result := adapter.Apply(context.Background(), plan)
	if result.Status != ContextApplyFailed || result.Outcomes[0].Failure != ContextFailureStale || len(applier.calls) != 0 {
		t.Fatalf("stale action did not fail closed: result %+v, calls %d", result, len(applier.calls))
	}
}

func TestContextAdapterForgedPlansAndActionsNeverReachApplier(t *testing.T) {
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	snapshot := contextFixture(now)
	applier := &recordingContextApplier{}
	adapter := newFixtureAdapter(snapshot, now, applier, nil, ContextAdapterOptions{})
	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	plan := adapter.Plan(validContextInput(), observation, ContextPlanRequest{
		ResourceKind: ContextResourcePrefix,
		Target:       locationOfKind(t, observation, ContextLocationLocal),
		ReuseHorizon: time.Minute,
	})

	forgedPlanID := plan
	forgedPlanID.ID = "cplan_00000000000000000000000000000000"
	forgedBytes := plan
	forgedBytes.Actions = append([]ContextAction(nil), plan.Actions...)
	forgedBytes.Actions[0].Bytes++
	forgedTypedAction := plan
	forgedTypedAction.Actions = append([]ContextAction(nil), plan.Actions...)
	forgedTypedAction.Actions[0].Kind = ContextActionCompact
	forgedTypedAction.Actions[0].ID = makeContextActionID(observation.ID, forgedTypedAction.Actions[0])
	forgedTypedAction.ID = makeContextPlanID(forgedTypedAction.Reference, forgedTypedAction.Actions[0])
	forgedLocation := plan
	forgedLocation.Actions = append([]ContextAction(nil), plan.Actions...)
	forgedLocation.Actions[0].Destination = ContextLocation{Kind: ContextLocationKind("private"), StableID: "loc_00000000000000000000000000000000"}
	forgedLocation.Actions[0].ID = makeContextActionID(observation.ID, forgedLocation.Actions[0])
	forgedLocation.ID = makeContextPlanID(forgedLocation.Reference, forgedLocation.Actions[0])

	for name, forged := range map[string]ContextPlan{
		"plan ID":       forgedPlanID,
		"sizes":         forgedBytes,
		"typed action":  forgedTypedAction,
		"location kind": forgedLocation,
	} {
		t.Run(name, func(t *testing.T) {
			result := adapter.Apply(context.Background(), forged)
			if result.Status != ContextApplyFailed || len(result.Outcomes) != 1 || result.Outcomes[0].Failure != ContextFailureInvalid {
				t.Fatalf("forgery did not fail with a typed invalid result: %+v", result)
			}
		})
	}
	if len(applier.calls) != 0 {
		t.Fatalf("forged operation reached applier %d times", len(applier.calls))
	}
}

func TestContextAdapterObservationIDBindsAllPlanRelevantMetadata(t *testing.T) {
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	base := contextFixture(now)
	observeID := func(t *testing.T, snapshot ContextSourceSnapshot) string {
		t.Helper()
		observation, err := newFixtureAdapter(snapshot, now, nil, nil, ContextAdapterOptions{}).Observe(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		return observation.ID
	}
	baseID := observeID(t, base)
	mutations := map[string]func(*ContextSourceSnapshot){
		"residency age": func(snapshot *ContextSourceSnapshot) {
			snapshot.Residencies[0].ResidentSince = snapshot.Residencies[0].ResidentSince.Add(-time.Second)
		},
		"residency freshness": func(snapshot *ContextSourceSnapshot) {
			snapshot.Residencies[0].FreshUntil = snapshot.Residencies[0].FreshUntil.Add(-time.Second)
		},
		"transfer duration": func(snapshot *ContextSourceSnapshot) {
			snapshot.Residencies[0].TransferCost.Duration++
		},
		"cost measurement age": func(snapshot *ContextSourceSnapshot) {
			snapshot.Residencies[0].TransferCost.MeasuredAt = snapshot.Residencies[0].TransferCost.MeasuredAt.Add(-time.Second)
		},
		"cost uncertainty": func(snapshot *ContextSourceSnapshot) {
			snapshot.Residencies[0].RehydrationCost.Uncertainty += .01
		},
		"compaction": func(snapshot *ContextSourceSnapshot) {
			snapshot.Residencies[0].Compaction = ContextCompactionRecommended
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			candidate := base
			candidate.Residencies = append([]ContextSourceResidency(nil), base.Residencies...)
			mutate(&candidate)
			if got := observeID(t, candidate); got == baseID {
				t.Fatalf("observation ID did not bind %s", name)
			}
		})
	}
}

func TestContextAdapterDisabledDelegatesToExistingPolicy(t *testing.T) {
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	fallback := &recordingContextFallback{}
	adapter := newFixtureAdapter(contextFixture(now), now, nil, fallback, ContextAdapterOptions{Disabled: true})
	input := validContextInput()
	input.ContextState = ContextState{Managed: true, ReusablePrefixBytes: 77, Pressure: .4}
	plan := adapter.Plan(input, ContextObservation{}, ContextPlanRequest{})
	if !plan.UsesFallback || plan.ProjectedContext != input.ContextState || !plan.Coordination.ContextReuse {
		t.Fatalf("disabled adapter replaced existing policy input: %+v", plan)
	}
	result := adapter.Apply(context.Background(), plan)
	if fallback.calls != 1 || !result.UsedFallback || result.Status != ContextApplyFallback {
		t.Fatalf("fallback was not used: calls %d, result %+v", fallback.calls, result)
	}
}

func TestContextAdapterSelfCheckCapturesReferencedFlowWithoutSecrets(t *testing.T) {
	now := time.Date(2026, time.August, 26, 12, 0, 0, 0, time.UTC)
	snapshot := contextFixture(now)
	adapter := newFixtureAdapter(snapshot, now, &recordingContextApplier{}, nil, ContextAdapterOptions{})
	input := validContextInput()
	input.HarnessIntent.Task = "prompt text private-tenant-key"
	input.HarnessIntent.Outcome = "private outcome"

	observation, err := adapter.Observe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	check := adapter.SelfCheck(context.Background(), input, ContextPlanRequest{
		ResourceKind: ContextResourcePrefix,
		Target:       locationOfKind(t, observation, ContextLocationLocal),
		ReuseHorizon: time.Minute,
	})
	if check.Failure != ContextFailureNone || len(check.Trace) != 3 {
		t.Fatalf("selfcheck failed: %+v", check)
	}
	if check.Trace[0].Phase != ContextTraceObserve || check.Trace[1].Phase != ContextTracePlan || check.Trace[2].Phase != ContextTraceApply {
		t.Fatalf("wrong selfcheck phases: %+v", check.Trace)
	}
	if check.Plan.Reference.ObservationID != check.Observation.ID || check.Apply.PlanID != check.Plan.ID {
		t.Fatalf("selfcheck references are disconnected: %+v", check)
	}
	encoded, err := json.Marshal(check)
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)
	for _, secret := range []string{"prompt text", "private-tenant-key", "private outcome", string(snapshot.Residencies[0].Identity), string(snapshot.Residencies[0].LocationIdentity)} {
		if strings.Contains(text, secret) {
			t.Fatalf("selfcheck leaked %q: %s", secret, text)
		}
	}
}

func contextFixture(now time.Time) ContextSourceSnapshot {
	identity := []byte("private-tenant-key/prompt text/hash")
	base := ContextSourceResidency{
		Identity:              identity,
		Kind:                  ContextResourcePrefix,
		Generation:            7,
		CurrentGeneration:     7,
		Bytes:                 8192,
		Tokens:                2048,
		ResidentSince:         now.Add(-10 * time.Second),
		FreshUntil:            now.Add(5 * time.Minute),
		EstimatedReuseHorizon: time.Minute,
		Compaction:            ContextCompactionNone,
		TransferCost:          freshCost(now, 4*time.Second, 8192, 0),
		RehydrationCost:       freshCost(now, 8*time.Second, 8192, .25),
	}
	localCold := base
	localCold.LocationKind = ContextLocationLocal
	localCold.LocationIdentity = []byte("private-local-tenant-key")
	localCold.Warm = false
	remoteWarm := base
	remoteWarm.LocationKind = ContextLocationRemote
	remoteWarm.LocationIdentity = []byte("private-remote-tenant-key")
	remoteWarm.Warm = true
	return ContextSourceSnapshot{
		Generation:  11,
		CapturedAt:  now.Add(-time.Second),
		Managed:     true,
		Pressure:    .3,
		Residencies: []ContextSourceResidency{localCold, remoteWarm},
	}
}

func freshCost(now time.Time, duration time.Duration, bytes int64, uncertainty float64) ContextCostEstimate {
	return ContextCostEstimate{
		Duration: duration, Bytes: bytes, Uncertainty: uncertainty,
		MeasuredAt: now.Add(-time.Second), FreshUntil: now.Add(time.Minute),
	}
}

func newFixtureAdapter(snapshot ContextSourceSnapshot, now time.Time, applier ContextActionApplier, fallback ContextPolicyFallback, options ContextAdapterOptions) *ContextAdapter {
	options.Now = func() time.Time { return now }
	return NewContextAdapter(contextObserverFunc(func(context.Context) (ContextSourceSnapshot, error) {
		return snapshot, nil
	}), applier, fallback, options)
}

func locationOfKind(t *testing.T, observation ContextObservation, kind ContextLocationKind) ContextLocation {
	t.Helper()
	for _, residency := range observation.Residencies {
		if residency.Location.Kind == kind {
			return residency.Location
		}
	}
	t.Fatalf("no %q location in observation", kind)
	return ContextLocation{}
}

func validContextInput() Input {
	return Input{
		HarnessIntent: HarnessIntent{Kind: HarnessKindFakNative, Task: "coordinate context", Outcome: "context ready"},
		ComputeState:  ComputeState{Engine: "fak_native", Available: true},
		ServeState:    ServeState{Admitted: true},
	}
}
