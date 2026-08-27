package gateway

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type decodeFootprintTestPlanner struct {
	worker           string
	completionTokens int
	err              error
	started          chan struct{}
	wait             <-chan struct{}
	startOnce        sync.Once
	calls            atomic.Int32
}

func (p *decodeFootprintTestPlanner) Model() string { return "Qwen3.8" }

func (p *decodeFootprintTestPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls.Add(1)
	if p.started != nil {
		p.startOnce.Do(func() { close(p.started) })
	}
	if p.wait != nil {
		select {
		case <-p.wait:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if p.err != nil {
		return nil, p.err
	}
	return &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: p.worker},
		Model:   "Qwen3.8",
		Usage:   agent.Usage{CompletionTokens: p.completionTokens},
	}, nil
}

type decodeFootprintStreamPlanner struct {
	worker           string
	completionTokens int
}

func (p *decodeFootprintStreamPlanner) Model() string            { return "Qwen3.8" }
func (p *decodeFootprintStreamPlanner) StreamingSupported() bool { return true }
func (p *decodeFootprintStreamPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, errors.New("buffered path was not expected")
}
func (p *decodeFootprintStreamPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if sink != nil {
		if err := sink("native "); err != nil {
			return nil, err
		}
		if err := sink("stream"); err != nil {
			return nil, err
		}
	}
	return &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: "native stream"},
		Model:   "Qwen3.8",
		Usage:   agent.Usage{CompletionTokens: p.completionTokens},
	}, nil
}

func decodeFootprintRouter(t *testing.T, membership *FleetMembership, policy *CacheAwarePolicy, planners ...agent.Planner) *ReplicaRouter {
	t.Helper()
	replicas := make([]PlannerReplica, len(planners))
	for i, planner := range planners {
		worker := planner.(*decodeFootprintTestPlanner).worker
		replicas[i] = PlannerReplica{Name: worker, Planner: planner}
	}
	router, err := NewReplicaRouter("Qwen3.8", replicas)
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	return router.WithMembership(membership).WithPickPolicy(policy)
}

func TestDecodeFootprintLiveRouteAvoidsMateriallyLargerBooking(t *testing.T) {
	membership := nativeReservationFleet(t, "w1", "w2")
	policy := NewCacheAwarePolicy(NewPrefixResidencyIndex(8), DefaultSkewThreshold())
	messages := []agent.Message{{Role: agent.RoleSystem, Content: "same-prefix"}}
	prefix := prefixSegments(messages)
	policy.Index().Observe("w1", prefix)
	policy.Index().Observe("w2", prefix)

	unblock := make(chan struct{})
	long := &decodeFootprintTestPlanner{worker: "w1", completionTokens: 2048, started: make(chan struct{}), wait: unblock}
	short := &decodeFootprintTestPlanner{worker: "w2", completionTokens: 32}
	router := decodeFootprintRouter(t, membership, policy, long, short)

	longDone := make(chan error, 1)
	go func() {
		_, err := router.Complete(context.Background(), messages, nil, agent.WithMaxTokens(4096))
		longDone <- err
	}()
	<-long.started

	const longBlocks = 4096 / 16
	if status, ok := fleetWorkerStatus(membership, "w1"); !ok || status.Inflight != 1 || status.BookedOutputBlocks != longBlocks {
		t.Fatalf("long booking status = %+v present=%v, want inflight=1 booked=%d", status, ok, longBlocks)
	}
	var metrics strings.Builder
	NewFleetMembershipMetrics().Render(&metrics, membership.Snapshot())
	if want := `fak_gateway_fleet_worker_booked_output_blocks{worker="w1"} 256`; !strings.Contains(metrics.String(), want) {
		t.Fatalf("fleet metrics missing live booked output %q\n%s", want, metrics.String())
	}
	// Equalize request count. Both workers also hold the same prefix; only the
	// anticipated output booking should distinguish their live route scores.
	if err := membership.Acquire("w2"); err != nil {
		t.Fatalf("sentinel Acquire(w2): %v", err)
	}
	got, err := router.Complete(context.Background(), messages, nil, agent.WithMaxTokens(64))
	if err != nil {
		t.Fatalf("short Complete: %v", err)
	}
	if got.Message.Content != "w2" {
		t.Fatalf("equal overlap/count routed to %q, want w2 away from long booking", got.Message.Content)
	}

	decisions := policy.DecodeFootprintDecisions()
	if len(decisions) != 2 {
		t.Fatalf("decision journal length = %d, want 2", len(decisions))
	}
	second := decisions[1]
	if len(second.Candidates) != 2 || second.Candidates[0].BaseLoad != second.Candidates[1].BaseLoad {
		t.Fatalf("candidate base loads were not equal: %+v", second.Candidates)
	}
	var w1Score, w2Score DecodeFootprintScore
	for _, score := range second.Candidates {
		if score.Worker == "w1" {
			w1Score = score
		} else if score.Worker == "w2" {
			w2Score = score
		}
	}
	if w1Score.BookedOutputBlocks != longBlocks || w2Score.BookedOutputBlocks != 0 || !(w2Score.Score > w1Score.Score) {
		t.Fatalf("booked-output score did not steer: w1=%+v w2=%+v", w1Score, w2Score)
	}
	if active := policy.DecodeFootprintActiveByWorker(); active["w1"] != longBlocks || active["w2"] != 0 {
		t.Fatalf("active bookings after short completion = %+v, want only w1=%d", active, longBlocks)
	}

	close(unblock)
	if err := <-longDone; err != nil {
		t.Fatalf("long Complete: %v", err)
	}
	membership.Release("w2")
	assertReservationFleetClean(t, membership)
	if active := policy.DecodeFootprintActiveByWorker(); len(active) != 0 {
		t.Fatalf("active bookings leaked after completion: %+v", active)
	}
}

func TestDecodeFootprintSharedMembershipIsVisibleAcrossPolicies(t *testing.T) {
	membership := nativeReservationFleet(t, "w1", "w2")
	messages := []agent.Message{{Role: agent.RoleSystem, Content: "shared-fleet-prefix"}}
	prefix := prefixSegments(messages)
	policyA := NewCacheAwarePolicy(NewPrefixResidencyIndex(8), DefaultSkewThreshold())
	policyB := NewCacheAwarePolicy(NewPrefixResidencyIndex(8), DefaultSkewThreshold())
	for _, policy := range []*CacheAwarePolicy{policyA, policyB} {
		policy.Index().Observe("w1", prefix)
		policy.Index().Observe("w2", prefix)
	}

	unblock := make(chan struct{})
	aLong := &decodeFootprintTestPlanner{worker: "w1", completionTokens: 100, started: make(chan struct{}), wait: unblock}
	routerA := decodeFootprintRouter(t, membership, policyA, aLong, &decodeFootprintTestPlanner{worker: "w2"})
	routerB := decodeFootprintRouter(t, membership, policyB, &decodeFootprintTestPlanner{worker: "w1"}, &decodeFootprintTestPlanner{worker: "w2", completionTokens: 8})
	done := make(chan error, 1)
	go func() {
		_, err := routerA.Complete(context.Background(), messages, nil, agent.WithMaxTokens(4096))
		done <- err
	}()
	<-aLong.started
	if err := membership.Acquire("w2"); err != nil {
		t.Fatalf("sentinel Acquire(w2): %v", err)
	}

	got, err := routerB.Complete(context.Background(), messages, nil, agent.WithMaxTokens(64))
	if err != nil {
		t.Fatalf("router B Complete: %v", err)
	}
	if got.Message.Content != "w2" {
		t.Fatalf("second policy ignored shared fleet booking and chose %q, want w2", got.Message.Content)
	}
	bDecision := policyB.DecodeFootprintDecisions()[0]
	var sharedSeen int
	for _, score := range bDecision.InitialCandidates {
		if score.Worker == "w1" {
			sharedSeen = score.BookedOutputBlocks
		}
	}
	if sharedSeen != 256 {
		t.Fatalf("policy B score saw w1 booked blocks=%d, want shared membership value 256: %+v", sharedSeen, bDecision)
	}
	close(unblock)
	if err := <-done; err != nil {
		t.Fatalf("router A Complete: %v", err)
	}
	membership.Release("w2")
	assertReservationFleetClean(t, membership)
}

func TestDecodeFootprintUnknownCapIdentityAndPredictionErrorObserved(t *testing.T) {
	membership := nativeReservationFleet(t, "w1")
	policy := NewCacheAwarePolicy(nil, DefaultSkewThreshold())
	planner := &decodeFootprintTestPlanner{worker: "w1", completionTokens: 40}
	router := decodeFootprintRouter(t, membership, policy, planner)
	messages := []agent.Message{{Role: agent.RoleUser, Content: "bounded"}}

	if _, err := router.Complete(context.Background(), messages, nil); err != nil {
		t.Fatalf("unknown Complete: %v", err)
	}
	planner.completionTokens = 5000
	const huge = 1 << 20
	if _, err := router.Complete(context.Background(), messages, nil, agent.WithMaxTokens(huge)); err != nil {
		t.Fatalf("capped Complete: %v", err)
	}

	decisions := policy.DecodeFootprintDecisions()
	if len(decisions) != 2 {
		t.Fatalf("decision journal length = %d, want 2", len(decisions))
	}
	unknown, capped := decisions[0], decisions[1]
	if !unknown.UnknownDefault || unknown.RequestedOutputTokens != 0 || unknown.ExpectedOutputTokens != 256 || unknown.OutputCapTokens != 4096 {
		t.Fatalf("unknown-output decision is not conservative and bounded: %+v", unknown)
	}
	if unknown.Engine != TurnIngressEngine || unknown.Model != "Qwen3.8" {
		t.Fatalf("native execution identity changed: engine=%q model=%q", unknown.Engine, unknown.Model)
	}
	if !unknown.Reconciled || unknown.ObservedOutputTokens != 40 || unknown.PredictionErrorTokens != 40-256 || !unknown.Released || unknown.ReleaseCount != 1 {
		t.Fatalf("unknown decision lifecycle = %+v", unknown)
	}
	if !capped.Capped || capped.RequestedOutputTokens != huge || capped.ExpectedOutputTokens != 4096 || capped.BookedOutputBlocks != 256 {
		t.Fatalf("large hint did not expose its cap: %+v", capped)
	}
	if !capped.Reconciled || capped.PredictionErrorTokens != 5000-4096 || capped.ReleaseCount != 1 {
		t.Fatalf("capped prediction reconciliation = %+v", capped)
	}
	stats := policy.DecodeFootprintStats()
	if stats.Routes != 2 || stats.UnknownDefaults != 1 || stats.CappedRoutes != 1 || stats.CappedOutputTokens != huge-4096 || stats.Reconciliations != 2 || stats.Releases != 2 || stats.ActiveBookings != 0 || stats.ActiveBookedOutputBlocks != 0 {
		t.Fatalf("decode footprint meters = %+v", stats)
	}
	if stats.PredictionErrorTokens != int64((40-256)+(5000-4096)) || stats.AbsolutePredictionErrorTokens != uint64((256-40)+(5000-4096)) {
		t.Fatalf("prediction-error meters = %+v", stats)
	}
	assertReservationFleetClean(t, membership)
}

func TestDecodeFootprintRetryMovesOneLogicalBookingWithoutInflation(t *testing.T) {
	membership := nativeReservationFleet(t, "w1", "w2")
	policy := NewCacheAwarePolicy(nil, DefaultSkewThreshold())
	failed := &decodeFootprintTestPlanner{worker: "w1", err: errReservationTestEndpoint}
	succeeded := &decodeFootprintTestPlanner{worker: "w2", completionTokens: 40}
	router := decodeFootprintRouter(t, membership, policy, failed, succeeded)

	got, err := router.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "retry"}}, nil, agent.WithMaxTokens(320))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got.Message.Content != "w2" || failed.calls.Load() != 1 || succeeded.calls.Load() != 1 {
		t.Fatalf("retry result=%q calls=%d/%d, want w2 and 1/1", got.Message.Content, failed.calls.Load(), succeeded.calls.Load())
	}
	decisions := policy.DecodeFootprintDecisions()
	if len(decisions) != 1 {
		t.Fatalf("retry created %d logical route records, want 1", len(decisions))
	}
	decision := decisions[0]
	if decision.InitialWorker != "w1" || decision.Worker != "w2" || decision.Retargets != 1 || decision.BookedOutputBlocks != 20 {
		t.Fatalf("booking was not moved once: %+v", decision)
	}
	if len(decision.InitialCandidates) != 2 || len(decision.Candidates) != 1 || decision.InitialSelectedScore < 0 {
		t.Fatalf("retry journal lost initial/current score evidence: %+v", decision)
	}
	if decision.ObservedOutputTokens != 40 || decision.PredictionErrorTokens != 40-320 || !decision.Reconciled || decision.ReleaseCount != 1 {
		t.Fatalf("retry inflated or skipped reconciliation: %+v", decision)
	}
	if decision.Engine != TurnIngressEngine || decision.Model != "Qwen3.8" {
		t.Fatalf("retry crossed native identity: %+v", decision)
	}
	stats := policy.DecodeFootprintStats()
	if stats.Routes != 1 || stats.Retargets != 1 || stats.Reconciliations != 1 || stats.Releases != 1 || stats.ActiveBookings != 0 {
		t.Fatalf("retry lifecycle meters = %+v", stats)
	}
	assertReservationFleetClean(t, membership)
}

func TestDecodeFootprintConfigurationHasHardGlobalBounds(t *testing.T) {
	membership := nativeReservationFleet(t, "w1")
	policy := NewCacheAwarePolicy(nil, DefaultSkewThreshold()).WithDecodeFootprintConfig(DecodeFootprintConfig{
		BlockTokens:         16,
		UnknownOutputTokens: maxIntValue(),
		MaxOutputTokens:     maxIntValue(),
		Scale:               1,
		JournalCapacity:     maxIntValue(),
	})
	router := decodeFootprintRouter(t, membership, policy, &decodeFootprintTestPlanner{worker: "w1", completionTokens: 1})
	if _, err := router.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "hard-bound"}}, nil, agent.WithMaxTokens(maxIntValue())); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	decision := policy.DecodeFootprintDecisions()[0]
	if decision.OutputCapTokens != maxDecodeFootprintOutputTokens || decision.ExpectedOutputTokens != maxDecodeFootprintOutputTokens || decision.BookedOutputBlocks <= 0 || !decision.Capped {
		t.Fatalf("configured projection escaped hard envelope or overflowed free: %+v", decision)
	}
	if stats := policy.DecodeFootprintStats(); stats.CappedOutputTokens != uint64(maxIntValue()-maxDecodeFootprintOutputTokens) {
		t.Fatalf("configured cap meter = %+v", stats)
	}
	assertReservationFleetClean(t, membership)
}

func TestDecodeFootprintCancellationReleasesExactlyOnce(t *testing.T) {
	membership := nativeReservationFleet(t, "w1")
	policy := NewCacheAwarePolicy(nil, DefaultSkewThreshold())
	planner := &decodeFootprintTestPlanner{worker: "w1", started: make(chan struct{}), wait: make(chan struct{})}
	router := decodeFootprintRouter(t, membership, policy, planner)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := router.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: "cancel"}}, nil, agent.WithMaxTokens(512))
		done <- err
	}()
	<-planner.started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("Complete error = %v, want context.Canceled", err)
	}
	decisions := policy.DecodeFootprintDecisions()
	if len(decisions) != 1 || decisions[0].ReleaseReason != "cancellation" || decisions[0].ReleaseCount != 1 || decisions[0].Reconciled {
		t.Fatalf("cancellation lifecycle = %+v", decisions)
	}
	stats := policy.DecodeFootprintStats()
	if stats.Releases != 1 || stats.CancellationReleases != 1 || stats.Reconciliations != 0 || stats.ActiveBookings != 0 {
		t.Fatalf("cancellation meters = %+v", stats)
	}
	assertReservationFleetClean(t, membership)
}

func TestDecodeFootprintStreamCompletionReconcilesThenReleasesExactlyOnce(t *testing.T) {
	membership := nativeReservationFleet(t, "w1")
	policy := NewCacheAwarePolicy(nil, DefaultSkewThreshold())
	planner := &decodeFootprintStreamPlanner{worker: "w1", completionTokens: 7}
	router, err := NewReplicaRouter("Qwen3.8", []PlannerReplica{{Name: "w1", Planner: planner}})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	router.WithMembership(membership).WithPickPolicy(policy)
	var streamed string
	comp, err := router.CompleteStream(context.Background(), func(fragment string) error {
		streamed += fragment
		return nil
	}, []agent.Message{{Role: agent.RoleUser, Content: "stream"}}, nil, agent.WithMaxTokens(64))
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if streamed != "native stream" || comp.Usage.CompletionTokens != 7 {
		t.Fatalf("stream=%q usage=%+v", streamed, comp.Usage)
	}
	decisions := policy.DecodeFootprintDecisions()
	if len(decisions) != 1 || !decisions[0].Reconciled || decisions[0].ObservedOutputTokens != 7 || decisions[0].ReleaseReason != "stream_completion" || decisions[0].ReleaseCount != 1 {
		t.Fatalf("stream lifecycle = %+v", decisions)
	}
	stats := policy.DecodeFootprintStats()
	if stats.Reconciliations != 1 || stats.Releases != 1 || stats.StreamCompletionReleases != 1 || stats.ActiveBookings != 0 {
		t.Fatalf("stream meters = %+v", stats)
	}
	assertReservationFleetClean(t, membership)
}

func TestDecodeFootprintNoMembershipHedgeBooksBothPhysicalAttempts(t *testing.T) {
	policy := NewCacheAwarePolicy(nil, DefaultSkewThreshold())
	primary := &decodeFootprintTestPlanner{worker: "w1", started: make(chan struct{}), wait: make(chan struct{})}
	alternate := &decodeFootprintTestPlanner{worker: "w2", completionTokens: 11}
	router, err := NewReplicaRouter("Qwen3.8", []PlannerReplica{
		{Name: "w1", Planner: primary},
		{Name: "w2", Planner: alternate},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	router.WithPickPolicy(policy)
	router.Hedge = eligibleHedgePolicy(nil)
	got, err := router.Complete(
		context.Background(),
		[]agent.Message{{Role: agent.RoleUser, Content: "hedge"}},
		nil,
		agent.WithMaxTokens(128),
		zeroTemperatureOpt(),
	)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got.Message.Content != "w2" {
		t.Fatalf("hedge winner = %q, want w2", got.Message.Content)
	}
	decisions := policy.DecodeFootprintDecisions()
	if len(decisions) != 2 {
		t.Fatalf("hedge physical booking records = %d, want 2", len(decisions))
	}
	for _, decision := range decisions {
		if decision.RequestedOutputTokens != 128 || decision.BookedOutputBlocks != 8 || decision.ReleaseCount != 1 {
			t.Fatalf("hedge booking did not carry opts/release once: %+v", decision)
		}
	}
	stats := policy.DecodeFootprintStats()
	if stats.Routes != 2 || stats.Releases != 2 || stats.Reconciliations != 1 || stats.ActiveBookings != 0 || stats.ActiveBookedOutputBlocks != 0 {
		t.Fatalf("hedge lifecycle meters = %+v", stats)
	}
	if active := policy.DecodeFootprintActiveByWorker(); len(active) != 0 {
		t.Fatalf("hedge bookings leaked: %+v", active)
	}
}
