package servingsupervision_test

import (
	"context"
	"errors"
	"fmt"
	"net"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/servingsupervision"
)

// TestReplicaLifecycleTransitions validates full replica state progression:
// starting -> ready -> draining -> recovering -> quarantined -> reset,
// as well as error-induced failures during readiness and recovery hooks.
func TestReplicaLifecycleTransitions(t *testing.T) {
	ctx := context.Background()

	t.Run("successful readiness transitions starting to ready", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{
			DomainID:      "rep-life-domain",
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		readyCalled := false
		rep := servingsupervision.NewReplicaSupervisor(
			spec,
			"rep-life-1",
			servingsupervision.WithReadinessCheck(func(ctx context.Context) error {
				readyCalled = true
				return nil
			}),
		)

		if rep.Phase() != servingsupervision.PhaseStarting {
			t.Fatalf("initial phase = %q, want %q", rep.Phase(), servingsupervision.PhaseStarting)
		}
		if rep.IsHealthy() {
			t.Fatal("replica should not report healthy in PhaseStarting")
		}

		if err := rep.Start(ctx); err != nil {
			t.Fatalf("start failed: %v", err)
		}

		if !readyCalled {
			t.Fatal("readiness check was not invoked on Start")
		}
		if rep.Phase() != servingsupervision.PhaseReady {
			t.Fatalf("phase after start = %q, want %q", rep.Phase(), servingsupervision.PhaseReady)
		}
		if !rep.IsHealthy() {
			t.Fatal("replica should report healthy in PhaseReady")
		}
		if rep.Generation() != 1 {
			t.Fatalf("initial generation = %d, want 1", rep.Generation())
		}
		if rep.RestartCount() != 0 {
			t.Fatalf("initial restart count = %d, want 0", rep.RestartCount())
		}
		if rep.Quarantined() {
			t.Fatal("replica should not be quarantined initially")
		}

		// Execute succeeds in PhaseReady
		execCalled := false
		err := rep.Execute(ctx, func() error {
			execCalled = true
			return nil
		})
		if err != nil {
			t.Fatalf("execute failed: %v", err)
		}
		if !execCalled {
			t.Fatal("inference callback was not executed")
		}
	})

	t.Run("failed initial readiness probe transitions to PhaseFailed", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{
			DomainID:      "rep-fail-ready",
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		rep := servingsupervision.NewReplicaSupervisor(
			spec,
			"rep-fail-1",
			servingsupervision.WithReadinessCheck(func(ctx context.Context) error {
				return errors.New("gpu device allocation timeout")
			}),
		)

		err := rep.Start(ctx)
		if err == nil {
			t.Fatal("expected error on failing readiness probe")
		}
		if rep.Phase() != servingsupervision.PhaseFailed {
			t.Fatalf("phase = %q, want %q", rep.Phase(), servingsupervision.PhaseFailed)
		}
		if rep.IsHealthy() {
			t.Fatal("replica reporting healthy despite failed readiness")
		}
	})

	t.Run("restart hook failure transitions replica to PhaseFailed", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{
			DomainID:      "rep-fail-hook",
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		rep := servingsupervision.NewReplicaSupervisor(
			spec,
			"rep-hook-1",
			servingsupervision.WithReplicaRestartHook(func(ctx context.Context, id string) error {
				return errors.New("failed to fork model worker process")
			}),
		)
		if err := rep.Start(ctx); err != nil {
			t.Fatalf("start: %v", err)
		}

		receipt, err := rep.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
		if err == nil {
			t.Fatal("expected error from failing restart hook")
		}
		if rep.Phase() != servingsupervision.PhaseFailed {
			t.Fatalf("phase = %q, want %q", rep.Phase(), servingsupervision.PhaseFailed)
		}
		if receipt == nil {
			t.Fatal("expected receipt even when restart hook fails")
		}
	})

	t.Run("readiness failure during restart transitions replica to PhaseFailed", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{
			DomainID:      "rep-post-restart-fail",
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		first := true
		rep := servingsupervision.NewReplicaSupervisor(
			spec,
			"rep-post-1",
			servingsupervision.WithReadinessCheck(func(ctx context.Context) error {
				if first {
					first = false
					return nil
				}
				return errors.New("weights checksum mismatch on reload")
			}),
		)
		if err := rep.Start(ctx); err != nil {
			t.Fatalf("start: %v", err)
		}

		_, err := rep.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
		if err == nil {
			t.Fatal("expected error when post-restart readiness check fails")
		}
		if rep.Phase() != servingsupervision.PhaseFailed {
			t.Fatalf("phase = %q, want %q", rep.Phase(), servingsupervision.PhaseFailed)
		}
	})

	t.Run("liveness check failure triggers leaf restart", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{
			DomainID:      "rep-liveness-domain",
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		liveCount := 0
		rep := servingsupervision.NewReplicaSupervisor(
			spec,
			"rep-live-1",
			servingsupervision.WithLivenessCheck(func(ctx context.Context) error {
				liveCount++
				if liveCount == 1 {
					return nil
				}
				return errors.New("worker heartbeat missed: deadlocked")
			}),
		)
		if err := rep.Start(ctx); err != nil {
			t.Fatalf("start: %v", err)
		}

		// First check succeeds
		if err := rep.CheckLiveness(ctx); err != nil {
			t.Fatalf("first liveness check failed: %v", err)
		}
		if rep.RestartCount() != 0 {
			t.Fatalf("restart count = %d, want 0", rep.RestartCount())
		}

		// Second check fails and triggers recovery
		genBefore := rep.Generation()
		if err := rep.CheckLiveness(ctx); err != nil {
			t.Fatalf("expected liveness recovery to succeed, got %v", err)
		}
		if rep.Generation() != genBefore+1 {
			t.Fatalf("generation = %d, want %d", rep.Generation(), genBefore+1)
		}
		if rep.RestartCount() != 1 {
			t.Fatalf("restart count = %d, want 1", rep.RestartCount())
		}
		if !rep.IsHealthy() {
			t.Fatal("replica should be healthy after successful liveness recovery")
		}
	})

	t.Run("reset quarantine restores service and clears restart accounting", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{
			DomainID:      "rep-reset-domain",
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 1,
		}
		rep := servingsupervision.NewReplicaSupervisor(spec, "rep-reset-1")
		if err := rep.Start(ctx); err != nil {
			t.Fatalf("start: %v", err)
		}

		// Crash 1: restarts within budget
		_, err := rep.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
		if err != nil {
			t.Fatalf("first crash: %v", err)
		}
		// Crash 2: budget exhausted -> quarantine
		_, err = rep.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
		if !errors.Is(err, servingsupervision.ErrBudgetExhausted) {
			t.Fatalf("expected ErrBudgetExhausted, got %v", err)
		}
		if !rep.Quarantined() {
			t.Fatal("replica should be quarantined")
		}

		// Operator reset
		if err := rep.ResetQuarantine(ctx); err != nil {
			t.Fatalf("reset quarantine failed: %v", err)
		}
		if rep.Quarantined() {
			t.Fatal("replica still quarantined after reset")
		}
		if rep.RestartCount() != 0 {
			t.Fatalf("restart count after reset = %d, want 0", rep.RestartCount())
		}
		if rep.Phase() != servingsupervision.PhaseReady {
			t.Fatalf("phase after reset = %q, want %q", rep.Phase(), servingsupervision.PhaseReady)
		}
		if !rep.IsHealthy() {
			t.Fatal("replica not healthy after reset")
		}
	})

	t.Run("reset quarantine fails closed if readiness fails", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{
			DomainID:      "rep-reset-fail-domain",
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 1,
		}
		allowReady := true
		rep := servingsupervision.NewReplicaSupervisor(
			spec,
			"rep-reset-fail-1",
			servingsupervision.WithReadinessCheck(func(ctx context.Context) error {
				if !allowReady {
					return errors.New("hardware memory error on reset")
				}
				return nil
			}),
		)
		if err := rep.Start(ctx); err != nil {
			t.Fatalf("start: %v", err)
		}

		// Exhaust budget
		_, _ = rep.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
		_, _ = rep.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
		if !rep.Quarantined() {
			t.Fatal("replica should be quarantined")
		}

		// Disallow readiness probe during reset
		allowReady = false
		err := rep.ResetQuarantine(ctx)
		if err == nil {
			t.Fatal("expected error on reset quarantine when readiness probe fails")
		}
		if rep.Phase() != servingsupervision.PhaseFailed {
			t.Fatalf("phase = %q, want %q", rep.Phase(), servingsupervision.PhaseFailed)
		}
	})
}

// TestControllerSupervisionScaleAndTopology tests scale up, scale down, and
// complex topologies with standalone routers and coupled KV fabrics.
func TestControllerSupervisionScaleAndTopology(t *testing.T) {
	ctx := context.Background()

	t.Run("scale up creates missing replicas non-destructively", func(t *testing.T) {
		topo, err := servingsupervision.BuildDefaultTopology("scale-up-dep", 4, 50*time.Millisecond, 3)
		if err != nil {
			t.Fatalf("build topology: %v", err)
		}

		// Start with desired count = 1
		desired1 := servingsupervision.DesiredServingState{
			DeploymentID:  "scale-up-dep",
			ModelArtifact: "native-weights",
			ReplicaCount:  1,
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired1, topo)
		if err := ctrl.Start(ctx); err != nil {
			t.Fatalf("controller start: %v", err)
		}

		report1, err := ctrl.Reconcile(ctx, nil, nil)
		if err != nil {
			t.Fatalf("reconcile 1: %v", err)
		}
		if len(report1.CreatedReplicas) != 1 {
			t.Fatalf("created = %d, want 1", len(report1.CreatedReplicas))
		}

		initialReps := ctrl.HealthyReplicas()
		if len(initialReps) != 1 {
			t.Fatalf("healthy reps = %d, want 1", len(initialReps))
		}
		initialRepGen := initialReps[0].Generation()

		// Scale up to 4 replicas
		desired4 := servingsupervision.DesiredServingState{
			DeploymentID:  "scale-up-dep",
			ModelArtifact: "native-weights",
			ReplicaCount:  4,
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		newCtrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired4, topo)
		if err := newCtrl.Start(ctx); err != nil {
			t.Fatalf("new ctrl start: %v", err)
		}

		report2, err := newCtrl.Reconcile(ctx, initialReps, ctrl.Proxy())
		if err != nil {
			t.Fatalf("reconcile 2: %v", err)
		}

		if len(report2.AdoptedReplicas) != 1 {
			t.Fatalf("adopted = %d, want 1", len(report2.AdoptedReplicas))
		}
		if len(report2.CreatedReplicas) != 3 {
			t.Fatalf("created = %d, want 3", len(report2.CreatedReplicas))
		}
		if len(newCtrl.HealthyReplicas()) != 4 {
			t.Fatalf("healthy replicas = %d, want 4", len(newCtrl.HealthyReplicas()))
		}
		// Adopted replica was never restarted
		if initialReps[0].Generation() != initialRepGen {
			t.Fatalf("adopted replica generation changed: %d vs %d", initialReps[0].Generation(), initialRepGen)
		}
	})

	t.Run("scale down drains and removes excess replicas", func(t *testing.T) {
		topo, err := servingsupervision.BuildDefaultTopology("scale-down-dep", 4, 50*time.Millisecond, 3)
		if err != nil {
			t.Fatalf("build topology: %v", err)
		}

		// Start with desired count = 4
		desired4 := servingsupervision.DesiredServingState{
			DeploymentID:  "scale-down-dep",
			ModelArtifact: "native-weights",
			ReplicaCount:  4,
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired4, topo)
		if err := ctrl.Start(ctx); err != nil {
			t.Fatalf("start: %v", err)
		}
		_, err = ctrl.Reconcile(ctx, nil, nil)
		if err != nil {
			t.Fatalf("initial reconcile: %v", err)
		}
		allReps := ctrl.HealthyReplicas()
		if len(allReps) != 4 {
			t.Fatalf("healthy reps = %d, want 4", len(allReps))
		}

		// Scale down to 2 replicas
		desired2 := servingsupervision.DesiredServingState{
			DeploymentID:  "scale-down-dep",
			ModelArtifact: "native-weights",
			ReplicaCount:  2,
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		scaleDownCtrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired2, topo)
		if err := scaleDownCtrl.Start(ctx); err != nil {
			t.Fatalf("scale down start: %v", err)
		}

		report, err := scaleDownCtrl.Reconcile(ctx, allReps, ctrl.Proxy())
		if err != nil {
			t.Fatalf("scale down reconcile: %v", err)
		}

		if len(report.AdoptedReplicas) != 4 {
			t.Fatalf("adopted in report = %d, want 4", len(report.AdoptedReplicas))
		}
		if len(report.RemovedReplicas) != 2 {
			t.Fatalf("removed = %d, want 2", len(report.RemovedReplicas))
		}
		if len(scaleDownCtrl.HealthyReplicas()) != 2 {
			t.Fatalf("remaining healthy = %d, want 2", len(scaleDownCtrl.HealthyReplicas()))
		}
	})

	t.Run("topology with standalone router and coupled KV fabrics", func(t *testing.T) {
		ctrlSpec := servingsupervision.ServingDomainSpec{DomainID: "topo-ctrl", DrainTimeout: 50 * time.Millisecond, RestartBudget: 3}
		proxySpec := servingsupervision.ServingDomainSpec{DomainID: "topo-proxy", DrainTimeout: 50 * time.Millisecond, RestartBudget: 3}
		routerSpec := servingsupervision.ServingDomainSpec{DomainID: "topo-router", DrainTimeout: 50 * time.Millisecond, RestartBudget: 3}
		kvSpec0 := servingsupervision.ServingDomainSpec{DomainID: "topo-kv-0", DrainTimeout: 50 * time.Millisecond, RestartBudget: 3}
		kvSpec1 := servingsupervision.ServingDomainSpec{DomainID: "topo-kv-1", DrainTimeout: 50 * time.Millisecond, RestartBudget: 3}

		rep0Spec := servingsupervision.ServingDomainSpec{
			DomainID:       "topo-rep-0",
			DrainTimeout:   50 * time.Millisecond,
			RestartBudget:  3,
			CoupledDomains: []string{"topo-kv-0"},
		}
		rep1Spec := servingsupervision.ServingDomainSpec{
			DomainID:       "topo-rep-1",
			DrainTimeout:   50 * time.Millisecond,
			RestartBudget:  3,
			CoupledDomains: []string{"topo-kv-1"},
		}

		topo, err := servingsupervision.NewServingTopology(
			"complex-topo",
			ctrlSpec,
			proxySpec,
			[]servingsupervision.ServingDomainSpec{rep0Spec, rep1Spec},
			servingsupervision.WithTopologyRouter(routerSpec),
			servingsupervision.WithTopologyKVFabric(kvSpec0),
			servingsupervision.WithTopologyKVFabric(kvSpec1),
		)
		if err != nil {
			t.Fatalf("create complex topology: %v", err)
		}

		if topo.Router == nil || topo.Router.DomainID != "topo-router" {
			t.Fatal("router domain missing or incorrect")
		}
		if len(topo.KVFabrics) != 2 {
			t.Fatalf("kv fabrics count = %d, want 2", len(topo.KVFabrics))
		}
		if len(topo.Domains()) != 7 {
			t.Fatalf("total domains = %d, want 7", len(topo.Domains()))
		}

		desired := servingsupervision.DesiredServingState{
			DeploymentID:  "complex-topo",
			ModelArtifact: "native-model",
			ReplicaCount:  2,
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo)
		if err := ctrl.Start(ctx); err != nil {
			t.Fatalf("start: %v", err)
		}
		report, err := ctrl.Reconcile(ctx, nil, nil)
		if err != nil {
			t.Fatalf("reconcile: %v", err)
		}
		if len(report.CreatedReplicas) != 2 {
			t.Fatalf("created = %d, want 2", len(report.CreatedReplicas))
		}
	})
}

// TestDomainIsolationMultiReplicaConcurrentFailures verifies that multiple replicas
// failing concurrently recover independently without interfering with each other
// or disrupting unaffected sibling replicas, controller, or proxy.
func TestDomainIsolationMultiReplicaConcurrentFailures(t *testing.T) {
	ctx := context.Background()

	topo, err := servingsupervision.BuildDefaultTopology("concurrent-fail", 4, 50*time.Millisecond, 3)
	if err != nil {
		t.Fatalf("topology: %v", err)
	}

	desired := servingsupervision.DesiredServingState{
		DeploymentID:  "concurrent-fail",
		ModelArtifact: "fak-model:qwen38",
		ReplicaCount:  4,
		DrainTimeout:  50 * time.Millisecond,
		RestartBudget: 3,
	}

	ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo)
	if err := ctrl.Start(ctx); err != nil {
		t.Fatalf("ctrl start: %v", err)
	}
	if _, err := ctrl.Reconcile(ctx, nil, nil); err != nil {
		t.Fatalf("ctrl reconcile: %v", err)
	}

	reps := ctrl.HealthyReplicas()
	if len(reps) != 4 {
		t.Fatalf("expected 4 replicas, got %d", len(reps))
	}
	proxy := ctrl.Proxy()

	// Pick 2 failing replicas and 2 unaffected sibling replicas
	failing0 := reps[0]
	failing1 := reps[1]
	stable0 := reps[2]
	stable1 := reps[3]

	fail0GenBefore := failing0.Generation()
	fail1GenBefore := failing1.Generation()
	stab0GenBefore := stable0.Generation()
	stab1GenBefore := stable1.Generation()

	// Concurrently crash failing0 and failing1
	var wg sync.WaitGroup
	var err0, err1 error
	var rec0, rec1 *servingsupervision.ServingReceipt

	wg.Add(2)
	go func() {
		defer wg.Done()
		rec0, err0 = failing0.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
	}()
	go func() {
		defer wg.Done()
		rec1, err1 = failing1.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
	}()

	wg.Wait()

	if err0 != nil {
		t.Fatalf("failing0 recovery error: %v", err0)
	}
	if err1 != nil {
		t.Fatalf("failing1 recovery error: %v", err1)
	}

	// Verify both failing replicas bumped generations independently
	if failing0.Generation() != fail0GenBefore+1 {
		t.Fatalf("failing0 gen = %d, want %d", failing0.Generation(), fail0GenBefore+1)
	}
	if failing1.Generation() != fail1GenBefore+1 {
		t.Fatalf("failing1 gen = %d, want %d", failing1.Generation(), fail1GenBefore+1)
	}
	if rec0.MemberID != failing0.ReplicaID() || rec1.MemberID != failing1.ReplicaID() {
		t.Fatalf("receipt member IDs mismatch: %s vs %s", rec0.MemberID, rec1.MemberID)
	}

	// Stable replicas MUST be completely unaffected
	if stable0.Generation() != stab0GenBefore || stable0.RestartCount() != 0 {
		t.Fatalf("stable0 disrupted: gen=%d, restarts=%d", stable0.Generation(), stable0.RestartCount())
	}
	if stable1.Generation() != stab1GenBefore || stable1.RestartCount() != 0 {
		t.Fatalf("stable1 disrupted: gen=%d, restarts=%d", stable1.Generation(), stable1.RestartCount())
	}
	if !stable0.IsHealthy() || !stable1.IsHealthy() {
		t.Fatal("stable replicas are not healthy")
	}

	// Controller and proxy remained healthy
	if ctrl.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("controller phase = %q, want ready", ctrl.Phase())
	}
	if proxy.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("proxy phase = %q, want ready", proxy.Phase())
	}

	// All 4 replicas are healthy again and proxy routes to all of them
	seen := make(map[string]bool)
	for i := 0; i < 12; i++ {
		routed, err := proxy.Route()
		if err != nil {
			t.Fatalf("proxy route error: %v", err)
		}
		seen[routed.ReplicaID()] = true
	}
	if len(seen) != 4 {
		t.Fatalf("proxy routed to %d distinct replicas, want 4", len(seen))
	}
}

// TestDrainManagerConcurrencyAndDeadlines tests high-concurrency request tracking,
// clean drains, and drain timeout deadlines where slow requests become lost work.
func TestDrainManagerConcurrencyAndDeadlines(t *testing.T) {
	ctx := context.Background()

	t.Run("concurrent acquire and clean drain", func(t *testing.T) {
		dm := servingsupervision.NewDrainManager("dm-conc-domain", "dm-conc-1", servingsupervision.RoleReplica, 100*time.Millisecond, 1)

		const workers = 20
		var wg sync.WaitGroup
		releases := make([]func(), workers)

		for i := 0; i < workers; i++ {
			rel, err := dm.Acquire()
			if err != nil {
				t.Fatalf("acquire %d failed: %v", i, err)
			}
			releases[i] = rel
		}

		if dm.Inflight() != workers {
			t.Fatalf("inflight = %d, want %d", dm.Inflight(), workers)
		}

		// Concurrently release half before drain, half during drain
		for i := 0; i < workers/2; i++ {
			releases[i]()
		}
		if dm.Inflight() != workers/2 {
			t.Fatalf("inflight after half releases = %d, want %d", dm.Inflight(), workers/2)
		}

		var receipt *servingsupervision.ServingReceipt
		var drainErr error

		wg.Add(1)
		go func() {
			defer wg.Done()
			receipt, drainErr = dm.Drain(ctx, servingsupervision.ErrWorkerProcessFailure, servingsupervision.ScopeLeafOnly, false, 2)
		}()

		// Release remaining requests concurrently
		go func() {
			for dm.Phase() != servingsupervision.PhaseDraining {
				runtime.Gosched()
			}
			for i := workers / 2; i < workers; i++ {
				releases[i]()
			}
		}()

		wg.Wait()

		if drainErr != nil {
			t.Fatalf("drain error: %v", drainErr)
		}
		if receipt.InflightDrained != workers/2 {
			t.Fatalf("inflight_drained = %d, want %d", receipt.InflightDrained, workers/2)
		}
		if receipt.InflightLost != 0 {
			t.Fatalf("inflight_lost = %d, want 0", receipt.InflightLost)
		}
		if receipt.ObservedGen != 1 || receipt.NextGen != 2 {
			t.Fatalf("generation mismatch in receipt: %d -> %d", receipt.ObservedGen, receipt.NextGen)
		}

		// Reset restores PhaseReady
		dm.Reset(2)
		if dm.Phase() != servingsupervision.PhaseReady {
			t.Fatalf("phase after reset = %q, want ready", dm.Phase())
		}
		rel, err := dm.Acquire()
		if err != nil {
			t.Fatalf("acquire after reset failed: %v", err)
		}
		rel()
	})

	t.Run("drain deadline expiration records lost work accurately", func(t *testing.T) {
		dm := servingsupervision.NewDrainManager("dm-timeout-domain", "dm-timeout-1", servingsupervision.RoleReplica, 25*time.Millisecond, 5)

		// Acquire 3 requests; release 1, leave 2 unreleased
		rel1, err1 := dm.Acquire()
		if err1 != nil {
			t.Fatalf("acquire 1: %v", err1)
		}
		rel2, err2 := dm.Acquire()
		if err2 != nil {
			t.Fatalf("acquire 2: %v", err2)
		}
		defer rel2()
		rel3, err3 := dm.Acquire()
		if err3 != nil {
			t.Fatalf("acquire 3: %v", err3)
		}
		defer rel3()

		// Release 1 before drain
		rel1()

		receipt, err := dm.Drain(ctx, servingsupervision.ErrFailedLiveness, servingsupervision.ScopeLeafOnly, false, 6)
		if err != nil {
			t.Fatalf("drain failed: %v", err)
		}

		if receipt.InflightLost != 2 {
			t.Fatalf("inflight_lost = %d, want 2", receipt.InflightLost)
		}
		if receipt.InflightDrained != 0 {
			t.Fatalf("inflight_drained = %d, want 0", receipt.InflightDrained)
		}
		if receipt.ErrorKind != servingsupervision.ErrorKindFailedLiveness {
			t.Fatalf("error kind = %q, want %q", receipt.ErrorKind, servingsupervision.ErrorKindFailedLiveness)
		}
	})
}

// TestDomainSpecValidationInvariants tests domain specification constraint
// enforcement and fail-closed validation rules.
func TestDomainSpecValidationInvariants(t *testing.T) {
	t.Run("negative drain timeout rejected", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: "ctrl", DrainTimeout: -1}
		proxy := servingsupervision.ServingDomainSpec{DomainID: "proxy"}
		rep := servingsupervision.ServingDomainSpec{DomainID: "rep"}
		_, err := servingsupervision.NewServingTopology("test", ctrl, proxy, []servingsupervision.ServingDomainSpec{rep})
		if err == nil {
			t.Fatal("expected error for negative drain timeout")
		}
	})

	t.Run("negative restart budget rejected", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: "ctrl"}
		proxy := servingsupervision.ServingDomainSpec{DomainID: "proxy", RestartBudget: -5}
		rep := servingsupervision.ServingDomainSpec{DomainID: "rep"}
		_, err := servingsupervision.NewServingTopology("test", ctrl, proxy, []servingsupervision.ServingDomainSpec{rep})
		if err == nil {
			t.Fatal("expected error for negative restart budget")
		}
	})

	t.Run("empty deployment ID rejected", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: "ctrl"}
		proxy := servingsupervision.ServingDomainSpec{DomainID: "proxy"}
		rep := servingsupervision.ServingDomainSpec{DomainID: "rep"}
		_, err := servingsupervision.NewServingTopology("", ctrl, proxy, []servingsupervision.ServingDomainSpec{rep})
		if err == nil {
			t.Fatal("expected error for empty deployment ID")
		}
	})

	t.Run("empty controller domain ID rejected", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: ""}
		proxy := servingsupervision.ServingDomainSpec{DomainID: "proxy"}
		rep := servingsupervision.ServingDomainSpec{DomainID: "rep"}
		_, err := servingsupervision.NewServingTopology("test", ctrl, proxy, []servingsupervision.ServingDomainSpec{rep})
		if err == nil {
			t.Fatal("expected error for empty controller domain ID")
		}
	})

	t.Run("empty proxy domain ID rejected", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: "ctrl"}
		proxy := servingsupervision.ServingDomainSpec{DomainID: ""}
		rep := servingsupervision.ServingDomainSpec{DomainID: "rep"}
		_, err := servingsupervision.NewServingTopology("test", ctrl, proxy, []servingsupervision.ServingDomainSpec{rep})
		if err == nil {
			t.Fatal("expected error for empty proxy domain ID")
		}
	})

	t.Run("zero replicas rejected", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: "ctrl"}
		proxy := servingsupervision.ServingDomainSpec{DomainID: "proxy"}
		_, err := servingsupervision.NewServingTopology("test", ctrl, proxy, nil)
		if err == nil {
			t.Fatal("expected error for empty replica list")
		}
	})

	t.Run("self-coupling rejected", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: "ctrl"}
		proxy := servingsupervision.ServingDomainSpec{DomainID: "proxy"}
		rep := servingsupervision.ServingDomainSpec{DomainID: "rep-self", CoupledDomains: []string{"rep-self"}}
		_, err := servingsupervision.NewServingTopology("test", ctrl, proxy, []servingsupervision.ServingDomainSpec{rep})
		if err == nil {
			t.Fatal("expected error for self-coupled domain")
		}
	})

	t.Run("coupling to unknown domain rejected", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: "ctrl"}
		proxy := servingsupervision.ServingDomainSpec{DomainID: "proxy"}
		rep := servingsupervision.ServingDomainSpec{DomainID: "rep-unknown", CoupledDomains: []string{"ghost-domain"}}
		_, err := servingsupervision.NewServingTopology("test", ctrl, proxy, []servingsupervision.ServingDomainSpec{rep})
		if err == nil {
			t.Fatal("expected error for coupling to unknown domain")
		}
	})

	t.Run("BuildDefaultTopology rejects non-positive replica count", func(t *testing.T) {
		_, err0 := servingsupervision.BuildDefaultTopology("test", 0, 50*time.Millisecond, 3)
		if err0 == nil {
			t.Fatal("expected error for 0 replicas in default topology")
		}
		_, errNeg := servingsupervision.BuildDefaultTopology("test", -2, 50*time.Millisecond, 3)
		if errNeg == nil {
			t.Fatal("expected error for negative replicas in default topology")
		}
	})

	t.Run("Domain lookup returns matching spec or false", func(t *testing.T) {
		topo, err := servingsupervision.BuildDefaultTopology("lookup-test", 2, 50*time.Millisecond, 3)
		if err != nil {
			t.Fatalf("build topo: %v", err)
		}
		spec, found := topo.Domain("lookup-test-controller")
		if !found || spec.DomainID != "lookup-test-controller" {
			t.Fatalf("expected to find controller domain: found=%v, spec=%+v", found, spec)
		}
		_, notFound := topo.Domain("non-existent-domain")
		if notFound {
			t.Fatal("expected notFound for non-existent domain")
		}
	})
}

// BenchmarkSupervisionController measures controller reconciliation and adoption latency.
func BenchmarkSupervisionController(b *testing.B) {
	ctx := context.Background()
	topo, err := servingsupervision.BuildDefaultTopology("bench-ctrl", 4, 100*time.Millisecond, 5)
	if err != nil {
		b.Fatalf("build topology: %v", err)
	}

	desired := servingsupervision.DesiredServingState{
		DeploymentID:  "bench-ctrl",
		ModelArtifact: "fak-model:bench",
		ReplicaCount:  4,
		DrainTimeout:  100 * time.Millisecond,
		RestartBudget: 5,
	}

	ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo)
	if err := ctrl.Start(ctx); err != nil {
		b.Fatalf("ctrl start: %v", err)
	}
	if _, err := ctrl.Reconcile(ctx, nil, nil); err != nil {
		b.Fatalf("initial reconcile: %v", err)
	}

	existingReplicas := ctrl.HealthyReplicas()
	existingProxy := ctrl.Proxy()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := ctrl.Reconcile(ctx, existingReplicas, existingProxy)
		if err != nil {
			b.Fatalf("reconcile failed: %v", err)
		}
	}
}

// BenchmarkReplicaSupervisorExecute measures overhead of ReplicaSupervisor.Execute
// with concurrent request acquisition and completion.
func BenchmarkReplicaSupervisorExecute(b *testing.B) {
	ctx := context.Background()
	spec := servingsupervision.ServingDomainSpec{
		DomainID:      "bench-rep-domain",
		DrainTimeout:  100 * time.Millisecond,
		RestartBudget: 10,
	}
	rep := servingsupervision.NewReplicaSupervisor(spec, "bench-rep-0")
	if err := rep.Start(ctx); err != nil {
		b.Fatalf("start replica: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			err := rep.Execute(ctx, func() error {
				return nil
			})
			if err != nil {
				b.Errorf("execute error: %v", err)
			}
		}
	})
}

// BenchmarkProxySupervisorRoute measures round-robin routing throughput across replicas.
func BenchmarkProxySupervisorRoute(b *testing.B) {
	ctx := context.Background()
	spec := servingsupervision.ServingDomainSpec{
		DomainID:      "bench-proxy-domain",
		DrainTimeout:  100 * time.Millisecond,
		RestartBudget: 5,
	}

	reps := make([]*servingsupervision.ReplicaSupervisor, 8)
	for i := 0; i < 8; i++ {
		repSpec := servingsupervision.ServingDomainSpec{
			DomainID:      fmt.Sprintf("bench-rep-%d", i),
			DrainTimeout:  100 * time.Millisecond,
			RestartBudget: 5,
		}
		reps[i] = servingsupervision.NewReplicaSupervisor(repSpec, fmt.Sprintf("rep-%d", i))
		if err := reps[i].Start(ctx); err != nil {
			b.Fatalf("start replica %d: %v", i, err)
		}
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		b.Fatalf("listen: %v", err)
	}
	endpoint := "http://" + ln.Addr().String()
	ln.Close()

	proxy := servingsupervision.NewProxySupervisor(spec, "bench-proxy", endpoint)
	if err := proxy.Start(ctx); err != nil {
		b.Fatalf("start proxy: %v", err)
	}
	if err := proxy.Reconstruct(ctx, reps); err != nil {
		b.Fatalf("reconstruct proxy: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			target, err := proxy.Route()
			if err != nil || target == nil {
				b.Errorf("route error: %v", err)
			}
		}
	})
}

// BenchmarkDrainManagerAcquireRelease measures concurrent Acquire/Release performance.
func BenchmarkDrainManagerAcquireRelease(b *testing.B) {
	dm := servingsupervision.NewDrainManager("bench-dm-domain", "bench-dm", servingsupervision.RoleReplica, 100*time.Millisecond, 1)

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			release, err := dm.Acquire()
			if err != nil {
				b.Errorf("acquire error: %v", err)
				return
			}
			release()
		}
	})
}
