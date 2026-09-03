package servingsupervision_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/servingsupervision"
)

func TestTypesAndErrorClassification(t *testing.T) {
	cases := []struct {
		name      string
		err       error
		wantKind  servingsupervision.ServingErrorKind
		wantScope servingsupervision.RestartScope
	}{
		{
			name:      "nil error",
			err:       nil,
			wantKind:  "",
			wantScope: servingsupervision.ScopeNone,
		},
		{
			name:      "sentinel request application error",
			err:       servingsupervision.ErrRequestApplication,
			wantKind:  servingsupervision.ErrorKindRequestApplication,
			wantScope: servingsupervision.ScopeNone,
		},
		{
			name:      "sentinel model state corruption",
			err:       servingsupervision.ErrModelStateCorruption,
			wantKind:  servingsupervision.ErrorKindModelStateCorruption,
			wantScope: servingsupervision.ScopeDeploymentDomain,
		},
		{
			name:      "sentinel kv fabric failure",
			err:       servingsupervision.ErrKVFabricFailure,
			wantKind:  servingsupervision.ErrorKindKVFabricFailure,
			wantScope: servingsupervision.ScopeDeploymentDomain,
		},
		{
			name:      "sentinel failed readiness",
			err:       servingsupervision.ErrFailedReadiness,
			wantKind:  servingsupervision.ErrorKindFailedReadiness,
			wantScope: servingsupervision.ScopeLeafOnly,
		},
		{
			name:      "sentinel failed liveness",
			err:       servingsupervision.ErrFailedLiveness,
			wantKind:  servingsupervision.ErrorKindFailedLiveness,
			wantScope: servingsupervision.ScopeLeafOnly,
		},
		{
			name:      "sentinel controller failure",
			err:       servingsupervision.ErrControllerFailure,
			wantKind:  servingsupervision.ErrorKindControllerFailure,
			wantScope: servingsupervision.ScopeLeafOnly,
		},
		{
			name:      "sentinel worker process failure",
			err:       servingsupervision.ErrWorkerProcessFailure,
			wantKind:  servingsupervision.ErrorKindWorkerProcessFailure,
			wantScope: servingsupervision.ScopeLeafOnly,
		},
		{
			name:      "wrapped classified error",
			err:       servingsupervision.WrapClassifiedError(servingsupervision.ErrorKindModelStateCorruption, servingsupervision.ScopeDeploymentDomain, errors.New("custom corruption")),
			wantKind:  servingsupervision.ErrorKindModelStateCorruption,
			wantScope: servingsupervision.ScopeDeploymentDomain,
		},
		{
			name:      "string pattern: bad request",
			err:       errors.New("user error: bad request validation failed"),
			wantKind:  servingsupervision.ErrorKindRequestApplication,
			wantScope: servingsupervision.ScopeNone,
		},
		{
			name:      "string pattern: NaN tensor corruption",
			err:       errors.New("fatal: NaN tensor detected in model state"),
			wantKind:  servingsupervision.ErrorKindModelStateCorruption,
			wantScope: servingsupervision.ScopeDeploymentDomain,
		},
		{
			name:      "string pattern: kv fabric channel failure",
			err:       errors.New("remote kv fabric connection lost"),
			wantKind:  servingsupervision.ErrorKindKVFabricFailure,
			wantScope: servingsupervision.ScopeDeploymentDomain,
		},
		{
			name:      "string pattern: failed readiness probe",
			err:       errors.New("worker failed readiness probe timeout"),
			wantKind:  servingsupervision.ErrorKindFailedReadiness,
			wantScope: servingsupervision.ScopeLeafOnly,
		},
		{
			name:      "string pattern: deadlock liveness probe",
			err:       errors.New("liveness probe: worker deadlock observed"),
			wantKind:  servingsupervision.ErrorKindFailedLiveness,
			wantScope: servingsupervision.ScopeLeafOnly,
		},
		{
			name:      "unknown error fallback to leaf worker failure",
			err:       errors.New("unexpected os signal 9"),
			wantKind:  servingsupervision.ErrorKindWorkerProcessFailure,
			wantScope: servingsupervision.ScopeLeafOnly,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			kind, scope := servingsupervision.ClassifyError(tc.err)
			if kind != tc.wantKind {
				t.Fatalf("kind = %q, want %q", kind, tc.wantKind)
			}
			if scope != tc.wantScope {
				t.Fatalf("scope = %q, want %q", scope, tc.wantScope)
			}
		})
	}
}

func TestTopologyValidationAndFailureDomainSeparation(t *testing.T) {
	t.Run("valid topology with distinct failure domains", func(t *testing.T) {
		topo, err := servingsupervision.BuildDefaultTopology("prod", 3, 2*time.Second, 3)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(topo.Domains()) != 5 { // 1 controller + 1 proxy + 3 replicas
			t.Fatalf("expected 5 domains, got %d", len(topo.Domains()))
		}
	})

	t.Run("rejects controller and proxy sharing failure domain", func(t *testing.T) {
		sharedSpec := servingsupervision.ServingDomainSpec{DomainID: "shared-domain"}
		replicaSpec := servingsupervision.ServingDomainSpec{DomainID: "replica-0"}
		_, err := servingsupervision.NewServingTopology("test", sharedSpec, sharedSpec, []servingsupervision.ServingDomainSpec{replicaSpec})
		if err == nil {
			t.Fatal("expected error when controller and proxy share domain ID")
		}
	})

	t.Run("rejects replicas sharing failure domain", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: "ctrl"}
		proxy := servingsupervision.ServingDomainSpec{DomainID: "proxy"}
		rep0 := servingsupervision.ServingDomainSpec{DomainID: "rep-shared"}
		rep1 := servingsupervision.ServingDomainSpec{DomainID: "rep-shared"}

		_, err := servingsupervision.NewServingTopology("test", ctrl, proxy, []servingsupervision.ServingDomainSpec{rep0, rep1})
		if err == nil {
			t.Fatal("expected error when replicas share domain ID")
		}
	})

	t.Run("rejects replica coupled to sibling replica", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: "ctrl"}
		proxy := servingsupervision.ServingDomainSpec{DomainID: "proxy"}
		rep0 := servingsupervision.ServingDomainSpec{DomainID: "rep-0", CoupledDomains: []string{"rep-1"}}
		rep1 := servingsupervision.ServingDomainSpec{DomainID: "rep-1"}

		_, err := servingsupervision.NewServingTopology("test", ctrl, proxy, []servingsupervision.ServingDomainSpec{rep0, rep1})
		if err == nil {
			t.Fatal("expected error when replica couples to sibling replica")
		}
	})

	t.Run("allows replica coupled to dedicated KV fabric", func(t *testing.T) {
		ctrl := servingsupervision.ServingDomainSpec{DomainID: "ctrl"}
		proxy := servingsupervision.ServingDomainSpec{DomainID: "proxy"}
		kvSpec := servingsupervision.ServingDomainSpec{DomainID: "kv-fabric-0"}
		rep0 := servingsupervision.ServingDomainSpec{DomainID: "rep-0", CoupledDomains: []string{"kv-fabric-0"}}

		topo, err := servingsupervision.NewServingTopology(
			"test", ctrl, proxy, []servingsupervision.ServingDomainSpec{rep0},
			servingsupervision.WithTopologyKVFabric(kvSpec),
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(topo.KVFabrics) != 1 {
			t.Fatalf("expected 1 kv fabric, got %d", len(topo.KVFabrics))
		}
	})
}

func TestDrainBoundedTrackingAndReceipt(t *testing.T) {
	ctx := context.Background()
	dm := servingsupervision.NewDrainManager("domain-1", "member-1", servingsupervision.RoleReplica, 50*time.Millisecond, 1)

	// Acquire 2 requests
	release1, err1 := dm.Acquire()
	if err1 != nil {
		t.Fatalf("acquire 1 failed: %v", err1)
	}
	release2, err2 := dm.Acquire()
	if err2 != nil {
		t.Fatalf("acquire 2 failed: %v", err2)
	}
	if dm.Inflight() != 2 {
		t.Fatalf("inflight = %d, want 2", dm.Inflight())
	}

	var wg sync.WaitGroup
	var receipt *servingsupervision.ServingReceipt
	var drainErr error

	// Release 1 request during drain, leave 1 slow request
	go func() {
		time.Sleep(10 * time.Millisecond)
		release1()
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		receipt, drainErr = dm.Drain(ctx, servingsupervision.ErrWorkerProcessFailure, servingsupervision.ScopeLeafOnly, false, 2)
	}()

	// Wait for drain to finish (due to timeout on remaining inflight request)
	wg.Wait()
	if drainErr != nil {
		t.Fatalf("drain failed: %v", drainErr)
	}

	// Verify that new requests are rejected during/after drain
	if _, err := dm.Acquire(); !errors.Is(err, servingsupervision.ErrTrafficWithdrawn) {
		t.Fatalf("expected ErrTrafficWithdrawn, got %v", err)
	}

	// Complete the slow request
	release2()

	// Verify receipt contents
	if receipt.Schema != servingsupervision.ServingReceiptSchema {
		t.Fatalf("schema = %q, want %q", receipt.Schema, servingsupervision.ServingReceiptSchema)
	}
	if receipt.Engine != servingsupervision.EngineNative {
		t.Fatalf("engine = %q, want %q", receipt.Engine, servingsupervision.EngineNative)
	}
	if receipt.FallbackUsed {
		t.Fatalf("fallback_used must be false")
	}
	if receipt.InflightDrained != 1 {
		t.Fatalf("inflight_drained = %d, want 1", receipt.InflightDrained)
	}
	if receipt.InflightLost != 1 {
		t.Fatalf("inflight_lost = %d, want 1", receipt.InflightLost)
	}
	if receipt.RestartScope != servingsupervision.ScopeLeafOnly {
		t.Fatalf("restart_scope = %q, want %q", receipt.RestartScope, servingsupervision.ScopeLeafOnly)
	}
	if receipt.ObservedGen != 1 || receipt.NextGen != 2 {
		t.Fatalf("gen mismatch: observed %d, next %d", receipt.ObservedGen, receipt.NextGen)
	}
}

func TestReplicaIsolationReadinessAndQuarantine(t *testing.T) {
	ctx := context.Background()
	spec := servingsupervision.ServingDomainSpec{
		DomainID:      "rep-0-domain",
		DrainTimeout:  50 * time.Millisecond,
		RestartBudget: 2,
	}

	readyCount := 0
	r := servingsupervision.NewReplicaSupervisor(
		spec,
		"rep-0",
		servingsupervision.WithReadinessCheck(func(ctx context.Context) error {
			readyCount++
			return nil
		}),
	)

	if err := r.Start(ctx); err != nil {
		t.Fatalf("start failed: %v", err)
	}
	if !r.IsHealthy() {
		t.Fatalf("replica should be healthy")
	}
	if readyCount != 1 {
		t.Fatalf("readyCount = %d, want 1", readyCount)
	}

	// Request/application error should NOT restart the replica
	reqErr := errors.New("bad request: invalid prompt JSON")
	receipt, err := r.HandleError(ctx, reqErr)
	if err != nil {
		t.Fatalf("handle request error failed: %v", err)
	}
	if receipt.RestartScope != servingsupervision.ScopeNone {
		t.Fatalf("scope = %q, want %q", receipt.RestartScope, servingsupervision.ScopeNone)
	}
	if r.RestartCount() != 0 {
		t.Fatalf("replica restart count should be 0, got %d", r.RestartCount())
	}
	if !r.IsHealthy() {
		t.Fatalf("replica should remain healthy after request error")
	}

	// First failure: within budget -> restarts leaf, runs readiness
	rec1, err1 := r.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
	if err1 != nil {
		t.Fatalf("restart 1 failed: %v", err1)
	}
	if rec1.RestartScope != servingsupervision.ScopeLeafOnly {
		t.Fatalf("scope = %q, want %q", rec1.RestartScope, servingsupervision.ScopeLeafOnly)
	}
	if r.RestartCount() != 1 {
		t.Fatalf("restart count = %d, want 1", r.RestartCount())
	}
	if readyCount != 2 {
		t.Fatalf("readyCount = %d, want 2", readyCount)
	}
	if !r.IsHealthy() {
		t.Fatalf("replica should be healthy after successful restart")
	}

	// Second failure: within budget
	_, err2 := r.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
	if err2 != nil {
		t.Fatalf("restart 2 failed: %v", err2)
	}
	if r.RestartCount() != 2 {
		t.Fatalf("restart count = %d, want 2", r.RestartCount())
	}

	// Third failure: budget exhausted -> quarantine!
	rec3, err3 := r.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
	if !errors.Is(err3, servingsupervision.ErrBudgetExhausted) {
		t.Fatalf("expected ErrBudgetExhausted, got %v", err3)
	}
	if !r.Quarantined() {
		t.Fatalf("replica should be quarantined")
	}
	if r.Phase() != servingsupervision.PhaseQuarantined {
		t.Fatalf("phase = %q, want %q", r.Phase(), servingsupervision.PhaseQuarantined)
	}
	if rec3.RestartScope != servingsupervision.ScopeQuarantine || !rec3.Quarantined {
		t.Fatalf("receipt should reflect quarantine: %+v", rec3)
	}

	// Traffic should be rejected
	if err := r.Execute(ctx, func() error { return nil }); !errors.Is(err, servingsupervision.ErrTrafficWithdrawn) {
		t.Fatalf("expected ErrTrafficWithdrawn for quarantined replica, got %v", err)
	}
}

func TestProxyReconstructibleRestart(t *testing.T) {
	ctx := context.Background()
	proxySpec := servingsupervision.ServingDomainSpec{
		DomainID:     "proxy-domain",
		DrainTimeout: 50 * time.Millisecond,
	}

	// Create 2 healthy replicas
	rep0Spec := servingsupervision.ServingDomainSpec{DomainID: "rep0-domain", DrainTimeout: 50 * time.Millisecond, RestartBudget: 3}
	rep1Spec := servingsupervision.ServingDomainSpec{DomainID: "rep1-domain", DrainTimeout: 50 * time.Millisecond, RestartBudget: 3}
	rep0 := servingsupervision.NewReplicaSupervisor(rep0Spec, "rep-0")
	rep1 := servingsupervision.NewReplicaSupervisor(rep1Spec, "rep-1")

	if err := rep0.Start(ctx); err != nil {
		t.Fatalf("rep0 start: %v", err)
	}
	if err := rep1.Start(ctx); err != nil {
		t.Fatalf("rep1 start: %v", err)
	}

	proxy := servingsupervision.NewProxySupervisor(proxySpec, "proxy-0", "http://127.0.0.1:9090")
	if err := proxy.Start(ctx); err != nil {
		t.Fatalf("proxy start: %v", err)
	}
	if err := proxy.Reconstruct(ctx, []*servingsupervision.ReplicaSupervisor{rep0, rep1}); err != nil {
		t.Fatalf("reconstruct: %v", err)
	}

	// Route traffic
	target1, err := proxy.Route()
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	target2, err := proxy.Route()
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	if target1.ReplicaID() == target2.ReplicaID() {
		t.Fatalf("round robin should alternate replicas: %s vs %s", target1.ReplicaID(), target2.ReplicaID())
	}

	// Restart proxy independently
	rep0GenBefore := rep0.Generation()
	rep1GenBefore := rep1.Generation()

	receipt, err := proxy.Restart(ctx, []*servingsupervision.ReplicaSupervisor{rep0, rep1})
	if err != nil {
		t.Fatalf("proxy restart: %v", err)
	}

	// Replicas MUST NOT be torn down or restarted
	if rep0.Generation() != rep0GenBefore || !rep0.IsHealthy() {
		t.Fatalf("replica 0 was disrupted by proxy restart")
	}
	if rep1.Generation() != rep1GenBefore || !rep1.IsHealthy() {
		t.Fatalf("replica 1 was disrupted by proxy restart")
	}

	// Stable endpoint identity is preserved
	if proxy.Endpoint() != "http://127.0.0.1:9090" {
		t.Fatalf("endpoint identity changed: %s", proxy.Endpoint())
	}

	if receipt.Role != servingsupervision.RoleProxy || receipt.Engine != servingsupervision.EngineNative || receipt.FallbackUsed {
		t.Fatalf("invalid proxy receipt: %+v", receipt)
	}
}

func TestControllerStateReconciliationAndAdoption(t *testing.T) {
	ctx := context.Background()
	topo, err := servingsupervision.BuildDefaultTopology("cluster", 2, 50*time.Millisecond, 3)
	if err != nil {
		t.Fatalf("build topology: %v", err)
	}

	desired := servingsupervision.DesiredServingState{
		DeploymentID:  "cluster",
		ModelArtifact: "fak-model:qwen38",
		ReplicaCount:  2,
		DrainTimeout:  50 * time.Millisecond,
		RestartBudget: 3,
	}

	ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo)
	if err := ctrl.Start(ctx); err != nil {
		t.Fatalf("controller start: %v", err)
	}

	// Initial reconciliation: creates 2 replicas and 1 proxy
	report, err := ctrl.Reconcile(ctx, nil, nil)
	if err != nil {
		t.Fatalf("initial reconcile: %v", err)
	}
	if len(report.CreatedReplicas) != 2 {
		t.Fatalf("created = %d, want 2", len(report.CreatedReplicas))
	}

	reps := ctrl.HealthyReplicas()
	if len(reps) != 2 {
		t.Fatalf("healthy replicas = %d, want 2", len(reps))
	}
	proxy := ctrl.Proxy()
	if proxy == nil {
		t.Fatalf("proxy should not be nil")
	}

	// Record replica generations before simulated controller crash
	rep0Gen := reps[0].Generation()
	rep1Gen := reps[1].Generation()

	// Simulate controller crash and replacement from desired state
	newCtrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo)
	if err := newCtrl.Start(ctx); err != nil {
		t.Fatalf("new controller start: %v", err)
	}

	// Non-destructive adoption of existing healthy replicas & proxy
	adoptReport, err := newCtrl.Reconcile(ctx, reps, proxy)
	if err != nil {
		t.Fatalf("reconcile adopt: %v", err)
	}

	if len(adoptReport.AdoptedReplicas) != 2 {
		t.Fatalf("adopted = %d, want 2", len(adoptReport.AdoptedReplicas))
	}
	if len(adoptReport.CreatedReplicas) != 0 {
		t.Fatalf("created replicas during adoption = %d, want 0", len(adoptReport.CreatedReplicas))
	}
	if !adoptReport.PreservedProxy {
		t.Fatalf("proxy should be preserved")
	}

	// Verify replicas were never restarted
	if reps[0].Generation() != rep0Gen || !reps[0].IsHealthy() {
		t.Fatalf("replica 0 was disrupted during controller adoption")
	}
	if reps[1].Generation() != rep1Gen || !reps[1].IsHealthy() {
		t.Fatalf("replica 1 was disrupted during controller adoption")
	}
}

func TestIssue10574AcceptanceCriteria(t *testing.T) {
	ctx := context.Background()

	t.Run("Kill one native replica: controller, proxy, and sibling replicas stay alive", func(t *testing.T) {
		topo, err := servingsupervision.BuildDefaultTopology("witness-serving", 3, 50*time.Millisecond, 3)
		if err != nil {
			t.Fatalf("topology: %v", err)
		}

		desired := servingsupervision.DesiredServingState{
			DeploymentID:  "witness-serving",
			ModelArtifact: "native-model",
			ReplicaCount:  3,
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
		if len(reps) != 3 {
			t.Fatalf("expected 3 healthy replicas, got %d", len(reps))
		}
		proxy := ctrl.Proxy()

		victim := reps[0]
		sibling1 := reps[1]
		sibling2 := reps[2]

		// Kill victim replica
		receipt, err := victim.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
		if err != nil {
			t.Fatalf("victim handle error: %v", err)
		}

		// Controller, proxy, and sibling replicas must remain alive
		if ctrl.Phase() != servingsupervision.PhaseReady {
			t.Fatalf("controller died: %s", ctrl.Phase())
		}
		if proxy.Phase() != servingsupervision.PhaseReady {
			t.Fatalf("proxy died: %s", proxy.Phase())
		}
		if !sibling1.IsHealthy() || !sibling2.IsHealthy() {
			t.Fatalf("sibling replicas were impacted by victim failure")
		}

		// Victim recovered with native engine receipt
		if receipt.Engine != servingsupervision.EngineNative || receipt.FallbackUsed {
			t.Fatalf("invalid receipt engine: %+v", receipt)
		}
		if !victim.IsHealthy() {
			t.Fatalf("victim should be healthy after recovery")
		}
	})

	t.Run("Kill one proxy: replicas remain loaded and endpoint identity preserved", func(t *testing.T) {
		topo, _ := servingsupervision.BuildDefaultTopology("proxy-test", 2, 50*time.Millisecond, 3)
		desired := servingsupervision.DesiredServingState{
			DeploymentID:  "proxy-test",
			ModelArtifact: "native-model",
			ReplicaCount:  2,
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo)
		_ = ctrl.Start(ctx)
		_, _ = ctrl.Reconcile(ctx, nil, nil)

		reps := ctrl.HealthyReplicas()
		proxy := ctrl.Proxy()
		endpointBefore := proxy.Endpoint()
		rep0Gen := reps[0].Generation()

		// Kill proxy
		receipt, err := proxy.Restart(ctx, reps)
		if err != nil {
			t.Fatalf("proxy restart: %v", err)
		}

		if proxy.Endpoint() != endpointBefore {
			t.Fatalf("endpoint identity changed: %s vs %s", proxy.Endpoint(), endpointBefore)
		}
		if reps[0].Generation() != rep0Gen {
			t.Fatalf("replica reloaded when only proxy was restarted")
		}
		if receipt.Engine != servingsupervision.EngineNative || receipt.FallbackUsed {
			t.Fatalf("receipt engine violated: %+v", receipt)
		}
	})

	t.Run("Kill serving controller: root restores it without worker teardown", func(t *testing.T) {
		topo, _ := servingsupervision.BuildDefaultTopology("ctrl-test", 2, 50*time.Millisecond, 3)
		desired := servingsupervision.DesiredServingState{
			DeploymentID:  "ctrl-test",
			ModelArtifact: "native-model",
			ReplicaCount:  2,
			DrainTimeout:  50 * time.Millisecond,
			RestartBudget: 3,
		}
		ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo)
		_ = ctrl.Start(ctx)
		_, _ = ctrl.Reconcile(ctx, nil, nil)
		reps := ctrl.HealthyReplicas()
		proxy := ctrl.Proxy()

		rep0Gen := reps[0].Generation()
		rep1Gen := reps[1].Generation()

		// Kill controller: root restarts it
		receipt, report, err := ctrl.Restart(ctx, reps, proxy)
		if err != nil {
			t.Fatalf("ctrl restart: %v", err)
		}

		if len(report.AdoptedReplicas) != 2 {
			t.Fatalf("adopted %d replicas, want 2", len(report.AdoptedReplicas))
		}
		if reps[0].Generation() != rep0Gen || reps[1].Generation() != rep1Gen {
			t.Fatalf("workers were torn down upon controller restart")
		}
		if receipt.Engine != servingsupervision.EngineNative || receipt.FallbackUsed {
			t.Fatalf("invalid controller receipt: %+v", receipt)
		}
	})

	t.Run("Request application exception does not restart healthy replica", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{DomainID: "rep-app-test", DrainTimeout: 50 * time.Millisecond, RestartBudget: 3}
		rep := servingsupervision.NewReplicaSupervisor(spec, "rep-app")
		_ = rep.Start(ctx)

		genBefore := rep.Generation()
		appErr := errors.New("request error: context length exceeded")

		receipt, err := rep.HandleError(ctx, appErr)
		if err != nil {
			t.Fatalf("handle error: %v", err)
		}
		if receipt.RestartScope != servingsupervision.ScopeNone {
			t.Fatalf("scope = %q, want %q", receipt.RestartScope, servingsupervision.ScopeNone)
		}
		if rep.Generation() != genBefore {
			t.Fatalf("replica restarted on request application error")
		}
		if !rep.IsHealthy() {
			t.Fatalf("replica unready on request error")
		}
	})

	t.Run("Corrupt model state replaces deployment domain", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{DomainID: "rep-corrupt-test", DrainTimeout: 50 * time.Millisecond, RestartBudget: 3}
		rep := servingsupervision.NewReplicaSupervisor(spec, "rep-corrupt")
		_ = rep.Start(ctx)

		corruptErr := errors.New("fatal: NaN tensor detected in model state corruption")
		receipt, err := rep.HandleError(ctx, corruptErr)
		if err != nil {
			t.Fatalf("handle error: %v", err)
		}

		if receipt.RestartScope != servingsupervision.ScopeDeploymentDomain {
			t.Fatalf("scope = %q, want %q", receipt.RestartScope, servingsupervision.ScopeDeploymentDomain)
		}
		if receipt.ErrorKind != servingsupervision.ErrorKindModelStateCorruption {
			t.Fatalf("error_kind = %q, want %q", receipt.ErrorKind, servingsupervision.ErrorKindModelStateCorruption)
		}
	})

	t.Run("Drain timeout produces explicit receipt with lost work", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{DomainID: "rep-drain-test", DrainTimeout: 20 * time.Millisecond, RestartBudget: 3}
		rep := servingsupervision.NewReplicaSupervisor(spec, "rep-drain")
		_ = rep.Start(ctx)

		// Force a drain via DrainManager with an unreleased inflight request
		dm := servingsupervision.NewDrainManager("d-test", "m-test", servingsupervision.RoleReplica, 20*time.Millisecond, 1)
		rel, err := dm.Acquire()
		if err != nil {
			t.Fatalf("acquire failed: %v", err)
		}
		defer rel()

		receipt, err := dm.Drain(ctx, servingsupervision.ErrWorkerProcessFailure, servingsupervision.ScopeLeafOnly, false, 2)
		if err != nil {
			t.Fatalf("drain error: %v", err)
		}
		if receipt.InflightLost != 1 {
			t.Fatalf("inflight_lost = %d, want 1", receipt.InflightLost)
		}
		if receipt.InflightDrained != 0 {
			t.Fatalf("inflight_drained = %d, want 0", receipt.InflightDrained)
		}
		if receipt.Engine != servingsupervision.EngineNative || receipt.FallbackUsed {
			t.Fatalf("invalid receipt: %+v", receipt)
		}
	})
}
