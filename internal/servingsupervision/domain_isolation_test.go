package servingsupervision_test

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/servingsupervision"
)

// TestProxyFailureIsolation verifies that a proxy crash/restart restarts only the proxy domain;
// registered model replicas retain their in-memory model weights and remain in PhaseReady.
func TestProxyFailureIsolation(t *testing.T) {
	ctx := context.Background()

	// Track weight loading / initialization events per replica.
	var rep0WeightLoads int32
	var rep1WeightLoads int32

	rep0Spec := servingsupervision.ServingDomainSpec{
		DomainID:      "replica-domain-0",
		DrainTimeout:  50 * time.Millisecond,
		RestartBudget: 3,
		Role:          servingsupervision.RoleReplica,
	}
	rep1Spec := servingsupervision.ServingDomainSpec{
		DomainID:      "replica-domain-1",
		DrainTimeout:  50 * time.Millisecond,
		RestartBudget: 3,
		Role:          servingsupervision.RoleReplica,
	}

	rep0 := servingsupervision.NewReplicaSupervisor(
		rep0Spec,
		"rep-0",
		servingsupervision.WithReadinessCheck(func(ctx context.Context) error {
			atomic.AddInt32(&rep0WeightLoads, 1)
			return nil
		}),
	)
	rep1 := servingsupervision.NewReplicaSupervisor(
		rep1Spec,
		"rep-1",
		servingsupervision.WithReadinessCheck(func(ctx context.Context) error {
			atomic.AddInt32(&rep1WeightLoads, 1)
			return nil
		}),
	)

	if err := rep0.Start(ctx); err != nil {
		t.Fatalf("start replica 0: %v", err)
	}
	if err := rep1.Start(ctx); err != nil {
		t.Fatalf("start replica 1: %v", err)
	}

	// Model weights loaded once initially on replica start.
	if loads := atomic.LoadInt32(&rep0WeightLoads); loads != 1 {
		t.Fatalf("rep0 initial weight loads = %d, want 1", loads)
	}
	if loads := atomic.LoadInt32(&rep1WeightLoads); loads != 1 {
		t.Fatalf("rep1 initial weight loads = %d, want 1", loads)
	}

	rep0GenBefore := rep0.Generation()
	rep1GenBefore := rep1.Generation()

	var proxyRestarts int32
	proxySpec := servingsupervision.ServingDomainSpec{
		DomainID:      "proxy-ingress-domain",
		DrainTimeout:  50 * time.Millisecond,
		RestartBudget: 5,
		Role:          servingsupervision.RoleProxy,
	}
	proxy := servingsupervision.NewProxySupervisor(
		proxySpec,
		"proxy-main",
		"http://127.0.0.1:9090",
		servingsupervision.WithProxyRestartHook(func(ctx context.Context) error {
			atomic.AddInt32(&proxyRestarts, 1)
			return nil
		}),
	)

	if err := proxy.Start(ctx); err != nil {
		t.Fatalf("start proxy: %v", err)
	}
	if err := proxy.Reconstruct(ctx, []*servingsupervision.ReplicaSupervisor{rep0, rep1}); err != nil {
		t.Fatalf("proxy reconstruct: %v", err)
	}

	proxyGenBefore := proxy.Generation()
	proxyEndpointBefore := proxy.Endpoint()

	// Verify proxy can route to replicas before crash
	target1, err := proxy.Route()
	if err != nil {
		t.Fatalf("route before crash: %v", err)
	}
	if target1 == nil || !target1.IsHealthy() {
		t.Fatalf("routed target before crash is unready: %v", target1)
	}

	// Simulate proxy crash and independent restart with existing replicas
	receipt, err := proxy.Restart(ctx, []*servingsupervision.ReplicaSupervisor{rep0, rep1})
	if err != nil {
		t.Fatalf("proxy restart failed: %v", err)
	}

	// Assert proxy state
	if proxy.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("proxy phase = %q, want %q", proxy.Phase(), servingsupervision.PhaseReady)
	}
	if proxy.Generation() != proxyGenBefore+1 {
		t.Fatalf("proxy generation = %d, want %d", proxy.Generation(), proxyGenBefore+1)
	}
	if proxy.Endpoint() != proxyEndpointBefore {
		t.Fatalf("proxy endpoint changed: got %q, want %q", proxy.Endpoint(), proxyEndpointBefore)
	}
	if restarts := atomic.LoadInt32(&proxyRestarts); restarts != 1 {
		t.Fatalf("proxy restart hook calls = %d, want 1", restarts)
	}

	// Assert proxy receipt
	if receipt == nil {
		t.Fatal("proxy receipt is nil")
	}
	if receipt.Role != servingsupervision.RoleProxy {
		t.Fatalf("receipt role = %q, want %q", receipt.Role, servingsupervision.RoleProxy)
	}
	if receipt.RestartScope != servingsupervision.ScopeLeafOnly {
		t.Fatalf("receipt restart scope = %q, want %q", receipt.RestartScope, servingsupervision.ScopeLeafOnly)
	}

	// Replicas MUST NOT be torn down or have weights reloaded:
	if loads := atomic.LoadInt32(&rep0WeightLoads); loads != 1 {
		t.Fatalf("replica 0 reloaded weights during proxy crash: load count = %d, want 1", loads)
	}
	if loads := atomic.LoadInt32(&rep1WeightLoads); loads != 1 {
		t.Fatalf("replica 1 reloaded weights during proxy crash: load count = %d, want 1", loads)
	}
	if rep0.Generation() != rep0GenBefore {
		t.Fatalf("replica 0 generation changed: %d vs %d", rep0.Generation(), rep0GenBefore)
	}
	if rep1.Generation() != rep1GenBefore {
		t.Fatalf("replica 1 generation changed: %d vs %d", rep1.Generation(), rep1GenBefore)
	}
	if rep0.RestartCount() != 0 || rep1.RestartCount() != 0 {
		t.Fatalf("replicas had restarts: rep0=%d, rep1=%d", rep0.RestartCount(), rep1.RestartCount())
	}
	if !rep0.IsHealthy() || rep0.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("replica 0 became unready: healthy=%v, phase=%s", rep0.IsHealthy(), rep0.Phase())
	}
	if !rep1.IsHealthy() || rep1.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("replica 1 became unready: healthy=%v, phase=%s", rep1.IsHealthy(), rep1.Phase())
	}

	// Proxy can route to both replicas after restart
	seen := make(map[string]bool)
	for i := 0; i < 4; i++ {
		target, routeErr := proxy.Route()
		if routeErr != nil {
			t.Fatalf("route after proxy restart failed: %v", routeErr)
		}
		seen[target.ReplicaID()] = true
	}
	if !seen["rep-0"] || !seen["rep-1"] {
		t.Fatalf("proxy did not route to both replicas after restart: %v", seen)
	}
}

// TestReplicaFailureIsolation verifies that a crash in one replica isolates to that replica;
// controller, proxy, and sibling replicas remain in PhaseReady and continue serving traffic.
func TestReplicaFailureIsolation(t *testing.T) {
	ctx := context.Background()

	topo, err := servingsupervision.BuildDefaultTopology("isolation-cluster", 3, 50*time.Millisecond, 3)
	if err != nil {
		t.Fatalf("build topology: %v", err)
	}

	desired := servingsupervision.DesiredServingState{
		DeploymentID:  "isolation-cluster",
		ModelArtifact: "fak-model:qwen38",
		ReplicaCount:  3,
		DrainTimeout:  50 * time.Millisecond,
		RestartBudget: 3,
	}

	ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo)
	if err := ctrl.Start(ctx); err != nil {
		t.Fatalf("start controller: %v", err)
	}
	if _, err := ctrl.Reconcile(ctx, nil, nil); err != nil {
		t.Fatalf("reconcile controller: %v", err)
	}

	reps := ctrl.HealthyReplicas()
	if len(reps) != 3 {
		t.Fatalf("healthy replicas count = %d, want 3", len(reps))
	}
	proxy := ctrl.Proxy()
	if proxy == nil {
		t.Fatal("proxy is nil")
	}

	victim := reps[0]
	sibling1 := reps[1]
	sibling2 := reps[2]

	ctrlGenBefore := ctrl.Generation()
	proxyGenBefore := proxy.Generation()
	sib1GenBefore := sibling1.Generation()
	sib2GenBefore := sibling2.Generation()
	victimGenBefore := victim.Generation()

	// Crash the victim replica
	victimReceipt, victimErr := victim.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
	if victimErr != nil {
		t.Fatalf("victim handle error: %v", victimErr)
	}

	// Verify failure receipt isolates to leaf only
	if victimReceipt.RestartScope != servingsupervision.ScopeLeafOnly {
		t.Fatalf("victim restart scope = %q, want %q", victimReceipt.RestartScope, servingsupervision.ScopeLeafOnly)
	}
	if victimReceipt.MemberID != victim.ReplicaID() {
		t.Fatalf("victim receipt member id = %q, want %q", victimReceipt.MemberID, victim.ReplicaID())
	}
	if victimReceipt.ErrorKind != servingsupervision.ErrorKindWorkerProcessFailure {
		t.Fatalf("victim error kind = %q, want %q", victimReceipt.ErrorKind, servingsupervision.ErrorKindWorkerProcessFailure)
	}

	// Controller MUST remain in PhaseReady, generation unchanged
	if ctrl.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("controller phase impacted: got %q, want %q", ctrl.Phase(), servingsupervision.PhaseReady)
	}
	if ctrl.Generation() != ctrlGenBefore {
		t.Fatalf("controller generation changed: %d vs %d", ctrl.Generation(), ctrlGenBefore)
	}

	// Proxy MUST remain in PhaseReady, generation unchanged
	if proxy.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("proxy phase impacted: got %q, want %q", proxy.Phase(), servingsupervision.PhaseReady)
	}
	if proxy.Generation() != proxyGenBefore {
		t.Fatalf("proxy generation changed: %d vs %d", proxy.Generation(), proxyGenBefore)
	}

	// Sibling replicas MUST stay healthy, generations unchanged, restart count 0
	if !sibling1.IsHealthy() || sibling1.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("sibling1 disrupted: healthy=%v, phase=%s", sibling1.IsHealthy(), sibling1.Phase())
	}
	if !sibling2.IsHealthy() || sibling2.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("sibling2 disrupted: healthy=%v, phase=%s", sibling2.IsHealthy(), sibling2.Phase())
	}
	if sibling1.Generation() != sib1GenBefore || sibling1.RestartCount() != 0 {
		t.Fatalf("sibling1 generation or restart count changed: gen=%d, restarts=%d", sibling1.Generation(), sibling1.RestartCount())
	}
	if sibling2.Generation() != sib2GenBefore || sibling2.RestartCount() != 0 {
		t.Fatalf("sibling2 generation or restart count changed: gen=%d, restarts=%d", sibling2.Generation(), sibling2.RestartCount())
	}

	// Proxy continues serving traffic through healthy siblings
	for i := 0; i < 6; i++ {
		target, routeErr := proxy.Route()
		if routeErr != nil {
			t.Fatalf("routing through proxy failed during replica failure: %v", routeErr)
		}
		if !target.IsHealthy() {
			t.Fatalf("proxy routed to unready target: %s", target.ReplicaID())
		}
		execErr := target.Execute(ctx, func() error { return nil })
		if execErr != nil {
			t.Fatalf("executing request on healthy replica failed: %v", execErr)
		}
	}

	// Victim replica recovered cleanly
	if !victim.IsHealthy() || victim.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("victim replica failed to recover to PhaseReady: healthy=%v, phase=%s", victim.IsHealthy(), victim.Phase())
	}
	if victim.Generation() != victimGenBefore+1 {
		t.Fatalf("victim generation = %d, want %d", victim.Generation(), victimGenBefore+1)
	}
	if victim.RestartCount() != 1 {
		t.Fatalf("victim restart count = %d, want 1", victim.RestartCount())
	}
}

// TestControllerRestorationFromDesiredState verifies that a controller crash and replacement
// reconstructs control plane state from DesiredServingState and non-destructively adopts
// existing healthy replicas and proxy without restarting or disrupting them.
func TestControllerRestorationFromDesiredState(t *testing.T) {
	ctx := context.Background()

	topo, err := servingsupervision.BuildDefaultTopology("restore-cluster", 2, 50*time.Millisecond, 3)
	if err != nil {
		t.Fatalf("build default topology: %v", err)
	}

	desired := servingsupervision.DesiredServingState{
		DeploymentID:  "restore-cluster",
		ModelArtifact: "fak-model:qwen38",
		ReplicaCount:  2,
		DrainTimeout:  50 * time.Millisecond,
		RestartBudget: 3,
	}

	// Start initial controller
	ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo)
	if err := ctrl.Start(ctx); err != nil {
		t.Fatalf("start initial controller: %v", err)
	}

	initialReport, err := ctrl.Reconcile(ctx, nil, nil)
	if err != nil {
		t.Fatalf("initial reconcile: %v", err)
	}
	if len(initialReport.CreatedReplicas) != 2 {
		t.Fatalf("initial created replicas = %d, want 2", len(initialReport.CreatedReplicas))
	}

	existingReplicas := ctrl.HealthyReplicas()
	existingProxy := ctrl.Proxy()

	rep0 := existingReplicas[0]
	rep1 := existingReplicas[1]
	rep0GenBefore := rep0.Generation()
	rep1GenBefore := rep1.Generation()
	proxyGenBefore := existingProxy.Generation()
	proxyEndpointBefore := existingProxy.Endpoint()

	// Scenario A: Controller in-place Restart
	restartReceipt, restartReport, restartErr := ctrl.Restart(ctx, existingReplicas, existingProxy)
	if restartErr != nil {
		t.Fatalf("controller in-place restart: %v", restartErr)
	}
	if restartReceipt == nil || restartReceipt.Role != servingsupervision.RoleController {
		t.Fatalf("invalid controller restart receipt: %+v", restartReceipt)
	}
	if len(restartReport.AdoptedReplicas) != 2 {
		t.Fatalf("in-place restart adopted = %d, want 2", len(restartReport.AdoptedReplicas))
	}
	if len(restartReport.CreatedReplicas) != 0 {
		t.Fatalf("in-place restart created = %d, want 0", len(restartReport.CreatedReplicas))
	}
	if !restartReport.PreservedProxy {
		t.Fatal("in-place restart did not preserve proxy")
	}

	// Scenario B: Total controller crash and replacement from desired state
	newCtrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo)
	if err := newCtrl.Start(ctx); err != nil {
		t.Fatalf("start replacement controller: %v", err)
	}

	adoptionReport, err := newCtrl.Reconcile(ctx, existingReplicas, existingProxy)
	if err != nil {
		t.Fatalf("replacement controller reconcile: %v", err)
	}

	// Check adoption results
	if len(adoptionReport.AdoptedReplicas) != 2 {
		t.Fatalf("adopted replicas count = %d, want 2", len(adoptionReport.AdoptedReplicas))
	}
	if len(adoptionReport.CreatedReplicas) != 0 {
		t.Fatalf("created replicas count during adoption = %d, want 0", len(adoptionReport.CreatedReplicas))
	}
	if len(adoptionReport.RemovedReplicas) != 0 {
		t.Fatalf("removed replicas count during adoption = %d, want 0", len(adoptionReport.RemovedReplicas))
	}
	if !adoptionReport.PreservedProxy {
		t.Fatal("proxy was not preserved during adoption")
	}

	// Replicas MUST NOT have restarted or changed generation
	if rep0.Generation() != rep0GenBefore {
		t.Fatalf("rep0 restarted during controller replacement: %d vs %d", rep0.Generation(), rep0GenBefore)
	}
	if rep1.Generation() != rep1GenBefore {
		t.Fatalf("rep1 restarted during controller replacement: %d vs %d", rep1.Generation(), rep1GenBefore)
	}
	if rep0.RestartCount() != 0 || rep1.RestartCount() != 0 {
		t.Fatalf("replicas had restarts: rep0=%d, rep1=%d", rep0.RestartCount(), rep1.RestartCount())
	}
	if !rep0.IsHealthy() || !rep1.IsHealthy() {
		t.Fatalf("replicas not healthy after adoption: rep0=%v, rep1=%v", rep0.IsHealthy(), rep1.IsHealthy())
	}

	// Proxy MUST retain generation and endpoint
	if existingProxy.Generation() != proxyGenBefore {
		t.Fatalf("proxy generation changed during adoption: %d vs %d", existingProxy.Generation(), proxyGenBefore)
	}
	if existingProxy.Endpoint() != proxyEndpointBefore {
		t.Fatalf("proxy endpoint changed during adoption: %s vs %s", existingProxy.Endpoint(), proxyEndpointBefore)
	}

	// New controller is functional and can route traffic through preserved proxy
	if newCtrl.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("new controller phase = %q, want %q", newCtrl.Phase(), servingsupervision.PhaseReady)
	}
	routed, routeErr := newCtrl.Proxy().Route()
	if routeErr != nil {
		t.Fatalf("route after controller adoption failed: %v", routeErr)
	}
	if routed == nil || !routed.IsHealthy() {
		t.Fatalf("routed replica is unready: %v", routed)
	}
}

// TestRequestErrorDoesNotRestartReplica verifies that application/client errors
// (invalid argument, context length exceeded, prompt validation failure) do not
// trigger replica restarts or change serving state.
func TestRequestErrorDoesNotRestartReplica(t *testing.T) {
	ctx := context.Background()

	var restartHookCalls int32
	spec := servingsupervision.ServingDomainSpec{
		DomainID:      "req-error-domain",
		DrainTimeout:  50 * time.Millisecond,
		RestartBudget: 3,
		Role:          servingsupervision.RoleReplica,
	}
	rep := servingsupervision.NewReplicaSupervisor(
		spec,
		"rep-req-err",
		servingsupervision.WithReplicaRestartHook(func(ctx context.Context, replicaID string) error {
			atomic.AddInt32(&restartHookCalls, 1)
			return nil
		}),
	)

	if err := rep.Start(ctx); err != nil {
		t.Fatalf("replica start: %v", err)
	}

	genBefore := rep.Generation()

	appErrors := []error{
		servingsupervision.ErrRequestApplication,
		errors.New("request error: context length exceeded"),
		errors.New("bad request: invalid JSON prompt payload"),
		errors.New("invalid argument: temperature -1.0 out of range"),
		errors.New("prompt error: max token budget exceeded"),
		errors.New("user error: model input validation failed"),
		servingsupervision.WrapClassifiedError(
			servingsupervision.ErrorKindRequestApplication,
			servingsupervision.ScopeNone,
			errors.New("custom wrapped application error"),
		),
	}

	for i, appErr := range appErrors {
		t.Run(fmt.Sprintf("app_error_%d", i), func(t *testing.T) {
			// Test execution through Execute
			execErr := rep.Execute(ctx, func() error {
				return appErr
			})
			if !errors.Is(execErr, appErr) && execErr != appErr {
				t.Fatalf("expected original error %v, got %v", appErr, execErr)
			}

			// Verify receipt produced for request error
			receipt := rep.LastReceipt()
			if receipt == nil {
				t.Fatal("receipt is nil for request error")
			}
			if receipt.RestartScope != servingsupervision.ScopeNone {
				t.Fatalf("restart scope = %q, want %q", receipt.RestartScope, servingsupervision.ScopeNone)
			}
			if receipt.ErrorKind != servingsupervision.ErrorKindRequestApplication {
				t.Fatalf("error kind = %q, want %q", receipt.ErrorKind, servingsupervision.ErrorKindRequestApplication)
			}
			if receipt.Quarantined {
				t.Fatal("replica marked quarantined for request application error")
			}
			if receipt.ObservedGen != genBefore || receipt.NextGen != genBefore {
				t.Fatalf("generation bumped on request error: observed %d, next %d", receipt.ObservedGen, receipt.NextGen)
			}

			// Replica state must be completely unaffected
			if rep.Generation() != genBefore {
				t.Fatalf("replica generation bumped: %d vs %d", rep.Generation(), genBefore)
			}
			if rep.RestartCount() != 0 {
				t.Fatalf("replica restart count incremented: %d", rep.RestartCount())
			}
			if !rep.IsHealthy() {
				t.Fatal("replica became unready after request error")
			}
			if rep.Phase() != servingsupervision.PhaseReady {
				t.Fatalf("replica phase = %q, want %q", rep.Phase(), servingsupervision.PhaseReady)
			}
			if calls := atomic.LoadInt32(&restartHookCalls); calls != 0 {
				t.Fatalf("restart hook was called %d times for request errors", calls)
			}

			// Replica can immediately serve a subsequent valid request
			validErr := rep.Execute(ctx, func() error {
				return nil
			})
			if validErr != nil {
				t.Fatalf("valid request execution failed: %v", validErr)
			}
		})
	}
}

// TestQuarantineOnBudgetExhaustion verifies that repeated crashes exhaust the restart budget,
// causing the replica to transition to PhaseQuarantined and reject subsequent traffic.
func TestQuarantineOnBudgetExhaustion(t *testing.T) {
	ctx := context.Background()

	const budget = 2
	spec := servingsupervision.ServingDomainSpec{
		DomainID:      "quarantine-domain",
		DrainTimeout:  30 * time.Millisecond,
		RestartBudget: budget,
		Role:          servingsupervision.RoleReplica,
	}

	rep := servingsupervision.NewReplicaSupervisor(spec, "rep-quarantine")
	if err := rep.Start(ctx); err != nil {
		t.Fatalf("start replica: %v", err)
	}

	// Crashes within budget should recover to PhaseReady
	for i := 1; i <= budget; i++ {
		receipt, err := rep.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
		if err != nil {
			t.Fatalf("crash %d failed to recover within budget: %v", i, err)
		}
		if receipt.RestartScope != servingsupervision.ScopeLeafOnly {
			t.Fatalf("crash %d restart scope = %q, want %q", i, receipt.RestartScope, servingsupervision.ScopeLeafOnly)
		}
		if receipt.Quarantined {
			t.Fatalf("crash %d unexpectedly quarantined", i)
		}
		if rep.RestartCount() != i {
			t.Fatalf("crash %d restart count = %d, want %d", i, rep.RestartCount(), i)
		}
		if rep.Quarantined() {
			t.Fatalf("replica quarantined prematurely on crash %d", i)
		}
		if !rep.IsHealthy() || rep.Phase() != servingsupervision.PhaseReady {
			t.Fatalf("replica not ready after crash %d: healthy=%v, phase=%s", i, rep.IsHealthy(), rep.Phase())
		}
	}

	// Next crash: budget exhausted (restartCount == budget) -> quarantine!
	receipt, err := rep.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
	if !errors.Is(err, servingsupervision.ErrBudgetExhausted) {
		t.Fatalf("expected ErrBudgetExhausted, got %v", err)
	}
	if receipt == nil {
		t.Fatal("receipt is nil on quarantine")
	}
	if !receipt.Quarantined {
		t.Fatal("receipt does not reflect quarantined status")
	}
	if receipt.RestartScope != servingsupervision.ScopeQuarantine {
		t.Fatalf("receipt restart scope = %q, want %q", receipt.RestartScope, servingsupervision.ScopeQuarantine)
	}

	// Replica state must be quarantined
	if !rep.Quarantined() {
		t.Fatal("replica Quarantined() is false")
	}
	if rep.Phase() != servingsupervision.PhaseQuarantined {
		t.Fatalf("replica phase = %q, want %q", rep.Phase(), servingsupervision.PhaseQuarantined)
	}
	if rep.IsHealthy() {
		t.Fatal("quarantined replica reports healthy")
	}

	// Traffic MUST be withdrawn and rejected
	reqErr := rep.Execute(ctx, func() error { return nil })
	if !errors.Is(reqErr, servingsupervision.ErrTrafficWithdrawn) {
		t.Fatalf("expected ErrTrafficWithdrawn on quarantined replica, got %v", reqErr)
	}

	// Proxy must exclude quarantined replica
	proxySpec := servingsupervision.ServingDomainSpec{
		DomainID:      "proxy-quarantine-test",
		DrainTimeout:  30 * time.Millisecond,
		RestartBudget: 3,
		Role:          servingsupervision.RoleProxy,
	}
	proxy := servingsupervision.NewProxySupervisor(proxySpec, "proxy-q", "http://127.0.0.1:9090")
	if err := proxy.Start(ctx); err != nil {
		t.Fatalf("proxy start: %v", err)
	}

	// Create another healthy replica
	healthySpec := servingsupervision.ServingDomainSpec{
		DomainID:      "healthy-domain",
		DrainTimeout:  30 * time.Millisecond,
		RestartBudget: 3,
		Role:          servingsupervision.RoleReplica,
	}
	healthyRep := servingsupervision.NewReplicaSupervisor(healthySpec, "rep-healthy")
	if err := healthyRep.Start(ctx); err != nil {
		t.Fatalf("start healthy replica: %v", err)
	}

	if err := proxy.Reconstruct(ctx, []*servingsupervision.ReplicaSupervisor{rep, healthyRep}); err != nil {
		t.Fatalf("proxy reconstruct: %v", err)
	}

	// All routed traffic must go exclusively to the healthy replica
	for i := 0; i < 5; i++ {
		target, routeErr := proxy.Route()
		if routeErr != nil {
			t.Fatalf("proxy route failed: %v", routeErr)
		}
		if target.ReplicaID() != "rep-healthy" {
			t.Fatalf("proxy routed to quarantined replica: %s", target.ReplicaID())
		}
	}

	// Operator reset of quarantine restores service
	if err := rep.ResetQuarantine(ctx); err != nil {
		t.Fatalf("reset quarantine failed: %v", err)
	}
	if rep.Quarantined() {
		t.Fatal("replica still quarantined after reset")
	}
	if !rep.IsHealthy() || rep.Phase() != servingsupervision.PhaseReady {
		t.Fatalf("replica not ready after quarantine reset: healthy=%v, phase=%s", rep.IsHealthy(), rep.Phase())
	}
	if rep.RestartCount() != 0 {
		t.Fatalf("restart count not cleared on quarantine reset: %d", rep.RestartCount())
	}
	if err := rep.Execute(ctx, func() error { return nil }); err != nil {
		t.Fatalf("request execution failed after quarantine reset: %v", err)
	}
}

// TestServingReceiptNativeContract enforces that all serving receipts strictly adhere to
// the FAK native execution contract: schema is fak-serving-receipt/1, engine is native
// or inkernel, and FallbackUsed is strictly false (no silent fallback to non-native engines).
func TestServingReceiptNativeContract(t *testing.T) {
	ctx := context.Background()

	assertReceiptContract := func(t *testing.T, r *servingsupervision.ServingReceipt, expectedRole servingsupervision.ServingRole) {
		t.Helper()
		if r == nil {
			t.Fatal("expected non-nil ServingReceipt")
		}
		if r.Schema != servingsupervision.ServingReceiptSchema {
			t.Errorf("schema = %q, want %q", r.Schema, servingsupervision.ServingReceiptSchema)
		}
		if r.Engine != servingsupervision.EngineNative && r.Engine != servingsupervision.EngineInKernel {
			t.Errorf("engine = %q, must be native or inkernel", r.Engine)
		}
		if r.FallbackUsed {
			t.Errorf("FallbackUsed = true; native inference contract strictly forbids fallback")
		}
		if r.Timestamp.IsZero() {
			t.Errorf("timestamp is zero")
		}
		if r.DomainID == "" {
			t.Errorf("domain_id is empty")
		}
		if r.MemberID == "" {
			t.Errorf("member_id is empty")
		}
		if expectedRole != "" && r.Role != expectedRole {
			t.Errorf("role = %q, want %q", r.Role, expectedRole)
		}
	}

	t.Run("DrainManager receipt contract", func(t *testing.T) {
		dm := servingsupervision.NewDrainManager("dm-domain", "dm-member", servingsupervision.RoleReplica, 20*time.Millisecond, 1)
		rel, err := dm.Acquire()
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		rel()

		receipt, err := dm.Drain(ctx, servingsupervision.ErrWorkerProcessFailure, servingsupervision.ScopeLeafOnly, false, 2)
		if err != nil {
			t.Fatalf("drain error: %v", err)
		}
		assertReceiptContract(t, receipt, servingsupervision.RoleReplica)
	})

	t.Run("ReplicaSupervisor request error receipt contract", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{DomainID: "rep-receipt-1", DrainTimeout: 20 * time.Millisecond, RestartBudget: 2}
		rep := servingsupervision.NewReplicaSupervisor(spec, "rep-1", servingsupervision.WithReplicaBackend(servingsupervision.EngineNative))
		if err := rep.Start(ctx); err != nil {
			t.Fatalf("start: %v", err)
		}

		receipt, err := rep.HandleError(ctx, servingsupervision.ErrRequestApplication)
		if err != nil {
			t.Fatalf("handle error: %v", err)
		}
		assertReceiptContract(t, receipt, servingsupervision.RoleReplica)
		if receipt.RestartScope != servingsupervision.ScopeNone {
			t.Errorf("restart scope = %q, want %q", receipt.RestartScope, servingsupervision.ScopeNone)
		}
	})

	t.Run("ReplicaSupervisor leaf restart receipt contract", func(t *testing.T) {
		spec := servingsupervision.ServingDomainSpec{DomainID: "rep-receipt-2", DrainTimeout: 20 * time.Millisecond, RestartBudget: 2}
		rep := servingsupervision.NewReplicaSupervisor(spec, "rep-2", servingsupervision.WithReplicaBackend(servingsupervision.EngineInKernel))
		if err := rep.Start(ctx); err != nil {
			t.Fatalf("start: %v", err)
		}

		receipt, err := rep.HandleError(ctx, servingsupervision.ErrWorkerProcessFailure)
		if err != nil {
			t.Fatalf("handle error: %v", err)
		}
		assertReceiptContract(t, receipt, servingsupervision.RoleReplica)
		if receipt.Engine != servingsupervision.EngineInKernel {
			t.Errorf("engine = %q, want %q", receipt.Engine, servingsupervision.EngineInKernel)
		}
		if receipt.RestartScope != servingsupervision.ScopeLeafOnly {
			t.Errorf("restart scope = %q, want %q", receipt.RestartScope, servingsupervision.ScopeLeafOnly)
		}
	})

	t.Run("ProxySupervisor restart receipt contract", func(t *testing.T) {
		repSpec := servingsupervision.ServingDomainSpec{DomainID: "rep-p-receipt", DrainTimeout: 20 * time.Millisecond, RestartBudget: 2}
		rep := servingsupervision.NewReplicaSupervisor(repSpec, "rep-p")
		if err := rep.Start(ctx); err != nil {
			t.Fatalf("start rep: %v", err)
		}

		proxySpec := servingsupervision.ServingDomainSpec{DomainID: "proxy-receipt-domain", DrainTimeout: 20 * time.Millisecond, RestartBudget: 3}
		proxy := servingsupervision.NewProxySupervisor(proxySpec, "proxy-receipt", "http://127.0.0.1:9090", servingsupervision.WithProxyBackend(servingsupervision.EngineNative))
		if err := proxy.Start(ctx); err != nil {
			t.Fatalf("start proxy: %v", err)
		}

		receipt, err := proxy.Restart(ctx, []*servingsupervision.ReplicaSupervisor{rep})
		if err != nil {
			t.Fatalf("proxy restart: %v", err)
		}
		assertReceiptContract(t, receipt, servingsupervision.RoleProxy)
	})

	t.Run("ControllerSupervisor restart receipt contract", func(t *testing.T) {
		topo, err := servingsupervision.BuildDefaultTopology("receipt-cluster", 1, 20*time.Millisecond, 2)
		if err != nil {
			t.Fatalf("build topo: %v", err)
		}
		desired := servingsupervision.DesiredServingState{
			DeploymentID:  "receipt-cluster",
			ModelArtifact: "fak-model:qwen38",
			ReplicaCount:  1,
			DrainTimeout:  20 * time.Millisecond,
			RestartBudget: 2,
		}
		ctrl := servingsupervision.NewControllerSupervisor(topo.Controller, desired, topo, servingsupervision.WithControllerBackend(servingsupervision.EngineNative))
		if err := ctrl.Start(ctx); err != nil {
			t.Fatalf("start ctrl: %v", err)
		}
		if _, err := ctrl.Reconcile(ctx, nil, nil); err != nil {
			t.Fatalf("reconcile: %v", err)
		}

		receipt, _, err := ctrl.Restart(ctx, ctrl.HealthyReplicas(), ctrl.Proxy())
		if err != nil {
			t.Fatalf("ctrl restart: %v", err)
		}
		assertReceiptContract(t, receipt, servingsupervision.RoleController)
	})
}
