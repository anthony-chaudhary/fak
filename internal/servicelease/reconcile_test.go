package servicelease

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// fakeNode is the fake remote node the reconcile tests observe: its state is
// plain data, and evidence() renders it into the Evidence channels exactly as
// a live heartbeat + native read-back would. The fake clock is an explicit
// int64 the tests advance by hand — nothing reads wall time.
type fakeNode struct {
	name     string
	bootID   string
	phase    servicespec.Phase
	lastExit *servicespec.ExitRecord
	lastHBMS int64
}

func (n *fakeNode) evidence() Evidence {
	return Evidence{
		LastHeartbeatMS:    n.lastHBMS,
		HeartbeatBootID:    n.bootID,
		HeartbeatTimeoutMS: 10000,
		KnownBootID:        n.bootID,
		ReadBack: &servicespec.Observed{
			Schema:   servicespec.ObservedSchemaV1,
			Identity: servicespec.Identity{Node: n.name, Service: "svc"},
			Phase:    n.phase,
			LastExit: n.lastExit,
		},
	}
}

// crashedNode is a fake node whose process just crashed (fresh heartbeat,
// failed read-back, crash exit).
func crashedNode(name string, atMS int64) *fakeNode {
	return &fakeNode{
		name:     name,
		bootID:   name + "-boot-1",
		phase:    servicespec.PhaseFailed,
		lastExit: &servicespec.ExitRecord{Class: servicespec.ExitCrash, Code: 1, RunMS: 50},
		lastHBMS: atMS,
	}
}

func reconcileSpec(node string, desired servicespec.DesiredState) *servicespec.Spec {
	s := &servicespec.Spec{
		Schema:   servicespec.SchemaV1,
		Identity: servicespec.Identity{Node: node, Service: "svc"},
		Kind:     servicespec.KindService,
		Desired:  desired,
		Command:  []string{"fak", "serve"},
	}
	s.Normalize()
	return s
}

func TestStepWithoutAuthorityRefused(t *testing.T) {
	tb := NewTable(0)
	r := NewReconciler(tb, "ctrl-a", "epoch-1")
	n := crashedNode("node-1", 1000)
	_, err := r.Step(StepInput{Spec: reconcileSpec("node-1", servicespec.DesiredRunning), Evidence: n.evidence(), NowMS: 1000})
	if !errors.Is(err, ErrNoAuthority) {
		t.Fatalf("step without authority: err = %v, want ErrNoAuthority", err)
	}
}

// TestDesiredDriftProducesAction is the desired!=observed core: a crashed node
// under desired-running yields a paced restart (with growing backoff), a
// healthy node yields none, and desired-stopped over a still-running node
// yields a stop.
func TestDesiredDriftProducesAction(t *testing.T) {
	tb := NewTable(0)
	r := NewReconciler(tb, "ctrl-a", "epoch-1")
	clk := int64(1000)
	if _, err := r.AcquireAuthority("node-1", clk); err != nil {
		t.Fatalf("acquire authority: %v", err)
	}
	spec := reconcileSpec("node-1", servicespec.DesiredRunning)
	n := crashedNode("node-1", clk)

	res, err := r.Step(StepInput{Spec: spec, Evidence: n.evidence(), NowMS: clk})
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	if res.Plan.Action != ActionRestartLocal || res.Decision == nil || !res.Decision.Restart {
		t.Fatalf("crashed node under desired-running: plan=%+v decision=%+v", res.Plan, res.Decision)
	}
	if !res.Plan.DryRun {
		t.Fatalf("reconcile plan must be dry-run: %+v", res.Plan)
	}
	if res.Decision.DelayMS != servicespec.DefaultInitialBackoffMS {
		t.Fatalf("first restart delay = %d, want %d", res.Decision.DelayMS, servicespec.DefaultInitialBackoffMS)
	}

	clk += 2000
	n.lastHBMS = clk // still crashed at the next tick
	res, err = r.Step(StepInput{Spec: spec, Evidence: n.evidence(), NowMS: clk})
	if err != nil {
		t.Fatalf("second step: %v", err)
	}
	if !res.Decision.Restart || res.Decision.DelayMS != 2*servicespec.DefaultInitialBackoffMS {
		t.Fatalf("second restart not paced by doubled backoff: %+v", res.Decision)
	}

	clk += 1000
	n.phase, n.lastExit, n.lastHBMS = servicespec.PhaseReady, nil, clk
	res, err = r.Step(StepInput{Spec: spec, Evidence: n.evidence(), NowMS: clk})
	if err != nil {
		t.Fatalf("healthy step: %v", err)
	}
	if res.Plan.Action != ActionNone || res.Plan.Condition != CondHealthy || res.Decision != nil {
		t.Fatalf("healthy node must reconcile to none: plan=%+v decision=%+v", res.Plan, res.Decision)
	}

	// Intent flips to stopped while the node still runs: the loop chases it.
	stop := reconcileSpec("node-1", servicespec.DesiredStopped)
	clk += 1000
	n.lastHBMS = clk
	res, err = r.Step(StepInput{Spec: stop, Evidence: n.evidence(), NowMS: clk})
	if err != nil {
		t.Fatalf("stop step: %v", err)
	}
	if res.Plan.Action != ActionStop {
		t.Fatalf("desired-stopped over running node: action = %q, want %q", res.Plan.Action, ActionStop)
	}

	// Once the node is down, desired-stopped is met: nothing to do.
	clk += 1000
	n.phase, n.lastHBMS = servicespec.PhaseStopped, clk
	res, err = r.Step(StepInput{Spec: stop, Evidence: n.evidence(), NowMS: clk})
	if err != nil {
		t.Fatalf("stopped step: %v", err)
	}
	if res.Plan.Action != ActionNone || res.Plan.Reason != servicespec.ReasonDesiredStopped {
		t.Fatalf("desired-stopped met: plan=%+v", res.Plan)
	}
}

// TestStaleFenceRefused covers both stale-owner shapes: a rival controller is
// refused while the fence is valid, and after the fence moves on the OLD
// owner's steps refuse (superseded holder, then no authority at all). A
// restarted controller epoch supersedes its own previous epoch immediately.
func TestStaleFenceRefused(t *testing.T) {
	tb := NewTable(0) // 30s default authority TTL
	a := NewReconciler(tb, "ctrl-a", "epoch-1")
	b := NewReconciler(tb, "ctrl-b", "epoch-1")
	clk := int64(1000)
	if _, err := a.AcquireAuthority("node-1", clk); err != nil {
		t.Fatalf("a acquire: %v", err)
	}
	// Standby cannot fence in while the active lease is valid.
	if _, err := b.AcquireAuthority("node-1", clk+1); !errors.Is(err, ErrLeaseHeld) {
		t.Fatalf("rival acquire while lease valid: err = %v, want ErrLeaseHeld", err)
	}

	// The active controller goes silent past the TTL; the standby fences in.
	clk += DefaultLeaseTTLMS + 1
	if _, err := b.AcquireAuthority("node-1", clk); err != nil {
		t.Fatalf("b acquire after expiry: %v", err)
	}

	// The old owner wakes up and tries to reconcile: refused, no plan.
	n := crashedNode("node-1", clk)
	spec := reconcileSpec("node-1", servicespec.DesiredRunning)
	if _, err := a.Step(StepInput{Spec: spec, Evidence: n.evidence(), NowMS: clk}); !errors.Is(err, ErrNotHolder) {
		t.Fatalf("stale owner step: err = %v, want ErrNotHolder", err)
	}
	// The refusal dropped its authority copy: the next step refuses earlier.
	if _, err := a.Step(StepInput{Spec: spec, Evidence: n.evidence(), NowMS: clk}); !errors.Is(err, ErrNoAuthority) {
		t.Fatalf("stale owner second step: err = %v, want ErrNoAuthority", err)
	}

	// A restarted controller (new epoch) supersedes its own old epoch at once —
	// no TTL wait: the recorded incarnation invalidates the old lease.
	tb2 := NewTable(0)
	c1 := NewReconciler(tb2, "ctrl", "epoch-1")
	c2 := NewReconciler(tb2, "ctrl", "epoch-2")
	if _, err := c1.AcquireAuthority("node-2", 1000); err != nil {
		t.Fatalf("c1 acquire: %v", err)
	}
	if _, err := c2.AcquireAuthority("node-2", 1001); err != nil {
		t.Fatalf("c2 acquire after epoch bump: %v", err)
	}
	n2 := crashedNode("node-2", 1002)
	spec2 := reconcileSpec("node-2", servicespec.DesiredRunning)
	if _, err := c1.Step(StepInput{Spec: spec2, Evidence: n2.evidence(), NowMS: 1002}); !errors.Is(err, ErrStaleIncarnation) {
		t.Fatalf("old epoch step: err = %v, want ErrStaleIncarnation", err)
	}
}

// TestOwnerChangeNewOwnerReconciles drives the whole handoff on the fake
// clock and asserts the anti-double-restart invariant at every tick: across
// both controllers exactly one actionable restart is ever produced.
func TestOwnerChangeNewOwnerReconciles(t *testing.T) {
	tb := NewTable(0)
	a := NewReconciler(tb, "ctrl-a", "epoch-1")
	b := NewReconciler(tb, "ctrl-b", "epoch-1")
	clk := int64(1000)
	if _, err := a.AcquireAuthority("node-1", clk); err != nil {
		t.Fatalf("a acquire: %v", err)
	}
	spec := reconcileSpec("node-1", servicespec.DesiredRunning)
	n := crashedNode("node-1", clk)

	step := func(r *Reconciler) (StepResult, error) {
		n.lastHBMS = clk
		return r.Step(StepInput{Spec: spec, Evidence: n.evidence(), NowMS: clk})
	}
	restarts := func(results []StepResult, errs []error) int {
		c := 0
		for i := range results {
			if errs[i] == nil && results[i].Plan.Action == ActionRestartLocal && results[i].Decision != nil && results[i].Decision.Restart {
				c++
			}
		}
		return c
	}

	var aToken FencingToken
	// Phase 1: the active owner reconciles; the standby cannot even acquire.
	for i := 0; i < 3; i++ {
		clk += 1000
		ra, errA := step(a)
		_, errB := b.AcquireAuthority("node-1", clk)
		if !errors.Is(errB, ErrLeaseHeld) {
			t.Fatalf("tick %d: standby acquire err = %v, want ErrLeaseHeld", i, errB)
		}
		rb, errBStep := b.Step(StepInput{Spec: spec, Evidence: n.evidence(), NowMS: clk})
		if got := restarts([]StepResult{ra, rb}, []error{errA, errBStep}); got != 1 {
			t.Fatalf("tick %d: %d actionable restarts, want exactly 1", i, got)
		}
		aToken = ra.Token
	}

	// Phase 2: the active controller goes silent; its fence lapses and the
	// standby takes over. The new owner reconciles, the old owner is refused.
	clk += DefaultLeaseTTLMS + 1
	if _, err := b.AcquireAuthority("node-1", clk); err != nil {
		t.Fatalf("b acquire after lapse: %v", err)
	}
	for i := 0; i < 3; i++ {
		clk += 1000
		rb, errB := step(b)
		ra, errA := a.Step(StepInput{Spec: spec, Evidence: n.evidence(), NowMS: clk})
		if errA == nil {
			t.Fatalf("tick %d: old owner step succeeded after handoff: %+v", i, ra)
		}
		if got := restarts([]StepResult{ra, rb}, []error{errA, errB}); got != 1 {
			t.Fatalf("tick %d after handoff: %d actionable restarts, want exactly 1", i, got)
		}
		if rb.Token.LeaseSeq <= aToken.LeaseSeq {
			t.Fatalf("new owner token %+v does not supersede old %+v", rb.Token, aToken)
		}
	}
}

// TestCircuitOpensAfterRepeatedCrashes proves the loop converts a restart
// storm into a held circuit instead of restarting forever.
func TestCircuitOpensAfterRepeatedCrashes(t *testing.T) {
	tb := NewTable(0)
	r := NewReconciler(tb, "ctrl-a", "epoch-1")
	clk := int64(1000)
	if _, err := r.AcquireAuthority("node-1", clk); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	spec := reconcileSpec("node-1", servicespec.DesiredRunning)
	n := crashedNode("node-1", clk)

	for i := 0; i < servicespec.DefaultWindowMaxRestarts; i++ {
		clk += 1000
		n.lastHBMS = clk
		res, err := r.Step(StepInput{Spec: spec, Evidence: n.evidence(), NowMS: clk})
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		if res.Decision == nil || !res.Decision.Restart {
			t.Fatalf("step %d should still restart: %+v", i, res.Decision)
		}
	}
	clk += 1000
	n.lastHBMS = clk
	res, err := r.Step(StepInput{Spec: spec, Evidence: n.evidence(), NowMS: clk})
	if err != nil {
		t.Fatalf("circuit step: %v", err)
	}
	if res.Decision == nil || !res.Decision.CircuitOpen {
		t.Fatalf("window cap did not open the circuit: %+v", res.Decision)
	}
	if res.Plan.Action != ActionNone || res.Plan.Reason != servicespec.ReasonCircuitOpen {
		t.Fatalf("open circuit must hold, not restart: %+v", res.Plan)
	}
}
