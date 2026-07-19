package servicelease

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

const wl = "bench-runner"

func boot(node, id string) Incarnation { return Incarnation{Node: node, BootID: id} }

// mustAcquire is a test helper: acquire or fail the test.
func mustAcquire(t *testing.T, tb *Table, w string, by Incarnation, now int64) *Lease {
	t.Helper()
	l, err := tb.Acquire(w, by, now)
	if err != nil {
		t.Fatalf("Acquire(%s by %s@%s) = %v", w, by.Node, by.BootID, err)
	}
	return l
}

func TestStaleIncarnationCannotRenewOrPublish(t *testing.T) {
	tb := NewTable(10000)
	old := boot("n1", "b1")
	tb.RecordIncarnation(old)
	l := mustAcquire(t, tb, wl, old, 0)

	// The host reboots: recording the new boot supersedes the old forever.
	if !tb.RecordIncarnation(boot("n1", "b2")) {
		t.Fatal("recording a new boot ID must report a change")
	}
	if _, err := tb.Renew(wl, old, l.Token, 1000); !errors.Is(err, ErrStaleIncarnation) {
		t.Fatalf("stale-incarnation renew = %v, want ErrStaleIncarnation", err)
	}
	if err := tb.PublishCompletion(wl, old, l.Token, Checkpoint{Seq: 1}); !errors.Is(err, ErrStaleIncarnation) {
		t.Fatalf("stale-incarnation publish = %v, want ErrStaleIncarnation", err)
	}
	if got := tb.ValidOwner(wl, 1000); !got.Zero() {
		t.Fatalf("a lease held by a superseded incarnation must not be valid, got owner %+v", got)
	}
}

func TestGenerationBumpFencesOutstandingToken(t *testing.T) {
	tb := NewTable(10000)
	inc := boot("n1", "b1")
	tb.RecordIncarnation(inc)
	l := mustAcquire(t, tb, wl, inc, 0)

	if g := tb.BumpGeneration(wl); g != 1 {
		t.Fatalf("BumpGeneration = %d, want 1", g)
	}
	if _, err := tb.Renew(wl, inc, l.Token, 1000); !errors.Is(err, ErrFenced) {
		t.Fatalf("old-generation renew = %v, want ErrFenced", err)
	}
	if err := tb.PublishCompletion(wl, inc, l.Token, Checkpoint{Seq: 1}); !errors.Is(err, ErrFenced) {
		t.Fatalf("old-generation publish = %v, want ErrFenced", err)
	}
	ok, why := RemoteReassignAllowed(tb, wl, 1000)
	if !ok || why != "generation-bumped" {
		t.Fatalf("RemoteReassignAllowed after bump = %v %q, want true generation-bumped", ok, why)
	}
	// Re-acquiring mints a token under the new generation, which works again.
	l2 := mustAcquire(t, tb, wl, inc, 1000)
	if l2.Token.Generation != 1 {
		t.Fatalf("re-acquired token generation = %d, want 1", l2.Token.Generation)
	}
	if _, err := tb.Renew(wl, inc, l2.Token, 2000); err != nil {
		t.Fatalf("new-generation renew = %v, want nil", err)
	}
}

func TestAcquireRefusedWhileValidLeaseHeld(t *testing.T) {
	tb := NewTable(10000)
	a, b := boot("n1", "b1"), boot("n2", "b1")
	tb.RecordIncarnation(a)
	tb.RecordIncarnation(b)
	la := mustAcquire(t, tb, wl, a, 0)

	if _, err := tb.Acquire(wl, b, 5000); !errors.Is(err, ErrLeaseHeld) {
		t.Fatalf("takeover of a valid lease = %v, want ErrLeaseHeld", err)
	}
	if ok, why := RemoteReassignAllowed(tb, wl, 5000); ok || why != "lease-valid" {
		t.Fatalf("RemoteReassignAllowed mid-lease = %v %q, want false lease-valid", ok, why)
	}

	// After clock expiry the takeover is permitted, and the old token dies.
	if ok, why := RemoteReassignAllowed(tb, wl, 10000); !ok || why != "lease-expired" {
		t.Fatalf("RemoteReassignAllowed at expiry = %v %q, want true lease-expired", ok, why)
	}
	lb := mustAcquire(t, tb, wl, b, 10000)
	if lb.Token.LeaseSeq <= la.Token.LeaseSeq {
		t.Fatalf("new lease seq %d must exceed old %d", lb.Token.LeaseSeq, la.Token.LeaseSeq)
	}
	if _, err := tb.Renew(wl, a, la.Token, 11000); !errors.Is(err, ErrNotHolder) {
		t.Fatalf("superseded-owner renew = %v, want ErrNotHolder", err)
	}
	if err := tb.PublishCompletion(wl, a, la.Token, Checkpoint{Seq: 9}); !errors.Is(err, ErrNotHolder) {
		t.Fatalf("superseded-owner publish = %v, want ErrNotHolder", err)
	}
	if got := tb.ValidOwner(wl, 11000); got != b {
		t.Fatalf("valid owner = %+v, want %+v", got, b)
	}
}

func TestRenewExpiredLeaseRefused(t *testing.T) {
	tb := NewTable(10000)
	inc := boot("n1", "b1")
	tb.RecordIncarnation(inc)
	l := mustAcquire(t, tb, wl, inc, 0)
	if _, err := tb.Renew(wl, inc, l.Token, 10000); !errors.Is(err, ErrLeaseExpired) {
		t.Fatalf("expired renew = %v, want ErrLeaseExpired", err)
	}
	// The holder recovers by re-acquiring (nobody else took it).
	if _, err := tb.Acquire(wl, inc, 10000); err != nil {
		t.Fatalf("holder re-acquire after expiry = %v, want nil", err)
	}
}

func TestPublishCompletionCheckpointRules(t *testing.T) {
	tb := NewTable(10000)
	inc := boot("n1", "b1")
	tb.RecordIncarnation(inc)
	l := mustAcquire(t, tb, wl, inc, 0)

	if err := tb.PublishCompletion(wl, inc, l.Token, Checkpoint{Seq: 3, ID: "c3"}); err != nil {
		t.Fatalf("publish = %v, want nil", err)
	}
	if err := tb.PublishCompletion(wl, inc, l.Token, Checkpoint{Seq: 2}); !errors.Is(err, ErrCheckpointRegression) {
		t.Fatalf("regressing publish = %v, want ErrCheckpointRegression", err)
	}
	// Expiry alone does not refuse a publish under the still-newest token.
	if err := tb.PublishCompletion(wl, inc, l.Token, Checkpoint{Seq: 4}); err != nil {
		t.Fatalf("post-expiry publish with newest token = %v, want nil", err)
	}
	// Progress survives an ownership change.
	other := boot("n2", "b1")
	tb.RecordIncarnation(other)
	l2 := mustAcquire(t, tb, wl, other, 20000)
	if l2.Checkpoint.Seq != 4 {
		t.Fatalf("checkpoint after reassignment = %d, want 4", l2.Checkpoint.Seq)
	}
}

func TestLocalRestartAllowedIsOfflineAndIncarnationBound(t *testing.T) {
	self := boot("n1", "b1")
	l := &Lease{Workload: wl, Holder: self}
	if !LocalRestartAllowed(l, self) {
		t.Fatal("holder incarnation must be allowed to restart locally offline")
	}
	if LocalRestartAllowed(l, boot("n1", "b2")) {
		t.Fatal("a rebooted incarnation must NOT inherit local restart rights")
	}
	if LocalRestartAllowed(l, boot("n2", "b1")) {
		t.Fatal("another node must NOT restart the workload locally")
	}
	if LocalRestartAllowed(nil, self) {
		t.Fatal("no lease, no local restart")
	}
}

func TestClassifyCoversTheVocabulary(t *testing.T) {
	ready := &servicespec.Observed{Phase: servicespec.PhaseReady}
	failed := &servicespec.Observed{Phase: servicespec.PhaseFailed}
	stopped := &servicespec.Observed{
		Phase:    servicespec.PhaseStopped,
		LastExit: &servicespec.ExitRecord{Class: servicespec.ExitOperatorStop},
	}
	fresh := Evidence{NowMS: 10000, LastHeartbeatMS: 9000, HeartbeatTimeoutMS: 2000,
		HeartbeatBootID: "b1", KnownBootID: "b1"}

	cases := []struct {
		name string
		ev   Evidence
		want Condition
	}{
		{"healthy", func() Evidence { e := fresh; e.ReadBack = ready; return e }(), CondHealthy},
		{"process-crashed", func() Evidence { e := fresh; e.ReadBack = failed; return e }(), CondProcessCrashed},
		{"host-rebooted", func() Evidence { e := fresh; e.HeartbeatBootID = "b2"; e.ReadBack = ready; return e }(), CondHostRebooted},
		{"network-partitioned", Evidence{NowMS: 10000, LastHeartbeatMS: 1000, HeartbeatTimeoutMS: 2000}, CondNetworkPartitioned},
		{"intentionally-stopped-by-desire", func() Evidence { e := fresh; e.DesiredStopped = true; return e }(), CondIntentionallyStopped},
		{"intentionally-stopped-by-exit", func() Evidence { e := fresh; e.ReadBack = stopped; return e }(), CondIntentionallyStopped},
		{"dependency-blocked", func() Evidence { e := fresh; e.DependencyBlocked = true; e.ReadBack = ready; return e }(), CondDependencyBlocked},
		{"unknown", func() Evidence { e := fresh; e.ReadBack = nil; return e }(), CondUnknown},
	}
	seen := map[Condition]bool{}
	for _, c := range cases {
		if got := Classify(c.ev); got != c.want {
			t.Errorf("%s: Classify = %q, want %q", c.name, got, c.want)
		}
		seen[c.want] = true
	}
	for _, c := range AllConditions {
		if !seen[c] {
			t.Errorf("vocabulary condition %q has no classification case", c)
		}
	}
}

func TestBuildPlanRefusesReassignWhileLeaseValid(t *testing.T) {
	tb := NewTable(10000)
	inc := boot("n1", "b1")
	tb.RecordIncarnation(inc)
	mustAcquire(t, tb, wl, inc, 0)

	// Holder silent, lease still valid: the ONLY safe plan is to wait.
	part := Evidence{NowMS: 5000, LastHeartbeatMS: 1000, HeartbeatTimeoutMS: 2000}
	p := BuildPlan(tb, wl, part)
	if p.Condition != CondNetworkPartitioned || p.Action != ActionWaitLease || p.Reason != "lease-valid" {
		t.Fatalf("mid-lease partition plan = %+v, want wait-lease/lease-valid", p)
	}
	if !p.DryRun {
		t.Fatal("every plan this leaf emits must be dry-run")
	}

	// Same silence after expiry: reassignment becomes permitted.
	part.NowMS = 12000
	p = BuildPlan(tb, wl, part)
	if p.Action != ActionReassign || p.Reason != "lease-expired" {
		t.Fatalf("post-expiry partition plan = %+v, want reassign/lease-expired", p)
	}

	// The plan is a JSON document with the versioned schema.
	raw, err := p.JSON()
	if err != nil {
		t.Fatalf("Plan.JSON = %v", err)
	}
	var back Plan
	if err := json.Unmarshal(raw, &back); err != nil || back != p {
		t.Fatalf("plan round-trip = %+v (err %v), want %+v", back, err, p)
	}
	if !strings.Contains(string(raw), ReconcileSchemaV1) {
		t.Fatalf("plan JSON lacks schema %q: %s", ReconcileSchemaV1, raw)
	}
}

func TestBuildPlanLocalRecoveryForCrash(t *testing.T) {
	tb := NewTable(10000)
	inc := boot("n1", "b1")
	tb.RecordIncarnation(inc)
	mustAcquire(t, tb, wl, inc, 0)

	ev := Evidence{NowMS: 2000, LastHeartbeatMS: 2000, HeartbeatTimeoutMS: 2000,
		HeartbeatBootID: "b1", KnownBootID: "b1",
		ReadBack: &servicespec.Observed{Phase: servicespec.PhaseFailed}}
	p := BuildPlan(tb, wl, ev)
	if p.Condition != CondProcessCrashed || p.Action != ActionRestartLocal {
		t.Fatalf("crash plan = %+v, want process-crashed/restart-local", p)
	}
}
