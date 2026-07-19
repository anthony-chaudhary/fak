package servicelease

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// step advances the sim one tick and fails the test the instant the safety
// invariant breaks: more than one simultaneously valid running owner.
func step(t *testing.T, s *Sim) {
	t.Helper()
	if n := s.Step(); n > 1 {
		t.Fatalf("at %dms: %d valid running owners (%v) — overlapping ownership",
			s.NowMS, n, s.ValidRunningOwners())
	}
}

func newTwoNodeSim() *Sim {
	// TTL 10s, 1s steps, 2.5s heartbeat window — all logical.
	return NewSim(wl, 10000, 1000, 2500, "alpha", "beta")
}

// TestSimPartitionNeverOverlapsOwnership is the core acceptance scenario: the
// owner is partitioned away, keeps running blind, the controller WAITS out the
// still-valid lease (no premature reassignment), takes over only after expiry,
// and the healed old owner is fenced — at no step do two valid owners overlap.
func TestSimPartitionNeverOverlapsOwnership(t *testing.T) {
	s := newTwoNodeSim()
	if err := s.Grant("alpha"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		step(t, s)
	}
	if got := s.ValidRunningOwners(); !reflect.DeepEqual(got, []string{"alpha"}) {
		t.Fatalf("pre-fault owner = %v, want [alpha]", got)
	}

	s.Partition("alpha")
	sawWait := false
	// alpha renewed at 3000ms, so its lease is valid until 13000ms: the
	// controller must refuse reassignment for every step before that.
	for s.NowMS < 12000 {
		step(t, s)
		if s.LastReconcile.Condition == CondNetworkPartitioned && s.LastReconcile.Action == ActionWaitLease {
			sawWait = true
		}
		if owner := s.Table.ValidOwner(wl, s.NowMS); owner.Node == "beta" {
			t.Fatalf("at %dms: reassigned to beta while alpha's lease was still valid", s.NowMS)
		}
	}
	if !sawWait {
		t.Fatal("never witnessed the wait-lease plan during the valid-lease partition window")
	}

	// Lease expiry passes: the controller may now reassign to beta.
	step(t, s) // 13000ms
	if got := s.ValidRunningOwners(); !reflect.DeepEqual(got, []string{"beta"}) {
		t.Fatalf("post-expiry owner = %v, want [beta]", got)
	}
	// Alpha is STILL RUNNING blind on the far side of the partition — the
	// split is real — but it is running, not valid, and its stale claim to
	// have finished the work is refused.
	if got := s.RunningNodes(); !reflect.DeepEqual(got, []string{"alpha", "beta"}) {
		t.Fatalf("running nodes = %v, want [alpha beta] (split process)", got)
	}
	alpha := s.node("alpha")
	err := s.Table.PublishCompletion(wl, alpha.Incarnation(), alpha.LeaseCopy.Token, Checkpoint{Seq: 99})
	if !errors.Is(err, ErrNotHolder) {
		t.Fatalf("stale owner publish = %v, want ErrNotHolder", err)
	}

	// The partition heals: alpha's next renewal is refused and it stops.
	s.Heal("alpha")
	step(t, s)
	if got := s.RunningNodes(); !reflect.DeepEqual(got, []string{"beta"}) {
		t.Fatalf("post-heal running nodes = %v, want [beta] (alpha fenced itself)", got)
	}
	fenced := false
	for _, r := range s.Refusals {
		if r.Node == "alpha" && r.Op == "renew" {
			fenced = true
		}
	}
	if !fenced {
		t.Fatal("healed stale owner's renewal refusal was never witnessed")
	}

	// Converged: steady healthy state, one owner, high-water mark holds.
	for i := 0; i < 3; i++ {
		step(t, s)
	}
	if s.LastReconcile.Condition != CondHealthy || s.LastReconcile.Action != ActionNone {
		t.Fatalf("converged plan = %+v, want healthy/none", s.LastReconcile)
	}
	if s.MaxOwners > 1 {
		t.Fatalf("MaxOwners = %d, want <= 1", s.MaxOwners)
	}
}

// TestSimCrashRecoversLocallyWithoutReassignment: a process crash is repaired
// by the owning incarnation itself — offline-capable — and ownership never
// moves (the lease sequence stays put).
func TestSimCrashRecoversLocallyWithoutReassignment(t *testing.T) {
	s := newTwoNodeSim()
	if err := s.Grant("alpha"); err != nil {
		t.Fatal(err)
	}
	seq := s.Table.Leases[wl].Token.LeaseSeq
	for i := 0; i < 2; i++ {
		step(t, s)
	}

	s.Crash("alpha")
	step(t, s)
	if got := s.ValidRunningOwners(); !reflect.DeepEqual(got, []string{"alpha"}) {
		t.Fatalf("post-crash owner = %v, want [alpha] (local restart)", got)
	}
	if s.LastReconcile.Condition != CondProcessCrashed || s.LastReconcile.Action != ActionRestartLocal {
		t.Fatalf("crash-step plan = %+v, want process-crashed/restart-local", s.LastReconcile)
	}
	step(t, s)
	if s.LastReconcile.Condition != CondHealthy {
		t.Fatalf("converged plan condition = %q, want healthy", s.LastReconcile.Condition)
	}
	if got := s.Table.Leases[wl].Token.LeaseSeq; got != seq {
		t.Fatalf("lease seq moved %d -> %d: crash recovery must not change ownership", seq, got)
	}
	if s.MaxOwners > 1 {
		t.Fatalf("MaxOwners = %d, want <= 1", s.MaxOwners)
	}
}

// TestSimRebootFencesOldIncarnation: a host reboot comes back as a NEW
// incarnation; the old one is superseded the moment its first heartbeat
// lands, ownership is re-granted under a fresh token, and the dead
// incarnation can never publish completion.
func TestSimRebootFencesOldIncarnation(t *testing.T) {
	s := newTwoNodeSim()
	if err := s.Grant("alpha"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		step(t, s)
	}
	oldInc := s.node("alpha").Incarnation()
	oldTok := s.node("alpha").LeaseCopy.Token

	s.Reboot("alpha")
	step(t, s)
	if s.LastReconcile.Condition != CondHostRebooted || s.LastReconcile.Action != ActionReassign {
		t.Fatalf("reboot-step plan = %+v, want host-rebooted/reassign", s.LastReconcile)
	}
	l := s.Table.Leases[wl]
	if l.Holder.BootID != "alpha-boot-2" {
		t.Fatalf("post-reboot holder = %+v, want alpha-boot-2", l.Holder)
	}
	if l.Token.LeaseSeq <= oldTok.LeaseSeq {
		t.Fatalf("post-reboot lease seq %d must exceed pre-reboot %d", l.Token.LeaseSeq, oldTok.LeaseSeq)
	}
	if err := s.Table.PublishCompletion(wl, oldInc, oldTok, Checkpoint{Seq: 7}); !errors.Is(err, ErrStaleIncarnation) {
		t.Fatalf("dead-incarnation publish = %v, want ErrStaleIncarnation", err)
	}
	for i := 0; i < 2; i++ {
		step(t, s)
	}
	if got := s.ValidRunningOwners(); !reflect.DeepEqual(got, []string{"alpha"}) {
		t.Fatalf("converged owner = %v, want [alpha] under the new incarnation", got)
	}
	if s.MaxOwners > 1 {
		t.Fatalf("MaxOwners = %d, want <= 1", s.MaxOwners)
	}
}

// TestSimDelayedHeartbeatDoesNotReassign: silence shorter than the lease is
// classified as a partition but changes NOTHING — no reassignment, same
// token, and the owner resumes cleanly when heartbeats return.
func TestSimDelayedHeartbeatDoesNotReassign(t *testing.T) {
	s := newTwoNodeSim()
	if err := s.Grant("alpha"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		step(t, s)
	}
	seq := s.Table.Leases[wl].Token.LeaseSeq

	s.Partition("alpha") // delay heartbeats for 4s, well inside the lease
	sawPartition := false
	for i := 0; i < 4; i++ {
		step(t, s)
		if s.LastReconcile.Condition == CondNetworkPartitioned {
			sawPartition = true
			if s.LastReconcile.Action != ActionWaitLease {
				t.Fatalf("delayed-heartbeat plan = %+v, want wait-lease", s.LastReconcile)
			}
		}
	}
	if !sawPartition {
		t.Fatal("heartbeat delay was never classified as network-partitioned")
	}
	s.Heal("alpha")
	for i := 0; i < 2; i++ {
		step(t, s)
	}
	if s.LastReconcile.Condition != CondHealthy {
		t.Fatalf("converged plan condition = %q, want healthy", s.LastReconcile.Condition)
	}
	if got := s.Table.Leases[wl].Token.LeaseSeq; got != seq {
		t.Fatalf("lease seq moved %d -> %d: a heartbeat delay must not reassign", seq, got)
	}
	if got := s.ValidRunningOwners(); !reflect.DeepEqual(got, []string{"alpha"}) {
		t.Fatalf("converged owner = %v, want [alpha]", got)
	}
	if s.MaxOwners > 1 {
		t.Fatalf("MaxOwners = %d, want <= 1", s.MaxOwners)
	}
}

// TestSimDeterministic replays the full partition script twice and demands
// byte-identical traces: same owner sequence, same refusals, same final table.
func TestSimDeterministic(t *testing.T) {
	run := func() (trace []string, table []byte) {
		s := newTwoNodeSim()
		if err := s.Grant("alpha"); err != nil {
			t.Fatal(err)
		}
		script := func(i int) {
			switch i {
			case 3:
				s.Partition("alpha")
			case 8:
				s.Crash("beta")
			case 15:
				s.Heal("alpha")
			case 18:
				s.Reboot("beta")
			}
		}
		for i := 0; i < 25; i++ {
			script(i)
			n := s.Step()
			trace = append(trace, fmt.Sprintf("%d now=%d owners=%v running=%v plan=%s/%s",
				n, s.NowMS, s.ValidRunningOwners(), s.RunningNodes(), s.LastReconcile.Condition, s.LastReconcile.Action))
		}
		for _, r := range s.Refusals {
			trace = append(trace, fmt.Sprintf("refusal %d %s %s %v", r.AtMS, r.Node, r.Op, r.Err))
		}
		raw, err := json.Marshal(s.Table)
		if err != nil {
			t.Fatal(err)
		}
		return trace, raw
	}
	t1, tab1 := run()
	t2, tab2 := run()
	if !reflect.DeepEqual(t1, t2) {
		t.Fatalf("trace diverged between identical runs:\n%v\n%v", t1, t2)
	}
	if string(tab1) != string(tab2) {
		t.Fatalf("final table diverged:\n%s\n%s", tab1, tab2)
	}
}
