package toolprocgate

import (
	"context"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// recordingReaper is the deterministic OS lever for tests: records every pid
// it is asked to terminate and answers a scripted verdict.
type recordingReaper struct {
	pids   []int
	ok     bool
	detail string
}

func (r *recordingReaper) reap(pid int) (bool, string) {
	r.pids = append(r.pids, pid)
	return r.ok, r.detail
}

// TestBindPIDDeadlineKillReapsTree is the seam-6 pipeline witness: a bound
// call blows its deadline → the same tick that cancels and revokes also
// terminates the bound OS process tree, exactly once.
func TestBindPIDDeadlineKillReapsTree(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	sup := NewSupervisor(toolproc.Config{})
	rec := &recordingReaper{ok: true, detail: "tree terminated"}
	sup.SetReaper(rec.reap)

	ctx, cancel := context.WithCancel(context.Background())
	if err := sup.Spawn("b-dl", "bg_build", "s1", 10_000, 0, 1_000, cancel); err != nil {
		t.Fatal(err)
	}
	if err := sup.BindPID("b-dl", 4242); err != nil {
		t.Fatal(err)
	}

	rep, err := sup.Tick(12_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 1 {
		t.Fatalf("want one action, got %+v", rep.Actions)
	}
	act := rep.Actions[0]
	if !act.Cancelled || !act.Reaped || act.ReapDetail != "tree terminated" {
		t.Fatalf("want cancelled+reaped kill, got %+v", act)
	}
	if ctx.Err() == nil {
		t.Fatal("context must be cancelled alongside the OS reap")
	}
	if len(rec.pids) != 1 || rec.pids[0] != 4242 {
		t.Fatalf("reaper must fire once on the bound pid, got %v", rec.pids)
	}

	// The binding is retired with the kill: a later tick re-invokes nothing.
	if _, err := sup.Tick(20_000); err != nil {
		t.Fatal(err)
	}
	if len(rec.pids) != 1 {
		t.Fatalf("killed call must not be reaped again, got %v", rec.pids)
	}
}

// TestBindPIDOrphanReap: the session-end orphan path carries the same teeth.
func TestBindPIDOrphanReap(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	sup := NewSupervisor(toolproc.Config{})
	rec := &recordingReaper{ok: true}
	sup.SetReaper(rec.reap)
	if err := sup.Spawn("b-orph", "watch_dir", "s2", 0, 0, 1_000, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.BindPID("b-orph", 777); err != nil {
		t.Fatal(err)
	}
	if err := sup.SessionEnd("s2", 5_000); err != nil {
		t.Fatal(err)
	}
	rep, err := sup.Tick(6_000)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Actions) != 1 || rep.Actions[0].Advice != toolproc.AdviceReap || !rep.Actions[0].Reaped {
		t.Fatalf("want one reaped orphan action, got %+v", rep.Actions)
	}
	if len(rec.pids) != 1 || rec.pids[0] != 777 {
		t.Fatalf("reaper must fire on the orphan's pid, got %v", rec.pids)
	}
}

// TestReaperFailureIsRecordedNotReaped: a failing OS lever reports its detail
// but never claims the reap; cancel and revocation still land.
func TestReaperFailureIsRecordedNotReaped(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	sup := NewSupervisor(toolproc.Config{})
	rec := &recordingReaper{ok: false, detail: "access denied"}
	sup.SetReaper(rec.reap)
	if err := sup.Spawn("b-fail", "bg_job", "s1", 1_000, 0, 500, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.BindPID("b-fail", 555); err != nil {
		t.Fatal(err)
	}
	rep, err := sup.Tick(2_000)
	if err != nil {
		t.Fatal(err)
	}
	act := rep.Actions[0]
	if act.Reaped || act.ReapDetail != "access denied" {
		t.Fatalf("failed reap must record detail without claiming success, got %+v", act)
	}
	if _, ok := KilledReason("b-fail"); !ok {
		t.Fatal("revocation must land even when the OS reap fails")
	}
}

// TestNoBindingNoReaperNoTeeth: unbound calls and reaper-less supervisors keep
// the pre-seam-6 behavior byte-for-byte — cancel + revoke, no OS act.
func TestNoBindingNoReaperNoTeeth(t *testing.T) {
	Reset()
	t.Cleanup(Reset)

	// Reaper set, no binding: nothing to aim at.
	sup := NewSupervisor(toolproc.Config{})
	rec := &recordingReaper{ok: true}
	sup.SetReaper(rec.reap)
	if err := sup.Spawn("n-1", "x", "s1", 1_000, 0, 500, nil); err != nil {
		t.Fatal(err)
	}
	rep, err := sup.Tick(2_000)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Actions[0].Reaped || len(rec.pids) != 0 {
		t.Fatalf("unbound call must not reap, got %+v pids=%v", rep.Actions, rec.pids)
	}

	// Binding set, no reaper: advice-only, no panic.
	sup2 := NewSupervisor(toolproc.Config{})
	if err := sup2.Spawn("n-2", "x", "s1", 1_000, 0, 500, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup2.BindPID("n-2", 888); err != nil {
		t.Fatal(err)
	}
	rep, err = sup2.Tick(2_000)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Actions[0].Reaped {
		t.Fatalf("reaper-less supervisor must stay advice-only, got %+v", rep.Actions)
	}
}

// TestBindPIDRefusals: the fail-closed binding contract — no invalid pid, no
// unknown call, never the supervisor's own process.
func TestBindPIDRefusals(t *testing.T) {
	Reset()
	t.Cleanup(Reset)
	sup := NewSupervisor(toolproc.Config{})
	if err := sup.Spawn("r-1", "x", "s1", 0, 0, 1_000, nil); err != nil {
		t.Fatal(err)
	}
	if err := sup.BindPID("r-1", 0); err == nil {
		t.Error("pid 0 must refuse")
	}
	if err := sup.BindPID("r-1", -7); err == nil {
		t.Error("negative pid must refuse")
	}
	if err := sup.BindPID("ghost", 1234); err == nil {
		t.Error("bind for unknown call must refuse")
	}
	if err := sup.BindPID("r-1", os.Getpid()); err == nil {
		t.Error("self pid must refuse")
	}

	// Exit retires the binding: the pid is not reaped after a clean exit.
	rec := &recordingReaper{ok: true}
	sup.SetReaper(rec.reap)
	if err := sup.BindPID("r-1", 999); err != nil {
		t.Fatal(err)
	}
	if err := sup.Exit("r-1", 2_000, "ok"); err != nil {
		t.Fatal(err)
	}
	if _, err := sup.Tick(3_000); err != nil {
		t.Fatal(err)
	}
	if len(rec.pids) != 0 {
		t.Fatalf("exited call must never be reaped, got %v", rec.pids)
	}
}

// TestNewOSSupervisorWiresProcguard: the production constructor arms the OS
// lever (never fired here — no binding is made).
func TestNewOSSupervisorWiresProcguard(t *testing.T) {
	sup := NewOSSupervisor(toolproc.Config{})
	if sup.reaper == nil {
		t.Fatal("NewOSSupervisor must register the procguard reaper")
	}
}
