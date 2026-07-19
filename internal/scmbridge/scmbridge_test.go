package scmbridge

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

func testSpec(t *testing.T) *servicespec.Spec {
	t.Helper()
	s := &servicespec.Spec{
		Schema:   servicespec.SchemaV1,
		Identity: servicespec.Identity{Node: "lab-1", Service: "fak-guard", Workload: "FakGuardControl"},
		Kind:     servicespec.KindService,
		Desired:  servicespec.DesiredRunning,
		Command:  []string{`C:\opt\fak\fak.exe`, "service", "windows-run"},
	}
	s.Normalize()
	return s
}

func TestProjectMachineRoleDerivesSCMFormFromOneContract(t *testing.T) {
	p, err := Project(testSpec(t), ProjectInput{Role: RoleMachine, BinarySHA256: "ABCD"})
	if err != nil {
		t.Fatal(err)
	}
	if p.Manager != ManagerSCM || p.Principal != MachinePrincipal || p.LogonType != "" {
		t.Fatalf("machine projection = %+v", p)
	}
	if !p.StartOnBoot || p.StartOnLogon || p.DesiredStopped {
		t.Fatalf("machine triggers = %+v", p)
	}
	// The SCM recovery ladder is the first three delays of the ONE portable
	// restart contract (defaults: 1s, 2s, 4s), reset from stable-run reset.
	want := []RecoveryStep{{DelayMS: 1000}, {DelayMS: 2000}, {DelayMS: 4000}}
	if len(p.Recovery) != 3 || p.Recovery[0] != want[0] || p.Recovery[1] != want[1] || p.Recovery[2] != want[2] {
		t.Fatalf("recovery ladder = %v", p.Recovery)
	}
	if p.RecoveryResetSec != servicespec.DefaultStableRunResetMS/1000 {
		t.Fatalf("reset = %d", p.RecoveryResetSec)
	}
	if p.BinarySHA256 != "abcd" {
		t.Fatalf("provenance not canonicalized: %q", p.BinarySHA256)
	}
}

func TestProjectPlacementRefusals(t *testing.T) {
	job := testSpec(t)
	job.Kind = servicespec.KindJob
	if _, err := Project(job, ProjectInput{Role: RoleMachine}); !errors.Is(err, ErrJobOnSCM) {
		t.Fatalf("job on SCM: err = %v", err)
	}
	for _, role := range []Role{RoleWatchdog, RoleBroker} {
		if _, err := Project(testSpec(t), ProjectInput{Role: role}); !errors.Is(err, ErrTaskNeedsPrincipal) {
			t.Fatalf("%s without principal: err = %v", role, err)
		}
	}
	if _, err := Project(testSpec(t), ProjectInput{Role: Role("cron")}); !errors.Is(err, ErrUnknownRole) {
		t.Fatalf("unknown role: err = %v", err)
	}
}

func TestProjectTaskRolesReserveScheduledTasksForTheDesktopSeam(t *testing.T) {
	w, err := Project(testSpec(t), ProjectInput{Role: RoleWatchdog, TaskPrincipal: "labops"})
	if err != nil {
		t.Fatal(err)
	}
	if w.Manager != ManagerTask || w.LogonType != LogonS4U || !w.StartOnBoot || w.StartOnLogon {
		t.Fatalf("watchdog projection = %+v", w)
	}
	b, err := Project(testSpec(t), ProjectInput{Role: RoleBroker, TaskPrincipal: "labops"})
	if err != nil {
		t.Fatal(err)
	}
	if b.Manager != ManagerTask || b.LogonType != LogonInteractive || b.StartOnBoot || !b.StartOnLogon {
		t.Fatalf("broker projection = %+v", b)
	}
}

func TestProjectDesiredStoppedProjectsStartDisabled(t *testing.T) {
	s := testSpec(t)
	s.Desired = servicespec.DesiredStopped
	p, err := Project(s, ProjectInput{Role: RoleMachine})
	if err != nil {
		t.Fatal(err)
	}
	if !p.DesiredStopped || p.StartOnBoot {
		t.Fatalf("desired-stopped projection = %+v", p)
	}
}

func observedFrom(p *Projection) Observed {
	return Observed{
		Present:          true,
		Manager:          p.Manager,
		UnitName:         p.UnitName,
		Principal:        p.Principal,
		LogonType:        p.LogonType,
		Command:          append([]string(nil), p.Command...),
		BinarySHA256:     p.BinarySHA256,
		StartOnBoot:      p.StartOnBoot,
		StartOnLogon:     p.StartOnLogon,
		Recovery:         append([]RecoveryStep(nil), p.Recovery...),
		RecoveryResetSec: p.RecoveryResetSec,
		StartDisabled:    p.DesiredStopped,
		Status:           "running",
		PID:              4242,
	}
}

func TestReconcileInSyncAndPrincipalSpellingFold(t *testing.T) {
	p, err := Project(testSpec(t), ProjectInput{Role: RoleMachine, BinarySHA256: "abcd"})
	if err != nil {
		t.Fatal(err)
	}
	got := observedFrom(p)
	if r := Reconcile(p, got); !r.InSync || len(r.Divergences) != 0 {
		t.Fatalf("round trip diverged: %+v", r)
	}
	// SCM reports the short spelling; the fold must not flag it.
	got.Principal = "LocalService"
	if r := Reconcile(p, got); !r.InSync {
		t.Fatalf("principal spelling diverged: %+v", r)
	}
}

func TestReconcileFlagsEveryContractAxis(t *testing.T) {
	p, err := Project(testSpec(t), ProjectInput{Role: RoleMachine, BinarySHA256: "abcd"})
	if err != nil {
		t.Fatal(err)
	}
	mutate := map[string]func(*Observed){
		AxisManager:     func(o *Observed) { o.Manager = ManagerTask },
		AxisUnit:        func(o *Observed) { o.UnitName = "OtherUnit" },
		AxisPrincipal:   func(o *Observed) { o.Principal = "LocalSystem" },
		AxisAction:      func(o *Observed) { o.Command = []string{`C:\evil.exe`} },
		AxisProvenance:  func(o *Observed) { o.BinarySHA256 = "ffff" },
		AxisTrigger:     func(o *Observed) { o.StartOnBoot = false },
		AxisRecovery:    func(o *Observed) { o.Recovery[0].DelayMS = 99999 },
		AxisDesiredStop: func(o *Observed) { o.StartDisabled = true },
	}
	for axis, fn := range mutate {
		got := observedFrom(p)
		fn(&got)
		r := Reconcile(p, got)
		if r.InSync || len(r.Divergences) != 1 || r.Divergences[0].Axis != axis {
			t.Errorf("axis %s: report = %+v", axis, r)
		}
	}
	// Absent dominates everything.
	r := Reconcile(p, Observed{})
	if r.InSync || len(r.Divergences) != 1 || r.Divergences[0].Axis != AxisAbsent {
		t.Fatalf("absent report = %+v", r)
	}
}

func TestReconcileSkipsNotReadFieldsInsteadOfGuessing(t *testing.T) {
	p, err := Project(testSpec(t), ProjectInput{Role: RoleMachine, BinarySHA256: "abcd"})
	if err != nil {
		t.Fatal(err)
	}
	got := observedFrom(p)
	got.BinarySHA256 = "" // digest not read
	got.Recovery = nil    // recovery not read
	got.RecoveryResetSec = 0
	if r := Reconcile(p, got); !r.InSync {
		t.Fatalf("not-read fields diverged: %+v", r)
	}
}

func TestPhaseMapsAreAuthoritativeReadBack(t *testing.T) {
	cases := []struct {
		state string
		pid   int
		want  servicespec.Phase
	}{
		{"running", 100, servicespec.PhaseReady},
		{"running", 0, servicespec.PhaseStarting},
		{"start-pending", 0, servicespec.PhaseStarting},
		{"paused", 0, servicespec.PhaseDegraded},
		{"stopped", 0, servicespec.PhaseStopped},
		{"gone", 0, servicespec.PhaseUnknown},
	}
	for _, c := range cases {
		if got := PhaseFromSCMState(c.state, c.pid); got != c.want {
			t.Errorf("scm %s/%d = %s, want %s", c.state, c.pid, got, c.want)
		}
	}
	if PhaseFromTaskState("Running") != servicespec.PhaseReady ||
		PhaseFromTaskState("Ready") != servicespec.PhaseStopped ||
		PhaseFromTaskState("Queued") != servicespec.PhaseStarting ||
		PhaseFromTaskState("Disabled") != servicespec.PhaseStopped ||
		PhaseFromTaskState("??") != servicespec.PhaseUnknown {
		t.Fatal("task state map broke")
	}
}

func TestExecutableFromCommandLineHandlesUnquotedSpaces(t *testing.T) {
	exists := func(p string) bool { return p == `C:\Program Files\fak\fak.exe` }
	cases := []struct{ in, want string }{
		{`"C:\Program Files\fak\fak.exe" service windows-run`, `C:\Program Files\fak\fak.exe`},
		{`C:\opt\fak.exe -k netsvcs`, `C:\opt\fak.exe`},
		{`C:\Program Files\fak\fak.exe service windows-run`, `C:\Program Files\fak\fak.exe`},
		{``, ``},
	}
	for _, c := range cases {
		if got := ExecutableFromCommandLine(c.in, exists); got != c.want {
			t.Errorf("%q => %q, want %q", c.in, got, c.want)
		}
	}
	// No resolver: first token wins.
	if got := ExecutableFromCommandLine(`C:\a b`, nil); got != `C:\a` {
		t.Errorf("nil resolver => %q", got)
	}
	// Nothing exists: fall back to the first token, not the whole line.
	if got := ExecutableFromCommandLine(`C:\x y z`, func(string) bool { return false }); got != `C:\x` {
		t.Errorf("unresolved => %q", got)
	}
	if !strings.HasPrefix(cases[0].want, `C:\Program`) {
		t.Fatal("sanity")
	}
}
