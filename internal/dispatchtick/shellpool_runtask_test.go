package dispatchtick

// RunTask is the rack's reuse-spine entry point (#3405): the call a spawn site
// makes instead of creating a process per task. These tests pin its contract —
// one shell serves many tasks, a shell that fails a task is retired instead of
// reused, and the churn dividend is countable — so the cmd/fak host-probe
// wiring that rides on it has a specified seam underneath.

import (
	"errors"
	"testing"
	"time"
)

type fakeTaskShell struct {
	fakeWarmShell
	tasks   []string
	failNth int // 1-based task index that breaks the shell (0 = never)
}

func (s *fakeTaskShell) RunTask(task string) ([]byte, error) {
	s.tasks = append(s.tasks, task)
	if s.failNth == len(s.tasks) {
		s.healthy = false
		return nil, errors.New("task broke the shell")
	}
	return []byte("out:" + task), nil
}

type fakeTaskSpawner struct {
	calls   int
	shells  []*fakeTaskShell
	failNth int
}

func (f *fakeTaskSpawner) spawn(string) (WarmShell, error) {
	f.calls++
	s := &fakeTaskShell{fakeWarmShell: fakeWarmShell{id: f.calls, healthy: true}, failNth: f.failNth}
	f.shells = append(f.shells, s)
	return s, nil
}

func TestShellRackRunTaskReusesOneShellAcrossTasks(t *testing.T) {
	sp := &fakeTaskSpawner{}
	p, err := NewShellRack(1, time.Minute, sp.spawn)
	if err != nil {
		t.Fatalf("NewShellRack: %v", err)
	}

	for _, task := range []string{"probe-a", "probe-b", "probe-c"} {
		out, err := p.RunTask("host", task)
		if err != nil {
			t.Fatalf("RunTask(%s): %v", task, err)
		}
		if string(out) != "out:"+task {
			t.Fatalf("RunTask(%s) = %q, want the shell's own output", task, out)
		}
	}

	if sp.calls != 1 {
		t.Fatalf("spawner ran %d times for 3 tasks, want 1 (reuse, not respawn)", sp.calls)
	}
	if got := len(sp.shells[0].tasks); got != 3 {
		t.Fatalf("shell ran %d tasks, want 3", got)
	}
	st := p.Stats()
	if st.ColdSpawns != 1 || st.WarmReuses != 2 || st.TasksRun != 3 {
		t.Fatalf("stats = %+v, want ColdSpawns=1 WarmReuses=2 TasksRun=3", st)
	}
	if got := st.SpawnsAvoided(); got != 2 {
		t.Fatalf("SpawnsAvoided = %d, want 2 (3 tasks would have been 3 spawns)", got)
	}
}

func TestShellRackRunTaskRetiresAShellThatFailedItsTask(t *testing.T) {
	sp := &fakeTaskSpawner{failNth: 2}
	p, err := NewShellRack(2, time.Minute, sp.spawn)
	if err != nil {
		t.Fatalf("NewShellRack: %v", err)
	}

	if _, err := p.RunTask("host", "first"); err != nil {
		t.Fatalf("first RunTask: %v", err)
	}
	if _, err := p.RunTask("host", "second"); err == nil {
		t.Fatalf("a task that breaks the shell must surface its error")
	}
	if !sp.shells[0].closed {
		t.Fatalf("the broken shell was not closed")
	}
	if p.IdleCount("host") != 0 {
		t.Fatalf("the broken shell was retained for reuse")
	}

	if _, err := p.RunTask("host", "third"); err != nil {
		t.Fatalf("third RunTask: %v", err)
	}
	if sp.calls != 2 {
		t.Fatalf("spawner ran %d times, want 2 (retire + replace)", sp.calls)
	}
	if st := p.Stats(); st.UnhealthyRetired != 1 || st.TasksRun != 3 {
		t.Fatalf("stats = %+v, want UnhealthyRetired=1 TasksRun=3", st)
	}
}

func TestShellRackRunTaskRefusesAShellThatCannotRunTasks(t *testing.T) {
	sp := &fakeShellSpawner{} // plain WarmShell, no RunTask
	p, err := NewShellRack(1, time.Minute, sp.spawn)
	if err != nil {
		t.Fatalf("NewShellRack: %v", err)
	}
	if _, err := p.RunTask("host", "probe"); !errors.Is(err, ErrShellNotTaskCapable) {
		t.Fatalf("RunTask error = %v, want ErrShellNotTaskCapable so the caller can fall back", err)
	}
	if p.IdleCount("host") != 1 {
		t.Fatalf("a task-incapable shell should be returned, not leaked; idle=%d", p.IdleCount("host"))
	}
}

func TestShellRackRunTaskRefusesOnAClosedRack(t *testing.T) {
	sp := &fakeTaskSpawner{}
	p, err := NewShellRack(1, time.Minute, sp.spawn)
	if err != nil {
		t.Fatalf("NewShellRack: %v", err)
	}
	p.Close()
	if _, err := p.RunTask("host", "probe"); err == nil {
		t.Fatalf("RunTask on a closed rack must refuse")
	}
	if sp.calls != 0 {
		t.Fatalf("a closed rack must not spawn; spawner ran %d times", sp.calls)
	}
}
