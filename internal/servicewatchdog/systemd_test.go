package servicewatchdog

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestParseSystemdShowStableProperties(t *testing.T) {
	raw := "LoadState=loaded\nActiveState=active\nSubState=running\nResult=watchdog\nMainPID=42\nInvocationID=abc\nNRestarts=3\nWatchdogTimestampUSec=9001\nBootID=boot\n"
	got, err := ParseSystemdShow([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if got.MainPID != 42 || got.NRestarts != 3 || got.Result != "watchdog" || got.InvocationID != "abc" || got.BootID != "boot" {
		t.Fatalf("unexpected read-back: %+v", got)
	}
	if args := strings.Join(SystemdShowArgs("fak.service"), " "); strings.Contains(args, "status") || !strings.Contains(args, "--property=WatchdogTimestampUSec") {
		t.Fatalf("unsafe/incomplete show args: %s", args)
	}
}

func TestParseSystemdShowFailsClosedOnPropertyDrift(t *testing.T) {
	if _, err := ParseSystemdShow([]byte("LoadState=loaded\n")); err == nil || !strings.Contains(err.Error(), "missing property") {
		t.Fatalf("err = %v", err)
	}
}

func TestNotifierHeartbeatsOnlyOnLoopProgress(t *testing.T) {
	now := time.Unix(100, 0)
	var sent []string
	n := &Notifier{socket: "fixture", interval: 5 * time.Second, now: func() time.Time { return now }, send: func(_, state string) error { sent = append(sent, state); return nil }}
	if err := n.Ready(); err != nil {
		t.Fatal(err)
	}
	now = now.Add(4 * time.Second)
	_ = n.Progress()
	now = now.Add(2 * time.Second)
	_ = n.Progress()
	now = now.Add(6 * time.Second)
	_ = n.Progress()
	_ = n.Stopping()
	want := []string{"READY=1", "WATCHDOG=1", "WATCHDOG=1", "STOPPING=1"}
	if strings.Join(sent, ",") != strings.Join(want, ",") {
		t.Fatalf("sent %v, want %v", sent, want)
	}
}

type fixtureCommand struct {
	calls  []string
	output []byte
	errAt  string
}

func (f *fixtureCommand) Run(args ...string) error {
	call := strings.Join(args, " ")
	f.calls = append(f.calls, call)
	if call == f.errAt {
		return errors.New("denied")
	}
	return nil
}
func (f *fixtureCommand) Output(args ...string) ([]byte, error) {
	call := strings.Join(args, " ")
	f.calls = append(f.calls, call)
	if call == f.errAt {
		return nil, errors.New("missing")
	}
	return f.output, nil
}
func TestManagerLifecyclePreservesDesiredStop(t *testing.T) {
	f := &fixtureCommand{}
	m := Manager{Command: f, Unit: "fak-x.service"}
	if err := m.Install(); err != nil {
		t.Fatal(err)
	}
	if err := m.Stop(); err != nil {
		t.Fatal(err)
	}
	if err := m.Start(); err != nil {
		t.Fatal(err)
	}
	if err := m.Restart(); err != nil {
		t.Fatal(err)
	}
	if err := m.Remove(); err != nil {
		t.Fatal(err)
	}
	got := strings.Join(f.calls, "|")
	want := "daemon-reload|enable --now fak-x.service|disable --now fak-x.service|enable --now fak-x.service|restart fak-x.service|disable --now fak-x.service|daemon-reload"
	if got != want {
		t.Fatalf("calls %q, want %q", got, want)
	}
}
func TestManagerStopsOnDaemonReloadFailure(t *testing.T) {
	f := &fixtureCommand{errAt: "daemon-reload"}
	m := Manager{Command: f, Unit: "fak-x.service"}
	if err := m.Install(); err == nil {
		t.Fatal("expected failure")
	}
	if len(f.calls) != 1 {
		t.Fatalf("calls=%v", f.calls)
	}
}
