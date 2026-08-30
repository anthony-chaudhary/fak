//go:build windows

package main

import (
	"errors"
	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/shellprov"
)

func TestLaunchHostSessionPlatformQueuesBeforeInteractiveBroker(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_HOST_RELAUNCH_DIR", dir)
	old := runInteractiveBrokerTask
	called := false
	runInteractiveBrokerTask = func(task string) error {
		called = true
		if task != "FakHostRelaunchBroker" {
			t.Fatalf("task=%q", task)
		}
		return os.ErrNotExist
	}
	t.Cleanup(func() { runInteractiveBrokerTask = old })
	req := hostresurrect.Request{Schema: hostresurrect.Schema, EventID: "evt", Session: "g1", CWD: `C:\work`, Command: []string{"claude", "--resume", "g1"}, ResumeHandle: "g1"}
	if _, err := launchHostSessionPlatform(req); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("broker not signaled")
	}
	pending, err := hostresurrect.Pending(dir)
	if err != nil || len(pending) != 1 {
		t.Fatalf("pending=%v err=%v", pending, err)
	}
	got, err := hostresurrect.ReadQueued(pending[0])
	if err != nil || got.Session != "g1" {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}

func TestRunOwnedBrokerPowerShellReceipts(t *testing.T) {
	fixed := time.UnixMilli(1_700_000_000_000)
	tests := []struct {
		name                           string
		startErr, identityErr, waitErr error
		wantErr                        string
		want                           []shellprov.Outcome
		wantClass                      []shellprov.ErrorClass
	}{
		{name: "success", want: []shellprov.Outcome{shellprov.OutcomeStarted, shellprov.OutcomeSucceeded}, wantClass: []shellprov.ErrorClass{shellprov.ErrorNone, shellprov.ErrorNone}},
		{name: "launch failure", startErr: errors.New("start"), wantErr: "start", want: []shellprov.Outcome{shellprov.OutcomeFailed}, wantClass: []shellprov.ErrorClass{shellprov.ErrorLaunch}},
		{name: "identity failure", identityErr: errors.New("identity"), wantErr: "identity", want: []shellprov.Outcome{shellprov.OutcomeFailed}, wantClass: []shellprov.ErrorClass{shellprov.ErrorUnknown}},
		{name: "exit failure", waitErr: errors.New("exit"), wantErr: "exit", want: []shellprov.Outcome{shellprov.OutcomeStarted, shellprov.OutcomeFailed}, wantClass: []shellprov.ErrorClass{shellprov.ErrorNone, shellprov.ErrorExitNonzero}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := exec.Command("cmd.exe", "/c", "exit 0")
			receipts := make([]shellprov.Receipt, 0, 2)
			deps := brokerRunDeps{
				start: func(cmd *exec.Cmd) error {
					if tt.startErr == nil {
						cmd.Process = &os.Process{Pid: 4321}
					}
					return tt.startErr
				},
				wait: func(*exec.Cmd) error { return tt.waitErr },
				createdMS: func(pid int) (int64, error) {
					if pid != 4321 {
						t.Fatalf("identity pid = %d", pid)
					}
					return fixed.UnixMilli(), tt.identityErr
				},
				append: func(r shellprov.Receipt) error { receipts = append(receipts, r); return nil },
				now:    func() time.Time { return fixed },
			}
			err := runOwnedBrokerPowerShell(cmd, deps)
			if tt.wantErr == "" && err != nil {
				t.Fatal(err)
			}
			if tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)) {
				t.Fatalf("err = %v, want %q", err, tt.wantErr)
			}
			if len(receipts) != len(tt.want) {
				t.Fatalf("receipts = %+v", receipts)
			}
			for i, receipt := range receipts {
				if receipt.Outcome != tt.want[i] || receipt.ErrorClass != tt.wantClass[i] {
					t.Fatalf("receipt[%d] = %+v", i, receipt)
				}
				if receipt.LaunchClass != shellprov.LaunchHook || receipt.ShellImage != shellprov.ShellPowerShell || receipt.ShellEdition != shellprov.EditionDesktop {
					t.Fatalf("bounded fields = %+v", receipt)
				}
				version, err := strconv.ParseFloat(receipt.ShellVersion, 64)
				if err != nil || version <= 5.0 {
					t.Fatalf("PowerShell version %q must parse and follow the prior desktop release: %v", receipt.ShellVersion, err)
				}
				if i > 0 && receipt.LaunchID != receipts[0].LaunchID {
					t.Fatalf("launch IDs differ: %q %q", receipts[0].LaunchID, receipt.LaunchID)
				}
			}
			if tt.startErr != nil && receipts[0].ChildPID != 0 {
				t.Fatalf("launch failure child PID = %d", receipts[0].ChildPID)
			}
		})
	}
}

func TestRunOwnedBrokerPowerShellReturnsAppendError(t *testing.T) {
	want := errors.New("append")
	cmd := exec.Command("cmd.exe", "/c", "exit 0")
	err := runOwnedBrokerPowerShell(cmd, brokerRunDeps{
		start:     func(cmd *exec.Cmd) error { cmd.Process = &os.Process{Pid: 7}; return nil },
		wait:      func(*exec.Cmd) error { return nil },
		createdMS: func(int) (int64, error) { return 8, nil },
		append:    func(shellprov.Receipt) error { return want },
		now:       func() time.Time { return time.UnixMilli(9) },
	})
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want append error", err)
	}
}
