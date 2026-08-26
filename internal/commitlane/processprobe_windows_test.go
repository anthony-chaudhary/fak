//go:build windows

package commitlane

import (
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/shellprov"
)

func testObservedDeps(t *testing.T, waitErr error, appendErr error) (processProbeDeps, *[]shellprov.Receipt) {
	t.Helper()
	receipts := []shellprov.Receipt{}
	deps := processProbeDeps{
		command:   exec.CommandContext,
		configure: func(*exec.Cmd) {},
		start: func(cmd *exec.Cmd) error {
			cmd.Process = &os.Process{Pid: 4321}
			_, _ = io.WriteString(cmd.Stdout, `[{"pid":1}]`)
			return nil
		},
		wait:      func(*exec.Cmd) error { return waitErr },
		createdMS: func(int) (int64, error) { return 1720000000123, nil },
		append:    func(r shellprov.Receipt) error { receipts = append(receipts, r); return appendErr },
		now:       func() time.Time { return time.UnixMilli(1720000000456) },
	}
	return deps, &receipts
}

func TestRunObservedProcessJSONReceipts(t *testing.T) {
	deps, receipts := testObservedDeps(t, nil, nil)
	out, err := runWindowsProcessJSONWithDeps(context.Background(), "powershell", nil, deps)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `[{"pid":1}]` {
		t.Fatalf("stdout = %q", out)
	}
	if len(*receipts) != 2 {
		t.Fatalf("receipts = %d", len(*receipts))
	}
	if (*receipts)[0].Outcome != shellprov.OutcomeStarted || (*receipts)[1].Outcome != shellprov.OutcomeSucceeded {
		t.Fatalf("receipts = %+v", *receipts)
	}
	for _, r := range *receipts {
		if r.ChildPID != 4321 || r.ChildCreatedUTCMS != 1720000000123 || r.LaunchClass != shellprov.LaunchProbe {
			t.Fatalf("receipt = %+v", r)
		}
		if r.ShellImage != shellprov.ShellPowerShell || r.ShellEdition != shellprov.EditionDesktop {
			t.Fatalf("metadata = %+v", r)
		}
	}
}

func TestRunObservedProcessJSONFailureAndBestEffortAppend(t *testing.T) {
	deps, receipts := testObservedDeps(t, errors.New("exit status 9"), errors.New("append failed"))
	deps.start = func(cmd *exec.Cmd) error {
		cmd.Process = &os.Process{Pid: 4321}
		_, _ = io.WriteString(cmd.Stderr, "stderr detail\n")
		return nil
	}
	out, err := runWindowsProcessJSONWithDeps(context.Background(), "pwsh", nil, deps)
	if out != nil || err == nil || err.Error() != "pwsh: stderr detail" {
		t.Fatalf("output=%q error=%v", out, err)
	}
	if len(*receipts) != 2 || (*receipts)[1].Outcome != shellprov.OutcomeFailed || (*receipts)[1].ErrorClass != shellprov.ErrorExitNonzero {
		t.Fatalf("receipts = %+v", *receipts)
	}
	if (*receipts)[0].ShellImage != shellprov.ShellPwsh || (*receipts)[0].ShellEdition != shellprov.EditionCore {
		t.Fatalf("metadata = %+v", (*receipts)[0])
	}
}

func TestRunObservedProcessJSONLaunchFailure(t *testing.T) {
	deps, receipts := testObservedDeps(t, nil, nil)
	deps.start = func(*exec.Cmd) error { return errors.New("not found") }
	out, err := runWindowsProcessJSONWithDeps(context.Background(), "powershell", nil, deps)
	if out != nil || err == nil || err.Error() != "powershell: not found" {
		t.Fatalf("output=%q error=%v", out, err)
	}
	if len(*receipts) != 1 || (*receipts)[0].ErrorClass != shellprov.ErrorLaunch || (*receipts)[0].ChildPID != 0 {
		t.Fatalf("receipts = %+v", *receipts)
	}
}
