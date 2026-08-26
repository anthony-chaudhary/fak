//go:build windows

package main

import (
	"encoding/base64"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
	"unicode/utf16"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
	"github.com/anthony-chaudhary/fak/internal/shellprov"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func encodedPowerShellCommand(script string) string {
	units := utf16.Encode([]rune(script))
	b := make([]byte, len(units)*2)
	for i, v := range units {
		b[2*i], b[2*i+1] = byte(v), byte(v>>8)
	}
	return base64.StdEncoding.EncodeToString(b)
}

type brokerRunDeps struct {
	start     func(*exec.Cmd) error
	wait      func(*exec.Cmd) error
	createdMS func(int) (int64, error)
	append    func(shellprov.Receipt) error
	now       func() time.Time
}

var runInteractiveBrokerTask = func(taskName string) error {
	script := "Start-ScheduledTask -TaskName '" + strings.ReplaceAll(taskName, "'", "''") + "'"
	cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encodedPowerShellCommand(script))
	windowgate.ConfigureBackgroundCommand(cmd)
	return runOwnedBrokerPowerShell(cmd, brokerRunDeps{
		start:     func(cmd *exec.Cmd) error { return cmd.Start() },
		wait:      func(cmd *exec.Cmd) error { return cmd.Wait() },
		createdMS: processCreatedAtMS,
		append: func(receipt shellprov.Receipt) error {
			return shellprov.Append(filepath.Join(hostRelaunchSpoolDir(), "shell-provenance.jsonl"), receipt, shellprov.DefaultMaxRows)
		},
		now: time.Now,
	})
}

func runOwnedBrokerPowerShell(cmd *exec.Cmd, deps brokerRunDeps) error {
	now := deps.now()
	launchID := "attempt-" + shellprov.ChildIdentity(os.Getpid(), now.UnixNano())
	if err := deps.start(cmd); err != nil {
		receiptErr := recordBrokerShell(deps, launchID, 0, 0, shellprov.OutcomeFailed, shellprov.ErrorLaunch)
		return errors.Join(err, receiptErr)
	}

	pid := cmd.Process.Pid
	createdMS, identityErr := deps.createdMS(pid)
	var receiptErr error
	if identityErr != nil {
		receiptErr = recordBrokerShell(deps, launchID, pid, 0, shellprov.OutcomeFailed, shellprov.ErrorUnknown)
	} else {
		receiptErr = recordBrokerShell(deps, launchID, pid, createdMS, shellprov.OutcomeStarted, shellprov.ErrorNone)
	}

	waitErr := deps.wait(cmd)
	if identityErr == nil {
		if waitErr == nil {
			receiptErr = errors.Join(receiptErr, recordBrokerShell(deps, launchID, pid, createdMS, shellprov.OutcomeSucceeded, shellprov.ErrorNone))
		} else {
			receiptErr = errors.Join(receiptErr, recordBrokerShell(deps, launchID, pid, createdMS, shellprov.OutcomeFailed, shellprov.ErrorExitNonzero))
		}
	}
	return errors.Join(waitErr, identityErr, receiptErr)
}

func recordBrokerShell(deps brokerRunDeps, launchID string, pid int, createdMS int64, outcome shellprov.Outcome, errorClass shellprov.ErrorClass) error {
	receipt, err := shellprov.New(deps.now(), shellprov.Fields{
		LaunchID:          launchID,
		ParentPID:         os.Getpid(),
		ChildPID:          pid,
		ChildCreatedUTCMS: createdMS,
		LaunchClass:       shellprov.LaunchHook,
		ShellImage:        shellprov.ShellPowerShell,
		ShellEdition:      shellprov.EditionDesktop,
		ShellVersion:      "5.1",
		Outcome:           outcome,
		ErrorClass:        errorClass,
	})
	if err != nil {
		return err
	}
	return deps.append(receipt)
}
func processCreatedAtMS(pid int) (int64, error) {
	const processQueryLimitedInformation = 0x1000
	handle, err := syscall.OpenProcess(processQueryLimitedInformation, false, uint32(pid))
	if err != nil {
		return 0, err
	}
	defer syscall.CloseHandle(handle)
	var created, exited, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(handle, &created, &exited, &kernel, &user); err != nil {
		return 0, err
	}
	return created.Nanoseconds() / int64(time.Millisecond), nil
}

func hostRelaunchSpoolDir() string {
	if dir := strings.TrimSpace(os.Getenv("FAK_HOST_RELAUNCH_DIR")); dir != "" {
		return dir
	}
	base, err := os.UserConfigDir()
	if err != nil || base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "fak", "host", "relaunch")
}

// The S4U watchdog persists before signaling the desktop broker. If TermService
// is unavailable, Start-ScheduledTask may fail but the request remains queued and
// the broker's AtLogOn trigger drains it after the interactive host recovers.
func launchHostSessionPlatform(req hostresurrect.Request) (int, error) {
	if len(req.Command) == 0 {
		return 0, errors.New("empty relaunch command")
	}
	if _, err := hostresurrect.Enqueue(hostRelaunchSpoolDir(), req); err != nil {
		return 0, err
	}
	_ = runInteractiveBrokerTask("FakHostRelaunchBroker")
	return 0, nil
}
