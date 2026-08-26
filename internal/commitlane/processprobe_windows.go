//go:build windows

package commitlane

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/shellprov"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type processProbeDeps struct {
	command   func(context.Context, string, ...string) *exec.Cmd
	configure func(*exec.Cmd)
	start     func(*exec.Cmd) error
	wait      func(*exec.Cmd) error
	createdMS func(int) (int64, error)
	append    func(shellprov.Receipt) error
	now       func() time.Time
}

func runWindowsProcessJSON(ctx context.Context, name string, args ...string) ([]byte, error) {
	return runWindowsProcessJSONWithDeps(ctx, name, args, defaultProcessProbeDeps())
}

func defaultProcessProbeDeps() processProbeDeps {
	return processProbeDeps{
		command:   exec.CommandContext,
		configure: windowgate.ConfigureBackgroundCommand,
		start:     func(cmd *exec.Cmd) error { return cmd.Start() },
		wait:      func(cmd *exec.Cmd) error { return cmd.Wait() },
		createdMS: processProbeCreatedAtMS,
		append: func(receipt shellprov.Receipt) error {
			base, err := os.UserConfigDir()
			if err != nil || base == "" {
				base = os.TempDir()
			}
			return shellprov.Append(filepath.Join(base, "fak", "shell-provenance.jsonl"), receipt, shellprov.DefaultMaxRows)
		},
		now: time.Now,
	}
}

func runWindowsProcessJSONWithDeps(ctx context.Context, name string, args []string, deps processProbeDeps) ([]byte, error) {
	cmd := deps.command(ctx, name, args...)
	deps.configure(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	launchID := "attempt-" + shellprov.ChildIdentity(os.Getpid(), deps.now().UnixNano())
	shellImage, edition, version := processProbeShell(name)

	if err := deps.start(cmd); err != nil {
		recordProcessProbe(deps, launchID, 0, 0, shellImage, edition, version, shellprov.OutcomeFailed, shellprov.ErrorLaunch)
		return nil, decorateProcessProbeError(name, stderr.String(), err)
	}
	pid := cmd.Process.Pid
	createdMS, identityErr := deps.createdMS(pid)
	if identityErr == nil {
		launchID = shellprov.ChildIdentity(pid, createdMS)
		recordProcessProbe(deps, launchID, pid, createdMS, shellImage, edition, version, shellprov.OutcomeStarted, shellprov.ErrorNone)
	}

	waitErr := deps.wait(cmd)
	if identityErr == nil {
		outcome, class := shellprov.OutcomeSucceeded, shellprov.ErrorNone
		if waitErr != nil {
			outcome, class = shellprov.OutcomeFailed, shellprov.ErrorExitNonzero
			if ctx.Err() != nil {
				class = shellprov.ErrorTimeout
			}
		}
		recordProcessProbe(deps, launchID, pid, createdMS, shellImage, edition, version, outcome, class)
	}
	if waitErr != nil {
		return nil, decorateProcessProbeError(name, stderr.String(), waitErr)
	}
	return stdout.Bytes(), nil
}

func decorateProcessProbeError(name, stderr string, err error) error {
	detail := strings.TrimSpace(stderr)
	if detail == "" {
		detail = err.Error()
	}
	return fmt.Errorf("%s: %s", name, detail)
}

func processProbeShell(name string) (shellprov.ShellImage, shellprov.ShellEdition, string) {
	if strings.EqualFold(filepath.Base(name), "pwsh") || strings.EqualFold(filepath.Base(name), "pwsh.exe") {
		return shellprov.ShellPwsh, shellprov.EditionCore, "7"
	}
	return shellprov.ShellPowerShell, shellprov.EditionDesktop, "5.1"
}

func recordProcessProbe(deps processProbeDeps, launchID string, pid int, createdMS int64, image shellprov.ShellImage, edition shellprov.ShellEdition, version string, outcome shellprov.Outcome, class shellprov.ErrorClass) {
	receipt, err := shellprov.New(deps.now(), shellprov.Fields{
		LaunchID: launchID, ParentPID: os.Getpid(), ChildPID: pid, ChildCreatedUTCMS: createdMS,
		LaunchClass: shellprov.LaunchProbe, ShellImage: image, ShellEdition: edition, ShellVersion: version,
		Outcome: outcome, ErrorClass: class,
	})
	if err == nil {
		_ = deps.append(receipt)
	}
}

func processProbeCreatedAtMS(pid int) (int64, error) {
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
