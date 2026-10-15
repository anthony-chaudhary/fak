//go:build !windows

package devcmd

import (
	"errors"
	"io"
	"os"
	"os/exec"
	"syscall"
	"time"
)

// The anchor owns no Git fixture state: handle its private mode before TestMain.
func init() {
	if len(os.Args) < 2 || os.Args[len(os.Args)-1] != "companion-probe-anchor" {
		return
	}
	pid := os.Getpid()
	if pgid, err := syscall.Getpgid(pid); err != nil || pgid != pid {
		os.Exit(2)
	}
	// Only the parent owns the pipe writer. Parent death closes it, so externally
	// killing the test process cannot leave the separate probe group running.
	// Backstop a broken owner/pipe fixture without permitting a long-lived anchor.
	// Normal probe admission is capped at 30 seconds plus a one-second pipe drain.
	watchdog := time.AfterFunc(35*time.Second, func() { _ = syscall.Kill(-pid, syscall.SIGKILL) })
	defer watchdog.Stop()
	_, _ = io.Copy(io.Discard, os.Stdin)
	_ = syscall.Kill(-pid, syscall.SIGKILL)
	os.Exit(2)
}

func runOwnedCompanionProbe(cmd *exec.Cmd) (err error) {
	// Retain a separate group leader unreaped until all group signaling is done.
	// Its PID cannot be recycled even if the probe exits before its descendants.
	readEnd, writeEnd, err := os.Pipe()
	if err != nil {
		return err
	}
	anchor := exec.Command(os.Args[0], "-test.run=^TestCompanionProbeBounds$", "--", "companion-probe-anchor")
	anchor.Env = cmd.Env
	anchor.Stdin = readEnd
	anchor.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	startErr := anchor.Start()
	_ = readEnd.Close()
	if startErr != nil {
		_ = writeEnd.Close()
		return startErr
	}
	pgid := anchor.Process.Pid
	killGroup := func() error {
		err := syscall.Kill(-pgid, syscall.SIGKILL)
		if errors.Is(err, syscall.ESRCH) {
			return nil
		}
		return err
	}
	defer func() {
		// Never signal this numeric group again after reaping its identity anchor.
		err = errors.Join(err, killGroup())
		_ = writeEnd.Close()
		waitErr := anchor.Wait()
		var exitErr *exec.ExitError
		if waitErr != nil && !errors.As(waitErr, &exitErr) {
			err = errors.Join(err, waitErr)
		}
	}()
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pgid: pgid}
	cmd.Cancel = killGroup
	return cmd.Run()
}

func companionProbeSocketClosed(err error) bool {
	return errors.Is(err, io.EOF) || errors.Is(err, syscall.ECONNRESET)
}
