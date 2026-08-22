package windowgate

import (
	"errors"
	"os/exec"
	"sync"
)

// CleanupOwner is implemented by testing.T and testing.B. Cleanup callbacks run
// even when a test stops at Fatal, FailNow, or a parent-context cancellation.
type CleanupOwner interface {
	Cleanup(func())
}

// OwnedProcess binds a started command to a cleanup owner. On Windows the
// command and all descendants share a Job Object. Wait reports the command's
// exit result; Close terminates its envelope and joins the root.
type OwnedProcess struct {
	cmd  *exec.Cmd
	job  *JobObject
	done chan struct{}

	releaseOnce sync.Once
	releaseErr  error
	waitErr     error
}

// StartOwnedProcess starts cmd inside the platform process envelope and
// immediately registers its teardown with owner. On Windows the envelope is a
// KILL_ON_JOB_CLOSE Job Object, so a successful or failed root exit cannot
// strand descendants that retain test scratch paths.
func StartOwnedProcess(owner CleanupOwner, cmd *exec.Cmd) (*OwnedProcess, error) {
	if owner == nil {
		return nil, errors.New("windowgate: StartOwnedProcess requires a cleanup owner")
	}
	job, err := StartInNewJob(cmd)
	if err != nil {
		return nil, err
	}
	p := &OwnedProcess{cmd: cmd, job: job, done: make(chan struct{})}
	go func() {
		p.waitErr = cmd.Wait()
		p.release()
		close(p.done)
	}()
	owner.Cleanup(func() { _ = p.Close() })
	return p, nil
}

// PID returns the root process ID, or zero when no root was started.
func (p *OwnedProcess) PID() int {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return 0
	}
	return p.cmd.Process.Pid
}

// Wait joins the root and returns its exit result. On Windows the Job Object is
// released before Wait returns, so every contained descendant is also gone.
func (p *OwnedProcess) Wait() error {
	if p == nil {
		return nil
	}
	<-p.done
	return p.waitErr
}

// Close terminates the process envelope and joins the root. It is idempotent.
func (p *OwnedProcess) Close() error {
	if p == nil {
		return nil
	}
	p.release()
	<-p.done
	return p.releaseErr
}

func (p *OwnedProcess) release() {
	p.releaseOnce.Do(func() {
		p.releaseErr = p.job.Close()
		// Off Windows the job handle is a no-op. Killing the root keeps cleanup
		// bounded there too; Windows normally reaps it through the job first.
		if p.cmd != nil && p.cmd.Process != nil {
			_ = p.cmd.Process.Kill()
		}
	})
}
