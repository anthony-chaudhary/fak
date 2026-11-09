package service

// The scheduled-task driver: the Windows user-scope control plane, driven
// through the Command boundary (schtasks /Query /End /Create /Run /Delete).
// The task XML is captured from the registered task itself (/Query /XML) so a
// restore re-registers exactly what was there; schtasks may emit a UTF-16 BOM
// on that output, which is stripped before capture. The control sequence is
// /End → publish task XML → /Create /F → /Run → read back, with restore
// re-registering the prior XML and prior running state on failure.

import (
	"errors"
	"fmt"
	"strings"
)

type scheduledTaskDriver struct{}

// taskXML captures the registered task's XML definition. A missing task (or an
// unrunnable schtasks) yields nil rather than an error: snapshot treats both
// as "nothing registered yet".
func (m *Manager) taskXML() ([]byte, error) {
	out, err := m.command().Output("/Query", "/TN", m.name, "/XML")
	if err != nil {
		return nil, nil
	}
	return []byte(strings.TrimPrefix(string(out), "\ufeff")), nil
}

// taskStatusFrom extracts the Status value from `schtasks /Query /FO LIST`
// output ("Status:         Running").
func taskStatusFrom(listOutput string) string {
	for _, line := range strings.Split(listOutput, "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "Status:"); ok {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func (scheduledTaskDriver) snapshot(m *Manager, def Definition) (Snapshot, error) {
	prior, err := m.taskXML()
	if err != nil {
		return Snapshot{}, fmt.Errorf("query task %s: %w", m.name, err)
	}
	running := false
	if out, qerr := m.command().Output("/Query", "/TN", m.name, "/FO", "LIST"); qerr == nil {
		running = strings.EqualFold(taskStatusFrom(string(out)), "Running")
	}
	return Snapshot{
		Path:       def.Path,
		Prior:      prior,
		WasRunning: running,
		WasEnabled: prior != nil,
	}, nil
}

func (scheduledTaskDriver) apply(m *Manager, snap Snapshot, def Definition) error {
	if def.Path == "" {
		return phaseErr("write task xml", errors.New("Definition.Path (task XML location) is required"))
	}
	if snap.exists() {
		// /End fails when the task is missing or already stopped; the port-free
		// preflight the caller runs afterwards is the real gate, so note it in
		// the log stream instead of failing the update.
		if err := m.command().Run("/End", "/TN", m.name); err != nil {
			_ = err // tolerated: see above
		}
	}
	if err := stageDocument(def.Path, def.Text); err != nil {
		return phaseErr("write task xml", err)
	}
	if err := m.command().Run("/Create", "/TN", m.name, "/XML", def.Path, "/F"); err != nil {
		return phaseErr("register task", err)
	}
	if err := m.command().Run("/Run", "/TN", m.name); err != nil {
		return phaseErr("start", err)
	}
	rb, err := (scheduledTaskDriver{}).readback(m)
	if err != nil {
		return phaseErr("read back", err)
	}
	if rb.State != Running {
		return phaseErr("read back", fmt.Errorf("task %s is %s after run", m.name, rb.State))
	}
	return nil
}

func (scheduledTaskDriver) restore(m *Manager, snap Snapshot) error {
	var errs []error
	if snap.Prior != nil {
		if snap.Path != "" {
			if err := stageDocument(snap.Path, string(snap.Prior)); err != nil {
				errs = append(errs, fmt.Errorf("restore task xml: %w", err))
			}
		}
		if err := m.command().Run("/Create", "/TN", m.name, "/XML", snap.Path, "/F"); err != nil {
			errs = append(errs, fmt.Errorf("re-register task: %w", err))
		}
		if snap.WasRunning {
			if err := m.command().Run("/Run", "/TN", m.name); err != nil {
				errs = append(errs, fmt.Errorf("run task: %w", err))
			}
		}
	} else if err := m.command().Run("/Delete", "/TN", m.name, "/F"); err != nil {
		// A fresh-install failure must not leave the half-registered task behind.
		errs = append(errs, fmt.Errorf("remove partial task: %w", err))
	}
	return errors.Join(errs...)
}

func (scheduledTaskDriver) readback(m *Manager) (Readback, error) {
	out, err := m.command().Output("/Query", "/TN", m.name, "/FO", "LIST")
	if err != nil {
		return Readback{State: Unknown}, nil // task not registered
	}
	rb := Readback{} // schtasks does not expose the command line or PID
	if strings.EqualFold(taskStatusFrom(string(out)), "Running") {
		rb.State = Running
	} else {
		rb.State = Stopped
	}
	return rb, nil
}

func (scheduledTaskDriver) start(m *Manager) error {
	return m.command().Run("/Run", "/TN", m.name)
}

func (scheduledTaskDriver) stop(m *Manager) error {
	return m.command().Run("/End", "/TN", m.name)
}

func (scheduledTaskDriver) restart(m *Manager) error {
	if err := m.command().Run("/End", "/TN", m.name); err != nil {
		_ = err // a missing/stopped task makes /End fail; /Run below is the goal
	}
	return m.command().Run("/Run", "/TN", m.name)
}
