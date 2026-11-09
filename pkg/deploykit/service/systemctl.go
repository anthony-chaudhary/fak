package service

// The systemd driver: both scopes, driven through the Command boundary
// (systemctl [--user] ...). Read-back reuses internal/servicewatchdog's typed
// systemctl-show parsing; the control sequence is stop → publish unit →
// daemon-reload → enable → start → read back, with restore putting the prior
// unit bytes and prior running state back on failure.

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/servicewatchdog"
)

type systemdDriver struct{}

func (m *Manager) unitName() string {
	if strings.HasSuffix(m.name, ".service") {
		return m.name
	}
	return m.name + ".service"
}

// unitDirPath resolves the systemd unit directory: overridden, or the
// platform default for the scope.
func (m *Manager) unitDirPath() string {
	if m.unitDir != "" {
		return m.unitDir
	}
	if m.scope == ScopeUser {
		home, err := os.UserHomeDir()
		if err == nil && home != "" {
			return filepath.Join(home, ".config", "systemd", "user")
		}
		return "/home/user/.config/systemd/user"
	}
	return "/etc/systemd/system"
}

func (m *Manager) unitPath() string { return filepath.Join(m.unitDirPath(), m.unitName()) }

// runSystemctl runs one systemctl invocation through the effective command
// boundary (which carries --user for user scope).
func (m *Manager) runSystemctl(args ...string) error {
	return m.command().Run(args...)
}

// systemdManager wraps the Command boundary in a servicewatchdog.Manager for
// the typed read-back path.
func (m *Manager) systemdManager() servicewatchdog.Manager {
	return servicewatchdog.Manager{Command: m.command(), Unit: m.unitName()}
}

func (systemdDriver) snapshot(m *Manager, def Definition) (Snapshot, error) {
	path := m.unitPath()
	prior, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		prior = nil
	} else if err != nil {
		return Snapshot{}, fmt.Errorf("read prior unit %s: %w", path, err)
	}
	running := false
	if st, serr := m.systemdManager().Status(); serr == nil {
		running = st.ActiveState == "active"
	}
	enabled := m.runSystemctl("is-enabled", m.unitName()) == nil
	return Snapshot{
		Path:       path,
		Prior:      prior,
		WasRunning: running,
		WasEnabled: enabled,
	}, nil
}

func (systemdDriver) apply(m *Manager, snap Snapshot, def Definition) error {
	unit := m.unitName()
	if snap.WasRunning {
		if err := m.runSystemctl("stop", unit); err != nil {
			return phaseErr("stop prior service", err)
		}
	}
	if err := stageDocument(m.unitPath(), def.Text); err != nil {
		return phaseErr("write unit", err)
	}
	if err := m.runSystemctl("daemon-reload"); err != nil {
		return phaseErr("daemon-reload", err)
	}
	if err := m.runSystemctl("enable", unit); err != nil {
		return phaseErr("enable", err)
	}
	if err := m.runSystemctl("start", unit); err != nil {
		return phaseErr("start", err)
	}
	rb, err := (systemdDriver{}).readback(m)
	if err != nil {
		return phaseErr("read back", err)
	}
	if rb.State != Running {
		return phaseErr("read back", fmt.Errorf("unit %s is %s after start", unit, rb.State))
	}
	return nil
}

func (systemdDriver) restore(m *Manager, snap Snapshot) error {
	unit := m.unitName()
	var errs []error
	if snap.exists() {
		if snap.Prior != nil {
			if err := stageDocument(m.unitPath(), string(snap.Prior)); err != nil {
				errs = append(errs, fmt.Errorf("restore unit file: %w", err))
			}
		} else if err := os.Remove(m.unitPath()); err != nil && !errors.Is(err, os.ErrNotExist) {
			errs = append(errs, fmt.Errorf("remove unit file: %w", err))
		}
		if err := m.runSystemctl("daemon-reload"); err != nil {
			errs = append(errs, fmt.Errorf("daemon-reload: %w", err))
		}
		if snap.WasEnabled {
			if err := m.runSystemctl("enable", unit); err != nil {
				errs = append(errs, fmt.Errorf("enable: %w", err))
			}
		} else if err := m.runSystemctl("disable", unit); err != nil {
			errs = append(errs, fmt.Errorf("disable: %w", err))
		}
		if snap.WasRunning {
			if err := m.runSystemctl("start", unit); err != nil {
				errs = append(errs, fmt.Errorf("start: %w", err))
			}
		}
	} else {
		// Nothing was registered before: drop the partially published unit so a
		// fresh-install failure leaves no orphan behind.
		if err := os.Remove(m.unitPath()); err != nil && !errors.Is(err, os.ErrNotExist) {
			errs = append(errs, fmt.Errorf("remove partial unit: %w", err))
		}
		if err := m.runSystemctl("daemon-reload"); err != nil {
			errs = append(errs, fmt.Errorf("daemon-reload: %w", err))
		}
	}
	return errors.Join(errs...)
}

func (systemdDriver) readback(m *Manager) (Readback, error) {
	st, err := m.systemdManager().Status()
	if err != nil {
		return Readback{State: Unknown}, fmt.Errorf("systemctl show: %w", err)
	}
	rb := Readback{PID: st.MainPID}
	if st.ActiveState == "active" {
		rb.State = Running
	} else {
		rb.State = Stopped
	}
	if out, oerr := m.command().Output("show", m.unitName(), "--no-pager", "--property=ExecStart"); oerr == nil {
		rb.Command = systemdExecStart(out)
	}
	return rb, nil
}

// systemdExecStart extracts the ExecStart value from systemctl show output.
func systemdExecStart(out []byte) string {
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if v, ok := strings.CutPrefix(line, "ExecStart="); ok {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func (systemdDriver) start(m *Manager) error   { return m.runSystemctl("start", m.unitName()) }
func (systemdDriver) stop(m *Manager) error    { return m.runSystemctl("stop", m.unitName()) }
func (systemdDriver) restart(m *Manager) error { return m.runSystemctl("restart", m.unitName()) }
