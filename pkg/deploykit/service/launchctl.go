package service

// The launchd driver: both scopes, driven through the Command boundary
// (launchctl bootstrap/bootout/kickstart/print). Read-back reuses
// internal/systemservice's typed launchctl-print parsing. The control sequence
// is bootout → publish plist → bootstrap → kickstart (only when the job has
// not come up) → read back, with restore putting the prior plist and prior
// running state back on failure.

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/systemservice"
)

type launchdDriver struct{}

// launchdDomain resolves the launchctl domain: "system" for system scope,
// gui/<uid> for user scope.
func (m *Manager) launchdDomain() string {
	if m.scope == ScopeSystem {
		return "system"
	}
	return fmt.Sprintf("gui/%d", m.uid)
}

func (m *Manager) launchdTarget() string { return m.launchdDomain() + "/" + m.name }

// plistDirPath resolves the launchd plist directory: overridden, or the
// platform default for the scope.
func (m *Manager) plistDirPath() string {
	if m.plistDir != "" {
		return m.plistDir
	}
	if m.scope == ScopeSystem {
		return "/Library/LaunchDaemons"
	}
	home, err := os.UserHomeDir()
	if err == nil && home != "" {
		return filepath.Join(home, "Library", "LaunchAgents")
	}
	return "/Library/LaunchAgents"
}

func (m *Manager) plistPath() string { return filepath.Join(m.plistDirPath(), m.name+".plist") }

// launchdStatus prints the job and parses it into a typed status.
func (m *Manager) launchdStatus() (*systemservice.LaunchdStatus, error) {
	out, err := m.command().Output("print", m.launchdTarget())
	if err != nil {
		return nil, err
	}
	return systemservice.ParseLaunchctlPrint(string(out))
}

// bootout unloads the job, tolerating "not loaded" refusals so a stop of an
// already-unloaded job is a no-op.
func (m *Manager) bootout() error {
	err := m.command().Run("bootout", m.launchdTarget())
	if err == nil {
		return nil
	}
	msg := strings.ToLower(err.Error())
	for _, sentinel := range []string{"not found", "no such process", "not loaded", "could not find"} {
		if strings.Contains(msg, sentinel) {
			return nil
		}
	}
	return err
}

func (launchdDriver) snapshot(m *Manager, def Definition) (Snapshot, error) {
	path := m.plistPath()
	prior, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		prior = nil
	} else if err != nil {
		return Snapshot{}, fmt.Errorf("read prior plist %s: %w", path, err)
	}
	running := false
	if st, perr := m.launchdStatus(); perr == nil {
		running = st.IsRunning()
	}
	return Snapshot{
		Path:       path,
		Prior:      prior,
		WasRunning: running,
		WasEnabled: prior != nil,
	}, nil
}

func (launchdDriver) apply(m *Manager, snap Snapshot, def Definition) error {
	if snap.exists() {
		if err := m.bootout(); err != nil {
			return phaseErr("stop prior service", err)
		}
	}
	if err := stageDocument(m.plistPath(), def.Text); err != nil {
		return phaseErr("write plist", err)
	}
	if err := m.command().Run("bootstrap", m.launchdDomain(), m.plistPath()); err != nil {
		return phaseErr("bootstrap", err)
	}
	// bootstrap honors RunAtLoad; kickstart only when the job has not come up.
	st, err := m.launchdStatus()
	if err != nil {
		return phaseErr("read back", err)
	}
	if !st.IsRunning() {
		if err := m.command().Run("kickstart", "-k", m.launchdTarget()); err != nil {
			return phaseErr("start", err)
		}
	}
	rb, err := (launchdDriver{}).readback(m)
	if err != nil {
		return phaseErr("read back", err)
	}
	if rb.State != Running {
		return phaseErr("read back", fmt.Errorf("job %s is %s after bootstrap", m.launchdTarget(), rb.State))
	}
	return nil
}

func (launchdDriver) restore(m *Manager, snap Snapshot) error {
	var errs []error
	if snap.Prior != nil {
		if err := stageDocument(m.plistPath(), string(snap.Prior)); err != nil {
			errs = append(errs, fmt.Errorf("restore plist: %w", err))
		}
	} else if err := os.Remove(m.plistPath()); err != nil && !errors.Is(err, os.ErrNotExist) {
		errs = append(errs, fmt.Errorf("remove plist: %w", err))
	}
	if err := m.bootout(); err != nil {
		errs = append(errs, fmt.Errorf("bootout: %w", err))
	}
	if snap.WasEnabled {
		if err := m.command().Run("bootstrap", m.launchdDomain(), m.plistPath()); err != nil {
			errs = append(errs, fmt.Errorf("bootstrap: %w", err))
		}
		if snap.WasRunning {
			if err := m.command().Run("kickstart", "-k", m.launchdTarget()); err != nil {
				errs = append(errs, fmt.Errorf("kickstart: %w", err))
			}
		}
	}
	return errors.Join(errs...)
}

func (launchdDriver) readback(m *Manager) (Readback, error) {
	st, err := m.launchdStatus()
	if err != nil {
		if errors.Is(err, systemservice.ErrLaunchctlServiceNotFound) {
			return Readback{State: Unknown}, nil
		}
		return Readback{State: Unknown}, err
	}
	rb := Readback{Command: st.Program, PID: st.PID}
	if st.IsRunning() {
		rb.State = Running
	} else {
		rb.State = Stopped
	}
	return rb, nil
}

func (launchdDriver) start(m *Manager) error {
	if err := m.command().Run("bootstrap", m.launchdDomain(), m.plistPath()); err != nil {
		// bootstrap fails when the job is already booted; kickstart below is
		// the force-start that covers both cases.
		if !strings.Contains(strings.ToLower(err.Error()), "already") {
			return err
		}
	}
	return m.command().Run("kickstart", "-k", m.launchdTarget())
}

func (launchdDriver) stop(m *Manager) error { return m.bootout() }

func (launchdDriver) restart(m *Manager) error {
	if err := m.bootout(); err != nil {
		return err
	}
	return (launchdDriver{}).start(m)
}
