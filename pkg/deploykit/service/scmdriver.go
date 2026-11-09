package service

// The Windows SCM driver: the system-scope control plane, driven against the
// plain SCMBoundary value types so the full sequence — snapshot → stop →
// sha256-named staged executable → config update → recovery ladder → start →
// read back, restore on failure — runs and is fake-testable on every GOOS.
// The live adapter (golang.org/x/sys/windows/svc/mgr) lives in scm_windows.go;
// non-Windows hosts get a refusal from scm_other.go unless a boundary is
// injected.

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

type scmDriver struct{}

// scmPrior carries the driver-specific prior state for an SCM registration.
type scmPrior struct {
	cfg         SCMConfig
	actions     []RecoveryAction
	resetPeriod uint32
	nonCrash    bool
	wasActive   bool
}

// The fak recovery ladder: restart after 3s / 10s / 30s, failure counter
// reset after 86400s of stable running, applied to non-crash failures too.
var scmRecoveryLadder = []RecoveryAction{
	{Type: ActionRestart, Delay: 3 * time.Second},
	{Type: ActionRestart, Delay: 10 * time.Second},
	{Type: ActionRestart, Delay: 30 * time.Second},
}

const scmRecoveryResetSec = 86400

// Stop-poll knobs; tests shrink them or let the fakes settle instantly.
var (
	scmStopTimeout = 30 * time.Second
	scmStopPoll    = 200 * time.Millisecond
)

func (m *Manager) openSCMService() (SCMService, error) {
	b, err := m.scmBoundary()
	if err != nil {
		return nil, err
	}
	svc, err := b.OpenService(m.name)
	if err != nil {
		return nil, fmt.Errorf("open service %s: %w", m.name, err)
	}
	return svc, nil
}

func (scmDriver) snapshot(m *Manager, def Definition) (Snapshot, error) {
	svc, err := m.openSCMService()
	if err != nil {
		if errors.Is(err, ErrServiceDoesNotExist) {
			return Snapshot{}, nil
		}
		return Snapshot{}, err
	}
	defer svc.Close()
	cfg, err := svc.Config()
	if err != nil {
		return Snapshot{}, fmt.Errorf("query configuration: %w", err)
	}
	actions, err := svc.RecoveryActions()
	if err != nil {
		return Snapshot{}, fmt.Errorf("query recovery actions: %w", err)
	}
	resetPeriod, err := svc.ResetPeriod()
	if err != nil {
		return Snapshot{}, fmt.Errorf("query recovery reset period: %w", err)
	}
	nonCrash, err := svc.RecoveryActionsOnNonCrashFailures()
	if err != nil {
		return Snapshot{}, fmt.Errorf("query recovery failure flag: %w", err)
	}
	status, err := svc.Query()
	if err != nil {
		return Snapshot{}, fmt.Errorf("query state: %w", err)
	}
	wasActive := status.State != StateStopped && status.State != StateStopPending
	return Snapshot{
		Path:       cfg.BinaryPathName,
		WasRunning: wasActive,
		Platform: &scmPrior{
			cfg:         cfg,
			actions:     append([]RecoveryAction(nil), actions...),
			resetPeriod: resetPeriod,
			nonCrash:    nonCrash,
			wasActive:   wasActive,
		},
	}, nil
}

// stageExecutable publishes def.Executable to its content-addressed path under
// the manager's binary directory.
func (m *Manager) stageExecutable(def Definition) (string, error) {
	if def.Executable == "" {
		return "", errors.New("Definition.Executable is required for the Windows SCM driver")
	}
	target, err := planStagedExecutable(m.binaryDirPath(), def.Executable)
	if err != nil {
		return "", phaseErr("plan service executable", err)
	}
	if err := stageExecutable(def.Executable, target); err != nil {
		return "", phaseErr("stage service executable", err)
	}
	return target, nil
}

// scmConfigFor builds the new SCM configuration for def at the staged path.
func (m *Manager) scmConfigFor(def Definition, staged string) SCMConfig {
	return SCMConfig{
		StartType:        StartAutomatic,
		ErrorControl:     ErrorNormal,
		BinaryPathName:   scmCommandLine(staged, def.Args...),
		DisplayName:      "fak service " + m.name,
		Description:      "fak service " + m.name + " (deploykit)",
		ServiceStartName: "NT AUTHORITY\\LocalService",
		SidType:          SidTypeRestricted,
	}
}

func configureSCMRecovery(s SCMService) error {
	if err := s.SetRecoveryActions(scmRecoveryLadder, scmRecoveryResetSec); err != nil {
		return err
	}
	return s.SetRecoveryActionsOnNonCrashFailures(true)
}

// stopSCMService requests a stop and polls until the service reports stopped,
// bounded by timeout.
func stopSCMService(s SCMService, timeout, poll time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		status, err := s.Query()
		if err != nil {
			return fmt.Errorf("query while stopping: %w", err)
		}
		if status.State == StateStopped {
			return nil
		}
		if status.State != StateStopPending {
			if _, err := s.Control(CmdStop); err != nil {
				return fmt.Errorf("request stop: %w", err)
			}
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return fmt.Errorf("service did not stop within %s (state=%d)", timeout, status.State)
		}
		sleep := poll
		if sleep > remaining {
			sleep = remaining
		}
		time.Sleep(sleep)
	}
}

// startSCMService starts the service, tolerating "already running".
func startSCMService(s SCMService) error {
	err := s.Start()
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "already running") {
		return nil
	}
	return err
}

func (scmDriver) apply(m *Manager, snap Snapshot, def Definition) error {
	prior, hadPrior := snap.Platform.(*scmPrior)

	if hadPrior {
		svc, err := m.openSCMService()
		if err != nil {
			return err
		}
		defer svc.Close()
		if prior.wasActive {
			if err := stopSCMService(svc, scmStopTimeout, scmStopPoll); err != nil {
				return phaseErr("stop prior service", err)
			}
		}
		staged, err := m.stageExecutable(def)
		if err != nil {
			return err
		}
		if err := svc.UpdateConfig(m.scmConfigFor(def, staged)); err != nil {
			return phaseErr("update service configuration", err)
		}
		if err := configureSCMRecovery(svc); err != nil {
			return phaseErr("update service recovery", err)
		}
		if err := startSCMService(svc); err != nil {
			return phaseErr("start", err)
		}
		return verifySCMReadback(m)
	}

	// Fresh install: nothing was registered, so a failure removes the partial
	// service rather than restoring a prior one.
	staged, err := m.stageExecutable(def)
	if err != nil {
		return err
	}
	b, err := m.scmBoundary()
	if err != nil {
		return err
	}
	args := append([]string(nil), def.Args...)
	svc, err := b.CreateService(m.name, staged, m.scmConfigFor(def, staged), args...)
	if err != nil {
		return phaseErr("create service", err)
	}
	defer svc.Close()
	cleanupFresh := func(phase string, cause error) error {
		if deleteErr := svc.Delete(); deleteErr != nil {
			return updateFailure(phase, cause, fmt.Errorf("remove incomplete service: %w", deleteErr), "")
		}
		return updateFailure(phase, cause, nil, "")
	}
	if err := configureSCMRecovery(svc); err != nil {
		return cleanupFresh("configure service recovery", err)
	}
	if err := startSCMService(svc); err != nil {
		return cleanupFresh("start new service", err)
	}
	return verifySCMReadback(m)
}

// verifySCMReadback reads the service back and requires it to report running.
func verifySCMReadback(m *Manager) error {
	rb, err := (scmDriver{}).readback(m)
	if err != nil {
		return phaseErr("read back", err)
	}
	if rb.State != Running {
		return phaseErr("read back", fmt.Errorf("service %s is %s after start", m.name, rb.State))
	}
	return nil
}

func (scmDriver) restore(m *Manager, snap Snapshot) error {
	prior, ok := snap.Platform.(*scmPrior)
	if !ok {
		return errors.New("snapshot carries no Windows SCM prior state")
	}
	svc, err := m.openSCMService()
	if err != nil {
		return err
	}
	defer svc.Close()
	var errs []error
	configRestored := true
	if err := svc.UpdateConfig(prior.cfg); err != nil {
		configRestored = false
		errs = append(errs, fmt.Errorf("restore configuration: %w", err))
	}
	if err := svc.SetRecoveryActions(prior.actions, prior.resetPeriod); err != nil {
		errs = append(errs, fmt.Errorf("restore recovery actions: %w", err))
	}
	if err := svc.SetRecoveryActionsOnNonCrashFailures(prior.nonCrash); err != nil {
		errs = append(errs, fmt.Errorf("restore recovery failure flag: %w", err))
	}
	if prior.wasActive {
		if !configRestored {
			errs = append(errs, errors.New("old service not restarted because its configuration could not be restored"))
		} else if err := startSCMService(svc); err != nil {
			errs = append(errs, fmt.Errorf("restart old service: %w", err))
		}
	}
	return errors.Join(errs...)
}

func (scmDriver) readback(m *Manager) (Readback, error) {
	svc, err := m.openSCMService()
	if err != nil {
		if errors.Is(err, ErrServiceDoesNotExist) {
			return Readback{State: Unknown}, nil
		}
		return Readback{State: Unknown}, err
	}
	defer svc.Close()
	cfg, cfgErr := svc.Config()
	status, qErr := svc.Query()
	if qErr != nil {
		return Readback{State: Unknown}, qErr
	}
	rb := Readback{PID: int(status.PID)}
	if cfgErr == nil {
		rb.Command = scmConfigExecutable(cfg.BinaryPathName)
	}
	if status.State == StateRunning {
		rb.State = Running
	} else {
		rb.State = Stopped
	}
	return rb, nil
}

// scmConfigExecutable extracts the executable from an SCM BinaryPathName: the
// first quoted token when present, else the first whitespace-delimited field.
func scmConfigExecutable(commandLine string) string {
	commandLine = strings.TrimSpace(commandLine)
	if strings.HasPrefix(commandLine, `"`) {
		if end := strings.Index(commandLine[1:], `"`); end >= 0 {
			return commandLine[1 : end+1]
		}
	}
	if fields := strings.Fields(commandLine); len(fields) != 0 {
		return fields[0]
	}
	return ""
}

func (scmDriver) start(m *Manager) error {
	svc, err := m.openSCMService()
	if err != nil {
		return err
	}
	defer svc.Close()
	return startSCMService(svc)
}

func (scmDriver) stop(m *Manager) error {
	svc, err := m.openSCMService()
	if err != nil {
		return err
	}
	defer svc.Close()
	status, err := svc.Query()
	if err != nil {
		return err
	}
	if status.State == StateStopped {
		return nil
	}
	return stopSCMService(svc, scmStopTimeout, scmStopPoll)
}

func (scmDriver) restart(m *Manager) error {
	if err := (scmDriver{}).stop(m); err != nil {
		return err
	}
	return (scmDriver{}).start(m)
}
