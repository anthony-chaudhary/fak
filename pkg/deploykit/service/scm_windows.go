//go:build windows

package service

import (
	"errors"
	"fmt"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

// defaultSCMBoundary connects to the live service control manager.
func defaultSCMBoundary() (SCMBoundary, error) {
	m, err := mgr.Connect()
	if err != nil {
		return nil, fmt.Errorf("connect to service control manager: %w", err)
	}
	return &scmManagerAdapter{mgr: m}, nil
}

type scmManagerAdapter struct {
	mgr *mgr.Mgr
}

func (a *scmManagerAdapter) Disconnect() error { return a.mgr.Disconnect() }

func (a *scmManagerAdapter) OpenService(name string) (SCMService, error) {
	s, err := a.mgr.OpenService(name)
	if err != nil {
		if errors.Is(err, windows.ERROR_SERVICE_DOES_NOT_EXIST) {
			return nil, fmt.Errorf("%w: %s", ErrServiceDoesNotExist, name)
		}
		return nil, err
	}
	return &scmServiceAdapter{s: s}, nil
}

func (a *scmManagerAdapter) CreateService(name, path string, cfg SCMConfig, args ...string) (SCMService, error) {
	s, err := a.mgr.CreateService(name, path, toMgrConfig(cfg), args...)
	if err != nil {
		return nil, err
	}
	return &scmServiceAdapter{s: s}, nil
}

type scmServiceAdapter struct {
	s *mgr.Service
}

func (v *scmServiceAdapter) Close() error { return v.s.Close() }

func (v *scmServiceAdapter) Config() (SCMConfig, error) {
	c, err := v.s.Config()
	if err != nil {
		return SCMConfig{}, err
	}
	return fromMgrConfig(c), nil
}

func (v *scmServiceAdapter) UpdateConfig(cfg SCMConfig) error {
	return v.s.UpdateConfig(toMgrConfig(cfg))
}

func (v *scmServiceAdapter) RecoveryActions() ([]RecoveryAction, error) {
	a, err := v.s.RecoveryActions()
	if err != nil {
		return nil, err
	}
	return fromMgrActions(a), nil
}

func (v *scmServiceAdapter) ResetPeriod() (uint32, error) { return v.s.ResetPeriod() }

func (v *scmServiceAdapter) SetRecoveryActions(actions []RecoveryAction, resetPeriodSec uint32) error {
	return v.s.SetRecoveryActions(toMgrActions(actions), resetPeriodSec)
}

func (v *scmServiceAdapter) RecoveryActionsOnNonCrashFailures() (bool, error) {
	return v.s.RecoveryActionsOnNonCrashFailures()
}

func (v *scmServiceAdapter) SetRecoveryActionsOnNonCrashFailures(on bool) error {
	return v.s.SetRecoveryActionsOnNonCrashFailures(on)
}

func (v *scmServiceAdapter) Query() (Status, error) {
	st, err := v.s.Query()
	if err != nil {
		return Status{}, err
	}
	return fromSvcStatus(st), nil
}

func (v *scmServiceAdapter) Control(cmd int) (Status, error) {
	st, err := v.s.Control(svc.Cmd(cmd))
	if err != nil {
		return Status{}, err
	}
	return fromSvcStatus(st), nil
}

func (v *scmServiceAdapter) Start(args ...string) error { return v.s.Start(args...) }

func (v *scmServiceAdapter) Delete() error { return v.s.Delete() }

// ── plain-value ↔ x/sys conversions ─────────────────────────────────────────

func toMgrConfig(c SCMConfig) mgr.Config {
	return mgr.Config{
		StartType:        uint32(c.StartType),
		ErrorControl:     uint32(c.ErrorControl),
		BinaryPathName:   c.BinaryPathName,
		DisplayName:      c.DisplayName,
		Description:      c.Description,
		ServiceStartName: c.ServiceStartName,
		SidType:          uint32(c.SidType),
	}
}

func fromMgrConfig(c mgr.Config) SCMConfig {
	return SCMConfig{
		StartType:        int(c.StartType),
		ErrorControl:     int(c.ErrorControl),
		BinaryPathName:   c.BinaryPathName,
		DisplayName:      c.DisplayName,
		Description:      c.Description,
		ServiceStartName: c.ServiceStartName,
		SidType:          int(c.SidType),
	}
}

func fromSvcStatus(st svc.Status) Status {
	return Status{State: int(st.State), PID: st.ProcessId}
}

func fromMgrActions(a []mgr.RecoveryAction) []RecoveryAction {
	out := make([]RecoveryAction, len(a))
	for i, x := range a {
		out[i] = RecoveryAction{Type: x.Type, Delay: x.Delay}
	}
	return out
}

func toMgrActions(a []RecoveryAction) []mgr.RecoveryAction {
	out := make([]mgr.RecoveryAction, len(a))
	for i, x := range a {
		out[i] = mgr.RecoveryAction{Type: x.Type, Delay: x.Delay}
	}
	return out
}
