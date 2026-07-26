//go:build windows

package main

import (
	"errors"
	"os"

	"github.com/anthony-chaudhary/fak/internal/scmbridge"
	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

func init() { bridgeLiveReadBack = windowsSCMReadBack }

// windowsSCMReadBack builds the authoritative Observed document from the
// LOCAL Service Control Manager using query-only access rights
// (SC_MANAGER_CONNECT + SERVICE_QUERY_CONFIG|SERVICE_QUERY_STATUS): it runs
// unelevated and can never mutate service state. mgr.Connect is deliberately
// avoided — it demands SC_MANAGER_ALL_ACCESS.
func windowsSCMReadBack(unit string) (scmbridge.Observed, error) {
	m, err := windows.OpenSCManager(nil, nil, windows.SC_MANAGER_CONNECT)
	if err != nil {
		return scmbridge.Observed{}, err
	}
	defer windows.CloseServiceHandle(m)
	u16, err := windows.UTF16PtrFromString(unit)
	if err != nil {
		return scmbridge.Observed{}, err
	}
	h, err := windows.OpenService(m, u16, windows.SERVICE_QUERY_CONFIG|windows.SERVICE_QUERY_STATUS)
	if err != nil {
		if errors.Is(err, windows.ERROR_SERVICE_DOES_NOT_EXIST) {
			return scmbridge.Observed{Present: false, Manager: scmbridge.ManagerSCM, UnitName: unit}, nil
		}
		return scmbridge.Observed{}, err
	}
	s := &mgr.Service{Name: unit, Handle: h}
	defer s.Close()
	cfg, err := s.Config()
	if err != nil {
		return scmbridge.Observed{}, err
	}
	st, err := s.Query()
	if err != nil {
		return scmbridge.Observed{}, err
	}
	got := scmbridge.Observed{
		Present:       true,
		Manager:       scmbridge.ManagerSCM,
		UnitName:      unit,
		Principal:     cfg.ServiceStartName,
		Command:       []string{cfg.BinaryPathName},
		StartOnBoot:   cfg.StartType == mgr.StartAutomatic,
		StartDisabled: cfg.StartType == mgr.StartDisabled,
		Status:        windowsSCMStateName(st.State),
		PID:           int(st.ProcessId),
	}
	exe := scmbridge.ExecutableFromCommandLine(cfg.BinaryPathName, func(p string) bool {
		fi, statErr := os.Stat(p)
		return statErr == nil && !fi.IsDir()
	})
	if sum, sumErr := sha256File(exe); sumErr == nil {
		got.BinarySHA256 = sum
	}
	// Recovery read-back is best-effort under query-only rights: unread stays
	// unread (Reconcile never guesses a zero-valued optional field).
	if ra, raErr := s.RecoveryActions(); raErr == nil {
		for _, a := range ra {
			got.Recovery = append(got.Recovery, scmbridge.RecoveryStep{DelayMS: a.Delay.Milliseconds()})
		}
		if reset, resetErr := s.ResetPeriod(); resetErr == nil {
			got.RecoveryResetSec = int64(reset)
		}
	}
	return got, nil
}

// windowsSCMStateName maps the SCM state enum onto the contract's status
// strings (the exact inputs scmbridge.PhaseFromSCMState reads).
func windowsSCMStateName(s svc.State) string {
	switch s {
	case svc.Stopped:
		return "stopped"
	case svc.StartPending:
		return "start-pending"
	case svc.StopPending:
		return "stop-pending"
	case svc.Running:
		return "running"
	case svc.ContinuePending:
		return "continue-pending"
	case svc.PausePending:
		return "pause-pending"
	case svc.Paused:
		return "paused"
	default:
		return "unknown"
	}
}
