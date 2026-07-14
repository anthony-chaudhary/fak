//go:build windows

package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

const windowsGuardServiceName = "FakGuardControl"

type fakWindowsService struct{ stdout, stderr io.Writer }

func (h fakWindowsService) Execute(_ []string, changes <-chan svc.ChangeRequest, statuses chan<- svc.Status) (bool, uint32) {
	statuses <- svc.Status{State: svc.StartPending}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan int, 1)
	go func() { done <- runServiceLoopContext(ctx, h.stdout, h.stderr, 15*time.Second) }()
	statuses <- svc.Status{State: svc.Running, Accepts: svc.AcceptStop | svc.AcceptShutdown}
	for {
		select {
		case c := <-changes:
			switch c.Cmd {
			case svc.Interrogate:
				statuses <- c.CurrentStatus
			case svc.Stop, svc.Shutdown:
				statuses <- svc.Status{State: svc.StopPending}
				cancel()
			}
		case rc := <-done:
			statuses <- svc.Status{State: svc.Stopped}
			return rc != 0, uint32(rc)
		}
	}
}

func runWindowsServiceDispatcher(stdout, stderr io.Writer) int {
	isService, err := svc.IsWindowsService()
	if err != nil || !isService {
		fmt.Fprintln(stderr, "fak service windows-run must be launched by SCM")
		return 2
	}
	if err := svc.Run(windowsGuardServiceName, fakWindowsService{stdout, stderr}); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func secureWindowsServiceState(path string) error {
	// SYSTEM + Builtin Administrators + LocalService full control, protected from
	// permissive parent inheritance. CI/interactive users do not own this state.
	sd, err := windows.SecurityDescriptorFromString("D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;LS)")
	if err != nil {
		return err
	}
	dacl, _, err := sd.DACL()
	if err != nil {
		return err
	}
	return windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION, nil, nil, dacl, nil)
}
func windowsServiceAction(action string, stdout, stderr io.Writer, dry bool) (serviceResult, int) {
	exe, err := os.Executable()
	if err != nil {
		return serviceResult{}, 1
	}
	state := filepath.Join(os.Getenv("ProgramData"), "fak", "guard-control")
	result := serviceResult{Manager: "windows-scm", Unit: windowsGuardServiceName, Path: exe}
	if dry {
		return result, 0
	}
	m, err := mgr.Connect()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return result, 1
	}
	defer m.Disconnect()
	switch action {
	case "install":
		if err := os.MkdirAll(state, 0o700); err != nil {
			return result, 1
		}
		s, err := m.OpenService(windowsGuardServiceName)
		if err == nil {
			_ = s.Close()
			fmt.Fprintln(stderr, "service already exists")
			return result, 1
		}
		cfg := mgr.Config{StartType: mgr.StartAutomatic, ErrorControl: mgr.ErrorNormal, DisplayName: "fak Guard control plane", Description: "Session-0 Guard sensor, policy, durable state, and recovery control plane", ServiceStartName: "NT AUTHORITY\\LocalService", SidType: 1}
		s, err = m.CreateService(windowsGuardServiceName, exe, cfg, "service", "windows-run")
		if err != nil {
			fmt.Fprintln(stderr, err)
			return result, 1
		}
		defer s.Close()
		if err = s.SetRecoveryActions([]mgr.RecoveryAction{{Type: mgr.ServiceRestart, Delay: 3 * time.Second}, {Type: mgr.ServiceRestart, Delay: 10 * time.Second}, {Type: mgr.ServiceRestart, Delay: 30 * time.Second}}, 86400); err != nil {
			return result, 1
		}
		if err = s.Start(); err != nil {
			return result, 1
		}
	case "status":
		s, err := m.OpenService(windowsGuardServiceName)
		if err != nil {
			return result, 3
		}
		defer s.Close()
		st, err := s.Query()
		if err != nil {
			return result, 1
		}
		result.Active = st.State == svc.Running
	case "uninstall":
		s, err := m.OpenService(windowsGuardServiceName)
		if err != nil {
			return result, 3
		}
		defer s.Close()
		_, _ = s.Control(svc.Stop)
		if err = s.Delete(); err != nil {
			return result, 1
		}
	default:
		return result, 2
	}
	return result, 0
}
