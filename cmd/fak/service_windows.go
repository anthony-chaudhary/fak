//go:build windows

package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
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
	go func() { done <- runWindowsControlLoop(ctx, h.stdout, h.stderr, 15*time.Second) }()
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

func windowsServiceStateDir() string {
	return filepath.Join(os.Getenv("ProgramData"), "fak", "guard-control")
}

var windowsControlCrashTick = func(stdout, stderr io.Writer, state string) int {
	return runHostCrash(stdout, stderr, []string{"--once", "--since", "5m", "--log", filepath.Join(state, "host-crashes.jsonl"), "--reg-dir", filepath.Join(state, "registry"), "--resurrect"})
}
var windowsControlResumeTick = serviceTick

func runWindowsControlLoop(ctx context.Context, stdout, stderr io.Writer, interval time.Duration) int {
	state := windowsServiceStateDir()
	_ = os.Setenv("FLEET_REG_DIR", filepath.Join(state, "registry"))
	_ = os.Setenv("FAK_HOST_RELAUNCH_DIR", filepath.Join(state, "relaunch"))
	for {
		if rc := windowsControlCrashTick(stdout, stderr, state); rc != 0 {
			return rc
		}
		if rc := windowsControlResumeTick(stdout, stderr); rc != 0 {
			return rc
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return 0
		case <-timer.C:
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

// applyWindowsSDDL parses sddl and installs its DACL on path, replacing whatever the
// parent directory would otherwise have granted. Both the service-state and the shared-dir
// hardeners below differ only in the SDDL they demand, so the parse/extract/apply sequence
// lives here once and neither can drift into a weaker apply.
func applyWindowsSDDL(path, sddl string) error {
	sd, err := windows.SecurityDescriptorFromString(sddl)
	if err != nil {
		return err
	}
	dacl, _, err := sd.DACL()
	if err != nil {
		return err
	}
	return windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION, nil, nil, dacl, nil)
}

func secureWindowsServiceState(path string) error {
	// SYSTEM + Builtin Administrators + LocalService full control, protected from
	// permissive parent inheritance. CI/interactive users do not own this state.
	return applyWindowsSDDL(path, "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;LS)")
}
func secureWindowsSharedDir(path string) error {
	return applyWindowsSDDL(path, "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;LS)(A;OICI;GRGW;;;AU)")
}

func windowsServiceAction(action string, stdout, stderr io.Writer, dry bool) (serviceResult, int) {
	exe, err := os.Executable()
	if err != nil {
		return serviceResult{}, 1
	}
	state := windowsServiceStateDir()
	result := serviceResult{Manager: "windows-scm", Unit: windowsGuardServiceName, Path: exe}
	if dry {
		if action == "witness" {
			result.StateKept = true
		}
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
		if err := secureWindowsServiceState(state); err != nil {
			fmt.Fprintln(stderr, "secure state:", err)
			return result, 1
		}
		for _, shared := range []string{filepath.Join(state, "registry"), filepath.Join(state, "relaunch")} {
			if err := os.MkdirAll(shared, 0o770); err != nil {
				return result, 1
			}
			if err := secureWindowsSharedDir(shared); err != nil {
				fmt.Fprintln(stderr, "secure shared dir:", err)
				return result, 1
			}
		}
		s, err := m.OpenService(windowsGuardServiceName)
		if err == nil {
			_ = s.Close()
			fmt.Fprintln(stderr, "service already exists")
			return result, 1
		}
		cfg := mgr.Config{StartType: mgr.StartAutomatic, ErrorControl: mgr.ErrorNormal, DisplayName: "fak Guard control plane", Description: "Session-0 Guard sensor, policy, durable state, and recovery control plane", ServiceStartName: "NT AUTHORITY\\LocalService", SidType: windows.SERVICE_SID_TYPE_RESTRICTED}
		s, err = m.CreateService(windowsGuardServiceName, exe, cfg, "service", "windows-run")
		if err != nil {
			fmt.Fprintln(stderr, err)
			return result, 1
		}
		defer s.Close()
		if err = s.SetRecoveryActions([]mgr.RecoveryAction{{Type: mgr.ServiceRestart, Delay: 3 * time.Second}, {Type: mgr.ServiceRestart, Delay: 10 * time.Second}, {Type: mgr.ServiceRestart, Delay: 30 * time.Second}}, 86400); err != nil {
			return result, 1
		}
		if err = s.SetRecoveryActionsOnNonCrashFailures(true); err != nil {
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
	case "witness":
		s, err := m.OpenService(windowsGuardServiceName)
		if err != nil {
			return result, 3
		}
		defer s.Close()
		st, err := s.Query()
		if err != nil || st.State != svc.Running || st.ProcessId == 0 {
			fmt.Fprintln(stderr, "service is not running")
			return result, 1
		}
		result.PIDBefore = st.ProcessId
		marker := filepath.Join(state, "scm-witness-state.txt")
		markerValue := strconv.FormatInt(time.Now().UnixNano(), 10)
		if err := os.WriteFile(marker, []byte(markerValue), 0o600); err != nil {
			fmt.Fprintln(stderr, "write witness state:", err)
			return result, 1
		}
		process, err := os.FindProcess(int(st.ProcessId))
		if err != nil {
			fmt.Fprintln(stderr, "open service process:", err)
			return result, 1
		}
		if err := process.Kill(); err != nil {
			fmt.Fprintln(stderr, "kill service process:", err)
			return result, 1
		}
		deadline := time.Now().Add(45 * time.Second)
		for time.Now().Before(deadline) {
			time.Sleep(500 * time.Millisecond)
			st, err = s.Query()
			if err == nil && st.State == svc.Running && st.ProcessId != 0 && st.ProcessId != result.PIDBefore {
				result.PIDAfter = st.ProcessId
				break
			}
		}
		if result.PIDAfter == 0 {
			fmt.Fprintln(stderr, "SCM did not replace the killed service process within 45s")
			return result, 1
		}
		retained, err := os.ReadFile(marker)
		if err != nil || string(retained) != markerValue {
			fmt.Fprintln(stderr, "machine state was not retained across SCM restart")
			return result, 1
		}
		result.StateKept = true
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
