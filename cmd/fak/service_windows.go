//go:build windows

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

const (
	windowsGuardServiceName     = "FakGuardControl"
	windowsServiceBinarySDDL    = "D:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;GRGX;;;LS)"
	windowsServiceBinaryDirSDDL = "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;GRGX;;;LS)"
)

var windowsServiceProgramData = func() string { return os.Getenv("ProgramData") }
var windowsServiceExecutable = os.Executable

type windowsSCMManager interface {
	Disconnect() error
	OpenService(string) (windowsSCMService, error)
	CreateService(string, string, mgr.Config, ...string) (windowsSCMService, error)
}

type windowsSCMService interface {
	Close() error
	Config() (mgr.Config, error)
	UpdateConfig(mgr.Config) error
	RecoveryActions() ([]mgr.RecoveryAction, error)
	ResetPeriod() (uint32, error)
	SetRecoveryActions([]mgr.RecoveryAction, uint32) error
	RecoveryActionsOnNonCrashFailures() (bool, error)
	SetRecoveryActionsOnNonCrashFailures(bool) error
	Query() (svc.Status, error)
	Control(svc.Cmd) (svc.Status, error)
	Start(...string) error
	Delete() error
}

type windowsSCMManagerAdapter struct{ manager *mgr.Mgr }

func (m windowsSCMManagerAdapter) Disconnect() error { return m.manager.Disconnect() }
func (m windowsSCMManagerAdapter) OpenService(name string) (windowsSCMService, error) {
	s, err := m.manager.OpenService(name)
	if err != nil {
		return nil, err
	}
	return s, nil
}
func (m windowsSCMManagerAdapter) CreateService(name, path string, cfg mgr.Config, args ...string) (windowsSCMService, error) {
	s, err := m.manager.CreateService(name, path, cfg, args...)
	if err != nil {
		return nil, err
	}
	return s, nil
}

var windowsConnectSCM = func() (windowsSCMManager, error) {
	m, err := mgr.Connect()
	if err != nil {
		return nil, err
	}
	return windowsSCMManagerAdapter{manager: m}, nil
}

var (
	windowsMakeDirAll                    = os.MkdirAll
	windowsCopyServiceExecutable         = io.Copy
	windowsRenameServiceExecutable       = renameWindowsServiceExecutableNoReplace
	windowsApplyServiceBinarySecurity    = secureWindowsServiceBinary
	windowsApplyServiceBinaryDirSecurity = secureWindowsServiceBinaryDir
	windowsPlanStagedServiceExecutable   = planWindowsServiceExecutable
	windowsStageServiceExecutable        = stageWindowsServiceExecutable
	windowsPrepareServiceState           = prepareWindowsServiceState
	windowsServiceStopTimeout            = 30 * time.Second
	windowsServiceStopPollInterval       = 200 * time.Millisecond
)

func windowsServiceBinaryDir() string {
	return filepath.Join(windowsServiceProgramData(), "fak", "bin")
}

func prepareWindowsServiceBinaryTree(binaryDir string) error {
	if err := windowsMakeDirAll(binaryDir, 0o755); err != nil {
		return fmt.Errorf("create service binary directory: %w", err)
	}
	// Harden the machine-owned directories from root to leaf before publishing a
	// binary. LocalService receives read/execute (directory traversal) only;
	// SYSTEM and Administrators retain the only write-capable grants.
	for _, dir := range []string{filepath.Dir(binaryDir), binaryDir} {
		if err := windowsApplyServiceBinaryDirSecurity(dir); err != nil {
			return fmt.Errorf("secure service binary directory %s: %w", dir, err)
		}
	}
	return nil
}

func renameWindowsServiceExecutableNoReplace(source, target string) error {
	from, err := windows.UTF16PtrFromString(source)
	if err != nil {
		return err
	}
	to, err := windows.UTF16PtrFromString(target)
	if err != nil {
		return err
	}
	// Omitting MOVEFILE_REPLACE_EXISTING is the immutability guarantee: a
	// concurrent publisher can make this move fail, but can never be overwritten.
	return windows.MoveFileEx(from, to, windows.MOVEFILE_WRITE_THROUGH)
}

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
	return filepath.Join(windowsServiceProgramData(), "fak", "guard-control")
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

func secureWindowsServiceBinary(path string) error {
	return applyWindowsSDDL(path, windowsServiceBinarySDDL)
}

func secureWindowsServiceBinaryDir(path string) error {
	return applyWindowsSDDL(path, windowsServiceBinaryDirSDDL)
}

func windowsExecutableSHA256(path string) ([sha256.Size]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return [sha256.Size]byte{}, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return [sha256.Size]byte{}, err
	}
	var digest [sha256.Size]byte
	copy(digest[:], h.Sum(nil))
	return digest, nil
}

func planWindowsServiceExecutable(source string) (string, error) {
	digest, err := windowsExecutableSHA256(source)
	if err != nil {
		return "", err
	}
	return filepath.Join(windowsServiceBinaryDir(), "fak-"+hex.EncodeToString(digest[:])+".exe"), nil
}

func stageWindowsServiceExecutable(source, target string) error {
	if source == target {
		return fmt.Errorf("refusing to stage service executable onto itself: %s", target)
	}
	if err := prepareWindowsServiceBinaryTree(filepath.Dir(target)); err != nil {
		return err
	}
	if targetDigest, err := windowsExecutableSHA256(target); err == nil {
		sourceDigest, sourceErr := windowsExecutableSHA256(source)
		if sourceErr != nil {
			return sourceErr
		}
		if targetDigest != sourceDigest {
			return fmt.Errorf("refusing to overwrite non-matching immutable service executable: %s", target)
		}
		return windowsApplyServiceBinarySecurity(target)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	in, err := os.Open(source)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp, err := os.CreateTemp(filepath.Dir(target), ".fak-service-*.exe")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = windowsCopyServiceExecutable(tmp, in); err == nil {
		err = tmp.Sync()
	}
	if e := tmp.Close(); err == nil {
		err = e
	}
	if err != nil {
		return err
	}
	stagedDigest, err := windowsExecutableSHA256(name)
	if err != nil {
		return err
	}
	contentTarget := filepath.Join(filepath.Dir(target), "fak-"+hex.EncodeToString(stagedDigest[:])+".exe")
	if !strings.EqualFold(filepath.Clean(target), filepath.Clean(contentTarget)) {
		return fmt.Errorf("staged executable changed while copying: content target %s does not match planned target %s", contentTarget, target)
	}
	if err = windowsApplyServiceBinarySecurity(name); err != nil {
		return err
	}
	if err = windowsRenameServiceExecutable(name, target); err != nil {
		return err
	}
	return nil
}
func windowsServiceConfig(binary string) mgr.Config {
	return mgr.Config{StartType: mgr.StartAutomatic, ErrorControl: mgr.ErrorNormal, BinaryPathName: fmt.Sprintf("\"%s\" service windows-run", binary), DisplayName: "fak Guard control plane", Description: "Session-0 Guard sensor, policy, durable state, and recovery control plane", ServiceStartName: "NT AUTHORITY\\LocalService", SidType: windows.SERVICE_SID_TYPE_RESTRICTED}
}

func windowsServiceConfigExecutable(commandLine string) string {
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

func secureWindowsServiceState(path string) error {
	// SYSTEM + Builtin Administrators + LocalService full control, protected from
	// permissive parent inheritance. CI/interactive users do not own this state.
	return applyWindowsSDDL(path, "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;LS)")
}
func secureWindowsSharedDir(path string) error {
	return applyWindowsSDDL(path, "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;LS)(A;OICI;GRGW;;;AU)")
}

func prepareWindowsServiceState(state string) error {
	if err := windowsMakeDirAll(state, 0o700); err != nil {
		return fmt.Errorf("create state directory: %w", err)
	}
	if err := secureWindowsServiceState(state); err != nil {
		return fmt.Errorf("secure state directory: %w", err)
	}
	for _, shared := range []string{filepath.Join(state, "registry"), filepath.Join(state, "relaunch")} {
		if err := windowsMakeDirAll(shared, 0o770); err != nil {
			return fmt.Errorf("create shared directory %s: %w", shared, err)
		}
		if err := secureWindowsSharedDir(shared); err != nil {
			return fmt.Errorf("secure shared directory %s: %w", shared, err)
		}
	}
	return nil
}

type windowsServiceSnapshot struct {
	config          mgr.Config
	recoveryActions []mgr.RecoveryAction
	resetPeriod     uint32
	nonCrash        bool
	wasActive       bool
	status          svc.Status
}

func snapshotWindowsService(s windowsSCMService) (windowsServiceSnapshot, error) {
	cfg, err := s.Config()
	if err != nil {
		return windowsServiceSnapshot{}, fmt.Errorf("query configuration: %w", err)
	}
	actions, err := s.RecoveryActions()
	if err != nil {
		return windowsServiceSnapshot{}, fmt.Errorf("query recovery actions: %w", err)
	}
	resetPeriod, err := s.ResetPeriod()
	if err != nil {
		return windowsServiceSnapshot{}, fmt.Errorf("query recovery reset period: %w", err)
	}
	nonCrash, err := s.RecoveryActionsOnNonCrashFailures()
	if err != nil {
		return windowsServiceSnapshot{}, fmt.Errorf("query recovery failure flag: %w", err)
	}
	status, err := s.Query()
	if err != nil {
		return windowsServiceSnapshot{}, fmt.Errorf("query state: %w", err)
	}
	return windowsServiceSnapshot{
		config:          cfg,
		recoveryActions: append([]mgr.RecoveryAction(nil), actions...),
		resetPeriod:     resetPeriod,
		nonCrash:        nonCrash,
		wasActive:       status.State != svc.Stopped && status.State != svc.StopPending,
		status:          status,
	}, nil
}

func waitForWindowsServiceStopped(s windowsSCMService, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		status, err := s.Query()
		if err != nil {
			return fmt.Errorf("query while stopping: %w", err)
		}
		if status.State == svc.Stopped {
			return nil
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return fmt.Errorf("service did not stop within %s (state=%d)", timeout, status.State)
		}
		sleep := windowsServiceStopPollInterval
		if sleep > remaining {
			sleep = remaining
		}
		time.Sleep(sleep)
	}
}

func stopWindowsService(s windowsSCMService, status svc.Status) error {
	if status.State == svc.Stopped {
		return nil
	}
	if status.State != svc.StopPending {
		if _, err := s.Control(svc.Stop); err != nil {
			return fmt.Errorf("request stop: %w", err)
		}
	}
	return waitForWindowsServiceStopped(s, windowsServiceStopTimeout)
}

var windowsServiceRecoveryActions = []mgr.RecoveryAction{
	{Type: mgr.ServiceRestart, Delay: 3 * time.Second},
	{Type: mgr.ServiceRestart, Delay: 10 * time.Second},
	{Type: mgr.ServiceRestart, Delay: 30 * time.Second},
}

func configureWindowsServiceRecovery(s windowsSCMService) error {
	if err := s.SetRecoveryActions(windowsServiceRecoveryActions, 86400); err != nil {
		return err
	}
	return s.SetRecoveryActionsOnNonCrashFailures(true)
}

func restoreWindowsService(s windowsSCMService, snapshot windowsServiceSnapshot, configurationMayHaveChanged bool) error {
	var restoreErrs []error
	configurationRestored := true
	if configurationMayHaveChanged {
		if err := s.UpdateConfig(snapshot.config); err != nil {
			configurationRestored = false
			restoreErrs = append(restoreErrs, fmt.Errorf("restore configuration: %w", err))
		}
		if err := s.SetRecoveryActions(snapshot.recoveryActions, snapshot.resetPeriod); err != nil {
			restoreErrs = append(restoreErrs, fmt.Errorf("restore recovery actions: %w", err))
		}
		if err := s.SetRecoveryActionsOnNonCrashFailures(snapshot.nonCrash); err != nil {
			restoreErrs = append(restoreErrs, fmt.Errorf("restore recovery failure flag: %w", err))
		}
	}
	if snapshot.wasActive {
		if !configurationRestored {
			restoreErrs = append(restoreErrs, errors.New("old service not restarted because its configuration could not be restored"))
		} else if err := s.Start(); err != nil && !strings.Contains(strings.ToLower(err.Error()), "already running") {
			restoreErrs = append(restoreErrs, fmt.Errorf("restart old service: %w", err))
		}
	}
	return errors.Join(restoreErrs...)
}

func windowsInstallFailure(phase string, cause, rollbackErr error, oldImagePath, stagedPath string) error {
	if rollbackErr != nil {
		return fmt.Errorf("%s: %w; rollback failed: %v; old ImagePath [%s]; staged executable [%s] retained because SCM may reference it", phase, cause, rollbackErr, oldImagePath, stagedPath)
	}
	return fmt.Errorf("%s: %w; previous service configuration preserved at [%s]", phase, cause, oldImagePath)
}

func installWindowsService(m windowsSCMManager, sourceExe, state string) (string, error) {
	service, openErr := m.OpenService(windowsGuardServiceName)
	var snapshot windowsServiceSnapshot
	existing := openErr == nil
	if existing {
		defer service.Close()
		var err error
		snapshot, err = snapshotWindowsService(service)
		if err != nil {
			return "", fmt.Errorf("inspect existing service: %w", err)
		}
		if err := stopWindowsService(service, snapshot.status); err != nil {
			return "", windowsInstallFailure("stop existing service", err, restoreWindowsService(service, snapshot, false), snapshot.config.BinaryPathName, "")
		}
	} else if !errors.Is(openErr, windows.ERROR_SERVICE_DOES_NOT_EXIST) {
		return "", fmt.Errorf("open existing service: %w", openErr)
	}

	var (
		target string
		err    error
	)
	failExisting := func(phase string, cause error, configurationMayHaveChanged bool) (string, error) {
		if !existing {
			return "", fmt.Errorf("%s: %w", phase, cause)
		}
		return "", windowsInstallFailure(phase, cause, restoreWindowsService(service, snapshot, configurationMayHaveChanged), snapshot.config.BinaryPathName, target)
	}

	if err := windowsPrepareServiceState(state); err != nil {
		return failExisting("prepare service state", err, false)
	}
	target, err = windowsPlanStagedServiceExecutable(sourceExe)
	if err != nil {
		return failExisting("plan service executable", err, false)
	}
	if err := windowsStageServiceExecutable(sourceExe, target); err != nil {
		return failExisting("stage service executable", err, false)
	}
	cfg := windowsServiceConfig(target)
	if existing {
		if err := service.UpdateConfig(cfg); err != nil {
			return failExisting("update service configuration", err, true)
		}
		if err := configureWindowsServiceRecovery(service); err != nil {
			return failExisting("update service recovery", err, true)
		}
		if err := service.Start(); err != nil && !strings.Contains(strings.ToLower(err.Error()), "already running") {
			return failExisting("start upgraded service", err, true)
		}
		return target, nil
	}

	service, err = m.CreateService(windowsGuardServiceName, target, cfg, "service", "windows-run")
	if err != nil {
		return "", fmt.Errorf("create service: %w", err)
	}
	defer service.Close()
	cleanupFresh := func(phase string, cause error) (string, error) {
		if deleteErr := service.Delete(); deleteErr != nil {
			return "", fmt.Errorf("%s: %w; remove incomplete service: %v", phase, cause, deleteErr)
		}
		return "", fmt.Errorf("%s: %w; incomplete service removed", phase, cause)
	}
	if err := configureWindowsServiceRecovery(service); err != nil {
		return cleanupFresh("configure service recovery", err)
	}
	if err := service.Start(); err != nil && !strings.Contains(strings.ToLower(err.Error()), "already running") {
		return cleanupFresh("start new service", err)
	}
	return target, nil
}

func windowsServiceAction(action string, stdout, stderr io.Writer, dry bool) (serviceResult, int) {
	sourceExe, err := windowsServiceExecutable()
	if err != nil {
		return serviceResult{}, 1
	}
	state := windowsServiceStateDir()
	result := serviceResult{Manager: "windows-scm", Unit: windowsGuardServiceName}
	if action == "install" {
		if selectedPath, planErr := windowsPlanStagedServiceExecutable(sourceExe); planErr == nil {
			result.Path = selectedPath
		}
	}
	if dry {
		if action == "witness" {
			result.StateKept = true
		}
		return result, 0
	}
	m, err := windowsConnectSCM()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return result, 1
	}
	defer m.Disconnect()
	switch action {
	case "install":
		result.Path, err = installWindowsService(m, sourceExe, state)
		if err != nil {
			fmt.Fprintln(stderr, "install service:", err)
			return result, 1
		}
	case "status":
		s, err := m.OpenService(windowsGuardServiceName)
		if err != nil {
			return result, 3
		}
		defer s.Close()
		if cfg, configErr := s.Config(); configErr == nil {
			result.Path = windowsServiceConfigExecutable(cfg.BinaryPathName)
		}
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
		if cfg, configErr := s.Config(); configErr == nil {
			result.Path = windowsServiceConfigExecutable(cfg.BinaryPathName)
		}
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
		if cfg, configErr := s.Config(); configErr == nil {
			result.Path = windowsServiceConfigExecutable(cfg.BinaryPathName)
		}
		_, _ = s.Control(svc.Stop)
		if err = s.Delete(); err != nil {
			return result, 1
		}
	default:
		return result, 2
	}
	return result, 0
}
