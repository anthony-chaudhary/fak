//go:build windows

package main

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

type fakeWindowsSCMManager struct {
	existing   *fakeWindowsSCMService
	created    *fakeWindowsSCMService
	openErr    error
	createErr  error
	createPath string
	createCfg  mgr.Config
	events     *[]string
}

func (m *fakeWindowsSCMManager) event(event string) {
	if m.events != nil {
		*m.events = append(*m.events, event)
	}
}
func (m *fakeWindowsSCMManager) Disconnect() error { m.event("disconnect"); return nil }
func (m *fakeWindowsSCMManager) OpenService(string) (windowsSCMService, error) {
	m.event("open")
	if m.openErr != nil {
		return nil, m.openErr
	}
	return m.existing, nil
}
func (m *fakeWindowsSCMManager) CreateService(_ string, path string, cfg mgr.Config, _ ...string) (windowsSCMService, error) {
	m.event("create")
	m.createPath, m.createCfg = path, cfg
	if m.createErr != nil {
		return nil, m.createErr
	}
	if m.created == nil {
		m.created = &fakeWindowsSCMService{events: m.events}
	}
	m.created.config = cfg
	return m.created, nil
}

type fakeWindowsSCMService struct {
	config          mgr.Config
	recoveryActions []mgr.RecoveryAction
	resetPeriod     uint32
	nonCrash        bool
	status          svc.Status
	queryStates     []svc.Status
	updateErrs      []error
	recoveryErrs    []error
	nonCrashErrs    []error
	startErrs       []error
	updateHistory   []mgr.Config
	startCalls      int
	deleteCalls     int
	events          *[]string
}

func (s *fakeWindowsSCMService) event(event string) {
	if s.events != nil {
		*s.events = append(*s.events, event)
	}
}
func (s *fakeWindowsSCMService) Close() error { s.event("close"); return nil }
func (s *fakeWindowsSCMService) Config() (mgr.Config, error) {
	s.event("config")
	return s.config, nil
}
func (s *fakeWindowsSCMService) UpdateConfig(cfg mgr.Config) error {
	s.event("update")
	s.updateHistory = append(s.updateHistory, cfg)
	s.config = cfg // ChangeServiceConfig can succeed before a later config2 call fails.
	return popWindowsTestError(&s.updateErrs)
}
func (s *fakeWindowsSCMService) RecoveryActions() ([]mgr.RecoveryAction, error) {
	s.event("recovery-query")
	return append([]mgr.RecoveryAction(nil), s.recoveryActions...), nil
}
func (s *fakeWindowsSCMService) ResetPeriod() (uint32, error) {
	s.event("reset-query")
	return s.resetPeriod, nil
}
func (s *fakeWindowsSCMService) SetRecoveryActions(actions []mgr.RecoveryAction, reset uint32) error {
	s.event("recovery-set")
	s.recoveryActions = append([]mgr.RecoveryAction(nil), actions...)
	s.resetPeriod = reset
	return popWindowsTestError(&s.recoveryErrs)
}
func (s *fakeWindowsSCMService) RecoveryActionsOnNonCrashFailures() (bool, error) {
	s.event("flag-query")
	return s.nonCrash, nil
}
func (s *fakeWindowsSCMService) SetRecoveryActionsOnNonCrashFailures(value bool) error {
	s.event("flag-set")
	s.nonCrash = value
	return popWindowsTestError(&s.nonCrashErrs)
}
func (s *fakeWindowsSCMService) Query() (svc.Status, error) {
	s.event("query")
	if len(s.queryStates) != 0 {
		s.status = s.queryStates[0]
		s.queryStates = s.queryStates[1:]
	}
	return s.status, nil
}
func (s *fakeWindowsSCMService) Control(cmd svc.Cmd) (svc.Status, error) {
	s.event("control")
	if cmd == svc.Stop {
		s.status.State = svc.StopPending
	}
	return s.status, nil
}
func (s *fakeWindowsSCMService) Start(...string) error {
	s.event("start")
	s.startCalls++
	err := popWindowsTestError(&s.startErrs)
	if err == nil {
		s.status.State = svc.Running
	}
	return err
}
func (s *fakeWindowsSCMService) Delete() error {
	s.event("delete")
	s.deleteCalls++
	return nil
}

func popWindowsTestError(errs *[]error) error {
	if len(*errs) == 0 {
		return nil
	}
	err := (*errs)[0]
	*errs = (*errs)[1:]
	return err
}

func stubWindowsInstallFilesystem(t *testing.T, events *[]string, target string) {
	t.Helper()
	oldPrepare := windowsPrepareServiceState
	oldPlan := windowsPlanStagedServiceExecutable
	oldStage := windowsStageServiceExecutable
	windowsPrepareServiceState = func(string) error {
		*events = append(*events, "prepare")
		return nil
	}
	windowsPlanStagedServiceExecutable = func(string) (string, error) {
		*events = append(*events, "plan")
		return target, nil
	}
	windowsStageServiceExecutable = func(_, got string) error {
		*events = append(*events, "stage")
		if got != target {
			t.Fatalf("stage target=%q want=%q", got, target)
		}
		return nil
	}
	t.Cleanup(func() {
		windowsPrepareServiceState = oldPrepare
		windowsPlanStagedServiceExecutable = oldPlan
		windowsStageServiceExecutable = oldStage
	})
}

func runningWindowsService(events *[]string) (*fakeWindowsSCMService, mgr.Config, []mgr.RecoveryAction) {
	oldCfg := windowsServiceConfig(`C:\ProgramData\fak\bin\fak-old.exe`)
	oldRecovery := []mgr.RecoveryAction{{Type: mgr.ServiceRestart, Delay: time.Minute}}
	return &fakeWindowsSCMService{
		config:          oldCfg,
		recoveryActions: append([]mgr.RecoveryAction(nil), oldRecovery...),
		resetPeriod:     42,
		status:          svc.Status{State: svc.Running},
		queryStates:     []svc.Status{{State: svc.Running}, {State: svc.Stopped}},
		events:          events,
	}, oldCfg, oldRecovery
}

func TestWindowsServiceHandlerReportsRunningAndStops(t *testing.T) {
	oldCrash, oldResume := windowsControlCrashTick, windowsControlResumeTick
	windowsControlCrashTick = func(io.Writer, io.Writer, string) int { return 0 }
	windowsControlResumeTick = func(io.Writer, io.Writer) int { return 0 }
	t.Cleanup(func() { windowsControlCrashTick, windowsControlResumeTick = oldCrash, oldResume })
	changes := make(chan svc.ChangeRequest, 1)
	statuses := make(chan svc.Status, 8)
	done := make(chan struct{})
	go func() { fakWindowsService{io.Discard, io.Discard}.Execute(nil, changes, statuses); close(done) }()
	seenRunning := false
	deadline := time.After(time.Second)
	for !seenRunning {
		select {
		case st := <-statuses:
			seenRunning = st.State == svc.Running
		case <-deadline:
			t.Fatal("no Running status")
		}
	}
	changes <- svc.ChangeRequest{Cmd: svc.Stop}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("service did not stop")
	}
}
func TestWindowsServiceDryRunNamesLeastPrivilegeSCMUnit(t *testing.T) {
	r, rc := windowsServiceAction("install", io.Discard, io.Discard, true)
	if rc != 0 || r.Manager != "windows-scm" || r.Unit != windowsGuardServiceName {
		t.Fatalf("r=%+v rc=%d", r, rc)
	}
}

func TestWindowsServiceWitnessDryRunIsNonDestructive(t *testing.T) {
	r, rc := windowsServiceAction("witness", io.Discard, io.Discard, true)
	if rc != 0 || r.Manager != "windows-scm" || r.Unit != windowsGuardServiceName || !r.StateKept || r.PIDBefore != 0 || r.PIDAfter != 0 {
		t.Fatalf("r=%+v rc=%d", r, rc)
	}
}

func TestWindowsControlCrashTickUsesMachineInteractiveRegistry(t *testing.T) {
	state := t.TempDir()
	registry := filepath.Join(state, "registry")
	if rc := windowsControlCrashTick(io.Discard, io.Discard, state); rc != 0 {
		t.Fatalf("tick rc=%d", rc)
	}
	if _, err := os.Stat(filepath.Join(registry, hostresurrect.CohortFileName)); err != nil {
		t.Fatalf("cohort not written to interactive registry: %v", err)
	}
}

func TestWindowsVersionedServiceExecutableUsesProgramData(t *testing.T) {
	old := windowsServiceProgramData
	programData := t.TempDir()
	windowsServiceProgramData = func() string { return programData }
	t.Cleanup(func() { windowsServiceProgramData = old })
	source := filepath.Join(t.TempDir(), "source.exe")
	if err := os.WriteFile(source, []byte("versioned fak executable"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := planWindowsServiceExecutable(source)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Dir(got) != windowsServiceBinaryDir() || filepath.Base(got) == "fak.exe" || !strings.HasPrefix(filepath.Base(got), "fak-") || !strings.HasSuffix(got, ".exe") {
		t.Fatalf("versioned path=%q binary-dir=%q", got, windowsServiceBinaryDir())
	}
	cfg := windowsServiceConfig(got)
	if cfg.StartType != mgr.StartAutomatic || cfg.ServiceStartName != `NT AUTHORITY\LocalService` || !strings.Contains(cfg.BinaryPathName, got) {
		t.Fatalf("config=%+v", cfg)
	}
}

func TestInstallWindowsServiceFreshInstallIsLeastPrivilegeAutoStart(t *testing.T) {
	events := []string{}
	target := `C:\ProgramData\fak\bin\fak-new.exe`
	stubWindowsInstallFilesystem(t, &events, target)
	created := &fakeWindowsSCMService{events: &events, status: svc.Status{State: svc.Stopped}}
	manager := &fakeWindowsSCMManager{openErr: windows.ERROR_SERVICE_DOES_NOT_EXIST, created: created, events: &events}

	got, err := installWindowsService(manager, `C:\incoming\fak.exe`, `C:\ProgramData\fak\guard-control`)
	if err != nil {
		t.Fatal(err)
	}
	if got != target || manager.createPath != target {
		t.Fatalf("path=%q createPath=%q want=%q", got, manager.createPath, target)
	}
	if manager.createCfg.StartType != mgr.StartAutomatic || manager.createCfg.ServiceStartName != `NT AUTHORITY\LocalService` || manager.createCfg.SidType != windows.SERVICE_SID_TYPE_RESTRICTED {
		t.Fatalf("create config=%+v", manager.createCfg)
	}
	if created.startCalls != 1 || created.deleteCalls != 0 || !reflect.DeepEqual(created.recoveryActions, windowsServiceRecoveryActions) || !created.nonCrash {
		t.Fatalf("created=%+v", created)
	}
	wantEvents := []string{"open", "prepare", "plan", "stage", "create", "recovery-set", "flag-set", "start", "close"}
	if !reflect.DeepEqual(events, wantEvents) {
		t.Fatalf("events=%v want=%v", events, wantEvents)
	}
}

func TestInstallWindowsServiceStopsRunningServiceBeforeUpgrade(t *testing.T) {
	events := []string{}
	target := `C:\ProgramData\fak\bin\fak-new.exe`
	stubWindowsInstallFilesystem(t, &events, target)
	service, _, _ := runningWindowsService(&events)
	manager := &fakeWindowsSCMManager{existing: service, events: &events}

	got, err := installWindowsService(manager, `C:\incoming\fak.exe`, `C:\ProgramData\fak\guard-control`)
	if err != nil {
		t.Fatal(err)
	}
	if got != target || service.startCalls != 1 || len(service.updateHistory) != 1 {
		t.Fatalf("path=%q starts=%d updates=%d", got, service.startCalls, len(service.updateHistory))
	}
	pathUpdated := strings.Contains(service.config.BinaryPathName, target)
	recoveryUpdated := reflect.DeepEqual(service.recoveryActions, windowsServiceRecoveryActions)
	if !pathUpdated || !recoveryUpdated || !service.nonCrash {
		t.Fatalf("service config=%+v pathUpdated=%v recovery=%+v recoveryUpdated=%v nonCrash=%v", service.config, pathUpdated, service.recoveryActions, recoveryUpdated, service.nonCrash)
	}
	wantEvents := []string{"open", "config", "recovery-query", "reset-query", "flag-query", "query", "control", "query", "prepare", "plan", "stage", "update", "recovery-set", "flag-set", "start", "close"}
	if !reflect.DeepEqual(events, wantEvents) {
		t.Fatalf("events=%v want=%v", events, wantEvents)
	}
}

func TestInstallWindowsServiceStopWaitIsBoundedAndPrecedesStaging(t *testing.T) {
	events := []string{}
	stubWindowsInstallFilesystem(t, &events, `C:\ProgramData\fak\bin\fak-new.exe`)
	service, oldCfg, _ := runningWindowsService(&events)
	service.queryStates = []svc.Status{{State: svc.Running}}
	manager := &fakeWindowsSCMManager{existing: service, events: &events}
	oldTimeout := windowsServiceStopTimeout
	windowsServiceStopTimeout = 0
	t.Cleanup(func() { windowsServiceStopTimeout = oldTimeout })

	_, err := installWindowsService(manager, `C:\incoming\fak.exe`, `C:\ProgramData\fak\guard-control`)
	if err == nil || !strings.Contains(err.Error(), "did not stop within") {
		t.Fatalf("err=%v", err)
	}
	if strings.Contains(strings.Join(events, ","), "stage") || service.startCalls != 1 || len(service.updateHistory) != 0 || !reflect.DeepEqual(service.config, oldCfg) {
		t.Fatalf("events=%v starts=%d updates=%v config=%+v", events, service.startCalls, service.updateHistory, service.config)
	}
}

func TestInstallWindowsServiceStageFailuresRestartOldService(t *testing.T) {
	for _, failure := range []string{"copy", "ACL"} {
		t.Run(failure, func(t *testing.T) {
			events := []string{}
			target := `C:\ProgramData\fak\bin\fak-new.exe`
			stubWindowsInstallFilesystem(t, &events, target)
			windowsStageServiceExecutable = func(_, _ string) error {
				events = append(events, "stage-"+failure+"-failure")
				return errors.New(failure + " failed")
			}
			service, oldCfg, _ := runningWindowsService(&events)
			manager := &fakeWindowsSCMManager{existing: service, events: &events}

			_, err := installWindowsService(manager, `C:\incoming\fak.exe`, `C:\ProgramData\fak\guard-control`)
			if err == nil || !strings.Contains(err.Error(), "stage service executable") || !strings.Contains(err.Error(), "previous service configuration preserved") {
				t.Fatalf("err=%v", err)
			}
			if service.startCalls != 1 || len(service.updateHistory) != 0 || !reflect.DeepEqual(service.config, oldCfg) {
				t.Fatalf("starts=%d updates=%d config=%+v", service.startCalls, len(service.updateHistory), service.config)
			}
		})
	}
}

func TestStageWindowsServiceExecutableFailureLeavesConfiguredTargetIntact(t *testing.T) {
	for _, test := range []struct {
		name string
		fail func()
	}{
		{name: "copy", fail: func() {
			windowsCopyServiceExecutable = func(io.Writer, io.Reader) (int64, error) { return 0, errors.New("copy failed") }
		}},
		{name: "ACL", fail: func() {
			windowsApplyServiceBinarySecurity = func(string) error { return errors.New("ACL failed") }
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			oldCopy := windowsCopyServiceExecutable
			oldBinarySecurity := windowsApplyServiceBinarySecurity
			oldDirSecurity := windowsApplyServiceBinaryDirSecurity
			windowsApplyServiceBinaryDirSecurity = func(string) error { return nil }
			test.fail()
			t.Cleanup(func() {
				windowsCopyServiceExecutable = oldCopy
				windowsApplyServiceBinarySecurity = oldBinarySecurity
				windowsApplyServiceBinaryDirSecurity = oldDirSecurity
			})
			dir := t.TempDir()
			source := filepath.Join(dir, "incoming.exe")
			configured := filepath.Join(dir, "fak-old.exe")
			if err := os.WriteFile(source, []byte("new executable"), 0o600); err != nil {
				t.Fatal(err)
			}
			planned, err := planWindowsServiceExecutable(source)
			if err != nil {
				t.Fatal(err)
			}
			target := filepath.Join(dir, filepath.Base(planned))
			if err := os.WriteFile(configured, []byte("old executable"), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := stageWindowsServiceExecutable(source, target); err == nil {
				t.Fatal("stage unexpectedly succeeded")
			}
			oldBytes, err := os.ReadFile(configured)
			if err != nil || string(oldBytes) != "old executable" {
				t.Fatalf("configured target missing or changed: bytes=%q err=%v", oldBytes, err)
			}
			if _, err := os.Stat(target); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("failed stage published target: %v", err)
			}
		})
	}
}

func TestRenameWindowsServiceExecutableNeverReplacesPublishedTarget(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "source.exe")
	target := filepath.Join(dir, "target.exe")
	if err := os.WriteFile(source, []byte("new"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("published"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := renameWindowsServiceExecutableNoReplace(source, target); err == nil {
		t.Fatal("no-replace move unexpectedly overwrote the published target")
	}
	got, err := os.ReadFile(target)
	if err != nil || string(got) != "published" {
		t.Fatalf("published target changed: bytes=%q err=%v", got, err)
	}
	if _, err := os.Stat(source); err != nil {
		t.Fatalf("failed no-replace move consumed source: %v", err)
	}
}

func TestInstallWindowsServiceFailuresRestoreOldConfiguration(t *testing.T) {
	for _, test := range []struct {
		name       string
		configure  func(*fakeWindowsSCMService)
		wantStarts int
	}{
		{name: "update", configure: func(s *fakeWindowsSCMService) {
			s.updateErrs = []error{errors.New("update failed"), nil}
		}, wantStarts: 1},
		{name: "recovery", configure: func(s *fakeWindowsSCMService) {
			s.recoveryErrs = []error{errors.New("recovery failed"), nil}
		}, wantStarts: 1},
		{name: "start", configure: func(s *fakeWindowsSCMService) {
			s.startErrs = []error{errors.New("start failed"), nil}
		}, wantStarts: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			events := []string{}
			stubWindowsInstallFilesystem(t, &events, `C:\ProgramData\fak\bin\fak-new.exe`)
			service, oldCfg, oldRecovery := runningWindowsService(&events)
			test.configure(service)
			manager := &fakeWindowsSCMManager{existing: service, events: &events}

			_, err := installWindowsService(manager, `C:\incoming\fak.exe`, `C:\ProgramData\fak\guard-control`)
			if err == nil || !strings.Contains(err.Error(), "previous service configuration preserved") {
				t.Fatalf("err=%v", err)
			}
			if service.startCalls != test.wantStarts || !reflect.DeepEqual(service.config, oldCfg) || !reflect.DeepEqual(service.recoveryActions, oldRecovery) || service.resetPeriod != 42 || service.nonCrash {
				t.Fatalf("starts=%d config=%+v recovery=%+v reset=%d nonCrash=%v", service.startCalls, service.config, service.recoveryActions, service.resetPeriod, service.nonCrash)
			}
			if len(service.updateHistory) != 2 {
				t.Fatalf("update history=%+v", service.updateHistory)
			}
		})
	}
}

func TestInstallWindowsServiceStartFailureKeepsOldExecutable(t *testing.T) {
	programData := t.TempDir()
	source := filepath.Join(t.TempDir(), "incoming.exe")
	oldTarget := filepath.Join(programData, "fak", "bin", "fak-old.exe")
	if err := os.MkdirAll(filepath.Dir(oldTarget), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("new executable"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(oldTarget, []byte("old executable"), 0o600); err != nil {
		t.Fatal(err)
	}
	oldProgramData := windowsServiceProgramData
	oldPrepare := windowsPrepareServiceState
	oldBinarySecurity := windowsApplyServiceBinarySecurity
	oldDirSecurity := windowsApplyServiceBinaryDirSecurity
	windowsServiceProgramData = func() string { return programData }
	windowsPrepareServiceState = func(string) error { return nil }
	windowsApplyServiceBinarySecurity = func(string) error { return nil }
	windowsApplyServiceBinaryDirSecurity = func(string) error { return nil }
	t.Cleanup(func() {
		windowsServiceProgramData = oldProgramData
		windowsPrepareServiceState = oldPrepare
		windowsApplyServiceBinarySecurity = oldBinarySecurity
		windowsApplyServiceBinaryDirSecurity = oldDirSecurity
	})
	events := []string{}
	service, _, _ := runningWindowsService(&events)
	service.config = windowsServiceConfig(oldTarget)
	service.startErrs = []error{errors.New("start failed"), nil}
	manager := &fakeWindowsSCMManager{existing: service, events: &events}

	newTarget, err := planWindowsServiceExecutable(source)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := installWindowsService(manager, source, windowsServiceStateDir()); err == nil {
		t.Fatal("install unexpectedly succeeded")
	}
	oldBytes, err := os.ReadFile(oldTarget)
	if err != nil || string(oldBytes) != "old executable" {
		t.Fatalf("old executable missing or changed: bytes=%q err=%v", oldBytes, err)
	}
	newBytes, err := os.ReadFile(newTarget)
	if err != nil || string(newBytes) != "new executable" {
		t.Fatalf("staged executable not retained for safe rollback: bytes=%q err=%v", newBytes, err)
	}
	if !strings.Contains(service.config.BinaryPathName, oldTarget) {
		t.Fatalf("restored config=%+v", service.config)
	}
}

func TestInstallWindowsServiceRollbackFailureReportsRetainedImagePaths(t *testing.T) {
	events := []string{}
	target := `C:\ProgramData\fak\bin\fak-new.exe`
	stubWindowsInstallFilesystem(t, &events, target)
	service, oldCfg, _ := runningWindowsService(&events)
	service.recoveryErrs = []error{errors.New("recovery failed")}
	service.updateErrs = []error{nil, errors.New("rollback config failed")}
	manager := &fakeWindowsSCMManager{existing: service, events: &events}

	_, err := installWindowsService(manager, `C:\incoming\fak.exe`, `C:\ProgramData\fak\guard-control`)
	if err == nil || !strings.Contains(err.Error(), "rollback failed") || !strings.Contains(err.Error(), oldCfg.BinaryPathName) || !strings.Contains(err.Error(), target) || !strings.Contains(err.Error(), "retained because SCM may reference it") {
		t.Fatalf("err=%v", err)
	}
	if service.startCalls != 0 {
		t.Fatalf("unsafe restart attempted after rollback failure: starts=%d", service.startCalls)
	}
}

func TestInstallWindowsServiceFreshStartFailureRemovesIncompleteService(t *testing.T) {
	events := []string{}
	stubWindowsInstallFilesystem(t, &events, `C:\ProgramData\fak\bin\fak-new.exe`)
	created := &fakeWindowsSCMService{events: &events, startErrs: []error{errors.New("start failed")}}
	manager := &fakeWindowsSCMManager{openErr: windows.ERROR_SERVICE_DOES_NOT_EXIST, created: created, events: &events}

	_, err := installWindowsService(manager, `C:\incoming\fak.exe`, `C:\ProgramData\fak\guard-control`)
	if err == nil || !strings.Contains(err.Error(), "incomplete service removed") || created.deleteCalls != 1 {
		t.Fatalf("err=%v deletes=%d", err, created.deleteCalls)
	}
}

func TestStageWindowsServiceExecutableCapturedACLPlanIsOrderedAndIdempotent(t *testing.T) {
	oldMakeDir := windowsMakeDirAll
	oldDirSecurity := windowsApplyServiceBinaryDirSecurity
	oldBinarySecurity := windowsApplyServiceBinarySecurity
	oldRename := windowsRenameServiceExecutable
	t.Cleanup(func() {
		windowsMakeDirAll = oldMakeDir
		windowsApplyServiceBinaryDirSecurity = oldDirSecurity
		windowsApplyServiceBinarySecurity = oldBinarySecurity
		windowsRenameServiceExecutable = oldRename
	})

	dir := t.TempDir()
	profile := filepath.Join(dir, "Users", "operator")
	programData := filepath.Join(dir, "ProgramData")
	oldProgramData := windowsServiceProgramData
	windowsServiceProgramData = func() string { return programData }
	t.Cleanup(func() { windowsServiceProgramData = oldProgramData })
	source := filepath.Join(profile, "bin", "fak.exe")
	if err := os.MkdirAll(filepath.Dir(source), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("stable service image"), 0o600); err != nil {
		t.Fatal(err)
	}
	target, err := planWindowsServiceExecutable(source)
	if err != nil {
		t.Fatal(err)
	}
	binaryDir := windowsServiceBinaryDir()
	machineDir := filepath.Dir(binaryDir)
	if filepath.Dir(target) != binaryDir || strings.HasPrefix(strings.ToLower(target), strings.ToLower(profile)+string(filepath.Separator)) {
		t.Fatalf("SCM target=%q must be staged under machine binary dir=%q, not invoking profile=%q", target, binaryDir, profile)
	}

	var events []string
	windowsMakeDirAll = func(path string, mode os.FileMode) error {
		events = append(events, "mkdir:"+path)
		return os.MkdirAll(path, mode)
	}
	windowsApplyServiceBinaryDirSecurity = func(path string) error {
		events = append(events, "acl-dir:"+path)
		return nil
	}
	windowsApplyServiceBinarySecurity = func(path string) error {
		events = append(events, "acl-binary:"+path)
		return nil
	}
	windowsRenameServiceExecutable = func(from, to string) error {
		events = append(events, "publish:"+to)
		return os.Rename(from, to)
	}

	if err := stageWindowsServiceExecutable(source, target); err != nil {
		t.Fatal(err)
	}
	wantFirst := []string{
		"mkdir:" + binaryDir,
		"acl-dir:" + machineDir,
		"acl-dir:" + binaryDir,
	}
	terminalEvents := len(wantFirst)
	if len(events) != terminalEvents+2 || !reflect.DeepEqual(events[:terminalEvents], wantFirst) || !strings.HasPrefix(events[terminalEvents], "acl-binary:") || events[terminalEvents+1] != "publish:"+target {
		t.Fatalf("captured first-install plan=%v", events)
	}

	events = nil
	if err := stageWindowsServiceExecutable(source, target); err != nil {
		t.Fatal(err)
	}
	wantReinstall := append(append([]string{}, wantFirst...), "acl-binary:"+target)
	if !reflect.DeepEqual(events, wantReinstall) {
		t.Fatalf("captured idempotent reinstall plan=%v want=%v", events, wantReinstall)
	}
}

func TestWindowsServiceBinaryACLIsLeastPrivilege(t *testing.T) {
	for name, sddl := range map[string]string{
		"binary":    windowsServiceBinarySDDL,
		"directory": windowsServiceBinaryDirSDDL,
	} {
		t.Run(name, func(t *testing.T) {
			if !strings.Contains(sddl, ";GRGX;;;LS)") {
				t.Fatalf("LocalService lacks read/execute/traverse grant: %s", sddl)
			}
			if strings.Contains(sddl, ";FA;;;LS)") || strings.Contains(sddl, ";GW;;;LS)") || strings.Contains(sddl, ";GRGW;;;LS)") {
				t.Fatalf("LocalService unexpectedly has a write-capable grant: %s", sddl)
			}
			if !strings.HasPrefix(sddl, "D:P") {
				t.Fatalf("ACL inherits broad parent grants: %s", sddl)
			}
		})
	}
}
