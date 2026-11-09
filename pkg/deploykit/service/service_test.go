package service

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ── fakes ────────────────────────────────────────────────────────────────────

// fakeCommand is the injectable Command boundary for the three shell-driven
// drivers. Outputs are canned per joined-args key, as an ordered queue (the
// last entry repeats); a missing key panics so a test wiring bug surfaces
// instead of reading as "service not found".
type fakeCommand struct {
	calls   [][]string
	outputs map[string][][]byte
	fail    map[string]int
}

func newFakeCommand() *fakeCommand {
	return &fakeCommand{outputs: map[string][][]byte{}, fail: map[string]int{}}
}

func (f *fakeCommand) key(args ...string) string { return strings.Join(args, "\x00") }

func (f *fakeCommand) Run(args ...string) error {
	f.calls = append(f.calls, args)
	key := f.key(args...)
	if f.fail[key] > 0 {
		f.fail[key]--
		return fmt.Errorf("injected failure for %q", key)
	}
	return nil
}

func (f *fakeCommand) Output(args ...string) ([]byte, error) {
	f.calls = append(f.calls, args)
	key := f.key(args...)
	if f.fail[key] > 0 {
		f.fail[key]--
		return nil, fmt.Errorf("injected failure for %q", key)
	}
	q, ok := f.outputs[key]
	if !ok || len(q) == 0 {
		panic("fakeCommand: no canned output for " + key)
	}
	if len(q) > 1 {
		f.outputs[key] = q[1:]
	}
	return q[0], nil
}

func (f *fakeCommand) countCalls(match func([]string) bool) int {
	n := 0
	for _, c := range f.calls {
		if match(c) {
			n++
		}
	}
	return n
}

func (f *fakeCommand) callsWithVerb(verb string) int {
	return f.countCalls(func(c []string) bool {
		for _, a := range c {
			if a == verb {
				return true
			}
		}
		return false
	})
}

// fakeSCMManager / fakeSCMService drive the Windows SCM driver on any GOOS.
type fakeSCMManager struct {
	svc *fakeSCMService
}

func (m *fakeSCMManager) Disconnect() error { return nil }

func (m *fakeSCMManager) OpenService(string) (SCMService, error) { return m.svc, nil }

func (m *fakeSCMManager) CreateService(_ string, _ string, cfg SCMConfig, _ ...string) (SCMService, error) {
	s := &fakeSCMService{config: cfg, status: Status{State: StateStopped}}
	m.svc = s
	return s, nil
}

type recoverySet struct {
	actions []RecoveryAction
	reset   uint32
}

type fakeSCMService struct {
	config          SCMConfig
	actions         []RecoveryAction
	resetPeriod     uint32
	nonCrash        bool
	status          Status
	updateHistory   []SCMConfig
	recoveryHistory []recoverySet
	startCalls      int
	startErrs       []error
}

func (s *fakeSCMService) Close() error { return nil }

func (s *fakeSCMService) Config() (SCMConfig, error) { return s.config, nil }

func (s *fakeSCMService) UpdateConfig(cfg SCMConfig) error {
	s.updateHistory = append(s.updateHistory, cfg)
	s.config = cfg
	return nil
}

func (s *fakeSCMService) RecoveryActions() ([]RecoveryAction, error) {
	return append([]RecoveryAction(nil), s.actions...), nil
}

func (s *fakeSCMService) ResetPeriod() (uint32, error) { return s.resetPeriod, nil }

func (s *fakeSCMService) SetRecoveryActions(actions []RecoveryAction, reset uint32) error {
	s.recoveryHistory = append(s.recoveryHistory, recoverySet{
		actions: append([]RecoveryAction(nil), actions...),
		reset:   reset,
	})
	s.actions = append([]RecoveryAction(nil), actions...)
	s.resetPeriod = reset
	return nil
}

func (s *fakeSCMService) RecoveryActionsOnNonCrashFailures() (bool, error) { return s.nonCrash, nil }

func (s *fakeSCMService) SetRecoveryActionsOnNonCrashFailures(on bool) error {
	s.nonCrash = on
	return nil
}

func (s *fakeSCMService) Query() (Status, error) { return s.status, nil }

func (s *fakeSCMService) Control(cmd int) (Status, error) {
	if cmd == CmdStop {
		s.status.State = StateStopped
	}
	return s.status, nil
}

func (s *fakeSCMService) Start(...string) error {
	s.startCalls++
	if len(s.startErrs) > 0 {
		err := s.startErrs[0]
		s.startErrs = s.startErrs[1:]
		return err
	}
	s.status.State = StateRunning
	return nil
}

func (s *fakeSCMService) Delete() error { return nil }

// newTestManager builds a Manager with a forced driver kind so every driver
// runs on every host, independent of runtime.GOOS.
func newTestManager(kind driverKind, scope Scope, name string, opts ...Option) *Manager {
	m := &Manager{scope: scope, name: name, kind: kind, cmd: execCommand{}, uid: 501}
	for _, o := range opts {
		o(m)
	}
	return m
}

// ── the negative witness ─────────────────────────────────────────────────────

// TestServiceUpdateFailedStartRestoresPriorRegistration is the leaf's negative
// witness: a start failure injected AFTER the configuration update must leave
// the prior registration in place AND the prior running state (running→running,
// stopped→stopped), on all four driver kinds, on any host.
func TestServiceUpdateFailedStartRestoresPriorRegistration(t *testing.T) {
	cases := []struct {
		kind  driverKind
		scope Scope
		label string
	}{
		{driverSystemd, ScopeUser, "systemd"},
		{driverLaunchd, ScopeUser, "launchd"},
		{driverScheduledTask, ScopeUser, "scheduled-task"},
		{driverSCM, ScopeSystem, "scm"},
	}
	for _, tc := range cases {
		for _, priorRunning := range []bool{true, false} {
			prior := "stopped"
			if priorRunning {
				prior = "running"
			}
			t.Run(tc.label+"/prior-"+prior, func(t *testing.T) {
				switch tc.kind {
				case driverSystemd:
					testSystemdUpdateFailedStart(t, priorRunning)
				case driverLaunchd:
					testLaunchdUpdateFailedStart(t, priorRunning)
				case driverScheduledTask:
					testScheduledTaskUpdateFailedStart(t, priorRunning)
				case driverSCM:
					testSCMUpdateFailedStart(t, priorRunning)
				}
			})
		}
	}
}

func testSystemdUpdateFailedStart(t *testing.T, priorRunning bool) {
	t.Helper()
	tmp := t.TempDir()
	unitDir := filepath.Join(tmp, "units")
	if err := os.MkdirAll(unitDir, 0o755); err != nil {
		t.Fatal(err)
	}
	unitPath := filepath.Join(unitDir, "fak-test.service")
	if err := os.WriteFile(unitPath, []byte("OLD UNIT"), 0o644); err != nil {
		t.Fatal(err)
	}

	active := "inactive"
	mainPID := "0"
	if priorRunning {
		active, mainPID = "active", "123"
	}
	showOut := "LoadState=loaded\nActiveState=" + active + "\nSubState=running\nResult=success\nMainPID=" + mainPID +
		"\nInvocationID=inv\nNRestarts=0\nWatchdogTimestampUSec=0\nBootID=boot\n"

	fake := newFakeCommand()
	// The user-scope driver prefixes every invocation with --user.
	prefixed := func(args ...string) string { return fake.key(append([]string{"--user"}, args...)...) }
	unit := "fak-test.service"
	fake.outputs[prefixed("show", unit, "--no-pager",
		"--property=LoadState", "--property=ActiveState", "--property=SubState", "--property=Result",
		"--property=MainPID", "--property=InvocationID", "--property=NRestarts",
		"--property=WatchdogTimestampUSec", "--property=BootID")] = [][]byte{[]byte(showOut)}
	fake.fail[prefixed("start", unit)] = 1 // the apply's start fails; the restore's start succeeds

	m := newTestManager(driverSystemd, ScopeUser, "fak-test", WithCommand(fake), WithUnitDir(unitDir))
	err := m.Update(Definition{Text: "NEW UNIT"})
	if err == nil {
		t.Fatal("expected update failure")
	}
	if !strings.HasPrefix(err.Error(), "start:") || !strings.Contains(err.Error(), "previous service configuration preserved at ["+unitPath+"]") {
		t.Fatalf("error shape = %q", err)
	}
	got, rerr := os.ReadFile(unitPath)
	if rerr != nil {
		t.Fatal(rerr)
	}
	if string(got) != "OLD UNIT" {
		t.Fatalf("prior unit not restored: %q", got)
	}
	wantStarts := 1
	if priorRunning {
		wantStarts = 2 // the failed apply start + the restore start
	}
	if got := fake.callsWithVerb("start"); got != wantStarts {
		t.Fatalf("start calls = %d, want %d (prior running=%v)", got, wantStarts, priorRunning)
	}
}

func launchdPrintOut(plistPath string, running bool) string {
	state, pid := "not running", "0"
	if running {
		state, pid = "running", "123"
	}
	return "gui/501/com.fak.test = {\n" +
		"\ttype = LaunchAgent\n" +
		"\tpath = " + plistPath + "\n" +
		"\tname = com.fak.test\n" +
		"\tstate = " + state + "\n" +
		"\tpid = " + pid + "\n" +
		"\tlast exit code = 0\n" +
		"}\n"
}

func testLaunchdUpdateFailedStart(t *testing.T, priorRunning bool) {
	t.Helper()
	tmp := t.TempDir()
	plistDir := filepath.Join(tmp, "plists")
	if err := os.MkdirAll(plistDir, 0o755); err != nil {
		t.Fatal(err)
	}
	plistPath := filepath.Join(plistDir, "com.fak.test.plist")
	if err := os.WriteFile(plistPath, []byte("OLD PLIST"), 0o644); err != nil {
		t.Fatal(err)
	}

	fake := newFakeCommand()
	target := "gui/501/com.fak.test"
	// Print queue: first call is the snapshot (prior state), the second is the
	// post-bootstrap probe, which reports not-running so the kickstart fires.
	fake.outputs[fake.key("print", target)] = [][]byte{
		[]byte(launchdPrintOut(plistPath, priorRunning)),
		[]byte(launchdPrintOut(plistPath, false)),
	}
	fake.fail[fake.key("kickstart", "-k", target)] = 1

	m := newTestManager(driverLaunchd, ScopeUser, "com.fak.test", WithCommand(fake), WithPlistDir(plistDir), WithUID(501))
	err := m.Update(Definition{Text: "NEW PLIST"})
	if err == nil {
		t.Fatal("expected update failure")
	}
	if !strings.HasPrefix(err.Error(), "start:") || !strings.Contains(err.Error(), "previous service configuration preserved at ["+plistPath+"]") {
		t.Fatalf("error shape = %q", err)
	}
	got, rerr := os.ReadFile(plistPath)
	if rerr != nil {
		t.Fatal(rerr)
	}
	if string(got) != "OLD PLIST" {
		t.Fatalf("prior plist not restored: %q", got)
	}
	wantKickstarts := 1
	if priorRunning {
		wantKickstarts = 2 // the failed apply kickstart + the restore kickstart
	}
	if got := fake.callsWithVerb("kickstart"); got != wantKickstarts {
		t.Fatalf("kickstart calls = %d, want %d (prior running=%v)", got, wantKickstarts, priorRunning)
	}
}

func testScheduledTaskUpdateFailedStart(t *testing.T, priorRunning bool) {
	t.Helper()
	tmp := t.TempDir()
	xmlPath := filepath.Join(tmp, "serve-task.xml")

	const priorXML = "PRIOR TASK XML"
	statusLine := "Ready"
	if priorRunning {
		statusLine = "Running"
	}
	listOut := "Folder: \\\nTask Name:      \\FakTest\nStatus:         " + statusLine + "\n"

	fake := newFakeCommand()
	// schtasks /Query /XML carries a UTF-16 BOM on some builds; the driver
	// must strip it so the captured prior is byte-comparable.
	fake.outputs[fake.key("/Query", "/TN", "FakTest", "/XML")] = [][]byte{[]byte("\ufeff" + priorXML)}
	fake.outputs[fake.key("/Query", "/TN", "FakTest", "/FO", "LIST")] = [][]byte{[]byte(listOut)}
	// The apply's /Run fails; the restore's /Run succeeds.
	fake.fail[fake.key("/Run", "/TN", "FakTest")] = 1

	var creates []string
	// Intercept /Create to record the task XML registered at each call.
	wrapped := &createRecordingCommand{inner: fake, creates: &creates}
	m := newTestManager(driverScheduledTask, ScopeUser, "FakTest", WithCommand(wrapped))
	err := m.Update(Definition{Path: xmlPath, Text: "NEW TASK XML"})
	if err == nil {
		t.Fatal("expected update failure")
	}
	if !strings.HasPrefix(err.Error(), "start:") || !strings.Contains(err.Error(), "previous service configuration preserved at ["+xmlPath+"]") {
		t.Fatalf("error shape = %q", err)
	}
	got, rerr := os.ReadFile(xmlPath)
	if rerr != nil {
		t.Fatal(rerr)
	}
	if string(got) != priorXML {
		t.Fatalf("prior task xml not restored (want BOM-free %q): %q", priorXML, got)
	}
	if len(creates) != 2 || creates[0] != "NEW TASK XML" || creates[1] != priorXML {
		t.Fatalf("create registrations = %v, want [NEW TASK XML %s]", creates, priorXML)
	}
	wantRuns := 1
	if priorRunning {
		wantRuns = 2 // the failed apply /Run + the restore /Run
	}
	if got := fake.callsWithVerb("/Run"); got != wantRuns {
		t.Fatalf("/Run calls = %d, want %d (prior running=%v)", got, wantRuns, priorRunning)
	}
}

// createRecordingCommand wraps a Command and records the task XML file content
// passed to each schtasks /Create invocation.
type createRecordingCommand struct {
	inner   Command
	creates *[]string
}

func (c *createRecordingCommand) Run(args ...string) error {
	if args[0] == "/Create" {
		for i, a := range args {
			if a == "/XML" && i+1 < len(args) {
				b, err := os.ReadFile(args[i+1])
				if err != nil {
					*c.creates = append(*c.creates, "<unreadable: "+err.Error()+">")
				} else {
					*c.creates = append(*c.creates, string(b))
				}
			}
		}
	}
	return c.inner.Run(args...)
}

func (c *createRecordingCommand) Output(args ...string) ([]byte, error) {
	return c.inner.Output(args...)
}

func testSCMUpdateFailedStart(t *testing.T, priorRunning bool) {
	t.Helper()
	tmp := t.TempDir()
	binDir := filepath.Join(tmp, "bin")
	srcExe := filepath.Join(tmp, "fak-src.exe")
	if err := os.WriteFile(srcExe, []byte("new executable bytes"), 0o755); err != nil {
		t.Fatal(err)
	}

	priorImage := `"C:\old\fak-old.exe"`
	priorCfg := SCMConfig{
		StartType:        StartAutomatic,
		ErrorControl:     ErrorNormal,
		BinaryPathName:   priorImage,
		DisplayName:      "fak old",
		ServiceStartName: "NT AUTHORITY\\LocalService",
		SidType:          SidTypeRestricted,
	}
	svc := &fakeSCMService{
		config:      priorCfg,
		actions:     []RecoveryAction{{Type: ActionRestart, Delay: 60 * time.Second}},
		resetPeriod: 42,
		nonCrash:    false,
		status:      Status{State: StateStopped},
		startErrs:   []error{fmt.Errorf("injected start failure")},
	}
	if priorRunning {
		svc.status.State = StateRunning
	}
	mgr := &fakeSCMManager{svc: svc}

	m := newTestManager(driverSCM, ScopeSystem, "FakTest", WithSCM(mgr), WithBinaryDir(binDir))
	err := m.Update(Definition{Executable: srcExe, Args: []string{"service", "windows-run"}})
	if err == nil {
		t.Fatal("expected update failure")
	}
	if !strings.HasPrefix(err.Error(), "start:") || !strings.Contains(err.Error(), "previous service configuration preserved at ["+priorImage+"]") {
		t.Fatalf("error shape = %q", err)
	}
	// The prior configuration must be the LAST one applied.
	last := svc.updateHistory[len(svc.updateHistory)-1]
	if last != priorCfg {
		t.Fatalf("last config = %+v, want prior %+v", last, priorCfg)
	}
	lastRec := svc.recoveryHistory[len(svc.recoveryHistory)-1]
	if lastRec.reset != 42 || len(lastRec.actions) != 1 || lastRec.actions[0] != (RecoveryAction{Type: ActionRestart, Delay: 60 * time.Second}) {
		t.Fatalf("last recovery = %+v, want prior ladder", lastRec)
	}
	if svc.nonCrash {
		t.Fatal("prior non-crash flag not restored")
	}
	wantStarts := 1
	if priorRunning {
		wantStarts = 2 // the failed apply start + the restore start
	}
	if svc.startCalls != wantStarts {
		t.Fatalf("start calls = %d, want %d (prior running=%v)", svc.startCalls, wantStarts, priorRunning)
	}
	// The staged executable must have been published content-addressed.
	entries, rerr := os.ReadDir(binDir)
	if rerr != nil {
		t.Fatal(rerr)
	}
	if len(entries) != 1 || !strings.HasPrefix(entries[0].Name(), "fak-") || !strings.HasSuffix(entries[0].Name(), ".exe") {
		t.Fatalf("staged executables = %v", entries)
	}
}

// ── restore / snapshot behavior ──────────────────────────────────────────────

// TestServiceRestoreRecoversPriorSCMState exercises the exported Snapshot +
// Restore pair on the SCM driver: damage the live configuration, then restore
// the captured snapshot and require the prior configuration, recovery ladder,
// flag, and running state back.
func TestServiceRestoreRecoversPriorSCMState(t *testing.T) {
	for _, wasActive := range []bool{true, false} {
		state := "stopped"
		if wasActive {
			state = "active"
		}
		t.Run("was-"+state, func(t *testing.T) {
			svc := &fakeSCMService{
				config:      SCMConfig{BinaryPathName: `"C:\old\fak-old.exe"`, StartType: StartManual},
				actions:     []RecoveryAction{{Type: ActionRestart, Delay: 60 * time.Second}},
				resetPeriod: 42,
				nonCrash:    false,
				status:      Status{State: StateStopped},
			}
			if wasActive {
				svc.status.State = StateRunning
			}
			m := newTestManager(driverSCM, ScopeSystem, "FakTest", WithSCM(&fakeSCMManager{svc: svc}))

			snap, err := m.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			if snap.WasRunning != wasActive {
				t.Fatalf("snapshot WasRunning = %v, want %v", snap.WasRunning, wasActive)
			}
			// Damage the live registration.
			damaged := SCMConfig{BinaryPathName: `"C:\new\fak-new.exe"`, StartType: StartDisabled}
			if err := svc.UpdateConfig(damaged); err != nil {
				t.Fatal(err)
			}
			if err := svc.SetRecoveryActions(nil, 0); err != nil {
				t.Fatal(err)
			}
			if err := svc.SetRecoveryActionsOnNonCrashFailures(true); err != nil {
				t.Fatal(err)
			}

			if err := m.Restore(snap); err != nil {
				t.Fatal(err)
			}
			if svc.config.BinaryPathName != `"C:\old\fak-old.exe"` || svc.config.StartType != StartManual {
				t.Fatalf("config not restored: %+v", svc.config)
			}
			if svc.resetPeriod != 42 || len(svc.actions) != 1 {
				t.Fatalf("recovery not restored: reset=%d actions=%v", svc.resetPeriod, svc.actions)
			}
			if svc.nonCrash {
				t.Fatal("non-crash flag not restored")
			}
			if wasActive && svc.status.State != StateRunning {
				t.Fatalf("prior running state not restored: state=%d", svc.status.State)
			}
			if !wasActive && svc.startCalls != 0 {
				t.Fatalf("restore must not start a previously-stopped service, startCalls=%d", svc.startCalls)
			}
		})
	}
}

// TestServiceReadbackReportsObservedState pins the Readback mapping for the
// SCM driver, including the not-found sentinel.
func TestServiceReadbackReportsObservedState(t *testing.T) {
	svc := &fakeSCMService{
		config: SCMConfig{BinaryPathName: `"C:\fak\fak.exe" service windows-run`},
		status: Status{State: StateRunning, PID: 777},
	}
	m := newTestManager(driverSCM, ScopeSystem, "FakTest", WithSCM(&fakeSCMManager{svc: svc}))
	rb, err := m.Readback()
	if err != nil {
		t.Fatal(err)
	}
	if rb.State != Running || rb.PID != 777 || rb.Command != `C:\fak\fak.exe` {
		t.Fatalf("readback = %+v", rb)
	}

	m2 := newTestManager(driverSCM, ScopeSystem, "FakTest", WithSCM(notFoundManager{}))
	rb2, err := m2.Readback()
	if err != nil {
		t.Fatal(err)
	}
	if rb2.State != Unknown {
		t.Fatalf("readback of missing service = %+v, want unknown", rb2)
	}
}

// notFoundManager is an SCMBoundary whose OpenService fails with the
// not-found sentinel.
type notFoundManager struct{}

func (notFoundManager) Disconnect() error { return nil }

func (notFoundManager) OpenService(name string) (SCMService, error) {
	return nil, fmt.Errorf("%w: %s", ErrServiceDoesNotExist, name)
}

func (notFoundManager) CreateService(string, string, SCMConfig, ...string) (SCMService, error) {
	return nil, fmt.Errorf("%w: cannot create", ErrServiceDoesNotExist)
}

// ── pure renderer goldens ────────────────────────────────────────────────────

// TestServiceRenderGolden pins the six platform × scope renders byte-for-byte.
func TestServiceRenderGolden(t *testing.T) {
	uniDef := Definition{Executable: `/usr/local/bin/fak`, Args: []string{"serve", "--addr", "127.0.0.1:8080"}}
	winDef := Definition{Executable: `C:\Program Files\fak\fak.exe`, Args: []string{"serve"}}

	const linuxSystem = `[Unit]
Description=fak service fak-gw
After=network.target

[Service]
ExecStart=/usr/local/bin/fak serve --addr 127.0.0.1:8080
Restart=always

[Install]
WantedBy=multi-user.target
`
	linuxUser := strings.Replace(linuxSystem, "WantedBy=multi-user.target", "WantedBy=default.target", 1)

	const darwinPlist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.fak.gw</string>
<key>ProgramArguments</key><array><string>/usr/local/bin/fak</string><string>serve</string><string>--addr</string><string>127.0.0.1:8080</string></array>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
</dict></plist>
`

	const taskXML = `<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>fak service FakGw — written by deploykit</Description>
  </RegistrationInfo>
  <Triggers>
    <BootTrigger><Enabled>true</Enabled></BootTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>C:\Program Files\fak\fak.exe</Command>
      <Arguments>serve</Arguments>
    </Exec>
  </Actions>
</Task>
`

	const scmDef = `sc.exe create FakGw binPath= "\"C:\Program Files\fak\fak.exe\" serve" start= auto obj= "NT AUTHORITY\LocalService" DisplayName= "fak service FakGw"
sc.exe failure FakGw reset= 86400 actions= restart/3000/restart/10000/restart/30000
sc.exe failureflag FakGw 1
`

	cases := []struct {
		kind  driverKind
		scope Scope
		name  string
		def   Definition
		want  string
	}{
		{driverSystemd, ScopeSystem, "fak-gw", uniDef, linuxSystem},
		{driverSystemd, ScopeUser, "fak-gw", uniDef, linuxUser},
		{driverLaunchd, ScopeSystem, "com.fak.gw", uniDef, darwinPlist},
		{driverLaunchd, ScopeUser, "com.fak.gw", uniDef, darwinPlist},
		{driverScheduledTask, ScopeUser, "FakGw", winDef, taskXML},
		{driverSCM, ScopeSystem, "FakGw", winDef, scmDef},
	}
	for _, tc := range cases {
		t.Run(tc.name+"/"+string(tc.scope), func(t *testing.T) {
			m := newTestManager(tc.kind, tc.scope, tc.name)
			got, err := m.Render(tc.def)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("render mismatch:\n--- got ---\n%s\n--- want ---\n%s", got, tc.want)
			}
		})
	}
}

// TestServiceRenderRefusesAmbiguousValues pins the refuse-not-escape rule.
func TestServiceRenderRefusesAmbiguousValues(t *testing.T) {
	m := newTestManager(driverSystemd, ScopeUser, "fak-gw")
	if _, err := m.Render(Definition{Executable: `/bin/fak "quoted"`}); err == nil {
		t.Fatal("expected refusal for a double quote in the executable")
	}
	if _, err := m.Render(Definition{Executable: "/bin/fak\x00"}); err == nil {
		t.Fatal("expected refusal for a control character")
	}
	if _, err := m.Render(Definition{}); err == nil {
		t.Fatal("expected refusal for an empty executable")
	}
}

// ── small behavior pins ──────────────────────────────────────────────────────

func TestServiceParseScope(t *testing.T) {
	if s, err := ParseScope("system"); err != nil || s != ScopeSystem {
		t.Fatalf("ParseScope(system) = %v, %v", s, err)
	}
	if s, err := ParseScope("user"); err != nil || s != ScopeUser {
		t.Fatalf("ParseScope(user) = %v, %v", s, err)
	}
	if _, err := ParseScope("machine"); err == nil {
		t.Fatal("expected refusal for unknown scope")
	}
}

func TestServiceUpdateFailureShape(t *testing.T) {
	cause := fmt.Errorf("boom")
	ok := updateFailure("start", cause, nil, `/etc/systemd/system/fak.service`)
	if want := "start: boom; previous service configuration preserved at [/etc/systemd/system/fak.service]"; ok.Error() != want {
		t.Fatalf("success shape = %q, want %q", ok, want)
	}
	bad := updateFailure("start", cause, fmt.Errorf("restore blew up"), `/x/y`)
	if want := "start: boom; rollback failed: restore blew up; previous registration retained at [/x/y]"; bad.Error() != want {
		t.Fatalf("rollback shape = %q, want %q", bad, want)
	}
}

func TestServiceStageDocumentAndMoveNoReplace(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "unit.service")
	if err := os.WriteFile(target, []byte("OLD"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := stageDocument(target, "NEW"); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(target)
	if err != nil || string(got) != "NEW" {
		t.Fatalf("target = %q, %v", got, err)
	}
	// No temp litter beside the target.
	entries, _ := os.ReadDir(tmp)
	if len(entries) != 1 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Fatalf("dir entries = %v", names)
	}
	// moveNoReplace refuses to overwrite an existing target.
	src := filepath.Join(tmp, "src.bin")
	if err := os.WriteFile(src, []byte("X"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := moveNoReplace(src, target); err == nil {
		t.Fatal("expected moveNoReplace to refuse an existing target")
	}
	if err := moveNoReplace(src, filepath.Join(tmp, "fresh.bin")); err != nil {
		t.Fatal(err)
	}
}

func TestServiceStageExecutableContentAddressed(t *testing.T) {
	tmp := t.TempDir()
	src := filepath.Join(tmp, "fak-src.exe")
	if err := os.WriteFile(src, []byte("payload"), 0o755); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(tmp, "bin")
	planned, err := planStagedExecutable(dir, src)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(filepath.Base(planned), "fak-") || !strings.HasSuffix(planned, ".exe") {
		t.Fatalf("planned path = %s", planned)
	}
	if err := stageExecutable(src, planned); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(planned); err != nil {
		t.Fatalf("staged executable missing: %v", err)
	}
	// Re-staging identical bytes is a no-op on the same path.
	if err := stageExecutable(src, planned); err != nil {
		t.Fatal(err)
	}
	// A different payload gets a different path and never overwrites.
	other := filepath.Join(tmp, "fak-other.exe")
	if err := os.WriteFile(other, []byte("different"), 0o755); err != nil {
		t.Fatal(err)
	}
	otherPlanned, err := planStagedExecutable(dir, other)
	if err != nil {
		t.Fatal(err)
	}
	if otherPlanned == planned {
		t.Fatal("distinct payloads must plan distinct paths")
	}
}
