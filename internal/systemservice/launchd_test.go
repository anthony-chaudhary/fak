package systemservice

import (
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

func boolPtr(b bool) *bool { return &b }

func baseLaunchdSpec() *servicespec.Spec {
	return &servicespec.Spec{
		Schema:   servicespec.SchemaV1,
		Identity: servicespec.Identity{Node: "node-macos-1", Service: "gateway"},
		Kind:     servicespec.KindService,
		Desired:  servicespec.DesiredRunning,
		Command:  []string{"/opt/fak/bin/fak", "serve", "--addr", ":8080"},
		Dir:      "/var/lib/fak",
		Env: []servicespec.EnvRef{
			{Name: "FAK_TOKEN", SecretRef: "secret://fak/token"},
			{Name: "FAK_MODE", Value: "fleet"},
		},
		Readiness:     &servicespec.Readiness{Kind: "http", Target: "http://127.0.0.1:8080/healthz", TimeoutMS: 15000},
		CheckpointDir: "/var/lib/fak/ckpt",
		DependsOn:     []string{"registry"},
	}
}

func assertValidXML(t *testing.T, xmlText string) {
	t.Helper()
	dec := xml.NewDecoder(strings.NewReader(xmlText))
	for {
		_, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("rendered plist is not valid XML: %v\n%s", err, xmlText)
		}
	}
}

func TestRenderLaunchDaemonIsSystemOwnedWithoutTerminal(t *testing.T) {
	p, e := RenderLaunchDaemon(LaunchdConfig{Executable: "/usr/local/libexec/fak", StateDir: "/var/db/fak", StdoutPath: "/var/log/fak/out", StderrPath: "/var/log/fak/err", UserName: "_fakguard"})
	if e != nil {
		t.Fatal(e)
	}
	for _, w := range []string{
		"<string>com.fak.guard-control</string>", "<key>KeepAlive</key><true/>", "<key>RunAtLoad</key><true/>",
		"<key>ProcessType</key><string>Background</string>", "<key>UserName</key><string>_fakguard</string>",
		"<string>launchd-system</string>", "service</string><string>run",
	} {
		if !strings.Contains(p, w) {
			t.Fatalf("missing %q: %s", w, p)
		}
	}
	for _, forbidden := range []string{"Terminal.app", "Aqua", "LimitLoadToSessionType"} {
		if strings.Contains(p, forbidden) {
			t.Fatalf("system daemon has GUI dependency %q", forbidden)
		}
	}
}

func TestRenderLaunchDaemonEscapesAndRejectsUnsafeInput(t *testing.T) {
	p, e := RenderLaunchDaemon(LaunchdConfig{Executable: "/opt/a & b/fak", StateDir: "/var/db/fak", StdoutPath: "/var/log/fak/out", StderrPath: "/var/log/fak/err", UserName: "_fakguard"})
	if e != nil || !strings.Contains(p, "/opt/a &amp; b/fak") {
		t.Fatalf("XML escaping failed: %v %s", e, p)
	}
	if _, e := RenderLaunchDaemon(LaunchdConfig{Executable: "/x\ny", StateDir: "/s", StdoutPath: "/o", StderrPath: "/e", UserName: "u"}); e == nil {
		t.Fatal("accepted injection")
	}
	if _, e := RenderLaunchDaemon(LaunchdConfig{Executable: "/x", StateDir: "/s", StdoutPath: "/o", StderrPath: "/e"}); e == nil {
		t.Fatal("accepted missing principal")
	}
}

func TestProjectLaunchdGoldenDaemon(t *testing.T) {
	spec := baseLaunchdSpec()
	in := LaunchdInput{
		Type:              LaunchDaemon,
		User:              "_fakguard",
		GroupName:         "_fakguard",
		EnvironmentFiles:  []string{"/etc/fak/gateway.env"},
		StandardOutPath:   "/var/log/fak/gateway.out.log",
		StandardErrorPath: "/var/log/fak/gateway.err.log",
	}

	p, err := ProjectLaunchd(spec, in)
	if err != nil {
		t.Fatalf("ProjectLaunchd failed: %v", err)
	}

	if p.Schema != LaunchdProjectionSchemaV1 {
		t.Fatalf("schema = %q, want %q", p.Schema, LaunchdProjectionSchemaV1)
	}
	if p.Type != LaunchDaemon {
		t.Fatalf("type = %q, want %q", p.Type, LaunchDaemon)
	}
	if p.Label != "com.fak.gateway" {
		t.Fatalf("label = %q, want com.fak.gateway", p.Label)
	}
	if p.Domain != "system" {
		t.Fatalf("domain = %q, want system", p.Domain)
	}
	if p.Target != "system/com.fak.gateway" {
		t.Fatalf("target = %q, want system/com.fak.gateway", p.Target)
	}
	if p.PlistPath != "/Library/LaunchDaemons/com.fak.gateway.plist" {
		t.Fatalf("plist_path = %q, want /Library/LaunchDaemons/com.fak.gateway.plist", p.PlistPath)
	}
	if !p.Enabled || p.DesiredStopped {
		t.Fatalf("desired-running must project enabled: %+v", p)
	}

	// Modern launchctl commands
	wantBootstrap := "launchctl bootstrap system /Library/LaunchDaemons/com.fak.gateway.plist"
	if p.Commands.Bootstrap != wantBootstrap || p.BootstrapCommand() != wantBootstrap {
		t.Fatalf("bootstrap cmd = %q, want %q", p.Commands.Bootstrap, wantBootstrap)
	}
	wantBootout := "launchctl bootout system/com.fak.gateway"
	if p.Commands.Bootout != wantBootout || p.BootoutCommand() != wantBootout {
		t.Fatalf("bootout cmd = %q, want %q", p.Commands.Bootout, wantBootout)
	}
	wantKickstart := "launchctl kickstart -k system/com.fak.gateway"
	if p.Commands.Kickstart != wantKickstart || p.KickstartCommand() != wantKickstart {
		t.Fatalf("kickstart cmd = %q, want %q", p.Commands.Kickstart, wantKickstart)
	}
	wantPrint := "launchctl print system/com.fak.gateway"
	if p.Commands.Print != wantPrint || p.PrintCommand() != wantPrint {
		t.Fatalf("print cmd = %q, want %q", p.Commands.Print, wantPrint)
	}

	// XML validity
	assertValidXML(t, p.PlistXML)
	if p.PlistText() != p.PlistXML || p.UnitText() != p.PlistXML {
		t.Fatal("PlistText/UnitText accessor mismatch")
	}

	for _, w := range []string{
		"<key>Label</key><string>com.fak.gateway</string>",
		"<key>UserName</key><string>_fakguard</string>",
		"<key>GroupName</key><string>_fakguard</string>",
		"<key>WorkingDirectory</key><string>/var/lib/fak</string>",
		"<key>FAK_SERVICE_MANAGER</key><string>launchd-system</string>",
		"<key>FAK_MODE</key><string>fleet</string>",
		"<key>RunAtLoad</key><true/>",
		"<key>KeepAlive</key><true/>",
		"<key>ProcessType</key><string>Background</string>",
		"<key>StandardOutPath</key><string>/var/log/fak/gateway.out.log</string>",
		"<key>StandardErrorPath</key><string>/var/log/fak/gateway.err.log</string>",
	} {
		if !strings.Contains(p.PlistXML, w) {
			t.Fatalf("plist missing %q:\n%s", w, p.PlistXML)
		}
	}

	// Secrets must never be serialized
	if strings.Contains(p.PlistXML, "secret://fak/token") {
		t.Fatal("secret reference serialized into plist XML")
	}

	// Determinism
	p2, err := ProjectLaunchd(spec, in)
	if err != nil {
		t.Fatal(err)
	}
	if p2.PlistXML != p.PlistXML {
		t.Fatal("projection is not deterministic")
	}
}

func TestProjectLaunchdGoldenAgent(t *testing.T) {
	spec := &servicespec.Spec{
		Schema:   servicespec.SchemaV1,
		Identity: servicespec.Identity{Node: "node-macos-2", Service: "sweeper"},
		Kind:     servicespec.KindService,
		Desired:  servicespec.DesiredRunning,
		Command:  []string{"/usr/local/bin/fak", "sweep"},
	}
	in := LaunchdInput{
		Type:     LaunchAgent,
		UID:      501,
		PlistDir: "~/Library/LaunchAgents",
	}

	p, err := ProjectLaunchd(spec, in)
	if err != nil {
		t.Fatalf("ProjectLaunchd failed: %v", err)
	}

	if p.Type != LaunchAgent {
		t.Fatalf("type = %q, want %q", p.Type, LaunchAgent)
	}
	if p.Label != "com.fak.sweeper" {
		t.Fatalf("label = %q, want com.fak.sweeper", p.Label)
	}
	if p.Domain != "gui/501" {
		t.Fatalf("domain = %q, want gui/501", p.Domain)
	}
	if p.Target != "gui/501/com.fak.sweeper" {
		t.Fatalf("target = %q, want gui/501/com.fak.sweeper", p.Target)
	}
	if p.PlistPath != "~/Library/LaunchAgents/com.fak.sweeper.plist" {
		t.Fatalf("plist_path = %q", p.PlistPath)
	}

	wantBootstrap := "launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.fak.sweeper.plist"
	if p.Commands.Bootstrap != wantBootstrap {
		t.Fatalf("bootstrap cmd = %q, want %q", p.Commands.Bootstrap, wantBootstrap)
	}
	wantBootout := "launchctl bootout gui/501/com.fak.sweeper"
	if p.Commands.Bootout != wantBootout {
		t.Fatalf("bootout cmd = %q, want %q", p.Commands.Bootout, wantBootout)
	}

	assertValidXML(t, p.PlistXML)
	if strings.Contains(p.PlistXML, "<key>UserName</key>") {
		t.Fatal("LaunchAgent must not set UserName in plist")
	}
	if !strings.Contains(p.PlistXML, "<key>FAK_SERVICE_MANAGER</key><string>launchd-agent</string>") {
		t.Fatal("missing launchd-agent manager in LaunchAgent env")
	}
}

func TestProjectLaunchdGlobalAgentInLibraryLaunchAgents(t *testing.T) {
	spec := &servicespec.Spec{
		Schema:   servicespec.SchemaV1,
		Identity: servicespec.Identity{Node: "node-macos-1", Service: "daemon-agent"},
		Kind:     servicespec.KindService,
		Desired:  servicespec.DesiredRunning,
		Command:  []string{"/usr/local/bin/fak", "agent"},
	}
	in := LaunchdInput{
		Type:     LaunchAgent,
		Domain:   "system",
		PlistDir: "/Library/LaunchAgents",
	}

	p, err := ProjectLaunchd(spec, in)
	if err != nil {
		t.Fatal(err)
	}
	if p.PlistPath != "/Library/LaunchAgents/com.fak.daemon-agent.plist" {
		t.Fatalf("plist path = %q", p.PlistPath)
	}
	if p.Commands.Bootstrap != "launchctl bootstrap system /Library/LaunchAgents/com.fak.daemon-agent.plist" {
		t.Fatalf("bootstrap = %q", p.Commands.Bootstrap)
	}
}

func TestProjectLaunchdKeepAliveBounded(t *testing.T) {
	baseSpec := func(kind servicespec.Kind) *servicespec.Spec {
		return &servicespec.Spec{
			Schema:   servicespec.SchemaV1,
			Identity: servicespec.Identity{Node: "node-1", Service: "worker"},
			Kind:     kind,
			Desired:  servicespec.DesiredRunning,
			Command:  []string{"/usr/local/bin/fak", "work"},
		}
	}

	// Case A: KindJob defaults to SuccessfulExit: false
	jobProj, err := ProjectLaunchd(baseSpec(servicespec.KindJob), LaunchdInput{Type: LaunchDaemon})
	if err != nil {
		t.Fatal(err)
	}
	assertValidXML(t, jobProj.PlistXML)
	if !strings.Contains(jobProj.PlistXML, "<key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>") {
		t.Fatalf("KindJob must default to SuccessfulExit: false keepalive:\n%s", jobProj.PlistXML)
	}

	// Case B: Explicit SuccessfulExit: false on a Service
	svcProjB, err := ProjectLaunchd(baseSpec(servicespec.KindService), LaunchdInput{
		Type:      LaunchDaemon,
		KeepAlive: &LaunchdKeepAlive{SuccessfulExit: boolPtr(false)},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertValidXML(t, svcProjB.PlistXML)
	if !strings.Contains(svcProjB.PlistXML, "<key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>") {
		t.Fatalf("explicit SuccessfulExit: false not rendered:\n%s", svcProjB.PlistXML)
	}

	// Case C: Explicit Crashed: true
	svcProjC, err := ProjectLaunchd(baseSpec(servicespec.KindService), LaunchdInput{
		Type:      LaunchDaemon,
		KeepAlive: &LaunchdKeepAlive{Crashed: boolPtr(true)},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertValidXML(t, svcProjC.PlistXML)
	if !strings.Contains(svcProjC.PlistXML, "<key>KeepAlive</key><dict><key>Crashed</key><true/></dict>") {
		t.Fatalf("explicit Crashed: true not rendered:\n%s", svcProjC.PlistXML)
	}

	// Case D: Both Crashed: true and SuccessfulExit: false
	svcProjD, err := ProjectLaunchd(baseSpec(servicespec.KindService), LaunchdInput{
		Type:      LaunchDaemon,
		KeepAlive: &LaunchdKeepAlive{Crashed: boolPtr(true), SuccessfulExit: boolPtr(false)},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertValidXML(t, svcProjD.PlistXML)
	if !strings.Contains(svcProjD.PlistXML, "<key>KeepAlive</key><dict><key>Crashed</key><true/><key>SuccessfulExit</key><false/></dict>") {
		t.Fatalf("combined keepalive dict not rendered in sorted order:\n%s", svcProjD.PlistXML)
	}

	// Case E: Explicit Always: true
	svcProjE, err := ProjectLaunchd(baseSpec(servicespec.KindJob), LaunchdInput{
		Type:      LaunchDaemon,
		KeepAlive: &LaunchdKeepAlive{Always: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertValidXML(t, svcProjE.PlistXML)
	if !strings.Contains(svcProjE.PlistXML, "<key>KeepAlive</key><true/>") {
		t.Fatalf("explicit Always: true not rendered:\n%s", svcProjE.PlistXML)
	}

	// Case F: Explicit disabled keepalive
	svcProjF, err := ProjectLaunchd(baseSpec(servicespec.KindService), LaunchdInput{
		Type:      LaunchDaemon,
		KeepAlive: &LaunchdKeepAlive{},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertValidXML(t, svcProjF.PlistXML)
	if !strings.Contains(svcProjF.PlistXML, "<key>KeepAlive</key><false/>") {
		t.Fatalf("empty keepalive must render false:\n%s", svcProjF.PlistXML)
	}
}

func TestProjectLaunchdDesiredStopIsUnitInvariant(t *testing.T) {
	in := LaunchdInput{
		Type:             LaunchDaemon,
		User:             "_fakguard",
		EnvironmentFiles: []string{"/etc/fak/gateway.env"},
	}

	runningSpec := baseLaunchdSpec()
	running, err := ProjectLaunchd(runningSpec, in)
	if err != nil {
		t.Fatal(err)
	}

	stoppedSpec := baseLaunchdSpec()
	stoppedSpec.Desired = servicespec.DesiredStopped
	stopped, err := ProjectLaunchd(stoppedSpec, in)
	if err != nil {
		t.Fatal(err)
	}

	if running.PlistXML != stopped.PlistXML {
		t.Fatal("desired-stopped changed the rendered plist XML; it must be invariant")
	}
	if !running.Enabled || running.DesiredStopped {
		t.Fatalf("running projection enablement wrong: %+v", running)
	}
	if stopped.Enabled || !stopped.DesiredStopped {
		t.Fatalf("stopped projection enablement wrong: %+v", stopped)
	}
}

func TestProjectLaunchdRefusals(t *testing.T) {
	base := func() *servicespec.Spec { return baseLaunchdSpec() }
	in := LaunchdInput{
		Type:             LaunchDaemon,
		EnvironmentFiles: []string{"/etc/fak/gateway.env"},
	}

	if _, err := ProjectLaunchd(nil, in); !errors.Is(err, ErrNilSpec) {
		t.Fatalf("nil spec: %v", err)
	}
	if _, err := ProjectLaunchd(base(), LaunchdInput{Type: "unknown"}); !errors.Is(err, ErrUnknownLaunchdType) {
		t.Fatalf("unknown launchd type: %v", err)
	}

	// Secret without environment file
	if _, err := ProjectLaunchd(base(), LaunchdInput{Type: LaunchDaemon}); !errors.Is(err, ErrSecretNeedsFile) {
		t.Fatalf("secret without env file: %v", err)
	}

	// Bad workload token
	s := base()
	s.Identity.Workload = "evil/traversal"
	if _, err := ProjectLaunchd(s, in); !errors.Is(err, ErrBadLabelToken) {
		t.Fatalf("bad workload token: %v", err)
	}

	// Bad dependency token
	s = base()
	s.DependsOn = []string{"dep/invalid"}
	if _, err := ProjectLaunchd(s, in); !errors.Is(err, ErrBadLabelToken) {
		t.Fatalf("bad dependency token: %v", err)
	}

	// Control-character injections
	s = base()
	s.Command = []string{"/bin/sh\n<key>evil</key>"}
	if _, err := ProjectLaunchd(s, in); err == nil {
		t.Fatal("accepted command with control character")
	}

	s = base()
	s.Dir = "/var/lib\r\nfak"
	if _, err := ProjectLaunchd(s, in); err == nil {
		t.Fatal("accepted dir with control character")
	}

	badIn := in
	badIn.User = "user\nroot"
	if _, err := ProjectLaunchd(base(), badIn); err == nil {
		t.Fatal("accepted user with control character")
	}
}

const sampleRealSpotlightPrint = `gui/502/com.apple.Spotlight = {
	active count = 11
	path = /System/Library/LaunchAgents/com.apple.Spotlight.plist
	type = LaunchAgent
	state = running

	program = /System/Library/CoreServices/Spotlight.app/Contents/MacOS/Spotlight
	arguments = {
		/System/Library/CoreServices/Spotlight.app/Contents/MacOS/Spotlight
	}

	domain = gui/502 [100022]
	asid = 100022
	minimum runtime = 1
	runs = 1
	pid = 773
	immediate reason = ipc (mach)
	last exit code = (never exited)
}
`

func TestParseLaunchctlPrintRealSpotlight(t *testing.T) {
	st, err := ParseLaunchctlPrint(sampleRealSpotlightPrint)
	if err != nil {
		t.Fatalf("ParseLaunchctlPrint failed: %v", err)
	}

	if st.Label != "com.apple.Spotlight" {
		t.Fatalf("label = %q", st.Label)
	}
	if st.Domain != "gui/502" {
		t.Fatalf("domain = %q", st.Domain)
	}
	if st.Target != "gui/502/com.apple.Spotlight" {
		t.Fatalf("target = %q", st.Target)
	}
	if st.Type != "LaunchAgent" {
		t.Fatalf("type = %q", st.Type)
	}
	if st.State != "running" {
		t.Fatalf("state = %q", st.State)
	}
	if st.PID != 773 {
		t.Fatalf("pid = %d", st.PID)
	}
	if st.PlistPath != "/System/Library/LaunchAgents/com.apple.Spotlight.plist" {
		t.Fatalf("plist path = %q", st.PlistPath)
	}
	if st.LastExitCode != nil {
		t.Fatalf("last exit code = %v, want nil", *st.LastExitCode)
	}
	if !st.IsRunning() {
		t.Fatal("IsRunning() should be true")
	}
	if st.Phase != servicespec.PhaseReady {
		t.Fatalf("phase = %q, want %q", st.Phase, servicespec.PhaseReady)
	}
	if st.Observed.Phase != servicespec.PhaseReady {
		t.Fatalf("observed phase = %q", st.Observed.Phase)
	}
	if st.ObservedStatus.Phase != servicespec.PhaseReady {
		t.Fatalf("observed_status phase = %q", st.ObservedStatus.Phase)
	}
	if st.Observed.LastExit != nil {
		t.Fatalf("unexpected last exit: %+v", st.Observed.LastExit)
	}
}

const sampleRealAnthony350Print = `gui/502/com.sample.campaign.job350 = {
	active count = 0
	path = /Library/LaunchAgents/com.sample.campaign.job350.plist
	type = LaunchAgent
	state = spawn scheduled

	program = /bin/zsh
	domain = gui/502 [100022]
	minimum runtime = 60
	runs = 2747
	last exit code = 3
}
`

func TestParseLaunchctlPrintRealExitedJob(t *testing.T) {
	st, err := ParseLaunchctlPrint(sampleRealAnthony350Print)
	if err != nil {
		t.Fatalf("ParseLaunchctlPrint failed: %v", err)
	}

	if st.Label != "com.sample.campaign.job350" {
		t.Fatalf("label = %q", st.Label)
	}
	if st.State != "spawn scheduled" {
		t.Fatalf("state = %q", st.State)
	}
	if st.PID != 0 {
		t.Fatalf("pid = %d", st.PID)
	}
	code, ok := st.ExitCode()
	if !ok || code != 3 {
		t.Fatalf("ExitCode = (%d, %v), want (3, true)", code, ok)
	}
	if st.Phase != servicespec.PhaseStarting {
		t.Fatalf("phase = %q, want %q", st.Phase, servicespec.PhaseStarting)
	}
	if st.Observed.LastExit == nil || st.Observed.LastExit.Code != 3 || st.Observed.LastExit.Class != servicespec.ExitCrash {
		t.Fatalf("last exit = %+v, want code 3 / crash", st.Observed.LastExit)
	}
}

const sampleRealFresh300Print = `gui/502/com.sample.campaign.fresh300 = {
	active count = 0
	path = /Library/LaunchAgents/com.sample.campaign.fresh300.plist
	type = LaunchAgent
	state = not running

	program = /usr/bin/python3
	stdout path = /var/log/fresh300.out
	domain = gui/502 [100022]
	runs = 1
	last exit code = 0
}
`

func TestParseLaunchctlPrintCleanExitedJob(t *testing.T) {
	st, err := ParseLaunchctlPrint(sampleRealFresh300Print)
	if err != nil {
		t.Fatalf("ParseLaunchctlPrint failed: %v", err)
	}

	if st.Label != "com.sample.campaign.fresh300" {
		t.Fatalf("label = %q", st.Label)
	}
	if st.State != "not running" {
		t.Fatalf("state = %q", st.State)
	}
	if st.PlistPath != "/Library/LaunchAgents/com.sample.campaign.fresh300.plist" {
		t.Fatalf("plist path = %q", st.PlistPath)
	}
	code, ok := st.ExitCode()
	if !ok || code != 0 {
		t.Fatalf("ExitCode = (%d, %v), want (0, true)", code, ok)
	}
	if st.Phase != servicespec.PhaseStopped {
		t.Fatalf("phase = %q, want %q", st.Phase, servicespec.PhaseStopped)
	}
	if st.Observed.LastExit == nil || st.Observed.LastExit.Code != 0 || st.Observed.LastExit.Class != servicespec.ExitClean {
		t.Fatalf("last exit = %+v, want code 0 / clean", st.Observed.LastExit)
	}
}

func TestParseLaunchctlPrintSyntheticIncumbent(t *testing.T) {
	raw := "gui/501/com.fak.model = {\n\tpath = /Library/LaunchAgents/com.fak.model.plist\n\tpid = 50123\n}\n"
	st, err := ParseLaunchctlPrint(raw)
	if err != nil {
		t.Fatal(err)
	}

	if st.Label != "com.fak.model" {
		t.Fatalf("label = %q", st.Label)
	}
	if st.PID != 50123 {
		t.Fatalf("pid = %d", st.PID)
	}
	if st.State != "running" {
		t.Fatalf("state = %q", st.State)
	}
	if st.PlistPath != "/Library/LaunchAgents/com.fak.model.plist" {
		t.Fatalf("plist path = %q", st.PlistPath)
	}
	if st.Phase != servicespec.PhaseReady {
		t.Fatalf("phase = %q", st.Phase)
	}
}

func TestParseLaunchctlPrintWatchdogExit(t *testing.T) {
	raw := `system/com.fak.daemon = {
	state = not running
	last exit code = 137
	last exit reason = (watchdog timeout)
}
`
	st, err := ParseLaunchctlPrint(raw)
	if err != nil {
		t.Fatal(err)
	}

	if st.Label != "com.fak.daemon" {
		t.Fatalf("label = %q", st.Label)
	}
	if st.Phase != servicespec.PhaseFailed {
		t.Fatalf("phase = %q, want %q", st.Phase, servicespec.PhaseFailed)
	}
	if st.ExitReason != "(watchdog timeout)" {
		t.Fatalf("exit reason = %q", st.ExitReason)
	}
	if st.Observed.LastExit == nil || st.Observed.LastExit.Class != servicespec.ExitWatchdog {
		t.Fatalf("exit record = %+v, want ExitWatchdog", st.Observed.LastExit)
	}
}

func TestParseLaunchctlPrintErrors(t *testing.T) {
	if _, err := ParseLaunchctlPrint(""); !errors.Is(err, ErrEmptyLaunchctlPrint) {
		t.Fatalf("empty input: %v", err)
	}
	if _, err := ParseLaunchctlPrint("   \n\t  "); !errors.Is(err, ErrEmptyLaunchctlPrint) {
		t.Fatalf("whitespace input: %v", err)
	}
	if _, err := ParseLaunchctlPrint("Could not find service \"com.fak.absent\" in domain"); !errors.Is(err, ErrLaunchctlServiceNotFound) {
		t.Fatalf("not found: %v", err)
	}
	if _, err := ParseLaunchctlPrint("service is not loaded"); !errors.Is(err, ErrLaunchctlServiceNotFound) {
		t.Fatalf("not loaded: %v", err)
	}
	if _, err := ParseLaunchctlPrint("gui/501 = {\n\tactive count = 10\n}"); !errors.Is(err, ErrMalformedLaunchctlOutput) {
		t.Fatalf("domain-only print: %v", err)
	}
	if _, err := ParseLaunchctlPrint("gui/501/com.fak.bad = {\n\tpid = abc\n}"); !errors.Is(err, ErrMalformedLaunchctlOutput) {
		t.Fatalf("invalid pid: %v", err)
	}
	if _, err := ParseLaunchctlPrint("some random prose without label or target"); !errors.Is(err, ErrMalformedLaunchctlOutput) {
		t.Fatalf("missing label: %v", err)
	}
}

// sampleNotRunningWithCoalitionsPrint is a stopped job whose nested
// resource/jetsam coalition blocks still report `state = active` and
// `type = resource|jetsam`. Only the top-level keys may drive the status.
const sampleNotRunningWithCoalitionsPrint = `gui/501/com.fak.up = {
	active count = 0
	path = /Users/example/Library/LaunchAgents/com.fak.up.plist
	type = LaunchAgent
	state = not running

	program = /Users/example/.local/bin/fak-native
	arguments = {
		/Users/example/.local/bin/fak-native
		up
		--headless
	}

	environment = {
		XPC_SERVICE_NAME => com.fak.up
	}

	domain = gui/501 [100024]
	runs = 3
	last exit code = 0

	resource coalition = {
		ID = 4295
		type = resource
		state = active
		active count = 0
		name = com.fak.up
	}

	jetsam coalition = {
		ID = 4296
		type = jetsam
		state = active
		active count = 0
		name = com.fak.up
	}

	spawn type = daemon (3)
	properties = keepalive | runatload | inferred program
}
`

func TestParseLaunchctlPrintNotRunningIgnoresNestedCoalitionState(t *testing.T) {
	st, err := ParseLaunchctlPrint(sampleNotRunningWithCoalitionsPrint)
	if err != nil {
		t.Fatalf("ParseLaunchctlPrint failed: %v", err)
	}
	if st.State != "not running" {
		t.Fatalf("state = %q, want %q (nested coalition state leaked)", st.State, "not running")
	}
	if st.Type != "LaunchAgent" {
		t.Fatalf("type = %q, want LaunchAgent (nested coalition type leaked)", st.Type)
	}
	if st.IsRunning() {
		t.Fatal("IsRunning() = true for a not-running job")
	}
	code, ok := st.ExitCode()
	if !ok || code != 0 {
		t.Fatalf("ExitCode = (%d, %v), want (0, true)", code, ok)
	}
	if st.Phase != servicespec.PhaseStopped {
		t.Fatalf("phase = %q, want %q", st.Phase, servicespec.PhaseStopped)
	}
}

// sampleRealFakUpRunningPrint is a scrubbed `launchctl print gui/<uid>/com.fak.up`
// capture of a running LaunchAgent on macOS 26 (uid -> 501, home ->
// /Users/example, SSH_AUTH_SOCK value redacted). The nested coalition blocks
// carry their own `type = ...` / `state = active` keys.
const sampleRealFakUpRunningPrint = `gui/501/com.fak.up = {
	active count = 1
	path = /Users/example/Library/LaunchAgents/com.fak.up.plist
	type = LaunchAgent
	state = running

	program = /Users/example/.local/bin/fak-native
	arguments = {
		/Users/example/.local/bin/fak-native
		up
		--headless
		--model
		27B
		--context
		20480
		--max-rss
		25769803776
		--max-rss-sustain
		3m
	}

	stdout path = /tmp/fak-up.log
	stderr path = /tmp/fak-up.err
	inherited environment = {
		SSH_AUTH_SOCK => <redacted>
	}

	default environment = {
		PATH => /usr/bin:/bin:/usr/sbin:/sbin
	}

	environment = {
		OSLogRateLimit => 64
		XPC_SERVICE_NAME => com.fak.up
	}

	domain = gui/501 [100024]
	asid = 100024
	minimum runtime = 30
	exit timeout = 5
	runs = 50
	pid = 74723
	immediate reason = inefficient
	forks = 27
	execs = 1
	initialized = 1
	trampolined = 1
	started suspended = 0
	proxy started suspended = 0
	checked allocations = 0 (queried = 1)
	checked allocations reason = no host
	checked allocations flags = 0x0
	last exit code = 0

	resource coalition = {
		ID = 4295
		type = resource
		state = active
		active count = 1
		name = com.fak.up
	}

	jetsam coalition = {
		ID = 4296
		type = jetsam
		state = active
		active count = 1
		name = com.fak.up
	}

	spawn type = daemon (3)
	jetsam priority = 40
	jetsam memory limit (active) = (unlimited)
	jetsam memory limit (inactive) = (unlimited)
	jetsamproperties category = daemon
	jetsam thread limit = 32
	cpumon = default

	properties = keepalive | runatload | inferred program
}
`

func TestParseLaunchctlPrintRealFakUpRunning(t *testing.T) {
	st, err := ParseLaunchctlPrint(sampleRealFakUpRunningPrint)
	if err != nil {
		t.Fatalf("ParseLaunchctlPrint failed: %v", err)
	}
	if st.Label != "com.fak.up" {
		t.Fatalf("label = %q", st.Label)
	}
	if st.Domain != "gui/501" {
		t.Fatalf("domain = %q, want gui/501", st.Domain)
	}
	if st.Target != "gui/501/com.fak.up" {
		t.Fatalf("target = %q", st.Target)
	}
	if st.State != "running" {
		t.Fatalf("state = %q, want running (nested coalition state leaked?)", st.State)
	}
	if st.Type != "LaunchAgent" {
		t.Fatalf("type = %q, want LaunchAgent (nested coalition type leaked?)", st.Type)
	}
	if st.PID != 74723 {
		t.Fatalf("pid = %d, want 74723", st.PID)
	}
	if st.PlistPath != "/Users/example/Library/LaunchAgents/com.fak.up.plist" {
		t.Fatalf("plist path = %q", st.PlistPath)
	}
	if st.Program != "/Users/example/.local/bin/fak-native" {
		t.Fatalf("program = %q", st.Program)
	}
	wantArgs := []string{
		"/Users/example/.local/bin/fak-native", "up", "--headless",
		"--model", "27B", "--context", "20480",
		"--max-rss", "25769803776", "--max-rss-sustain", "3m",
	}
	if !reflect.DeepEqual(st.Arguments, wantArgs) {
		t.Fatalf("arguments = %q, want %q", st.Arguments, wantArgs)
	}
	if st.Runs != 50 {
		t.Fatalf("runs = %d, want 50", st.Runs)
	}
	wantProps := []string{"keepalive", "runatload", "inferred program"}
	if !reflect.DeepEqual(st.Properties, wantProps) {
		t.Fatalf("properties = %q, want %q", st.Properties, wantProps)
	}
	if !slices.Contains(st.Properties, "keepalive") {
		t.Fatalf("properties %q missing keepalive", st.Properties)
	}
	code, ok := st.ExitCode()
	if !ok || code != 0 {
		t.Fatalf("ExitCode = (%d, %v), want (0, true)", code, ok)
	}
	if st.ExitReason != "inefficient" {
		t.Fatalf("exit reason = %q, want inefficient", st.ExitReason)
	}
	if !st.IsRunning() {
		t.Fatal("IsRunning() should be true")
	}
	if st.Phase != servicespec.PhaseReady {
		t.Fatalf("phase = %q, want %q", st.Phase, servicespec.PhaseReady)
	}
}

func TestParseLaunchctlPrintNestedBlocksDoNotLeakIntoTopLevel(t *testing.T) {
	// Nested blocks reuse top-level key names; none of them may win.
	raw := `gui/501/com.fak.nested = {
	path = /Users/example/Library/LaunchAgents/com.fak.nested.plist
	type = LaunchAgent
	state = not running
	program = /usr/local/bin/fak
	endpoints = {
		"com.fak.nested.port" = {
			port = 0x1234
			active = 0
			state = active
			pid = 999
		}
	}
	event triggers = {
		trigger = {
			type = com.apple.notifyd.matching
			state = active
			path = /tmp/nested
			domain = gui/999
			last exit code = 9
		}
	}
	last exit code = (never exited)
}
`
	st, err := ParseLaunchctlPrint(raw)
	if err != nil {
		t.Fatalf("ParseLaunchctlPrint failed: %v", err)
	}
	if st.State != "not running" || st.Type != "LaunchAgent" || st.PID != 0 {
		t.Fatalf("nested keys leaked: state=%q type=%q pid=%d", st.State, st.Type, st.PID)
	}
	if st.PlistPath != "/Users/example/Library/LaunchAgents/com.fak.nested.plist" || st.Domain != "gui/501" {
		t.Fatalf("nested path/domain leaked: path=%q domain=%q", st.PlistPath, st.Domain)
	}
	if st.LastExitCode != nil {
		t.Fatalf("last exit code = %d, want nil (never exited)", *st.LastExitCode)
	}
	if st.Arguments != nil || st.Properties != nil {
		t.Fatalf("absent arguments/properties should be nil: %q %q", st.Arguments, st.Properties)
	}
	if st.Phase != servicespec.PhaseStopped {
		t.Fatalf("phase = %q, want %q", st.Phase, servicespec.PhaseStopped)
	}
}

func TestParseLaunchctlPrintHeaderlessTreatsDepthZeroAsTopLevel(t *testing.T) {
	raw := "\tstate = not running\n\tprogram = /usr/local/bin/fak\n\tresource coalition = {\n\t\tstate = active\n\t}\n\tenvironment = {\n\t\tXPC_SERVICE_NAME => com.fak.headless\n\t}\n\tlast exit code = 0\n"
	st, err := ParseLaunchctlPrint(raw)
	if err != nil {
		t.Fatalf("ParseLaunchctlPrint failed: %v", err)
	}
	if st.Label != "com.fak.headless" || st.Target != "com.fak.headless" {
		t.Fatalf("label/target = %q/%q", st.Label, st.Target)
	}
	if st.State != "not running" || st.Program != "/usr/local/bin/fak" {
		t.Fatalf("state=%q program=%q", st.State, st.Program)
	}
	if st.Phase != servicespec.PhaseStopped {
		t.Fatalf("phase = %q, want %q", st.Phase, servicespec.PhaseStopped)
	}
}

func TestParseLaunchctlPrintExitCodeWithTrailingText(t *testing.T) {
	raw := `gui/501/com.fak.up = {
	path = /Users/example/Library/LaunchAgents/com.fak.up.plist
	type = LaunchAgent
	state = not running
	program = /Users/example/.local/bin/fak-native
	domain = gui/501 [100024]
	runs = 7
	last exit code = 1: Operation not permitted
}
`
	st, err := ParseLaunchctlPrint(raw)
	if err != nil {
		t.Fatalf("ParseLaunchctlPrint failed: %v", err)
	}
	code, ok := st.ExitCode()
	if !ok || code != 1 {
		t.Fatalf("ExitCode = (%d, %v), want (1, true)", code, ok)
	}
	if st.Phase != servicespec.PhaseFailed {
		t.Fatalf("phase = %q, want %q", st.Phase, servicespec.PhaseFailed)
	}
	if st.Observed.LastExit == nil || st.Observed.LastExit.Code != 1 || st.Observed.LastExit.Class != servicespec.ExitCrash {
		t.Fatalf("last exit = %+v, want code 1 / crash", st.Observed.LastExit)
	}
	if st.Runs != 7 {
		t.Fatalf("runs = %d, want 7", st.Runs)
	}
}

func TestParseLeadingInt(t *testing.T) {
	for in, want := range map[string]int{"0": 0, "3": 3, "1: Operation not permitted": 1, " 78:EX_CONFIG": 78, "-1": -1} {
		if got, ok := parseLeadingInt(in); !ok || got != want {
			t.Fatalf("parseLeadingInt(%q) = (%d, %v), want (%d, true)", in, got, ok, want)
		}
	}
	for _, in := range []string{"", "(never exited)", "-", "abc"} {
		if got, ok := parseLeadingInt(in); ok {
			t.Fatalf("parseLeadingInt(%q) = (%d, true), want not ok", in, got)
		}
	}
}

func TestParseLaunchctlPrintDisabled(t *testing.T) {
	modern := "\n\tdisabled services = {\n\t\t\"com.docker.helper\" => enabled\n\t\t\"com.fak.up\" => enabled\n\t\t\"com.fak.up.old\" => disabled\n\t\t\"com.apple.Siri.agent\" => disabled\n\t}\n"
	legacy := "\n\tdisabled services = {\n\t\t\"com.fak.up\" => false\n\t\t\"com.fak.guard\" => true\n\t}\n"
	cases := []struct {
		name, output, label  string
		wantDisabled, wantOK bool
	}{
		{"modern enabled", modern, "com.fak.up", false, true},
		{"modern disabled", modern, "com.fak.up.old", true, true},
		{"modern disabled other", modern, "com.apple.Siri.agent", true, true},
		{"legacy false means enabled", legacy, "com.fak.up", false, true},
		{"legacy true means disabled", legacy, "com.fak.guard", true, true},
		{"not found", modern, "com.fak.absent", false, false},
		{"prefix is not a match", modern, "com.fak", false, false},
		{"empty output", "", "com.fak.up", false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			disabled, found := ParseLaunchctlPrintDisabled(tc.output, tc.label)
			if disabled != tc.wantDisabled || found != tc.wantOK {
				t.Fatalf("ParseLaunchctlPrintDisabled(%q) = (%v, %v), want (%v, %v)", tc.label, disabled, found, tc.wantDisabled, tc.wantOK)
			}
		})
	}
}

func TestProjectLaunchdEnableDisableCommands(t *testing.T) {
	agent, err := ProjectLaunchd(&servicespec.Spec{
		Schema:   servicespec.SchemaV1,
		Identity: servicespec.Identity{Node: "node-macos-2", Service: "up"},
		Kind:     servicespec.KindService,
		Desired:  servicespec.DesiredRunning,
		Command:  []string{"/usr/local/bin/fak", "up"},
	}, LaunchdInput{Type: LaunchAgent, UID: 501, PlistDir: "~/Library/LaunchAgents"})
	if err != nil {
		t.Fatal(err)
	}
	if want := "launchctl enable gui/501/com.fak.up"; agent.Commands.Enable != want || agent.EnableCommand() != want {
		t.Fatalf("enable cmd = %q, want %q", agent.Commands.Enable, want)
	}
	if want := "launchctl disable gui/501/com.fak.up"; agent.Commands.Disable != want || agent.DisableCommand() != want {
		t.Fatalf("disable cmd = %q, want %q", agent.Commands.Disable, want)
	}

	daemon, err := ProjectLaunchd(baseLaunchdSpec(), LaunchdInput{Type: LaunchDaemon, EnvironmentFiles: []string{"/etc/fak/gateway.env"}})
	if err != nil {
		t.Fatal(err)
	}
	if want := "launchctl enable system/com.fak.gateway"; daemon.EnableCommand() != want {
		t.Fatalf("enable cmd = %q, want %q", daemon.EnableCommand(), want)
	}
	if want := "launchctl disable system/com.fak.gateway"; daemon.DisableCommand() != want {
		t.Fatalf("disable cmd = %q, want %q", daemon.DisableCommand(), want)
	}

	raw, err := json.Marshal(agent.Commands)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []string{`"enable":"launchctl enable gui/501/com.fak.up"`, `"disable":"launchctl disable gui/501/com.fak.up"`} {
		if !strings.Contains(string(raw), w) {
			t.Fatalf("commands JSON missing %s: %s", w, raw)
		}
	}
}

// TestParseLaunchctlPrintLiveWitness parses the host's real, read-only
// `launchctl print gui/<uid>/<label>` and `launchctl print-disabled gui/<uid>`
// output. Opt-in: set FAK_LAUNCHD_LIVE_LABEL (e.g. com.fak.up) on macOS.
func TestParseLaunchctlPrintLiveWitness(t *testing.T) {
	label := os.Getenv("FAK_LAUNCHD_LIVE_LABEL")
	if runtime.GOOS != "darwin" || label == "" {
		t.Skip("set FAK_LAUNCHD_LIVE_LABEL on macOS to parse live launchctl output")
	}
	domain := fmt.Sprintf("gui/%d", os.Getuid())
	out, err := exec.Command("launchctl", "print", domain+"/"+label).CombinedOutput()
	if err != nil {
		t.Skipf("launchctl print %s/%s: %v: %s", domain, label, err, out)
	}
	st, err := ParseLaunchctlPrint(string(out))
	if err != nil {
		t.Fatalf("ParseLaunchctlPrint(live) failed: %v", err)
	}
	if st.Label != label || st.Domain != domain {
		t.Fatalf("label/domain = %q/%q, want %q/%q", st.Label, st.Domain, label, domain)
	}
	if st.Type != string(LaunchAgent) && st.Type != string(LaunchDaemon) {
		t.Fatalf("type = %q, want a launchd job type (nested block leaked?)", st.Type)
	}
	if st.State == "active" {
		t.Fatalf("state = %q: nested coalition state leaked into the job state", st.State)
	}
	if st.Program == "" && len(st.Arguments) == 0 {
		t.Fatalf("neither program nor arguments parsed from live output")
	}
	if st.State == "running" && (st.PID <= 0 || st.Phase != servicespec.PhaseReady) {
		t.Fatalf("running job parsed pid=%d phase=%q", st.PID, st.Phase)
	}
	t.Logf("live %s: state=%q type=%q pid=%d runs=%d phase=%q program=%q args=%q properties=%q",
		st.Target, st.State, st.Type, st.PID, st.Runs, st.Phase, st.Program, st.Arguments, st.Properties)

	dis, err := exec.Command("launchctl", "print-disabled", domain).CombinedOutput()
	if err != nil {
		t.Skipf("launchctl print-disabled %s: %v", domain, err)
	}
	disabled, found := ParseLaunchctlPrintDisabled(string(dis), label)
	t.Logf("live print-disabled %s: disabled=%v found=%v", label, disabled, found)
}

// TestParseLaunchctlPrintReviewRegressions pins the real-output shapes an
// independent review of 868 live prints found (#13535): inline block closes,
// argv that mentions a not-found phrase, verbatim argv, truncation, a merged
// stderr preamble, and the no-GUI-domain refusal.
func TestParseLaunchctlPrintReviewRegressions(t *testing.T) {
	// launchctl prints data descriptors without a trailing newline, so a
	// block's "}" lands at the end of a content line (networkserviceproxy).
	inlineClose := "gui/501/com.apple.networkserviceproxy = {\n\tstate = running\n\tpid = 812\n" +
		"\tevent triggers = {\n\t\tnew-handles => {\n\t\t\tdescriptor = {\n\t\t\t\t\"aux-data\" => {\n" +
		"\t\t\t\t\t\"NSPServerAuxilaryData\" => \t\t\t\t\t\"NSPProxyAgentManagerAuxilaryData\" => \t\t\t\t}\n" +
		"\t\t\t\t\"handles-array\" => [\n\t\t\t\t\t2 = {\n\t\t\t\t\t\t\"key-material\" => \t\t\t\t\t}\n" +
		"\t\t\t\t]\n\t\t\t}\n\t\t}\n\t}\n" +
		"\tproperties = supports transactions | supports pressured exit | system service | exponential throttling | tle system\n}\n"
	st, err := ParseLaunchctlPrint(inlineClose)
	if err != nil {
		t.Fatalf("inline close: %v", err)
	}
	if len(st.Properties) != 5 || st.State != "running" || st.PID != 812 {
		t.Fatalf("inline close: properties=%q state=%q pid=%d", st.Properties, st.State, st.PID)
	}

	// A running job whose own argv and exit text contain not-found phrases.
	mentions := "gui/501/com.fak.x = {\n\tstate = running\n\targuments = {\n\t\t/usr/local/bin/x\n\t\t--msg=service is not loaded\n\t}\n" +
		"\tpid = 4242\n\tlast exit code = 3: No such process\n}\n"
	st, err = ParseLaunchctlPrint(mentions)
	if err != nil {
		t.Fatalf("phrase in argv read as not found: %v", err)
	}
	if st.PID != 4242 || st.LastExitCode == nil || *st.LastExitCode != 3 || len(st.Arguments) != 2 {
		t.Fatalf("mentions: pid=%d exit=%v args=%q", st.PID, st.LastExitCode, st.Arguments)
	}

	// argv is copied verbatim: surrounding spaces, an empty entry, and a "}".
	verbatim := "gui/501/com.fak.v = {\n\tstate = running\n\targuments = {\n\t\t/usr/local/bin/x\n\t\t  spaced  \n\t\t\n\t\t}\n\t\t--after\n\t}\n\tpid = 7\n\tresource coalition = {\n\t\tstate = active\n\t}\n}\n"
	st, err = ParseLaunchctlPrint(verbatim)
	if err != nil {
		t.Fatalf("verbatim: %v", err)
	}
	want := []string{"/usr/local/bin/x", "  spaced  ", "", "}", "--after"}
	if strings.Join(st.Arguments, "|") != strings.Join(want, "|") || st.State != "running" || st.PID != 7 {
		t.Fatalf("verbatim: args=%q state=%q pid=%d", st.Arguments, st.State, st.PID)
	}

	// Truncated output must not pass as a (stale-looking) definition.
	for name, raw := range map[string]string{
		"mid-args":     "gui/501/com.fak.t = {\n\tstate = running\n\targuments = {\n\t\t/usr/local/bin/x\n\t\tup\n",
		"never-closed": "gui/501/com.fak.t = {\n\tstate = running\n\tpid = 9\n",
	} {
		if _, err := ParseLaunchctlPrint(raw); !errors.Is(err, ErrMalformedLaunchctlOutput) {
			t.Fatalf("truncated %s: err = %v, want ErrMalformedLaunchctlOutput", name, err)
		}
	}

	// A merged stderr preamble before the header.
	st, err = ParseLaunchctlPrint("launchctl: warning: something\ngui/501/com.fak.p = {\n\tstate = running\n\tpid = 11\n}\n")
	if err != nil || st.Target != "gui/501/com.fak.p" || st.State != "running" || st.PID != 11 {
		t.Fatalf("preamble: st=%+v err=%v", st, err)
	}

	// No GUI session for the uid (e.g. over SSH) is a not-found, not malformed.
	if _, err := ParseLaunchctlPrint("Bad request.\nCould not find domain for user gui: 99999\n"); !errors.Is(err, ErrLaunchctlServiceNotFound) {
		t.Fatalf("no gui domain: err = %v", err)
	}

	// A signal termination with no exit-code line is surfaced as the reason.
	st, err = ParseLaunchctlPrint("gui/501/com.fak.s = {\n\tstate = not running\n\tlast terminating signal = Terminated: 15\n}\n")
	if err != nil || st.LastExitCode != nil || st.ExitReason != "terminated by signal: Terminated: 15" {
		t.Fatalf("signal: st=%+v err=%v", st, err)
	}
}
