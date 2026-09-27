package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	fakeUpServiceUID   = 501
	fakeUpServiceHome  = "/Users/example"
	fakeUpServiceTgt   = "gui/501/com.fak.up"
	fakeUpServicePlist = "/Users/example/Library/LaunchAgents/com.fak.up.plist"
)

// The live #13535 shape: the plist on disk gained --gpu-idle-exit 0, but the job
// launchd loaded before that edit still runs without it.
var (
	fakeUpServicePlistArgs = []string{"/Users/example/.local/bin/fak-native", "up", "--headless", "--gpu-idle-exit", "0", "--model", "27B", "--context", "20480"}
	fakeUpServiceStaleArgs = []string{"/Users/example/.local/bin/fak-native", "up", "--headless", "--model", "27B", "--context", "20480"}
)

// fakeUpLaunchd is a scripted launchd + plutil + service for the verbs.
type fakeUpLaunchd struct {
	loaded     bool
	disabled   bool
	pid        int
	runs       int
	loadedArgs []string
	plistArgs  []string // nil: no plist on disk
	listening  bool
	// readyAfter is how many health probes return warming_up after a bootstrap;
	// negative never becomes ready.
	readyAfter int
	probes     int
	alive      map[int]bool
	calls      []string
	spawned    [][]string
	now        time.Time
}

func newFakeUpLaunchdRunning(loadedArgs []string) *fakeUpLaunchd {
	return &fakeUpLaunchd{
		loaded: true, pid: 74723, runs: 50, loadedArgs: loadedArgs,
		plistArgs: fakeUpServicePlistArgs, listening: true,
		alive: map[int]bool{74723: true},
		now:   time.Date(2026, 9, 25, 18, 0, 0, 0, time.UTC),
	}
}

func (f *fakeUpLaunchd) printOutput() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s = {\n\tactive count = 1\n\tpath = %s\n\ttype = LaunchAgent\n\tstate = running\n\n", fakeUpServiceTgt, fakeUpServicePlist)
	fmt.Fprintf(&b, "\tprogram = %s\n\targuments = {\n", f.loadedArgs[0])
	for _, a := range f.loadedArgs {
		fmt.Fprintf(&b, "\t\t%s\n", a)
	}
	b.WriteString("\t}\n\n")
	b.WriteString("\tenvironment = {\n\t\tXPC_SERVICE_NAME => com.fak.up\n\t}\n\n")
	fmt.Fprintf(&b, "\tdomain = gui/501 [100024]\n\truns = %d\n\tpid = %d\n\tlast exit code = 0\n\n", f.runs, f.pid)
	b.WriteString("\tresource coalition = {\n\t\tID = 4295\n\t\ttype = resource\n\t\tstate = active\n\t}\n\n")
	b.WriteString("\tjetsam coalition = {\n\t\tID = 4296\n\t\ttype = jetsam\n\t\tstate = active\n\t}\n\n")
	b.WriteString("\tproperties = keepalive | runatload | inferred program\n}\n")
	return b.String()
}

func (f *fakeUpLaunchd) run(_ context.Context, name string, args ...string) ([]byte, error) {
	exitErr := func(code int) error { return fmt.Errorf("exit status %d", code) }
	switch name {
	case "/usr/bin/plutil":
		if f.plistArgs == nil || len(args) < 2 || args[0] != "-extract" {
			return []byte("unexpected plutil " + strings.Join(args, " ")), exitErr(1)
		}
		if args[1] != "ProgramArguments" {
			return []byte("No value at that key path or invalid key path: " + args[1]), exitErr(1)
		}
		raw, _ := json.Marshal(f.plistArgs)
		return raw, nil
	case "/bin/launchctl":
	default:
		return nil, fmt.Errorf("unexpected tool %s", name)
	}
	verb := args[0]
	if verb != "print" && verb != "print-disabled" {
		f.calls = append(f.calls, "launchctl "+strings.Join(args, " "))
	}
	switch verb {
	case "print":
		if !f.loaded {
			return []byte("Bad request.\nCould not find service \"com.fak.up\" in domain for user gui: 501\n"), exitErr(113)
		}
		return []byte(f.printOutput()), nil
	case "print-disabled":
		state := "enabled"
		if f.disabled {
			state = "disabled"
		}
		return []byte("disabled services = {\n\t\t\"com.fak.keep-awake\" => enabled\n\t\t\"com.fak.up\" => " + state + "\n}\n"), nil
	case "enable":
		f.disabled = false
		return nil, nil
	case "disable":
		f.disabled = true
		return nil, nil
	case "bootout":
		if !f.loaded {
			return []byte("Boot-out failed: 3: No such process\n"), exitErr(3)
		}
		f.loaded, f.listening = false, false
		f.alive[f.pid] = false
		return nil, nil
	case "bootstrap":
		if f.loaded || f.disabled {
			return []byte("Bootstrap failed: 5: Input/output error\n"), exitErr(5)
		}
		f.loaded, f.listening = true, true
		f.pid++
		f.runs++
		f.alive[f.pid] = true
		f.loadedArgs = slices.Clone(f.plistArgs)
		f.probes = 0
		return nil, nil
	}
	return nil, fmt.Errorf("unexpected launchctl verb %q", verb)
}

func (f *fakeUpLaunchd) deps(t *testing.T, stateDir string) upServiceDeps {
	t.Helper()
	return upServiceDeps{
		goos: "darwin",
		uid:  func() int { return fakeUpServiceUID },
		home: func() (string, error) { return fakeUpServiceHome, nil },
		stat: func(path string) error {
			if f.plistArgs == nil {
				return &os.PathError{Op: "stat", Path: path, Err: os.ErrNotExist}
			}
			return nil
		},
		stateDir: func() (string, error) { return stateDir, nil },
		run:      f.run,
		httpGet: func(_ context.Context, url string) (int, []byte, error) {
			if url != "http://127.0.0.1:8080/healthz" {
				t.Errorf("health probe url = %q", url)
			}
			f.probes++
			if f.readyAfter < 0 || f.probes <= f.readyAfter {
				return 200, []byte(`{"ok":false,"status":"warming_up"}`), nil
			}
			return 200, []byte(`{"ok":true,"status":"ok"}`), nil
		},
		listening:  func(string) bool { return f.listening },
		alive:      func(pid int) bool { return f.alive[pid] },
		lease:      func() upServiceLease { return upServiceLease{Held: f.loaded, PID: f.pid} },
		now:        func() time.Time { return f.now },
		sleep:      func(d time.Duration) { f.now = f.now.Add(d) },
		executable: func() (string, error) { return "/tmp/fak-test-bin", nil },
		spawnLapse: func(exe string, args []string, logPath string) (int, error) {
			f.spawned = append(f.spawned, append([]string{exe}, args...))
			if filepath.Dir(logPath) != stateDir {
				t.Errorf("lapse log %q not under state dir %q", logPath, stateDir)
			}
			return 4242, nil
		},
	}
}

func runUpServiceJSON(t *testing.T, deps upServiceDeps, argv ...string) (upServiceStatus, int, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	rc := runUpService(&stdout, &stderr, append(argv, "--json"), deps)
	var st upServiceStatus
	if err := json.Unmarshal(stdout.Bytes(), &st); err != nil {
		t.Fatalf("%v: stdout is not a status record: %v\nstdout=%s\nstderr=%s", argv, err, stdout.String(), stderr.String())
	}
	return st, rc, stderr.String()
}

// TestUpServiceStatusFlagsStaleDefinition reproduces the #13535 root cause:
// launchd keeps running the definition from the last bootstrap, so a plist edit
// to --gpu-idle-exit 0 never reached the process. status must name the drift.
func TestUpServiceStatusFlagsStaleDefinition(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServiceStaleArgs)
	st, rc, _ := runUpServiceJSON(t, f.deps(t, t.TempDir()), "status")
	if st.Verdict != upServiceVerdictStale || rc != 1 {
		t.Fatalf("verdict=%s rc=%d, want %s rc=1 (%+v)", st.Verdict, rc, upServiceVerdictStale, st)
	}
	if !st.Loaded || st.PID != 74723 || st.Runs != 50 || st.State != "running" {
		t.Fatalf("launchd facts not parsed from the live print shape: %+v", st)
	}
	if !slices.Equal(st.LoadedArgs, fakeUpServiceStaleArgs) || !slices.Equal(st.PlistArgs, fakeUpServicePlistArgs) {
		t.Fatalf("loaded=%v plist=%v", st.LoadedArgs, st.PlistArgs)
	}
	if !strings.Contains(st.Next, "fak up on") {
		t.Fatalf("next step should name the repair verb: %q", st.Next)
	}
	if len(f.calls) != 0 {
		t.Fatalf("status must not change launchd state, ran %v", f.calls)
	}
}

func TestUpServiceStatusOnWhenFreshAndHealthy(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	st, rc, _ := runUpServiceJSON(t, f.deps(t, t.TempDir()), "status")
	if st.Verdict != upServiceVerdictOn || rc != 0 || st.Stale || !st.HealthReady || st.HealthStatus != "ok" {
		t.Fatalf("verdict=%s rc=%d %+v", st.Verdict, rc, st)
	}
	if st.Disabled {
		t.Fatalf("enabled job reported disabled")
	}
	if !st.Lease.Held || st.Lease.PID != st.PID {
		t.Fatalf("lease %+v should be held by the service pid %d", st.Lease, st.PID)
	}
}

func TestUpServiceOffStopsDisablesAndMarks(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	dir := t.TempDir()
	st, rc, stderr := runUpServiceJSON(t, f.deps(t, dir), "off")
	if rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("rc=%d verdict=%s stderr=%s %+v", rc, st.Verdict, stderr, st)
	}
	wantCalls := []string{"launchctl disable " + fakeUpServiceTgt, "launchctl bootout " + fakeUpServiceTgt}
	if !slices.Equal(f.calls, wantCalls) {
		t.Fatalf("calls = %v, want %v", f.calls, wantCalls)
	}
	if f.loaded || !f.disabled || f.listening || st.Listening || st.Lease.Held {
		t.Fatalf("service still up after off: loaded=%v disabled=%v listening=%v lease=%+v", f.loaded, f.disabled, f.listening, st.Lease)
	}
	m, err := readUpDevOffMarker("com.fak.up", f.deps(t, dir))
	if err != nil {
		t.Fatalf("dev-off marker not persisted: %v", err)
	}
	if !m.Until.IsZero() || m.Token == "" || len(f.spawned) != 0 {
		t.Fatalf("indefinite off must have no lapse window or waker: %+v spawned=%v", m, f.spawned)
	}
	// Persisted: a fresh status still reads OFF (not STOPPED) from the marker.
	st2, rc2, _ := runUpServiceJSON(t, f.deps(t, dir), "status")
	if st2.Verdict != upServiceVerdictOff || rc2 != 0 || !st2.Disabled || st2.DevOff == nil {
		t.Fatalf("status after off: verdict=%s rc=%d %+v", st2.Verdict, rc2, st2)
	}
}

func TestUpServiceOffIsIdempotentWhenAlreadyStopped(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	f.loaded, f.listening = false, false
	st, rc, _ := runUpServiceJSON(t, f.deps(t, t.TempDir()), "off")
	if rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("rc=%d verdict=%s", rc, st.Verdict)
	}
	if slices.Contains(f.calls, "launchctl bootout "+fakeUpServiceTgt) {
		t.Fatalf("bootout of an unloaded job: %v", f.calls)
	}
}

func TestUpServiceOffForSpawnsBoundLapseWaker(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	dir := t.TempDir()
	start := f.now
	st, rc, _ := runUpServiceJSON(t, f.deps(t, dir), "off", "--for", "30m")
	if rc != 0 || st.Verdict != upServiceVerdictOff {
		t.Fatalf("rc=%d verdict=%s", rc, st.Verdict)
	}
	m, err := readUpDevOffMarker("com.fak.up", f.deps(t, dir))
	if err != nil {
		t.Fatal(err)
	}
	if !m.Until.Equal(start.Add(30 * time.Minute)) {
		t.Fatalf("until = %s, want %s", m.Until, start.Add(30*time.Minute))
	}
	if m.WakerPID != 4242 || len(f.spawned) != 1 {
		t.Fatalf("waker pid %d spawned=%v", m.WakerPID, f.spawned)
	}
	want := []string{"/tmp/fak-test-bin", "up", "on", "--label", "com.fak.up", "--plist", fakeUpServicePlist,
		"--lapse-token", m.Token, "--lapse-at", strconv.FormatInt(m.Until.Unix(), 10)}
	if !slices.Equal(f.spawned[0], want) {
		t.Fatalf("waker argv = %v\nwant %v", f.spawned[0], want)
	}
}

func TestUpServiceOnRestoresAndClearsMarker(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	dir := t.TempDir()
	deps := f.deps(t, dir)
	if _, rc, _ := runUpServiceJSON(t, deps, "off"); rc != 0 {
		t.Fatalf("off rc=%d", rc)
	}
	f.calls = nil
	f.readyAfter = 3 // model load: /healthz answers warming_up first
	st, rc, stderr := runUpServiceJSON(t, deps, "on")
	if rc != 0 || st.Verdict != upServiceVerdictOn {
		t.Fatalf("rc=%d verdict=%s stderr=%s %+v", rc, st.Verdict, stderr, st)
	}
	wantCalls := []string{"launchctl enable " + fakeUpServiceTgt, "launchctl bootstrap gui/501 " + fakeUpServicePlist}
	if !slices.Equal(f.calls, wantCalls) {
		t.Fatalf("calls = %v, want %v", f.calls, wantCalls)
	}
	if !f.loaded || f.disabled || !st.HealthReady || st.HealthCode != 200 {
		t.Fatalf("not restored: loaded=%v disabled=%v %+v", f.loaded, f.disabled, st)
	}
	if _, err := readUpDevOffMarker("com.fak.up", deps); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("dev-off marker should be cleared, got %v", err)
	}
	if f.probes <= 3 {
		t.Fatalf("on returned before /healthz reported ready (probes=%d)", f.probes)
	}
}

// TestUpServiceOnReloadsStaleDefinition pins the repair: `on` re-bootstraps a
// loaded job whose cached definition differs from the plist, so the plist's
// --gpu-idle-exit 0 finally reaches the process.
func TestUpServiceOnReloadsStaleDefinition(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServiceStaleArgs)
	st, rc, stderr := runUpServiceJSON(t, f.deps(t, t.TempDir()), "on")
	if rc != 0 || st.Verdict != upServiceVerdictOn {
		t.Fatalf("rc=%d verdict=%s stderr=%s", rc, st.Verdict, stderr)
	}
	wantCalls := []string{"launchctl enable " + fakeUpServiceTgt, "launchctl bootout " + fakeUpServiceTgt, "launchctl bootstrap gui/501 " + fakeUpServicePlist}
	if !slices.Equal(f.calls, wantCalls) {
		t.Fatalf("calls = %v, want %v", f.calls, wantCalls)
	}
	if !slices.Equal(f.loadedArgs, fakeUpServicePlistArgs) || st.Stale {
		t.Fatalf("loaded definition still stale: %v", f.loadedArgs)
	}
}

func TestUpServiceOnLeavesFreshRunningJobAlone(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	st, rc, _ := runUpServiceJSON(t, f.deps(t, t.TempDir()), "on")
	if rc != 0 || st.Verdict != upServiceVerdictOn {
		t.Fatalf("rc=%d verdict=%s", rc, st.Verdict)
	}
	if !slices.Equal(f.calls, []string{"launchctl enable " + fakeUpServiceTgt}) || f.runs != 50 {
		t.Fatalf("a fresh healthy job must not be restarted: calls=%v runs=%d", f.calls, f.runs)
	}
}

func TestUpServiceOnTimesOutNotReady(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	f.loaded, f.listening = false, false
	f.readyAfter = -1
	st, rc, _ := runUpServiceJSON(t, f.deps(t, t.TempDir()), "on", "--wait", "10s")
	if rc != 1 || st.Verdict != upServiceVerdictNotReady || !st.Loaded {
		t.Fatalf("rc=%d verdict=%s %+v", rc, st.Verdict, st)
	}
}

func TestUpServiceOnRefusesWithoutPlist(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	f.loaded, f.plistArgs = false, nil
	st, rc, _ := runUpServiceJSON(t, f.deps(t, t.TempDir()), "on")
	if rc != 1 || st.Verdict != upServiceVerdictNotInstalled || len(f.calls) != 0 {
		t.Fatalf("rc=%d verdict=%s calls=%v", rc, st.Verdict, f.calls)
	}
}

func TestUpServiceStatusStoppedAndLapsed(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	f.loaded, f.listening = false, false
	dir := t.TempDir()
	deps := f.deps(t, dir)
	st, rc, _ := runUpServiceJSON(t, deps, "status")
	if st.Verdict != upServiceVerdictStopped || rc != 1 {
		t.Fatalf("no marker: verdict=%s rc=%d", st.Verdict, rc)
	}
	m := &upDevOffMarker{Schema: upDevOffMarkerSchema, Label: "com.fak.up", Since: f.now.Add(-2 * time.Hour), Until: f.now.Add(-time.Hour), Token: "t"}
	if err := writeUpDevOffMarker(m, deps); err != nil {
		t.Fatal(err)
	}
	st, rc, _ = runUpServiceJSON(t, deps, "status")
	if st.Verdict != upServiceVerdictOffLapsed || rc != 1 || !strings.Contains(st.Next, "fak up on") {
		t.Fatalf("lapsed marker: verdict=%s rc=%d next=%q", st.Verdict, rc, st.Next)
	}
}

func TestUpServiceLapseWakerRestoresOnlyItsOwnWindow(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	dir := t.TempDir()
	deps := f.deps(t, dir)
	if _, rc, _ := runUpServiceJSON(t, deps, "off", "--for", "90s"); rc != 0 {
		t.Fatalf("off rc=%d", rc)
	}
	m, err := readUpDevOffMarker("com.fak.up", deps)
	if err != nil {
		t.Fatal(err)
	}
	at := strconv.FormatInt(m.Until.Unix(), 10)

	// A waker bound to a different `off` is a no-op.
	f.calls = nil
	var stdout, stderr bytes.Buffer
	if rc := runUpService(&stdout, &stderr, []string{"on", "--lapse-token", "someone-else", "--lapse-at", at}, deps); rc != 0 || len(f.calls) != 0 {
		t.Fatalf("stale waker acted: rc=%d calls=%v", rc, f.calls)
	}

	// The bound waker sleeps on the wall clock until the window lapses, then restores.
	start := f.now
	stdout.Reset()
	stderr.Reset()
	if rc := runUpService(&stdout, &stderr, []string{"on", "--lapse-token", m.Token, "--lapse-at", at}, deps); rc != 0 {
		t.Fatalf("bound waker rc=%d stderr=%s", rc, stderr.String())
	}
	if f.now.Before(m.Until) || f.now.Sub(start) > 2*time.Minute {
		t.Fatalf("waker restored at %s, window lapsed at %s", f.now, m.Until)
	}
	if !f.loaded || f.disabled {
		t.Fatalf("waker did not restore: loaded=%v disabled=%v calls=%v", f.loaded, f.disabled, f.calls)
	}
}

func TestUpServiceNotSupportedOffDarwin(t *testing.T) {
	for _, goos := range []string{"linux", "windows"} {
		for _, verb := range upServiceVerbs {
			f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
			deps := f.deps(t, t.TempDir())
			deps.goos = goos
			deps.run = func(context.Context, string, ...string) ([]byte, error) {
				t.Fatalf("%s %s ran a tool", goos, verb)
				return nil, nil
			}
			var stdout, stderr bytes.Buffer
			if rc := runUpService(&stdout, &stderr, []string{verb}, deps); rc != 2 || !strings.Contains(stderr.String(), upServiceVerdictNotSupported) {
				t.Fatalf("%s %s: rc=%d stderr=%q", goos, verb, rc, stderr.String())
			}
			st, rc, _ := runUpServiceJSON(t, deps, verb)
			if rc != 2 || st.Verdict != upServiceVerdictNotSupported {
				t.Fatalf("%s %s --json: rc=%d verdict=%s", goos, verb, rc, st.Verdict)
			}
		}
	}
}

func TestIsUpServiceVerbPeelsOnlyArgv0(t *testing.T) {
	for _, tc := range []struct {
		argv []string
		want bool
	}{
		{[]string{"status"}, true},
		{[]string{"off", "--for", "1h"}, true},
		{[]string{"on"}, true},
		{nil, false},
		{[]string{"--headless", "--model", "27B"}, false},
		{[]string{"--model", "on"}, false},
		{[]string{"help"}, false},
	} {
		if got := isUpServiceVerb(tc.argv); got != tc.want {
			t.Errorf("isUpServiceVerb(%v) = %v, want %v", tc.argv, got, tc.want)
		}
	}
}

func TestUpServiceHelpExitsZero(t *testing.T) {
	f := newFakeUpLaunchdRunning(fakeUpServicePlistArgs)
	var stdout, stderr bytes.Buffer
	if rc := runUpService(&stdout, &stderr, []string{"off", "--help"}, f.deps(t, t.TempDir())); rc != 0 || !strings.Contains(stderr.String(), "fak up off [--for <dur>]") {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	if len(f.calls) != 0 {
		t.Fatalf("--help ran launchctl: %v", f.calls)
	}
	var help bytes.Buffer
	printUpHelp(&help)
	if !strings.Contains(help.String(), "fak up status") {
		t.Fatalf("fak up --help does not list the service verbs:\n%s", help.String())
	}
}

func TestUpServiceAddrFromArgs(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want string
	}{
		{fakeUpServicePlistArgs, "127.0.0.1:8080"},
		{[]string{"fak", "up", "--addr", "127.0.0.1:9090"}, "127.0.0.1:9090"},
		{[]string{"fak", "up", "--addr=:8181"}, "127.0.0.1:8181"},
		{[]string{"fak", "up", "-addr", "0.0.0.0:8282"}, "127.0.0.1:8282"},
	} {
		if got := upServiceAddrFromArgs(tc.args); got != tc.want {
			t.Errorf("upServiceAddrFromArgs(%v) = %q, want %q", tc.args, got, tc.want)
		}
	}
}

func TestParseUpHealthBody(t *testing.T) {
	for _, tc := range []struct {
		code   int
		body   string
		ready  bool
		status string
	}{
		{200, `{"ok":true,"status":"ok"}`, true, "ok"},
		{200, `{"ok":false,"status":"warming_up"}`, false, "warming_up"},
		{200, `{"ok":false,"status":"stopping"}`, false, "stopping"},
		{200, `plain ok`, true, ""},
		{503, `{"ok":true}`, false, ""},
	} {
		ready, status := parseUpHealthBody(tc.code, []byte(tc.body))
		if ready != tc.ready || status != tc.status {
			t.Errorf("parseUpHealthBody(%d, %s) = %v %q, want %v %q", tc.code, tc.body, ready, status, tc.ready, tc.status)
		}
	}
}
