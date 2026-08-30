package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

func TestLaunchDefaultAndDisablePersist(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(dir, "launch.json"))
	command := fakeLaunchProvider(t, dir)
	if err := launchshim.Save(launchshim.Config{Providers: map[string]launchshim.Provider{"claude": {Command: command}}}); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runLaunchDefault(&out, &errOut, []string{"claude"}); code != 0 {
		t.Fatalf("default code=%d stderr=%s", code, errOut.String())
	}
	if code := runLaunchToggle(&out, &errOut, true); code != 0 {
		t.Fatalf("disable code=%d stderr=%s", code, errOut.String())
	}
	c, err := launchshim.Load()
	if err != nil {
		t.Fatal(err)
	}
	if c.Default != "claude" || !c.Disabled {
		t.Fatalf("config=%+v", c)
	}
}

func TestWriteLaunchShimForwardsDirectEscape(t *testing.T) {
	dir := t.TempDir()
	if err := writeLaunchShim(dir, "claude", filepath.Join(dir, "fak binary")); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, shimName("claude")))
	if err != nil {
		t.Fatal(err)
	}
	text := string(b)
	for _, want := range []string{"launch claude --", "fak binary"} {
		if !strings.Contains(text, want) {
			t.Fatalf("shim %q missing %q", text, want)
		}
	}
	if runtime.GOOS != "windows" {
		if info, _ := os.Stat(filepath.Join(dir, "claude")); info.Mode()&0o111 == 0 {
			t.Fatal("unix shim is not executable")
		}
	}
}

func TestLaunchDirectRunsRecordedProvider(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("native test executables are blocked on this host; covered under WSL/CI")
	}
	dir := t.TempDir()
	command := filepath.Join(dir, "provider")
	output := filepath.Join(dir, "args")
	if err := os.WriteFile(command, []byte("#!/bin/sh\nprintf '%s' \"$*\" > \"$OUT\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OUT", output)
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(dir, "launch.json"))
	if err := launchshim.Save(launchshim.Config{Default: "claude", Providers: map[string]launchshim.Provider{"claude": {Command: command}}}); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runLaunch(&out, &errOut, []string{"--direct", "claude", "hello", "world"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	b, _ := os.ReadFile(output)
	if string(b) != "hello world" {
		t.Fatalf("args=%q", b)
	}
}

func TestLaunchCustomAliasLifecycleAndRedactedList(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(dir, "launch.json"))
	t.Setenv("FAK_LAUNCH_BIN", filepath.Join(dir, "bin"))
	command := fakeLaunchProvider(t, dir)
	var out, errOut bytes.Buffer
	args := []string{"add", "--command", command, "--arg", "space value", "--arg=--leading", "--default", "--shim", "third-provider"}
	if code := runLaunch(&out, &errOut, args); code != 0 {
		t.Fatalf("add code=%d stderr=%s", code, errOut.String())
	}
	c, err := launchshim.Load()
	if err != nil {
		t.Fatal(err)
	}
	p := c.Providers["third-provider"]
	if c.Default != "third-provider" || !p.InstallShim || !reflect.DeepEqual(p.Args, []string{"space value", "--leading"}) {
		t.Fatalf("config=%+v", c)
	}
	if _, err := os.Stat(filepath.Join(dir, "bin", shimName("third-provider"))); err != nil {
		t.Fatalf("shim: %v", err)
	}
	out.Reset()
	errOut.Reset()
	if code := runLaunchList(&out, &errOut, []string{"--json"}); code != 0 {
		t.Fatalf("list code=%d stderr=%s", code, errOut.String())
	}
	if strings.Contains(out.String(), command) || !strings.Contains(out.String(), `"name": "third-provider"`) || !strings.Contains(out.String(), `"template_args": 2`) {
		t.Fatalf("list=%s", out.String())
	}
	out.Reset()
	errOut.Reset()
	if code := runLaunchRemove(&out, &errOut, []string{"third-provider"}); code != 0 {
		t.Fatalf("remove code=%d stderr=%s", code, errOut.String())
	}
	c, _ = launchshim.Load()
	if _, ok := c.Providers["third-provider"]; ok || c.Default != "" {
		t.Fatalf("after remove=%+v", c)
	}
	if _, err := os.Stat(filepath.Join(dir, "bin", shimName("third-provider"))); !os.IsNotExist(err) {
		t.Fatalf("shim remains: %v", err)
	}
}

func TestLaunchAliasRejectsReservedAndPathLikeNames(t *testing.T) {
	for _, name := range []string{"serve", "fak", "../other", "two words"} {
		if _, err := validateLaunchAlias(name); err == nil {
			t.Errorf("alias %q accepted", name)
		}
	}
}

func fakeLaunchProvider(t *testing.T, dir string) string {
	t.Helper()
	name := "provider"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("provider"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLaunchCustomTemplatePreservesArgvBoundaries(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(dir, "launch.json"))
	command := fakeLaunchProvider(t, dir)
	config := launchshim.Config{Default: "third", Providers: map[string]launchshim.Provider{"third": {Command: command, Args: []string{"space value", "--leading", `quote"value`, "\u03bb"}}}}
	if err := launchshim.Save(config); err != nil {
		t.Fatal(err)
	}
	old := launchChildRunner
	defer func() { launchChildRunner = old }()
	var gotCommand string
	var gotArgs []string
	launchChildRunner = func(_ io.Reader, _ io.Writer, _ io.Writer, command string, args []string) int {
		gotCommand = command
		gotArgs = append([]string(nil), args...)
		return 0
	}
	var out, errOut bytes.Buffer
	if code := runLaunch(&out, &errOut, []string{"--direct", "third", "user value", ";touch never"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	want := []string{"space value", "--leading", `quote"value`, "\u03bb", "user value", ";touch never"}
	if gotCommand != command || !reflect.DeepEqual(gotArgs, want) {
		t.Fatalf("command=%q argv=%q want %q", gotCommand, gotArgs, want)
	}
}

// TestLaunchInteractiveProviderSuppressesStartupWall is the front-door render
// witness: launch must carry the attended-child fact into guard explicitly. A
// shim can make stdin look piped, so relying on guard's auto probe regresses to
// the full startup report before the provider paints its UI.
func TestLaunchInteractiveProviderSuppressesStartupWall(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(dir, "launch.json"))
	command := fakeLaunchProvider(t, dir)
	if err := launchshim.Save(launchshim.Config{Default: "claude", Providers: map[string]launchshim.Provider{"claude": {Command: command}}}); err != nil {
		t.Fatal(err)
	}
	old := launchChildRunner
	defer func() { launchChildRunner = old }()
	var got []string
	launchChildRunner = func(_ io.Reader, _ io.Writer, _ io.Writer, _ string, args []string) int {
		got = append([]string(nil), args...)
		return 0
	}
	var out, errOut bytes.Buffer
	if code := runLaunch(&out, &errOut, nil); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	want := []string{"guard", "--banner=animate", "--", command}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("guard argv=%q want %q", got, want)
	}
}
func TestStableLaunchTargetSurvivesSourceReplacement(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "install", "fak.exe")
	if err := os.MkdirAll(filepath.Dir(source), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("v1"), 0o755); err != nil {
		t.Fatal(err)
	}
	shimDir := filepath.Join(dir, "shims")
	if err := os.MkdirAll(shimDir, 0o755); err != nil {
		t.Fatal(err)
	}
	target, err := installStableLaunchTarget(shimDir, source)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeLaunchShim(shimDir, "third", target); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("v2"), 0o755); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(target)
	if string(got) != "v1" {
		t.Fatalf("stable target changed with source: %q", got)
	}
	bound, err := launchshim.StableExecutable(target)
	if err != nil || !samePath(bound, source) {
		t.Fatalf("stable executable binding=%q err=%v, want %q", bound, err, source)
	}
	shim, _ := os.ReadFile(filepath.Join(shimDir, shimName("third")))
	if !strings.Contains(string(shim), target) || strings.Contains(string(shim), source) {
		t.Fatalf("shim=%q target=%q source=%q", shim, target, source)
	}
	if _, err := installStableLaunchTarget(shimDir, source); err != nil {
		t.Fatal(err)
	}
	got, _ = os.ReadFile(target)
	if string(got) != "v2" {
		t.Fatalf("repair target=%q", got)
	}
}

func TestRepairLaunchShimsRefreshesStableTarget(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(dir, "launch.json"))
	t.Setenv("FAK_LAUNCH_BIN", filepath.Join(dir, "bin"))
	provider := fakeLaunchProvider(t, dir)
	if err := launchshim.Save(launchshim.Config{Providers: map[string]launchshim.Provider{"third": {Command: provider, InstallShim: true}}}); err != nil {
		t.Fatal(err)
	}
	if err := repairLaunchShims(); err != nil {
		t.Fatal(err)
	}
	target := stableLaunchTarget(filepath.Join(dir, "bin"), mustExecutable(t))
	if _, err := os.Stat(target); err != nil {
		t.Fatalf("stable target: %v", err)
	}
	shim, err := os.ReadFile(filepath.Join(dir, "bin", shimName("third")))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(shim), target) {
		t.Fatalf("shim=%q target=%q", shim, target)
	}
}

func writeInstalledProviderFixture(t *testing.T, path string) {
	t.Helper()
	body := "#!/bin/sh\nexit 0\n"
	if runtime.GOOS == "windows" {
		body = "@echo off\r\nexit /b 0\r\n"
	}
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
}
func TestLaunchInstalledLifecycleManagedCodexShim(t *testing.T) {
	dir := t.TempDir()
	home := filepath.Join(dir, "home")
	binDir := filepath.Join(dir, "bin")
	configPath := filepath.Join(dir, "config", "launch.json")
	providerDir := filepath.Join(dir, "provider")
	providerName := "codex"
	providerPath := filepath.Join(providerDir, providerName)
	if runtime.GOOS == "windows" {
		providerPath += ".cmd"
	}
	if err := os.MkdirAll(home, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(providerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeInstalledProviderFixture(t, providerPath)
	t.Setenv("FAK_LAUNCH_CONFIG", configPath)
	t.Setenv("FAK_LAUNCH_BIN", binDir)
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("FAK_TEST_LAUNCH_CHILD", "")
	originalPath := os.Getenv("PATH")
	t.Setenv("PATH", providerDir+string(os.PathListSeparator)+originalPath)

	oldExecutable := osExecutableForLaunch
	oldRunner := launchChildRunner
	oldResolve := stableLaunchResolve
	t.Cleanup(func() {
		osExecutableForLaunch = oldExecutable
		launchChildRunner = oldRunner
		stableLaunchResolve = oldResolve
	})
	osExecutableForLaunch = func() (string, error) { return mustExecutable(t), nil }

	var out, errOut bytes.Buffer
	if code := runLaunch(&out, &errOut, []string{"install", "--provider", "codex", "--default", "codex", "--no-path"}); code != 0 {
		t.Fatalf("install code=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+originalPath)
	stable := stableLaunchTarget(binDir, mustExecutable(t))
	if _, err := os.Stat(stable); err != nil {
		t.Fatalf("stable target: %v", err)
	}
	shimPath := filepath.Join(binDir, shimName("codex"))
	if _, err := os.Stat(shimPath); err != nil {
		t.Fatalf("shim: %v", err)
	}
	cfg, err := launchshim.Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Default != "codex" {
		t.Fatalf("default=%q", cfg.Default)
	}
	if got := cfg.Providers["codex"].Command; got != providerPath {
		t.Fatalf("provider command=%q want %q", got, providerPath)
	}

	shimBody, err := os.ReadFile(shimPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(shimBody), stable) {
		t.Fatalf("shim=%q stable=%q", shimBody, stable)
	}

	statusOut, statusErr := &bytes.Buffer{}, &bytes.Buffer{}
	if code := runLaunch(statusOut, statusErr, []string{"status"}); code != 0 {
		t.Fatalf("status code=%d stdout=%s stderr=%s", code, statusOut.String(), statusErr.String())
	}
	for _, want := range []string{"default: codex", "interception: active", "codex: " + providerPath} {
		if !strings.Contains(statusOut.String(), want) {
			t.Fatalf("status missing %q: %s", want, statusOut.String())
		}
	}

	doctorOut, doctorErr := &bytes.Buffer{}, &bytes.Buffer{}
	doctorCode := runLaunchDoctor(doctorOut, doctorErr, []string{})
	if doctorCode != 0 && doctorCode != 1 {
		t.Fatalf("doctor code=%d stdout=%s stderr=%s", doctorCode, doctorOut.String(), doctorErr.String())
	}
	for _, want := range []string{"LAUNCH DOCTOR", "default=codex", "codex  READY", "role=canonical    ready=true  reason=READY"} {
		if !strings.Contains(doctorOut.String(), want) {
			t.Fatalf("doctor missing %q: %s", want, doctorOut.String())
		}
	}

	stableLaunchResolve = func(target, policy string, wait time.Duration) (string, error) {
		if target != stable || policy != launchshim.UpdatePolicyPrior {
			t.Fatalf("resolve target=%q policy=%q", target, policy)
		}
		return stable, nil
	}
	childCalls := 0
	var childCommand string
	var childArgs []string
	launchChildRunner = func(_ io.Reader, stdout, _ io.Writer, command string, args []string) int {
		childCalls++
		childCommand = command
		childArgs = append([]string(nil), args...)
		fmt.Fprintln(stdout, "codex fixture 1.0")
		return 0
	}
	out.Reset()
	errOut.Reset()
	if code := runLaunch(&out, &errOut, []string{"codex", "--version"}); code != 0 {
		t.Fatalf("child code=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
	if childCalls != 1 || !samePath(childCommand, mustExecutable(t)) {
		t.Fatalf("child calls=%d command=%q want deployed fak %q", childCalls, childCommand, mustExecutable(t))
	}
	wantArgs := []string{"guard", "--banner=animate", "--", providerPath, "--version"}
	if !reflect.DeepEqual(childArgs, wantArgs) {
		t.Fatalf("child args=%q want %q", childArgs, wantArgs)
	}
	if samePath(childCommand, shimPath) || !strings.Contains(out.String(), "codex fixture 1.0") {
		t.Fatalf("recursive or missing fixture command=%q shim=%q stdout=%q", childCommand, shimPath, out.String())
	}

	out.Reset()
	errOut.Reset()
	if code := runLaunch(&out, &errOut, []string{"uninstall", "--provider", "codex"}); code != 0 {
		t.Fatalf("uninstall code=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
	if _, err := os.Stat(shimPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("shim still present err=%v", err)
	}
	cfg, err = launchshim.Load()
	if err != nil {
		t.Fatalf("load post-uninstall config: %v", err)
	}
	if len(cfg.Providers) != 0 || cfg.Default != "" {
		t.Fatalf("provider state remains after uninstall: %+v", cfg)
	}
}

func mustExecutable(t *testing.T) string {
	t.Helper()
	path, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLaunchCorruptConfigFailsClosedWithoutOverwrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "launch.json")
	t.Setenv("FAK_LAUNCH_CONFIG", path)
	want := []byte(`{"broken":`)
	if err := os.WriteFile(path, want, 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runLaunch(&out, &errOut, []string{"--direct", "claude"}); code == 0 {
		t.Fatalf("corrupt config accepted")
	}
	got, _ := os.ReadFile(path)
	if !bytes.Equal(got, want) {
		t.Fatalf("corrupt config overwritten: %q", got)
	}
	if !strings.Contains(errOut.String(), "parse") {
		t.Fatalf("stderr=%q", errOut.String())
	}
}

func TestLaunchBypassScopesAndExitPropagation(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(dir, "launch.json"))
	command := fakeLaunchProvider(t, dir)
	if err := launchshim.Save(launchshim.Config{Default: "third", Providers: map[string]launchshim.Provider{"third": {Command: command}}}); err != nil {
		t.Fatal(err)
	}
	old := launchChildRunner
	defer func() { launchChildRunner = old }()
	var gotCommand string
	var gotArgs []string
	launchChildRunner = func(_ io.Reader, _ io.Writer, _ io.Writer, command string, args []string) int {
		gotCommand, gotArgs = command, append([]string(nil), args...)
		return 23
	}
	for _, tc := range []struct {
		name     string
		env      bool
		disabled bool
		argv     []string
	}{
		{"explicit", false, false, []string{"--direct", "third", "unicode-value"}},
		{"shim-token", false, false, []string{"third", "--fak-direct", `quote"value`}},
		{"environment", true, false, []string{"third", "env"}},
		{"persisted", false, true, []string{"third", "disabled"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_DIRECT", "")
			c, _ := launchshim.Load()
			c.Disabled = tc.disabled
			if err := launchshim.Save(c); err != nil {
				t.Fatal(err)
			}
			if tc.env {
				t.Setenv("FAK_DIRECT", "1")
			}
			var out, errOut bytes.Buffer
			if code := runLaunch(&out, &errOut, tc.argv); code != 23 {
				t.Fatalf("exit=%d stderr=%s", code, errOut.String())
			}
			if gotCommand != command || len(gotArgs) != 1 {
				t.Fatalf("command=%q args=%q", gotCommand, gotArgs)
			}
		})
	}
}

func TestLaunchStatusEmptyConfigIsInactiveAndRejectsProviderArg(t *testing.T) {
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(t.TempDir(), "missing.json"))
	var out, errb bytes.Buffer
	if code := runLaunch(&out, &errb, []string{"status"}); code != 0 {
		t.Fatalf("status code=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{"default: (unset)", "interception: inactive (no configured providers)", "build: unknown"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("status missing %q: %s", want, out.String())
		}
	}
	out.Reset()
	errb.Reset()
	if code := runLaunch(&out, &errb, []string{"status", "codex"}); code != 2 || !strings.Contains(errb.String(), `unexpected argument "codex"`) {
		t.Fatalf("provider arg code=%d stdout=%s stderr=%s", code, out.String(), errb.String())
	}
}

func TestStableLaunchUpdatePoliciesPreserveProcessContract(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(dir, "launch.json"))
	t.Setenv("FAK_UPDATE_LAUNCH_POLICY", "")
	t.Setenv("FAK_UPDATE_LAUNCH_WAIT", "")
	target := filepath.Join(dir, "install", "fak")
	if runtime.GOOS == "windows" {
		target += ".exe"
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("prior"), 0o755); err != nil {
		t.Fatal(err)
	}
	shimDir := filepath.Join(dir, "shims")
	if err := os.MkdirAll(shimDir, 0o755); err != nil {
		t.Fatal(err)
	}
	stable, err := installStableLaunchTarget(shimDir, target)
	if err != nil {
		t.Fatal(err)
	}
	if err := launchshim.Save(launchshim.Config{
		Executable: target,
		Providers:  map[string]launchshim.Provider{},
	}); err != nil {
		t.Fatal(err)
	}
	finish, err := selfinstall.BeginLaunchTransaction(target)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(finish)

	oldExecutable := osExecutableForLaunch
	oldInput := launchInput
	oldRunner := launchChildRunner
	oldResolve := stableLaunchResolve
	t.Cleanup(func() {
		osExecutableForLaunch = oldExecutable
		launchInput = oldInput
		launchChildRunner = oldRunner
		stableLaunchResolve = oldResolve
	})
	osExecutableForLaunch = func() (string, error) { return stable, nil }

	t.Run("default prior is immediate and lossless", func(t *testing.T) {
		launchInput = strings.NewReader("stdin-prior")
		var gotCommand string
		var gotArgs []string
		launchChildRunner = func(stdin io.Reader, stdout, stderr io.Writer, command string, args []string) int {
			gotCommand, gotArgs = command, append([]string(nil), args...)
			gotInput, _ := io.ReadAll(stdin)
			fmt.Fprintf(stdout, "stdout:%s", gotInput)
			fmt.Fprint(stderr, "stderr:prior")
			return 23
		}
		var stdout, stderr bytes.Buffer
		code := runLaunch(&stdout, &stderr, []string{"codex", "--", "space value", `quote"value`})
		if code != 23 || gotCommand != selfinstall.LaunchPriorPath(target) {
			t.Fatalf("code=%d command=%q stderr=%s", code, gotCommand, stderr.String())
		}
		wantArgs := []string{"launch", "codex", "--", "space value", `quote"value`}
		if !reflect.DeepEqual(gotArgs, wantArgs) {
			t.Fatalf("argv=%q want %q", gotArgs, wantArgs)
		}
		if stdout.String() != "stdout:stdin-prior" || stderr.String() != "stderr:prior" {
			t.Fatalf("stdout=%q stderr=%q", stdout.String(), stderr.String())
		}
	})

	t.Run("fail is strict and does not spawn", func(t *testing.T) {
		t.Setenv("FAK_UPDATE_LAUNCH_POLICY", "fail")
		called := false
		launchChildRunner = func(io.Reader, io.Writer, io.Writer, string, []string) int {
			called = true
			return 0
		}
		var stdout, stderr bytes.Buffer
		code := runLaunch(&stdout, &stderr, []string{"codex", "arg"})
		if code != 75 || called || !strings.Contains(stderr.String(), "self-update is replacing") {
			t.Fatalf("code=%d called=%t stdout=%q stderr=%q", code, called, stdout.String(), stderr.String())
		}
	})

	t.Run("wait is bounded then launches new with lossless process state", func(t *testing.T) {
		t.Setenv("FAK_UPDATE_LAUNCH_POLICY", "")
		resolverEntered := make(chan struct{})
		releaseResolver := make(chan struct{})
		stableLaunchResolve = func(gotTarget, policy string, wait time.Duration) (string, error) {
			if gotTarget != target || policy != launchshim.UpdatePolicyWait || wait != 2*time.Second {
				t.Errorf("resolve target=%q policy=%q wait=%s", gotTarget, policy, wait)
			}
			close(resolverEntered)
			<-releaseResolver
			return target, nil
		}
		launchInput = strings.NewReader("stdin-wait")
		childCalled := make(chan struct{})
		var gotArgs []string
		launchChildRunner = func(stdin io.Reader, stdout, stderr io.Writer, command string, args []string) int {
			if command != target {
				t.Errorf("command=%q want target %q", command, target)
			}
			gotArgs = append([]string(nil), args...)
			gotInput, _ := io.ReadAll(stdin)
			fmt.Fprintf(stdout, "stdout:%s", gotInput)
			fmt.Fprint(stderr, "stderr:wait")
			close(childCalled)
			return 37
		}
		type result struct {
			code           int
			stdout, stderr string
		}
		resultc := make(chan result, 1)
		go func() {
			var stdout, stderr bytes.Buffer
			code := runLaunch(&stdout, &stderr, []string{"--update-launch-policy=wait", "--update-launch-wait=2s", "codex", "--", "space value", `quote"value`})
			resultc <- result{code, stdout.String(), stderr.String()}
		}()
		<-resolverEntered
		select {
		case <-childCalled:
			t.Fatal("child launched before wait policy released the replacement boundary")
		default:
		}
		close(releaseResolver)
		got := <-resultc
		if got.code != 37 || got.stdout != "stdout:stdin-wait" || got.stderr != "stderr:wait" {
			t.Fatalf("result=%+v", got)
		}
		wantArgs := []string{"launch", "codex", "--", "space value", `quote"value`}
		if !reflect.DeepEqual(gotArgs, wantArgs) {
			t.Fatalf("argv=%q want %q", gotArgs, wantArgs)
		}
	})

	t.Run("provider update-looking args are not consumed", func(t *testing.T) {
		stableLaunchResolve = func(string, string, time.Duration) (string, error) {
			return target, nil
		}
		launchInput = strings.NewReader("")
		var gotArgs []string
		launchChildRunner = func(_ io.Reader, _ io.Writer, _ io.Writer, _ string, args []string) int {
			gotArgs = append([]string(nil), args...)
			return 0
		}
		var stdout, stderr bytes.Buffer
		code := runLaunch(&stdout, &stderr, []string{"codex", "--", "--update-launch-policy=fail"})
		wantArgs := []string{"launch", "codex", "--", "--update-launch-policy=fail"}
		if code != 0 || !reflect.DeepEqual(gotArgs, wantArgs) {
			t.Fatalf("code=%d argv=%q want %q stderr=%q", code, gotArgs, wantArgs, stderr.String())
		}
	})
}

func TestInstalledLaunchTopologyWindows(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows launch topology")
	}

	root := t.TempDir()
	nodeSource := filepath.Join(os.Getenv("WINDIR"), "System32", "whoami.exe")
	nodeBytes, err := os.ReadFile(nodeSource)
	if err != nil {
		t.Fatal(err)
	}
	nodeDir := filepath.Join(root, "node runtime")
	if err := os.MkdirAll(nodeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	node := filepath.Join(nodeDir, "node.exe")
	if err := os.WriteFile(node, nodeBytes, 0o755); err != nil {
		t.Fatal(err)
	}

	const provider = "codex"
	entrypointRel := filepath.Join("node_modules", "@openai", "codex", "bin", "codex.js")
	makeInstall := func(name string, entrypoint bool) string {
		dir := filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, provider+".cmd"), []byte("@echo off\r\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if entrypoint {
			path := filepath.Join(dir, entrypointRel)
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte("// hermetic fixture\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		return dir
	}

	managed := makeInstall("managed wrapper", false)
	npm := makeInstall("npm provider", true)
	npmSpaces := makeInstall("npm provider with spaces", true)
	missing := makeInstall("npm missing entrypoint", false)
	first := makeInstall("npm first", true)
	second := makeInstall("npm second", true)

	tests := []struct {
		name       string
		path       []string
		wantDir    string
		wantDirect bool
	}{
		{name: "managed wrapper directory precedes npm provider", path: []string{managed, npm}, wantDir: npm, wantDirect: true},
		{name: "npm provider precedes managed", path: []string{npm, managed}, wantDir: npm, wantDirect: true},
		{name: "provider path contains spaces", path: []string{npmSpaces}, wantDir: npmSpaces, wantDirect: true},
		{name: "provider cmd missing Node entrypoint", path: []string{missing}, wantDirect: false},
		{name: "two provider installations choose first non-managed", path: []string{managed, first, second}, wantDir: first, wantDirect: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pathEnv := strings.Join(tt.path, string(os.PathListSeparator))
			original := []string{provider, "--version"}
			resolved := resolveNodeBatchCommandFromPath(original, entrypointRel, node, pathEnv)
			if !tt.wantDirect {
				if !reflect.DeepEqual(resolved, original) {
					t.Fatalf("resolved=%q want unresolved %q", resolved, original)
				}
				return // Resolution failed before any spawn attempt.
			}

			wantEntrypoint := filepath.Join(tt.wantDir, entrypointRel)
			want := []string{node, wantEntrypoint, "--version"}
			if !reflect.DeepEqual(resolved, want) {
				t.Fatalf("resolved=%q want %q", resolved, want)
			}

			t.Setenv("PATH", pathEnv+string(os.PathListSeparator)+nodeDir)
			cmd := newResolvedExecCommand(original)
			if !strings.EqualFold(filepath.Base(cmd.Path), "node.exe") {
				t.Fatalf("command path=%q want node.exe", cmd.Path)
			}
			if strings.EqualFold(filepath.Base(cmd.Path), "cmd.exe") {
				t.Fatalf("command unexpectedly uses cmd.exe: %q", cmd.Path)
			}
			if len(cmd.Args) < 2 || !strings.EqualFold(cmd.Args[0], node) || cmd.Args[1] != wantEntrypoint {
				t.Fatalf("command argv=%q want node and direct entrypoint %q", cmd.Args, wantEntrypoint)
			}
			if err := cmd.Run(); err == nil {
				return
			} else if _, ok := err.(*os.PathError); ok {
				t.Fatalf("node.exe was not spawned: %v", err)
			}
		})
	}
}
