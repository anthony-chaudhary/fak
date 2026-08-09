package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
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
	launchChildRunner = func(_ io.Writer, _ io.Writer, command string, args []string) int {
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
	launchChildRunner = func(_ io.Writer, _ io.Writer, command string, args []string) int {
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
