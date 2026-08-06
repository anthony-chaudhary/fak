package main

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
)

func TestLaunchDefaultAndDisablePersist(t *testing.T) {
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(t.TempDir(), "launch.json"))
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
