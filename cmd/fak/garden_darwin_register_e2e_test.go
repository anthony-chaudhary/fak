package main

import (
	"bytes"
	"encoding/xml"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=20ms
func TestGardenRegisterDarwinWritesParseablePlist(t *testing.T) {
	home := filepath.Join(t.TempDir(), "home dir & co")
	if err := os.MkdirAll(home, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	const fakBin = `/opt/fak bin/fak&co`
	const root = `C:/tmp/a&b <c> "d" 'e'`

	var stdout, stderr bytes.Buffer
	if code := registerDarwinLaunchdAgent(&stdout, &stderr, fakBin, root, time.Hour, true); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	plistPath := filepath.Join(home, "Library", "LaunchAgents", gardenLaunchdLabel+".plist")
	raw, err := os.ReadFile(plistPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("plist:\n%s", raw)
	var got struct {
		Args []string `xml:"dict>array>string"`
	}
	if err := xml.Unmarshal(raw, &got); err != nil {
		t.Fatalf("written plist is not well-formed XML: %v", err)
	}
	want := []string{fakBin, "garden", "watchdog", "--repo", root, "--live"}
	if !reflect.DeepEqual(got.Args, want) {
		t.Fatalf("ProgramArguments = %q, want %q", got.Args, want)
	}
	if !strings.Contains(stdout.String(), gardenDarwinLoadCommand(plistPath)) {
		t.Fatalf("stdout lacks the quoted load command for %s", plistPath)
	}
	t.Logf("guidance: %s", gardenDarwinLoadCommand(plistPath))
}
