package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
)

func TestLaunchDoctorGoldenReasons(t *testing.T) {
	root := t.TempDir()
	shimDir := filepath.Join(root, "shims")
	underlying := filepath.Join(root, "real", "claude")
	shim := filepath.Join(shimDir, shimName("claude"))
	infos := map[string]bool{underlying: true, filepath.Join(root, "real", "codex"): true}
	stat := func(path string) (os.FileInfo, error) {
		if infos[path] {
			return fakeLaunchInfo{name: filepath.Base(path)}, nil
		}
		return nil, os.ErrNotExist
	}
	look := func(provider string) (string, error) {
		if provider == "claude" {
			return shim, nil
		}
		return filepath.Join(root, "other", provider), nil
	}
	c := launchshim.Config{Default: "claude", Providers: map[string]launchshim.Provider{
		"claude": {Command: underlying},
		"codex":  {Command: filepath.Join(root, "real", "codex")},
	}}
	report := buildLaunchDoctor(c, nil, filepath.Join(root, "launch.json"), shimDir, look, stat)
	if report.Schema != launchDoctorSchema || report.ConfigPath != "<local>/launch.json" || len(report.Rows) != 2 {
		t.Fatalf("report=%+v", report)
	}
	if report.Rows[0].Provider != "claude" || report.Rows[0].Reason != "READY" || !report.Rows[0].InterceptReady {
		t.Fatalf("claude=%+v", report.Rows[0])
	}
	if report.Rows[1].Provider != "codex" || report.Rows[1].Reason != "SHADOWED" || report.Rows[1].Action == "" {
		t.Fatalf("codex=%+v", report.Rows[1])
	}
}

func TestLaunchDoctorMalformedConfigJSON(t *testing.T) {
	root := t.TempDir()
	t.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(root, "launch.json"))
	t.Setenv("FAK_LAUNCH_BIN", filepath.Join(root, "bin"))
	if err := os.WriteFile(filepath.Join(root, "launch.json"), []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runLaunchDoctor(&out, &errOut, []string{"--json"}); code != 1 {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
	for _, want := range []string{`"schema": "fak.launch-doctor.v1"`, `"reason": "CONFIG_INVALID"`, `"action": "fak launch install --provider claude"`} {
		if !bytes.Contains(out.Bytes(), []byte(want)) {
			t.Fatalf("missing %q in %s", want, out.String())
		}
	}
	if bytes.Contains(out.Bytes(), []byte(root)) {
		t.Fatalf("JSON leaked local root: %s", out.String())
	}
}

func TestLaunchDoctorClosedReasonMatrix(t *testing.T) {
	root := t.TempDir()
	shimDir := filepath.Join(root, "bin")
	real := filepath.Join(root, "real")
	stat := func(path string) (os.FileInfo, error) {
		if path == real {
			return fakeLaunchInfo{name: "real"}, nil
		}
		return nil, os.ErrNotExist
	}
	cases := []struct {
		name   string
		config launchshim.Config
		look   launchLookPath
		want   string
	}{
		{"not on path", launchshim.Config{Providers: map[string]launchshim.Provider{"claude": {Command: real}}}, func(string) (string, error) { return "", errors.New("missing") }, "NOT_ON_PATH"},
		{"underlying missing", launchshim.Config{Providers: map[string]launchshim.Provider{"claude": {Command: filepath.Join(root, "gone")}}}, func(string) (string, error) { return filepath.Join(shimDir, shimName("claude")), nil }, "UNDERLYING_MISSING"},
		{"recursive", launchshim.Config{Providers: map[string]launchshim.Provider{"claude": {Command: filepath.Join(shimDir, shimName("claude"))}}}, func(string) (string, error) { return filepath.Join(shimDir, shimName("claude")), nil }, "RECURSIVE"},
		{"disabled", launchshim.Config{Disabled: true, Providers: map[string]launchshim.Provider{"claude": {Command: real}}}, func(string) (string, error) { return filepath.Join(shimDir, shimName("claude")), nil }, "DISABLED"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := buildLaunchDoctor(tc.config, nil, "config", shimDir, tc.look, stat)
			var got string
			for _, row := range r.Rows {
				if row.Provider == "claude" {
					got = row.Reason
					if row.Action == "" {
						t.Fatal("missing action")
					}
				}
			}
			if got != tc.want {
				t.Fatalf("reason=%s want=%s", got, tc.want)
			}
		})
	}
}

type fakeLaunchInfo struct{ name string }

func (f fakeLaunchInfo) Name() string     { return f.name }
func (fakeLaunchInfo) Size() int64        { return 1 }
func (fakeLaunchInfo) Mode() os.FileMode  { return 0o755 }
func (fakeLaunchInfo) ModTime() time.Time { return time.Time{} }
func (fakeLaunchInfo) IsDir() bool        { return false }
func (fakeLaunchInfo) Sys() any           { return nil }
