package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
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

func TestLaunchDoctorNamesExistingUnmanagedPathWinner(t *testing.T) {
	root := t.TempDir()
	shimDir := filepath.Join(root, "bin")
	underlying := filepath.Join(root, "npm", "codex.cmd")
	if err := os.MkdirAll(filepath.Dir(underlying), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(underlying, []byte("@echo off\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	look := func(name string) (string, error) {
		if name == "codex" {
			return underlying, nil
		}
		return "", exec.ErrNotFound
	}
	report := buildLaunchDoctor(launchshim.Config{Providers: map[string]launchshim.Provider{}}, nil, filepath.Join(root, "launch.json"), shimDir, look, os.Stat)
	var codex launchDoctorRow
	for _, row := range report.Rows {
		if row.Provider == "codex" {
			codex = row
		}
	}
	if codex.Reason != "UNMANAGED" || codex.PathWinner != "<local>/codex.cmd" || codex.InterceptReady {
		t.Fatalf("codex row=%+v", codex)
	}
	if report.Binary.AppVersion == "" || report.Binary.Go == "" {
		t.Fatalf("binary provenance missing: %+v", report.Binary)
	}
}

func TestBuildLaunchDoctorEntryPointsClassifiesCodexSurfaces(t *testing.T) {
	entries := buildLaunchDoctorEntryPoints([]launchDoctorRow{{Provider: "codex", Reason: "READY", InterceptReady: true}})
	if len(entries) != 3 {
		t.Fatalf("entry points=%d want 3: %+v", len(entries), entries)
	}
	want := []struct {
		command, role, reason string
		ready                 bool
	}{
		{"codex", "canonical", "READY", true},
		{"fak m codex", "noncanonical", "PATH_RESOLUTION_AMBIGUOUS", false},
		{"fak codex", "specialized", "PATH_RESOLUTION_AMBIGUOUS", false},
	}
	seen := map[string]bool{}
	for i, entry := range entries {
		if seen[entry.Command] {
			t.Fatalf("duplicate command %q", entry.Command)
		}
		seen[entry.Command] = true
		if entry.Command != want[i].command || entry.Role != want[i].role || entry.Reason != want[i].reason || entry.Ready != want[i].ready {
			t.Errorf("entry[%d]=%+v want command=%q role=%q reason=%q ready=%t", i, entry, want[i].command, want[i].role, want[i].reason, want[i].ready)
		}
		if len(entry.Pipeline) < 3 {
			t.Errorf("entry[%d] pipeline too weak: %+v", i, entry.Pipeline)
		}
	}
	if got := strings.Join(entries[0].Pipeline, " -> "); !strings.Contains(got, "recorded-provider") {
		t.Fatalf("canonical pipeline=%q", got)
	}
	if got := strings.Join(entries[2].Pipeline, " -> "); !strings.Contains(got, "freshness-admission") || !strings.Contains(got, "loop-gate") {
		t.Fatalf("specialized pipeline=%q", got)
	}
}

func TestBuildLaunchDoctorEntryPointsPropagatesProviderFailure(t *testing.T) {
	entries := buildLaunchDoctorEntryPoints([]launchDoctorRow{{Provider: "codex", Reason: "SHADOWED", Action: "prepend bin to PATH"}})
	for _, entry := range entries {
		if entry.Ready || entry.Reason != "SHADOWED" || entry.Action != "prepend bin to PATH" {
			t.Fatalf("entry=%+v", entry)
		}
	}
}

func TestLaunchDoctorHumanOutputNamesEntryPointRoles(t *testing.T) {
	oldConfig, oldBin := os.Getenv("FAK_LAUNCH_CONFIG"), os.Getenv("FAK_LAUNCH_BIN")
	t.Cleanup(func() { _ = os.Setenv("FAK_LAUNCH_CONFIG", oldConfig); _ = os.Setenv("FAK_LAUNCH_BIN", oldBin) })
	root := t.TempDir()
	_ = os.Setenv("FAK_LAUNCH_CONFIG", filepath.Join(root, "missing.json"))
	_ = os.Setenv("FAK_LAUNCH_BIN", filepath.Join(root, "bin"))
	var out, errOut bytes.Buffer
	_ = runLaunchDoctor(&out, &errOut, nil)
	text := out.String()
	for _, want := range []string{"ENTRY POINTS", "codex        role=canonical", "fak m codex  role=noncanonical", "fak codex    role=specialized"} {
		if !strings.Contains(text, want) {
			t.Errorf("human output missing %q:\n%s", want, text)
		}
	}
}

func TestInstalledLaunchQualificationReceipt(t *testing.T) {
	readyRows := []launchDoctorRow{{
		Provider:       "codex",
		Reason:         "READY",
		PathWinner:     redactLocalPath(filepath.Join(t.TempDir(), shimName("codex"))),
		Underlying:     redactLocalPath(filepath.Join(t.TempDir(), "codex.cmd")),
		InterceptReady: true,
	}}
	receipt := buildInstalledLaunchQualification(readyRows)
	if receipt.Schema != installedLaunchQualificationSchema || !receipt.Qualified || len(receipt.Providers) != 1 {
		t.Fatalf("success receipt = %#v", receipt)
	}
	provider := receipt.Providers[0]
	if provider.Harness != "fak-launch/codex" || provider.Status != "READY" || provider.Failure != "" {
		t.Fatalf("success provider = %#v", provider)
	}
	if want := []string{"<local>/codex.cmd", "fak guard", "<local>/codex.cmd"}; !reflect.DeepEqual(provider.Chain, want) {
		t.Fatalf("resolved chain = %#v, want %#v", provider.Chain, want)
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), t.TempDir()) || strings.Contains(string(encoded), `C:\`) {
		t.Fatalf("receipt leaks local path: %s", encoded)
	}

	failed := buildInstalledLaunchQualification([]launchDoctorRow{{Provider: "codex", Reason: "UNDERLYING_MISSING"}})
	if failed.Qualified || len(failed.Providers) != 1 || failed.Providers[0].Failure != "UNDERLYING_MISSING" || len(failed.Providers[0].Chain) != 0 {
		t.Fatalf("failure receipt = %#v", failed)
	}

	report := buildLaunchDoctor(launchshim.Config{Providers: map[string]launchshim.Provider{
		"codex": {Command: filepath.Join(t.TempDir(), "codex.cmd")},
	}}, nil, filepath.Join(t.TempDir(), "launch.json"), t.TempDir(), func(name string) (string, error) {
		return "", exec.ErrNotFound
	}, func(path string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})
	if report.Qualification.Schema != installedLaunchQualificationSchema || report.Qualification.Qualified {
		t.Fatalf("doctor qualification = %#v", report.Qualification)
	}
	if got := report.Qualification.Providers[0].Failure; got == "" {
		t.Fatalf("doctor failure is untyped: %#v", report.Qualification)
	}
}
