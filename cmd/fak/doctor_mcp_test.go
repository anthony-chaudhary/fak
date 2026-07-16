package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestDoctorMCPReadsCodexEntryWithoutSecrets(t *testing.T) {
	p := filepath.Join(t.TempDir(), "config.toml")
	body := `[mcp_servers.fak]
command = 'C:\tools\fak.exe'
args = ["serve", "--stdio", "--policy", 'C:\repo\policy.json']
env = { API_KEY = "must-not-leak" }
`
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	cmd, args, err := readCodexMCPEntry(p, "fak")
	if err != nil {
		t.Fatal(err)
	}
	if cmd != `C:\tools\fak.exe` || strings.Join(args, "|") != `serve|--stdio|--policy|C:\repo\policy.json` {
		t.Fatalf("cmd=%q args=%q", cmd, args)
	}
	b, _ := json.Marshal(doctorMCPReport{Command: cmd, Args: args})
	if strings.Contains(string(b), "must-not-leak") {
		t.Fatalf("secret leaked: %s", b)
	}
}

func TestDoctorMCPMissingExecutableIsTyped(t *testing.T) {
	rep := diagnoseMCP("fak", "", "definitely-missing-fak-mcp-4659", nil, time.Second)
	if rep.OK {
		t.Fatal("missing executable reported healthy")
	}
	if got := stageCause(rep, "executable_resolution"); got != "EXECUTABLE_MISSING" {
		t.Fatalf("cause=%q", got)
	}
}

func TestDoctorMCPMalformedPolicyIsTyped(t *testing.T) {
	policy := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(policy, []byte("{broken"), 0o600); err != nil {
		t.Fatal(err)
	}
	cmd := "cmd.exe"
	args := []string{"/d", "/c", "exit 0", "--policy", policy}
	if runtime.GOOS != "windows" {
		cmd = "sh"
		args = []string{"-c", "exit 0", "--policy", policy}
	}
	rep := diagnoseMCP("fixture", "", cmd, args, time.Second)
	if got := stageCause(rep, "policy_readability"); got != "POLICY_MALFORMED" {
		t.Fatalf("cause=%q stages=%+v", got, rep.Stages)
	}
}

func TestDoctorMCPInitializeTimeoutIsTyped(t *testing.T) {
	cmd := "cmd.exe"
	args := []string{"/d", "/c", "ping -n 6 127.0.0.1 >nul"}
	if runtime.GOOS != "windows" {
		cmd = "sh"
		args = []string{"-c", "sleep 5"}
	}
	rep := diagnoseMCP("fixture", "", cmd, args, 50*time.Millisecond)
	if got := stageCause(rep, "initialize_response"); got != "INITIALIZE_TIMEOUT" {
		t.Fatalf("cause=%q stages=%+v", got, rep.Stages)
	}
}

func TestDoctorMCPPermissionFailureIsTyped(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows executable ACL fixture is host-policy dependent")
	}
	path := filepath.Join(t.TempDir(), "not-executable")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	rep := diagnoseMCP("fixture", "", path, nil, time.Second)
	if got := stageCause(rep, "executable_resolution"); got != "EXECUTABLE_PERMISSION_DENIED" {
		t.Fatalf("cause=%q stages=%+v", got, rep.Stages)
	}
}

func TestDoctorMCPCorruptProductionRegistryIsUntouched(t *testing.T) {
	exe := buildDoctorSelfcheck(t)
	registry := filepath.Join(t.TempDir(), "production-registry.json")
	before := []byte("{corrupt-production-state")
	if err := os.WriteFile(registry, before, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(sessionRegistryEnv, registry)
	policy := filepath.Join(repoRootForDoctorTest(t), "examples", "dev-agent-policy.json")
	rep := diagnoseMCP("self", "", exe, []string{"serve", "--stdio", "--policy", policy}, 10*time.Second)
	if !rep.OK {
		t.Fatalf("probe failed: %+v", rep.Stages)
	}
	after, err := os.ReadFile(registry)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatalf("production registry mutated: before=%q after=%q", before, after)
	}
	matches, err := filepath.Glob(registry + ".corrupt-*")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 0 {
		t.Fatalf("diagnosis quarantined production state: %v", matches)
	}
}

func TestDoctorMCPStdoutContaminationAndEarlyExit(t *testing.T) {
	if runtime.GOOS == "windows" {
		rep := diagnoseMCP("fixture", "", "cmd.exe", []string{"/d", "/c", "echo banner"}, 2*time.Second)
		if got := stageCause(rep, "stdout_protocol_purity"); got != "STDOUT_CONTAMINATION" {
			t.Fatalf("contamination cause=%q stages=%+v", got, rep.Stages)
		}
		rep = diagnoseMCP("fixture", "", "cmd.exe", []string{"/d", "/c", "exit 7"}, 2*time.Second)
		if got := stageCause(rep, "stdout_protocol_purity"); got != "EARLY_EXIT" {
			t.Fatalf("early-exit cause=%q stages=%+v", got, rep.Stages)
		}
	}
}

func TestDoctorMCPLiveSelfCheck(t *testing.T) {
	exe := buildDoctorSelfcheck(t)
	policy := filepath.Join(repoRootForDoctorTest(t), "examples", "dev-agent-policy.json")
	rep := diagnoseMCP("self", "", exe, []string{"serve", "--stdio", "--policy", policy}, 10*time.Second)
	if !rep.OK {
		t.Fatalf("live self-check failed: %+v", rep.Stages)
	}
	if got := stageStatus(rep, "initialize_response"); got != "pass" {
		t.Fatalf("initialize=%q", got)
	}
}

func buildDoctorSelfcheck(t *testing.T) string {
	t.Helper()
	exe := filepath.Join(t.TempDir(), "fak-doctor-selfcheck")
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	build := exec.Command("go", "build", "-o", exe, "./cmd/fak")
	build.Dir = repoRootForDoctorTest(t)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build self-check binary: %v\n%s", err, out)
	}
	return exe
}

func stageCause(r doctorMCPReport, name string) string {
	for _, s := range r.Stages {
		if s.Name == name {
			return s.Cause
		}
	}
	return ""
}
func stageStatus(r doctorMCPReport, name string) string {
	for _, s := range r.Stages {
		if s.Name == name {
			return s.Status
		}
	}
	return ""
}
func repoRootForDoctorTest(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for dir := wd; ; dir = filepath.Dir(dir) {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		if filepath.Dir(dir) == dir {
			t.Fatalf("go.mod not found above %s", wd)
		}
	}
}
