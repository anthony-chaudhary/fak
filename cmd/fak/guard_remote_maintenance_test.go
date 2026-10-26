package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

func remoteMaintenanceTestConfig(t *testing.T, expires time.Time, ids ...string) remotemaint.Config {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	cfg := remotemaint.Config{Schema: remotemaint.ConfigSchema}
	for _, id := range ids {
		cfg.Profiles = append(cfg.Profiles, remotemaint.Profile{
			ProfileID: id, Target: "maintenance.example", HostKey: "SHA256:" + strings.Repeat("A", 43),
			Operation:     remotemaint.OperationSystemdProfileInstallReload,
			PayloadSHA256: strings.Repeat("1", 64), Holder: "operator-reviewed",
			Paths: []string{"/etc/systemd/system/fak.service"}, PathPrefixes: []remotemaint.PathPrefix{{Directory: "/var/tmp", FilenamePrefix: "maintenance-stage."}},
			Units: []string{"fak.service"}, LeaseNodes: []string{"maintenance-node"}, ExpiresAt: expires.UTC(),
			Authority: remotemaint.Command{Executable: executable, SHA256: hex.EncodeToString(sum[:]), Args: []string{"remote-maintenance"}},
		})
	}
	return cfg
}

func writeRemoteMaintenanceTestConfig(t *testing.T, path string, cfg remotemaint.Config) {
	t.Helper()
	body, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=2s
func TestRemoteMaintenanceOperatorInstallClampsTTLAndRevokes(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	dir := t.TempDir()
	reviewed := filepath.Join(dir, "reviewed.json")
	destination := filepath.Join(dir, "home", ".fak", "guard", "remote-maintenance.json")
	writeRemoteMaintenanceTestConfig(t, reviewed, remoteMaintenanceTestConfig(t, now.Add(time.Hour), "profile-a", "profile-b"))

	var stdout, stderr bytes.Buffer
	if code := runGuardRemoteMaintenanceAt(&stdout, &stderr, destination, reviewed, 10*time.Minute, false, now); code != 0 {
		t.Fatalf("install code=%d stderr=%s", code, stderr.String())
	}
	installed, err := readGuardRemoteMaintenanceConfig(destination)
	if err != nil {
		t.Fatal(err)
	}
	wantExpiry := now.Add(10 * time.Minute)
	for _, profile := range installed.Profiles {
		if !profile.ExpiresAt.Equal(wantExpiry) {
			t.Fatalf("profile %s expiry=%s, want clamped %s", profile.ProfileID, profile.ExpiresAt, wantExpiry)
		}
	}

	stdout.Reset()
	stderr.Reset()
	if code := runGuardRemoteMaintenanceAt(&stdout, &stderr, destination, "profile-a", 0, true, now); code != 0 {
		t.Fatalf("revoke one code=%d stderr=%s", code, stderr.String())
	}
	installed, err = readGuardRemoteMaintenanceConfig(destination)
	if err != nil {
		t.Fatal(err)
	}
	if len(installed.Profiles) != 1 || installed.Profiles[0].ProfileID != "profile-b" {
		t.Fatalf("profiles after revoke=%+v", installed.Profiles)
	}
	if code := runGuardRemoteMaintenanceAt(&stdout, &stderr, destination, "profile-b", 0, true, now); code != 0 {
		t.Fatalf("revoke final code=%d stderr=%s", code, stderr.String())
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatalf("final revoke left context: %v", err)
	}
}

// fak-test:runtime fast est=1s
func TestRemoteMaintenanceOperatorRequiresBoundedTTL(t *testing.T) {
	destination := filepath.Join(t.TempDir(), "remote-maintenance.json")
	for _, ttl := range []time.Duration{0, -time.Second, guardRemoteMaintenanceMaxTTL + time.Second} {
		var stdout, stderr bytes.Buffer
		if code := runGuardRemoteMaintenanceAt(&stdout, &stderr, destination, "reviewed.json", ttl, false, time.Now().UTC()); code != 2 {
			t.Fatalf("ttl=%s code=%d stderr=%s, want usage refusal", ttl, code, stderr.String())
		}
	}
}

// fak-test:runtime fast est=2s
func TestRemoteMaintenanceNormalFloorLoadsAndDeniesSelfModify(t *testing.T) {
	now := time.Now().UTC()
	dir := t.TempDir()
	configPath := filepath.Join(dir, "remote-maintenance.json")
	writeRemoteMaintenanceTestConfig(t, configPath, remoteMaintenanceTestConfig(t, now.Add(time.Hour), "normal-route"))

	originalDefault := guardRemoteMaintenanceDefaultPath
	originalLoad := guardRemoteMaintenanceLoadAuthority
	originalProtected := guardRemoteMaintenanceProtectedPaths
	t.Cleanup(func() {
		guardRemoteMaintenanceDefaultPath = originalDefault
		guardRemoteMaintenanceLoadAuthority = originalLoad
		guardRemoteMaintenanceProtectedPaths = originalProtected
	})
	guardRemoteMaintenanceDefaultPath = func() (string, error) { return configPath, nil }
	loads := 0
	guardRemoteMaintenanceLoadAuthority = func(path string) (remotemaint.Authority, error) {
		loads++
		return remotemaint.LoadAuthority(path)
	}
	guardRemoteMaintenanceProtectedPaths = remotemaint.ProtectedPaths

	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	rt = protectRemoteMaintenanceAuthority(rt)
	if loads != 1 {
		t.Fatalf("normal floor authority loads=%d, want 1", loads)
	}
	wants := map[string]bool{filepath.ToSlash(filepath.Clean(configPath)): false}
	for _, profile := range remoteMaintenanceTestConfig(t, now.Add(time.Hour), "normal-route").Profiles {
		wants[filepath.ToSlash(filepath.Clean(profile.Authority.Executable))] = false
	}
	for _, got := range rt.Adjudicator.SelfModifyGlobs {
		if _, ok := wants[filepath.ToSlash(got)]; ok {
			wants[filepath.ToSlash(got)] = true
		}
	}
	for path, found := range wants {
		if !found {
			t.Fatalf("self_modify_globs=%v, missing %q", rt.Adjudicator.SelfModifyGlobs, path)
		}
	}

	a := adjudicator.New(rt.Adjudicator)
	for path := range wants {
		call := guardToolCall(t, "Bash", map[string]any{"command": "echo x > " + path})
		verdict := a.Adjudicate(context.Background(), call)
		if verdict.Kind != abi.VerdictDeny || verdict.Reason != abi.ReasonSelfModify {
			t.Fatalf("write %q verdict=(%v,%v), want deny/SELF_MODIFY", path, verdict.Kind, verdict.Reason)
		}
	}
}

// fak-test:runtime fast est=1s
func TestRemoteMaintenanceMissingContextPathIsStillProtected(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "not-created", "remote-maintenance.json")
	originalDefault := guardRemoteMaintenanceDefaultPath
	originalLoad := guardRemoteMaintenanceLoadAuthority
	originalProtected := guardRemoteMaintenanceProtectedPaths
	t.Cleanup(func() {
		guardRemoteMaintenanceDefaultPath = originalDefault
		guardRemoteMaintenanceLoadAuthority = originalLoad
		guardRemoteMaintenanceProtectedPaths = originalProtected
	})
	guardRemoteMaintenanceDefaultPath = func() (string, error) { return configPath, nil }
	loads := 0
	guardRemoteMaintenanceLoadAuthority = func(path string) (remotemaint.Authority, error) {
		loads++
		return remotemaint.LoadAuthority(path)
	}
	guardRemoteMaintenanceProtectedPaths = remotemaint.ProtectedPaths

	rt := protectRemoteMaintenanceAuthority(policy.Runtime{})
	if loads != 0 {
		t.Fatalf("missing context invoked loader %d time(s), want disabled authority", loads)
	}
	want := filepath.ToSlash(filepath.Clean(configPath))
	if len(rt.Adjudicator.SelfModifyGlobs) != 1 || rt.Adjudicator.SelfModifyGlobs[0] != want {
		t.Fatalf("self_modify_globs=%v, want missing default path %q protected", rt.Adjudicator.SelfModifyGlobs, want)
	}
}

// fak-test:runtime fast est=1s
func TestRemoteMaintenanceExpiredContextDoesNotBrickBaseFloor(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "remote-maintenance.json")
	writeRemoteMaintenanceTestConfig(t, configPath, remoteMaintenanceTestConfig(t, time.Now().Add(-time.Minute), "expired"))
	originalDefault := guardRemoteMaintenanceDefaultPath
	t.Cleanup(func() { guardRemoteMaintenanceDefaultPath = originalDefault })
	guardRemoteMaintenanceDefaultPath = func() (string, error) { return configPath, nil }
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	rt = protectRemoteMaintenanceAuthority(rt)
	if !rt.Adjudicator.Allow["Read"] || !rt.Adjudicator.Allow["fak_remote_maintenance"] {
		t.Fatalf("expired optional authority changed base allow floor: %+v", rt.Adjudicator.Allow)
	}
	want := filepath.ToSlash(filepath.Clean(configPath))
	found := false
	for _, path := range rt.Adjudicator.SelfModifyGlobs {
		found = found || filepath.ToSlash(path) == want
	}
	if !found {
		t.Fatalf("expired context path %q is not protected: %v", want, rt.Adjudicator.SelfModifyGlobs)
	}
}

// fak-test:runtime fast est=1s
func TestRemoteMaintenanceMalformedContextDoesNotBrickBaseFloor(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "remote-maintenance.json")
	if err := os.WriteFile(configPath, []byte(`{"broken":`), 0o600); err != nil {
		t.Fatal(err)
	}
	originalDefault := guardRemoteMaintenanceDefaultPath
	t.Cleanup(func() { guardRemoteMaintenanceDefaultPath = originalDefault })
	guardRemoteMaintenanceDefaultPath = func() (string, error) { return configPath, nil }
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	rt = protectRemoteMaintenanceAuthority(rt)
	if !rt.Adjudicator.Allow["Read"] || !rt.Adjudicator.Allow["fak_remote_maintenance"] {
		t.Fatalf("malformed optional authority changed base allow floor: %+v", rt.Adjudicator.Allow)
	}
	want := filepath.ToSlash(filepath.Clean(configPath))
	found := false
	for _, path := range rt.Adjudicator.SelfModifyGlobs {
		found = found || filepath.ToSlash(path) == want
	}
	if !found {
		t.Fatalf("malformed context path %q is not protected: %v", want, rt.Adjudicator.SelfModifyGlobs)
	}
}

// fak-test:runtime fast est=1s
func TestRemoteMaintenanceExpiredProfileCanBeRevokedAndMalformedContextCleared(t *testing.T) {
	now := time.Now().UTC()
	destination := filepath.Join(t.TempDir(), "remote-maintenance.json")
	writeRemoteMaintenanceTestConfig(t, destination, remoteMaintenanceTestConfig(t, now.Add(-time.Minute), "drop", "keep"))
	var stdout, stderr bytes.Buffer
	if code := runGuardRemoteMaintenanceAt(&stdout, &stderr, destination, "drop", 0, true, now); code != 0 {
		t.Fatalf("expired revoke code=%d stderr=%s", code, stderr.String())
	}
	cfg, err := readGuardRemoteMaintenanceConfigUnchecked(destination)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Profiles) != 1 || cfg.Profiles[0].ProfileID != "keep" {
		t.Fatalf("remaining profiles=%+v", cfg.Profiles)
	}
	if err := os.WriteFile(destination, []byte(`{"broken":`), 0o600); err != nil {
		t.Fatal(err)
	}
	stderr.Reset()
	if code := runGuardRemoteMaintenanceAt(&stdout, &stderr, destination, "*", 0, true, now); code != 0 {
		t.Fatalf("malformed clear-all code=%d stderr=%s", code, stderr.String())
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatalf("clear-all left malformed context: %v", err)
	}
}

// fak-test:runtime fast est=1s
func TestRemoteMaintenanceOperatorCLIRefusesGuardedSession(t *testing.T) {
	t.Setenv(guardAllowSessionIDEnv, "agent-session")
	var stdout, stderr bytes.Buffer
	if code := runGuardRemoteMaintenance(&stdout, &stderr, "reviewed.json", time.Minute, false); code == 0 {
		t.Fatal("guarded agent session installed operator authority")
	}
}
