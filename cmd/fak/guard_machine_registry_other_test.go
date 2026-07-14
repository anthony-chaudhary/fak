//go:build !windows

package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMachineGuardRegistryDirUsesInstalledOSServiceState(t *testing.T) {
	// The production path is intentionally absolute and cannot be redirected by
	// an untrusted environment variable. This test proves absence is fail-closed.
	old := os.Getenv("FLEET_REG_DIR")
	t.Setenv("FLEET_REG_DIR", filepath.Join(t.TempDir(), "attacker-controlled"))
	got := machineGuardRegistryDir()
	if got == os.Getenv("FLEET_REG_DIR") {
		t.Fatalf("machine registry trusted process environment: %q (old=%q)", got, old)
	}
	if got != "" && !filepath.IsAbs(got) {
		t.Fatalf("machine registry is not absolute: %q", got)
	}
}
