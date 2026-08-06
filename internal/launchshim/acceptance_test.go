package launchshim

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestProviderAcceptanceMatrix is intentionally stdlib-only and runs on every OS in
// .github/workflows/cross-platform.yml. The cmd layer owns fak/guard process launch;
// this package witnesses the platform-sensitive executable identity and lossless
// config lifecycle for both provider names.
func TestProviderAcceptanceMatrix(t *testing.T) {
	for _, provider := range []string{"claude", "codex"} {
		t.Run(provider, func(t *testing.T) {
			dir := t.TempDir()
			ext := ""
			if runtime.GOOS == "windows" {
				ext = ".cmd"
			}
			real := filepath.Join(dir, "real-"+provider+ext)
			body := []byte("#!/bin/sh\nexit 0\n")
			if runtime.GOOS == "windows" {
				body = []byte("@echo off\r\nexit /b 0\r\n")
			}
			if err := os.WriteFile(real, body, 0o755); err != nil {
				t.Fatal(err)
			}
			canonical, err := CanonicalCommand(real)
			if err != nil {
				t.Fatal(err)
			}
			config := filepath.Join(dir, "launch.json")
			t.Setenv("FAK_LAUNCH_CONFIG", config)
			if err := Save(Config{Default: provider, Providers: map[string]Provider{provider: {Command: real, Canonical: canonical}}}); err != nil {
				t.Fatal(err)
			}
			got, err := Load()
			if err != nil {
				t.Fatal(err)
			}
			if got.Default != provider || !SameCommand(got.Providers[provider].Canonical, real) {
				t.Fatalf("binding=%+v", got)
			}
			unrelated := filepath.Join(dir, "unrelated")
			want := []byte("keep-byte-identical")
			if err := os.WriteFile(unrelated, want, 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.Remove(config); err != nil {
				t.Fatal(err)
			}
			gotBytes, err := os.ReadFile(unrelated)
			if err != nil {
				t.Fatal(err)
			}
			if string(gotBytes) != string(want) {
				t.Fatalf("uninstall collateral=%q", gotBytes)
			}
			if _, err := os.Stat(real); err != nil {
				t.Fatalf("provider lost: %v", err)
			}
		})
	}
}
