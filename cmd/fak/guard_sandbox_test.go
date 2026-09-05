package main

import (
	"flag"
	"os"
	"strings"
	"testing"
)

func TestGuardSandboxFlag(t *testing.T) {
	fs := flag.NewFlagSet("guard", flag.ContinueOnError)
	sandboxMode := fs.String("sandbox", "auto", "execution sandbox tier for tool dispatches: auto|runsc|l1|wasi|off (default: auto; sets FAK_SANDBOX_TIER)")

	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty flags: %v", err)
	}
	if *sandboxMode != "auto" {
		t.Fatalf("expected default sandbox mode 'auto', got %q", *sandboxMode)
	}

	testCases := []struct {
		val  string
		want string
	}{
		{"runsc", "runsc"},
		{"l1", "l1"},
		{"wasi", "wasi"},
		{"off", "off"},
	}

	for _, tc := range testCases {
		origEnv := os.Getenv("FAK_SANDBOX_TIER")
		t.Cleanup(func() {
			_ = os.Setenv("FAK_SANDBOX_TIER", origEnv)
		})

		fsIter := flag.NewFlagSet("guard", flag.ContinueOnError)
		sb := fsIter.String("sandbox", "auto", "execution sandbox tier for tool dispatches: auto|runsc|l1|wasi|off (default: auto; sets FAK_SANDBOX_TIER)")
		if err := fsIter.Parse([]string{"--sandbox", tc.val}); err != nil {
			t.Fatalf("failed to parse --sandbox %s: %v", tc.val, err)
		}
		if *sb != tc.want {
			t.Errorf("sandboxMode = %q, want %q", *sb, tc.want)
		}

		if *sb != "" {
			_ = os.Setenv("FAK_SANDBOX_TIER", *sb)
		}
		if got := os.Getenv("FAK_SANDBOX_TIER"); got != tc.want {
			t.Errorf("FAK_SANDBOX_TIER = %q, want %q", got, tc.want)
		}
	}
}

func TestGuardSandboxFlagDeclaredOnManageCommand(t *testing.T) {
	guardSrc := readEntrypoint(t, "guard.go")
	if !strings.Contains(guardSrc, `sandboxMode := fs.String("sandbox", "auto"`) {
		t.Errorf("guard.go must register --sandbox flag in cmdManageCommand")
	}
	if !strings.Contains(guardSrc, `os.Setenv("FAK_SANDBOX_TIER", *sandboxMode)`) {
		t.Errorf("guard.go must set FAK_SANDBOX_TIER from parsed sandbox flag")
	}
}
