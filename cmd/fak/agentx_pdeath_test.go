package main

import (
	"bytes"
	"os"
	"strconv"
	"strings"
	"testing"
)

func TestParentDeathGuard(t *testing.T) {
	verdicts := []struct {
		name     string
		expected int
		observed int
		want     parentDeathAction
	}{
		{"no supervisor declared", 0, 4242, parentDeathSkip},
		{"negative expected is off", -1, 4242, parentDeathSkip},
		{"supervisor still parent", 4242, 4242, parentDeathContinue},
		{"reparented to init", 4242, 1, parentDeathExit},
		{"reparented to user subreaper", 4242, 977, parentDeathExit},
		{"observed ppid zero", 4242, 0, parentDeathExit},
	}
	for _, tc := range verdicts {
		t.Run("verdict/"+tc.name, func(t *testing.T) {
			if got := parentDeathVerdict(tc.expected, tc.observed); got != tc.want {
				t.Fatalf("parentDeathVerdict(%d, %d) = %v, want %v", tc.expected, tc.observed, got, tc.want)
			}
		})
	}

	envCases := []struct {
		value string
		want  int
	}{
		{"", 0},
		{"garbage", 0},
		{"-7", 0},
		{"31337", 31337},
	}
	for _, tc := range envCases {
		t.Run("env/"+tc.value, func(t *testing.T) {
			t.Setenv(agentxParentPIDEnv, tc.value)
			if got := agentxDefaultParentPID(); got != tc.want {
				t.Fatalf("agentxDefaultParentPID() with %q = %d, want %d", tc.value, got, tc.want)
			}
		})
	}

	live := os.Getppid()
	stranger := live + 1_000_003

	t.Run("install/live parent continues", func(t *testing.T) {
		var stderr bytes.Buffer
		if !installAgentXParentDeathGuard(&stderr, live) {
			t.Fatalf("guard refused the live parent %d: %s", live, stderr.String())
		}
	})
	t.Run("install/disabled continues", func(t *testing.T) {
		var stderr bytes.Buffer
		if !installAgentXParentDeathGuard(&stderr, 0) || stderr.Len() != 0 {
			t.Fatalf("disabled guard should be silent and continue, got %q", stderr.String())
		}
	})
	t.Run("install/dead supervisor exits", func(t *testing.T) {
		var stderr bytes.Buffer
		if installAgentXParentDeathGuard(&stderr, stranger) {
			t.Fatal("guard continued although the declared supervisor is not our parent")
		}
		if !strings.Contains(stderr.String(), "no longer the parent") {
			t.Fatalf("stderr missing orphan reason: %q", stderr.String())
		}
	})

	t.Run("agentx/refuses orphaned run", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runAgentX(&stdout, &stderr, []string{"--selfcheck", "--parent-pid", strconv.Itoa(stranger)})
		if code != 1 {
			t.Fatalf("runAgentX exit = %d, want 1; stderr=%q", code, stderr.String())
		}
		if stdout.Len() != 0 {
			t.Fatalf("orphaned run produced output: %q", stdout.String())
		}
	})
	t.Run("agentx/rejects negative parent pid", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := runAgentX(&stdout, &stderr, []string{"--parent-pid", "-2"}); code != 2 {
			t.Fatalf("runAgentX exit = %d, want 2; stderr=%q", code, stderr.String())
		}
	})
}
