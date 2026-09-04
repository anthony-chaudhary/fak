package main

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func TestGuardPostureFlags(t *testing.T) {
	if os.Getenv("TEST_GUARD_POSTURE_INVALID") == "1" {
		loadGuardCapabilityFloor("", "invalid_posture_value")
		return
	}

	t.Run("default posture without flag", func(t *testing.T) {
		t.Setenv("FAK_GUARD_POSTURE", "")
		rt, _, _, _ := loadGuardCapabilityFloor("")
		if rt.Adjudicator.Posture != adjudicator.PostureDefaultOpen {
			t.Fatalf("expected PostureDefaultOpen, got %v", rt.Adjudicator.Posture)
		}
	})

	t.Run("explicit posture fail_closed", func(t *testing.T) {
		t.Setenv("FAK_GUARD_POSTURE", "")
		rt, _, _, _ := loadGuardCapabilityFloor("", "fail_closed")
		if rt.Adjudicator.Posture != adjudicator.PostureFailClosed {
			t.Fatalf("expected PostureFailClosed, got %v", rt.Adjudicator.Posture)
		}
	})

	t.Run("explicit posture admit_and_log", func(t *testing.T) {
		t.Setenv("FAK_GUARD_POSTURE", "")
		rt, _, _, _ := loadGuardCapabilityFloor("", "admit_and_log")
		if rt.Adjudicator.Posture != adjudicator.PostureAdmitAndLog {
			t.Fatalf("expected PostureAdmitAndLog, got %v", rt.Adjudicator.Posture)
		}
	})

	t.Run("FAK_GUARD_POSTURE env var", func(t *testing.T) {
		t.Setenv("FAK_GUARD_POSTURE", "fail_closed")
		rt, _, _, _ := loadGuardCapabilityFloor("")
		if rt.Adjudicator.Posture != adjudicator.PostureFailClosed {
			t.Fatalf("expected PostureFailClosed, got %v", rt.Adjudicator.Posture)
		}
	})

	t.Run("flag overrides env var", func(t *testing.T) {
		t.Setenv("FAK_GUARD_POSTURE", "fail_closed")
		rt, _, _, _ := loadGuardCapabilityFloor("", "default_open")
		if rt.Adjudicator.Posture != adjudicator.PostureDefaultOpen {
			t.Fatalf("expected PostureDefaultOpen, got %v", rt.Adjudicator.Posture)
		}
	})

	t.Run("invalid posture fails loudly", func(t *testing.T) {
		cmd := exec.Command(os.Args[0], "-test.run=^TestGuardPostureFlags$")
		cmd.Env = append(os.Environ(), "TEST_GUARD_POSTURE_INVALID=1")
		out, err := cmd.CombinedOutput()
		if err == nil {
			t.Fatalf("expected invalid posture to fail loudly, but succeeded: %s", string(out))
		}
		if !strings.Contains(string(out), "unknown posture") {
			t.Fatalf("expected output to mention 'unknown posture', got: %s", string(out))
		}
	})
}
