package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=500ms lane=default
func TestServeDFlash2RefusesBeforeAcquisition(t *testing.T) {
	const child = "FAK_TEST_DFLASH2_STARTUP_CHILD"
	if os.Getenv(child) == "1" {
		cmdServe([]string{
			"--gguf=" + os.Getenv("FAK_TEST_DFLASH2_MISSING_MODEL"),
			"--backend=dflash2-test-unregistered-backend",
			"--speculative=" + os.Getenv("FAK_TEST_DFLASH2_FLAG"),
			"--keep-awake=off",
		})
		return
	}
	for _, tc := range []struct {
		name, flag, fak, legacy string
	}{
		{"flag", " DFlash2 ", "", ""},
		{"primary-env", "mtp", " DFlash2 ", ""},
		{"legacy-env", "mtp", "", "dflash2"},
		{"conflicting-env", "mtp", "mtp", "dflash2"},
		{"primary-conflict", "mtp", "dflash2", "ngram"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeDFlash2RefusesBeforeAcquisition$")
			for _, entry := range os.Environ() {
				key, _, _ := strings.Cut(entry, "=")
				if key != "FAK_SPECULATIVE" && key != "SPECULATIVE" && key != child {
					cmd.Env = append(cmd.Env, entry)
				}
			}
			cmd.Env = append(cmd.Env, child+"=1", "FAK_TEST_DFLASH2_FLAG="+tc.flag,
				"FAK_SPECULATIVE="+tc.fak, "SPECULATIVE="+tc.legacy,
				"FAK_TEST_DFLASH2_MISSING_MODEL="+filepath.Join(t.TempDir(), "missing.gguf"))
			out, err := cmd.CombinedOutput()
			var exit *exec.ExitError
			if !errors.As(err, &exit) || exit.ExitCode() != 2 {
				t.Fatalf("startup exit = %v, want 2; output: %s", err, out)
			}
			if !strings.Contains(string(out), "DFlash2 serve startup requires an authenticated target owner") {
				t.Fatalf("startup reached an earlier acquisition/selection failure instead of refusing unowned DFlash2: %s", out)
			}
		})
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestServeDFlash2PreservesModelFreeInspection(t *testing.T) {
	const child = "FAK_TEST_DFLASH2_INSPECTION_CHILD"
	if mode := os.Getenv(child); mode != "" {
		// The malformed HF source must never reach source resolution or transport.
		args := []string{"--speculative=dflash2", "--gguf=hf://", "--keep-awake=off"}
		if mode == "policy-check" {
			args = append(args, "--policy-check", "--policy="+os.Getenv("FAK_TEST_DFLASH2_POLICY"))
		} else {
			args = append(args, "--print-effective-config")
		}
		cmdServe(args)
		return
	}
	policy := filepath.Join(t.TempDir(), "policy.json")
	if err := os.WriteFile(policy, []byte(minimalPolicyManifest), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"policy-check", "print-effective-config"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeDFlash2PreservesModelFreeInspection$")
			cmd.Env = append(os.Environ(), child+"="+mode, "FAK_TEST_DFLASH2_POLICY="+policy,
				"FAK_SPECULATIVE=", "SPECULATIVE=")
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("model-free inspection failed or reached model acquisition: %v; output: %s", err, out)
			}
			if mode == "policy-check" {
				if !strings.Contains(string(out), "manifest valid;") {
					t.Fatalf("missing policy validation: %s", out)
				}
			} else {
				line := strings.TrimSpace(strings.SplitN(string(out), "\n", 2)[0])
				var report effectiveServeConfigReport
				if err := json.Unmarshal([]byte(line), &report); err != nil {
					t.Fatalf("invalid effective configuration: %v; output: %s", err, out)
				}
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestServeDFlash2StartupPreservesOtherSelectors(t *testing.T) {
	for _, tc := range []struct {
		flag, fak, legacy, want string
	}{
		{"mtp", "", "", "mtp"},
		{"ngram", "", "", "ngram"},
		{"", "", "", ""},
		{"dflash2", "mtp", "", "mtp"},
		{"dflash2", "", "ngram", "ngram"},
		{"mtp", "ngram", "mtp", "ngram"},
	} {
		getenv := func(key string) string {
			if key == "FAK_SPECULATIVE" {
				return tc.fak
			}
			return tc.legacy
		}
		mode, err := resolveSpeculativeMode(tc.flag, getenv)
		if err != nil || mode != tc.want {
			t.Fatalf("resolve %+v = %q, %v", tc, mode, err)
		}
		if err := validateServeDFlash2Startup(mode, getenv); err != nil {
			t.Fatalf("ordinary/overridden selector %+v refused: %v", tc, err)
		}
	}
	if err := validateServeDFlash2Startup("dflash2", func(string) string { return "" }); !errors.Is(err, ErrServeDFlash2OwnerUnavailable) {
		t.Fatalf("unowned DFlash2 = %v, want typed startup refusal", err)
	}
}
