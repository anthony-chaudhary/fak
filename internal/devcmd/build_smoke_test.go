package devcmd

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

func init() {
	mode := os.Getenv("FAK_BUILD_SMOKE_HELPER")
	if mode == "" {
		return
	}
	if len(os.Args) < 2 {
		os.Exit(2)
	}
	verb := os.Args[1]
	if verb == "version" && (len(os.Args) != 3 || os.Args[2] != "--json") || verb == "runtime-capabilities" && len(os.Args) != 2 {
		os.Exit(2)
	}
	if mode == "null" {
		fmt.Fprintln(os.Stdout, "null")
		os.Exit(0)
	}
	if verb == "version" {
		switch mode {
		case "array":
			fmt.Fprintln(os.Stdout, "[]")
			os.Exit(0)
		case "empty-object":
			fmt.Fprintln(os.Stdout, "{}")
			os.Exit(0)
		case "invalid-json":
			fmt.Fprintln(os.Stdout, "{")
			os.Exit(0)
		case "timeout":
			time.Sleep(time.Minute)
		case "overflow":
			fmt.Fprint(os.Stdout, strings.Repeat(" ", buildSmokeOutputLimit+1))
		case "stderr-overflow":
			fmt.Fprint(os.Stderr, strings.Repeat("x", buildSmokeOutputLimit+1))
		case "warning":
			fmt.Fprintln(os.Stderr, "diagnostic warning")
		}
		_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
			"app_version": "test", "go": runtime.Version(), "os": runtime.GOOS,
			"arch": runtime.GOARCH, "stamped": false, "future_field": true,
		})
		if mode == "version-exit" {
			os.Exit(7)
		}
		os.Exit(0)
	}
	if verb != "runtime-capabilities" {
		os.Exit(2)
	}
	switch mode {
	case "runtime-timeout":
		time.Sleep(time.Minute)
	case "runtime-null":
		fmt.Fprintln(os.Stdout, "null")
		os.Exit(0)
	case "runtime-overflow":
		fmt.Fprint(os.Stdout, strings.Repeat(" ", buildSmokeOutputLimit+1))
	}
	report := map[string]any{
		"schema": "fak-runtime-capabilities/1", "binary_runnable": true,
		"control_plane_runnable": true, "goos": runtime.GOOS, "goarch": runtime.GOARCH,
		"model_execution": map[string]any{"runnable": false}, "future_field": true,
	}
	switch mode {
	case "runtime-schema":
		report["schema"] = "unrecognized"
	case "runtime-binary-false":
		report["binary_runnable"] = false
	case "runtime-control-false":
		report["control_plane_runnable"] = false
	case "runtime-platform":
		report["goarch"] = "other-architecture"
	}
	_ = json.NewEncoder(os.Stdout).Encode(report)
	if mode == "runtime-exit" {
		os.Exit(9)
	}
	os.Exit(0)
}

// fak-test:runtime fast est=100ms
func TestSmokeBuildArtifactRejectsNonIdentityJSON(t *testing.T) {
	path, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_BUILD_SMOKE_HELPER", "null")
	got := smokeBuildArtifact(path)
	if got.Outcome == "success" || got.ExitCode == 0 || got.Error == "" {
		t.Fatalf("non-identity JSON qualified: outcome=%s exit=%d error=%s", got.Outcome, got.ExitCode, got.Error)
	}
}

// fak-test:runtime medium est=4s
func TestSmokeBuildArtifactProcessContracts(t *testing.T) {
	path, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	failureCodes := map[string]string{
		"array": buildSmokeFailureIdentity, "empty-object": buildSmokeFailureIdentity,
		"invalid-json": buildSmokeFailureIdentity, "version-exit": buildSmokeFailureProcess,
		"timeout": buildSmokeFailureTimeout, "overflow": buildSmokeFailureOutput,
		"stderr-overflow": buildSmokeFailureOutput, "runtime-null": buildSmokeFailureCapabilities,
		"runtime-schema": buildSmokeFailureCapabilities, "runtime-binary-false": buildSmokeFailureCapabilities,
		"runtime-control-false": buildSmokeFailureCapabilities, "runtime-platform": buildSmokeFailurePlatform,
		"runtime-exit": buildSmokeFailureProcess, "runtime-timeout": buildSmokeFailureTimeout,
		"runtime-overflow": buildSmokeFailureOutput,
	}
	for _, mode := range []string{
		"valid", "warning", "array", "empty-object", "invalid-json", "version-exit",
		"timeout", "overflow", "stderr-overflow", "runtime-null", "runtime-schema",
		"runtime-binary-false", "runtime-control-false", "runtime-platform", "runtime-exit",
		"runtime-timeout", "runtime-overflow",
	} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("FAK_BUILD_SMOKE_HELPER", mode)
			if mode == "timeout" || mode == "runtime-timeout" {
				timeout := buildSmokeTimeout
				buildSmokeTimeout = 500 * time.Millisecond
				t.Cleanup(func() { buildSmokeTimeout = timeout })
			}
			start := time.Now()
			got := smokeBuildArtifact(path)
			success := mode == "valid" || mode == "warning"
			if success {
				if got.Outcome != "success" || got.ExitCode != 0 || got.Error != "" ||
					len(got.Command) != 3 || got.Command[1] != "version" || got.Command[2] != "--json" ||
					got.Runtime == nil || got.Runtime.Outcome != "success" ||
					len(got.Runtime.Command) != 2 || got.Runtime.Command[1] != "runtime-capabilities" {
					t.Fatalf("ordinary verbs failed: outcome=%s exit=%d code=%s error=%s", got.Outcome, got.ExitCode, got.FailureCode, got.Error)
				}
				if mode == "warning" && got.Stderr == "" {
					t.Fatal("stderr diagnostic was lost")
				}
				return
			}
			if got.Outcome == "success" || got.ExitCode == 0 || got.Error == "" {
				t.Fatalf("invalid artifact qualified (%s): outcome=%s exit=%d code=%s", mode, got.Outcome, got.ExitCode, got.FailureCode)
			}
			if got.FailureCode != failureCodes[mode] {
				t.Fatalf("failure code=%q, want %q", got.FailureCode, failureCodes[mode])
			}
			if strings.HasPrefix(mode, "runtime-") {
				var version map[string]any
				if json.Unmarshal([]byte(got.Output), &version) != nil || version["app_version"] != "test" ||
					got.Runtime == nil || got.Runtime.Outcome != "failed" || got.Runtime.FailureCode != failureCodes[mode] ||
					len(got.Runtime.Command) != 2 || got.Runtime.Command[1] != "runtime-capabilities" {
					t.Fatal("runtime failure lost version evidence or its own command/outcome")
				}
			}
			if mode == "version-exit" && got.ExitCode != 7 {
				t.Fatalf("version exit code=%d, want 7", got.ExitCode)
			}
			if mode == "runtime-exit" && got.ExitCode != 9 {
				t.Fatalf("runtime exit code=%d, want 9", got.ExitCode)
			}
			if len(got.Output) > buildSmokeOutputLimit || len(got.Stderr) > buildSmokeOutputLimit {
				t.Fatal("version output exceeded capture bound")
			}
			if got.Runtime != nil && (len(got.Runtime.Output) > buildSmokeOutputLimit || len(got.Runtime.Stderr) > buildSmokeOutputLimit) {
				t.Fatal("runtime output exceeded capture bound")
			}
			if mode == "timeout" || mode == "runtime-timeout" {
				if got.FailureCode != buildSmokeFailureTimeout || time.Since(start) > 3*time.Second {
					t.Fatalf("deadline was not reported/bounded: outcome=%s exit=%d error=%s elapsed=%s", got.Outcome, got.ExitCode, got.Error, time.Since(start))
				}
			}
			if strings.Contains(mode, "overflow") && got.FailureCode != buildSmokeFailureOutput {
				t.Fatalf("capture overflow did not fail explicitly: %s", got.Error)
			}
		})
	}
}
