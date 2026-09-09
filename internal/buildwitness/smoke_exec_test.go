package buildwitness

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestSmokeExecPropagatesOfflineAgentFailure(t *testing.T) {
	if _, err := exec.LookPath("make"); err != nil {
		t.Skip("make is required to execute the smoke-exec recipe")
	}
	helper := fmt.Sprintf("%q -test.run=^TestHelperProcess$$ --", os.Args[0])
	cases := []struct {
		name       string
		mode       string
		stale      bool
		wantOK     bool
		wantReport bool
	}{
		{name: "nonzero child", mode: "fail"},
		{name: "zero without report", mode: "missing"},
		{name: "zero cannot reuse stale report", mode: "missing", stale: true},
		{name: "fresh incomplete report", mode: "incomplete", wantReport: true},
		{name: "completed fresh report", mode: "success", wantOK: true, wantReport: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reportPath := filepath.Join(t.TempDir(), "smoke-agent.json")
			if tc.stale {
				stale := []byte("{\n  \"both_completed\": true,\n  \"live\": false\n}\n")
				if err := os.WriteFile(reportPath, stale, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			cmd := exec.Command("make", "-C", repoRoot(t), "-o", "build", "smoke-exec",
				"SMOKE_FAK="+helper, "SMOKE_AGENT_REPORT="+reportPath)
			cmd.Env = append(os.Environ(),
				"GO_WANT_SMOKE_EXEC_HELPER=1",
				"SMOKE_EXEC_HELPER_AGENT_MODE="+tc.mode,
			)
			out, runErr := cmd.CombinedOutput()
			marker := "SMOKE_EXEC_HELPER_AGENT:" + tc.mode
			if !strings.Contains(string(out), marker) {
				t.Fatalf("smoke-exec did not reach the injected offline agent (%s):\n%s", marker, out)
			}
			if tc.wantOK && runErr != nil {
				t.Fatalf("smoke-exec failed with a completed fresh report: %v\n%s", runErr, out)
			}
			if !tc.wantOK && runErr == nil {
				t.Fatalf("smoke-exec succeeded without a completed fresh report:\n%s", out)
			}
			if got := strings.Contains(string(out), "smoke-exec OK"); got != tc.wantOK {
				t.Fatalf("success marker = %v, want %v:\n%s", got, tc.wantOK, out)
			}
			_, statErr := os.Stat(reportPath)
			if got := statErr == nil; got != tc.wantReport {
				t.Fatalf("fresh report present = %v, want %v (stat error: %v)", got, tc.wantReport, statErr)
			}
		})
	}
}

func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_SMOKE_EXEC_HELPER") != "1" {
		return
	}
	separator := slices.Index(os.Args, "--")
	if separator < 0 || separator+1 >= len(os.Args) {
		os.Exit(2)
	}
	args := os.Args[separator+1:]
	switch args[0] {
	case "version":
		fmt.Println("test fak version")
	case "preflight":
		for i, arg := range args {
			if arg == "--tool" && i+1 < len(args) {
				if args[i+1] == "refund_payment" {
					fmt.Println("verdict=DENY")
				} else {
					fmt.Println("verdict=ALLOW")
				}
				return
			}
		}
		os.Exit(2)
	case "agent":
		mode := os.Getenv("SMOKE_EXEC_HELPER_AGENT_MODE")
		fmt.Fprintln(os.Stderr, "SMOKE_EXEC_HELPER_AGENT:"+mode)
		switch mode {
		case "fail":
			os.Exit(23)
		case "missing":
			return
		case "success", "incomplete":
			for i, arg := range args {
				if arg == "--out" && i+1 < len(args) {
					completed := mode == "success"
					report := []byte(fmt.Sprintf("{\n  \"fak\": {\"task_completed\": %t},\n  \"baseline\": {\"task_completed\": %t},\n  \"both_completed\": %t,\n  \"live\": false\n}\n", completed, completed, completed))
					if err := os.WriteFile(args[i+1], report, 0o600); err != nil {
						os.Exit(1)
					}
					return
				}
			}
			os.Exit(2)
		default:
			os.Exit(2)
		}
	default:
		os.Exit(2)
	}
}
