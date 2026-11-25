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

const smokeUpPlanOK = "{\n  \"schema\": \"fak-macfit-turnkey/1\",\n  \"tier\": {\n    \"name\": \"3B\"\n  }\n}\n"

func TestSmokeExecRequiresFreshUpPlan(t *testing.T) {
	if _, err := exec.LookPath("make"); err != nil {
		t.Skip("make is required to execute the smoke-exec recipe")
	}
	helper := fmt.Sprintf("%q -test.run=^TestHelperProcess$$ --", os.Args[0])
	cases := []struct {
		name     string
		mode     string
		stale    bool
		wantOK   bool
		wantDiag string
	}{
		{name: "nonzero child", mode: "fail"},
		{name: "zero without plan", mode: "missing", wantDiag: "emitted no stable plan schema"},
		{name: "zero cannot reuse stale plan", mode: "missing", stale: true, wantDiag: "emitted no stable plan schema"},
		{name: "wrong schema", mode: "wrong-schema", wantDiag: "emitted no stable plan schema"},
		{name: "wrong tier", mode: "wrong-tier", wantDiag: "did not select the README 3B tier"},
		{name: "fresh plan", mode: "success", wantOK: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			planPath := filepath.Join(t.TempDir(), "plan", "smoke-up-plan.txt")
			if tc.stale {
				if err := os.MkdirAll(filepath.Dir(planPath), 0o700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(planPath, []byte(smokeUpPlanOK), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			cmd := exec.Command("make", "-C", repoRoot(t), "-o", "build", "smoke-exec",
				"SMOKE_FAK="+helper, "SMOKE_UP_PLAN="+planPath)
			cmd.Env = append(os.Environ(),
				"GO_WANT_SMOKE_EXEC_HELPER=1",
				"SMOKE_EXEC_HELPER_UP_MODE="+tc.mode,
			)
			out, runErr := cmd.CombinedOutput()
			marker := "SMOKE_EXEC_HELPER_UP:" + tc.mode
			if !strings.Contains(string(out), marker) {
				t.Fatalf("smoke-exec did not reach the injected up planner (%s):\n%s", marker, out)
			}
			if tc.wantOK && runErr != nil {
				t.Fatalf("smoke-exec failed with a fresh 3B plan: %v\n%s", runErr, out)
			}
			if !tc.wantOK && runErr == nil {
				t.Fatalf("smoke-exec succeeded without a fresh 3B plan:\n%s", out)
			}
			if got := strings.Contains(string(out), "smoke-exec OK"); got != tc.wantOK {
				t.Fatalf("success marker = %v, want %v:\n%s", got, tc.wantOK, out)
			}
			if tc.wantDiag != "" && !strings.Contains(string(out), tc.wantDiag) {
				t.Fatalf("missing diagnostic %q:\n%s", tc.wantDiag, out)
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
	case "up":
		// The smoke must only plan: a recipe that drops --dry-run would start a real server.
		if !slices.Contains(args, "--dry-run") || !slices.Contains(args, "--json") {
			os.Exit(2)
		}
		mode := os.Getenv("SMOKE_EXEC_HELPER_UP_MODE")
		fmt.Fprintln(os.Stderr, "SMOKE_EXEC_HELPER_UP:"+mode)
		switch mode {
		case "fail":
			fmt.Print(smokeUpPlanOK)
			os.Exit(23)
		case "missing":
		case "wrong-schema":
			fmt.Print(strings.Replace(smokeUpPlanOK, "fak-macfit-turnkey/1", "fak-macfit-turnkey/0", 1))
		case "wrong-tier":
			fmt.Print(strings.Replace(smokeUpPlanOK, "\"3B\"", "\"1B\"", 1))
		case "success":
			fmt.Print(smokeUpPlanOK)
		default:
			os.Exit(2)
		}
	default:
		os.Exit(2)
	}
}
