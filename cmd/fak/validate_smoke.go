package main

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func validateSmokePolicyPath() string {
	return filepath.ToSlash(filepath.Join("examples", "customer-support-readonly-policy.json"))
}

// runValidateSmokePhase builds ./cmd/fak inside the isolated checkout directory
// and executes hermetic real-world CLI smoke tests to verify binary execution early.
func runValidateSmokePhase(ctx context.Context, stdout io.Writer, res *validateResult, recorder *validateRecorder, dir string, wslWorkspace, asJSON bool) (int, bool) {
	phase := recorder.start("smoke")

	smokeBin := "fak_smoke"
	if runtime.GOOS == "windows" && !wslWorkspace {
		smokeBin = "fak_smoke.exe"
	}
	smokePath := filepath.Join(dir, smokeBin)
	defer os.Remove(smokePath)

	detail, ok := validateRunGoCheckWithin(ctx, dir, wslWorkspace, "build", "-o", smokeBin, "./cmd/fak")
	if code, timedOut := finishValidateContextPhase(stdout, res, recorder, phase, "smoke", asJSON); timedOut {
		return code, true
	}
	if !ok {
		recordValidateFailure(res, phase, "smoke", detail, errors.New("smoke binary build failed"))
		return 0, false
	}

	// 1. Version check: verifies that binary links, loads dependencies, and runs cleanly.
	if out, ok := runValidateSmokeExec(ctx, dir, wslWorkspace, smokeBin, []string{"version"}); !ok {
		recordValidateFailure(res, phase, "smoke", out, errors.New("smoke check 'fak version' failed"))
		return 0, false
	}

	// 2. Preflight DENY check: verifies policy adjudication engine enforces security floor.
	policyRel := validateSmokePolicyPath()
	if out, ok := runValidateSmokeExec(ctx, dir, wslWorkspace, smokeBin, []string{"preflight", "--policy", policyRel, "--tool", "refund_payment", "--args", "{}"}); !ok || !strings.Contains(out, "verdict=DENY") {
		recordValidateFailure(res, phase, "smoke", out, errors.New("smoke check 'fak preflight' DENY failed"))
		return 0, false
	}

	// 3. Preflight ALLOW check: verifies non-blanket allow path executes.
	if out, ok := runValidateSmokeExec(ctx, dir, wslWorkspace, smokeBin, []string{"preflight", "--policy", policyRel, "--tool", "search_kb", "--args", "{}"}); !ok || !strings.Contains(out, "verdict=ALLOW") {
		recordValidateFailure(res, phase, "smoke", out, errors.New("smoke check 'fak preflight' ALLOW failed"))
		return 0, false
	}

	// 4. Agent offline check: verifies full mock agent turn execution and report output.
	tmpReport := filepath.Join(dir, "agent-smoke-report.json")
	defer os.Remove(tmpReport)
	tmpReportArg := filepath.ToSlash(tmpReport)
	if out, ok := runValidateSmokeExec(ctx, dir, wslWorkspace, smokeBin, []string{"agent", "--offline", "--out", tmpReportArg}); !ok || !strings.Contains(out, "booked") {
		recordValidateFailure(res, phase, "smoke", out, errors.New("smoke check 'fak agent --offline' failed"))
		return 0, false
	}

	phase.finish(nil)
	return 0, false
}

func runValidateSmokeExec(ctx context.Context, dir string, wslWorkspace bool, bin string, args []string) (string, bool) {
	if wslWorkspace {
		cmdArgs := append([]string{"./" + bin}, args...)
		out, err := runValidateWSLCommandWithin(ctx, filepath.ToSlash(dir), cmdArgs...)
		return strings.TrimSpace(string(out)), err == nil
	}
	target := filepath.Join(dir, bin)
	cmd := windowgate.CommandContext(ctx, target, args...)
	cmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(out)), err == nil
}
