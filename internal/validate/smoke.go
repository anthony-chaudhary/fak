package validate

import (
	"context"
	"encoding/json"
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

	// 4. Product-entry check: exercise the real fak up planner that the released
	// binary uses before model acquisition. Mock/offline demo output is not an
	// onboarding witness; the release qualification owns the weight-backed first
	// response on each native OS (#14508).
	if out, ok := runValidateSmokeExec(ctx, dir, wslWorkspace, smokeBin, validateSmokeUpArgs()); !ok || !validateSmokeUpOutput(out) {
		recordValidateFailure(res, phase, "smoke", out, errors.New("smoke check 'fak up --dry-run' failed"))
		return 0, false
	}

	phase.finish(nil)
	return 0, false
}

func validateSmokeUpArgs() []string {
	return []string{"up", "--model", "3B", "--dry-run", "--json", "--memory-gib", "32"}
}

func validateSmokeUpOutput(out string) bool {
	var plan struct {
		Schema string `json:"schema"`
		Tier   struct {
			Name string `json:"name"`
		} `json:"tier"`
	}
	return json.Unmarshal([]byte(out), &plan) == nil && plan.Schema == "fak-macfit-turnkey/1" && plan.Tier.Name == "3B"
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
