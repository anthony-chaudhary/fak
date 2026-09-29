package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// binPath is the real crossauditcalibrate binary compiled once in TestMain. The
// tests drive it as a subprocess (exit code + stdout + stderr) so a deleted or
// stubbed main cannot pass vacuously.
var binPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "crossauditcalibrate-bin-*")
	if err != nil {
		panic(err)
	}
	exe := "crossauditcalibrate"
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	binPath = filepath.Join(dir, exe)
	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		_ = os.RemoveAll(dir)
		panic("build crossauditcalibrate: " + err.Error())
	}
	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}

// runBin executes the compiled binary and returns (exitCode, stdout, stderr).
func runBin(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	cmd := exec.Command(binPath, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		var ee *exec.ExitError
		if !errors.As(err, &ee) {
			t.Fatalf("run %v: %v", args, err)
		}
		code = ee.ExitCode()
	}
	return code, stdout.String(), stderr.String()
}

// auditIdentity returns a fully-provenanced arm auditor; the manifest validator
// refuses any arm with an incomplete provenance row.
func auditIdentity(provider, family, model, weights, posture, driver string) modelroute.AuditIdentity {
	return modelroute.AuditIdentity{
		Provider:         provider,
		Family:           family,
		Model:            model,
		WeightsRevision:  weights,
		ReasoningPosture: posture,
		Driver:           driver,
	}
}

// writeSuccessFixtures derives a valid calibration-run manifest plus one
// observation file per available arm from the package's real corpus truth, so a
// success case exercises the true delegation path rather than a mock.
func writeSuccessFixtures(t *testing.T) (string, []string) {
	t.Helper()
	truth, err := modelroute.AccidentalCalibrationTruth()
	if err != nil {
		t.Fatalf("AccidentalCalibrationTruth: %v", err)
	}
	if len(truth) == 0 {
		t.Fatal("corpus truth is empty; cannot build a success fixture")
	}
	dir := t.TempDir()

	armA := auditIdentity("openai", "gpt", "gpt-5.6-sol", "rev-a", "xhigh", "codex")
	armB := auditIdentity("anthropic", "claude", "opus", "rev-b", "high", "claude")

	observations := func(a modelroute.AuditIdentity, policyDigest string) []modelroute.CalibrationObservation {
		rows := make([]modelroute.CalibrationObservation, 0, len(truth))
		for _, row := range truth {
			verdict := modelroute.CrossAuditPass
			if row.Corrupt {
				verdict = modelroute.CrossAuditRefute
			}
			rows = append(rows, modelroute.CalibrationObservation{
				ID:            row.ID,
				Auditor:       a,
				Verdict:       verdict,
				BundleDigest:  row.BundleDigest,
				PolicyDigest:  policyDigest,
				PromptVersion: modelroute.CrossAuditPromptVersion,
				PromptDigest:  "sha256:0000000000000000000000000000000000000000000000000000000000000000",
				DurationNanos: 1000,
			})
		}
		return rows
	}

	specs := []modelroute.CalibrationArmSpec{
		{Name: "arm-a", Auditor: armA, Status: "available"},
		{Name: "arm-b", Auditor: armB, Status: "available"},
	}
	manifest := modelroute.CalibrationRunManifest{
		Schema:                modelroute.CrossAuditCalibrationRunSchema,
		Corpus:                modelroute.AccidentalCorpusManifestSchema,
		MaxSamplesPerArm:      len(truth) + 1,
		MaxCostMicrosUSD:      0,
		MinHighSeverityRecall: 0.9,
		MaxFalsePositiveRate:  0.1,
		Arms:                  specs,
	}
	mb, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	manifestPath := filepath.Join(dir, "run-manifest.json")
	if err := os.WriteFile(manifestPath, mb, 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	obsArgs := make([]string, 0, len(specs))
	for i, spec := range specs {
		policyDigest := "sha256:" + strings.Repeat(string(rune('1'+i)), 64)
		rows := observations(spec.Auditor, policyDigest)
		rb, err := json.MarshalIndent(rows, "", "  ")
		if err != nil {
			t.Fatalf("marshal observations: %v", err)
		}
		p := filepath.Join(dir, spec.Name+"-observations.json")
		if err := os.WriteFile(p, rb, 0o644); err != nil {
			t.Fatalf("write observations: %v", err)
		}
		obsArgs = append(obsArgs, spec.Name+"="+p)
	}
	return manifestPath, obsArgs
}

func TestNoArgsPrintsUsageToStderr(t *testing.T) {
	code, stdout, stderr := runBin(t)
	if code == 0 {
		t.Fatalf("exit code = 0, want non-zero for missing required flags")
	}
	if code != 2 {
		t.Errorf("exit code = %d, want 2", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "usage: crossauditcalibrate") {
		t.Errorf("stderr = %q, want the usage banner", stderr)
	}
}

func TestUnknownFlagRejected(t *testing.T) {
	code, stdout, stderr := runBin(t, "--definitely-not-a-flag")
	if code != 2 {
		t.Errorf("exit code = %d, want 2", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "flag provided but not defined") {
		t.Errorf("stderr = %q, want the flag-parse error", stderr)
	}
}

func TestSuccessfulDelegationWritesReport(t *testing.T) {
	manifest, obsArgs := writeSuccessFixtures(t)
	outPath := filepath.Join(t.TempDir(), "report.json")

	args := []string{"--manifest", manifest, "--out", outPath}
	for _, o := range obsArgs {
		args = append(args, "--observations", o)
	}

	code, stdout, stderr := runBin(t, args...)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%s", code, stderr)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want empty", stdout)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want empty", stderr)
	}
	raw, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report modelroute.CrossAuditCalibrationReport
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v\n%s", err, raw)
	}
	if report.Schema != modelroute.CrossAuditCalibrationSchema {
		t.Errorf("report schema = %q, want %q", report.Schema, modelroute.CrossAuditCalibrationSchema)
	}
	if len(report.Arms) != len(obsArgs) {
		t.Errorf("report arms = %d, want %d", len(report.Arms), len(obsArgs))
	}
	if report.CorpusDigest == "" {
		t.Error("report corpus digest is empty; delegation produced no content addressing")
	}
}

func TestMalformedManifestExitsOne(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "bad-manifest.json")
	if err := os.WriteFile(manifest, []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	obs := filepath.Join(dir, "obs.json")
	if err := os.WriteFile(obs, []byte("[]"), 0o644); err != nil {
		t.Fatal(err)
	}
	outPath := filepath.Join(dir, "report.json")

	code, stdout, stderr := runBin(t, "--manifest", manifest, "--observations", "a="+obs, "--out", outPath)
	if code != 1 {
		t.Errorf("exit code = %d, want 1 (engine error class)", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want empty", stdout)
	}
	if strings.TrimSpace(stderr) == "" {
		t.Error("stderr is empty, want the delegation error")
	}
	if _, err := os.Stat(outPath); !os.IsNotExist(err) {
		t.Errorf("report file exists after a failed run (stat err = %v)", err)
	}
}

func TestMissingObservationFilePanicsNonZero(t *testing.T) {
	dir := t.TempDir()
	// A well-formed manifest is required so the run reaches the observation read.
	manifestPath, _ := writeSuccessFixtures(t)
	missing := filepath.Join(dir, "does-not-exist.json")
	outPath := filepath.Join(dir, "report.json")

	code, _, stderr := runBin(t, "--manifest", manifestPath, "--observations", "a="+missing, "--out", outPath)
	if code == 0 {
		t.Fatal("exit code = 0, want non-zero for an unreadable observation file")
	}
	if !strings.Contains(stderr, "panic:") {
		t.Errorf("stderr = %q, want a panic from the unreadable observation file", stderr)
	}
}
