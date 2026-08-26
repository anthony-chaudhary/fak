package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/systembaseline"
)

func TestSystemBaselineHelper(t *testing.T) {
	mode := os.Getenv("GO_WANT_SYSTEM_BASELINE_HELPER")
	if mode == "" {
		return
	}
	if mode == "timeout" {
		time.Sleep(5 * time.Second)
		return
	}
	if mode == "success" {
		time.Sleep(80 * time.Millisecond)
		return
	}
	time.Sleep(80 * time.Millisecond)
	os.Exit(7)
}

type failingSystemBaselineWriter struct{}

func (failingSystemBaselineWriter) Write([]byte) (int, error) {
	return 0, errors.New("injected stdout failure")
}

func TestBenchSystemBaselineReturnsFailureOnStdoutWriteError(t *testing.T) {
	t.Setenv("GO_WANT_SYSTEM_BASELINE_HELPER", "success")
	var stderr bytes.Buffer
	code := runBenchSystemBaseline(failingSystemBaselineWriter{}, &stderr, []string{"--interval", "20ms", "--baseline-duration", "20ms", "--max-sampler-duty-percent", "100", "--", os.Args[0], "-test.run=^TestSystemBaselineHelper$"})
	if code != 1 || !strings.Contains(stderr.String(), "write stdout") || !strings.Contains(stderr.String(), "injected stdout failure") {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
}

func TestBenchSystemBaselineCapturesFailedChild(t *testing.T) {
	t.Setenv("GO_WANT_SYSTEM_BASELINE_HELPER", "1")
	var stdout, stderr bytes.Buffer
	code := runBenchSystemBaseline(&stdout, &stderr, []string{"--interval", "20ms", "--baseline-duration", "40ms", "--max-sampler-duty-percent", "100", "--", os.Args[0], "-test.run=^TestSystemBaselineHelper$"})
	if code != 7 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var report systembaseline.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode: %v\n%s", err, stdout.String())
	}
	if report.CommandExitCode != 7 || report.Verdict != systembaseline.VerdictInvalid || report.Coverage.SUTRootPID <= 0 || report.Window.Samples < 2 || report.Baseline.Samples < 2 || report.Baseline.EndedAtUTC > report.Window.StartedAtUTC {
		t.Fatalf("report=%+v", report)
	}
	if report.Policy.MaximumSamplerDutyPercent != 100 || !report.BaselineSampler.DutyPercent.Available || !report.CommandSampler.DutyPercent.Available || report.BaselineSampler.CountedSamples != report.Baseline.Samples-1 || report.CommandSampler.CountedSamples != report.Window.Samples-1 {
		t.Fatalf("sampler overhead missing: baseline=%+v command=%+v policy=%+v", report.BaselineSampler, report.CommandSampler, report.Policy)
	}
	if err := report.Validate(); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, f := range report.Findings {
		if f.Code == "COMMAND_FAILED" {
			found = true
		}
	}
	if !found {
		t.Fatalf("failure finding absent: %+v", report.Findings)
	}
	if strings.Contains(stdout.String(), os.Args[0]) {
		t.Fatal("report leaked the child executable path")
	}
}

func TestBenchSystemBaselineTimeoutIsInvalid(t *testing.T) {
	t.Setenv("GO_WANT_SYSTEM_BASELINE_HELPER", "timeout")
	var stdout, stderr bytes.Buffer
	code := runBenchSystemBaseline(&stdout, &stderr, []string{"--interval", "10ms", "--baseline-duration", "20ms", "--timeout", "40ms", "--", os.Args[0], "-test.run=^TestSystemBaselineHelper$"})
	if code != 124 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var report systembaseline.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if !report.TimedOut || report.Verdict != systembaseline.VerdictInvalid || !hasSystemBaselineFinding(report, "COMMAND_TIMEOUT") {
		t.Fatalf("report=%+v", report)
	}
}

func TestBenchSystemBaselineVerifyRejectsTamper(t *testing.T) {
	t.Setenv("GO_WANT_SYSTEM_BASELINE_HELPER", "1")
	var stdout, stderr bytes.Buffer
	path := t.TempDir() + string(os.PathSeparator) + "attestation.json"
	if code := runBenchSystemBaseline(&stdout, &stderr, []string{"--interval", "20ms", "--baseline-duration", "20ms", "--out", path, "--", os.Args[0], "-test.run=^TestSystemBaselineHelper$"}); code != 7 {
		t.Fatalf("capture exit=%d stderr=%s", code, stderr.String())
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Fatalf("attestation mode=%o", info.Mode().Perm())
	}
	stdout.Reset()
	stderr.Reset()
	if code := runBenchSystemBaseline(&stdout, &stderr, []string{"--verify", path}); code != 0 || !strings.HasPrefix(stdout.String(), "VALID sha256:") {
		t.Fatalf("verify exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	raw = bytes.Replace(raw, []byte(`"command_exit_code": 7`), []byte(`"command_exit_code": 9`), 1)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := runBenchSystemBaseline(&stdout, &stderr, []string{"--verify", path}); code == 0 || !strings.Contains(stderr.String(), "canonical digest mismatch") {
		t.Fatalf("tamper exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
}

func hasSystemBaselineFinding(report systembaseline.Report, code string) bool {
	for _, finding := range report.Findings {
		if finding.Code == code {
			return true
		}
	}
	return false
}
