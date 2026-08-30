package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/systembaseline"
)

const (
	systemBaselineHelperModeEnv   = "GO_WANT_SYSTEM_BASELINE_HELPER"
	systemBaselineHelperMarkerEnv = "GO_WANT_SYSTEM_BASELINE_MARKER"
	systemBaselineChurnChildren   = 12
)

func TestSystemBaselineHelper(t *testing.T) {
	mode := os.Getenv(systemBaselineHelperModeEnv)
	if mode == "" {
		cmd := exec.Command(os.Args[0], "-test.run=^TestSystemBaselineHelper$")
		cmd.Env = systemBaselineHelperEnv("success")
		if output, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("success helper failed: %v: %s", err, output)
		}
		return
	}
	switch mode {
	case "timeout":
		time.Sleep(5 * time.Second)
		return
	case "success":
		time.Sleep(80 * time.Millisecond)
		return
	case "windows_cleanup_grandchild":
		time.Sleep(30 * time.Second)
		return
	case "windows_cleanup_success", "windows_cleanup_failure", "windows_cleanup_timeout":
		child := exec.Command(os.Args[0], "-test.run=^TestSystemBaselineHelper$")
		child.Env = systemBaselineHelperEnv("windows_cleanup_grandchild")
		if err := child.Start(); err != nil {
			os.Exit(101)
		}
		marker := os.Getenv(systemBaselineHelperMarkerEnv)
		if marker == "" || os.WriteFile(marker, []byte(strconv.Itoa(child.Process.Pid)), 0o600) != nil {
			_ = child.Process.Kill()
			os.Exit(102)
		}
		if mode == "windows_cleanup_timeout" {
			time.Sleep(30 * time.Second)
			return
		}
		// Keep the root alive briefly so the parent test can retain a handle to
		// the exact grandchild process object before this root exits.
		time.Sleep(150 * time.Millisecond)
		if mode == "windows_cleanup_failure" {
			os.Exit(7)
		}
		return
	case "windows_churn_root":
		for range systemBaselineChurnChildren {
			child := exec.Command(os.Args[0], "-test.run=^TestSystemBaselineHelper$")
			child.Env = systemBaselineHelperEnv("windows_churn_child")
			if err := child.Run(); err != nil {
				os.Exit(103)
			}
		}
		return
	case "windows_churn_child":
		grandchild := exec.Command(os.Args[0], "-test.run=^TestSystemBaselineHelper$")
		grandchild.Env = systemBaselineHelperEnv("windows_churn_grandchild")
		if err := grandchild.Run(); err != nil {
			os.Exit(104)
		}
		return
	case "windows_churn_grandchild":
		return
	}
	time.Sleep(80 * time.Millisecond)
	os.Exit(7)
}

func systemBaselineHelperEnv(mode string) []string {
	prefix := systemBaselineHelperModeEnv + "="
	env := make([]string, 0, len(os.Environ())+1)
	for _, item := range os.Environ() {
		if !strings.HasPrefix(item, prefix) {
			env = append(env, item)
		}
	}
	return append(env, prefix+mode)
}

type failingSystemBaselineWriter struct{}

func (failingSystemBaselineWriter) Write([]byte) (int, error) {
	return 0, errors.New("injected stdout failure")
}

type recordingSystemBaselineAttributor struct {
	startedPID  int
	finishCalls int
}

func (*recordingSystemBaselineAttributor) Configure(*exec.Cmd) bool { return true }
func (*recordingSystemBaselineAttributor) Active() bool             { return true }
func (a *recordingSystemBaselineAttributor) Started(pid int) error {
	a.startedPID = pid
	return nil
}
func (*recordingSystemBaselineAttributor) LaunchFailed(error) {}
func (a *recordingSystemBaselineAttributor) FinishAttribution() systembaseline.CommandAttribution {
	a.finishCalls++
	reason := "injected cgroup counter fallback"
	metric := func(unit string) systembaseline.Metric {
		return systembaseline.Metric{Unit: unit, Reason: reason}
	}
	axis := systembaseline.PressureAxis{Some: metric("microseconds"), Full: metric("microseconds")}
	cgroup := systembaseline.CgroupV2{
		State:  systembaseline.CgroupStateUnavailable,
		Reason: reason,
		Membership: systembaseline.CgroupMembership{
			AtomicPlacement:  true,
			RootPID:          a.startedPID,
			AfterStart:       metric("processes"),
			AfterWait:        metric("processes"),
			PlacementSource:  "injected command-wrap seam",
			UnavailableCause: reason,
		},
		CPU: systembaseline.CounterSet{Reason: reason},
		Memory: systembaseline.CgroupMemory{
			CurrentBytes: metric("bytes"),
			PeakBytes:    metric("bytes"),
			Events:       systembaseline.CounterSet{Reason: reason},
		},
		Pressure: systembaseline.CgroupPressure{CPU: axis, Memory: axis, IO: axis},
		Cleanup:  systembaseline.CgroupCleanup{Attempted: true, Empty: true, Removed: true},
	}
	return systembaseline.CommandAttribution{CgroupV2: &cgroup}
}

func TestBenchSystemBaselineFinalizesCgroupForEveryCommandOutcome(t *testing.T) {
	tests := []struct {
		name, mode string
		timeout    string
		wantExit   int
	}{
		{name: "success", mode: "success", wantExit: 0},
		{name: "failure", mode: "failure", wantExit: 7},
		{name: "timeout", mode: "timeout", timeout: "40ms", wantExit: 124},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv("GO_WANT_SYSTEM_BASELINE_HELPER", test.mode)
			attributor := &recordingSystemBaselineAttributor{}
			args := []string{"--interval", "10ms", "--baseline-duration", "20ms", "--max-sampler-duty-percent", "100"}
			if test.timeout != "" {
				args = append(args, "--timeout", test.timeout)
			}
			args = append(args, "--", os.Args[0], "-test.run=^TestSystemBaselineHelper$")
			var stdout, stderr bytes.Buffer
			code := runBenchSystemBaselineWithAttributor(&stdout, &stderr, args, func() systemBaselineCommandAttributor { return attributor })
			if code != test.wantExit {
				t.Fatalf("exit=%d want=%d stderr=%s", code, test.wantExit, stderr.String())
			}
			var report systembaseline.Report
			if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
				t.Fatalf("decode: %v\n%s", err, stdout.String())
			}
			if attributor.startedPID <= 0 || attributor.finishCalls != 1 {
				t.Fatalf("started=%d finish_calls=%d", attributor.startedPID, attributor.finishCalls)
			}
			if report.CgroupV2 == nil || !report.CgroupV2.Cleanup.Attempted || !report.CgroupV2.Cleanup.Empty || !report.CgroupV2.Cleanup.Removed {
				t.Fatalf("cleanup receipt=%+v", report.CgroupV2)
			}
			if err := report.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestBenchSystemBaselineWindowsJobObjectCleansEveryTerminalPath(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows Job Object integration witness")
	}
	tests := []struct {
		name, mode string
		timeout    string
		wantExit   int
	}{
		{name: "success", mode: "windows_cleanup_success", wantExit: 0},
		{name: "failure", mode: "windows_cleanup_failure", wantExit: 7},
		{name: "timeout", mode: "windows_cleanup_timeout", timeout: "2s", wantExit: 124},
	}
	type runResult struct {
		code           int
		stdout, stderr string
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			marker := t.TempDir() + string(os.PathSeparator) + "grandchild.pid"
			t.Setenv(systemBaselineHelperModeEnv, test.mode)
			t.Setenv(systemBaselineHelperMarkerEnv, marker)
			args := []string{"--interval", "10ms", "--baseline-duration", "20ms", "--max-sampler-duty-percent", "100"}
			if test.timeout != "" {
				args = append(args, "--timeout", test.timeout)
			}
			args = append(args, "--", os.Args[0], "-test.run=^TestSystemBaselineHelper$")
			done := make(chan runResult, 1)
			go func() {
				var stdout, stderr bytes.Buffer
				code := runBenchSystemBaseline(&stdout, &stderr, args)
				done <- runResult{code: code, stdout: stdout.String(), stderr: stderr.String()}
			}()

			var grandchild *os.Process
			deadline := time.Now().Add(10 * time.Second)
			for grandchild == nil && time.Now().Before(deadline) {
				raw, err := os.ReadFile(marker)
				if err == nil && len(strings.TrimSpace(string(raw))) > 0 {
					pid, parseErr := strconv.Atoi(strings.TrimSpace(string(raw)))
					if parseErr != nil {
						t.Fatalf("parse grandchild PID %q: %v", raw, parseErr)
					}
					grandchild, err = os.FindProcess(pid)
					if err != nil {
						t.Fatalf("retain grandchild process %d: %v", pid, err)
					}
					break
				}
				select {
				case result := <-done:
					t.Fatalf("bench exited before grandchild could be retained: code=%d stderr=%s", result.code, result.stderr)
				case <-time.After(5 * time.Millisecond):
				}
			}
			if grandchild == nil {
				t.Fatal("timed out waiting for grandchild PID marker")
			}
			grandchildDone := make(chan error, 1)
			go func() {
				_, err := grandchild.Wait()
				grandchildDone <- err
			}()

			var result runResult
			select {
			case result = <-done:
			case <-time.After(15 * time.Second):
				_ = grandchild.Kill()
				t.Fatal("bench did not finish after the root terminal path")
			}
			if result.code != test.wantExit {
				t.Fatalf("exit=%d want=%d stderr=%s", result.code, test.wantExit, result.stderr)
			}
			select {
			case err := <-grandchildDone:
				if err != nil {
					t.Fatalf("wait retained grandchild: %v", err)
				}
			case <-time.After(5 * time.Second):
				_ = grandchild.Kill()
				t.Fatal("grandchild survived Job Object cleanup")
			}

			var report systembaseline.Report
			if err := json.Unmarshal([]byte(result.stdout), &report); err != nil {
				t.Fatalf("decode: %v\nstdout=%s\nstderr=%s", err, result.stdout, result.stderr)
			}
			job := report.WindowsJobObject
			if report.Coverage.DescendantAttribution != "job_object" || job == nil || job.State != systembaseline.WindowsJobStateMeasured {
				t.Fatalf("coverage=%q job=%+v", report.Coverage.DescendantAttribution, job)
			}
			if !job.Membership.AtomicPlacement || job.Membership.RootStartID == 0 || !job.Cleanup.Attempted || !job.Cleanup.KilledRemaining || !job.Cleanup.Empty || !job.Cleanup.Closed {
				t.Fatalf("membership=%+v cleanup=%+v", job.Membership, job.Cleanup)
			}
			if err := report.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestBenchSystemBaselineWindowsJobObjectAttributesRapidChurn(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows Job Object integration witness")
	}
	t.Setenv(systemBaselineHelperModeEnv, "windows_churn_root")
	var stdout, stderr bytes.Buffer
	code := runBenchSystemBaseline(&stdout, &stderr, []string{
		"--interval", "10ms",
		"--baseline-duration", "20ms",
		"--max-sampler-duty-percent", "100",
		"--", os.Args[0], "-test.run=^TestSystemBaselineHelper$",
	})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var report systembaseline.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode: %v\n%s", err, stdout.String())
	}
	job := report.WindowsJobObject
	// The job can also contain Windows console infrastructure. The load-bearing
	// bound is that none of the known root/child/grandchild processes disappeared
	// between Toolhelp samples.
	wantProcesses := uint64(1 + 2*systemBaselineChurnChildren)
	if report.Coverage.DescendantAttribution != "job_object" || job == nil || job.State != systembaseline.WindowsJobStateMeasured {
		t.Fatalf("coverage=%q job=%+v", report.Coverage.DescendantAttribution, job)
	}
	if got := job.Processes.Values["total_processes"]; got < wantProcesses {
		t.Fatalf("job lifetime process count=%d want at least %d (100%% root/child/grandchild attribution)", got, wantProcesses)
	}
	if job.Membership.RootStartID == 0 || !job.CPU.Available || !job.Memory.PeakJobCommitBytes.Available || !job.Cleanup.Empty || !job.Cleanup.Closed {
		t.Fatalf("job receipt=%+v", job)
	}
	if err := report.Validate(); err != nil {
		t.Fatal(err)
	}
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
