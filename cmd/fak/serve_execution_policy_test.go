package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

func cpuOnlyServeLookup(string) (compute.Backend, error) { return nil, errors.New("not linked") }

// fak-test:runtime fast est=2ms lane=default
func TestServeExecutionPolicyProductionKeepsHaloCPUOffloadRefusal(t *testing.T) {
	for _, name := range []string{"", "production", " Production "} {
		p, err := resolveServeExecutionPolicy(name, "qwen38-flash-next", "qwen38-flash-next", nil)
		if err != nil || p.Admitted || p.Name != serveExecutionPolicyProduction {
			t.Fatalf("policy %q: %+v err=%v, want unadmitted production", name, p, err)
		}
		var halo *haloGPURequiredError
		if err := p.admit(validateServeHaloCPUOffload(true, true, "", "")); !errors.As(err, &halo) || halo.Reason != haloGPUReasonCPUExpertOffload {
			t.Fatalf("policy %q offload: err=%v, want %s", name, err, haloGPUReasonCPUExpertOffload)
		}
		if _, err := resolveServeChatBackendForHostPolicy("cpu", "", "linux", cpuOnlyServeLookup, true, p); !errors.As(err, &halo) || halo.Reason != haloGPUReasonCPUBackend {
			t.Fatalf("policy %q cpu backend: err=%v, want %s", name, err, haloGPUReasonCPUBackend)
		}
		if !p.requireDeviceExecution(true) || len(p.Bypassed) != 0 || p.report() != nil {
			t.Fatalf("policy %q: production lifted a guard: %+v", name, p)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeExecutionPolicyBringupWithoutDedicationRefuses(t *testing.T) {
	for _, tc := range []struct{ model, dedicated string }{
		{"qwen38-flash-next", ""},
		{"qwen38-flash-next", "deepseek-v41-flash"},
		{"qwen38-flash-next", "qwen38-flash"},
		{"", "qwen38-flash-next"},
	} {
		_, err := resolveServeExecutionPolicy("bringup", tc.model, tc.dedicated, nil)
		var pe *executionPolicyError
		if !errors.As(err, &pe) || pe.Code != ExecutionPolicyNotDedicated {
			t.Fatalf("model=%q dedicated=%q: err=%v, want %s", tc.model, tc.dedicated, err, ExecutionPolicyNotDedicated)
		}
	}
	_, err := resolveServeExecutionPolicy("device-off", "m", "m", nil)
	var pe *executionPolicyError
	if !errors.As(err, &pe) || pe.Code != ExecutionPolicyUnknown {
		t.Fatalf("unknown policy: err=%v, want %s", err, ExecutionPolicyUnknown)
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestServeExecutionPolicyBringupDedicatedAdmitsHaloCPUOffload(t *testing.T) {
	var log bytes.Buffer
	p, err := resolveServeExecutionPolicy("bringup", "qwen38-flash-next", "deepseek-v41-flash, qwen38-flash-next", &log)
	if err != nil || !p.Admitted {
		t.Fatalf("dedicated bringup: %+v err=%v, want admitted", p, err)
	}
	if be, err := resolveServeChatBackendForHostPolicy("cpu", "", "linux", cpuOnlyServeLookup, true, p); err != nil || be != nil {
		t.Fatalf("cpu backend: be=%v err=%v, want host execution admitted", be, err)
	}
	if err := p.admit(validateServeHaloCPUOffload(true, true, "", "")); err != nil {
		t.Fatalf("cpu offload: %v, want admitted", err)
	}
	if err := p.admit(validateServeHaloCPUOffload(true, false, "bogus", "")); err == nil {
		t.Fatal("malformed n-cpu-moe grade must still refuse under bringup")
	}
	if p.requireDeviceExecution(true) {
		t.Fatal("bringup must lift the device-only session policy on a halo")
	}
	want := []string{haloGPUReasonCPUBackend, haloGPUReasonCPUExpertOffload, executionPolicyBypassDeviceOnlySession}
	if !slices.Equal(p.Bypassed, want) {
		t.Fatalf("bypassed=%v, want %v", p.Bypassed, want)
	}
	r := p.report()
	if r == nil || r.Policy != serveExecutionPolicyBringup || r.Model != "qwen38-flash-next" || !r.Dedicated || !slices.Equal(r.Bypassed, want) {
		t.Fatalf("report=%+v", r)
	}
	for _, token := range want {
		if !strings.Contains(log.String(), token) {
			t.Fatalf("stderr log lacks bypassed token %s:\n%s", token, log.String())
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeExecutionPolicyBringupDeviceBackendKeepsDeviceOnly(t *testing.T) {
	p, err := resolveServeExecutionPolicy("bringup", "m", "m", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.admit(validateServeHaloCPUOffload(true, false, "", "")); err != nil {
		t.Fatal(err)
	}
	if !p.requireDeviceExecution(true) || len(p.Bypassed) != 0 {
		t.Fatalf("bringup with nothing host-side lifted: bypassed=%v, want device-only kept (expert ring stays admissible)", p.Bypassed)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeExecutionPolicyBringupOffHaloBypassesNothing(t *testing.T) {
	p, err := resolveServeExecutionPolicy("bringup", "m", "m", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.admit(validateServeHaloCPUOffload(false, true, "", "")); err != nil || p.requireDeviceExecution(false) || len(p.Bypassed) != 0 {
		t.Fatalf("off-halo bringup: err=%v bypassed=%v, want nothing to lift", err, p.Bypassed)
	}
}

// fak-test:runtime slow est=3s lane=default
func TestServeDryAdmitExecutionPolicyVerdict(t *testing.T) {
	if policy, ok := os.LookupEnv("FAK_TEST_DRY_ADMIT_POLICY"); ok {
		args := []string{"--dry-admit", "--model", "mock", "--backend", "cpu", "--execution-policy", policy, "--session-registry", "off"}
		cmdServe(append(args, strings.Fields(os.Getenv("FAK_TEST_DRY_ADMIT_EXTRA"))...))
		return
	}
	extra := ""
	run := func(policy, dedicated string) (string, string, int) {
		cmd := exec.Command(os.Args[0], "-test.run=^TestServeDryAdmitExecutionPolicyVerdict$")
		cmd.Env = append(os.Environ(), "FAK_TEST_DRY_ADMIT_POLICY="+policy, ServeBringupDedicatedEnv+"="+dedicated, "FAK_BACKEND=", "FAK_METAL=", "FAK_SESSION_STATE=off", "FAK_TEST_DRY_ADMIT_EXTRA="+extra)
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		err := cmd.Run()
		code := 0
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			code = exit.ExitCode()
		} else if err != nil {
			t.Fatal(err)
		}
		return stdout.String(), stderr.String(), code
	}
	verdict := func(out string) serveDryAdmitVerdict {
		t.Helper()
		start := strings.Index(out, "{")
		var v serveDryAdmitVerdict
		if start < 0 || json.NewDecoder(strings.NewReader(out[start:])).Decode(&v) != nil {
			t.Fatalf("no dry-admit verdict on stdout:\n%s", out)
		}
		return v
	}

	out, stderr, code := run("production", "")
	if v := verdict(out); code != 0 || v.Schema != serveDryAdmitSchema || v.Verdict != serveDryAdmitAdmit || v.ExecutionPolicy != serveExecutionPolicyProduction || v.Backend != "cpu" || v.LoadFit != serveDryAdmitLoadFitNotRun {
		t.Fatalf("production: code=%d verdict=%+v stderr=%s", code, v, stderr)
	}
	if _, stderr, code := run("bringup", ""); code != 2 || !strings.Contains(stderr, ExecutionPolicyNotDedicated) {
		t.Fatalf("undedicated bringup: code=%d stderr=%s, want exit 2 %s", code, stderr, ExecutionPolicyNotDedicated)
	}
	out, stderr, code = run("bringup", "mock")
	if v := verdict(out); code != 0 || v.Verdict != serveDryAdmitAdmit || v.ExecutionPolicy != serveExecutionPolicyBringup || v.Policy == nil || v.Policy.Model != "mock" {
		t.Fatalf("dedicated bringup: code=%d verdict=%+v stderr=%s", code, v, stderr)
	}
	extra = "--qwen38-runtime llama-mtp"
	if out, stderr, code := run("production", ""); code != 2 || !strings.Contains(stderr, DryAdmitDelegatedRuntime) || strings.Contains(out, serveDryAdmitAdmit) {
		t.Fatalf("delegated runtime: code=%d stderr=%s, want exit 2 %s", code, stderr, DryAdmitDelegatedRuntime)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeExecutionPolicyBringupBypassIsSessionScoped(t *testing.T) {
	prod, err := resolveServeExecutionPolicy("production", "m", "m", nil)
	if err != nil || prod.sessionBypass() != "" || (*serveExecutionPolicy)(nil).sessionBypass() != "" {
		t.Fatalf("production: err=%v bypass=%q, want none", err, prod.sessionBypass())
	}
	var log bytes.Buffer
	p, err := resolveServeExecutionPolicy("bringup", "m", "m", &log)
	if err != nil {
		t.Fatal(err)
	}
	if !p.requireDeviceExecution(true) || p.sessionBypass() != "" {
		t.Fatalf("bringup with nothing lifted: bypass=%q, want device-only kept", p.sessionBypass())
	}
	if err := p.admit(validateServeHaloCPUOffload(true, true, "", "")); err != nil {
		t.Fatal(err)
	}
	if p.requireDeviceExecution(true) {
		t.Fatal("bringup must lift the serve-wide device-only load policy")
	}
	if got := p.sessionBypass(); got != executionPolicyBypassDeviceOnlySession {
		t.Fatalf("session bypass=%q, want %s handed to the planner", got, executionPolicyBypassDeviceOnlySession)
	}
	if !strings.Contains(log.String(), agent.HostPathDegradedToken) {
		t.Fatalf("bypass log does not name the perf-line token:\n%s", log.String())
	}
}
