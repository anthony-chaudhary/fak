package frontierswe

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestEnvAdapterPlanNoInternetLoopback(t *testing.T) {
	task, err := LoadTask(filepath.Join(fixtureDir, "git-to-zig"))
	if err != nil {
		t.Fatalf("LoadTask: %v", err)
	}

	plan := BuildEnvAdapterPlan(EnvAdapterConfig{Task: task})
	if plan.Schema != EnvAdapterSchema {
		t.Fatalf("schema = %q, want %q", plan.Schema, EnvAdapterSchema)
	}
	if !plan.Integrity.OK {
		t.Fatalf("loopback gateway/upstream should satisfy allow_internet=false: %+v", plan.Integrity)
	}
	if plan.Capability.Runnable != (plan.Capability.DockerPresent && plan.Integrity.OK) {
		t.Fatalf("capability inconsistent: %+v integrity=%+v", plan.Capability, plan.Integrity)
	}
	if plan.HealthzURL != "http://127.0.0.1:8080/healthz" {
		t.Errorf("HealthzURL = %q", plan.HealthzURL)
	}
	if plan.AgentTimeoutS != 72000 || plan.Resources.CPUs != 4 || plan.Resources.MemoryMB != 16384 {
		t.Errorf("task envelope not carried into plan: %+v timeout=%d", plan.Resources, plan.AgentTimeoutS)
	}
	for _, want := range []string{
		"--network none",
		"ghcr.io/proximal-labs/frontier-swe/git-to-zig:v4",
		"FAK_GATEWAY_URL=http://127.0.0.1:8080/v1",
		"curl -fsS http://127.0.0.1:8080/healthz",
		"/chat/completions",
		"FRONTIERSWE_RUN_CMD=python -m harbor run job.yaml",
	} {
		if !strings.Contains(plan.Command, want) {
			t.Errorf("docker command missing %q:\n%s", want, plan.Command)
		}
	}
	if !strings.Contains(plan.JobYAML, "harbor_ext.fak_routed:FakRoutedAgent") ||
		!strings.Contains(plan.JobYAML, "allow_internet: false") {
		t.Errorf("job yaml missing shim/no-internet contract:\n%s", plan.JobYAML)
	}
}

func TestEnvAdapterPlanRejectsExternalGatewayForNoInternet(t *testing.T) {
	task, err := LoadTask(filepath.Join(fixtureDir, "git-to-zig"))
	if err != nil {
		t.Fatalf("LoadTask: %v", err)
	}

	plan := BuildEnvAdapterPlan(EnvAdapterConfig{
		Task:            task,
		GatewayBaseURL:  "https://gateway.example.com/v1",
		UpstreamBaseURL: "http://127.0.0.1:11434/v1",
	})
	if plan.Integrity.OK {
		t.Fatalf("external gateway should fail allow_internet=false integrity: %+v", plan.Integrity)
	}
	if !strings.Contains(plan.Integrity.Reason, "gateway_base_url") {
		t.Errorf("reason should name gateway_base_url, got %q", plan.Integrity.Reason)
	}
	if plan.Capability.Runnable {
		t.Errorf("capability should not be runnable when integrity fails: %+v", plan.Capability)
	}

	pinned := BuildEnvAdapterPlan(EnvAdapterConfig{
		Task:            task,
		GatewayBaseURL:  "http://fak-gateway.local:8080/v1",
		UpstreamBaseURL: "http://model.local:11434/v1",
		PinnedHosts:     []string{"fak-gateway.local,model.local"},
	})
	if !pinned.Integrity.OK {
		t.Fatalf("pinned hosts should satisfy no-internet boundary: %+v", pinned.Integrity)
	}
}

func TestHealthzURL(t *testing.T) {
	tests := map[string]string{
		"http://127.0.0.1:8080/v1":       "http://127.0.0.1:8080/healthz",
		"http://127.0.0.1:8080/proxy/v1": "http://127.0.0.1:8080/proxy/healthz",
	}
	for in, want := range tests {
		if got := HealthzURL(in); got != want {
			t.Errorf("HealthzURL(%q) = %q, want %q", in, got, want)
		}
	}
}
