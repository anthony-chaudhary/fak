package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

func TestFrontiersweEnvAdapterEmitsGatedRecipe(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"env-adapter", "--json", "--tasks", repoRootTasksDir, "--task", "git-to-zig",
	})
	if code != 0 {
		t.Fatalf("env-adapter exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}

	var plan frontierswe.EnvAdapterPlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("stdout is not env-adapter JSON: %v\nstdout:\n%s", err, stdout.String())
	}
	if plan.Schema != frontierswe.EnvAdapterSchema || plan.Task != "git-to-zig" {
		t.Fatalf("unexpected plan identity: %+v", plan)
	}
	if !plan.Integrity.OK {
		t.Fatalf("default loopback adapter should satisfy no-internet gate: %+v", plan.Integrity)
	}
	for _, want := range []string{
		"docker run",
		"--network none",
		"curl -fsS http://127.0.0.1:8080/healthz",
		"/chat/completions",
		"FRONTIERSWE_RUN_CMD=python -m harbor run job.yaml",
	} {
		if !strings.Contains(plan.Command, want) {
			t.Errorf("command missing %q:\n%s", want, plan.Command)
		}
	}
	if !strings.Contains(plan.JobYAML, "harbor_ext.fak_routed:FakRoutedAgent") {
		t.Errorf("job yaml missing C6 shim block:\n%s", plan.JobYAML)
	}
	// --json suppresses the human summary.
	if strings.Contains(stderr.String(), "== fak frontierswe env-adapter ==") {
		t.Errorf("--json should suppress human summary; stderr:\n%s", stderr.String())
	}
}

func TestFrontiersweEnvAdapterRefusesExternalNoInternetGateway(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"env-adapter", "--json", "--tasks", repoRootTasksDir, "--task", "git-to-zig",
		"--gateway-base-url", "https://gateway.example.com/v1",
	})
	if code != 1 {
		t.Fatalf("external gateway exit = %d, want 1\nstderr:\n%s", code, stderr.String())
	}
	var plan frontierswe.EnvAdapterPlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("stdout should still carry the refused plan JSON: %v\n%s", err, stdout.String())
	}
	if plan.Integrity.OK || !strings.Contains(plan.Integrity.Reason, "allow_internet=false") {
		t.Fatalf("plan should refuse external gateway under allow_internet=false: %+v", plan.Integrity)
	}
}
