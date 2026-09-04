package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRunAMDGPUDirect_Inspect(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunAMDGPUDirect(&stdout, &stderr, []string{"inspect"})
	if code != 0 {
		t.Fatalf("RunAMDGPUDirect inspect returned %d: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "AMD GPU Direct Topology") {
		t.Errorf("expected header 'AMD GPU Direct Topology', got:\n%s", out)
	}
	if !strings.Contains(out, "Fabric=") {
		t.Errorf("expected fabric link in output, got:\n%s", out)
	}

	// Test JSON mode
	stdout.Reset()
	stderr.Reset()
	code = RunAMDGPUDirect(&stdout, &stderr, []string{"inspect", "--json"})
	if code != 0 {
		t.Fatalf("RunAMDGPUDirect inspect --json returned %d: %s", code, stderr.String())
	}
	var res map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if _, ok := res["nodes"]; !ok {
		t.Errorf("missing 'nodes' in JSON output: %+v", res)
	}
}

func TestRunAMDGPUDirect_Audit(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunAMDGPUDirect(&stdout, &stderr, []string{"audit"})
	if code != 0 {
		t.Fatalf("RunAMDGPUDirect audit returned %d: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Healthy:") {
		t.Errorf("expected 'Healthy:' in audit output, got:\n%s", out)
	}

	// Test JSON mode
	stdout.Reset()
	stderr.Reset()
	code = RunAMDGPUDirect(&stdout, &stderr, []string{"audit", "--json"})
	if code != 0 {
		t.Fatalf("RunAMDGPUDirect audit --json returned %d: %s", code, stderr.String())
	}
	var res map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if res["healthy"] != true {
		t.Errorf("expected healthy=true, got %+v", res)
	}
}

func TestRunAMDGPUDirect_Bench(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunAMDGPUDirect(&stdout, &stderr, []string{"bench"})
	if code != 0 {
		t.Fatalf("RunAMDGPUDirect bench returned %d: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Microbenchmark Results") {
		t.Errorf("expected 'Microbenchmark Results' in bench output, got:\n%s", out)
	}
	if !strings.Contains(out, "Staging Copies: 0") {
		t.Errorf("expected zero staging copies in bench output, got:\n%s", out)
	}

	// Test JSON mode
	stdout.Reset()
	stderr.Reset()
	code = RunAMDGPUDirect(&stdout, &stderr, []string{"bench", "--json"})
	if code != 0 {
		t.Fatalf("RunAMDGPUDirect bench --json returned %d: %s", code, stderr.String())
	}
	var res map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}
	p2p, ok := res["p2p_transfer"].(map[string]any)
	if !ok || p2p["staging_copies"] != float64(0) {
		t.Errorf("expected p2p staging_copies=0, got %+v", res)
	}
	rdma, ok := res["rdma_registration"].(map[string]any)
	if !ok || rdma["staging_copies"] != float64(0) {
		t.Errorf("expected rdma staging_copies=0, got %+v", res)
	}
	nvme, ok := res["nvme_direct_storage"].(map[string]any)
	if !ok || nvme["staging_copies"] != float64(0) {
		t.Errorf("expected nvme staging_copies=0, got %+v", res)
	}
}

func TestRunAMDGPUDirect_InvalidMode(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunAMDGPUDirect(&stdout, &stderr, []string{"invalid_subcommand"})
	if code != 2 {
		t.Fatalf("expected exit code 2 for invalid subcommand, got %d", code)
	}
	if !strings.Contains(stderr.String(), "unknown mode") {
		t.Errorf("expected 'unknown mode' in stderr, got: %s", stderr.String())
	}
}
