package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

func TestRunAMDGPU_Help(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runAMDGPU(&stdout, &stderr, []string{"help"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d", code)
	}
	if !strings.Contains(stdout.String(), "Usage: fak amdgpu") {
		t.Errorf("expected usage message, got:\n%s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "roofline") {
		t.Errorf("expected roofline subcommand mentioned, got:\n%s", stdout.String())
	}
}

func TestRunAMDGPU_RooflineJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runAMDGPU(&stdout, &stderr, []string{"roofline", "--device=gfx1151", "--json", "--mock"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %s", code, stderr.String())
	}

	var receipt amdgpu.EmpiricalRooflineReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON receipt: %v\nOutput: %s", err, stdout.String())
	}

	if receipt.Schema != amdgpu.EmpiricalRooflineSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, amdgpu.EmpiricalRooflineSchema)
	}
	if receipt.Device != "gfx1151" {
		t.Errorf("device = %q, want 'gfx1151'", receipt.Device)
	}
	if err := receipt.Verify(); err != nil {
		t.Errorf("receipt.Verify() failed: %v", err)
	}
}

func TestRunAMDGPU_DirectFlags(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runAMDGPU(&stdout, &stderr, []string{"--device=gfx1151", "--json", "--mock"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %s", code, stderr.String())
	}

	var receipt amdgpu.EmpiricalRooflineReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}
	if receipt.Device != "gfx1151" {
		t.Errorf("device = %q, want 'gfx1151'", receipt.Device)
	}
}

func TestRunAMDGPU_UnknownSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runAMDGPU(&stdout, &stderr, []string{"unknown-subcommand"})
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
	if !strings.Contains(stderr.String(), "unknown subcommand") {
		t.Errorf("expected stderr to mention unknown subcommand, got: %s", stderr.String())
	}
}

func TestRunAMDGPU_Containment(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runAMDGPU(&stdout, &stderr, []string{"containment", "--aperture-gb=64", "--json"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %s", code, stderr.String())
	}
	var res map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}
	if res["has_capacity"] != true {
		t.Errorf("expected has_capacity=true, got %v", res["has_capacity"])
	}
}
