package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/roofline"
)

func TestRunBenchRoofline_JSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchRoofline(&stdout, &stderr, []string{"--device=gfx1151", "--json"})
	if code != 0 {
		t.Fatalf("runBenchRoofline failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var receipt roofline.EmpiricalRooflineReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v\nOutput was: %s", err, stdout.String())
	}

	if receipt.Schema != roofline.EmpiricalRooflineSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, roofline.EmpiricalRooflineSchema)
	}
	if receipt.Device != "gfx1151" {
		t.Errorf("device = %q, want gfx1151", receipt.Device)
	}
	if !receipt.Verified {
		t.Errorf("receipt not verified")
	}
	if err := receipt.Verify(); err != nil {
		t.Errorf("receipt verification failed: %v", err)
	}
}

func TestRunBenchRoofline_SubcommandPrefix(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchRoofline(&stdout, &stderr, []string{"roofline", "--device=gfx1151", "--json"})
	if code != 0 {
		t.Fatalf("runBenchRoofline with subcommand prefix failed with exit code %d; stderr: %s", code, stderr.String())
	}

	var receipt roofline.EmpiricalRooflineReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v", err)
	}
	if receipt.Device != "gfx1151" {
		t.Errorf("device = %q, want gfx1151", receipt.Device)
	}
}

func TestRunBenchRoofline_HumanReadable(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchRoofline(&stdout, &stderr, []string{"--device=gfx1151"})
	if code != 0 {
		t.Fatalf("runBenchRoofline failed with exit code %d; stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "AMD Strix Halo (gfx1151)") {
		t.Errorf("expected output to mention AMD Strix Halo (gfx1151), got:\n%s", out)
	}
	if !strings.Contains(out, "Coalesced DRAM Read Streaming") {
		t.Errorf("expected output to mention Coalesced DRAM Read Streaming, got:\n%s", out)
	}
	if !strings.Contains(out, "MALL Cache Boundary Sweep") {
		t.Errorf("expected output to mention MALL Cache Boundary Sweep, got:\n%s", out)
	}
	if !strings.Contains(out, "Synthetic WMMA Compute Ceilings") {
		t.Errorf("expected output to mention Synthetic WMMA Compute Ceilings, got:\n%s", out)
	}
}

func TestRunBenchRoofline_OutAndVerify(t *testing.T) {
	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "roofline_receipt.json")

	var stdout, stderr bytes.Buffer
	code := runBenchRoofline(&stdout, &stderr, []string{"--device=gfx1151", "--json", "--out=" + outPath})
	if code != 0 {
		t.Fatalf("runBenchRoofline with --out failed with code %d: %s", code, stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	verifyCode := runBenchRoofline(&stdout, &stderr, []string{"--verify=" + outPath})
	if verifyCode != 0 {
		t.Fatalf("runBenchRoofline --verify failed with code %d: %s", verifyCode, stderr.String())
	}
	if !strings.Contains(stdout.String(), "VALID") {
		t.Errorf("expected verification output to contain VALID, got: %s", stdout.String())
	}
}

func TestRunBenchRoofline_Help(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchRoofline(&stdout, &stderr, []string{"--help"})
	// flag.ContinueOnError prints help to output
	if code != 0 && code != 2 {
		t.Fatalf("unexpected code for --help: %d", code)
	}
}

func TestRunBenchRoofline_InvalidDevice(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchRoofline(&stdout, &stderr, []string{"--device=unsupported_device_arch"})
	if code == 0 {
		t.Fatalf("expected error exit code for invalid device, got 0")
	}
}
