package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestHILProbe(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHIL(&stdout, &stderr, []string{"--probe"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "fak hil: Hardware Probe") {
		t.Errorf("expected hardware probe header in stdout, got:\n%s", out)
	}
	if !strings.Contains(out, "Platform/Arch:") {
		t.Errorf("expected Platform/Arch in stdout, got:\n%s", out)
	}
}

func TestHILProbeJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHIL(&stdout, &stderr, []string{"--probe", "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	var data map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &data); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v, raw: %s", err, stdout.String())
	}
	if kind, ok := data["kind"].(string); !ok || kind == "" {
		t.Errorf("expected non-empty kind, got %v", data["kind"])
	}
}

func TestHILMicroDoses(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHIL(&stdout, &stderr, []string{})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "fak hil — Hardware-in-the-Loop Micro-Dose Suite") {
		t.Errorf("expected micro-dose header in stdout, got:\n%s", out)
	}
	if !strings.Contains(out, "[PASS]") {
		t.Errorf("expected [PASS] in output, got:\n%s", out)
	}
}

func TestHILMicroDosesJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHIL(&stdout, &stderr, []string{"--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	var rep map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v, raw: %s", err, stdout.String())
	}
	if schema, ok := rep["schema"].(string); !ok || schema != "fak.hil.report.v1" {
		t.Errorf("expected schema 'fak.hil.report.v1', got %v", rep["schema"])
	}
	if allPassed, ok := rep["all_passed"].(bool); !ok || !allPassed {
		t.Errorf("expected all_passed to be true, got %v", rep["all_passed"])
	}
}

func TestHILAuditComparisonRealHardware(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHIL(&stdout, &stderr, []string{
		"--audit-comparison",
		"--candidate", "fak-native",
		"--candidate-val", "15.22",
		"--candidate-physical=true",
		"--baseline", "llama.cpp",
		"--baseline-val", "12.14",
		"--baseline-physical=true",
	})
	if rc != 0 {
		t.Fatalf("expected rc 0 for real hardware comparison, got %d, stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "VERIFIED_REAL_HARDWARE") {
		t.Errorf("expected VERIFIED_REAL_HARDWARE in stdout, got:\n%s", out)
	}
}

func TestHILAuditComparisonRejectsSimulation(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHIL(&stdout, &stderr, []string{
		"--audit-comparison",
		"--candidate", "fak-simulated",
		"--candidate-val", "250.0",
		"--candidate-physical=false",
		"--candidate-type", "analytical_bound",
		"--baseline", "llama.cpp",
		"--baseline-val", "50.0",
		"--baseline-physical=true",
	})
	if rc != 1 {
		t.Fatalf("expected rc 1 (refused) for simulated comparison, got %d", rc)
	}
	out := stdout.String()
	if !strings.Contains(out, "EARLY_INDICATOR_ONLY") {
		t.Errorf("expected EARLY_INDICATOR_ONLY in stdout, got:\n%s", out)
	}
}

func TestHILInventoryJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHIL(&stdout, &stderr, []string{"--inventory", "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0 for inventory probe, got %d, stderr: %s", rc, stderr.String())
	}
	var rep map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v, raw: %s", err, stdout.String())
	}
	if schema, ok := rep["schema"].(string); !ok || schema != "fak.hil.inventory.v1" {
		t.Errorf("expected schema 'fak.hil.inventory.v1', got %v", rep["schema"])
	}
	if _, ok := rep["local"].(map[string]any); !ok {
		t.Errorf("expected local hardware info object in inventory")
	}
	if _, ok := rep["lan"].(map[string]any); !ok {
		t.Errorf("expected lan node info object in inventory")
	}
}

func TestHILProbeWithLAN(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHIL(&stdout, &stderr, []string{"--probe", "--lan"})
	if rc != 0 {
		t.Fatalf("expected rc 0 for probe with LAN, got %d, stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "fak hil: Hardware Probe") {
		t.Errorf("expected hardware probe header, got:\n%s", out)
	}
	if !strings.Contains(out, "LAN Node") {
		t.Errorf("expected LAN Node in output, got:\n%s", out)
	}
}
