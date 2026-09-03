package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/superstream"
)

func TestSuperstreamCLIPlan(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runSuperstream(&stdout, &stderr, []string{"plan"})
	if code != 0 {
		t.Fatalf("runSuperstream plan exit = %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Super Workstream: stream-sample") {
		t.Fatalf("expected stream-sample header, got: %s", out)
	}
	if !strings.Contains(out, "task-gateway-parity") {
		t.Fatalf("expected queue item in output, got: %s", out)
	}
}

func TestSuperstreamCLIStepJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runSuperstream(&stdout, &stderr, []string{"step", "--json"})
	if code != 0 {
		t.Fatalf("runSuperstream step exit = %d, stderr: %s", code, stderr.String())
	}
	var dec superstream.StepDecision
	if err := json.Unmarshal(stdout.Bytes(), &dec); err != nil {
		t.Fatalf("failed to parse JSON decision: %v, raw: %s", err, stdout.String())
	}
	if dec.Action != superstream.ActionAcquireLease {
		t.Fatalf("dec.Action = %s, want %s", dec.Action, superstream.ActionAcquireLease)
	}
	if dec.Item == nil || dec.Item.ID != "task-gateway-parity" {
		t.Fatalf("dec.Item = %v, want task-gateway-parity", dec.Item)
	}
}

func TestSuperstreamCLICarryover(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runSuperstream(&stdout, &stderr, []string{"carryover", "--json"})
	if code != 0 {
		t.Fatalf("runSuperstream carryover exit = %d, stderr: %s", code, stderr.String())
	}
	var seed superstream.StreamCarryoverSeed
	if err := json.Unmarshal(stdout.Bytes(), &seed); err != nil {
		t.Fatalf("failed to parse JSON carryover: %v, raw: %s", err, stdout.String())
	}
	if seed.Schema != superstream.CarryoverSchema {
		t.Fatalf("seed.Schema = %s, want %s", seed.Schema, superstream.CarryoverSchema)
	}
	if seed.StreamID != "stream-sample" {
		t.Fatalf("seed.StreamID = %s, want stream-sample", seed.StreamID)
	}
}

func TestSuperstreamCLIStatus(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runSuperstream(&stdout, &stderr, []string{"status"})
	if code != 0 {
		t.Fatalf("runSuperstream status exit = %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Super Workstream Status: stream-sample") {
		t.Fatalf("expected status header, got: %s", out)
	}
	if !strings.Contains(out, "Context Safety: SAFE") {
		t.Fatalf("expected Context Safety SAFE, got: %s", out)
	}
}
