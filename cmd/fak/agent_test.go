package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestAgentRawMode(t *testing.T) {
	// 1. Verify flag parsing and defaults
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse empty args: %v", err)
	}
	if af.raw == nil || *af.raw {
		t.Fatal("--raw defaulted to true")
	}
	if af.mode == nil || *af.mode != "" {
		t.Fatalf("--mode defaulted to %q", *af.mode)
	}

	fsRaw, afRaw := newAgentFlagSet()
	if err := fsRaw.Parse([]string{"--raw"}); err != nil {
		t.Fatalf("Parse --raw: %v", err)
	}
	if afRaw.raw == nil || !*afRaw.raw {
		t.Fatal("--raw was not parsed")
	}

	fsMode, afMode := newAgentFlagSet()
	if err := fsMode.Parse([]string{"--mode", "raw"}); err != nil {
		t.Fatalf("Parse --mode raw: %v", err)
	}
	if afMode.mode == nil || *afMode.mode != "raw" {
		t.Fatalf("--mode was not parsed: %q", *afMode.mode)
	}

	// 2. Verify mode validation
	if isRaw, isNat, err := validateAgentMode(true, false, ""); err != nil || !isRaw || isNat {
		t.Fatalf("validateAgentMode(raw=true) = (%v, %v, %v)", isRaw, isNat, err)
	}
	if isRaw, isNat, err := validateAgentMode(false, false, "raw"); err != nil || !isRaw || isNat {
		t.Fatalf("validateAgentMode(mode=raw) = (%v, %v, %v)", isRaw, isNat, err)
	}
	if _, _, err := validateAgentMode(true, true, ""); err == nil {
		t.Fatal("validateAgentMode with both raw and native did not fail")
	}
	if _, _, err := validateAgentMode(false, false, "unknown"); err == nil {
		t.Fatal("validateAgentMode with unknown mode did not fail")
	}

	// 3. Verify receipt structure and constructor
	metrics := agent.ArmMetrics{
		Arm:                 "baseline",
		Turns:               4,
		TaskCompleted:       true,
		DestructiveExecuted: true,
		FinalAnswer:         "test answer",
	}
	receipt := newRawAgentReceipt("do work", "test-model", metrics)
	if receipt.Schema != "agent.raw-receipt.v1" {
		t.Fatalf("receipt schema = %q, want agent.raw-receipt.v1", receipt.Schema)
	}
	if receipt.Schema != rawAgentReceiptSchema {
		t.Fatalf("receipt schema = %q, want %q", receipt.Schema, rawAgentReceiptSchema)
	}
	if receipt.Mode != "raw" {
		t.Fatalf("receipt mode = %q, want raw", receipt.Mode)
	}
	if receipt.FakMediated != false {
		t.Fatalf("receipt FakMediated = %v, want false", receipt.FakMediated)
	}
	if receipt.Adjudications != 0 {
		t.Fatalf("receipt Adjudications = %d, want 0", receipt.Adjudications)
	}
	if receipt.Task != "do work" || receipt.Model != "test-model" {
		t.Fatalf("receipt task/model mismatch: %#v", receipt)
	}
	if receipt.Metrics.Arm != "baseline" || receipt.Metrics.FinalAnswer != "test answer" {
		t.Fatalf("receipt metrics mismatch: %#v", receipt.Metrics)
	}

	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("json.Marshal(receipt): %v", err)
	}
	if bytes.Contains(data, []byte(`"arm":"fak"`)) {
		t.Fatalf("raw receipt leaked fak arm: %s", data)
	}

	// 4. Verify CLI execution with --raw flag and offline mode writing to file
	out := filepath.Join(t.TempDir(), "raw.json")
	stdout, stderr := captureAgentStdio(t, func() {
		cmdAgent([]string{"--raw", "--offline", "--out", out})
	})
	if !strings.Contains(stdout, "Booked flight UA123") {
		t.Fatalf("stdout did not carry the final answer:\n%s", stdout)
	}
	if strings.Contains(stdout, "turn-use vs now") {
		t.Fatalf("raw mode rendered dual-arm A/B benchmark:\n%s", stdout)
	}
	if !strings.Contains(stderr, filepath.Dir(out)) {
		t.Fatalf("stderr did not announce receipt directory:\n%s", stderr)
	}

	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", out, err)
	}
	var fileReceipt rawAgentReceipt
	if err := json.Unmarshal(body, &fileReceipt); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if fileReceipt.Schema != rawAgentReceiptSchema {
		t.Fatalf("file receipt schema = %q, want %q", fileReceipt.Schema, rawAgentReceiptSchema)
	}
	if fileReceipt.Schema != "agent.raw-receipt.v1" {
		t.Fatalf("file receipt schema = %q, want agent.raw-receipt.v1", fileReceipt.Schema)
	}
	if fileReceipt.Mode != "raw" {
		t.Fatalf("file receipt mode = %q, want raw", fileReceipt.Mode)
	}
	if fileReceipt.FakMediated != false {
		t.Fatalf("file receipt FakMediated = %v, want false", fileReceipt.FakMediated)
	}
	if fileReceipt.Adjudications != 0 {
		t.Fatalf("file receipt Adjudications = %d, want 0", fileReceipt.Adjudications)
	}
	if fileReceipt.Metrics.Arm != "baseline" {
		t.Fatalf("file receipt arm = %q, want baseline", fileReceipt.Metrics.Arm)
	}
	if !fileReceipt.Metrics.TaskCompleted {
		t.Fatalf("file receipt TaskCompleted = false, want true")
	}
	if !fileReceipt.Metrics.DestructiveExecuted {
		t.Fatalf("expected DestructiveExecuted = true in unmediated raw arm")
	}
	if fileReceipt.Metrics.Denies != 0 {
		t.Fatalf("expected 0 denies in unmediated raw arm, got %d", fileReceipt.Metrics.Denies)
	}

	// 5. Verify CLI execution with --mode raw
	outMode := filepath.Join(t.TempDir(), "raw_mode.json")
	cmdAgent([]string{"--mode", "raw", "--offline", "--out", outMode})
	bodyMode, err := os.ReadFile(outMode)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", outMode, err)
	}
	var modeReceipt rawAgentReceipt
	if err := json.Unmarshal(bodyMode, &modeReceipt); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if modeReceipt.Schema != "agent.raw-receipt.v1" || modeReceipt.FakMediated != false {
		t.Fatalf("mode receipt = %#v", modeReceipt)
	}

	// 6. Verify CLI execution with --out - (stdout receipt)
	stdoutDash, _ := captureAgentStdio(t, func() {
		cmdAgent([]string{"--raw", "--offline", "--out", "-"})
	})
	var dashReceipt rawAgentReceipt
	if err := json.Unmarshal([]byte(stdoutDash), &dashReceipt); err != nil {
		t.Fatalf("json.Unmarshal stdout dash receipt: %v\nstdout was:\n%s", err, stdoutDash)
	}
	if dashReceipt.Schema != "agent.raw-receipt.v1" || dashReceipt.FakMediated != false || dashReceipt.Mode != "raw" {
		t.Fatalf("dash receipt mismatch: %#v", dashReceipt)
	}
}
