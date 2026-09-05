package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestNativeAgentReceiptIsSingleKernelArm(t *testing.T) {
	metrics := agent.ArmMetrics{
		Arm:           "fak",
		Turns:         2,
		TaskCompleted: true,
		FinalAnswer:   "done",
	}
	receipt := newNativeAgentReceipt("fix it", "fixture", metrics)
	if receipt.Schema != nativeAgentReceiptSchema || receipt.Task != "fix it" || receipt.Model != "fixture" {
		t.Fatalf("receipt identity = %#v", receipt)
	}
	if receipt.Metrics.Arm != "fak" || receipt.Metrics.FinalAnswer != "done" {
		t.Fatalf("receipt metrics = %#v", receipt.Metrics)
	}
	body, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(body, []byte(`"baseline"`)) {
		t.Fatalf("single-arm receipt leaked benchmark arm: %s", body)
	}
}

func TestNativeAgentOfflinePrintsAnswerAndWritesReceipt(t *testing.T) {
	out := filepath.Join(t.TempDir(), "native.json")
	stdout, stderr := captureAgentStdio(t, func() {
		cmdAgent([]string{"--native", "--offline", "--out", out})
	})
	if !strings.Contains(stdout, "Booked flight UA123") {
		t.Fatalf("stdout did not carry the final answer:\n%s", stdout)
	}
	if strings.Contains(stdout, "turn-use vs now") {
		t.Fatalf("native mode rendered the A/B benchmark:\n%s", stdout)
	}
	if !strings.Contains(stderr, filepath.Dir(out)) {
		t.Fatalf("stderr did not announce receipt directory:\n%s", stderr)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var receipt nativeAgentReceipt
	if err := json.Unmarshal(body, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != nativeAgentReceiptSchema || receipt.Metrics.Arm != "fak" || !receipt.Metrics.TaskCompleted {
		t.Fatalf("receipt = %#v", receipt)
	}
	if receipt.Metrics.FinalAnswer == "" || receipt.Metrics.Turns != 7 {
		t.Fatalf("native run envelope = %#v", receipt.Metrics)
	}
}

func TestRawAgentReceiptIsSingleBaselineArm(t *testing.T) {
	metrics := agent.ArmMetrics{
		Arm:           "baseline",
		Turns:         2,
		TaskCompleted: true,
		FinalAnswer:   "done",
	}
	receipt := newRawAgentReceipt("fix it", "fixture", metrics)
	if receipt.Schema != rawAgentReceiptSchema || receipt.Task != "fix it" || receipt.Model != "fixture" {
		t.Fatalf("receipt identity = %#v", receipt)
	}
	if receipt.Metrics.Arm != "baseline" || receipt.Metrics.FinalAnswer != "done" {
		t.Fatalf("receipt metrics = %#v", receipt.Metrics)
	}
	body, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(body, []byte(`"fak"`)) {
		t.Fatalf("single-arm raw receipt leaked kernel arm: %s", body)
	}
}

func TestRawAgentOfflinePrintsAnswerAndWritesReceipt(t *testing.T) {
	out := filepath.Join(t.TempDir(), "raw.json")
	stdout, stderr := captureAgentStdio(t, func() {
		cmdAgent([]string{"--raw", "--offline", "--out", out})
	})
	if !strings.Contains(stdout, "Booked flight UA123") {
		t.Fatalf("stdout did not carry the final answer:\n%s", stdout)
	}
	if strings.Contains(stdout, "turn-use vs now") {
		t.Fatalf("raw mode rendered the A/B benchmark:\n%s", stdout)
	}
	if !strings.Contains(stderr, filepath.Dir(out)) {
		t.Fatalf("stderr did not announce receipt directory:\n%s", stderr)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var receipt rawAgentReceipt
	if err := json.Unmarshal(body, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != rawAgentReceiptSchema || receipt.Metrics.Arm != "baseline" || !receipt.Metrics.TaskCompleted {
		t.Fatalf("receipt = %#v", receipt)
	}
	// Verify zero kernel mediation occurred:
	if receipt.Metrics.Repairs != 0 || receipt.Metrics.VDSOHits != 0 || receipt.Metrics.Denies != 0 || receipt.Metrics.Quarantines != 0 {
		t.Fatalf("kernel mediation occurred in raw mode: %#v", receipt.Metrics)
	}
	// Verify unmediated baseline behavior: destructive operation executed, injection in context, unrepaired syntax error.
	if !receipt.Metrics.DestructiveExecuted {
		t.Fatalf("destructive op was expected to execute in raw baseline mode: %#v", receipt.Metrics)
	}
	if !receipt.Metrics.InjectionInContext {
		t.Fatalf("injection was expected to reach context in raw baseline mode: %#v", receipt.Metrics)
	}
	if receipt.Metrics.ToolErrors == 0 {
		t.Fatalf("tool errors were expected from unmediated args in raw baseline mode: %#v", receipt.Metrics)
	}
	if receipt.Metrics.FinalAnswer == "" || receipt.Metrics.Turns != 9 {
		t.Fatalf("raw run envelope = %#v", receipt.Metrics)
	}
}

func TestRawAgentRespectsTaskOfflineAndOutFlags(t *testing.T) {
	customTask := "Plan trip to New York"
	out := filepath.Join(t.TempDir(), "custom_raw.json")
	captureAgentStdio(t, func() {
		cmdAgent([]string{"--raw", "--task", customTask, "--offline", "--out", out})
	})

	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("failed to read output file: %v", err)
	}
	var receipt rawAgentReceipt
	if err := json.Unmarshal(body, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != rawAgentReceiptSchema {
		t.Fatalf("schema = %q, want %q", receipt.Schema, rawAgentReceiptSchema)
	}
	if receipt.Task != customTask {
		t.Fatalf("task = %q, want %q", receipt.Task, customTask)
	}
	if receipt.Metrics.Arm != "baseline" {
		t.Fatalf("metrics arm = %q, want baseline", receipt.Metrics.Arm)
	}
}

func TestRawAgentModeFlag(t *testing.T) {
	outRaw := filepath.Join(t.TempDir(), "mode_raw.json")
	captureAgentStdio(t, func() {
		cmdAgent([]string{"--mode", "raw", "--offline", "--out", outRaw})
	})
	bodyRaw, err := os.ReadFile(outRaw)
	if err != nil {
		t.Fatal(err)
	}
	var recRaw rawAgentReceipt
	if err := json.Unmarshal(bodyRaw, &recRaw); err != nil {
		t.Fatal(err)
	}
	if recRaw.Schema != rawAgentReceiptSchema || recRaw.Metrics.Arm != "baseline" {
		t.Fatalf("mode raw receipt = %#v", recRaw)
	}

	outNative := filepath.Join(t.TempDir(), "mode_native.json")
	captureAgentStdio(t, func() {
		cmdAgent([]string{"--mode", "native", "--offline", "--out", outNative})
	})
	bodyNative, err := os.ReadFile(outNative)
	if err != nil {
		t.Fatal(err)
	}
	var recNative nativeAgentReceipt
	if err := json.Unmarshal(bodyNative, &recNative); err != nil {
		t.Fatal(err)
	}
	if recNative.Schema != nativeAgentReceiptSchema || recNative.Metrics.Arm != "fak" {
		t.Fatalf("mode native receipt = %#v", recNative)
	}
}

func TestRawAgentZeroAdjudicationsWitness(t *testing.T) {
	// Tighten the adjudicator policy to deny the tools the task uses.
	// If adjudicator were called, these would be denied.
	prevPolicy := adjudicator.Default.PolicySnapshot()
	defer adjudicator.Default.SetPolicy(prevPolicy)
	adjudicator.Default.SetPolicy(adjudicator.Policy{
		Deny: map[string]abi.ReasonCode{
			"get_user":       abi.ReasonPolicyBlock,
			"search_flights": abi.ReasonPolicyBlock,
			"book_flight":    abi.ReasonPolicyBlock,
		},
	})

	out := filepath.Join(t.TempDir(), "unmediated.json")
	captureAgentStdio(t, func() {
		cmdAgent([]string{"--raw", "--offline", "--out", out})
	})

	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var receipt rawAgentReceipt
	if err := json.Unmarshal(body, &receipt); err != nil {
		t.Fatal(err)
	}
	// Zero adjudications occurred: the policy deny was never triggered because the kernel/adjudicator is bypassed.
	if receipt.Metrics.Denies != 0 {
		t.Fatalf("expected 0 denies in raw mode, got %d", receipt.Metrics.Denies)
	}
	if !receipt.Metrics.TaskCompleted {
		t.Fatalf("task was expected to complete unmediated despite policy block on adjudicator: %#v", receipt.Metrics)
	}
}

func TestResolveAgentMode(t *testing.T) {
	cases := []struct {
		raw        bool
		native     bool
		mode       string
		wantRaw    bool
		wantNative bool
		wantErr    bool
	}{
		{raw: true, native: false, mode: "", wantRaw: true, wantNative: false, wantErr: false},
		{raw: false, native: true, mode: "", wantRaw: false, wantNative: true, wantErr: false},
		{raw: false, native: false, mode: "raw", wantRaw: true, wantNative: false, wantErr: false},
		{raw: false, native: false, mode: "native", wantRaw: false, wantNative: true, wantErr: false},
		{raw: false, native: false, mode: "ab", wantRaw: false, wantNative: false, wantErr: false},
		{raw: false, native: false, mode: "", wantRaw: false, wantNative: false, wantErr: false},
		{raw: true, native: true, mode: "", wantErr: true},
		{raw: true, native: false, mode: "native", wantErr: true},
		{raw: false, native: true, mode: "raw", wantErr: true},
		{raw: false, native: false, mode: "invalid", wantErr: true},
	}
	for _, tc := range cases {
		gotRaw, gotNative, err := resolveAgentMode(tc.raw, tc.native, tc.mode)
		if tc.wantErr {
			if err == nil {
				t.Errorf("resolveAgentMode(%v, %v, %q) expected error, got nil", tc.raw, tc.native, tc.mode)
			}
			continue
		}
		if err != nil {
			t.Errorf("resolveAgentMode(%v, %v, %q) unexpected error: %v", tc.raw, tc.native, tc.mode, err)
			continue
		}
		if gotRaw != tc.wantRaw || gotNative != tc.wantNative {
			t.Errorf("resolveAgentMode(%v, %v, %q) = (%v, %v), want (%v, %v)", tc.raw, tc.native, tc.mode, gotRaw, gotNative, tc.wantRaw, tc.wantNative)
		}
	}
}
