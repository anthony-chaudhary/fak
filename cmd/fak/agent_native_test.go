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
