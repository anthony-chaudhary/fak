package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/swebench"
)

func TestSwebenchSmokeContractCommandWritesPreRunContract(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "contract.json")
	md := filepath.Join(tmp, "contract.md")
	difficulty := filepath.Join("..", "..", "testdata", "swebench_smoke.json")

	cmdSwebenchSmokeContract([]string{
		"--difficulty", difficulty,
		"--python", "definitely-not-a-python-binary",
		"--out", out,
		"--md", md,
	})

	var contract swebench.OpusSmokeContract
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &contract); err != nil {
		t.Fatal(err)
	}
	if contract.Schema != swebench.OpusSmokeContractSchema {
		t.Fatalf("schema = %q", contract.Schema)
	}
	if contract.Status != "READY_FOR_REMOTE_GRADING" {
		t.Fatalf("status = %q, want READY_FOR_REMOTE_GRADING while only local grading is gated", contract.Status)
	}
	if contract.ResultClaimAllowed {
		t.Fatal("smoke-contract must not allow a result claim")
	}
	if len(contract.TaskSelection.TaskIDs) != 5 || !contract.TaskSelection.SameTaskIDs {
		t.Fatalf("task selection = %+v", contract.TaskSelection)
	}
	var sawRawGate bool
	for _, gate := range contract.Gates {
		if gate.Name == "raw_arm_command" {
			sawRawGate = true
			if !gate.OK {
				t.Fatalf("raw_arm_command gate must be true with the default raw command: %+v", gate)
			}
			if !strings.Contains(gate.Detail, "mini-extra swebench") {
				t.Fatalf("raw command does not use mini-swe-agent scaffold: %s", gate.Detail)
			}
			if !strings.Contains(gate.Detail, "django__django-12345|django__django-23456") {
				t.Fatalf("raw command does not pin selected task ids: %s", gate.Detail)
			}
			if !strings.Contains(gate.Detail, "predictions.json") {
				t.Fatalf("raw command does not normalize the predictions path: %s", gate.Detail)
			}
		}
	}
	if !sawRawGate {
		t.Fatalf("raw_arm_command gate missing: %+v", contract.Gates)
	}

	mb, err := os.ReadFile(md)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(mb), "Required Before Any Result Claim") {
		t.Fatalf("markdown did not render the no-claim requirements:\n%s", mb)
	}
}

func TestSwebenchSmokeContractCommandPreservesExplicitRawCommand(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "contract.json")
	difficulty := filepath.Join("..", "..", "testdata", "swebench_smoke.json")
	raw := "raw-opus-runner --tasks smoke.txt"

	cmdSwebenchSmokeContract([]string{
		"--difficulty", difficulty,
		"--python", "definitely-not-a-python-binary",
		"--raw-command", raw,
		"--out", out,
	})

	var contract swebench.OpusSmokeContract
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &contract); err != nil {
		t.Fatal(err)
	}
	for _, arm := range contract.Arms {
		if arm.Name == "raw-opus" && arm.Command != raw {
			t.Fatalf("explicit raw command was not preserved: %q", arm.Command)
		}
	}
}

func TestSwebenchSmokeContractCommandHonorsDifficultyEnv(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "contract.json")
	difficulty := filepath.Join("..", "..", "testdata", "swebench_smoke.json")
	t.Setenv("FAK_SWEBENCH_DIFFICULTY", difficulty)
	t.Setenv("FAK_SWEBENCH_DATASET", "")

	cmdSwebenchSmokeContract([]string{
		"--python", "definitely-not-a-python-binary",
		"--out", out,
	})

	var contract swebench.OpusSmokeContract
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &contract); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(contract.TaskSelection.Source, difficulty) {
		t.Fatalf("source did not use env difficulty: %q", contract.TaskSelection.Source)
	}
	var fakCommand string
	for _, arm := range contract.Arms {
		if arm.Name == "fak-opus" {
			fakCommand = arm.Command
			break
		}
	}
	if !strings.Contains(fakCommand, "--difficulty "+difficulty) {
		t.Fatalf("fak command did not preserve env difficulty:\n%s", fakCommand)
	}
}

func TestSwebenchDeepSWEContractCommandWritesPreRunContract(t *testing.T) {
	tmp := t.TempDir()
	out := filepath.Join(tmp, "deepswe-contract.json")
	md := filepath.Join(tmp, "deepswe-contract.md")
	difficulty := filepath.Join("..", "..", "testdata", "swebench_smoke.json")

	cmdSwebenchDeepSWEContract([]string{
		"--difficulty", difficulty,
		"--python", "definitely-not-a-python-binary",
		"--out", out,
		"--md", md,
	})

	var contract swebench.DeepSWERawFakContract
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &contract); err != nil {
		t.Fatal(err)
	}
	if contract.Schema != swebench.DeepSWERawFakContractSchema {
		t.Fatalf("schema = %q", contract.Schema)
	}
	if contract.Status != "READY_FOR_EXTERNAL_RUN" {
		t.Fatalf("status = %q", contract.Status)
	}
	if contract.ResultClaimAllowed {
		t.Fatal("deepswe-contract must not allow a result claim")
	}
	if len(contract.TaskSelection.TaskIDs) != 2 || !contract.TaskSelection.SameTaskIDs {
		t.Fatalf("task selection = %+v", contract.TaskSelection)
	}
	if contract.Adapter.Command != "deepswe-r2e-runner" || !contract.Adapter.SameAdapter {
		t.Fatalf("adapter not fixed: %+v", contract.Adapter)
	}

	arms := map[string]swebench.SmokeArm{}
	for _, arm := range contract.Arms {
		arms[arm.Name] = arm
	}
	raw := arms["raw-deepswe"].Command
	fak := arms["fak-deepswe"].Command
	for name, cmd := range map[string]string{"raw": raw, "fak": fak} {
		for _, want := range []string{"go run ./cmd/fak swebench run", "--agent deepswe", "--limit 2", "--model DeepSWE-Preview"} {
			if !strings.Contains(cmd, want) {
				t.Fatalf("%s command missing %q:\n%s", name, want, cmd)
			}
		}
	}
	if !strings.Contains(arms["raw-deepswe"].PredictionsPath, "predictions.json") ||
		!strings.Contains(arms["fak-deepswe"].PredictionsPath, "predictions.json") {
		t.Fatalf("prediction paths missing: %+v", contract.Arms)
	}
	if !strings.Contains(raw, "$env:FAK_DEEPSWE_BASE_URL=$env:RAW_DEEPSWE_BASE_URL") {
		t.Fatalf("raw command did not preserve raw base URL env reference:\n%s", raw)
	}
	if !strings.Contains(fak, "$env:FAK_DEEPSWE_BASE_URL='http://localhost:8080/v1'") {
		t.Fatalf("fak command did not target the fak gateway base URL:\n%s", fak)
	}

	mb, err := os.ReadFile(md)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(mb), "DeepSWE Raw-vs-fak SWE-bench Contract") {
		t.Fatalf("markdown did not render the DeepSWE contract:\n%s", mb)
	}
}
