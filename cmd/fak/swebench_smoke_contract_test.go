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
	if contract.Status != "INCOMPLETE_CONTRACT" {
		t.Fatalf("status = %q, want INCOMPLETE_CONTRACT while raw command is missing", contract.Status)
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
			if gate.OK {
				t.Fatalf("raw_arm_command gate must be false without --raw-command: %+v", gate)
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
