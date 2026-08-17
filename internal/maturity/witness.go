package maturity

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

const runtimeProofSchema = "fak-maturity-runtime-proofs/1"

// RuntimeProof is a reproducible runtime invocation that proves fak executes a capability.
type RuntimeProof struct {
	Lane    string `json:"lane"`
	Command string `json:"command"`
}

type runtimeProofFile struct {
	Schema    string         `json:"schema"`
	Witnesses []RuntimeProof `json:"witnesses"`
}

func loadRuntimeProofes(root string) (map[string]RuntimeProof, error) {
	path := filepath.Join(root, "internal", "maturity", "runtime-proofs.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read runtime witness registry: %w", err)
	}
	var file runtimeProofFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("decode runtime witness registry: %w", err)
	}
	if file.Schema != runtimeProofSchema {
		return nil, fmt.Errorf("runtime witness schema %q, want %q", file.Schema, runtimeProofSchema)
	}
	witnesses := make(map[string]RuntimeProof, len(file.Witnesses))
	for _, witness := range file.Witnesses {
		witness.Lane = strings.TrimSpace(witness.Lane)
		witness.Command = strings.TrimSpace(witness.Command)
		if witness.Lane == "" || witness.Command == "" {
			return nil, fmt.Errorf("runtime witness requires lane and command")
		}
		if _, exists := witnesses[witness.Lane]; exists {
			return nil, fmt.Errorf("duplicate runtime witness for %s", witness.Lane)
		}
		witnesses[witness.Lane] = witness
	}
	return witnesses, nil
}

func verifyRuntimeProofes(root string) (map[string]RuntimeProof, error) {
	witnesses, err := loadRuntimeProofes(root)
	if err != nil {
		return nil, err
	}
	if err := runRuntimeProofes(root, witnesses); err != nil {
		return nil, err
	}
	return witnesses, nil
}

// VerifyRuntimeProofes runs every declared witness in stable lane order.
func VerifyRuntimeProofes(root string) error {
	witnesses, err := loadRuntimeProofes(root)
	if err != nil {
		return err
	}
	return runRuntimeProofes(root, witnesses)
}

func runRuntimeProofes(root string, witnesses map[string]RuntimeProof) error {
	lanes := make([]string, 0, len(witnesses))
	for lane := range witnesses {
		lanes = append(lanes, lane)
	}
	sort.Strings(lanes)
	for _, lane := range lanes {
		witness := witnesses[lane]
		fields := strings.Fields(witness.Command)
		if len(fields) == 0 {
			return fmt.Errorf("runtime witness for %s is empty", lane)
		}
		command := exec.Command(fields[0], fields[1:]...)
		command.Dir = root
		if output, err := command.CombinedOutput(); err != nil {
			return fmt.Errorf("runtime witness for %s failed: %w: %s", lane, err, strings.TrimSpace(string(output)))
		}
	}
	return nil
}
