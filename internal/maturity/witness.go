package maturity

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const runtimeProofSchema = "fak-maturity-runtime-proofs/2"

// RuntimeProof is a reproducible runtime invocation that proves fak executes a capability.
type RuntimeProof struct {
	Lane           string `json:"lane"`
	Command        string `json:"command"`
	OutputContains string `json:"output_contains"`
	DefaultOn      bool   `json:"default_on,omitempty"`
	DefaultReason  string `json:"default_reason,omitempty"`
}

type runtimeProofFile struct {
	Schema    string         `json:"schema"`
	Witnesses []RuntimeProof `json:"witnesses"`
}

func loadRuntimeProofs(root string) (map[string]RuntimeProof, error) {
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
		witness.OutputContains = strings.TrimSpace(witness.OutputContains)
		if witness.Lane == "" || witness.Command == "" || witness.OutputContains == "" {
			return nil, fmt.Errorf("runtime witness requires lane, command, and output_contains")
		}
		if !strings.Contains(strings.ToLower(witness.OutputContains), strings.ToLower(witness.Lane)) {
			return nil, fmt.Errorf("runtime proof for %s: output_contains must identify the lane", witness.Lane)
		}
		witness.DefaultReason = strings.TrimSpace(witness.DefaultReason)
		if err := validateRuntimeCommand(witness.Command); err != nil {
			return nil, fmt.Errorf("runtime proof for %s: %w", witness.Lane, err)
		}
		if witness.DefaultOn && witness.DefaultReason == "" {
			return nil, fmt.Errorf("runtime proof for %s: default_on requires default_reason", witness.Lane)
		}
		if !witness.DefaultOn && witness.DefaultReason != "" {
			return nil, fmt.Errorf("runtime proof for %s: default_reason requires default_on", witness.Lane)
		}
		if _, exists := witnesses[witness.Lane]; exists {
			return nil, fmt.Errorf("duplicate runtime witness for %s", witness.Lane)
		}
		witnesses[witness.Lane] = witness
	}
	return witnesses, nil
}

func validateRuntimeCommand(command string) error {
	fields := strings.Fields(command)
	if len(fields) == 0 {
		return fmt.Errorf("command is empty")
	}
	program := strings.ToLower(filepath.Base(fields[0]))
	if program == "go" || program == "go.exe" {
		return fmt.Errorf("go commands are build/test evidence, not dogfooding")
	}
	for _, field := range fields[1:] {
		if field == "test" || strings.HasPrefix(field, "-test.") {
			return fmt.Errorf("test commands are test evidence, not dogfooding")
		}
	}
	return nil
}

func verifyRuntimeProofs(root string) (map[string]RuntimeProof, error) {
	return verifyRuntimeProofsWithProgress(root, newRuntimeProofProgress(""))
}

func verifyRuntimeProofsWithProgress(root string, progress func(string, int)) (map[string]RuntimeProof, error) {
	progress("load_proofs", 0)
	witnesses, err := loadRuntimeProofs(root)
	if err != nil {
		return nil, err
	}
	if err := runRuntimeProofs(root, witnesses, progress); err != nil {
		return nil, err
	}
	return witnesses, nil
}

// VerifyRuntimeProofs runs every declared witness in stable lane order.
func VerifyRuntimeProofs(root string) error {
	witnesses, err := loadRuntimeProofs(root)
	if err != nil {
		return err
	}
	return runRuntimeProofs(root, witnesses, newRuntimeProofProgress(""))
}

func runRuntimeProofs(root string, witnesses map[string]RuntimeProof, progress func(string, int)) error {
	progress("runtime_proofs_start", 0)
	lanes := make([]string, 0, len(witnesses))
	for lane := range witnesses {
		lanes = append(lanes, lane)
	}
	sort.Strings(lanes)
	for index, lane := range lanes {
		witness := witnesses[lane]
		fields := strings.Fields(witness.Command)
		if len(fields) == 0 {
			return fmt.Errorf("runtime witness for %s is empty", lane)
		}
		program := fields[0]
		if isFakProgram(program) {
			progress("verify_artifact", index+1)
			resolved, err := resolveRuntimeFak()
			if err != nil {
				return fmt.Errorf("runtime witness for %s cannot resolve fak artifact: %w", lane, err)
			}
			if err := verifyFakArtifact(root, resolved); err != nil {
				return fmt.Errorf("runtime witness for %s: %w", lane, err)
			}
			program = resolved
		}
		command := exec.Command(program, fields[1:]...)
		command.Dir = root
		progress("execute_proof", index+1)
		output, err := command.CombinedOutput()
		if err != nil {
			return fmt.Errorf("runtime witness for %s failed: %w: %s", lane, err, strings.TrimSpace(string(output)))
		}
		if !strings.Contains(string(output), witness.OutputContains) {
			return fmt.Errorf("runtime witness for %s did not emit required output %q: %s", lane, witness.OutputContains, strings.TrimSpace(string(output)))
		}
	}
	progress("runtime_proofs_complete", len(lanes))
	return nil
}

// newRuntimeProofProgress retains one atomic, bounded snapshot even when the
// outer garden runner kills this process tree. The proof ID is its one-based
// position in sorted lane order at the scored source, never an argument or path.
// Elapsed time is since Build began, not the enclosing process launch.
func newRuntimeProofProgress(path string) func(string, int) {
	started := time.Now()
	lastProof := 0
	return func(phase string, proof int) {
		if path == "" {
			return
		}
		if proof > 0 {
			lastProof = proof
		}
		data, _ := json.Marshal(struct {
			Schema     string `json:"schema"`
			Phase      string `json:"phase"`
			ElapsedMS  int64  `json:"elapsed_ms"`
			ProofIndex int    `json:"proof_index"`
		}{"fak-scorecard-progress/1", phase, time.Since(started).Milliseconds(), lastProof})
		// Diagnostic I/O is best-effort and cannot change proof success/failure.
		if temporary, err := os.CreateTemp(filepath.Dir(path), ".maturity-progress-*.tmp"); err == nil {
			_, writeErr := temporary.Write(append(data, '\n'))
			closeErr := temporary.Close()
			if writeErr == nil && closeErr == nil {
				_ = os.Rename(temporary.Name(), path)
			}
			_ = os.Remove(temporary.Name())
		}
		_, _ = fmt.Fprintln(os.Stderr, "FAK_SCORECARD_PROGRESS "+string(data))
	}
}

var resolveRuntimeFak = func() (string, error) {
	if executable, err := os.Executable(); err == nil && isFakProgram(executable) {
		return executable, nil
	}
	return exec.LookPath("fak")
}

func verifyFakArtifact(root, artifact string) error {
	headCommand := windowgate.Command("git", "rev-parse", "HEAD")
	headCommand.Dir = root
	headOutput, err := headCommand.CombinedOutput()
	if err != nil {
		return fmt.Errorf("resolve scored source revision: %w: %s", err, strings.TrimSpace(string(headOutput)))
	}
	head := strings.TrimSpace(string(headOutput))
	versionCommand := windowgate.Command(artifact, "version")
	versionCommand.Dir = root
	versionOutput, err := versionCommand.CombinedOutput()
	if err != nil {
		return fmt.Errorf("inspect fak artifact identity: %w: %s", err, strings.TrimSpace(string(versionOutput)))
	}
	build := ""
	for _, line := range strings.Split(string(versionOutput), "\n") {
		if value, ok := strings.CutPrefix(strings.TrimSpace(line), "build:"); ok {
			build = strings.TrimSpace(value)
			break
		}
	}
	if build == "" || strings.Contains(build, "no VCS stamp") {
		return fmt.Errorf("fak artifact has no VCS revision for scored source %s", shortRevision(head))
	}
	fields := strings.Fields(build)
	if len(fields) == 0 || !strings.HasPrefix(head, fields[0]) {
		return fmt.Errorf("fak artifact revision %s does not match scored source %s", build, shortRevision(head))
	}
	if len(fields) > 1 && fields[1] == "dirty" {
		return fmt.Errorf("fak artifact revision %s is dirty; runtime evidence must name an immutable source", build)
	}
	return nil
}

func isFakProgram(program string) bool {
	base := strings.ToLower(filepath.Base(program))
	return base == "fak" || base == "fak.exe"
}

func shortRevision(revision string) string {
	if len(revision) > 12 {
		return revision[:12]
	}
	return revision
}
