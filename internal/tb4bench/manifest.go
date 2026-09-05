package tb4bench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
)

// Supported task categories in Terminal-Bench 4.
const (
	CategorySysadmin = "sysadmin"
	CategoryBuild    = "build"
	CategoryGit      = "git"
	CategoryTest     = "test"
	CategoryRefactor = "refactor"
	CategoryDebug    = "debug"
	CategorySecurity = "security"
)

var validCategories = map[string]bool{
	CategorySysadmin: true,
	CategoryBuild:    true,
	CategoryGit:      true,
	CategoryTest:     true,
	CategoryRefactor: true,
	CategoryDebug:    true,
	CategorySecurity: true,
}

// TaskManifest represents a single task definition in the TB4 benchmark dataset.
type TaskManifest struct {
	TaskID                 string            `json:"task_id"`
	Category               string            `json:"category"`
	Prompt                 string            `json:"prompt"`
	EnvironmentImageDigest string            `json:"environment_image_digest"`
	SetupCommand           string            `json:"setup_command,omitempty"`
	VerificationOracle     string            `json:"verification_oracle"`
	VerificationOracleHash string            `json:"verification_oracle_hash"`
	TimeoutSeconds         int               `json:"timeout_seconds"`
	BudgetTurns            int               `json:"budget_turns"`
	Metadata               map[string]string `json:"metadata,omitempty"`
}

// Validate checks that the task manifest is well-formed, contains immutable image digests,
// and enforces valid task categories.
func (t *TaskManifest) Validate() error {
	if strings.TrimSpace(t.TaskID) == "" {
		return errors.New("task_id cannot be empty")
	}
	if !validCategories[t.Category] {
		return fmt.Errorf("invalid category %q for task %s; must be one of: sysadmin, build, git, test, refactor, debug, security", t.Category, t.TaskID)
	}
	if strings.TrimSpace(t.Prompt) == "" {
		return fmt.Errorf("prompt cannot be empty for task %s", t.TaskID)
	}
	if !strings.Contains(t.EnvironmentImageDigest, "sha256:") {
		return fmt.Errorf("environment_image_digest %q for task %s must contain immutable sha256 digest", t.EnvironmentImageDigest, t.TaskID)
	}
	if !strings.HasPrefix(t.VerificationOracleHash, "sha256:") {
		return fmt.Errorf("verification_oracle_hash %q for task %s must start with 'sha256:'", t.VerificationOracleHash, t.TaskID)
	}
	if t.TimeoutSeconds <= 0 {
		return fmt.Errorf("timeout_seconds must be positive for task %s, got %d", t.TaskID, t.TimeoutSeconds)
	}
	if t.BudgetTurns <= 0 {
		return fmt.Errorf("budget_turns must be positive for task %s, got %d", t.TaskID, t.BudgetTurns)
	}
	return nil
}

// VerifyOracleScript validates that the actual script bytes match the pinned hash.
func (t *TaskManifest) VerifyOracleScript(scriptBytes []byte) error {
	h := sha256.Sum256(scriptBytes)
	gotHash := "sha256:" + hex.EncodeToString(h[:])
	if gotHash != t.VerificationOracleHash {
		return fmt.Errorf("oracle script hash mismatch for task %s: got %s, want %s", t.TaskID, gotHash, t.VerificationOracleHash)
	}
	return nil
}

// ManifestSuite encapsulates a collection of tasks forming a TB4 benchmark suite.
type ManifestSuite struct {
	Benchmark string         `json:"benchmark"`
	Version   string         `json:"version"`
	Tasks     []TaskManifest `json:"tasks"`
}

// Validate checks the entire manifest suite.
func (s *ManifestSuite) Validate() error {
	if s.Benchmark != BenchmarkName {
		return fmt.Errorf("invalid benchmark name %q, expected %q", s.Benchmark, BenchmarkName)
	}
	if len(s.Tasks) == 0 {
		return errors.New("manifest suite contains no tasks")
	}
	seen := make(map[string]bool, len(s.Tasks))
	for i := range s.Tasks {
		task := &s.Tasks[i]
		if seen[task.TaskID] {
			return fmt.Errorf("duplicate task_id %q found in manifest suite", task.TaskID)
		}
		seen[task.TaskID] = true
		if err := task.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// ParseManifestJSON parses a JSON manifest into a ManifestSuite.
func ParseManifestJSON(data []byte) (*ManifestSuite, error) {
	var suite ManifestSuite
	if err := json.Unmarshal(data, &suite); err != nil {
		return nil, fmt.Errorf("failed to parse manifest JSON: %w", err)
	}
	if err := suite.Validate(); err != nil {
		return nil, fmt.Errorf("manifest validation failed: %w", err)
	}
	return &suite, nil
}

// LoadManifestFile loads and parses a manifest file from disk.
func LoadManifestFile(path string) (*ManifestSuite, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read manifest file: %w", err)
	}
	return ParseManifestJSON(data)
}
