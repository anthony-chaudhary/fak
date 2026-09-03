package tb4bench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"time"
)

const (
	ContractSchema = "fak.tb4-run-contract.v1"
	BenchmarkName  = "terminal-bench-4"

	StatusContractDraft    = "CONTRACT_DRAFT"
	StatusRehearsalReady   = "REHEARSAL_READY"
	StatusLockedEvaluation = "LOCKED_EVALUATION"

	DefaultSeed              int64   = 42
	DefaultTemperature       float64 = 0.0
	DefaultTopP              float64 = 1.0
	DefaultTopK              int     = 1
	DefaultMaxContextTokens  int     = 32768
	DefaultMaxOutputTokens   int     = 4096
	DefaultMaxTurns          int     = 30
	DefaultTimeBudgetSeconds int     = 600
)

// ModelConfig captures the exact pinned weights and quantization parameters.
type ModelConfig struct {
	Checkpoint   string `json:"checkpoint"`
	Sha256       string `json:"sha256"`
	Quantization string `json:"quantization"`
	Format       string `json:"format,omitempty"`
}

// DeterminismEnvelope encapsulates all decoding hyperparameters and sandbox boundaries.
type DeterminismEnvelope struct {
	Temperature       float64 `json:"temperature"`
	Seed              int64   `json:"seed"`
	TopP              float64 `json:"top_p"`
	TopK              int     `json:"top_k"`
	MaxContextTokens  int     `json:"max_context_tokens"`
	MaxOutputTokens   int     `json:"max_output_tokens"`
	MaxTurns          int     `json:"max_turns"`
	NetworkIsolated   bool    `json:"network_isolated"`
	TimeBudgetSeconds int     `json:"time_budget_seconds"`
}

// ArmConfig specifies the configuration for one benchmark evaluation arm.
type ArmConfig struct {
	ID            string      `json:"id"`
	Harness       string      `json:"harness"`
	ServingEngine string      `json:"serving_engine"`
	IPC           string      `json:"ipc"`
	Model         ModelConfig `json:"model"`
}

// ParityConstraints defines strict parity rules between benchmark arms.
type ParityConstraints struct {
	SameTaskIDsRequired bool `json:"same_task_ids_required"`
	SameImageRequired   bool `json:"same_image_required"`
	SameBudgetRequired  bool `json:"same_budget_required"`
	SameWeightsRequired bool `json:"same_weights_required"`
}

// TaskSelection captures the dataset and task roster to evaluate.
type TaskSelection struct {
	Dataset string            `json:"dataset"`
	Version string            `json:"version"`
	Tasks   []string          `json:"tasks"`
	Parity  ParityConstraints `json:"parity"`
}

// OfficialRunContract is the authoritative, immutable specification of a TB4 run.
type OfficialRunContract struct {
	Schema        string              `json:"schema"`
	GeneratedAt   string              `json:"generated_at"`
	Benchmark     string              `json:"benchmark"`
	Status        string              `json:"status"`
	Model         ModelConfig         `json:"model"`
	Determinism   DeterminismEnvelope `json:"determinism"`
	ArmA          ArmConfig           `json:"arm_a"`
	ArmB          ArmConfig           `json:"arm_b"`
	TaskSelection TaskSelection       `json:"task_selection"`
}

// DefaultDeterminismEnvelope returns the strictly pinned determinism envelope for TB4.
func DefaultDeterminismEnvelope() DeterminismEnvelope {
	return DeterminismEnvelope{
		Temperature:       DefaultTemperature,
		Seed:              DefaultSeed,
		TopP:              DefaultTopP,
		TopK:              DefaultTopK,
		MaxContextTokens:  DefaultMaxContextTokens,
		MaxOutputTokens:   DefaultMaxOutputTokens,
		MaxTurns:          DefaultMaxTurns,
		NetworkIsolated:   true,
		TimeBudgetSeconds: DefaultTimeBudgetSeconds,
	}
}

// DefaultRunContract creates an official run contract template with default parameters.
func DefaultRunContract(checkpoint, modelSha256, quant string, taskIDs []string) *OfficialRunContract {
	modelCfg := ModelConfig{
		Checkpoint:   checkpoint,
		Sha256:       modelSha256,
		Quantization: quant,
		Format:       "gguf",
	}

	tasks := append([]string(nil), taskIDs...)
	sort.Strings(tasks)

	return &OfficialRunContract{
		Schema:      ContractSchema,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Benchmark:   BenchmarkName,
		Status:      StatusContractDraft,
		Model:       modelCfg,
		Determinism: DefaultDeterminismEnvelope(),
		ArmA: ArmConfig{
			ID:            "fak_inkernel",
			Harness:       "fak-native",
			ServingEngine: "in-kernel",
			IPC:           "in-process",
			Model:         modelCfg,
		},
		ArmB: ArmConfig{
			ID:            "opencode_llamacpp",
			Harness:       "opencode",
			ServingEngine: "llama-server",
			IPC:           "http/loopback",
			Model:         modelCfg,
		},
		TaskSelection: TaskSelection{
			Dataset: "terminal-bench-4",
			Version: "1.0",
			Tasks:   tasks,
			Parity: ParityConstraints{
				SameTaskIDsRequired: true,
				SameImageRequired:   true,
				SameBudgetRequired:  true,
				SameWeightsRequired: true,
			},
		},
	}
}

// Validate checks the contract against the TB4 specification.
// If strictDeterminism is true, sampling hyperparameters and isolation flags must match pinned defaults exactly.
func (c *OfficialRunContract) Validate(strictDeterminism bool) error {
	if c.Schema != ContractSchema {
		return fmt.Errorf("invalid contract schema %q, expected %q", c.Schema, ContractSchema)
	}
	if c.Benchmark != BenchmarkName {
		return fmt.Errorf("invalid benchmark name %q, expected %q", c.Benchmark, BenchmarkName)
	}
	if c.Model.Checkpoint == "" {
		return errors.New("model checkpoint must be specified")
	}
	if c.Model.Sha256 == "" {
		return errors.New("model sha256 hash must be specified")
	}

	if strictDeterminism {
		if c.Determinism.Temperature != DefaultTemperature {
			return fmt.Errorf("strict determinism requires temperature=0.0, got %f", c.Determinism.Temperature)
		}
		if c.Determinism.Seed != DefaultSeed {
			return fmt.Errorf("strict determinism requires seed=42, got %d", c.Determinism.Seed)
		}
		if c.Determinism.TopP != DefaultTopP {
			return fmt.Errorf("strict determinism requires top_p=1.0, got %f", c.Determinism.TopP)
		}
		if c.Determinism.TopK != DefaultTopK {
			return fmt.Errorf("strict determinism requires top_k=1, got %d", c.Determinism.TopK)
		}
		if c.Determinism.MaxContextTokens <= 0 {
			return errors.New("max_context_tokens must be positive")
		}
		if !c.Determinism.NetworkIsolated {
			return errors.New("strict determinism requires network_isolated=true")
		}
	}

	if len(c.TaskSelection.Tasks) == 0 {
		return errors.New("task selection must contain at least one task")
	}

	if c.TaskSelection.Parity.SameWeightsRequired {
		if c.ArmA.Model.Sha256 != c.Model.Sha256 {
			return fmt.Errorf("arm_a model sha256 %q does not match contract model sha256 %q", c.ArmA.Model.Sha256, c.Model.Sha256)
		}
		if c.ArmB.Model.Sha256 != c.Model.Sha256 {
			return fmt.Errorf("arm_b model sha256 %q does not match contract model sha256 %q", c.ArmB.Model.Sha256, c.Model.Sha256)
		}
	}

	return nil
}

// Digest returns the SHA-256 hex digest of the canonical JSON representation of the contract.
func (c *OfficialRunContract) Digest() (string, error) {
	data, err := json.Marshal(c)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:]), nil
}

// LoadContractFile loads and validates a contract JSON file.
func LoadContractFile(path string, strictDeterminism bool) (*OfficialRunContract, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read contract file: %w", err)
	}
	var contract OfficialRunContract
	if err := json.Unmarshal(data, &contract); err != nil {
		return nil, fmt.Errorf("failed to unmarshal contract JSON: %w", err)
	}
	if err := contract.Validate(strictDeterminism); err != nil {
		return nil, fmt.Errorf("contract validation failed: %w", err)
	}
	return &contract, nil
}
