//go:build wip_coordination

package harnesskit

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// CoordinationContractVersion is the compatibility line implemented by the coordination plane.
const CoordinationContractVersion = "fak.harness.coordination/v1"

// AccessMode governs whether a worker operates purely in an observational/read-only mode or may cause side effects.
type AccessMode string

const (
	// AccessModeObserve specifies that the worker is strictly read-only and cannot mutate state or worktrees.
	AccessModeObserve AccessMode = "observe"
	// AccessModeEffect specifies that the worker may produce bounded mutations according to its tool scope.
	AccessModeEffect AccessMode = "effect"
)

// IsValid reports whether the access mode is recognized.
func (m AccessMode) IsValid() bool {
	return m == AccessModeObserve || m == AccessModeEffect
}

// AllowsMutations reports whether the access mode permits mutations.
func (m AccessMode) AllowsMutations() bool {
	return m == AccessModeEffect
}

// AllowsWorktree reports whether the access mode permits worktree modifications.
func (m AccessMode) AllowsWorktree() bool {
	return m == AccessModeEffect
}

// StrategyKind designates a scheduling and execution strategy used by a manager.
type StrategyKind string

const (
	// StrategyFanOutFanIn concurrently executes workers and aggregates their receipts.
	StrategyFanOutFanIn StrategyKind = "fan_out_fan_in"
	// StrategySequential executes worker roles in ordered sequential turns.
	StrategySequential StrategyKind = "sequential"
	// StrategyAdaptiveDAG schedules workers based on dependency graphs and runtime receipts.
	StrategyAdaptiveDAG StrategyKind = "adaptive_dag"
	// StrategySpeculative races multiple speculative worker implementations, keeping the first valid receipt.
	StrategySpeculative StrategyKind = "speculative_race"
)

// IsValid reports whether the strategy kind is recognized.
func (s StrategyKind) IsValid() bool {
	switch s {
	case StrategyFanOutFanIn, StrategySequential, StrategyAdaptiveDAG, StrategySpeculative:
		return true
	default:
		return false
	}
}

// ReceiptStatus specifies the terminal execution status of a worker task.
type ReceiptStatus string

const (
	// StatusCompleted indicates successful completion of the worker task.
	StatusCompleted ReceiptStatus = "COMPLETED"
	// StatusFailed indicates task failure during execution or witness verification.
	StatusFailed ReceiptStatus = "FAILED"
	// StatusAbstain indicates the worker gracefully abstained due to unmet preconditions or out-of-scope tasks.
	StatusAbstain ReceiptStatus = "ABSTAIN"
	// StatusTimedOut indicates the task exceeded its budget timeout.
	StatusTimedOut ReceiptStatus = "TIMED_OUT"
)

// IsValid reports whether the receipt status is recognized.
func (s ReceiptStatus) IsValid() bool {
	switch s {
	case StatusCompleted, StatusFailed, StatusAbstain, StatusTimedOut:
		return true
	default:
		return false
	}
}

// IsTerminal reports whether the status represents a finished task.
func (s ReceiptStatus) IsTerminal() bool {
	return s.IsValid()
}

// IsSuccess reports whether the status represents a successful outcome.
func (s ReceiptStatus) IsSuccess() bool {
	return s == StatusCompleted
}

// RiskEscalationAction defines the remediation action taken when an escalation gate triggers.
type RiskEscalationAction string

const (
	// EscalateAbstain halts execution and records an abstain receipt.
	EscalateAbstain RiskEscalationAction = "abstain"
	// EscalatePromptHuman halts automated execution and prompts human-in-the-loop review.
	EscalatePromptHuman RiskEscalationAction = "prompt_human"
	// EscalateRerouteRole redirects the task to a specialized role equipped for the detected risk.
	EscalateRerouteRole RiskEscalationAction = "reroute_role"
)

// IsValid reports whether the escalation action is recognized.
func (a RiskEscalationAction) IsValid() bool {
	switch a {
	case EscalateAbstain, EscalatePromptHuman, EscalateRerouteRole:
		return true
	default:
		return false
	}
}

// WorkerBudget bounds the resources a worker may consume in a single invocation.
type WorkerBudget struct {
	MaxTurns        int           `json:"max_turns,omitempty"`
	MaxInputTokens  int           `json:"max_input_tokens,omitempty"`
	MaxOutputTokens int           `json:"max_output_tokens,omitempty"`
	Timeout         time.Duration `json:"timeout,omitempty"`
}

// Validate checks budget limits for negative values.
func (b WorkerBudget) Validate() error {
	if b.MaxTurns < 0 {
		return fmt.Errorf("max_turns cannot be negative: %d", b.MaxTurns)
	}
	if b.MaxInputTokens < 0 {
		return fmt.Errorf("max_input_tokens cannot be negative: %d", b.MaxInputTokens)
	}
	if b.MaxOutputTokens < 0 {
		return fmt.Errorf("max_output_tokens cannot be negative: %d", b.MaxOutputTokens)
	}
	if b.Timeout < 0 {
		return fmt.Errorf("timeout cannot be negative: %v", b.Timeout)
	}
	return nil
}

// WitnessRequirements configures verification criteria that must pass before a receipt is considered completed.
type WitnessRequirements struct {
	RequireIndependentWitness bool          `json:"require_independent_witness,omitempty"`
	WitnessCommand            string        `json:"witness_command,omitempty"`
	RequireZeroExitCode       bool          `json:"require_zero_exit_code,omitempty"`
	WitnessTimeout            time.Duration `json:"witness_timeout,omitempty"`
	VerifyArtifactIntegrity   bool          `json:"verify_artifact_integrity,omitempty"`
}

// Validate checks witness settings.
func (w WitnessRequirements) Validate() error {
	if w.WitnessTimeout < 0 {
		return fmt.Errorf("witness_timeout cannot be negative: %v", w.WitnessTimeout)
	}
	return nil
}

// ModelPlacement defines model selection and sampling parameters for a worker.
type ModelPlacement struct {
	Provider       string  `json:"provider,omitempty"`
	Model          string  `json:"model,omitempty"`
	Effort         string  `json:"effort,omitempty"`
	ThinkingBudget int     `json:"thinking_budget,omitempty"`
	Temperature    float64 `json:"temperature,omitempty"`
}

// Validate checks placement parameters.
func (p ModelPlacement) Validate() error {
	if p.ThinkingBudget < 0 {
		return fmt.Errorf("thinking_budget cannot be negative: %d", p.ThinkingBudget)
	}
	if p.Temperature < 0.0 {
		return fmt.Errorf("temperature cannot be negative: %f", p.Temperature)
	}
	return nil
}

// WorkerSpec declares the behavior, permissions, and operational constraints for a worker role.
type WorkerSpec struct {
	RoleID              string              `json:"role_id"`
	Purpose             string              `json:"purpose,omitempty"`
	InstructionSnapshot string              `json:"instruction_snapshot,omitempty"`
	InstructionTemplate string              `json:"instruction_template,omitempty"`
	AccessMode          AccessMode          `json:"access_mode"`
	ToolScope           ToolScope           `json:"tool_scope"`
	Budget              WorkerBudget        `json:"budget"`
	Witness             WitnessRequirements `json:"witness,omitempty"`
	Placement           ModelPlacement      `json:"placement,omitempty"`
	Metadata            map[string]string   `json:"metadata,omitempty"`
}

// Validate checks worker role invariants, access mode consistency, and bounds.
func (w WorkerSpec) Validate() error {
	if strings.TrimSpace(w.RoleID) == "" {
		return &Error{Code: CodeInvalid, Op: "worker.validate", Err: errors.New("worker role_id is required")}
	}
	if !w.AccessMode.IsValid() {
		return &Error{Code: CodeInvalid, Op: "worker.validate", Err: fmt.Errorf("invalid access_mode %q", w.AccessMode)}
	}
	// AccessMode semantics: observe workers must not cause mutations or alter worktree.
	if w.AccessMode == AccessModeObserve {
		if w.ToolScope.AllowWorktree {
			return &Error{Code: CodeInvalid, Op: "worker.validate", Err: fmt.Errorf("worker %q: access_mode observe cannot allow worktree modifications", w.RoleID)}
		}
		if w.ToolScope.MaxMutations > 0 {
			return &Error{Code: CodeInvalid, Op: "worker.validate", Err: fmt.Errorf("worker %q: access_mode observe cannot permit mutations (max_mutations=%d)", w.RoleID, w.ToolScope.MaxMutations)}
		}
		if w.ToolScope.Mutability == MutabilityMutating || w.ToolScope.Mutability == MutabilityDestructive {
			return &Error{Code: CodeInvalid, Op: "worker.validate", Err: fmt.Errorf("worker %q: access_mode observe cannot declare mutability %q", w.RoleID, w.ToolScope.Mutability)}
		}
	}
	if err := w.Budget.Validate(); err != nil {
		return &Error{Code: CodeInvalid, Op: "worker.validate", Err: err}
	}
	if err := w.Witness.Validate(); err != nil {
		return &Error{Code: CodeInvalid, Op: "worker.validate", Err: err}
	}
	if err := w.Placement.Validate(); err != nil {
		return &Error{Code: CodeInvalid, Op: "worker.validate", Err: err}
	}
	return nil
}

// EscalationGate inspects task scope and patterns to intercept risk before execution.
type EscalationGate struct {
	Name         string               `json:"name"`
	PathPatterns []string             `json:"path_patterns,omitempty"`
	RiskKeywords []string             `json:"risk_keywords,omitempty"`
	Action       RiskEscalationAction `json:"action"`
	TargetRole   string               `json:"target_role,omitempty"`
}

// Validate checks escalation gate configuration.
func (g EscalationGate) Validate() error {
	if strings.TrimSpace(g.Name) == "" {
		return &Error{Code: CodeInvalid, Op: "escalation_gate.validate", Err: errors.New("gate name is required")}
	}
	if !g.Action.IsValid() {
		return &Error{Code: CodeInvalid, Op: "escalation_gate.validate", Err: fmt.Errorf("invalid escalation action %q", g.Action)}
	}
	if g.Action == EscalateRerouteRole && strings.TrimSpace(g.TargetRole) == "" {
		return &Error{Code: CodeInvalid, Op: "escalation_gate.validate", Err: errors.New("target_role is required when action is reroute_role")}
	}
	return nil
}

// ReceiptPolicy establishes acceptance gates and compact fold limits for receipts.
type ReceiptPolicy struct {
	StrictReceipt      bool `json:"strict_receipt,omitempty"`
	QuarantineFailures bool `json:"quarantine_failures,omitempty"`
	RequireWitnessPass bool `json:"require_witness_pass,omitempty"`
	MaxFoldTokens      int  `json:"max_fold_tokens,omitempty"`
}

// Validate checks receipt policy settings.
func (p ReceiptPolicy) Validate() error {
	if p.MaxFoldTokens < 0 {
		return &Error{Code: CodeInvalid, Op: "receipt_policy.validate", Err: fmt.Errorf("max_fold_tokens cannot be negative: %d", p.MaxFoldTokens)}
	}
	return nil
}

// DefaultMaxConcurrency is the default concurrency limit for manager dispatch.
const DefaultMaxConcurrency = 4

// ManagerSpec defines governance, concurrency, and strategy settings for orchestration.
type ManagerSpec struct {
	MaxConcurrency  int              `json:"max_concurrency,omitempty"`
	DefaultStrategy StrategyKind     `json:"default_strategy,omitempty"`
	ReceiptPolicy   ReceiptPolicy    `json:"receipt_policy,omitempty"`
	EscalationGates []EscalationGate `json:"escalation_gates,omitempty"`
	TokenCap        int              `json:"token_cap,omitempty"`
}

// ApplyDefaults fills in omitted values with sensible defaults.
func (m *ManagerSpec) ApplyDefaults() {
	if m.MaxConcurrency <= 0 {
		m.MaxConcurrency = DefaultMaxConcurrency
	}
	if m.DefaultStrategy == "" {
		m.DefaultStrategy = StrategyFanOutFanIn
	}
}

// Validate checks manager configuration and nested gates.
func (m ManagerSpec) Validate() error {
	if m.MaxConcurrency < 0 {
		return &Error{Code: CodeInvalid, Op: "manager.validate", Err: fmt.Errorf("max_concurrency cannot be negative: %d", m.MaxConcurrency)}
	}
	if m.TokenCap < 0 {
		return &Error{Code: CodeInvalid, Op: "manager.validate", Err: fmt.Errorf("token_cap cannot be negative: %d", m.TokenCap)}
	}
	if m.DefaultStrategy != "" && !m.DefaultStrategy.IsValid() {
		return &Error{Code: CodeInvalid, Op: "manager.validate", Err: fmt.Errorf("invalid default_strategy %q", m.DefaultStrategy)}
	}
	if err := m.ReceiptPolicy.Validate(); err != nil {
		return err
	}
	for _, g := range m.EscalationGates {
		if err := g.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// ManifestMetadata provides human and machine identity for a manifest.
type ManifestMetadata struct {
	Name        string            `json:"name"`
	Version     string            `json:"version,omitempty"`
	Description string            `json:"description,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// Validate checks metadata presence.
func (m ManifestMetadata) Validate() error {
	if strings.TrimSpace(m.Name) == "" {
		return &Error{Code: CodeInvalid, Op: "manifest_metadata.validate", Err: errors.New("metadata name is required")}
	}
	return nil
}

// TopologyRule defines permissible role-to-role transitions or delegation links.
type TopologyRule struct {
	FromRole  string `json:"from_role"`
	ToRole    string `json:"to_role"`
	Condition string `json:"condition,omitempty"`
	Required  bool   `json:"required,omitempty"`
}

// Validate checks rule role definitions.
func (t TopologyRule) Validate() error {
	if strings.TrimSpace(t.FromRole) == "" {
		return &Error{Code: CodeInvalid, Op: "topology_rule.validate", Err: errors.New("from_role is required")}
	}
	if strings.TrimSpace(t.ToRole) == "" {
		return &Error{Code: CodeInvalid, Op: "topology_rule.validate", Err: errors.New("to_role is required")}
	}
	return nil
}

// CoordinationManifest is the complete, declarative description of a manager/worker topology.
type CoordinationManifest struct {
	SchemaVersion string                `json:"schema_version"`
	Metadata      ManifestMetadata      `json:"metadata"`
	Manager       ManagerSpec           `json:"manager"`
	Workers       map[string]WorkerSpec `json:"workers"`
	Topology      []TopologyRule        `json:"topology,omitempty"`
}

// Validate verifies manifest structure, schemas, and referential integrity.
func (m CoordinationManifest) Validate() error {
	if strings.TrimSpace(m.SchemaVersion) == "" {
		return &Error{Code: CodeInvalid, Op: "manifest.validate", Err: errors.New("schema_version is required")}
	}
	if m.SchemaVersion != CoordinationContractVersion {
		return &Error{Code: CodeUnsupported, Op: "manifest.validate", Err: fmt.Errorf("unsupported schema_version %q (expected %q)", m.SchemaVersion, CoordinationContractVersion)}
	}
	if err := m.Metadata.Validate(); err != nil {
		return err
	}
	if err := m.Manager.Validate(); err != nil {
		return err
	}
	if len(m.Workers) == 0 {
		return &Error{Code: CodeInvalid, Op: "manifest.validate", Err: errors.New("at least one worker role must be defined")}
	}
	for roleKey, worker := range m.Workers {
		if strings.TrimSpace(worker.RoleID) == "" {
			return &Error{Code: CodeInvalid, Op: "manifest.validate", Err: fmt.Errorf("worker %q is missing role_id", roleKey)}
		}
		if worker.RoleID != roleKey {
			return &Error{Code: CodeConflict, Op: "manifest.validate", Err: fmt.Errorf("worker map key %q does not match role_id %q", roleKey, worker.RoleID)}
		}
		if err := worker.Validate(); err != nil {
			return err
		}
	}
	for _, top := range m.Topology {
		if err := top.Validate(); err != nil {
			return err
		}
		if _, exists := m.Workers[top.FromRole]; !exists {
			return &Error{Code: CodeInvalid, Op: "manifest.validate", Err: fmt.Errorf("topology references unknown from_role %q", top.FromRole)}
		}
		if _, exists := m.Workers[top.ToRole]; !exists {
			return &Error{Code: CodeInvalid, Op: "manifest.validate", Err: fmt.Errorf("topology references unknown to_role %q", top.ToRole)}
		}
	}
	for _, gate := range m.Manager.EscalationGates {
		if gate.TargetRole != "" {
			if _, exists := m.Workers[gate.TargetRole]; !exists {
				return &Error{Code: CodeInvalid, Op: "manifest.validate", Err: fmt.Errorf("escalation gate %q references unknown target_role %q", gate.Name, gate.TargetRole)}
			}
		}
	}
	return nil
}

// ParseCoordinationManifest decodes JSON into a CoordinationManifest, strictly disallowing unknown fields.
func ParseCoordinationManifest(b []byte) (CoordinationManifest, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var manifest CoordinationManifest
	if err := dec.Decode(&manifest); err != nil {
		return CoordinationManifest{}, &Error{Code: CodeInvalid, Op: "parse_manifest", Err: err}
	}
	if dec.More() {
		return CoordinationManifest{}, &Error{Code: CodeInvalid, Op: "parse_manifest", Err: errors.New("unexpected trailing content after json manifest")}
	}
	if manifest.Manager.MaxConcurrency == 0 {
		manifest.Manager.ApplyDefaults()
	}
	if err := manifest.Validate(); err != nil {
		return CoordinationManifest{}, err
	}
	return manifest, nil
}

// DiffSummary summarizes filesystem diffs produced by an effect worker.
type DiffSummary struct {
	FilesChanged int `json:"files_changed,omitempty"`
	Insertions   int `json:"insertions,omitempty"`
	Deletions    int `json:"deletions,omitempty"`
}

// WitnessResult captures execution telemetry and validation outcome from a verification witness.
type WitnessResult struct {
	Command      string        `json:"command,omitempty"`
	ExitCode     int           `json:"exit_code,omitempty"`
	Duration     time.Duration `json:"duration,omitempty"`
	OutputDigest string        `json:"output_digest,omitempty"`
	Passed       bool          `json:"passed"`
}

// TokenBreakdown accounts for LLM token usage during task execution.
type TokenBreakdown struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
	TotalTokens  int `json:"total_tokens,omitempty"`
}

// FailureDiagnosis provides machine-actionable triage information on failure or abstention.
type FailureDiagnosis struct {
	ReasonCategory   string   `json:"reason_category,omitempty"`
	Message          string   `json:"message,omitempty"`
	FailingSeam      string   `json:"failing_seam,omitempty"`
	SuggestedRole    string   `json:"suggested_role,omitempty"`
	UnmetAssumptions []string `json:"unmet_assumptions,omitempty"`
}

// WorkerReceipt is the immutable structured receipt emitted upon completion of a worker task.
type WorkerReceipt struct {
	TaskID          string            `json:"task_id"`
	WorkerID        string            `json:"worker_id"`
	RoleID          string            `json:"role_id"`
	Status          ReceiptStatus     `json:"status"`
	Summary         string            `json:"summary,omitempty"`
	TouchedFiles    []string          `json:"touched_files,omitempty"`
	GitOID          string            `json:"git_oid,omitempty"`
	Diff            DiffSummary       `json:"diff,omitempty"`
	Witness         WitnessResult     `json:"witness,omitempty"`
	Artifacts       map[string]string `json:"artifacts,omitempty"`
	Tokens          TokenBreakdown    `json:"tokens,omitempty"`
	Diagnosis       *FailureDiagnosis `json:"diagnosis,omitempty"`
	ExecutionTimeMS int64             `json:"execution_time_ms,omitempty"`
}

// Validate verifies mandatory receipt fields and status consistency.
func (r WorkerReceipt) Validate() error {
	if strings.TrimSpace(r.TaskID) == "" {
		return &Error{Code: CodeInvalid, Op: "receipt.validate", Err: errors.New("receipt task_id is required")}
	}
	if strings.TrimSpace(r.WorkerID) == "" {
		return &Error{Code: CodeInvalid, Op: "receipt.validate", Err: errors.New("receipt worker_id is required")}
	}
	if strings.TrimSpace(r.RoleID) == "" {
		return &Error{Code: CodeInvalid, Op: "receipt.validate", Err: errors.New("receipt role_id is required")}
	}
	if !r.Status.IsValid() {
		return &Error{Code: CodeInvalid, Op: "receipt.validate", Err: fmt.Errorf("invalid status %q", r.Status)}
	}
	if r.ExecutionTimeMS < 0 {
		return &Error{Code: CodeInvalid, Op: "receipt.validate", Err: fmt.Errorf("execution_time_ms cannot be negative: %d", r.ExecutionTimeMS)}
	}
	return nil
}

// Savings calculates the context savings realized by folding raw worker turns into this receipt.
func (r WorkerReceipt) Savings(rawTranscriptTokens int) ContextSavings {
	foldTokens := r.Tokens.TotalTokens
	return CalculateContextSavings(rawTranscriptTokens, foldTokens)
}

// ContextSavings quantifies token reduction achieved by receipt folding over intermediate conversation history.
type ContextSavings struct {
	RawTranscriptTokens int     `json:"raw_transcript_tokens"`
	ReceiptFoldTokens   int     `json:"receipt_fold_tokens"`
	SavedTokens         int     `json:"saved_tokens"`
	CompressionRatio    float64 `json:"compression_ratio"`
}

// CalculateContextSavings computes token savings and compression ratio.
func CalculateContextSavings(rawTranscriptTokens, receiptFoldTokens int) ContextSavings {
	if rawTranscriptTokens < 0 {
		rawTranscriptTokens = 0
	}
	if receiptFoldTokens < 0 {
		receiptFoldTokens = 0
	}
	saved := rawTranscriptTokens - receiptFoldTokens
	if saved < 0 {
		saved = 0
	}
	var ratio float64
	if receiptFoldTokens > 0 {
		ratio = float64(rawTranscriptTokens) / float64(receiptFoldTokens)
	} else if rawTranscriptTokens > 0 {
		ratio = float64(rawTranscriptTokens)
	}
	return ContextSavings{
		RawTranscriptTokens: rawTranscriptTokens,
		ReceiptFoldTokens:   receiptFoldTokens,
		SavedTokens:         saved,
		CompressionRatio:    ratio,
	}
}

// NetSavingsPercentage returns the percentage of tokens saved relative to raw transcript volume.
func (s ContextSavings) NetSavingsPercentage() float64 {
	if s.RawTranscriptTokens <= 0 {
		return 0.0
	}
	return (float64(s.SavedTokens) / float64(s.RawTranscriptTokens)) * 100.0
}

// CoordinationContract publishes machine-readable coordination metadata.
type CoordinationContract struct {
	SchemaVersion   string   `json:"schema_version"`
	Strategies      []string `json:"strategies"`
	AccessModes     []string `json:"access_modes"`
	ReceiptStatuses []string `json:"receipt_statuses"`
	Escalations     []string `json:"escalations"`
	Isolation       string   `json:"isolation"`
	ReceiptFolding  string   `json:"receipt_folding"`
	Security        string   `json:"security_reachability"`
	Cancellation    string   `json:"cancellation"`
	Errors          []Code   `json:"errors"`
}

// PublicCoordinationContract returns the normative coordination plane contract.
func PublicCoordinationContract() CoordinationContract {
	return CoordinationContract{
		SchemaVersion:   CoordinationContractVersion,
		Strategies:      []string{string(StrategyFanOutFanIn), string(StrategySequential), string(StrategyAdaptiveDAG), string(StrategySpeculative)},
		AccessModes:     []string{string(AccessModeObserve), string(AccessModeEffect)},
		ReceiptStatuses: []string{string(StatusCompleted), string(StatusFailed), string(StatusAbstain), string(StatusTimedOut)},
		Escalations:     []string{string(EscalateAbstain), string(EscalatePromptHuman), string(EscalateRerouteRole)},
		Isolation:       "workers execute in distinct isolation contexts without shared mutable memory",
		ReceiptFolding:  "completed tasks fold into compact structured receipts, suppressing raw intermediate turns from manager context",
		Security:        "worker reachability is bounded by declared tool scopes; observe workers cannot mutate state",
		Cancellation:    "context cancellation propagates to active worker tasks and aborts speculative branches",
		Errors:          []Code{CodeInvalid, CodeUnsupported, CodeConflict, CodeDenied, CodeCanceled, CodeBackpressure, CodeInternal},
	}
}

// Manager orchestrates worker roles according to a coordination manifest.
type Manager interface {
	Manifest() CoordinationManifest
	Dispatch(ctx context.Context, roleID string, input Invocation) (WorkerReceipt, error)
	ExecuteStrategy(ctx context.Context, strategy StrategyKind, inputs map[string]Invocation) (map[string]WorkerReceipt, error)
}

// WorkerPool manages the lifecycle and allocation of workers.
type WorkerPool interface {
	Acquire(ctx context.Context, roleID string) (WorkerDispatcher, error)
	Release(ctx context.Context, roleID string, worker WorkerDispatcher) error
	Available(roleID string) int
	Capacity(roleID string) int
}

// WorkerDispatcher dispatches tasks to an isolated worker instance.
type WorkerDispatcher interface {
	Role() string
	Dispatch(ctx context.Context, taskID string, input Invocation) (WorkerReceipt, error)
	Cancel(taskID string) error
}

// ReceiptFoldHandler processes worker receipts and compacts them into manager context.
type ReceiptFoldHandler interface {
	Fold(ctx context.Context, receipt WorkerReceipt) (ContextSavings, error)
}
