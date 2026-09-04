package microagent

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
)

// ReceiptStatus defines the closed status vocabulary for worker receipts.
type ReceiptStatus string

const (
	StatusCompleted ReceiptStatus = "COMPLETED"
	StatusFailed    ReceiptStatus = "FAILED"
	StatusAbstain   ReceiptStatus = "ABSTAIN"
)

// WorkerReceipt is the compact, author-neutral deliverable artifact handed back
// by an isolated worker subagent to the clean coordinator. It deliberately does
// not embed raw compiler outputs, shell stdout/stderr, or multi-turn transcripts.
type WorkerReceipt struct {
	TaskID           string        `json:"task_id"`
	Status           ReceiptStatus `json:"status"`
	TouchedFiles     []string      `json:"touched_files"`
	WitnessCommand   string        `json:"witness_command"`
	WitnessExitCode  int           `json:"witness_exit_code"`
	GitSHA           string        `json:"git_sha,omitempty"`
	AbstainRationale string        `json:"abstain_rationale,omitempty"`
	TokensUsed       int           `json:"tokens_used"`
	Summary          string        `json:"summary"`
}

// CompactJSON returns a single-line, deterministically serialized JSON string
// representing the receipt for compact inclusion in context.
func (r WorkerReceipt) CompactJSON() string {
	data, err := json.Marshal(r)
	if err != nil {
		return fmt.Sprintf(`{"task_id":%q,"status":%q,"summary":%q}`, r.TaskID, r.Status, r.Summary)
	}
	return string(data)
}

// Validate checks the structural invariants of the worker receipt.
func (r WorkerReceipt) Validate() error {
	if strings.TrimSpace(r.TaskID) == "" {
		return errors.New("microagent: task_id is required")
	}
	switch r.Status {
	case StatusCompleted, StatusFailed, StatusAbstain:
	default:
		return fmt.Errorf("microagent: invalid receipt status %q", r.Status)
	}
	if len(r.TouchedFiles) > 3 {
		return fmt.Errorf("microagent: touched files count %d exceeds atomic S0/S1 ceiling of 3", len(r.TouchedFiles))
	}
	if r.Status == StatusCompleted {
		if r.WitnessExitCode != 0 {
			return errors.New("microagent: completed status requires witness exit code 0")
		}
		if strings.TrimSpace(r.WitnessCommand) == "" {
			return errors.New("microagent: completed status requires witness command")
		}
	}
	if r.Status == StatusAbstain && strings.TrimSpace(r.AbstainRationale) == "" {
		return errors.New("microagent: abstain status requires abstain rationale")
	}
	return nil
}

// RiskCategory classifies high-difficulty architectural boundaries that require
// smaller or fast worker models to fail-to-abstain rather than speculate.
type RiskCategory string

const (
	RiskConcurrencyLocks  RiskCategory = "CONCURRENCY_LOCK_INVARIANTS"
	RiskFrozenABI         RiskCategory = "FROZEN_ABI"
	RiskLowLevelKernels   RiskCategory = "LOW_LEVEL_KERNELS"
	RiskSecurityPolicy    RiskCategory = "SECURITY_POLICY_GATE"
	RiskProtocolMigration RiskCategory = "PROTOCOL_MIGRATION"
)

// Escalation captures a structured ABSTAIN verdict routed to higher-capability
// models or human operators.
type Escalation struct {
	TaskID     string        `json:"task_id"`
	Risk       RiskCategory  `json:"risk_category"`
	Rationale  string        `json:"rationale"`
	EscalateTo string        `json:"escalate_to"`
	Receipt    WorkerReceipt `json:"receipt"`
}

// DetectHighRiskBoundary inspects target file paths and deliverable descriptions
// to detect if a task intersects known high-difficulty architectural boundaries.
func DetectHighRiskBoundary(files []string, description string) (RiskCategory, bool) {
	for _, f := range files {
		fLower := strings.ToLower(f)
		if strings.HasPrefix(fLower, "internal/abi") || strings.Contains(fLower, "/abi/") {
			return RiskFrozenABI, true
		}
		if strings.Contains(fLower, "cuda") || strings.Contains(fLower, "simd") || strings.Contains(fLower, "kernel") || strings.Contains(fLower, "vdso") {
			return RiskLowLevelKernels, true
		}
		if strings.Contains(fLower, "policy") || strings.Contains(fLower, "capability") || strings.Contains(fLower, "guard") {
			return RiskSecurityPolicy, true
		}
	}

	descLower := strings.ToLower(description)
	if strings.Contains(descLower, "concurrency") || strings.Contains(descLower, "mutex") || strings.Contains(descLower, "lock invariant") || strings.Contains(descLower, "lock ordering") || strings.Contains(descLower, "deadlock") {
		return RiskConcurrencyLocks, true
	}
	if strings.Contains(descLower, "frozen abi") || strings.Contains(descLower, "abi modification") || strings.Contains(descLower, "internal/abi") {
		return RiskFrozenABI, true
	}
	if strings.Contains(descLower, "cuda") || strings.Contains(descLower, "simd") || strings.Contains(descLower, "gpu kernel") {
		return RiskLowLevelKernels, true
	}
	if strings.Contains(descLower, "protocol migration") || strings.Contains(descLower, "wire format") {
		return RiskProtocolMigration, true
	}
	if strings.Contains(descLower, "security policy") || strings.Contains(descLower, "capability floor") {
		return RiskSecurityPolicy, true
	}

	return "", false
}

// CoordinatorTask is an atomic S0/S1 unit of work admitted to the coordinator.
type CoordinatorTask struct {
	ID             string   `json:"id"`
	Deliverable    string   `json:"deliverable"`
	TargetFiles    []string `json:"target_files"`
	WitnessCommand string   `json:"witness_command"`
	Subsystem      string   `json:"subsystem,omitempty"`
}

var (
	ErrTaskIDRequired       = errors.New("microagent: task id is required")
	ErrDeliverableRequired  = errors.New("microagent: deliverable is required")
	ErrInvalidFileCount     = errors.New("microagent: atomic S0/S1 tasks must target 1 to 3 files")
	ErrWitnessCommandNeeded = errors.New("microagent: atomic S0/S1 tasks require exactly one witness command")
	ErrChainedWitnessCmd    = errors.New("microagent: witness command must be exactly one command, not a chained pipeline")
	ErrDuplicateTask        = errors.New("microagent: duplicate task id")
	ErrTaskNotFound         = errors.New("microagent: task not registered")
)

// ValidateS0S1 validates that the task satisfies atomic S0/S1 constraints:
// 1. Exactly 1 observable deliverable.
// 2. 1 to 3 target files within a single package or lane.
// 3. Exactly 1 witness command (no chained scripts or pipes).
func (t CoordinatorTask) ValidateS0S1() error {
	if strings.TrimSpace(t.ID) == "" {
		return ErrTaskIDRequired
	}
	if strings.TrimSpace(t.Deliverable) == "" {
		return ErrDeliverableRequired
	}
	if len(t.TargetFiles) < 1 || len(t.TargetFiles) > 3 {
		return ErrInvalidFileCount
	}
	for _, f := range t.TargetFiles {
		if strings.TrimSpace(f) == "" {
			return errors.New("microagent: target file path cannot be empty")
		}
	}
	cmd := strings.TrimSpace(t.WitnessCommand)
	if cmd == "" {
		return ErrWitnessCommandNeeded
	}
	if strings.Contains(cmd, "&&") || strings.Contains(cmd, ";") || strings.Contains(cmd, "||") || strings.Contains(cmd, "\n") {
		return ErrChainedWitnessCmd
	}
	return nil
}

// ContextSavings tracks the token and byte reduction achieved by ingesting
// compact receipts instead of raw worker transcripts.
type ContextSavings struct {
	RawTokens      int     `json:"raw_tokens"`
	ReceiptTokens  int     `json:"receipt_tokens"`
	SavedTokens    int     `json:"saved_tokens"`
	ReductionRatio float64 `json:"reduction_ratio"`
	RawBytes       int     `json:"raw_bytes"`
	ReceiptBytes   int     `json:"receipt_bytes"`
	SavedBytes     int     `json:"saved_bytes"`
}

// CoordinatorConfig configures a clean Coordinator instance.
type CoordinatorConfig struct {
	TokenCap           int
	QuarantineFailures bool // When true, failed tasks do not append to active context
}

// Coordinator manages bulk backlog processing across bounded microagents.
// It enforces atomic S0/S1 task decomposition, protects its context from raw
// execution chatter, captures typed receipts, computes savings, and escalates
// structured ABSTAIN boundaries.
type Coordinator struct {
	mu             sync.Mutex
	cfg            CoordinatorConfig
	tasks          map[string]CoordinatorTask
	taskOrder      []string
	receipts       map[string]WorkerReceipt
	escalations    []Escalation
	context        *Context
	rawTokens      int
	rawBytes       int
	receiptTokens  int
	receiptBytes   int
	pollutedTokens int
}

// NewCoordinator initializes a new clean coordinator.
func NewCoordinator(cfg CoordinatorConfig) *Coordinator {
	if cfg.TokenCap <= 0 {
		cfg.TokenCap = DefaultContextCap
	}
	return &Coordinator{
		cfg:      cfg,
		tasks:    make(map[string]CoordinatorTask),
		receipts: make(map[string]WorkerReceipt),
		context:  NewContext(cfg.TokenCap),
	}
}

// RegisterTask admits a new atomic S0/S1 task after validating its bounds.
func (c *Coordinator) RegisterTask(t CoordinatorTask) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := t.ValidateS0S1(); err != nil {
		return err
	}
	if _, exists := c.tasks[t.ID]; exists {
		return fmt.Errorf("%w: %s", ErrDuplicateTask, t.ID)
	}

	c.tasks[t.ID] = t
	c.taskOrder = append(c.taskOrder, t.ID)
	return nil
}

// GetTask returns a registered task by ID.
func (c *Coordinator) GetTask(id string) (CoordinatorTask, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	t, ok := c.tasks[id]
	return t, ok
}

// Tasks returns all registered tasks in registration order.
func (c *Coordinator) Tasks() []CoordinatorTask {
	c.mu.Lock()
	defer c.mu.Unlock()
	res := make([]CoordinatorTask, 0, len(c.taskOrder))
	for _, id := range c.taskOrder {
		res = append(res, c.tasks[id])
	}
	return res
}

// IngestReceipt accepts a WorkerReceipt from an isolated worker, validates it,
// computes token and byte savings against optional raw transcript chatter,
// handles ABSTAIN escalations, and appends only compact typed information into
// coordinator context.
func (c *Coordinator) IngestReceipt(receipt WorkerReceipt, rawTranscript ...string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := receipt.Validate(); err != nil {
		return err
	}

	task, exists := c.tasks[receipt.TaskID]
	if !exists {
		return fmt.Errorf("%w: %s", ErrTaskNotFound, receipt.TaskID)
	}

	// Validate S0/S1 bounds of the receipt
	if len(receipt.TouchedFiles) > 3 {
		return fmt.Errorf("microagent: touched files count %d exceeds atomic S0/S1 ceiling of 3", len(receipt.TouchedFiles))
	}
	if receipt.Status == StatusCompleted && receipt.WitnessCommand != task.WitnessCommand {
		return fmt.Errorf("microagent: receipt witness command %q does not match task witness command %q", receipt.WitnessCommand, task.WitnessCommand)
	}

	// Calculate raw transcript size
	var rawText string
	if len(rawTranscript) > 0 {
		rawText = strings.Join(rawTranscript, "\n")
	}
	rawBytes := len(rawText)
	rawTokens := 0
	if rawText != "" {
		rawTokens = estContextTokens([]Msg{{Role: "assistant", Content: rawText}})
	} else if receipt.TokensUsed > 0 {
		rawTokens = receipt.TokensUsed
	}

	// Compact receipt representation for coordinator context
	compactJSON := receipt.CompactJSON()
	receiptBytes := len(compactJSON)
	receiptTokens := estContextTokens([]Msg{{Role: "assistant", Content: compactJSON}})

	c.rawTokens += rawTokens
	c.rawBytes += rawBytes
	c.receiptTokens += receiptTokens
	c.receiptBytes += receiptBytes

	c.receipts[receipt.TaskID] = receipt

	// Handle structured ABSTAIN escalation
	if receipt.Status == StatusAbstain {
		risk, ok := DetectHighRiskBoundary(task.TargetFiles, task.Deliverable+" "+task.Subsystem)
		if !ok {
			risk, ok = DetectHighRiskBoundary(receipt.TouchedFiles, receipt.AbstainRationale)
		}
		if !ok {
			risk = RiskCategory("UNSPECIFIED_RISK_BOUNDARY")
		}
		escalation := Escalation{
			TaskID:     receipt.TaskID,
			Risk:       risk,
			Rationale:  receipt.AbstainRationale,
			EscalateTo: "higher_capability_model",
			Receipt:    receipt,
		}
		c.escalations = append(c.escalations, escalation)
	}

	// Fail-closed context isolation:
	// If the sub-task failed and failure quarantine is active, zero context is appended.
	if receipt.Status == StatusFailed && c.cfg.QuarantineFailures {
		c.pollutedTokens += rawTokens
		return nil
	}

	// The clean coordinator invariant: NEVER append rawText. ONLY append the compact receipt.
	c.context.Append("assistant", compactJSON)
	return nil
}

// ContextSavings computes the token and byte reduction between the raw execution
// transcripts and the compact receipts retained in coordinator context.
func (c *Coordinator) ContextSavings() ContextSavings {
	c.mu.Lock()
	defer c.mu.Unlock()

	activeTokens := c.context.Tokens()
	savedTokens := c.rawTokens - activeTokens
	if savedTokens < 0 {
		savedTokens = 0
	}
	var ratio float64
	if c.rawTokens > 0 {
		ratio = float64(savedTokens) / float64(c.rawTokens)
	}

	savedBytes := c.rawBytes - c.receiptBytes
	if savedBytes < 0 {
		savedBytes = 0
	}

	return ContextSavings{
		RawTokens:      c.rawTokens,
		ReceiptTokens:  activeTokens,
		SavedTokens:    savedTokens,
		ReductionRatio: ratio,
		RawBytes:       c.rawBytes,
		ReceiptBytes:   c.receiptBytes,
		SavedBytes:     savedBytes,
	}
}

// Context returns the coordinator's clean message context.
func (c *Coordinator) Context() *Context {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.context
}

// Receipts returns all ingested worker receipts.
func (c *Coordinator) Receipts() map[string]WorkerReceipt {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make(map[string]WorkerReceipt, len(c.receipts))
	for k, v := range c.receipts {
		out[k] = v
	}
	return out
}

// GetReceipt retrieves an ingested receipt by task ID.
func (c *Coordinator) GetReceipt(taskID string) (WorkerReceipt, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	r, ok := c.receipts[taskID]
	return r, ok
}

// Escalations returns all typed ABSTAIN escalations recorded by the coordinator.
func (c *Coordinator) Escalations() []Escalation {
	c.mu.Lock()
	defer c.mu.Unlock()
	res := make([]Escalation, len(c.escalations))
	copy(res, c.escalations)
	return res
}

// CompletedTasks returns all receipts with status COMPLETED.
func (c *Coordinator) CompletedTasks() []WorkerReceipt {
	c.mu.Lock()
	defer c.mu.Unlock()
	var res []WorkerReceipt
	for _, id := range c.taskOrder {
		if r, ok := c.receipts[id]; ok && r.Status == StatusCompleted {
			res = append(res, r)
		}
	}
	return res
}

// FailedTasks returns all receipts with status FAILED.
func (c *Coordinator) FailedTasks() []WorkerReceipt {
	c.mu.Lock()
	defer c.mu.Unlock()
	var res []WorkerReceipt
	for _, id := range c.taskOrder {
		if r, ok := c.receipts[id]; ok && r.Status == StatusFailed {
			res = append(res, r)
		}
	}
	return res
}

// AbstainedTasks returns all receipts with status ABSTAIN.
func (c *Coordinator) AbstainedTasks() []WorkerReceipt {
	c.mu.Lock()
	defer c.mu.Unlock()
	var res []WorkerReceipt
	for _, id := range c.taskOrder {
		if r, ok := c.receipts[id]; ok && r.Status == StatusAbstain {
			res = append(res, r)
		}
	}
	return res
}
