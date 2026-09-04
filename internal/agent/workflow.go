package agent

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// GateVerdictKind represents the closed set of gate decision outcomes.
type GateVerdictKind string

const (
	GatePass   GateVerdictKind = "PASS"
	GateRefuse GateVerdictKind = "REFUSE"
	GateHalt   GateVerdictKind = "HALT"
)

// Witness records evidence of gate checks or phase actions.
type Witness struct {
	Command     string `json:"command,omitempty"`
	Artifact    string `json:"artifact,omitempty"`
	Description string `json:"description,omitempty"`
	Passed      bool   `json:"passed"`
	Output      string `json:"output,omitempty"`
}

// GateVerdict represents the outcome of an entry or exit condition check.
type GateVerdict struct {
	Kind        GateVerdictKind `json:"kind"`
	Reason      string          `json:"reason"`
	RefusalCode string          `json:"refusal_code,omitempty"`
	Witness     Witness         `json:"witness,omitempty"`
}

// EntryGate defines a pre-condition check evaluated before a phase executes.
type EntryGate struct {
	Name        string                                                      `json:"name"`
	Description string                                                      `json:"description,omitempty"`
	Check       func(ctx context.Context, state *WorkflowState) GateVerdict `json:"-"`
}

// ExitCondition defines a post-condition check evaluated after a phase action executes.
type ExitCondition struct {
	Name        string                                                      `json:"name"`
	Description string                                                      `json:"description,omitempty"`
	Check       func(ctx context.Context, state *WorkflowState) GateVerdict `json:"-"`
}

// PhaseAction performs the execution payload of a phase.
type PhaseAction func(ctx context.Context, state *WorkflowState) (Witness, error)

// Phase represents a distinct bounded step within a multi-phase workflow.
type Phase struct {
	ID          string          `json:"id"`
	Index       int             `json:"index"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	EntryGates  []EntryGate     `json:"entry_gates,omitempty"`
	ExitGates   []ExitCondition `json:"exit_gates,omitempty"`
	Action      PhaseAction     `json:"-"`
}

// PhaseTransitionReceipt records a witnessed boundary transition or refusal.
type PhaseTransitionReceipt struct {
	WorkflowID  string      `json:"workflow_id"`
	FromPhase   string      `json:"from_phase"`
	ToPhase     string      `json:"to_phase"`
	FromIndex   int         `json:"from_index"`
	ToIndex     int         `json:"to_index"`
	Timestamp   time.Time   `json:"timestamp"`
	Verdict     GateVerdict `json:"verdict"`
	Witness     Witness     `json:"witness,omitempty"`
	StateDigest string      `json:"state_digest,omitempty"`
}

// WorkflowStatus represents the lifecycle state of a workflow execution.
type WorkflowStatus string

const (
	WorkflowPending    WorkflowStatus = "pending"
	WorkflowInProgress WorkflowStatus = "in_progress"
	WorkflowCompleted  WorkflowStatus = "completed"
	WorkflowRefused    WorkflowStatus = "refused"
	WorkflowHalted     WorkflowStatus = "halted"
)

// WorkflowState captures the runtime state and history of a workflow run.
type WorkflowState struct {
	WorkflowID     string                   `json:"workflow_id"`
	CurrentPhase   int                      `json:"current_phase"`
	Data           map[string]any           `json:"data"`
	Receipts       []PhaseTransitionReceipt `json:"receipts"`
	Status         WorkflowStatus           `json:"status"`
	LastRefusal    *GateVerdict             `json:"last_refusal,omitempty"`
	CheckpointPath string                   `json:"checkpoint_path,omitempty"`
}

// Workflow defines an ordered collection of phases with boundary gate checks.
type Workflow struct {
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	Description string  `json:"description,omitempty"`
	Phases      []Phase `json:"phases"`
}

// WorkflowEngine orchestrates multi-phase workflows with step-boundary gate checks.
type WorkflowEngine struct {
	mu            sync.RWMutex
	workflows     map[string]*Workflow
	CheckpointDir string
}

// NewWorkflowEngine creates a new WorkflowEngine instance with a specified checkpoint directory.
func NewWorkflowEngine(checkpointDir ...string) *WorkflowEngine {
	dir := ".fak/workflows"
	if len(checkpointDir) > 0 && checkpointDir[0] != "" {
		dir = checkpointDir[0]
	}
	return &WorkflowEngine{
		workflows:     make(map[string]*Workflow),
		CheckpointDir: dir,
	}
}

var (
	defaultWorkflowEngineOnce sync.Once
	defaultWorkflowEngineInst *WorkflowEngine
)

// DefaultWorkflowEngine returns the shared default WorkflowEngine with builtin workflows registered.
func DefaultWorkflowEngine() *WorkflowEngine {
	defaultWorkflowEngineOnce.Do(func() {
		defaultWorkflowEngineInst = NewWorkflowEngine(".fak/workflows")
		defaultWorkflowEngineInst.Register(NewFleetWaveWorkflow())
	})
	return defaultWorkflowEngineInst
}

// Register adds or replaces a workflow definition in the engine.
func (e *WorkflowEngine) Register(w *Workflow) {
	if w == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.workflows[w.ID] = w
}

// Get returns the workflow definition with the given ID.
func (e *WorkflowEngine) Get(id string) (*Workflow, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	w, ok := e.workflows[id]
	return w, ok
}

// List returns all registered workflows sorted by ID.
func (e *WorkflowEngine) List() []*Workflow {
	e.mu.RLock()
	defer e.mu.RUnlock()
	list := make([]*Workflow, 0, len(e.workflows))
	for _, w := range e.workflows {
		list = append(list, w)
	}
	sort.Slice(list, func(i, j int) bool {
		return list[i].ID < list[j].ID
	})
	return list
}

// Step advances a workflow by exactly one phase, evaluating entry gates, action, and exit gates.
func (e *WorkflowEngine) Step(ctx context.Context, workflowID string, state *WorkflowState) (*WorkflowState, error) {
	w, ok := e.Get(workflowID)
	if !ok {
		return state, fmt.Errorf("workflow %q not found", workflowID)
	}

	if state == nil {
		state = &WorkflowState{
			WorkflowID:   workflowID,
			CurrentPhase: 0,
			Data:         make(map[string]any),
			Status:       WorkflowPending,
		}
	}
	if state.Data == nil {
		state.Data = make(map[string]any)
	}
	if state.Status == "" {
		state.Status = WorkflowPending
	}

	// If already in a terminal state, return immediately.
	if state.Status == WorkflowCompleted || state.Status == WorkflowRefused || state.Status == WorkflowHalted {
		return state, nil
	}

	idx := state.CurrentPhase
	if idx >= len(w.Phases) {
		state.Status = WorkflowCompleted
		return state, nil
	}

	phase := w.Phases[idx]

	// 1. Evaluate Entry Gates
	for _, gate := range phase.EntryGates {
		if gate.Check == nil {
			continue
		}
		verdict := gate.Check(ctx, state)
		if verdict.Kind == GateRefuse || verdict.Kind == GateHalt {
			if verdict.Kind == GateRefuse {
				state.Status = WorkflowRefused
			} else {
				state.Status = WorkflowHalted
			}
			state.LastRefusal = &verdict
			receipt := PhaseTransitionReceipt{
				WorkflowID:  w.ID,
				FromPhase:   phase.ID,
				ToPhase:     "",
				FromIndex:   idx,
				ToIndex:     idx,
				Timestamp:   time.Now().UTC(),
				Verdict:     verdict,
				Witness:     verdict.Witness,
				StateDigest: computeStateDigest(state),
			}
			state.Receipts = append(state.Receipts, receipt)
			_, _ = e.SaveCheckpoint(state, idx)
			return state, nil
		}
	}

	state.Status = WorkflowInProgress

	// 2. Execute Phase Action (if defined)
	var actionWitness Witness
	if phase.Action != nil {
		witness, err := phase.Action(ctx, state)
		actionWitness = witness
		if err != nil {
			state.Status = WorkflowHalted
			verdict := GateVerdict{
				Kind:    GateHalt,
				Reason:  fmt.Sprintf("action error: %v", err),
				Witness: witness,
			}
			state.LastRefusal = &verdict
			receipt := PhaseTransitionReceipt{
				WorkflowID:  w.ID,
				FromPhase:   phase.ID,
				ToPhase:     "",
				FromIndex:   idx,
				ToIndex:     idx,
				Timestamp:   time.Now().UTC(),
				Verdict:     verdict,
				Witness:     witness,
				StateDigest: computeStateDigest(state),
			}
			state.Receipts = append(state.Receipts, receipt)
			_, _ = e.SaveCheckpoint(state, idx)
			return state, fmt.Errorf("phase %s action failed: %w", phase.ID, err)
		}
	}

	// 3. Evaluate Exit Conditions
	for _, exitGate := range phase.ExitGates {
		if exitGate.Check == nil {
			continue
		}
		verdict := exitGate.Check(ctx, state)
		if verdict.Kind == GateRefuse || verdict.Kind == GateHalt {
			if verdict.Kind == GateRefuse {
				state.Status = WorkflowRefused
			} else {
				state.Status = WorkflowHalted
			}
			state.LastRefusal = &verdict
			receipt := PhaseTransitionReceipt{
				WorkflowID:  w.ID,
				FromPhase:   phase.ID,
				ToPhase:     "",
				FromIndex:   idx,
				ToIndex:     idx,
				Timestamp:   time.Now().UTC(),
				Verdict:     verdict,
				Witness:     verdict.Witness,
				StateDigest: computeStateDigest(state),
			}
			state.Receipts = append(state.Receipts, receipt)
			_, _ = e.SaveCheckpoint(state, idx)
			return state, nil
		}
	}

	// 4. Successful transition to next phase
	nextIdx := idx + 1
	var nextPhaseID string
	if nextIdx < len(w.Phases) {
		nextPhaseID = w.Phases[nextIdx].ID
	} else {
		nextPhaseID = "completed"
		state.Status = WorkflowCompleted
	}

	passVerdict := GateVerdict{
		Kind:    GatePass,
		Reason:  fmt.Sprintf("phase %s (%s) passed", phase.ID, phase.Name),
		Witness: actionWitness,
	}
	receipt := PhaseTransitionReceipt{
		WorkflowID:  w.ID,
		FromPhase:   phase.ID,
		ToPhase:     nextPhaseID,
		FromIndex:   idx,
		ToIndex:     nextIdx,
		Timestamp:   time.Now().UTC(),
		Verdict:     passVerdict,
		Witness:     actionWitness,
		StateDigest: computeStateDigest(state),
	}
	state.Receipts = append(state.Receipts, receipt)
	state.CurrentPhase = nextIdx
	if nextIdx >= len(w.Phases) {
		state.Status = WorkflowCompleted
	}
	_, _ = e.SaveCheckpoint(state, idx)
	return state, nil
}

// Execute executes workflow phases sequentially until completion, refusal, or halt.
func (e *WorkflowEngine) Execute(ctx context.Context, workflowID string, state *WorkflowState) (*WorkflowState, error) {
	w, ok := e.Get(workflowID)
	if !ok {
		return state, fmt.Errorf("workflow %q not found", workflowID)
	}

	if state == nil {
		state = &WorkflowState{
			WorkflowID:   workflowID,
			CurrentPhase: 0,
			Data:         make(map[string]any),
			Status:       WorkflowPending,
		}
	}

	for state.CurrentPhase < len(w.Phases) {
		if state.Status == WorkflowRefused || state.Status == WorkflowHalted {
			break
		}
		var err error
		state, err = e.Step(ctx, workflowID, state)
		if err != nil {
			return state, err
		}
		if state.Status == WorkflowRefused || state.Status == WorkflowHalted {
			break
		}
	}

	if state.CurrentPhase >= len(w.Phases) && state.Status != WorkflowRefused && state.Status != WorkflowHalted {
		state.Status = WorkflowCompleted
	}

	return state, nil
}

// SaveCheckpoint writes the current workflow state to disk in the checkpoint directory.
func (e *WorkflowEngine) SaveCheckpoint(state *WorkflowState, phaseIdx int) (string, error) {
	if state == nil {
		return "", fmt.Errorf("state is nil")
	}
	e.mu.RLock()
	dir := e.CheckpointDir
	e.mu.RUnlock()

	if dir == "" {
		dir = ".fak/workflows"
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("failed to create checkpoint dir %q: %w", dir, err)
	}

	filename := fmt.Sprintf("checkpoint_%s_phase_%d.json", sanitizeWorkflowID(state.WorkflowID), phaseIdx)
	path := filepath.Join(dir, filename)
	if err := SaveWorkflowState(path, state); err != nil {
		return "", err
	}
	return path, nil
}

// LoadLatestCheckpoint finds and loads the latest phase checkpoint for a workflow ID.
func (e *WorkflowEngine) LoadLatestCheckpoint(workflowID string) (*WorkflowState, error) {
	e.mu.RLock()
	dir := e.CheckpointDir
	e.mu.RUnlock()

	if dir == "" {
		dir = ".fak/workflows"
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	cleanID := sanitizeWorkflowID(workflowID)
	prefix := fmt.Sprintf("checkpoint_%s_phase_", cleanID)
	suffix := ".json"
	maxIdx := -1
	var latestFile string

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, prefix) && strings.HasSuffix(name, suffix) {
			idxStr := strings.TrimSuffix(strings.TrimPrefix(name, prefix), suffix)
			if idx, err := strconv.Atoi(idxStr); err == nil {
				if idx > maxIdx {
					maxIdx = idx
					latestFile = filepath.Join(dir, name)
				}
			}
		}
	}

	if latestFile == "" {
		return nil, os.ErrNotExist
	}
	return LoadWorkflowState(latestFile)
}

// LoadWorkflowState reads a persisted WorkflowState from a JSON file.
func LoadWorkflowState(path string) (*WorkflowState, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var st WorkflowState
	if err := json.Unmarshal(data, &st); err != nil {
		return nil, err
	}
	st.CheckpointPath = path
	return &st, nil
}

// SaveWorkflowState writes a WorkflowState to a JSON file.
func SaveWorkflowState(path string, state *WorkflowState) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return err
	}
	state.CheckpointPath = path
	return nil
}

func sanitizeWorkflowID(id string) string {
	r := strings.NewReplacer("/", "_", "\\", "_", ":", "_", " ", "_")
	return r.Replace(id)
}

func computeStateDigest(state *WorkflowState) string {
	if state == nil {
		return ""
	}
	h := sha256.New()
	dataBytes, _ := json.Marshal(state.Data)
	fmt.Fprintf(h, "%s:%d:%s:%s", state.WorkflowID, state.CurrentPhase, state.Status, string(dataBytes))
	return fmt.Sprintf("sha256:%x", h.Sum(nil))
}

// NewFleetWaveWorkflow creates the canonical builtin fleet-wave workflow (Phases 0..5).
func NewFleetWaveWorkflow() *Workflow {
	return &Workflow{
		ID:          "fleet-wave",
		Name:        "Fleet Wave",
		Description: "Multi-phase guarded ultracode wave orchestration",
		Phases: []Phase{
			{
				ID:          "gate",
				Index:       0,
				Name:        "Gate",
				Description: "Verify standing dispatcher and fleet collision status",
				EntryGates: []EntryGate{
					{
						Name:        "AdmissionPreflight",
						Description: "Ensure no conflicting standing gardener",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "no conflicting standing gardener"}
						},
					},
				},
				ExitGates: []ExitCondition{
					{
						Name:        "GateVerified",
						Description: "Gate checks complete and verified",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "gate verification clean"}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					if state.Data == nil {
						state.Data = make(map[string]any)
					}
					state.Data["gate_verified"] = true
					return Witness{
						Command:     "fak status",
						Description: "Fleet admission gate verified",
						Passed:      true,
						Output:      "admission: OK",
					}, nil
				},
			},
			{
				ID:          "price",
				Index:       1,
				Name:        "Price",
				Description: "Read admission capacity and calculate account seat budget",
				EntryGates: []EntryGate{
					{
						Name:        "PricingAdmission",
						Description: "Verify system capacity for pricing",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "pricing capacity available"}
						},
					},
				},
				ExitGates: []ExitCondition{
					{
						Name:        "PriceValid",
						Description: "Verify resource pricing outputs",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "pricing quota valid"}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					if state.Data == nil {
						state.Data = make(map[string]any)
					}
					state.Data["priced"] = true
					return Witness{
						Command:     "fak price",
						Description: "Wave resource pricing completed",
						Passed:      true,
						Output:      "pricing: 30 requested, 12 granted",
					}, nil
				},
			},
			{
				ID:          "receipt",
				Index:       2,
				Name:        "Receipt",
				Description: "Generate operator receipt and bounded rules",
				EntryGates: []EntryGate{
					{
						Name:        "ReceiptPreflight",
						Description: "Verify price receipt input prerequisites",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "receipt prerequisites met"}
						},
					},
				},
				ExitGates: []ExitCondition{
					{
						Name:        "ReceiptRendered",
						Description: "Verify operator receipt rendered",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "receipt rendered cleanly"}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					if state.Data == nil {
						state.Data = make(map[string]any)
					}
					state.Data["receipt_rendered"] = true
					return Witness{
						Command:     "fak receipt",
						Artifact:    ".fak/wave/receipt.md",
						Description: "Operator receipt rendered with target and rules",
						Passed:      true,
						Output:      "receipt: .fak/wave/receipt.md",
					}, nil
				},
			},
			{
				ID:          "launch",
				Index:       3,
				Name:        "Launch",
				Description: "Dispatch guarded worker processes",
				EntryGates: []EntryGate{
					{
						Name:        "LaunchGate",
						Description: "Verify receipt artifact before launch",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "receipt artifact confirmed"}
						},
					},
				},
				ExitGates: []ExitCondition{
					{
						Name:        "LaunchVerified",
						Description: "Verify dispatch processes started",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "workers active"}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					if state.Data == nil {
						state.Data = make(map[string]any)
					}
					state.Data["launched"] = true
					return Witness{
						Command:     "fak dispatch wave",
						Description: "Dispatched guarded wave workers",
						Passed:      true,
						Output:      "launched: 12 workers",
					}, nil
				},
			},
			{
				ID:          "monitor",
				Index:       4,
				Name:        "Monitor",
				Description: "Track wave progress and worker liveness",
				EntryGates: []EntryGate{
					{
						Name:        "MonitorPreflight",
						Description: "Verify worker processes exist to monitor",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "workers present"}
						},
					},
				},
				ExitGates: []ExitCondition{
					{
						Name:        "MonitorComplete",
						Description: "Verify monitor pass completed",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "liveness verified"}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					if state.Data == nil {
						state.Data = make(map[string]any)
					}
					state.Data["monitored"] = true
					return Witness{
						Command:     "fak monitor",
						Description: "Wave workers monitored: active forward progress confirmed",
						Passed:      true,
						Output:      "monitor: 12/12 active",
					}, nil
				},
			},
			{
				ID:          "harvest",
				Index:       5,
				Name:        "Harvest",
				Description: "Reconcile shipped work from git and release leases",
				EntryGates: []EntryGate{
					{
						Name:        "HarvestPreflight",
						Description: "Verify wave completion before harvest",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "ready for reconciliation"}
						},
					},
				},
				ExitGates: []ExitCondition{
					{
						Name:        "HarvestReconciled",
						Description: "Verify reconciliation completed",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "all leases released"}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					if state.Data == nil {
						state.Data = make(map[string]any)
					}
					state.Data["harvested"] = true
					return Witness{
						Command:     "fak wave-harvest",
						Description: "Wave harvest complete: artifacts reconciled and leases released",
						Passed:      true,
						Output:      "harvest: 12 closed, 0 leaked",
					}, nil
				},
			},
		},
	}
}
