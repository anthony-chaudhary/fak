package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// tasktools.go — kernel-mediated child task spawning and harness fan-out tools
// (task_spawn, task_wait, task_status, task_cancel) for the native agent harness (#11414, #11840).
//
// These tools provide parent-scoped task intent admission, capacity check, and stable
// task handles with BEST_EFFORT consistency: immediate in-process synthesized results,
// structural validation, and strict execution bounds.

const (
	ToolTaskSpawn  = "task_spawn"
	ToolTaskWait   = "task_wait"
	ToolTaskStatus = "task_status"
	ToolTaskCancel = "task_cancel"

	EngineTaskSpawn  = "agent.task_spawn"
	EngineTaskWait   = "agent.task_wait"
	EngineTaskStatus = "agent.task_status"
	EngineTaskCancel = "agent.task_cancel"

	RungNameTask = "tasktools"

	taskToolRank = 23

	TaskStatePending   = "pending"
	TaskStateRunning   = "running"
	TaskStateCompleted = "completed"
	TaskStateFailed    = "failed"
	TaskStateCancelled = "cancelled"
	TaskStateTimedOut  = "timed_out"

	DefaultMaxActiveTasks  = 16
	DefaultMaxBacklogTasks = 64
	DefaultMaxTotalTasks   = 100
)

// TaskItem represents a single managed child task within the harness.
type TaskItem struct {
	ID             string    `json:"id"`
	Prompt         string    `json:"prompt"`
	Description    string    `json:"description,omitempty"`
	SubagentType   string    `json:"subagent_type,omitempty"`
	ReadOnly       bool      `json:"read_only"`
	State          string    `json:"state"`
	CreatedAt      time.Time `json:"created_at"`
	CompletedAt    time.Time `json:"completed_at,omitempty"`
	Result         any       `json:"result,omitempty"`
	Error          string    `json:"error,omitempty"`
	IdempotencyKey string    `json:"idempotency_key,omitempty"`
}

// TaskSpawnRequest defines parameters for task_spawn.
type TaskSpawnRequest struct {
	Prompt         string `json:"prompt"`
	Description    string `json:"description,omitempty"`
	TaskID         string `json:"task_id,omitempty"`
	SubagentType   string `json:"subagent_type,omitempty"`
	ReadOnly       bool   `json:"read_only,omitempty"`
	IdempotencyKey string `json:"idempotency_key,omitempty"`
}

// TaskSpawnReceipt is the structured confirmation returned by task_spawn.
type TaskSpawnReceipt struct {
	Status       string    `json:"status"` // "accepted" | "queued" | "running" | "error"
	TaskID       string    `json:"task_id,omitempty"`
	SubagentType string    `json:"subagent_type,omitempty"`
	ReadOnly     bool      `json:"read_only"`
	Idempotent   bool      `json:"idempotent,omitempty"`
	CreatedAt    time.Time `json:"created_at,omitempty"`
	Error        string    `json:"error,omitempty"`
}

// TaskWaitRequest defines parameters for task_wait.
type TaskWaitRequest struct {
	TaskIDs   []string `json:"task_ids,omitempty"`
	TaskID    string   `json:"task_id,omitempty"`
	TimeoutMs int      `json:"timeout_ms,omitempty"`
	WaitAll   *bool    `json:"wait_all,omitempty"`
}

// TaskWaitReceipt is the structured payload returned by task_wait.
type TaskWaitReceipt struct {
	Status    string               `json:"status"` // "completed" | "timed_out" | "error"
	Tasks     map[string]*TaskItem `json:"tasks,omitempty"`
	Completed int                  `json:"completed"`
	Running   int                  `json:"running"`
	Failed    int                  `json:"failed"`
	Cancelled int                  `json:"cancelled"`
	TimedOut  int                  `json:"timed_out"`
	Error     string               `json:"error,omitempty"`
}

// TaskStatusRequest defines parameters for task_status.
type TaskStatusRequest struct {
	TaskID string `json:"task_id,omitempty"`
	Limit  int    `json:"limit,omitempty"`
}

// TaskStatusReceipt is the structured payload returned by task_status.
type TaskStatusReceipt struct {
	Tasks     []TaskItem `json:"tasks"`
	Total     int        `json:"total"`
	Active    int        `json:"active"`
	Pending   int        `json:"pending"`
	Completed int        `json:"completed"`
	Failed    int        `json:"failed"`
	Cancelled int        `json:"cancelled"`
	Error     string     `json:"error,omitempty"`
}

// TaskCancelRequest defines parameters for task_cancel.
type TaskCancelRequest struct {
	TaskID string `json:"task_id"`
	Reason string `json:"reason,omitempty"`
}

// TaskCancelReceipt is the structured payload returned by task_cancel.
type TaskCancelReceipt struct {
	Status    string `json:"status"` // "cancelled" | "not_found" | "error"
	TaskID    string `json:"task_id"`
	Cancelled bool   `json:"cancelled"`
	Error     string `json:"error,omitempty"`
}

// TaskState holds the thread-safe active task items and queue state for a session.
type TaskState struct {
	mu          sync.RWMutex
	tasks       map[string]*TaskItem
	order       []string
	idempotency map[string]string // idempotencyKey -> taskID
	maxActive   int
	maxBacklog  int
	taskSeq     int64
	doneChans   map[string]chan struct{}
	closeOnce   map[string]*sync.Once
}

// NewTaskState returns an empty initialized TaskState.
func NewTaskState() *TaskState {
	return &TaskState{
		tasks:       make(map[string]*TaskItem),
		order:       make([]string, 0),
		idempotency: make(map[string]string),
		maxActive:   DefaultMaxActiveTasks,
		maxBacklog:  DefaultMaxBacklogTasks,
		doneChans:   make(map[string]chan struct{}),
		closeOnce:   make(map[string]*sync.Once),
	}
}

// Spawn validates and creates a new child task intent with capacity enforcement.
func (s *TaskState) Spawn(req TaskSpawnRequest) (TaskSpawnReceipt, error) {
	if s == nil {
		return TaskSpawnReceipt{Status: "error", Error: "task state is nil"}, fmt.Errorf("task state is nil")
	}

	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		err := fmt.Errorf("task prompt is required and cannot be empty")
		return TaskSpawnReceipt{Status: "error", Error: err.Error()}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Scoped idempotency check
	if req.IdempotencyKey != "" {
		if existingID, ok := s.idempotency[req.IdempotencyKey]; ok {
			if existing, found := s.tasks[existingID]; found {
				return TaskSpawnReceipt{
					Status:       "accepted",
					TaskID:       existing.ID,
					SubagentType: existing.SubagentType,
					ReadOnly:     existing.ReadOnly,
					Idempotent:   true,
					CreatedAt:    existing.CreatedAt,
				}, nil
			}
		}
	}

	// Check total capacity
	if len(s.tasks) >= DefaultMaxTotalTasks {
		err := fmt.Errorf("task capacity exceeded: total tasks (%d) reached maximum (%d)", len(s.tasks), DefaultMaxTotalTasks)
		return TaskSpawnReceipt{Status: "error", Error: err.Error()}, err
	}

	// Check active + backlog capacity
	activeCount := 0
	for _, t := range s.tasks {
		if t.State == TaskStateRunning || t.State == TaskStatePending {
			activeCount++
		}
	}
	if activeCount >= s.maxActive+s.maxBacklog {
		err := fmt.Errorf("task admission rejected: active + backlog capacity reached (%d)", s.maxActive+s.maxBacklog)
		return TaskSpawnReceipt{Status: "error", Error: err.Error()}, err
	}

	// Task ID allocation
	taskID := strings.TrimSpace(req.TaskID)
	if taskID != "" {
		if _, exists := s.tasks[taskID]; exists {
			err := fmt.Errorf("task %q already exists", taskID)
			return TaskSpawnReceipt{Status: "error", Error: err.Error()}, err
		}
	} else {
		s.taskSeq++
		taskID = fmt.Sprintf("task-%d", s.taskSeq)
	}

	subagentType := strings.TrimSpace(req.SubagentType)
	if subagentType == "" {
		subagentType = "worker"
	}

	state := TaskStateRunning
	if activeCount >= s.maxActive {
		state = TaskStatePending
	}

	item := &TaskItem{
		ID:             taskID,
		Prompt:         prompt,
		Description:    strings.TrimSpace(req.Description),
		SubagentType:   subagentType,
		ReadOnly:       req.ReadOnly,
		State:          state,
		CreatedAt:      time.Now().UTC(),
		IdempotencyKey: req.IdempotencyKey,
	}

	s.tasks[taskID] = item
	s.order = append(s.order, taskID)
	if req.IdempotencyKey != "" {
		s.idempotency[req.IdempotencyKey] = taskID
	}
	s.doneChans[taskID] = make(chan struct{})
	s.closeOnce[taskID] = &sync.Once{}

	return TaskSpawnReceipt{
		Status:       "accepted",
		TaskID:       taskID,
		SubagentType: subagentType,
		ReadOnly:     req.ReadOnly,
		CreatedAt:    item.CreatedAt,
	}, nil
}

// Wait waits for target child tasks to reach a terminal state or timeout.
func (s *TaskState) Wait(ctx context.Context, req TaskWaitRequest) (TaskWaitReceipt, error) {
	if s == nil {
		return TaskWaitReceipt{Status: "error", Error: "task state is nil"}, fmt.Errorf("task state is nil")
	}

	s.mu.RLock()
	var targets []string
	if req.TaskID != "" {
		targets = append(targets, req.TaskID)
	}
	for _, id := range req.TaskIDs {
		trimmed := strings.TrimSpace(id)
		if trimmed != "" {
			already := false
			for _, existing := range targets {
				if existing == trimmed {
					already = true
					break
				}
			}
			if !already {
				targets = append(targets, trimmed)
			}
		}
	}

	// If no targets provided, wait on all known tasks
	if len(targets) == 0 {
		for _, id := range s.order {
			targets = append(targets, id)
		}
	}

	waitAll := true
	if req.WaitAll != nil {
		waitAll = *req.WaitAll
	}

	type waitTarget struct {
		id   string
		done <-chan struct{}
	}
	var pendingTargets []waitTarget

	for _, id := range targets {
		item, exists := s.tasks[id]
		if !exists {
			s.mu.RUnlock()
			err := fmt.Errorf("task %q not found", id)
			return TaskWaitReceipt{Status: "error", Error: err.Error()}, err
		}
		if item.State == TaskStateRunning || item.State == TaskStatePending {
			if ch, ok := s.doneChans[id]; ok {
				pendingTargets = append(pendingTargets, waitTarget{id: id, done: ch})
			}
		}
	}
	s.mu.RUnlock()

	timedOut := false
	if len(pendingTargets) > 0 {
		var timer *time.Timer
		var timerC <-chan time.Time
		if req.TimeoutMs > 0 {
			timer = time.NewTimer(time.Duration(req.TimeoutMs) * time.Millisecond)
			defer timer.Stop()
			timerC = timer.C
		}

		if waitAll {
			for _, pt := range pendingTargets {
				select {
				case <-pt.done:
				case <-timerC:
					timedOut = true
					break
				case <-ctx.Done():
					return s.summarizeWait(targets, "cancelled")
				}
				if timedOut {
					break
				}
			}
		} else {
			wakeCh := make(chan struct{}, len(pendingTargets))
			stopCh := make(chan struct{})
			defer close(stopCh)

			for _, pt := range pendingTargets {
				go func(ch <-chan struct{}) {
					select {
					case <-ch:
						select {
						case wakeCh <- struct{}{}:
						default:
						}
					case <-stopCh:
					}
				}(pt.done)
			}

			select {
			case <-wakeCh:
			case <-timerC:
				timedOut = true
			case <-ctx.Done():
				return s.summarizeWait(targets, "cancelled")
			}
		}
	}

	status := "completed"
	if timedOut {
		status = "timed_out"
	}
	return s.summarizeWait(targets, status)
}

func (s *TaskState) summarizeWait(targets []string, status string) (TaskWaitReceipt, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	receipt := TaskWaitReceipt{
		Status: status,
		Tasks:  make(map[string]*TaskItem, len(targets)),
	}

	for _, id := range targets {
		if item, exists := s.tasks[id]; exists {
			cp := *item
			receipt.Tasks[id] = &cp
			switch item.State {
			case TaskStateCompleted:
				receipt.Completed++
			case TaskStateRunning, TaskStatePending:
				receipt.Running++
			case TaskStateFailed:
				receipt.Failed++
			case TaskStateCancelled:
				receipt.Cancelled++
			case TaskStateTimedOut:
				receipt.TimedOut++
			}
		}
	}
	return receipt, nil
}

// Status returns lifecycle details for a specific task or all known tasks.
func (s *TaskState) Status(req TaskStatusRequest) (TaskStatusReceipt, error) {
	if s == nil {
		return TaskStatusReceipt{Error: "task state is nil"}, fmt.Errorf("task state is nil")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	receipt := TaskStatusReceipt{
		Tasks: make([]TaskItem, 0),
	}

	if req.TaskID != "" {
		item, exists := s.tasks[req.TaskID]
		if !exists {
			err := fmt.Errorf("task %q not found", req.TaskID)
			receipt.Error = err.Error()
			return receipt, err
		}
		receipt.Tasks = append(receipt.Tasks, *item)
		receipt.Total = 1
		switch item.State {
		case TaskStateRunning:
			receipt.Active = 1
		case TaskStatePending:
			receipt.Pending = 1
		case TaskStateCompleted:
			receipt.Completed = 1
		case TaskStateFailed:
			receipt.Failed = 1
		case TaskStateCancelled:
			receipt.Cancelled = 1
		}
		return receipt, nil
	}

	for _, id := range s.order {
		item := s.tasks[id]
		if req.Limit <= 0 || len(receipt.Tasks) < req.Limit {
			receipt.Tasks = append(receipt.Tasks, *item)
		}
		receipt.Total++
		switch item.State {
		case TaskStateRunning:
			receipt.Active++
		case TaskStatePending:
			receipt.Pending++
		case TaskStateCompleted:
			receipt.Completed++
		case TaskStateFailed:
			receipt.Failed++
		case TaskStateCancelled:
			receipt.Cancelled++
		}
	}

	return receipt, nil
}

// Cancel cancels a pending or running child task.
func (s *TaskState) Cancel(req TaskCancelRequest) (TaskCancelReceipt, error) {
	if s == nil {
		return TaskCancelReceipt{Status: "error", Error: "task state is nil"}, fmt.Errorf("task state is nil")
	}

	taskID := strings.TrimSpace(req.TaskID)
	if taskID == "" {
		err := fmt.Errorf("task_id is required for cancel")
		return TaskCancelReceipt{Status: "error", Error: err.Error()}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	item, exists := s.tasks[taskID]
	if !exists {
		err := fmt.Errorf("task %q not found", taskID)
		return TaskCancelReceipt{Status: "not_found", TaskID: taskID, Error: err.Error()}, err
	}

	if item.State == TaskStateCompleted || item.State == TaskStateFailed || item.State == TaskStateCancelled || item.State == TaskStateTimedOut {
		return TaskCancelReceipt{
			Status:    item.State,
			TaskID:    taskID,
			Cancelled: false,
		}, nil
	}

	item.State = TaskStateCancelled
	item.CompletedAt = time.Now().UTC()
	if req.Reason != "" {
		item.Error = req.Reason
	} else {
		item.Error = "task cancelled by operator"
	}

	if once, ok := s.closeOnce[taskID]; ok {
		once.Do(func() {
			if ch, ok := s.doneChans[taskID]; ok {
				close(ch)
			}
		})
	}

	return TaskCancelReceipt{
		Status:    "cancelled",
		TaskID:    taskID,
		Cancelled: true,
	}, nil
}

// CompleteTask transitions a task to completed or failed state with optional result payload.
func (s *TaskState) CompleteTask(taskID string, result any, err error) error {
	if s == nil {
		return fmt.Errorf("task state is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	item, exists := s.tasks[taskID]
	if !exists {
		return fmt.Errorf("task %q not found", taskID)
	}

	if err != nil {
		item.State = TaskStateFailed
		item.Error = err.Error()
	} else {
		item.State = TaskStateCompleted
		item.Result = result
	}
	item.CompletedAt = time.Now().UTC()

	if once, ok := s.closeOnce[taskID]; ok {
		once.Do(func() {
			if ch, ok := s.doneChans[taskID]; ok {
				close(ch)
			}
		})
	}
	return nil
}

// GetTasks returns a copy of all current tasks.
func (s *TaskState) GetTasks() []TaskItem {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]TaskItem, 0, len(s.order))
	for _, id := range s.order {
		if item, ok := s.tasks[id]; ok {
			out = append(out, *item)
		}
	}
	return out
}

// taskEngine implements abi.Engine for task_spawn, task_wait, task_status, and task_cancel.
type taskEngine struct {
	state *TaskState
	mu    sync.RWMutex
}

func (e *taskEngine) Caps() []abi.Capability { return nil }
func (e *taskEngine) WeightBearing() bool    { return false }

func (e *taskEngine) getState() *TaskState {
	if e != nil && e.state != nil {
		return e.state
	}
	return armedTaskTools.Load()
}

func (e *taskEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body, _ := decodeCallArgs(ctx, c.Args)
	st := e.getState()
	if st == nil {
		errResp, _ := json.Marshal(map[string]any{"status": "error", "error": "task tools are unarmed"})
		return engineResult(ctx, c, body, errResp, true, RungNameTask), nil
	}

	switch c.Tool {
	case ToolTaskSpawn:
		var req TaskSpawnRequest
		if len(body) > 0 {
			if err := json.Unmarshal(body, &req); err != nil {
				errResp, _ := json.Marshal(TaskSpawnReceipt{Status: "error", Error: fmt.Sprintf("invalid arguments JSON: %v", err)})
				return engineResult(ctx, c, body, errResp, true, EngineTaskSpawn), nil
			}
		}
		receipt, err := st.Spawn(req)
		respBytes, _ := json.Marshal(receipt)
		return engineResult(ctx, c, body, respBytes, err != nil, EngineTaskSpawn), nil

	case ToolTaskWait:
		var req TaskWaitRequest
		if len(body) > 0 {
			if err := json.Unmarshal(body, &req); err != nil {
				errResp, _ := json.Marshal(TaskWaitReceipt{Status: "error", Error: fmt.Sprintf("invalid arguments JSON: %v", err)})
				return engineResult(ctx, c, body, errResp, true, EngineTaskWait), nil
			}
		}
		receipt, err := st.Wait(ctx, req)
		respBytes, _ := json.Marshal(receipt)
		return engineResult(ctx, c, body, respBytes, err != nil, EngineTaskWait), nil

	case ToolTaskStatus:
		var req TaskStatusRequest
		if len(body) > 0 {
			if err := json.Unmarshal(body, &req); err != nil {
				errResp, _ := json.Marshal(TaskStatusReceipt{Error: fmt.Sprintf("invalid arguments JSON: %v", err)})
				return engineResult(ctx, c, body, errResp, true, EngineTaskStatus), nil
			}
		}
		receipt, err := st.Status(req)
		respBytes, _ := json.Marshal(receipt)
		return engineResult(ctx, c, body, respBytes, err != nil, EngineTaskStatus), nil

	case ToolTaskCancel:
		var req TaskCancelRequest
		if len(body) > 0 {
			if err := json.Unmarshal(body, &req); err != nil {
				errResp, _ := json.Marshal(TaskCancelReceipt{Status: "error", Error: fmt.Sprintf("invalid arguments JSON: %v", err)})
				return engineResult(ctx, c, body, errResp, true, EngineTaskCancel), nil
			}
		}
		receipt, err := st.Cancel(req)
		respBytes, _ := json.Marshal(receipt)
		return engineResult(ctx, c, body, respBytes, err != nil, EngineTaskCancel), nil

	default:
		return engineResult(ctx, c, body, []byte(fmt.Sprintf(`{"error":"unknown tool %q"}`, c.Tool)), true, RungNameTask), nil
	}
}

// taskToolGate adjudicates task tools (spawn, wait, status, cancel), pinning the engine.
type taskToolGate struct{}

func (taskToolGate) Caps() []abi.Capability { return nil }

func (taskToolGate) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil || armedTaskTools.Load() == nil {
		return abi.Verdict{Kind: abi.VerdictDefer, By: RungNameTask}
	}
	switch c.Tool {
	case ToolTaskSpawn:
		c.Engine = EngineTaskSpawn
		return abi.Verdict{Kind: abi.VerdictAllow, By: RungNameTask}
	case ToolTaskWait:
		c.Engine = EngineTaskWait
		return abi.Verdict{Kind: abi.VerdictAllow, By: RungNameTask}
	case ToolTaskStatus:
		c.Engine = EngineTaskStatus
		return abi.Verdict{Kind: abi.VerdictAllow, By: RungNameTask}
	case ToolTaskCancel:
		c.Engine = EngineTaskCancel
		return abi.Verdict{Kind: abi.VerdictAllow, By: RungNameTask}
	default:
		return abi.Verdict{Kind: abi.VerdictDefer, By: RungNameTask}
	}
}

var (
	armedTaskTools   atomic.Pointer[TaskState]
	taskGateOnce     sync.Once
	taskEnginesOnce  sync.Once
	activeTaskEngine *taskEngine
)

// ArmTaskTools initializes the native child task tools, registers their engines,
// installs the adjudicator gate once, and returns the planner-facing ToolDef declarations.
func ArmTaskTools() ([]ToolDef, error) {
	st := NewTaskState()
	armedTaskTools.Store(st)

	taskEnginesOnce.Do(func() {
		activeTaskEngine = &taskEngine{}
		abi.RegisterEngine(EngineTaskSpawn, activeTaskEngine)
		abi.RegisterEngine(EngineTaskWait, activeTaskEngine)
		abi.RegisterEngine(EngineTaskStatus, activeTaskEngine)
		abi.RegisterEngine(EngineTaskCancel, activeTaskEngine)
	})

	taskGateOnce.Do(func() {
		abi.RegisterAdjudicator(taskToolRank, taskToolGate{})
	})

	return TaskToolCatalog(), nil
}

// DisarmTaskTools unarms the task tools, restoring the inactive state.
func DisarmTaskTools() {
	armedTaskTools.Store(nil)
}

// GetActiveTaskState returns the active TaskState, or nil if unarmed.
func GetActiveTaskState() *TaskState {
	return armedTaskTools.Load()
}

// TaskToolCatalog renders the task tools as loop ToolDefs. Empty when unarmed.
func TaskToolCatalog() []ToolDef {
	if armedTaskTools.Load() == nil {
		return nil
	}
	return taskToolDefs()
}

func taskToolDefs() []ToolDef {
	return []ToolDef{
		{
			Type: "function",
			Function: ToolDefFunction{
				Name:        ToolTaskSpawn,
				Description: "Spawn a child agent task with prompt-scoped intent admission, capacity check, and stable handle return.",
				Parameters: rawSchema(`{
  "type": "object",
  "properties": {
    "prompt": {
      "type": "string",
      "description": "Task prompt, instructions, or goal for the child agent"
    },
    "description": {
      "type": "string",
      "description": "Short summary or label for the child task"
    },
    "task_id": {
      "type": "string",
      "description": "Optional unique identifier for the task; generated if omitted"
    },
    "subagent_type": {
      "type": "string",
      "description": "Optional specialized subagent profile or role (e.g. worker, researcher, explore)"
    },
    "read_only": {
      "type": "boolean",
      "description": "Whether the task is read-only / effect-safe"
    },
    "idempotency_key": {
      "type": "string",
      "description": "Optional idempotency key to prevent duplicate spawn submissions"
    }
  },
  "required": ["prompt"]
}`),
			},
		},
		{
			Type: "function",
			Function: ToolDefFunction{
				Name:        ToolTaskWait,
				Description: "Wait for child tasks to reach a terminal state (completed, failed, or timed out) or until timeout.",
				Parameters: rawSchema(`{
  "type": "object",
  "properties": {
    "task_ids": {
      "type": "array",
      "items": {"type": "string"},
      "description": "List of task IDs to wait on. If omitted, waits on all known active child tasks."
    },
    "task_id": {
      "type": "string",
      "description": "Optional single task ID to wait on"
    },
    "timeout_ms": {
      "type": "integer",
      "description": "Optional timeout in milliseconds"
    },
    "wait_all": {
      "type": "boolean",
      "description": "If true, waits for all specified tasks; if false, wakes on the first ready/terminal task"
    }
  }
}`),
			},
		},
		{
			Type: "function",
			Function: ToolDefFunction{
				Name:        ToolTaskStatus,
				Description: "Inspect the lifecycle status, execution state, and progress of child tasks without blocking.",
				Parameters: rawSchema(`{
  "type": "object",
  "properties": {
    "task_id": {
      "type": "string",
      "description": "Optional task ID to inspect. If omitted, returns status across active and completed tasks."
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of tasks to return"
    }
  }
}`),
			},
		},
		{
			Type: "function",
			Function: ToolDefFunction{
				Name:        ToolTaskCancel,
				Description: "Cancel or abort a pending or running child task by ID.",
				Parameters: rawSchema(`{
  "type": "object",
  "properties": {
    "task_id": {
      "type": "string",
      "description": "The unique task ID to cancel"
    },
    "reason": {
      "type": "string",
      "description": "Optional cancellation reason"
    }
  },
  "required": ["task_id"]
}`),
			},
		},
	}
}

// taskToolMeta returns the vDSO / consistency scope metadata for task tools.
func taskToolMeta(tool string) (map[string]string, bool) {
	if armedTaskTools.Load() == nil {
		return nil, false
	}
	switch tool {
	case ToolTaskSpawn:
		return map[string]string{
			"readOnlyHint":   "false",
			"idempotentHint": "false",
			"consistency":    "BEST_EFFORT",
		}, true
	case ToolTaskWait:
		return map[string]string{
			"readOnlyHint":   "true",
			"idempotentHint": "false",
			"consistency":    "BEST_EFFORT",
		}, true
	case ToolTaskStatus:
		return map[string]string{
			"readOnlyHint":   "true",
			"idempotentHint": "true",
			"consistency":    "BEST_EFFORT",
		}, true
	case ToolTaskCancel:
		return map[string]string{
			"readOnlyHint":   "false",
			"idempotentHint": "true",
			"destructive":    "true",
			"consistency":    "BEST_EFFORT",
		}, true
	default:
		return nil, false
	}
}

// taskToolAllow returns the tool names admitted when task tools are armed.
func taskToolAllow() []string {
	if armedTaskTools.Load() == nil {
		return nil
	}
	return []string{ToolTaskSpawn, ToolTaskWait, ToolTaskStatus, ToolTaskCancel}
}
