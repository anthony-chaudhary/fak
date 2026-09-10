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

	DefaultTaskWaitTimeout = 2 * time.Minute
	MaxTaskWaitTimeout     = 10 * time.Minute
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

// ChildTaskRunRequest is the immutable work envelope handed to an admitted child.
type ChildTaskRunRequest struct {
	TaskID       string
	Prompt       string
	Description  string
	SubagentType string
	ReadOnly     bool
}

// ChildTaskRunner executes one admitted child. Implementations must stop when ctx
// is cancelled and return only after all child effects have ceased.
type ChildTaskRunner func(ctx context.Context, req ChildTaskRunRequest) (any, error)

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
	runner      ChildTaskRunner
	rootCtx     context.Context
	rootCancel  context.CancelFunc
	cancels     map[string]context.CancelFunc
	executions  sync.WaitGroup
	closed      bool
}

// NewTaskState returns an empty initialized TaskState.
func NewTaskState() *TaskState {
	return newTaskState(nil)
}

// NewTaskStateWithRunner returns an initialized TaskState which executes
// admitted work through runner. A nil runner preserves intent-only behavior.
func NewTaskStateWithRunner(runner ChildTaskRunner) *TaskState {
	return newTaskState(runner)
}

func newTaskState(runner ChildTaskRunner) *TaskState {
	return newTaskStateWithContext(context.Background(), runner)
}

func newTaskStateWithContext(parent context.Context, runner ChildTaskRunner) *TaskState {
	if parent == nil {
		parent = context.Background()
	}
	rootCtx, rootCancel := context.WithCancel(parent)
	return &TaskState{
		tasks:       make(map[string]*TaskItem),
		order:       make([]string, 0),
		idempotency: make(map[string]string),
		maxActive:   DefaultMaxActiveTasks,
		maxBacklog:  DefaultMaxBacklogTasks,
		doneChans:   make(map[string]chan struct{}),
		closeOnce:   make(map[string]*sync.Once),
		runner:      runner,
		rootCtx:     rootCtx,
		rootCancel:  rootCancel,
		cancels:     make(map[string]context.CancelFunc),
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
	if s.closed {
		s.mu.Unlock()
		err := fmt.Errorf("task state is closed")
		return TaskSpawnReceipt{Status: "error", Error: err.Error()}, err
	}

	// Scoped idempotency check
	if req.IdempotencyKey != "" {
		if existingID, ok := s.idempotency[req.IdempotencyKey]; ok {
			if existing, found := s.tasks[existingID]; found {
				receipt := TaskSpawnReceipt{
					Status:       "accepted",
					TaskID:       existing.ID,
					SubagentType: existing.SubagentType,
					ReadOnly:     existing.ReadOnly,
					Idempotent:   true,
					CreatedAt:    existing.CreatedAt,
				}
				s.mu.Unlock()
				return receipt, nil
			}
		}
	}

	// Check total capacity
	if len(s.tasks) >= DefaultMaxTotalTasks {
		err := fmt.Errorf("task capacity exceeded: total tasks (%d) reached maximum (%d)", len(s.tasks), DefaultMaxTotalTasks)
		s.mu.Unlock()
		return TaskSpawnReceipt{Status: "error", Error: err.Error()}, err
	}

	// Check active + backlog capacity
	runningCount := s.activeReservationsLocked()
	pendingCount := 0
	for _, t := range s.tasks {
		switch t.State {
		case TaskStatePending:
			pendingCount++
		}
	}
	if runningCount+pendingCount >= s.maxActive+s.maxBacklog {
		err := fmt.Errorf("task admission rejected: active + backlog capacity reached (%d)", s.maxActive+s.maxBacklog)
		s.mu.Unlock()
		return TaskSpawnReceipt{Status: "error", Error: err.Error()}, err
	}

	// Task ID allocation
	taskID := strings.TrimSpace(req.TaskID)
	if taskID != "" {
		if _, exists := s.tasks[taskID]; exists {
			err := fmt.Errorf("task %q already exists", taskID)
			s.mu.Unlock()
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
	if runningCount >= s.maxActive {
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

	receipt := TaskSpawnReceipt{
		Status:       "accepted",
		TaskID:       taskID,
		SubagentType: subagentType,
		ReadOnly:     req.ReadOnly,
		CreatedAt:    item.CreatedAt,
	}
	shouldStart := state == TaskStateRunning && s.runner != nil
	s.mu.Unlock()
	if shouldStart {
		s.startTask(taskID)
	}
	return receipt, nil
}

func (s *TaskState) startTask(taskID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	item, ok := s.tasks[taskID]
	if !ok || s.closed || s.runner == nil || item.State != TaskStateRunning {
		s.mu.Unlock()
		return
	}
	if _, started := s.cancels[taskID]; started {
		s.mu.Unlock()
		return
	}
	ctx, cancel := context.WithCancel(s.rootCtx)
	s.cancels[taskID] = cancel
	runner := s.runner
	req := ChildTaskRunRequest{
		TaskID:       item.ID,
		Prompt:       item.Prompt,
		Description:  item.Description,
		SubagentType: item.SubagentType,
		ReadOnly:     item.ReadOnly,
	}
	s.executions.Add(1)
	s.mu.Unlock()

	go func() {
		defer s.executions.Done()
		result, err := runner(ctx, req)
		_ = s.finishTask(taskID, result, err)
	}()
}

func (s *TaskState) finishTask(taskID string, result any, runErr error) error {
	if s == nil {
		return fmt.Errorf("task state is nil")
	}
	s.mu.Lock()
	item, exists := s.tasks[taskID]
	if !exists {
		s.mu.Unlock()
		return fmt.Errorf("task %q not found", taskID)
	}
	if isTerminalTaskState(item.State) {
		_, wasRunning := s.cancels[taskID]
		delete(s.cancels, taskID)
		var promoted []string
		if wasRunning {
			promoted = s.promotePendingLocked()
		}
		s.mu.Unlock()
		for _, id := range promoted {
			s.startTask(id)
		}
		return nil
	}
	if cancel := s.cancels[taskID]; cancel != nil {
		cancel()
		delete(s.cancels, taskID)
	}
	if runErr != nil {
		item.State = TaskStateFailed
		item.Error = runErr.Error()
	} else {
		item.State = TaskStateCompleted
		item.Result = result
	}
	item.CompletedAt = time.Now().UTC()
	s.closeTaskLocked(taskID)
	promoted := s.promotePendingLocked()
	s.mu.Unlock()
	for _, id := range promoted {
		s.startTask(id)
	}
	return nil
}

func isTerminalTaskState(state string) bool {
	switch state {
	case TaskStateCompleted, TaskStateFailed, TaskStateCancelled, TaskStateTimedOut:
		return true
	default:
		return false
	}
}

func (s *TaskState) closeTaskLocked(taskID string) {
	if once, ok := s.closeOnce[taskID]; ok {
		once.Do(func() {
			if ch, found := s.doneChans[taskID]; found {
				close(ch)
			}
		})
	}
}

func (s *TaskState) activeReservationsLocked() int {
	active := len(s.cancels)
	for id, item := range s.tasks {
		if item.State != TaskStateRunning {
			continue
		}
		if _, started := s.cancels[id]; !started {
			active++
		}
	}
	return active
}

func (s *TaskState) promotePendingLocked() []string {
	if s.closed || s.runner == nil {
		return nil
	}
	running := s.activeReservationsLocked()
	if running >= s.maxActive {
		return nil
	}
	promoted := make([]string, 0, s.maxActive-running)
	for _, id := range s.order {
		item := s.tasks[id]
		if item.State != TaskStatePending {
			continue
		}
		item.State = TaskStateRunning
		promoted = append(promoted, id)
		running++
		if running >= s.maxActive {
			break
		}
	}
	return promoted
}

// Close cancels all admitted children and prevents further spawns.
func (s *TaskState) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	s.rootCancel()
	for id, item := range s.tasks {
		if !isTerminalTaskState(item.State) {
			item.State = TaskStateCancelled
			item.Error = "task tools disarmed"
			item.CompletedAt = time.Now().UTC()
			s.closeTaskLocked(id)
		}
	}
	s.cancels = make(map[string]context.CancelFunc)
	s.mu.Unlock()
}

// closeAndWait is the run-owned cleanup boundary. Close remains non-blocking for
// legacy callers; a scoped RunArm additionally joins its own admitted runners so
// no child effects survive the parent invocation.
func (s *TaskState) closeAndWait() {
	if s == nil {
		return
	}
	s.Close()
	s.executions.Wait()
}

// Wait waits for target child tasks to reach a terminal state or timeout.
func (s *TaskState) Wait(ctx context.Context, req TaskWaitRequest) (TaskWaitReceipt, error) {
	if s == nil {
		return TaskWaitReceipt{Status: "error", Error: "task state is nil"}, fmt.Errorf("task state is nil")
	}

	s.mu.RLock()
	hasRunner := s.runner != nil
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

	waitTimeout := time.Duration(req.TimeoutMs) * time.Millisecond
	if hasRunner && waitTimeout <= 0 {
		waitTimeout = DefaultTaskWaitTimeout
	}
	if waitTimeout > MaxTaskWaitTimeout {
		waitTimeout = MaxTaskWaitTimeout
	}
	timedOut := false
	if len(pendingTargets) > 0 {
		var timer *time.Timer
		var timerC <-chan time.Time
		if waitTimeout > 0 {
			timer = time.NewTimer(waitTimeout)
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

	item, exists := s.tasks[taskID]
	if !exists {
		s.mu.Unlock()
		err := fmt.Errorf("task %q not found", taskID)
		return TaskCancelReceipt{Status: "not_found", TaskID: taskID, Error: err.Error()}, err
	}

	if isTerminalTaskState(item.State) {
		receipt := TaskCancelReceipt{
			Status:    item.State,
			TaskID:    taskID,
			Cancelled: false,
		}
		s.mu.Unlock()
		return receipt, nil
	}

	wasRunning := item.State == TaskStateRunning
	item.State = TaskStateCancelled
	item.CompletedAt = time.Now().UTC()
	if req.Reason != "" {
		item.Error = req.Reason
	} else {
		item.Error = "task cancelled by operator"
	}

	cancel := s.cancels[taskID]
	if !wasRunning {
		delete(s.cancels, taskID)
	}
	s.closeTaskLocked(taskID)
	receipt := TaskCancelReceipt{
		Status:    "cancelled",
		TaskID:    taskID,
		Cancelled: true,
	}
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	return receipt, nil
}

// CompleteTask transitions a task to completed or failed state with optional result payload.
func (s *TaskState) CompleteTask(taskID string, result any, err error) error {
	return s.finishTask(taskID, result, err)
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

func (e *taskEngine) getState(ctx context.Context) *TaskState {
	if st := taskStateFromContext(ctx); st != nil {
		return st
	}
	if e != nil && e.state != nil {
		return e.state
	}
	return armedTaskTools.Load()
}

func (e *taskEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body, _ := decodeCallArgs(ctx, c.Args)
	st := e.getState(ctx)
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

func (taskToolGate) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil || (taskStateFromContext(ctx) == nil && armedTaskTools.Load() == nil) {
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

type taskStateContextKey struct{}

type childTaskRunConfig struct {
	maxActive  int
	maxBacklog int
	runner     ChildTaskRunner
}

// WithChildTaskRunner gives one RunArm invocation its own bounded child task
// namespace. The state is created when the arm starts, inherited by kernel
// dispatch through context, and cancelled and joined before the arm returns.
func WithChildTaskRunner(maxActive, maxBacklog int, runner ChildTaskRunner) RunOption {
	return func(c *runConfig) {
		c.childTasks = &childTaskRunConfig{
			maxActive:  maxActive,
			maxBacklog: maxBacklog,
			runner:     runner,
		}
		c.taskTools = true
	}
}

func (c *runConfig) bindChildTaskState(ctx context.Context) context.Context {
	if c == nil || c.childTasks == nil {
		return ctx
	}
	st := newTaskStateWithContext(ctx, c.childTasks.runner)
	st.SetLimits(c.childTasks.maxActive, c.childTasks.maxBacklog)
	c.taskState = st
	registerTaskToolRuntime()
	return withTaskState(ctx, st)
}

func (c *runConfig) closeChildTaskState() {
	if c == nil || c.taskState == nil {
		return
	}
	c.taskState.closeAndWait()
	c.taskState = nil
}

func withTaskState(ctx context.Context, state *TaskState) context.Context {
	if state == nil {
		return ctx
	}
	return context.WithValue(ctx, taskStateContextKey{}, state)
}

func taskStateFromContext(ctx context.Context) *TaskState {
	if ctx == nil {
		return nil
	}
	state, _ := ctx.Value(taskStateContextKey{}).(*TaskState)
	return state
}

func registerTaskToolRuntime() {
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
}

// SetLimits updates the max active and backlog capacity limits.
func (s *TaskState) SetLimits(maxActive, maxBacklog int) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if maxActive > 0 {
		s.maxActive = maxActive
	}
	if maxBacklog > 0 {
		s.maxBacklog = maxBacklog
	}
}

// Limits returns the current max active and backlog capacity limits.
func (s *TaskState) Limits() (maxActive, maxBacklog int) {
	if s == nil {
		return 0, 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.maxActive, s.maxBacklog
}

// ArmTaskTools initializes the native child task tools, registers their engines,
// installs the adjudicator gate once, and returns the planner-facing ToolDef declarations.
func ArmTaskTools() ([]ToolDef, error) {
	return ArmTaskToolsWithLimits(DefaultMaxActiveTasks, DefaultMaxBacklogTasks)
}

// ArmTaskToolsWithLimits initializes the native child task tools with explicit capacity limits,
// registers their engines, installs the adjudicator gate once, and returns planner ToolDefs.
func ArmTaskToolsWithLimits(maxActive, maxBacklog int) ([]ToolDef, error) {
	return ArmTaskToolsWithRunner(maxActive, maxBacklog, nil)
}

// ArmTaskToolsWithRunner arms bounded task execution through runner. Passing a
// nil runner preserves the intent-only task lifecycle used by legacy callers.
func ArmTaskToolsWithRunner(maxActive, maxBacklog int, runner ChildTaskRunner) ([]ToolDef, error) {
	st := NewTaskStateWithRunner(runner)
	if maxActive > 0 {
		st.maxActive = maxActive
	}
	if maxBacklog > 0 {
		st.maxBacklog = maxBacklog
	}
	if old := armedTaskTools.Swap(st); old != nil {
		old.Close()
	}

	registerTaskToolRuntime()

	return TaskToolCatalog(), nil
}

// DisarmTaskTools unarms the task tools, restoring the inactive state.
func DisarmTaskTools() {
	if old := armedTaskTools.Swap(nil); old != nil {
		old.Close()
	}
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
			"idempotentHint": "false",
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
	return taskToolNames()
}

func taskToolNames() []string {
	return []string{ToolTaskSpawn, ToolTaskWait, ToolTaskStatus, ToolTaskCancel}
}
