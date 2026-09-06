package agentqueue

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"
)

// PromptTask state values.
const (
	PromptTaskQueued    = "queued"
	PromptTaskRunning   = "running"
	PromptTaskCompleted = "completed"
	PromptTaskFailed    = "failed"
	PromptTaskHeld      = "held"

	PromptTaskStateQueued    = PromptTaskQueued
	PromptTaskStateRunning   = PromptTaskRunning
	PromptTaskStateCompleted = PromptTaskCompleted
	PromptTaskStateFailed    = PromptTaskFailed
	PromptTaskStateHeld      = PromptTaskHeld
)

// Typed errors for prompt task operations.
var (
	ErrTaskNotFound          = errors.New("agentqueue: prompt task not found")
	ErrQueueCapacityExceeded = errors.New("agentqueue: prompt task queue capacity exceeded")
	ErrQueueFull             = ErrQueueCapacityExceeded
	ErrInvalidState          = errors.New("agentqueue: invalid prompt task state")
	ErrInvalidTransition     = errors.New("agentqueue: invalid prompt task transition")
)

// PromptTaskSpec defines the specification for submitting a parent-scoped prompt task.
type PromptTaskSpec struct {
	TaskID          string    `json:"task_id,omitempty"`
	ParentSessionID string    `json:"parent_session_id,omitempty"`
	IdempotencyKey  string    `json:"idempotency_key,omitempty"`
	Prompt          string    `json:"prompt"`
	State           string    `json:"state,omitempty"`
	CreatedAt       time.Time `json:"created_at,omitempty"`
	Result          string    `json:"result,omitempty"`
}

// PromptTaskHandle represents a prompt task in the queue with its current status and outcome.
type PromptTaskHandle struct {
	TaskID          string    `json:"task_id"`
	ParentSessionID string    `json:"parent_session_id,omitempty"`
	IdempotencyKey  string    `json:"idempotency_key,omitempty"`
	Prompt          string    `json:"prompt"`
	State           string    `json:"state"`
	CreatedAt       time.Time `json:"created_at"`
	Result          string    `json:"result,omitempty"`
}

// IsTerminal returns true if the task is in completed or failed state.
func (h PromptTaskHandle) IsTerminal() bool {
	return h.State == PromptTaskCompleted || h.State == PromptTaskFailed
}

// CapacityPolicy determines queue behavior when backlog capacity is exhausted.
type CapacityPolicy string

const (
	CapacityPolicyReject CapacityPolicy = "reject"
	CapacityPolicyHold   CapacityPolicy = "hold"
)

// PromptTaskManagerConfig holds configuration options for PromptTaskManager.
type PromptTaskManagerConfig struct {
	Capacity       int
	CapacityPolicy CapacityPolicy
	HoldOnFull     bool
}

// PromptTaskManager manages admission, deduplication, capacity limits, and lifecycle
// state transitions for parent-scoped prompt tasks.
type PromptTaskManager struct {
	mu             sync.RWMutex
	capacity       int
	capacityPolicy CapacityPolicy
	holdOnFull     bool
	tasks          map[string]*PromptTaskHandle
	dedup          map[string]string // caller-scoped dedupKey -> taskID
	order          []string          // taskIDs in submission order
}

// NewPromptTaskManager creates a manager with bounded backlog capacity that rejects
// new tasks with ErrQueueCapacityExceeded when capacity is exhausted.
func NewPromptTaskManager(capacity int) *PromptTaskManager {
	return &PromptTaskManager{
		capacity:       capacity,
		capacityPolicy: CapacityPolicyReject,
		holdOnFull:     false,
		tasks:          make(map[string]*PromptTaskHandle),
		dedup:          make(map[string]string),
	}
}

// NewPromptTaskManagerWithHold creates a manager that places new tasks into "held"
// state when backlog capacity is exhausted.
func NewPromptTaskManagerWithHold(capacity int) *PromptTaskManager {
	return &PromptTaskManager{
		capacity:       capacity,
		capacityPolicy: CapacityPolicyHold,
		holdOnFull:     true,
		tasks:          make(map[string]*PromptTaskHandle),
		dedup:          make(map[string]string),
	}
}

// NewPromptTaskManagerWithConfig creates a manager with explicit configuration.
func NewPromptTaskManagerWithConfig(cfg PromptTaskManagerConfig) *PromptTaskManager {
	hold := cfg.HoldOnFull || cfg.CapacityPolicy == CapacityPolicyHold
	policy := cfg.CapacityPolicy
	if policy == "" {
		if hold {
			policy = CapacityPolicyHold
		} else {
			policy = CapacityPolicyReject
		}
	}
	return &PromptTaskManager{
		capacity:       cfg.Capacity,
		capacityPolicy: policy,
		holdOnFull:     hold,
		tasks:          make(map[string]*PromptTaskHandle),
		dedup:          make(map[string]string),
	}
}

// PromptTaskManager attaches a PromptTaskManager constructor to Store.
func (s Store) PromptTaskManager(capacity int) *PromptTaskManager {
	return NewPromptTaskManager(capacity)
}

func (m *PromptTaskManager) init() {
	if m.tasks == nil {
		m.tasks = make(map[string]*PromptTaskHandle)
	}
	if m.dedup == nil {
		m.dedup = make(map[string]string)
	}
}

// SetCapacity adjusts the backlog capacity limit dynamically.
func (m *PromptTaskManager) SetCapacity(capacity int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.capacity = capacity
}

// SetHoldOnFull controls whether tasks are held (true) or rejected (false) when full.
func (m *PromptTaskManager) SetHoldOnFull(hold bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.holdOnFull = hold
	if hold {
		m.capacityPolicy = CapacityPolicyHold
	} else {
		m.capacityPolicy = CapacityPolicyReject
	}
}

// SetCapacityPolicy configures the full-queue policy ("reject" or "hold").
func (m *PromptTaskManager) SetCapacityPolicy(policy CapacityPolicy) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.capacityPolicy = policy
	m.holdOnFull = (policy == CapacityPolicyHold)
}

func dedupKey(parentSessionID, idempotencyKey string) string {
	if parentSessionID == "" {
		return "\x00" + idempotencyKey
	}
	return parentSessionID + "\x00" + idempotencyKey
}

func generatePromptTaskID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("ptask-%d", time.Now().UnixNano())
	}
	return "ptask-" + hex.EncodeToString(b[:])
}

// Submit admits a prompt task into the bounded queue without requiring a synthetic
// GitHub issue number. Performs caller-scoped deduplication on IdempotencyKey.
func (m *PromptTaskManager) Submit(spec PromptTaskSpec) (PromptTaskHandle, error) {
	return m.submitInternal(spec, false)
}

// SubmitOrHold admits a prompt task, holding it in "held" state if capacity is full.
func (m *PromptTaskManager) SubmitOrHold(spec PromptTaskSpec) (PromptTaskHandle, error) {
	return m.submitInternal(spec, true)
}

func (m *PromptTaskManager) submitInternal(spec PromptTaskSpec, forceHoldOnFull bool) (PromptTaskHandle, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()

	// 1. Caller-scoped deduplication: submitting with the same IdempotencyKey
	// returns the existing task handle rather than creating a duplicate.
	if spec.IdempotencyKey != "" {
		key := dedupKey(spec.ParentSessionID, spec.IdempotencyKey)
		if existingID, ok := m.dedup[key]; ok {
			if existing, ok := m.tasks[existingID]; ok {
				return *existing, nil
			}
		}
		// If ParentSessionID was not specified, check fallback without session prefix
		if spec.ParentSessionID == "" {
			for _, t := range m.tasks {
				if t.IdempotencyKey == spec.IdempotencyKey {
					return *t, nil
				}
			}
		}
	}

	// 2. Determine initial state
	targetState := spec.State
	if targetState == "" {
		targetState = PromptTaskQueued
	}

	// 3. Enforce queue capacity limits (reject or hold when backlog capacity is exhausted)
	if targetState == PromptTaskQueued && m.capacity > 0 {
		backlog := m.backlogCountLocked()
		if backlog >= m.capacity {
			shouldHold := forceHoldOnFull || m.holdOnFull || m.capacityPolicy == CapacityPolicyHold
			if shouldHold {
				targetState = PromptTaskHeld
			} else {
				return PromptTaskHandle{}, ErrQueueCapacityExceeded
			}
		}
	}

	taskID := spec.TaskID
	if taskID == "" {
		taskID = generatePromptTaskID()
	} else if _, exists := m.tasks[taskID]; exists {
		return PromptTaskHandle{}, fmt.Errorf("agentqueue: task ID %q already exists", taskID)
	}

	createdAt := spec.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}

	handle := PromptTaskHandle{
		TaskID:          taskID,
		ParentSessionID: spec.ParentSessionID,
		IdempotencyKey:  spec.IdempotencyKey,
		Prompt:          spec.Prompt,
		State:           targetState,
		CreatedAt:       createdAt,
		Result:          spec.Result,
	}

	m.tasks[taskID] = &handle
	m.order = append(m.order, taskID)

	if spec.IdempotencyKey != "" {
		key := dedupKey(spec.ParentSessionID, spec.IdempotencyKey)
		m.dedup[key] = taskID
	}

	return handle, nil
}

func (m *PromptTaskManager) backlogCountLocked() int {
	count := 0
	for _, t := range m.tasks {
		if t.State == PromptTaskQueued {
			count++
		}
	}
	return count
}

func validateState(state string) error {
	switch state {
	case PromptTaskQueued, PromptTaskRunning, PromptTaskCompleted, PromptTaskFailed, PromptTaskHeld:
		return nil
	default:
		return fmt.Errorf("%w: %q", ErrInvalidState, state)
	}
}

func validateTransition(from, to string) error {
	if from == to {
		return nil
	}
	if err := validateState(to); err != nil {
		return err
	}
	switch from {
	case PromptTaskQueued:
		return nil
	case PromptTaskHeld:
		if to == PromptTaskCompleted {
			return fmt.Errorf("%w: cannot transition directly from held to completed", ErrInvalidTransition)
		}
		return nil
	case PromptTaskRunning:
		return nil
	case PromptTaskCompleted, PromptTaskFailed:
		return fmt.Errorf("%w: cannot transition from terminal state %q to %q", ErrInvalidTransition, from, to)
	default:
		return fmt.Errorf("%w: unknown source state %q", ErrInvalidTransition, from)
	}
}

// UpdateState transitions a task to a new state if valid.
func (m *PromptTaskManager) UpdateState(taskID string, newState string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()

	task, ok := m.tasks[taskID]
	if !ok {
		return ErrTaskNotFound
	}
	if err := validateTransition(task.State, newState); err != nil {
		return err
	}
	task.State = newState
	return nil
}

// UpdateStatus is an alias for UpdateState.
func (m *PromptTaskManager) UpdateStatus(taskID string, state string) error {
	return m.UpdateState(taskID, state)
}

// Start transitions a task from queued or held to running.
func (m *PromptTaskManager) Start(taskID string) error {
	return m.UpdateState(taskID, PromptTaskRunning)
}

// Complete transitions a task to completed and records its result.
func (m *PromptTaskManager) Complete(taskID string, result string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()

	task, ok := m.tasks[taskID]
	if !ok {
		return ErrTaskNotFound
	}
	if task.State == PromptTaskCompleted {
		task.Result = result
		return nil
	}
	if err := validateTransition(task.State, PromptTaskCompleted); err != nil {
		return err
	}
	task.State = PromptTaskCompleted
	task.Result = result
	return nil
}

// Fail transitions a task to failed and records its error or result.
func (m *PromptTaskManager) Fail(taskID string, result string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()

	task, ok := m.tasks[taskID]
	if !ok {
		return ErrTaskNotFound
	}
	if task.State == PromptTaskFailed {
		task.Result = result
		return nil
	}
	if err := validateTransition(task.State, PromptTaskFailed); err != nil {
		return err
	}
	task.State = PromptTaskFailed
	task.Result = result
	return nil
}

// UpdateResult updates both state and result on a task.
func (m *PromptTaskManager) UpdateResult(taskID string, state string, result string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()

	task, ok := m.tasks[taskID]
	if !ok {
		return ErrTaskNotFound
	}
	if state != "" {
		if err := validateTransition(task.State, state); err != nil {
			return err
		}
		task.State = state
	}
	task.Result = result
	return nil
}

// SetResult updates the result string on a task without changing its state.
func (m *PromptTaskManager) SetResult(taskID string, result string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()

	task, ok := m.tasks[taskID]
	if !ok {
		return ErrTaskNotFound
	}
	task.Result = result
	return nil
}

// Get returns the task handle for taskID, or ErrTaskNotFound.
func (m *PromptTaskManager) Get(taskID string) (PromptTaskHandle, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	m.init()

	task, ok := m.tasks[taskID]
	if !ok {
		return PromptTaskHandle{}, ErrTaskNotFound
	}
	return *task, nil
}

// GetTask returns the task handle and whether it was found.
func (m *PromptTaskManager) GetTask(taskID string) (PromptTaskHandle, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	m.init()

	task, ok := m.tasks[taskID]
	if !ok {
		return PromptTaskHandle{}, false
	}
	return *task, true
}

// GetStatus returns the current state of taskID.
func (m *PromptTaskManager) GetStatus(taskID string) (string, error) {
	h, err := m.Get(taskID)
	if err != nil {
		return "", err
	}
	return h.State, nil
}

// GetResult returns the current result of taskID.
func (m *PromptTaskManager) GetResult(taskID string) (string, error) {
	h, err := m.Get(taskID)
	if err != nil {
		return "", err
	}
	return h.Result, nil
}

// GetByIdempotencyKey looks up a task handle by caller-scoped idempotency key.
func (m *PromptTaskManager) GetByIdempotencyKey(parentSessionID, idempotencyKey string) (PromptTaskHandle, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	m.init()

	if idempotencyKey == "" {
		return PromptTaskHandle{}, false
	}
	key := dedupKey(parentSessionID, idempotencyKey)
	if taskID, ok := m.dedup[key]; ok {
		if task, ok := m.tasks[taskID]; ok {
			return *task, true
		}
	}
	if parentSessionID == "" {
		for _, t := range m.tasks {
			if t.IdempotencyKey == idempotencyKey {
				return *t, true
			}
		}
	}
	return PromptTaskHandle{}, false
}

// ReleaseHeld promotes held tasks to queued up to available capacity.
// Returns the count of tasks promoted.
func (m *PromptTaskManager) ReleaseHeld() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()

	promoted := 0
	for _, id := range m.order {
		if m.capacity > 0 && m.backlogCountLocked() >= m.capacity {
			break
		}
		t := m.tasks[id]
		if t != nil && t.State == PromptTaskHeld {
			t.State = PromptTaskQueued
			promoted++
		}
	}
	return promoted
}

// Count returns the total number of tasks in the manager.
func (m *PromptTaskManager) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.tasks)
}

// BacklogCount returns the number of tasks currently in "queued" state.
func (m *PromptTaskManager) BacklogCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.backlogCountLocked()
}

// List returns all prompt tasks in insertion order.
func (m *PromptTaskManager) List() []PromptTaskHandle {
	m.mu.RLock()
	defer m.mu.RUnlock()
	m.init()

	res := make([]PromptTaskHandle, 0, len(m.order))
	for _, id := range m.order {
		if t, ok := m.tasks[id]; ok {
			res = append(res, *t)
		}
	}
	return res
}

// ListByState returns all tasks matching the specified state.
func (m *PromptTaskManager) ListByState(state string) []PromptTaskHandle {
	m.mu.RLock()
	defer m.mu.RUnlock()
	m.init()

	var res []PromptTaskHandle
	for _, id := range m.order {
		if t, ok := m.tasks[id]; ok && t.State == state {
			res = append(res, *t)
		}
	}
	return res
}

// ListByParentSession returns all tasks associated with the given parent session ID.
func (m *PromptTaskManager) ListByParentSession(sessionID string) []PromptTaskHandle {
	m.mu.RLock()
	defer m.mu.RUnlock()
	m.init()

	var res []PromptTaskHandle
	for _, id := range m.order {
		if t, ok := m.tasks[id]; ok && t.ParentSessionID == sessionID {
			res = append(res, *t)
		}
	}
	return res
}
