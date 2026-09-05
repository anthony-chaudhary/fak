package agent

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// MinSubagentTimeoutMs enforces a sane minimum timeout floor for subagents (60 seconds).
const MinSubagentTimeoutMs = 60000

// DefaultMaxThreads is the default maximum thread limit preventing thread exhaustion.
const DefaultMaxThreads = 16

// Subagent thread execution states.
const (
	SubagentStateRunning   = "running"
	SubagentStateCompleted = "completed"
	SubagentStateFailed    = "failed"
	SubagentStateTimedOut  = "timed_out"
)

// ClampWaitTimeout clamps timeoutMs to >= MinSubagentTimeoutMs (or default 60s if <= 0).
func ClampWaitTimeout(timeoutMs int) int {
	if timeoutMs < MinSubagentTimeoutMs {
		return MinSubagentTimeoutMs
	}
	return timeoutMs
}

// GoalDescriptor represents the goal state projected from a parent session.
type GoalDescriptor struct {
	GoalID          string    `json:"goal_id"`
	Description     string    `json:"description"`
	ReadOnly        bool      `json:"read_only"`
	ProjectedAt     time.Time `json:"projected_at"`
	ParentSessionID string    `json:"parent_session_id"`
}

// SubagentThread represents an individual subagent thread and its lifecycle.
type SubagentThread struct {
	ID              string          `json:"id"`
	ParentSessionID string          `json:"parent_session_id"`
	State           string          `json:"state"`
	CreatedAt       time.Time       `json:"created_at"`
	CompletedAt     time.Time       `json:"completed_at,omitempty"`
	ExitErr         error           `json:"exit_err,omitempty"`
	ProjectedGoal   *GoalDescriptor `json:"projected_goal,omitempty"`

	done      chan struct{}
	closeOnce sync.Once
}

// SubagentSupervisor coordinates subagent threads, enforcing timeout floors,
// thread limit caps, automatic thread reaping, and leak prevention.
type SubagentSupervisor struct {
	mu             sync.RWMutex
	MaxThreadLimit int
	active         map[string]*SubagentThread
	completed      map[string]*SubagentThread
}

// NewSubagentSupervisor creates a new SubagentSupervisor with a max active thread limit.
func NewSubagentSupervisor(maxThreads int) *SubagentSupervisor {
	if maxThreads <= 0 {
		maxThreads = DefaultMaxThreads
	}
	return &SubagentSupervisor{
		MaxThreadLimit: maxThreads,
		active:         make(map[string]*SubagentThread),
		completed:      make(map[string]*SubagentThread),
	}
}

// Spawn starts tracking a new subagent thread.
// It checks if active thread limit is reached; if so, triggers proactive reaping of any completed threads first.
// If still saturated, returns error "collab spawn failed: agent thread limit reached".
// It projects the parent goal as a read-only snapshot into the child (ReadOnly: true).
func (s *SubagentSupervisor) Spawn(parentSessionID, subagentID string, parentGoal *GoalDescriptor) (*SubagentThread, error) {
	if subagentID == "" {
		return nil, errors.New("subagent ID cannot be empty")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.active) >= s.MaxThreadLimit {
		s.autoReapLocked()
	}
	if len(s.active) >= s.MaxThreadLimit {
		return nil, errors.New("collab spawn failed: agent thread limit reached")
	}

	if _, exists := s.active[subagentID]; exists {
		return nil, fmt.Errorf("subagent %q is already running", subagentID)
	}
	delete(s.completed, subagentID)

	var projected *GoalDescriptor
	if parentGoal != nil {
		projected = &GoalDescriptor{
			GoalID:          parentGoal.GoalID,
			Description:     parentGoal.Description,
			ReadOnly:        true,
			ProjectedAt:     time.Now().UTC(),
			ParentSessionID: parentSessionID,
		}
	}

	thread := &SubagentThread{
		ID:              subagentID,
		ParentSessionID: parentSessionID,
		State:           SubagentStateRunning,
		CreatedAt:       time.Now().UTC(),
		ProjectedGoal:   projected,
		done:            make(chan struct{}),
	}

	s.active[subagentID] = thread
	return thread, nil
}

// Wait blocks until the specified subagent completes, fails, or times out.
// Clamps requested timeoutMs to >= 60,000ms using ClampWaitTimeout.
func (s *SubagentSupervisor) Wait(subagentID string, timeoutMs int) (*SubagentThread, error) {
	clampedTimeout := ClampWaitTimeout(timeoutMs)

	s.mu.Lock()
	thread, ok := s.active[subagentID]
	if !ok {
		thread, ok = s.completed[subagentID]
	}
	if !ok {
		s.mu.Unlock()
		return nil, fmt.Errorf("subagent %q not found", subagentID)
	}

	if thread.State != SubagentStateRunning {
		s.mu.Unlock()
		return thread, thread.ExitErr
	}
	done := thread.done
	s.mu.Unlock()

	select {
	case <-done:
		s.mu.RLock()
		defer s.mu.RUnlock()
		return thread, thread.ExitErr
	case <-time.After(time.Duration(clampedTimeout) * time.Millisecond):
		s.mu.Lock()
		defer s.mu.Unlock()
		if thread.State == SubagentStateRunning {
			thread.State = SubagentStateTimedOut
			thread.CompletedAt = time.Now().UTC()
			thread.ExitErr = fmt.Errorf("subagent %q timed out after %dms", subagentID, clampedTimeout)
			thread.closeOnce.Do(func() {
				close(thread.done)
			})
		}
		return thread, thread.ExitErr
	}
}

// Complete marks the subagent completed or failed.
func (s *SubagentSupervisor) Complete(subagentID string, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	thread, ok := s.active[subagentID]
	if !ok {
		thread, ok = s.completed[subagentID]
	}
	if !ok {
		return
	}

	if thread.State == SubagentStateRunning {
		if err != nil {
			thread.State = SubagentStateFailed
			thread.ExitErr = err
		} else {
			thread.State = SubagentStateCompleted
		}
		thread.CompletedAt = time.Now().UTC()
		thread.closeOnce.Do(func() {
			close(thread.done)
		})
	}
}

// AutoReap automatically cleans up/reaps finished subagents, freeing active thread slots.
func (s *SubagentSupervisor) AutoReap() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.autoReapLocked()
}

func (s *SubagentSupervisor) autoReapLocked() {
	for id, thread := range s.active {
		if thread.State != SubagentStateRunning {
			delete(s.active, id)
			s.completed[id] = thread
		}
	}
}

// TeardownParent automatically reaps all child subagents owned by this parent.
func (s *SubagentSupervisor) TeardownParent(parentSessionID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now().UTC()
	for id, thread := range s.active {
		if thread.ParentSessionID == parentSessionID {
			if thread.State == SubagentStateRunning {
				thread.State = SubagentStateCompleted
				thread.CompletedAt = now
				thread.closeOnce.Do(func() {
					close(thread.done)
				})
			}
			delete(s.active, id)
			s.completed[id] = thread
		}
	}
}

// GetGoal returns the projected read-only parent goal (never null if parent had a goal).
func (s *SubagentSupervisor) GetGoal(subagentID string) (*GoalDescriptor, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	thread, ok := s.active[subagentID]
	if !ok {
		thread, ok = s.completed[subagentID]
	}
	if !ok {
		return nil, fmt.Errorf("subagent %q not found", subagentID)
	}

	if thread.ProjectedGoal == nil {
		return nil, fmt.Errorf("subagent %q has no projected goal", subagentID)
	}

	goalCopy := *thread.ProjectedGoal
	return &goalCopy, nil
}

// ActiveCount returns the number of active threads currently registered.
func (s *SubagentSupervisor) ActiveCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.active)
}

// CompletedCount returns the count of completed threads retained in the pool.
func (s *SubagentSupervisor) CompletedCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.completed)
}

// GetThread retrieves a subagent thread by ID from active or completed pools.
func (s *SubagentSupervisor) GetThread(subagentID string) (*SubagentThread, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if t, ok := s.active[subagentID]; ok {
		return t, true
	}
	if t, ok := s.completed[subagentID]; ok {
		return t, true
	}
	return nil, false
}
