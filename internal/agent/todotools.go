package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// todotools.go — kernel-mediated planning and todo list tools (todowrite / todoread)
// for the native agent harness (#11224).
//
// These tools provide native state-tracking and progress management with BEST_EFFORT
// consistency: immediate in-process synthesized results, structural validation, and
// strict bounds (max 100 entries, at most 1 item in_progress).

const (
	ToolTodoWrite = "todowrite"
	ToolTodoRead  = "todoread"

	EngineTodoWrite = "agent.todowrite"
	EngineTodoRead  = "agent.todoread"

	RungNameTodo = "todotools"

	todoToolRank = 22

	MaxTodoItems = 100

	StatusPending    = "pending"
	StatusInProgress = "in_progress"
	StatusCompleted  = "completed"
	StatusCancelled  = "cancelled"

	PriorityHigh   = "high"
	PriorityMedium = "medium"
	PriorityLow    = "low"
)

// TodoItem represents one structured task within a todo list.
type TodoItem struct {
	ID       string `json:"id,omitempty"`
	Content  string `json:"content"`
	Status   string `json:"status"`
	Priority string `json:"priority"`
}

// TodoList is the serialized wire input for todowrite.
type TodoList struct {
	Todos []TodoItem `json:"todos"`
}

// TodoReceipt is the compact structured confirmation returned by todowrite.
type TodoReceipt struct {
	Status     string `json:"status"`
	Total      int    `json:"total"`
	Pending    int    `json:"pending"`
	InProgress int    `json:"in_progress"`
	Completed  int    `json:"completed"`
	Cancelled  int    `json:"cancelled"`
	Error      string `json:"error,omitempty"`
}

// TodoReadResponse is the structured payload returned by todoread.
type TodoReadResponse struct {
	Todos      []TodoItem `json:"todos"`
	Total      int        `json:"total"`
	InProgress *TodoItem  `json:"in_progress,omitempty"`
}

// TodoState holds the thread-safe active todo items for a session.
type TodoState struct {
	mu    sync.RWMutex
	todos []TodoItem
}

// NewTodoState returns an empty initialized TodoState.
func NewTodoState() *TodoState {
	return &TodoState{
		todos: make([]TodoItem, 0),
	}
}

// GetTodos returns a copy of the current todos.
func (s *TodoState) GetTodos() []TodoItem {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]TodoItem, len(s.todos))
	copy(out, s.todos)
	return out
}

// SetTodos validates and atomically updates the full task list.
func (s *TodoState) SetTodos(todos []TodoItem) (TodoReceipt, error) {
	if s == nil {
		return TodoReceipt{}, fmt.Errorf("todo state is nil")
	}

	if len(todos) > MaxTodoItems {
		return TodoReceipt{Status: "error", Error: fmt.Sprintf("todo count %d exceeds maximum %d", len(todos), MaxTodoItems)},
			fmt.Errorf("todo count %d exceeds maximum %d", len(todos), MaxTodoItems)
	}

	inProgressCount := 0
	receipt := TodoReceipt{Total: len(todos)}
	sanitized := make([]TodoItem, len(todos))

	for i, item := range todos {
		content := strings.TrimSpace(item.Content)
		if content == "" {
			err := fmt.Errorf("todo at index %d has empty content", i)
			receipt.Status = "error"
			receipt.Error = err.Error()
			return receipt, err
		}

		status := strings.ToLower(strings.TrimSpace(item.Status))
		switch status {
		case StatusPending:
			receipt.Pending++
		case StatusInProgress:
			receipt.InProgress++
			inProgressCount++
		case StatusCompleted:
			receipt.Completed++
		case StatusCancelled:
			receipt.Cancelled++
		default:
			err := fmt.Errorf("todo at index %d has invalid status %q (allowed: pending, in_progress, completed, cancelled)", i, item.Status)
			receipt.Status = "error"
			receipt.Error = err.Error()
			return receipt, err
		}

		priority := strings.ToLower(strings.TrimSpace(item.Priority))
		if priority == "" {
			priority = PriorityMedium
		}
		switch priority {
		case PriorityHigh, PriorityMedium, PriorityLow:
		default:
			err := fmt.Errorf("todo at index %d has invalid priority %q (allowed: high, medium, low)", i, item.Priority)
			receipt.Status = "error"
			receipt.Error = err.Error()
			return receipt, err
		}

		id := strings.TrimSpace(item.ID)
		if id == "" {
			id = fmt.Sprintf("task-%d", i+1)
		}

		sanitized[i] = TodoItem{
			ID:       id,
			Content:  content,
			Status:   status,
			Priority: priority,
		}
	}

	if inProgressCount > 1 {
		err := fmt.Errorf("invalid plan: exactly 0 or 1 item may be in_progress (found %d)", inProgressCount)
		receipt.Status = "error"
		receipt.Error = err.Error()
		return receipt, err
	}

	receipt.Status = "ok"

	s.mu.Lock()
	s.todos = sanitized
	s.mu.Unlock()

	return receipt, nil
}

// todoEngine implements abi.Engine for todowrite and todoread.
type todoEngine struct {
	state *TodoState
}

func (e *todoEngine) Caps() []abi.Capability { return nil }
func (e *todoEngine) WeightBearing() bool    { return false }

func (e *todoEngine) getState() *TodoState {
	if e != nil && e.state != nil {
		return e.state
	}
	return armedTodoTools.Load()
}

func (e *todoEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body, _ := decodeCallArgs(ctx, c.Args)
	st := e.getState()
	if st == nil {
		errResp, _ := json.Marshal(TodoReceipt{Status: "error", Error: "todo state is unarmed"})
		return engineResult(ctx, c, body, errResp, true, "todotools"), nil
	}

	switch c.Tool {
	case ToolTodoWrite:
		var list TodoList
		if len(body) > 0 {
			if err := json.Unmarshal(body, &list); err != nil {
				errResp, _ := json.Marshal(TodoReceipt{Status: "error", Error: fmt.Sprintf("invalid arguments JSON: %v", err)})
				return engineResult(ctx, c, body, errResp, true, EngineTodoWrite), nil
			}
		}
		receipt, err := st.SetTodos(list.Todos)
		respBytes, _ := json.Marshal(receipt)
		return engineResult(ctx, c, body, respBytes, err != nil, EngineTodoWrite), nil

	case ToolTodoRead:
		todos := st.GetTodos()
		if todos == nil {
			todos = []TodoItem{}
		}
		var inProg *TodoItem
		for _, item := range todos {
			if item.Status == StatusInProgress {
				cp := item
				inProg = &cp
				break
			}
		}
		resp := TodoReadResponse{
			Todos:      todos,
			Total:      len(todos),
			InProgress: inProg,
		}
		respBytes, _ := json.Marshal(resp)
		return engineResult(ctx, c, body, respBytes, false, EngineTodoRead), nil

	default:
		return engineResult(ctx, c, body, []byte(fmt.Sprintf(`{"error":"unknown tool %q"}`, c.Tool)), true, "todotools"), nil
	}
}

// todoGate adjudicates todowrite and todoread calls, pinning the appropriate engine.
type todoGate struct{}

func (todoGate) Caps() []abi.Capability { return nil }

func (todoGate) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	st := armedTodoTools.Load()
	if st == nil {
		return abi.Verdict{Kind: abi.VerdictDefer, By: RungNameTodo}
	}
	switch c.Tool {
	case ToolTodoWrite:
		c.Engine = EngineTodoWrite
		return abi.Verdict{Kind: abi.VerdictAllow, By: RungNameTodo}
	case ToolTodoRead:
		c.Engine = EngineTodoRead
		return abi.Verdict{Kind: abi.VerdictAllow, By: RungNameTodo}
	default:
		return abi.Verdict{Kind: abi.VerdictDefer, By: RungNameTodo}
	}
}

var (
	armedTodoTools   atomic.Pointer[TodoState]
	todoGateOnce     sync.Once
	todoEnginesOnce  sync.Once
	activeTodoEngine *todoEngine
)

// ArmTodoTools initializes the native planning and todo list tools, registers their engines,
// installs the adjudicator gate once, and returns the planner-facing ToolDef declarations.
func ArmTodoTools() ([]ToolDef, error) {
	st := NewTodoState()
	armedTodoTools.Store(st)

	todoEnginesOnce.Do(func() {
		activeTodoEngine = &todoEngine{}
		abi.RegisterEngine(EngineTodoWrite, activeTodoEngine)
		abi.RegisterEngine(EngineTodoRead, activeTodoEngine)
	})

	todoGateOnce.Do(func() {
		abi.RegisterAdjudicator(todoToolRank, todoGate{})
	})

	return TodoToolCatalog(), nil
}

// DisarmTodoTools unarms the todo tools, restoring the inactive state.
func DisarmTodoTools() {
	armedTodoTools.Store(nil)
}

// GetActiveTodoState returns the active TodoState, or nil if unarmed.
func GetActiveTodoState() *TodoState {
	return armedTodoTools.Load()
}

// TodoToolCatalog renders the todo tools as loop ToolDefs. Empty when unarmed.
func TodoToolCatalog() []ToolDef {
	if armedTodoTools.Load() == nil {
		return nil
	}
	return todoToolDefs()
}

func todoToolDefs() []ToolDef {
	return []ToolDef{
		{
			Type: "function",
			Function: ToolDefFunction{
				Name: ToolTodoWrite,
				Description: "Create and maintain a structured task list. Organizes multi-step work, tracks progress, " +
					"and keeps exactly one task in_progress while active work remains.",
				Parameters: rawSchema(`{
  "type": "object",
  "properties": {
    "todos": {
      "type": "array",
      "description": "The updated todo list",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string", "description": "Optional unique identifier for the task"},
          "content": {"type": "string", "description": "Brief description of the task"},
          "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"], "description": "Current status"},
          "priority": {"type": "string", "enum": ["high", "medium", "low"], "description": "Priority level"}
        },
        "required": ["content", "status", "priority"]
      }
    }
  },
  "required": ["todos"]
}`),
			},
		},
		{
			Type: "function",
			Function: ToolDefFunction{
				Name:        ToolTodoRead,
				Description: "Read the current list of tasks, their execution statuses, and which task is currently active.",
				Parameters:  rawSchema(`{"type":"object","properties":{}}`),
			},
		},
	}
}

// todoToolMeta returns the vDSO / consistency scope metadata for todowrite and todoread.
func todoToolMeta(tool string) (map[string]string, bool) {
	if armedTodoTools.Load() == nil {
		return nil, false
	}
	switch tool {
	case ToolTodoRead:
		return map[string]string{
			"readOnlyHint":   "true",
			"idempotentHint": "false",
			"consistency":    "BEST_EFFORT",
		}, true
	case ToolTodoWrite:
		return map[string]string{
			"readOnlyHint":   "false",
			"idempotentHint": "false",
			"destructive":    "false",
			"consistency":    "BEST_EFFORT",
		}, true
	default:
		return nil, false
	}
}

// todoToolAllow returns the tool names admitted when todotools are armed.
func todoToolAllow() []string {
	if armedTodoTools.Load() == nil {
		return nil
	}
	return []string{ToolTodoWrite, ToolTodoRead}
}
