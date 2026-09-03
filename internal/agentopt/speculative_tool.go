package agentopt

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"sync"
	"time"
)

// ToolCall describes an invoked or predicted tool operation.
type ToolCall struct {
	ID       string         `json:"id"`
	Name     string         `json:"name"`
	Args     map[string]any `json:"args"`
	ReadOnly bool           `json:"read_only"`
}

// SpeculativeReceipt captures the result and performance metadata of an executed or cached tool call.
type SpeculativeReceipt struct {
	CallID      string        `json:"call_id"`
	ToolName    string        `json:"tool_name"`
	ArgsDigest  string        `json:"args_digest"`
	Output      string        `json:"output"`
	Duration    time.Duration `json:"duration"`
	Speculative bool          `json:"speculative"`
	Hit         bool          `json:"hit"`
	ExecutedAt  time.Time     `json:"executed_at"`
	Error       string        `json:"error,omitempty"`
}

// SpeculativeEngine manages in-process speculative execution and receipt caching for read-only tools.
type SpeculativeEngine struct {
	mu            sync.RWMutex
	readOnlyTools map[string]bool
	cache         map[string]SpeculativeReceipt
}

// NewSpeculativeEngine creates a new engine with standard read-only tools.
func NewSpeculativeEngine(readOnlyTools ...string) *SpeculativeEngine {
	tools := map[string]bool{
		"Read": true, "Glob": true, "Grep": true, "read_file": true, "list_dir": true,
	}
	for _, t := range readOnlyTools {
		tools[t] = true
	}
	return &SpeculativeEngine{
		readOnlyTools: tools,
		cache:         make(map[string]SpeculativeReceipt),
	}
}

// RegisterReadOnlyTool registers an additional tool as deterministic and read-only.
func (e *SpeculativeEngine) RegisterReadOnlyTool(name string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.readOnlyTools[name] = true
}

// IsReadOnly reports whether the tool is designated as read-only.
func (e *SpeculativeEngine) IsReadOnly(name string) bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.readOnlyTools[name]
}

// DigestCall computes a deterministic hash of the tool name and arguments.
func DigestCall(toolName string, args map[string]any) string {
	b, _ := json.Marshal(args)
	h := sha256.New()
	h.Write([]byte(toolName))
	h.Write([]byte{0})
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil))
}

// Speculate queues and executes a tool call in advance of model completion.
func (e *SpeculativeEngine) Speculate(ctx context.Context, call ToolCall, execFn func(context.Context) (string, error)) (SpeculativeReceipt, error) {
	if !e.IsReadOnly(call.Name) && !call.ReadOnly {
		return SpeculativeReceipt{}, errors.New("cannot speculate non-read-only tool call")
	}
	key := DigestCall(call.Name, call.Args)

	e.mu.Lock()
	if existing, ok := e.cache[key]; ok {
		e.mu.Unlock()
		return existing, nil
	}
	e.mu.Unlock()

	start := time.Now()
	out, err := execFn(ctx)
	elapsed := time.Since(start)

	receipt := SpeculativeReceipt{
		CallID:      call.ID,
		ToolName:    call.Name,
		ArgsDigest:  key,
		Output:      out,
		Duration:    elapsed,
		Speculative: true,
		Hit:         false,
		ExecutedAt:  time.Now().UTC(),
	}
	if err != nil {
		receipt.Error = err.Error()
	}

	e.mu.Lock()
	e.cache[key] = receipt
	e.mu.Unlock()

	return receipt, err
}

// ExecuteOrHit returns the cached speculative receipt with zero latency if present,
// or executes the tool synchronously if missed.
func (e *SpeculativeEngine) ExecuteOrHit(ctx context.Context, call ToolCall, execFn func(context.Context) (string, error)) (SpeculativeReceipt, error) {
	key := DigestCall(call.Name, call.Args)

	e.mu.Lock()
	if receipt, ok := e.cache[key]; ok {
		e.mu.Unlock()
		receipt.Hit = true
		receipt.Duration = 0
		return receipt, nil
	}
	e.mu.Unlock()

	start := time.Now()
	out, err := execFn(ctx)
	elapsed := time.Since(start)

	receipt := SpeculativeReceipt{
		CallID:      call.ID,
		ToolName:    call.Name,
		ArgsDigest:  key,
		Output:      out,
		Duration:    elapsed,
		Speculative: false,
		Hit:         false,
		ExecutedAt:  time.Now().UTC(),
	}
	if err != nil {
		receipt.Error = err.Error()
	}

	if e.IsReadOnly(call.Name) || call.ReadOnly {
		e.mu.Lock()
		e.cache[key] = receipt
		e.mu.Unlock()
	}

	return receipt, err
}

// InvalidateAll clears the speculative receipt cache.
func (e *SpeculativeEngine) InvalidateAll() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.cache = make(map[string]SpeculativeReceipt)
}

// InvalidateTool purges all cached receipts for a specified tool.
func (e *SpeculativeEngine) InvalidateTool(toolName string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for k, v := range e.cache {
		if v.ToolName == toolName {
			delete(e.cache, k)
		}
	}
}

// CacheSize returns the number of active cached speculative receipts.
func (e *SpeculativeEngine) CacheSize() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.cache)
}
