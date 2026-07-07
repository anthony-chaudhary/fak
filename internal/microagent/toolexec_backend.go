package microagent

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
)

// Backend is the dispatch-only half of the ToolExec seam (#2018, epic #2000
// M13/M18): it executes an action the in-process kernel floor has ALREADY
// allowed. A Backend never adjudicates and is never handed a non-Allowed
// action — the seam (ToolExec.Run) decides first and dispatches only on a
// kernel Allow. That split is the enforcement: the floor is a property of the
// SEAM, not a convention each backend re-implements (and can forget), so
// whatever the isolation level — goroutine, subprocess, container, microVM —
// the adjudication floor is identical.
//
// Name reports the backend's isolation-level name; it must match the
// internal/policy isolation-ladder vocabulary so the dial (M13) can select a
// backend per trust level.
type Backend interface {
	Name() string
	// Dispatch runs one already-allowed action and reports the captured
	// outcome. Implementations own execution mechanics only (spawn, capture,
	// timeout, reaping) — never policy.
	Dispatch(ctx context.Context, act ToolAction) (ToolResult, error)
}

// Backend names — aligned with the internal/policy isolation-ladder vocabulary
// (goroutine → subprocess → container → gvisor → firecracker → remote).
const (
	BackendGoroutine  = "goroutine"
	BackendSubprocess = "subprocess"
)

// Structured refusals for the backend seam and registry.
var (
	ErrNilBackend     = errors.New("microagent: NewToolExecBackend requires a backend (nil Backend)")
	ErrUnknownBackend = errors.New("microagent: no ToolExec backend registered under that name")
	ErrNoGoTool       = errors.New("microagent: goroutine backend has no registered func for the tool")
)

// The by-name backend registry (M13). It stores CONSTRUCTORS, not instances,
// so every executor gets its own backend state; and its only exit is
// NewToolExecFor, which requires the kernel floor — the registry can never
// issue a bare, unadjudicated executor.
var (
	backendsMu sync.RWMutex
	backends   = map[string]func() Backend{}
)

// RegisterBackend registers a backend constructor under an isolation-level
// name. Double-registration, an empty name, and a nil constructor are refused
// loud — a silent overwrite could swap a strong backend for a weak one.
//
// The #2018 conformance suite (toolexec_floor_conformance_test.go) pins the
// registered vocabulary: a newly-registered backend trips it until the suite
// proves a policy-denied action is blocked in that backend too.
func RegisterBackend(name string, mk func() Backend) error {
	if name == "" {
		return errors.New("microagent: RegisterBackend requires a backend name")
	}
	if mk == nil {
		return fmt.Errorf("microagent: RegisterBackend(%q) requires a constructor (nil func)", name)
	}
	backendsMu.Lock()
	defer backendsMu.Unlock()
	if _, dup := backends[name]; dup {
		return fmt.Errorf("microagent: backend %q already registered (no silent overwrite)", name)
	}
	backends[name] = mk
	return nil
}

// RegisteredBackends reports the registered backend names, sorted.
func RegisteredBackends() []string {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	names := make([]string, 0, len(backends))
	for name := range backends {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// NewToolExecFor builds a floor-adjudicated executor over the named registered
// backend. It is the registry's ONLY exit: there is no way to obtain a
// registered backend without the kernel floor wrapped around it.
func NewToolExecFor(name string, floor KernelFloor) (*ToolExec, error) {
	backendsMu.RLock()
	mk := backends[name]
	backendsMu.RUnlock()
	if mk == nil {
		return nil, fmt.Errorf("%w: %q (registered: %v)", ErrUnknownBackend, name, RegisteredBackends())
	}
	return NewToolExecBackend(floor, mk())
}

// The built-in isolation levels. Registration failure here is a programming
// error (duplicate built-in), not a runtime condition — fail loud at init.
func init() {
	if err := RegisterBackend(BackendGoroutine, func() Backend { return NewGoroutineBackend() }); err != nil {
		panic(err)
	}
	if err := RegisterBackend(BackendSubprocess, func() Backend { return subprocessBackend{} }); err != nil {
		panic(err)
	}
}

// GoToolFunc is one in-process tool implementation the goroutine backend
// dispatches to. It receives a deadline-bounded ctx (ToolAction.Timeout, else
// DefaultActionTimeout) and is TRUSTED to honor it — the goroutine tier has no
// kill isolation; that is exactly what the subprocess tier and above buy.
type GoToolFunc func(ctx context.Context, act ToolAction) (ToolResult, error)

// GoroutineBackend is the trusted in-process backend (#2003, M14 rung 0): tool
// actions run as plain Go funcs in the host process. It is the CHEAPEST — and
// therefore the most tempting place to skip adjudication; it gets none of
// that choice: like every backend it is dispatch-only, reachable at runtime
// only behind the ToolExec seam's kernel floor (#2018).
type GoroutineBackend struct {
	mu    sync.RWMutex
	funcs map[string]GoToolFunc
}

// NewGoroutineBackend builds an empty in-process backend; register tool funcs
// with Register.
func NewGoroutineBackend() *GoroutineBackend {
	return &GoroutineBackend{funcs: map[string]GoToolFunc{}}
}

// Name reports the isolation-ladder name for the in-process tier.
func (g *GoroutineBackend) Name() string { return BackendGoroutine }

// Register binds an in-process func to a logical tool name. Empty names, nil
// funcs, and double-registration are refused loud.
func (g *GoroutineBackend) Register(tool string, fn GoToolFunc) error {
	if tool == "" {
		return errors.New("microagent: GoroutineBackend.Register requires a tool name")
	}
	if fn == nil {
		return fmt.Errorf("microagent: GoroutineBackend.Register(%q) requires a func (nil GoToolFunc)", tool)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if _, dup := g.funcs[tool]; dup {
		return fmt.Errorf("microagent: GoroutineBackend tool %q already registered (no silent overwrite)", tool)
	}
	g.funcs[tool] = fn
	return nil
}

// Dispatch runs the registered func for the action's tool under the per-action
// deadline. The func's result is reported with Ran=true (it executed in this
// process); a deadline fire is recorded as TimedOut. An unregistered tool is a
// structured refusal (ErrNoGoTool) — reachable only AFTER a kernel Allow,
// because dispatch is only reachable through the seam.
func (g *GoroutineBackend) Dispatch(ctx context.Context, act ToolAction) (ToolResult, error) {
	g.mu.RLock()
	fn := g.funcs[act.Tool]
	g.mu.RUnlock()
	if fn == nil {
		return ToolResult{}, fmt.Errorf("%w: %q", ErrNoGoTool, act.Tool)
	}
	timeout := act.Timeout
	if timeout <= 0 {
		timeout = DefaultActionTimeout
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	res, err := fn(runCtx, act)
	res.Ran = true
	if runCtx.Err() == context.DeadlineExceeded {
		res.TimedOut = true
	}
	return res, err
}
