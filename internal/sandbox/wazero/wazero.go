package wazero

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

// ProviderName is the canonical registry identifier for the pure-Go Wazero WASI sandbox.
const ProviderName = "wazero_wasi"

func init() {
	sandbox.RegisterProvider(NewProvider())
}

// ---------------------------------------------------------------------------
// COMPILED MODULE CACHE
// ---------------------------------------------------------------------------

// ModuleCache maintains a thread-safe cache of pre-compiled Wasm modules keyed by SHA-256.
type ModuleCache struct {
	mu      sync.RWMutex
	modules map[string]*CompiledModule
	hits    int64
	misses  int64
}

// NewModuleCache creates an empty ModuleCache.
func NewModuleCache() *ModuleCache {
	return &ModuleCache{
		modules: make(map[string]*CompiledModule),
	}
}

// GetOrCompile retrieves a cached module or compiles and caches the bytecode.
func (c *ModuleCache) GetOrCompile(bytecode []byte) (*CompiledModule, bool, error) {
	hash := sha256.Sum256(bytecode)
	key := hex.EncodeToString(hash[:])

	c.mu.RLock()
	mod, ok := c.modules[key]
	c.mu.RUnlock()
	if ok {
		atomic.AddInt64(&c.hits, 1)
		return mod, true, nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if mod, ok = c.modules[key]; ok {
		atomic.AddInt64(&c.hits, 1)
		return mod, true, nil
	}

	compiled, err := Compile(bytecode)
	if err != nil {
		return nil, false, err
	}
	compiled.Hash = key
	c.modules[key] = compiled
	atomic.AddInt64(&c.misses, 1)
	return compiled, false, nil
}

// Hits returns the total number of cache hits.
func (c *ModuleCache) Hits() int64 {
	return atomic.LoadInt64(&c.hits)
}

// Misses returns the total number of cache misses.
func (c *ModuleCache) Misses() int64 {
	return atomic.LoadInt64(&c.misses)
}

// Len returns the number of cached modules.
func (c *ModuleCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.modules)
}

// Global process-wide module cache singleton.
var defaultModuleCache = NewModuleCache()

// ---------------------------------------------------------------------------
// PROVIDER IMPLEMENTATION
// ---------------------------------------------------------------------------

// Provider instantiates pure-Go Wasm & WASI sandboxes at TierL0Wasm.
type Provider struct {
	cache *ModuleCache
}

// NewProvider constructs a wazero_wasi Provider.
func NewProvider() *Provider {
	return &Provider{
		cache: defaultModuleCache,
	}
}

// Name returns the canonical provider name ("wazero_wasi").
func (p *Provider) Name() string {
	return ProviderName
}

// Tier returns sandbox.TierL0Wasm.
func (p *Provider) Tier() sandbox.Tier {
	return sandbox.TierL0Wasm
}

// Available reports whether the in-process pure-Go Wasm runtime is available (always true).
func (p *Provider) Available() bool {
	return true
}

// Create validates spec and returns a new sandbox Instance.
func (p *Provider) Create(ctx context.Context, spec sandbox.Spec) (sandbox.Instance, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	return NewInstance(spec, p.cache)
}

// ---------------------------------------------------------------------------
// INSTANCE IMPLEMENTATION
// ---------------------------------------------------------------------------

// Instance executes WebAssembly binaries and synthetic micro-tools under WASI confinement.
type Instance struct {
	spec   sandbox.Spec
	cache  *ModuleCache
	vfs    map[string][]byte
	mu     sync.Mutex
	closed bool
}

// NewInstance constructs an initialized Instance.
func NewInstance(spec sandbox.Spec, cache *ModuleCache) (*Instance, error) {
	if cache == nil {
		cache = defaultModuleCache
	}
	return &Instance{
		spec:  spec,
		cache: cache,
		vfs:   make(map[string][]byte),
	}, nil
}

// SetVFSFile registers an in-memory file within the sandbox's virtual filesystem.
func (inst *Instance) SetVFSFile(path string, data []byte) {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	clean := filepath.ToSlash(filepath.Clean(path))
	inst.vfs[clean] = data
}

// Spec returns the sandbox specification.
func (inst *Instance) Spec() sandbox.Spec {
	return inst.spec
}

// Reset clears transient state in the instance.
func (inst *Instance) Reset(ctx context.Context) error {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.closed {
		return errors.New("sandbox instance is closed")
	}
	return nil
}

// Close terminates and releases the sandbox instance.
func (inst *Instance) Close() error {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	inst.closed = true
	return nil
}

// Execute runs a command or wasm binary inside the isolated Wasm/WASI sandbox.
func (inst *Instance) Execute(ctx context.Context, req sandbox.ExecutionRequest) (sandbox.ExecutionResult, error) {
	startTime := time.Now()

	inst.mu.Lock()
	if inst.closed {
		inst.mu.Unlock()
		return sandbox.ExecutionResult{}, errors.New("sandbox instance is closed")
	}
	inst.mu.Unlock()

	// 1. Filesystem confinement & sensitive path verification
	if err := inst.checkConfinement(req); err != nil {
		durationMS := time.Since(startTime).Milliseconds()
		res := sandbox.NewExecutionResult(1, nil, []byte(err.Error()+"\n"), inst.spec.WorkspaceDir, durationMS, 0, 0)
		return res, err
	}

	// 2. Setup timeout
	timeout := time.Duration(inst.spec.TimeoutMS) * time.Millisecond
	if req.TimeoutMS > 0 {
		reqTimeout := time.Duration(req.TimeoutMS) * time.Millisecond
		if timeout == 0 || reqTimeout < timeout {
			timeout = reqTimeout
		}
	}
	execCtx := ctx
	if timeout > 0 {
		var cancel context.CancelFunc
		execCtx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	// 3. Resolve Wasm bytecode
	bytecode, err := inst.resolveBytecode(req)
	if err != nil {
		durationMS := time.Since(startTime).Milliseconds()
		res := sandbox.NewExecutionResult(1, nil, []byte(err.Error()+"\n"), inst.spec.WorkspaceDir, durationMS, 0, 0)
		return res, err
	}

	// 4. Compile or retrieve from thread-safe module cache
	mod, _, err := inst.cache.GetOrCompile(bytecode)
	if err != nil {
		durationMS := time.Since(startTime).Milliseconds()
		res := sandbox.NewExecutionResult(1, nil, []byte(fmt.Sprintf("wasm compile error: %v\n", err)), inst.spec.WorkspaceDir, durationMS, 0, 0)
		return res, err
	}

	// 5. Prepare environment & arguments
	var env []string
	if len(req.Env) > 0 {
		env = req.Env
	} else if len(inst.spec.Env) > 0 {
		env = inst.spec.Env
	}

	var argv []string
	if len(req.Argv) > 0 {
		argv = req.Argv
	} else if strings.TrimSpace(req.Command) != "" {
		argv = strings.Fields(req.Command)
	}

	// 6. Instantiate WASI context & VM
	wasi := NewWASIContext(req.Stdin, env, argv, inst.vfs)
	fuelLimit := inst.spec.FuelLimit
	if fuelLimit <= 0 {
		fuelLimit = 10_000_000 // default 10M instructions
	}

	vm, err := NewVM(mod, inst.spec.MemoryLimitBytes, fuelLimit, wasi)
	if err != nil {
		durationMS := time.Since(startTime).Milliseconds()
		res := sandbox.NewExecutionResult(1, nil, []byte(err.Error()+"\n"), inst.spec.WorkspaceDir, durationMS, 0, 0)
		return res, err
	}

	// 7. Execute VM
	exitCode, runErr := vm.Run(execCtx)
	durationMS := time.Since(startTime).Milliseconds()

	stdoutBytes := wasi.Stdout.Bytes()
	stderrBytes := wasi.Stderr.Bytes()
	if runErr != nil && len(stderrBytes) == 0 {
		stderrBytes = []byte(runErr.Error() + "\n")
	}

	res := sandbox.NewExecutionResult(
		exitCode,
		stdoutBytes,
		stderrBytes,
		inst.spec.WorkspaceDir,
		durationMS,
		vm.FuelUsed(),
		vm.MemoryBytes(),
	)

	return res, runErr
}

func (inst *Instance) resolveBytecode(req sandbox.ExecutionRequest) ([]byte, error) {
	// Case 1: Stdin provides raw Wasm binary
	if len(req.Stdin) >= 8 && bytes.Equal(req.Stdin[:4], wasmMagic) {
		return req.Stdin, nil
	}

	// Case 2: Command is a .wasm file path
	cleanCmd := strings.TrimSpace(req.Command)
	if strings.HasSuffix(cleanCmd, ".wasm") {
		return inst.readWasmFile(cleanCmd)
	}

	// Case 3: Argv[0] is a .wasm file path
	if len(req.Argv) > 0 && strings.HasSuffix(strings.TrimSpace(req.Argv[0]), ".wasm") {
		return inst.readWasmFile(strings.TrimSpace(req.Argv[0]))
	}

	// Case 4: Synthetic micro-tools (echo, math, json, cat, lint)
	return ResolveSyntheticTool(req)
}

func (inst *Instance) readWasmFile(path string) ([]byte, error) {
	slashPath := filepath.ToSlash(filepath.Clean(path))

	// Check in-memory VFS first
	inst.mu.Lock()
	data, ok := inst.vfs[slashPath]
	inst.mu.Unlock()
	if ok {
		return data, nil
	}

	// Check workspace directory
	ws := filepath.Clean(inst.spec.WorkspaceDir)
	absPath := path
	if !filepath.IsAbs(absPath) {
		absPath = filepath.Join(ws, path)
	}
	absPath = filepath.Clean(absPath)

	// Confinement check against workspace
	rel, err := filepath.Rel(ws, absPath)
	if err != nil || hasDotDot(rel) || rel == ".." {
		return nil, sandbox.NewSandboxError(sandbox.ErrLanePathEscape, fmt.Sprintf("path %q escapes workspace %q", path, ws))
	}

	// Read from disk within workspace
	b, err := os.ReadFile(absPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read wasm file %q: %w", path, err)
	}
	return b, nil
}

func (inst *Instance) checkConfinement(req sandbox.ExecutionRequest) error {
	ws := filepath.Clean(inst.spec.WorkspaceDir)

	// Check working directory
	wd := req.WorkingDir
	if strings.TrimSpace(wd) == "" {
		wd = ws
	}
	wd = filepath.Clean(wd)
	relWd, err := filepath.Rel(ws, wd)
	if err != nil || hasDotDot(relWd) || relWd == ".." {
		return sandbox.NewSandboxError(sandbox.ErrLanePathEscape, fmt.Sprintf("working directory %q escapes workspace %q", wd, ws))
	}

	// Check candidate paths in command and arguments
	candidates := extractPaths(req.Command, req.Argv)
	for _, cand := range candidates {
		if isSens, cat := isSensitivePath(cand); isSens {
			return sandbox.NewSandboxError(sandbox.ErrSecretExfiltrationAttempt, fmt.Sprintf("access to host sensitive path denied (%s): %s", cat, cand))
		}
		if filepath.IsAbs(cand) {
			rel, err := filepath.Rel(ws, filepath.Clean(cand))
			if err != nil || hasDotDot(rel) || rel == ".." {
				// Absolute host path outside workspace is strictly forbidden
				return sandbox.NewSandboxError(sandbox.ErrLanePathEscape, fmt.Sprintf("path %q escapes workspace %q", cand, ws))
			}
		}
	}

	return nil
}

func extractPaths(cmd string, argv []string) []string {
	var candidates []string
	tokens := append(strings.Fields(cmd), argv...)
	for _, t := range tokens {
		clean := strings.Trim(t, "'\"`<>|;&")
		if strings.Contains(clean, "/") || strings.Contains(clean, "\\") || strings.HasPrefix(clean, "~") || strings.HasPrefix(clean, ".") {
			candidates = append(candidates, clean)
		}
	}
	return candidates
}

func isSensitivePath(p string) (bool, string) {
	norm := strings.ToLower(filepath.ToSlash(p))
	if strings.Contains(norm, ".ssh") || strings.Contains(norm, "id_rsa") || strings.Contains(norm, "id_ed25519") {
		return true, "ssh_credentials"
	}
	if strings.Contains(norm, ".aws") || strings.Contains(norm, ".kube") || strings.Contains(norm, ".azure") {
		return true, "cloud_credentials"
	}
	if strings.Contains(norm, "/etc/passwd") || strings.Contains(norm, "/etc/shadow") || strings.Contains(norm, "/etc/sudoers") {
		return true, "system_credentials"
	}
	if strings.Contains(norm, "system32/config/sam") || strings.Contains(norm, "system32/config/system") {
		return true, "windows_system_credentials"
	}
	return false, ""
}

func hasDotDot(rel string) bool {
	return len(rel) >= 3 && rel[0] == '.' && rel[1] == '.' && (rel[2] == '/' || rel[2] == '\\')
}
