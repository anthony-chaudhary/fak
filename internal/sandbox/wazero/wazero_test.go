package wazero_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sandbox"
	"github.com/anthony-chaudhary/fak/internal/sandbox/wazero"
)

// TestWasmSandboxFuelAndMemoryConfinement tests:
// 1. Fuel exhaustion on infinite loop module.
// 2. Memory cap enforcement on linear memory growth.
// 3. Host filesystem access inaccessibility (fail-closed).
// 4. 1,000 isolated invocations in <50ms total.
func TestWasmSandboxFuelAndMemoryConfinement(t *testing.T) {
	ctx := context.Background()
	wsDir := t.TempDir()

	prov := wazero.NewProvider()
	if !prov.Available() {
		t.Fatalf("wazero provider reported unavailable")
	}

	// -----------------------------------------------------------------------
	// 1. Infinite loop fuel exhaustion
	// -----------------------------------------------------------------------
	specFuel := sandbox.Spec{
		Tier:             sandbox.TierL0Wasm,
		WorkspaceDir:     wsDir,
		EgressPolicy:     sandbox.EgressBlocked,
		FuelLimit:        5000,
		MemoryLimitBytes: 1024 * 1024,
	}
	instFuel, err := prov.Create(ctx, specFuel)
	if err != nil {
		t.Fatalf("failed to create fuel instance: %v", err)
	}
	defer instFuel.Close()

	infiniteWasm := wazero.BuildInfiniteLoopModule()
	reqFuel := sandbox.ExecutionRequest{
		Command: "infinite.wasm",
		Stdin:   infiniteWasm,
	}
	resFuel, fuelErr := instFuel.Execute(ctx, reqFuel)
	if fuelErr == nil {
		t.Fatalf("expected fuel exhaustion error, got nil result: %+v", resFuel)
	}
	if !sandbox.IsSandboxError(fuelErr, "FUEL_EXHAUSTED") && !strings.Contains(fuelErr.Error(), "FUEL_EXHAUSTED") {
		t.Fatalf("expected FUEL_EXHAUSTED error token, got: %v", fuelErr)
	}
	if resFuel.FuelUsed < 5000 {
		t.Fatalf("expected FuelUsed >= 5000, got: %d", resFuel.FuelUsed)
	}

	// -----------------------------------------------------------------------
	// 2. Linear memory cap enforcement
	// -----------------------------------------------------------------------
	// Limit memory to 2 pages (128 KB = 131072 bytes)
	specMem := sandbox.Spec{
		Tier:             sandbox.TierL0Wasm,
		WorkspaceDir:     wsDir,
		EgressPolicy:     sandbox.EgressBlocked,
		FuelLimit:        1_000_000,
		MemoryLimitBytes: 128 * 1024,
	}
	instMem, err := prov.Create(ctx, specMem)
	if err != nil {
		t.Fatalf("failed to create mem instance: %v", err)
	}
	defer instMem.Close()

	// Module attempts to grow memory by 10 pages (> 2 pages limit)
	growWasm := wazero.BuildMemoryGrowModule(10)
	reqMem := sandbox.ExecutionRequest{
		Command: "grow.wasm",
		Stdin:   growWasm,
	}
	resMem, err := instMem.Execute(ctx, reqMem)
	if err != nil {
		t.Fatalf("memory grow execution failed: %v", err)
	}
	// The module handles -1 from memory.grow by writing MEMORY_GROW_FAILED and exiting 42
	if resMem.ExitCode != 42 {
		t.Fatalf("expected exit code 42 (growth cap failed), got %d, stdout: %q, stderr: %q",
			resMem.ExitCode, string(resMem.Stdout), string(resMem.Stderr))
	}
	if !strings.Contains(string(resMem.Stderr), "MEMORY_GROW_FAILED") {
		t.Fatalf("expected MEMORY_GROW_FAILED in stderr, got: %q", string(resMem.Stderr))
	}

	// -----------------------------------------------------------------------
	// 3. Host filesystem is inaccessible (fails closed)
	// -----------------------------------------------------------------------
	specFS := sandbox.Spec{
		Tier:             sandbox.TierL0Wasm,
		WorkspaceDir:     wsDir,
		EgressPolicy:     sandbox.EgressBlocked,
		FuelLimit:        100_000,
		MemoryLimitBytes: 1024 * 1024,
	}
	instFS, err := prov.Create(ctx, specFS)
	if err != nil {
		t.Fatalf("failed to create fs instance: %v", err)
	}
	defer instFS.Close()

	// Test A: sensitive path denied
	reqSens := sandbox.ExecutionRequest{
		Command: "cat /etc/passwd",
	}
	_, sensErr := instFS.Execute(ctx, reqSens)
	if sensErr == nil {
		t.Fatalf("expected error accessing host /etc/passwd, got nil")
	}
	if !sandbox.IsSandboxError(sensErr, sandbox.ErrSecretExfiltrationAttempt) && !sandbox.IsSandboxError(sensErr, sandbox.ErrLanePathEscape) {
		t.Fatalf("expected ErrSecretExfiltrationAttempt or ErrLanePathEscape, got: %v", sensErr)
	}

	// Test B: traversal outside workspace denied
	reqEscape := sandbox.ExecutionRequest{
		Command: "../escape.wasm",
	}
	_, escErr := instFS.Execute(ctx, reqEscape)
	if escErr == nil {
		t.Fatalf("expected error accessing outside workspace, got nil")
	}
	if !sandbox.IsSandboxError(escErr, sandbox.ErrLanePathEscape) {
		t.Fatalf("expected ErrLanePathEscape, got: %v", escErr)
	}

	// -----------------------------------------------------------------------
	// 4. 1,000 isolated invocations in <50ms total
	// -----------------------------------------------------------------------
	specBench := sandbox.Spec{
		Tier:             sandbox.TierL0Wasm,
		WorkspaceDir:     wsDir,
		EgressPolicy:     sandbox.EgressBlocked,
		FuelLimit:        100_000,
		MemoryLimitBytes: 1024 * 1024,
	}
	instBench, err := prov.Create(ctx, specBench)
	if err != nil {
		t.Fatalf("failed to create bench instance: %v", err)
	}
	defer instBench.Close()

	echoBytecode := wazero.BuildEchoModule("bench\n")
	reqBench := sandbox.ExecutionRequest{
		Command: "echo.wasm",
		Stdin:   echoBytecode,
	}

	// Warmup once
	if _, err := instBench.Execute(ctx, reqBench); err != nil {
		t.Fatalf("warmup invocation failed: %v", err)
	}

	const iterations = 1000
	start := time.Now()
	for i := 0; i < iterations; i++ {
		res, err := instBench.Execute(ctx, reqBench)
		if err != nil {
			t.Fatalf("invocation %d failed: %v", i, err)
		}
		if res.ExitCode != 0 {
			t.Fatalf("invocation %d exited with %d", i, res.ExitCode)
		}
	}
	duration := time.Since(start)
	t.Logf("1,000 isolated invocations completed in %v (avg %v/op)", duration, duration/iterations)

	if duration >= 50*time.Millisecond {
		t.Fatalf("expected 1,000 invocations in <50ms, took %v", duration)
	}
}

// TestCompiledModuleCache tests thread-safe compiled module cache hits and reuses.
func TestCompiledModuleCache(t *testing.T) {
	cache := wazero.NewModuleCache()
	if cache.Len() != 0 {
		t.Fatalf("expected empty cache, got len=%d", cache.Len())
	}

	bytecode := wazero.BuildEchoModule("hello cache\n")

	// First compile -> miss
	mod1, hit1, err := cache.GetOrCompile(bytecode)
	if err != nil {
		t.Fatalf("initial compile failed: %v", err)
	}
	if hit1 {
		t.Fatalf("expected cache miss on first compile")
	}
	if cache.Misses() != 1 || cache.Hits() != 0 {
		t.Fatalf("expected 1 miss, 0 hits; got misses=%d, hits=%d", cache.Misses(), cache.Hits())
	}

	// Second compile -> hit
	mod2, hit2, err := cache.GetOrCompile(bytecode)
	if err != nil {
		t.Fatalf("second compile failed: %v", err)
	}
	if !hit2 {
		t.Fatalf("expected cache hit on second compile")
	}
	if mod1 != mod2 {
		t.Fatalf("expected identical CompiledModule pointer from cache")
	}
	if cache.Hits() != 1 {
		t.Fatalf("expected 1 hit; got hits=%d", cache.Hits())
	}
}

// TestWasiStreamsCaptureAndNormalization tests stdout/stderr stream separation
// and output normalization (ANSI stripping, CRLF normalization, /workspace path canonicalization).
func TestWasiStreamsCaptureAndNormalization(t *testing.T) {
	ctx := context.Background()
	wsDir := filepath.Join(os.TempDir(), "test_wazero_ws")
	_ = os.MkdirAll(wsDir, 0755)
	defer os.RemoveAll(wsDir)

	prov := wazero.NewProvider()
	spec := sandbox.Spec{
		Tier:             sandbox.TierL0Wasm,
		WorkspaceDir:     wsDir,
		EgressPolicy:     sandbox.EgressBlocked,
		FuelLimit:        100_000,
		MemoryLimitBytes: 1024 * 1024,
	}
	inst, err := prov.Create(ctx, spec)
	if err != nil {
		t.Fatalf("failed to create instance: %v", err)
	}
	defer inst.Close()

	// Construct module emitting ANSI escape codes, CRLF, and the host workspace path
	stdoutRaw := "\x1b[32mSuccess in " + wsDir + "\x1b[0m\r\nline 2   \r\n"
	stderrRaw := "\x1b[31mWarning in " + wsDir + "\x1b[0m\r\n"

	modBytes := wazero.BuildStdoutStderrModule(stdoutRaw, stderrRaw)
	req := sandbox.ExecutionRequest{
		Command: "test_streams.wasm",
		Stdin:   modBytes,
	}

	res, err := inst.Execute(ctx, req)
	if err != nil {
		t.Fatalf("execution failed: %v", err)
	}

	// Verify raw stream capture
	if string(res.Stdout) != stdoutRaw {
		t.Fatalf("stdout mismatch: got %q, want %q", string(res.Stdout), stdoutRaw)
	}
	if string(res.Stderr) != stderrRaw {
		t.Fatalf("stderr mismatch: got %q, want %q", string(res.Stderr), stderrRaw)
	}

	// Verify normalized stdout
	normOut := string(res.NormalizedStdout)
	if strings.Contains(normOut, "\x1b[") {
		t.Fatalf("normalized stdout contains ANSI escapes: %q", normOut)
	}
	if strings.Contains(normOut, "\r") {
		t.Fatalf("normalized stdout contains CR: %q", normOut)
	}
	if strings.Contains(normOut, wsDir) {
		t.Fatalf("normalized stdout contains raw workspace dir: %q", normOut)
	}
	if !strings.Contains(normOut, "/workspace") {
		t.Fatalf("normalized stdout missing canonical /workspace: %q", normOut)
	}

	// Verify normalized stderr
	normErr := string(res.NormalizedStderr)
	if strings.Contains(normErr, "\x1b[") {
		t.Fatalf("normalized stderr contains ANSI escapes: %q", normErr)
	}
	if strings.Contains(normErr, wsDir) {
		t.Fatalf("normalized stderr contains raw workspace dir: %q", normErr)
	}
	if !strings.Contains(normErr, "/workspace") {
		t.Fatalf("normalized stderr missing canonical /workspace: %q", normErr)
	}
}

// TestSyntheticMicroTools verifies pure-Go synthetic micro-tools (echo, math, json, cat, lint).
func TestSyntheticMicroTools(t *testing.T) {
	ctx := context.Background()
	wsDir := t.TempDir()

	prov := wazero.NewProvider()
	spec := sandbox.Spec{
		Tier:             sandbox.TierL0Wasm,
		WorkspaceDir:     wsDir,
		EgressPolicy:     sandbox.EgressBlocked,
		FuelLimit:        500_000,
		MemoryLimitBytes: 1024 * 1024,
	}
	inst, err := prov.Create(ctx, spec)
	if err != nil {
		t.Fatalf("failed to create instance: %v", err)
	}
	defer inst.Close()

	// 1. Echo micro-tool
	resEcho, err := inst.Execute(ctx, sandbox.ExecutionRequest{
		Command: "echo Hello Micro Tool",
	})
	if err != nil {
		t.Fatalf("echo execution failed: %v", err)
	}
	if !strings.Contains(string(resEcho.Stdout), "Hello Micro Tool") {
		t.Fatalf("echo output mismatch: %q", string(resEcho.Stdout))
	}

	// 2. Math micro-tool
	resMath, err := inst.Execute(ctx, sandbox.ExecutionRequest{
		Command: "math 35 + 7",
	})
	if err != nil {
		t.Fatalf("math execution failed: %v", err)
	}
	if strings.TrimSpace(string(resMath.Stdout)) != "42" {
		t.Fatalf("math output mismatch: %q", string(resMath.Stdout))
	}

	// 3. JSON validator micro-tool
	resJSON, err := inst.Execute(ctx, sandbox.ExecutionRequest{
		Command: "json",
		Stdin:   []byte(`{"status": "ok", "tier": "l0_wasm"}`),
	})
	if err != nil {
		t.Fatalf("json execution failed: %v", err)
	}
	if !strings.Contains(string(resJSON.Stdout), "status") {
		t.Fatalf("json output mismatch: %q", string(resJSON.Stdout))
	}

	// 4. Cat micro-tool
	resCat, err := inst.Execute(ctx, sandbox.ExecutionRequest{
		Command: "cat",
		Stdin:   []byte("piped input stream\n"),
	})
	if err != nil {
		t.Fatalf("cat execution failed: %v", err)
	}
	if string(resCat.Stdout) != "piped input stream\n" {
		t.Fatalf("cat output mismatch: %q", string(resCat.Stdout))
	}

	// 5. Lint micro-tool
	resLint, err := inst.Execute(ctx, sandbox.ExecutionRequest{
		Command: "lint",
		Stdin:   []byte("func CleanCode() {}"),
	})
	if err != nil {
		t.Fatalf("lint execution failed: %v", err)
	}
	if !strings.Contains(string(resLint.Stdout), "0 errors") {
		t.Fatalf("lint output mismatch: %q", string(resLint.Stdout))
	}
}

// TestProcExitCode tests clean propagation of WASI proc_exit codes.
func TestProcExitCode(t *testing.T) {
	ctx := context.Background()
	wsDir := t.TempDir()

	prov := wazero.NewProvider()
	spec := sandbox.Spec{
		Tier:             sandbox.TierL0Wasm,
		WorkspaceDir:     wsDir,
		EgressPolicy:     sandbox.EgressBlocked,
		FuelLimit:        100_000,
		MemoryLimitBytes: 1024 * 1024,
	}
	inst, err := prov.Create(ctx, spec)
	if err != nil {
		t.Fatalf("failed to create instance: %v", err)
	}
	defer inst.Close()

	exitModule := wazero.BuildExitModule(17)
	res, err := inst.Execute(ctx, sandbox.ExecutionRequest{
		Command: "exit.wasm",
		Stdin:   exitModule,
	})
	if err != nil {
		t.Fatalf("exit execution failed: %v", err)
	}
	if res.ExitCode != 17 {
		t.Fatalf("expected exit code 17, got %d", res.ExitCode)
	}
}

// TestRegistryResolution confirms registration in DefaultRegistry under TierL0Wasm.
func TestRegistryResolution(t *testing.T) {
	p, err := sandbox.DefaultRegistry().ResolveTier(sandbox.TierL0Wasm)
	if err != nil {
		t.Fatalf("failed to resolve TierL0Wasm from DefaultRegistry: %v", err)
	}
	if p.Name() != wazero.ProviderName {
		t.Fatalf("expected provider name %q, got %q", wazero.ProviderName, p.Name())
	}
	if p.Tier() != sandbox.TierL0Wasm {
		t.Fatalf("expected tier %v, got %v", sandbox.TierL0Wasm, p.Tier())
	}
}
