package oci

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

// ProviderName is the canonical identifier for the gVisor runsc OCI sandbox provider.
const ProviderName = "runsc"

func init() {
	sandbox.RegisterProvider(NewRunscProvider())
}

// ---------------------------------------------------------------------------
// RUNSC PROVIDER IMPLEMENTATION
// ---------------------------------------------------------------------------

// RunscProvider implements sandbox.Provider for the gVisor runsc user-space virtualization runtime.
type RunscProvider struct {
	name     string
	lookPath func(string) (string, error)
	runner   func(ctx context.Context, cmd *exec.Cmd) error
}

// NewRunscProvider constructs a standard RunscProvider.
func NewRunscProvider() *RunscProvider {
	return &RunscProvider{
		name:     ProviderName,
		lookPath: exec.LookPath,
	}
}

// NewRunscProviderWithLookPath constructs a RunscProvider with an injected LookPath lookup.
func NewRunscProviderWithLookPath(lp func(string) (string, error)) *RunscProvider {
	if lp == nil {
		lp = exec.LookPath
	}
	return &RunscProvider{
		name:     ProviderName,
		lookPath: lp,
	}
}

// Name returns "runsc".
func (p *RunscProvider) Name() string {
	return p.name
}

// Tier returns TierL2Virtual.
func (p *RunscProvider) Tier() sandbox.Tier {
	return sandbox.TierL2Virtual
}

// Available reports whether the runsc binary is present in the host system PATH.
func (p *RunscProvider) Available() bool {
	lp := p.lookPath
	if lp == nil {
		lp = exec.LookPath
	}
	_, err := lp(ProviderName)
	return err == nil
}

// Create instantiates a new RunscInstance. If runsc is not present on the host,
// it returns ErrSandboxUnavailable so callers can gracefully step down the isolation ladder.
func (p *RunscProvider) Create(ctx context.Context, spec sandbox.Spec) (sandbox.Instance, error) {
	if !p.Available() {
		return nil, sandbox.NewSandboxError(
			sandbox.ErrSandboxUnavailable,
			"runsc binary not found on host; gVisor L2 virtual sandbox unavailable",
		)
	}
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	return newRunscInstance(spec, p.lookPath, p.runner)
}

// ---------------------------------------------------------------------------
// RUNSC INSTANCE IMPLEMENTATION
// ---------------------------------------------------------------------------

// RunscInstance manages process execution within an ephemeral gVisor container bundle.
type RunscInstance struct {
	mu       sync.Mutex
	spec     sandbox.Spec
	lookPath func(string) (string, error)
	runner   func(ctx context.Context, cmd *exec.Cmd) error
	closed   bool
}

func newRunscInstance(spec sandbox.Spec, lp func(string) (string, error), runner func(context.Context, *exec.Cmd) error) (*RunscInstance, error) {
	if lp == nil {
		lp = exec.LookPath
	}
	return &RunscInstance{
		spec:     spec,
		lookPath: lp,
		runner:   runner,
	}, nil
}

// Spec returns the sandbox specification.
func (inst *RunscInstance) Spec() sandbox.Spec {
	return inst.spec
}

// Reset clears transient state in the sandbox.
func (inst *RunscInstance) Reset(ctx context.Context) error {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.closed {
		return errors.New("runsc sandbox instance is closed")
	}
	return nil
}

// Close terminates and releases the sandbox resources.
func (inst *RunscInstance) Close() error {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	inst.closed = true
	return nil
}

// Execute prepares an OCI bundle and invokes runsc to execute the requested command.
func (inst *RunscInstance) Execute(ctx context.Context, req sandbox.ExecutionRequest) (sandbox.ExecutionResult, error) {
	startTime := time.Now()

	inst.mu.Lock()
	if inst.closed {
		inst.mu.Unlock()
		return sandbox.ExecutionResult{}, errors.New("runsc sandbox instance is closed")
	}
	inst.mu.Unlock()

	// 1. Create temporary bundle directory
	bundleDir, err := os.MkdirTemp("", "fak-runsc-bundle-*")
	if err != nil {
		durationMS := time.Since(startTime).Milliseconds()
		return sandbox.NewExecutionResult(1, nil, []byte(err.Error()+"\n"), inst.spec.WorkspaceDir, durationMS, 0, 0), err
	}
	defer func() {
		_ = os.RemoveAll(bundleDir)
	}()

	// 2. Generate OCI bundle (config.json + rootfs mounts)
	if err := GenerateBundle(inst.spec, req, bundleDir); err != nil {
		durationMS := time.Since(startTime).Milliseconds()
		return sandbox.NewExecutionResult(1, nil, []byte(err.Error()+"\n"), inst.spec.WorkspaceDir, durationMS, 0, 0), err
	}

	// 3. Configure execution timeout
	timeout := time.Duration(inst.spec.TimeoutMS) * time.Millisecond
	if req.TimeoutMS > 0 {
		reqTimeout := time.Duration(req.TimeoutMS) * time.Millisecond
		if timeout == 0 || reqTimeout < timeout {
			timeout = reqTimeout
		}
	}
	runCtx := ctx
	if timeout > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	// 4. Resolve runsc binary path
	runscBin := ProviderName
	if inst.lookPath != nil {
		if resolved, err := inst.lookPath(ProviderName); err == nil {
			runscBin = resolved
		}
	}

	containerID := fmt.Sprintf("fak-runsc-%d-%d", os.Getpid(), time.Now().UnixNano())

	// Build runsc command: runsc --rootless --network=none run --bundle <bundleDir> <id>
	args := []string{"--rootless", "--network=none", "run", "--bundle", bundleDir, containerID}
	cmd := exec.CommandContext(runCtx, runscBin, args...)
	cmd.Dir = bundleDir

	if len(req.Stdin) > 0 {
		cmd.Stdin = bytes.NewReader(req.Stdin)
	}

	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	// 5. Execute command
	var runErr error
	if inst.runner != nil {
		runErr = inst.runner(runCtx, cmd)
	} else {
		runErr = cmd.Run()
	}

	durationMS := time.Since(startTime).Milliseconds()

	exitCode := 0
	if runErr != nil {
		exitCode = -1
		var ee *exec.ExitError
		if errors.As(runErr, &ee) {
			exitCode = ee.ExitCode()
		}
	}

	res := sandbox.NewExecutionResult(
		exitCode,
		stdoutBuf.Bytes(),
		stderrBuf.Bytes(),
		inst.spec.WorkspaceDir,
		durationMS,
		0,
		0,
	)

	return res, runErr
}

// Ensure interface conformance.
var (
	_ sandbox.Provider = (*RunscProvider)(nil)
	_ sandbox.Instance = (*RunscInstance)(nil)
)
