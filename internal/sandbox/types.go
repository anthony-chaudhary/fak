// Package sandbox defines the tiered isolation ladder (L0 Wasm, L1 Host Native,
// L2 Virtual/gVisor), low-ego OCI/WASI/MCP execution contracts, and gym lifecycle invariants.
package sandbox

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ---------------------------------------------------------------------------
// TIER DEFINITIONS & ISOLATION LADDER
// ---------------------------------------------------------------------------

// Tier denotes the isolation rung in the sandbox ladder.
type Tier string

const (
	// TierL0Wasm represents in-process WebAssembly/WASI execution (<1ms startup,
	// memory-isolated, capability-gated, deterministic).
	TierL0Wasm Tier = "l0_wasm"

	// TierL1NativeOS represents host-native OS sandboxing (<5ms startup,
	// Linux namespaces/cgroups/Landlock/seccomp, macOS seatbelt, Windows AppContainer).
	TierL1NativeOS Tier = "l1_native_os"

	// TierL2Virtual represents virtualized kernel isolation (~20-50ms cold, <10ms restore,
	// gVisor user-space kernel or microVM like Firecracker/Cloud-Hypervisor).
	TierL2Virtual Tier = "l2_virtual"
)

// String returns the string value of the tier.
func (t Tier) String() string {
	return string(t)
}

// Valid reports whether the tier is one of the recognized ladder rungs.
func (t Tier) Valid() bool {
	switch t {
	case TierL0Wasm, TierL1NativeOS, TierL2Virtual:
		return true
	default:
		return false
	}
}

// IsolationLevel returns the numeric isolation rank (0=Wasm, 1=Native OS, 2=Virtual).
// Returns -1 for invalid or unclassified tiers.
func (t Tier) IsolationLevel() int {
	switch t {
	case TierL0Wasm:
		return 0
	case TierL1NativeOS:
		return 1
	case TierL2Virtual:
		return 2
	default:
		return -1
	}
}

// RequiresVirtualization reports whether the tier mandates hypervisor or user-space kernel virtualization.
func (t Tier) RequiresVirtualization() bool {
	return t == TierL2Virtual
}

// ParseTier parses a string into a recognized Tier or returns an error.
func ParseTier(s string) (Tier, error) {
	switch Tier(strings.TrimSpace(s)) {
	case TierL0Wasm:
		return TierL0Wasm, nil
	case TierL1NativeOS:
		return TierL1NativeOS, nil
	case TierL2Virtual:
		return TierL2Virtual, nil
	default:
		return "", fmt.Errorf("invalid sandbox tier: %q", s)
	}
}

// ---------------------------------------------------------------------------
// EGRESS POLICIES & CAPABILITY ENVELOPE
// ---------------------------------------------------------------------------

// EgressPolicy defines outbound network containment rules.
type EgressPolicy string

const (
	// EgressBlocked represents strict default-deny network containment (no outbound sockets).
	EgressBlocked EgressPolicy = "blocked"

	// EgressLoopback permits communication with loopback (127.0.0.1) services only.
	EgressLoopback EgressPolicy = "loopback"

	// EgressAllowlist restricts outbound egress to strictly audited hostnames/IPs.
	EgressAllowlist EgressPolicy = "allowlist"
)

// CapabilityEnvelope specifies negotiated permissions and lane tree boundaries.
type CapabilityEnvelope struct {
	LaneTree     []string         `json:"lane_tree,omitempty"`
	Capabilities []abi.Capability `json:"capabilities,omitempty"`
	AllowNetwork bool             `json:"allow_network,omitempty"`
	AllowIPC     bool             `json:"allow_ipc,omitempty"`
}

// HasCapability reports whether the envelope contains the requested capability.
func (e CapabilityEnvelope) HasCapability(cap abi.Capability) bool {
	for _, c := range e.Capabilities {
		if c == cap {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// STANDARD REFUSAL TOKENS & ERRORS
// ---------------------------------------------------------------------------

const (
	// ErrSandboxUnavailable indicates the requested sandbox tier/provider is missing or unhealthy.
	ErrSandboxUnavailable = "SANDBOX_UNAVAILABLE"

	// ErrLanePathEscape indicates a filesystem traversal outside the designated workspace.
	ErrLanePathEscape = "LANE_PATH_ESCAPE"

	// ErrEgressBlocked indicates an unauthorized network egress attempt under fail-closed policy.
	ErrEgressBlocked = "EGRESS_BLOCKED"

	// ErrSiblingLaneTouch indicates an illegal file mutation across DOS sibling lane trees.
	ErrSiblingLaneTouch = "SIBLING_LANE_TOUCH"

	// ErrSecretExfiltrationAttempt indicates an environment or secret exfiltration attempt was caught.
	ErrSecretExfiltrationAttempt = "SECRET_EXFILTRATION_ATTEMPT"
)

// SandboxError represents a typed structured refusal or operational failure in the sandbox kernel.
type SandboxError struct {
	Token   string `json:"token"`
	Message string `json:"message"`
	Err     error  `json:"-"`
}

func (e *SandboxError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %s (%v)", e.Token, e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.Token, e.Message)
}

func (e *SandboxError) Unwrap() error {
	return e.Err
}

// NewSandboxError constructs a typed SandboxError for a closed refusal token.
func NewSandboxError(token, message string) *SandboxError {
	return &SandboxError{Token: token, Message: message}
}

// WrapSandboxError wraps an underlying error with a refusal token and explanation.
func WrapSandboxError(token, message string, err error) *SandboxError {
	return &SandboxError{Token: token, Message: message, Err: err}
}

// IsSandboxError reports whether the error or any wrapped cause carries the specified token.
func IsSandboxError(err error, token string) bool {
	var se *SandboxError
	if errors.As(err, &se) {
		return se.Token == token
	}
	return false
}

// ---------------------------------------------------------------------------
// SPECIFICATION & REQUEST/RESULT CONTRACTS
// ---------------------------------------------------------------------------

// Spec specifies the complete configuration for sandbox instantiation.
type Spec struct {
	Tier             Tier             `json:"tier"`
	Rootfs           string           `json:"rootfs,omitempty"`
	WorkspaceDir     string           `json:"workspace_dir"`
	LaneTree         []string         `json:"lane_tree,omitempty"`
	ReadOnlyPaths    []string         `json:"read_only_paths,omitempty"`
	WritablePaths    []string         `json:"writable_paths,omitempty"`
	MemoryLimitBytes int64            `json:"memory_limit_bytes,omitempty"`
	CPULimitPercent  int              `json:"cpu_limit_percent,omitempty"`
	FuelLimit        int64            `json:"fuel_limit,omitempty"`
	TimeoutMS        int64            `json:"timeout_ms,omitempty"`
	Env              []string         `json:"env,omitempty"`
	EgressPolicy     EgressPolicy     `json:"egress_policy"`
	Capabilities     []abi.Capability `json:"capabilities,omitempty"`
}

// Validate checks whether the Spec conforms to structural integrity requirements.
func (s Spec) Validate() error {
	if !s.Tier.Valid() {
		return NewSandboxError(ErrSandboxUnavailable, fmt.Sprintf("invalid tier: %q", s.Tier))
	}
	if strings.TrimSpace(s.WorkspaceDir) == "" {
		return NewSandboxError(ErrLanePathEscape, "workspace_dir is required")
	}
	if s.MemoryLimitBytes < 0 {
		return fmt.Errorf("memory_limit_bytes must be non-negative: %d", s.MemoryLimitBytes)
	}
	if s.CPULimitPercent < 0 || s.CPULimitPercent > 100 {
		return fmt.Errorf("cpu_limit_percent must be between 0 and 100: %d", s.CPULimitPercent)
	}
	if s.FuelLimit < 0 {
		return fmt.Errorf("fuel_limit must be non-negative: %d", s.FuelLimit)
	}
	if s.TimeoutMS < 0 {
		return fmt.Errorf("timeout_ms must be non-negative: %d", s.TimeoutMS)
	}
	if s.EgressPolicy == "" {
		return fmt.Errorf("egress_policy is required")
	}
	return nil
}

// ExecutionRequest represents a command dispatch within an active sandbox instance.
type ExecutionRequest struct {
	Command    string   `json:"command"`
	Argv       []string `json:"argv,omitempty"`
	Stdin      []byte   `json:"stdin,omitempty"`
	Env        []string `json:"env,omitempty"`
	TimeoutMS  int64    `json:"timeout_ms,omitempty"`
	WorkingDir string   `json:"working_dir,omitempty"`
}

// Audit records an isolated event, violation attempt, or measurement.
type Audit struct {
	TimestampMS int64  `json:"timestamp_ms"`
	Type        string `json:"type"`
	Message     string `json:"message"`
}

// ExecutionResult captures the output, accounting, and normalized streams of an execution.
type ExecutionResult struct {
	ExitCode         int     `json:"exit_code"`
	Stdout           []byte  `json:"stdout"`
	Stderr           []byte  `json:"stderr"`
	NormalizedStdout []byte  `json:"normalized_stdout"`
	NormalizedStderr []byte  `json:"normalized_stderr"`
	DurationMS       int64   `json:"duration_ms"`
	FuelUsed         int64   `json:"fuel_used"`
	MemoryBytes      int64   `json:"memory_bytes"`
	Audits           []Audit `json:"audits,omitempty"`
}

// ---------------------------------------------------------------------------
// OUTPUT NORMALIZATION HELPER
// ---------------------------------------------------------------------------

var ansiEscapeRE = regexp.MustCompile(`\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)`)

// NormalizeOutput sanitizes raw execution stream bytes for deterministic cryptographic comparison:
// 1. Strips terminal ANSI escape sequences.
// 2. Normalizes carriage-return line breaks (\r\n and \r -> \n).
// 3. Replaces host-specific workspace directory paths with canonical "/workspace".
// 4. Trims trailing whitespace from lines while preserving line structure.
func NormalizeOutput(raw []byte, workspaceDir string) []byte {
	if len(raw) == 0 {
		return []byte{}
	}

	// 1. Strip ANSI escape sequences.
	cleaned := ansiEscapeRE.ReplaceAll(raw, nil)

	// 2. Normalize CRLF and CR to LF.
	s := string(cleaned)
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")

	// 3. Canonicalize workspace path to /workspace.
	if strings.TrimSpace(workspaceDir) != "" {
		cleanWS := filepath.Clean(workspaceDir)
		wsForward := filepath.ToSlash(cleanWS)
		wsNative := cleanWS

		s = strings.ReplaceAll(s, wsNative, "/workspace")
		if wsForward != wsNative {
			s = strings.ReplaceAll(s, wsForward, "/workspace")
		}
	}

	// 4. Strip trailing whitespace per line and overall trailing newlines.
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	s = strings.Join(lines, "\n")
	s = strings.TrimRight(s, "\n")

	return []byte(s)
}

// NewExecutionResult constructs an ExecutionResult with automatic output normalization.
func NewExecutionResult(exitCode int, stdout, stderr []byte, workspaceDir string, durationMS, fuelUsed, memoryBytes int64) ExecutionResult {
	return ExecutionResult{
		ExitCode:         exitCode,
		Stdout:           stdout,
		Stderr:           stderr,
		NormalizedStdout: NormalizeOutput(stdout, workspaceDir),
		NormalizedStderr: NormalizeOutput(stderr, workspaceDir),
		DurationMS:       durationMS,
		FuelUsed:         fuelUsed,
		MemoryBytes:      memoryBytes,
	}
}

// ---------------------------------------------------------------------------
// CORE RUNTIME INTERFACES
// ---------------------------------------------------------------------------

// Instance represents a running or paused sandbox environment.
type Instance interface {
	Execute(ctx context.Context, req ExecutionRequest) (ExecutionResult, error)
	Reset(ctx context.Context) error
	Close() error
	Spec() Spec
}

// SnapshotHandle represents an immutable Copy-on-Write state snapshot.
type SnapshotHandle interface {
	ID() string
	Restore(ctx context.Context) error
	Release() error
}

// SnapshotableInstance extends Instance with snapshot creation capability.
type SnapshotableInstance interface {
	Instance
	Snapshot(ctx context.Context) (SnapshotHandle, error)
}

// Provider instantiates and manages the lifecycle of sandboxes at a designated tier.
type Provider interface {
	Name() string
	Tier() Tier
	Available() bool
	Create(ctx context.Context, spec Spec) (Instance, error)
}
