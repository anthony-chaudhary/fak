package sandbox

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
)

// osConfinement abstracts platform-specific OS isolation (Win32 JobObjects, Linux Landlock, Darwin Seatbelt).
type osConfinement interface {
	PrepareCommand(cmd *exec.Cmd, req ExecutionRequest) error
	OnProcessStart(pid int) error
	PostProcess(res *ExecutionResult) error
	Close() error
}

// ---------------------------------------------------------------------------
// L1 PROVIDER IMPLEMENTATION
// ---------------------------------------------------------------------------

// L1Provider instantiates Host-Native OS confined sandboxes.
type L1Provider struct {
	name string
}

// NewL1Provider returns an initialized L1Provider.
func NewL1Provider() *L1Provider {
	return &L1Provider{name: "l1_native_os"}
}

// Name returns the provider name.
func (p *L1Provider) Name() string {
	return p.name
}

// Tier returns TierL1NativeOS.
func (p *L1Provider) Tier() Tier {
	return TierL1NativeOS
}

// Available reports whether L1 host-native OS sandboxing is available on this platform.
func (p *L1Provider) Available() bool {
	return true
}

// Create instantiates an L1Instance configured according to spec.
func (p *L1Provider) Create(ctx context.Context, spec Spec) (Instance, error) {
	if err := spec.Validate(); err != nil {
		return nil, err
	}
	return newL1Instance(spec)
}

// ---------------------------------------------------------------------------
// L1 INSTANCE IMPLEMENTATION
// ---------------------------------------------------------------------------

// L1Instance manages process execution under host-native OS confinement.
type L1Instance struct {
	mu     sync.Mutex
	spec   Spec
	osConf osConfinement
	closed bool
}

func newL1Instance(spec Spec) (*L1Instance, error) {
	osConf, err := newOSConfinement(spec)
	if err != nil {
		return nil, WrapSandboxError(ErrSandboxUnavailable, "failed to initialize OS confinement", err)
	}
	return &L1Instance{
		spec:   spec,
		osConf: osConf,
	}, nil
}

// Spec returns the sandbox specification.
func (inst *L1Instance) Spec() Spec {
	return inst.spec
}

// Reset clears transient state.
func (inst *L1Instance) Reset(ctx context.Context) error {
	return nil
}

// Close terminates and releases the sandbox resources.
func (inst *L1Instance) Close() error {
	inst.mu.Lock()
	defer inst.mu.Unlock()
	if inst.closed {
		return nil
	}
	inst.closed = true
	if inst.osConf != nil {
		return inst.osConf.Close()
	}
	return nil
}

type l1SnapshotHandle struct {
	id string
}

func (s *l1SnapshotHandle) ID() string                        { return s.id }
func (s *l1SnapshotHandle) Restore(ctx context.Context) error { return nil }
func (s *l1SnapshotHandle) Release() error                    { return nil }

// Snapshot creates a snapshot handle for the active instance.
func (inst *L1Instance) Snapshot(ctx context.Context) (SnapshotHandle, error) {
	return &l1SnapshotHandle{id: fmt.Sprintf("l1-snap-%d", time.Now().UnixNano())}, nil
}

// Execute dispatches a command inside the L1 sandbox.
func (inst *L1Instance) Execute(ctx context.Context, req ExecutionRequest) (ExecutionResult, error) {
	startTime := time.Now()

	workingDir := req.WorkingDir
	if strings.TrimSpace(workingDir) == "" {
		workingDir = inst.spec.WorkspaceDir
	}
	workingDir = filepath.Clean(workingDir)

	// 1. Confinement checks (working directory and operand path checks)
	audit, err := inst.checkConfinement(workingDir, req)
	if err != nil {
		durationMS := time.Since(startTime).Milliseconds()
		res := NewExecutionResult(1, nil, []byte(err.Error()+"\n"), inst.spec.WorkspaceDir, durationMS, 0, 0)
		if audit != nil {
			res.Audits = append(res.Audits, *audit)
		}
		return res, err
	}

	// 2. Set timeout
	timeout := time.Duration(inst.spec.TimeoutMS) * time.Millisecond
	if req.TimeoutMS > 0 {
		reqTimeout := time.Duration(req.TimeoutMS) * time.Millisecond
		if timeout == 0 || reqTimeout < timeout {
			timeout = reqTimeout
		}
	}
	runCtx := ctx
	var cancel context.CancelFunc
	if timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	// 3. Prepare exec.Cmd
	cmd := prepareCommand(runCtx, req)
	cmd.Dir = workingDir

	// Setup environment
	if len(req.Env) > 0 {
		cmd.Env = req.Env
	} else if len(inst.spec.Env) > 0 {
		cmd.Env = inst.spec.Env
	} else {
		cmd.Env = os.Environ()
	}

	// Setup stdin
	if len(req.Stdin) > 0 {
		cmd.Stdin = bytes.NewReader(req.Stdin)
	}

	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	// 4. Attach OS confinement attributes
	if inst.osConf != nil {
		if err := inst.osConf.PrepareCommand(cmd, req); err != nil {
			return ExecutionResult{}, err
		}
	}

	// 5. Start process
	startErr := cmd.Start()
	if startErr != nil {
		durationMS := time.Since(startTime).Milliseconds()
		return NewExecutionResult(-1, stdoutBuf.Bytes(), stderrBuf.Bytes(), inst.spec.WorkspaceDir, durationMS, 0, 0), startErr
	}

	// 6. Bind to OS confinement
	if inst.osConf != nil && cmd.Process != nil {
		_ = inst.osConf.OnProcessStart(cmd.Process.Pid)
	}

	// 7. Wait for completion
	waitErr := cmd.Wait()
	durationMS := time.Since(startTime).Milliseconds()

	exitCode := 0
	if waitErr != nil {
		exitCode = -1
		var ee *exec.ExitError
		if errors.As(waitErr, &ee) {
			exitCode = ee.ExitCode()
		}
	}

	res := NewExecutionResult(exitCode, stdoutBuf.Bytes(), stderrBuf.Bytes(), inst.spec.WorkspaceDir, durationMS, 0, 0)

	// 8. Post-process stats from OS confinement
	if inst.osConf != nil {
		_ = inst.osConf.PostProcess(&res)
	}

	return res, waitErr
}

// ---------------------------------------------------------------------------
// LANE TREE & SENSITIVE PATH CONFINEMENT LOGIC
// ---------------------------------------------------------------------------

func (inst *L1Instance) checkConfinement(workingDir string, req ExecutionRequest) (*Audit, error) {
	ws := filepath.Clean(inst.spec.WorkspaceDir)

	// 1. Check WorkingDir against WorkspaceDir
	relWd, err := filepath.Rel(ws, workingDir)
	if err != nil || hasDotDotPrefix(relWd) || relWd == ".." {
		audit := Audit{
			TimestampMS: time.Now().UnixMilli(),
			Type:        ErrLanePathEscape,
			Message:     fmt.Sprintf("working directory %q escapes workspace %q", workingDir, ws),
		}
		return &audit, NewSandboxError(ErrLanePathEscape, audit.Message)
	}

	// Check WorkingDir against LaneTree
	if len(inst.spec.LaneTree) > 0 && relWd != "." {
		if !isLaneAllowed(inst.spec.LaneTree, relWd) {
			audit := Audit{
				TimestampMS: time.Now().UnixMilli(),
				Type:        ErrSiblingLaneTouch,
				Message:     fmt.Sprintf("SIBLING_LANE_TOUCH: working directory %q touches sibling lane outside %v", relWd, inst.spec.LaneTree),
			}
			return &audit, NewSandboxError(ErrSiblingLaneTouch, audit.Message)
		}
	}

	// 2. Scan Command and Argv for candidate target paths
	candidates := extractCandidatePaths(req.Command, req.Argv)

	for _, candidate := range candidates {
		// A. Check for host-sensitive file access
		if isSens, category := isSensitiveHostPath(candidate); isSens {
			audit := Audit{
				TimestampMS: time.Now().UnixMilli(),
				Type:        "host_sensitive_path_deny",
				Message:     fmt.Sprintf("access to host sensitive path denied (%s): %s", category, candidate),
			}
			return &audit, NewSandboxError(ErrSecretExfiltrationAttempt, audit.Message)
		}

		if isFlag(candidate) {
			continue
		}

		// B. Resolve target path
		var absTarget string
		if filepath.IsAbs(candidate) {
			absTarget = filepath.Clean(candidate)
		} else if strings.HasPrefix(candidate, "~") {
			home, _ := os.UserHomeDir()
			if home == "" {
				home = "/home/user"
			}
			absTarget = filepath.Clean(filepath.Join(home, strings.TrimPrefix(candidate, "~")))
		} else {
			absTarget = filepath.Clean(filepath.Join(workingDir, candidate))
		}

		// Re-check resolved path for sensitive keywords
		if isSens, category := isSensitiveHostPath(absTarget); isSens {
			audit := Audit{
				TimestampMS: time.Now().UnixMilli(),
				Type:        "host_sensitive_path_deny",
				Message:     fmt.Sprintf("access to host sensitive path denied (%s): %s", category, absTarget),
			}
			return &audit, NewSandboxError(ErrSecretExfiltrationAttempt, audit.Message)
		}

		// C. Allowed system path bypass (e.g. /dev/null, cmd.exe, system32, etc.)
		if isAllowedSystemPath(candidate, absTarget) {
			continue
		}

		// D. Check if target is inside declared WritablePaths
		inWritable := false
		for _, wp := range inst.spec.WritablePaths {
			if within(wp, absTarget) {
				inWritable = true
				break
			}
		}
		if inWritable {
			continue
		}

		// E. Check target relative to WorkspaceDir
		relTarget, relErr := filepath.Rel(ws, absTarget)
		if relErr != nil || hasDotDotPrefix(relTarget) || relTarget == ".." {
			// Path escapes workspace root
			audit := Audit{
				TimestampMS: time.Now().UnixMilli(),
				Type:        ErrLanePathEscape,
				Message:     fmt.Sprintf("LANE_PATH_ESCAPE: path %q (%s) escapes workspace %q", candidate, absTarget, ws),
			}
			return &audit, NewSandboxError(ErrLanePathEscape, audit.Message)
		}

		// F. Check target relative to LaneTree (SIBLING_LANE_TOUCH)
		if len(inst.spec.LaneTree) > 0 {
			if !isLaneAllowed(inst.spec.LaneTree, relTarget) {
				audit := Audit{
					TimestampMS: time.Now().UnixMilli(),
					Type:        ErrSiblingLaneTouch,
					Message:     fmt.Sprintf("SIBLING_LANE_TOUCH: attempted access outside lane tree to sibling path: %s", relTarget),
				}
				return &audit, NewSandboxError(ErrSiblingLaneTouch, audit.Message)
			}
		}
	}

	return nil, nil
}

// ---------------------------------------------------------------------------
// PATH & OPERAND INSPECTION HELPERS
// ---------------------------------------------------------------------------

var redirectPattern = regexp.MustCompile(`(>{1,2}|<)\s*([^\s;&|<>"]+)`)

func extractCandidatePaths(command string, argv []string) []string {
	var candidates []string
	joined := command
	for _, a := range argv {
		joined += " " + a
	}

	// 1. Redirection operands
	matches := redirectPattern.FindAllStringSubmatch(joined, -1)
	for _, m := range matches {
		if len(m) >= 3 {
			t := strings.Trim(m[2], "'\"`")
			if t != "" {
				candidates = append(candidates, t)
			}
		}
	}

	// 2. Whitespace-delimited tokens
	words := strings.Fields(joined)
	for _, w := range words {
		trimmed := strings.Trim(w, "'\"`")
		trimmed = strings.TrimLeft(trimmed, ">")
		trimmed = strings.TrimLeft(trimmed, "<")
		trimmed = strings.Trim(trimmed, "'\"`")
		if trimmed != "" && isPathLike(trimmed) {
			candidates = append(candidates, trimmed)
		}
	}

	return dedupeStrings(candidates)
}

func isPathLike(token string) bool {
	if strings.Contains(token, "/") || strings.Contains(token, "\\") {
		return true
	}
	if strings.HasPrefix(token, ".") || strings.HasPrefix(token, "~") {
		return true
	}
	low := strings.ToLower(token)
	if strings.Contains(low, "id_rsa") || strings.Contains(low, "id_ed25519") ||
		strings.Contains(low, "passwd") || strings.Contains(low, "shadow") {
		return true
	}
	return false
}

func isFlag(token string) bool {
	if strings.HasPrefix(token, "-") {
		return true
	}
	if strings.HasPrefix(token, "/") {
		// Windows cmd flags like /c, /d, /s, /q, /b, /pid, /t, /f
		if strings.Count(token, "/") == 1 && len(token) <= 5 {
			return true
		}
	}
	return false
}

func isSensitiveHostPath(p string) (bool, string) {
	norm := strings.ToLower(filepath.ToSlash(p))

	// SSH keys and authorization
	if strings.Contains(norm, ".ssh") ||
		strings.Contains(norm, "id_rsa") ||
		strings.Contains(norm, "id_ed25519") ||
		strings.Contains(norm, "id_dsa") ||
		strings.Contains(norm, "id_ecdsa") ||
		strings.Contains(norm, "authorized_keys") ||
		strings.Contains(norm, "known_hosts") {
		return true, "ssh_credentials"
	}

	// Cloud credentials
	if strings.Contains(norm, ".aws/credentials") ||
		strings.Contains(norm, ".aws/config") ||
		strings.Contains(norm, ".kube/config") ||
		strings.Contains(norm, ".azure") {
		return true, "cloud_credentials"
	}

	// Unix sensitive authentication and shadow databases
	if strings.Contains(norm, "/etc/shadow") ||
		strings.Contains(norm, "/etc/passwd") ||
		strings.Contains(norm, "/etc/sudoers") ||
		strings.Contains(norm, "/etc/master.passwd") {
		return true, "system_credentials"
	}

	// Windows system secrets
	if strings.Contains(norm, "system32/config/sam") ||
		strings.Contains(norm, "system32/config/system") ||
		strings.Contains(norm, "system32/config/security") {
		return true, "windows_system_credentials"
	}

	// Shell histories
	if strings.Contains(norm, ".bash_history") ||
		strings.Contains(norm, ".zsh_history") {
		return true, "shell_history"
	}

	return false, ""
}

func isAllowedSystemPath(candidate, abs string) bool {
	cLow := strings.ToLower(filepath.ToSlash(candidate))
	aLow := strings.ToLower(filepath.ToSlash(abs))

	// Null and zero devices
	if cLow == "nul" || cLow == "/dev/null" || strings.HasSuffix(cLow, "/dev/null") || strings.HasSuffix(cLow, "/nul") {
		return true
	}
	if cLow == "/dev/zero" || strings.HasSuffix(cLow, "/dev/zero") {
		return true
	}

	// Standard system execution binaries
	for _, sysPrefix := range []string{
		"/bin/", "/usr/bin/", "/usr/local/bin/", "/sbin/",
		"c:/windows/system32/", "c:/windows/",
	} {
		if strings.HasPrefix(aLow, sysPrefix) || strings.HasPrefix(cLow, sysPrefix) {
			return true
		}
	}
	return false
}

func matchLaneGlob(pattern, relPath string) bool {
	pattern = filepath.ToSlash(pattern)
	relPath = filepath.ToSlash(relPath)

	pattern = strings.TrimPrefix(pattern, "./")
	relPath = strings.TrimPrefix(relPath, "./")

	if pattern == "**" || pattern == "*" {
		return true
	}

	if strings.HasSuffix(pattern, "/**") {
		base := strings.TrimSuffix(pattern, "/**")
		if relPath == base || strings.HasPrefix(relPath, base+"/") {
			return true
		}
	}

	if strings.HasSuffix(pattern, "/*") {
		base := strings.TrimSuffix(pattern, "/*")
		if relPath == base {
			return true
		}
		if strings.HasPrefix(relPath, base+"/") {
			sub := strings.TrimPrefix(relPath, base+"/")
			if !strings.Contains(sub, "/") {
				return true
			}
		}
	}

	if pattern == relPath {
		return true
	}

	matched, err := filepath.Match(pattern, relPath)
	return err == nil && matched
}

func isLaneAllowed(laneTree []string, relPath string) bool {
	if len(laneTree) == 0 {
		return true
	}
	relPath = filepath.ToSlash(filepath.Clean(relPath))
	if relPath == "." {
		return true
	}
	for _, pattern := range laneTree {
		if matchLaneGlob(pattern, relPath) {
			return true
		}
	}
	return false
}

func hasDotDotPrefix(rel string) bool {
	return len(rel) >= 3 && rel[0] == '.' && rel[1] == '.' && (rel[2] == '/' || rel[2] == '\\')
}

func within(base, target string) bool {
	rel, err := filepath.Rel(filepath.Clean(base), filepath.Clean(target))
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !hasDotDotPrefix(rel))
}

func dedupeStrings(in []string) []string {
	seen := make(map[string]bool)
	var out []string
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

func isCmdBuiltin(cmd string) bool {
	low := strings.ToLower(cmd)
	switch low {
	case "echo", "type", "dir", "copy", "del", "mkdir", "rmdir", "move", "ren", "time", "date", "ver", "vol":
		return true
	default:
		return false
	}
}

func prepareCommand(ctx context.Context, req ExecutionRequest) *exec.Cmd {
	name := req.Command
	args := req.Argv

	if runtime.GOOS == "windows" {
		isCmd := strings.EqualFold(filepath.Base(name), "cmd.exe") || strings.EqualFold(name, "cmd")
		if !isCmd {
			if len(args) == 0 && (strings.ContainsAny(name, " <>&|*\"") || isCmdBuiltin(name)) {
				args = []string{"/d", "/s", "/c", name}
				name = "cmd.exe"
			} else if isCmdBuiltin(name) {
				args = append([]string{"/d", "/s", "/c", name}, args...)
				name = "cmd.exe"
			}
		}
	} else {
		isSh := strings.EqualFold(filepath.Base(name), "sh") || strings.EqualFold(filepath.Base(name), "bash")
		if !isSh && len(args) == 0 && strings.ContainsAny(name, " <>&|*\"'") {
			args = []string{"-c", name}
			name = "/bin/sh"
		}
	}

	return exec.CommandContext(ctx, name, args...)
}
