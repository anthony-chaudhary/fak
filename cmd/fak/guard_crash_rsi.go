package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/launchguard"
)

const guardCrashRSIMarkerEnv = "FAK_GUARD_CRASH_RSI"

const (
	guardFailureRSITriggerChildResourceReceipt = "child_resource_receipt"
	guardFailureRSIReasonContainmentSurvivors  = "CHILD_RESOURCE_CONTAINMENT_SURVIVORS"
	guardFailureRSISubsystemChildResource      = "child_resource"
	guardFailureRSIReceiptChildResource        = "fak.guard.child-resource.v1/reap_tree"
)

type guardCrashRSIRequest struct {
	Tag             string
	Source          string
	Agent           string
	Class           string
	ExitCode        int
	Workspace       string
	Trigger         string
	Reason          string
	Subsystem       string
	Signature       string
	BuildCommit     string
	BuildModule     string
	ReceiptIdentity string
	Prompt          string
	Provider        string
	Model           string
	FallbackModel   string
	Env             []string
	Stderr          io.Writer
	OnWait          func(guardCrashRSITerminalRecord)
}

// guardCrashRSITerminalRecord records the observable terminal outcome of an RSI child process.
type guardCrashRSITerminalRecord struct {
	Tag       string    `json:"tag"`
	Agent     string    `json:"agent"`
	Provider  string    `json:"provider,omitempty"`
	Model     string    `json:"model,omitempty"`
	ExitCode  int       `json:"exit_code"`
	Error     string    `json:"error,omitempty"`
	Stderr    string    `json:"stderr,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

var (
	guardCrashRSITerminalMu      sync.Mutex
	guardCrashRSITerminalRecords []guardCrashRSITerminalRecord
	guardCrashRSITerminalHook    func(guardCrashRSITerminalRecord)
)

func recordGuardCrashRSITerminal(rec guardCrashRSITerminalRecord) {
	guardCrashRSITerminalMu.Lock()
	guardCrashRSITerminalRecords = append(guardCrashRSITerminalRecords, rec)
	hook := guardCrashRSITerminalHook
	guardCrashRSITerminalMu.Unlock()
	if hook != nil {
		hook(rec)
	}
}

func getGuardCrashRSITerminalRecords() []guardCrashRSITerminalRecord {
	guardCrashRSITerminalMu.Lock()
	defer guardCrashRSITerminalMu.Unlock()
	out := make([]guardCrashRSITerminalRecord, len(guardCrashRSITerminalRecords))
	copy(out, guardCrashRSITerminalRecords)
	return out
}

func resetGuardCrashRSITerminalRecords() {
	guardCrashRSITerminalMu.Lock()
	defer guardCrashRSITerminalMu.Unlock()
	guardCrashRSITerminalRecords = nil
}

var guardCrashRSILaunch = launchGuardCrashRSI
var guardCrashRSIAdmit = admitGuardCrashRSILaunch
var guardCrashRSIDir string

// guardRSISession admits at most one launch attempt for the lifetime of one guard
// session. Admission claims the slot before starting the worker, so even a failed
// launch cannot cause a later crash or failure trigger to fan out another worker.
type guardRSISession struct {
	attempted atomic.Bool
}

func (s *guardRSISession) claim() bool {
	return s != nil && s.attempted.CompareAndSwap(false, true)
}

// guardTypedTerminalFailure is the closed adapter between a human-oriented
// terminal error and RSI admission. Its cause remains available to the guard's
// original fail-closed error chain, but is never copied into the RSI request.
type guardTypedTerminalFailure struct {
	Trigger   string
	Reason    string
	Subsystem string
	cause     error
}

func (e *guardTypedTerminalFailure) Error() string { return e.cause.Error() }
func (e *guardTypedTerminalFailure) Unwrap() error { return e.cause }

// guardTypeContainmentSurvivorError types only the receipt writer's exact,
// closed survivor reason. Callers preserve and return the original receipt
// error; this adapter exists solely for the fail-open RSI side channel.
func guardTypeContainmentSurvivorError(err error) error {
	if err == nil {
		return nil
	}
	var typed *guardTypedTerminalFailure
	if errors.As(err, &typed) {
		return err
	}
	const prefix = "verify child resource reap: " + guardFailureRSIReasonContainmentSurvivors + ":"
	if !strings.HasPrefix(strings.TrimSpace(err.Error()), prefix) {
		return err
	}
	return &guardTypedTerminalFailure{
		Trigger:   guardFailureRSITriggerChildResourceReceipt,
		Reason:    guardFailureRSIReasonContainmentSurvivors,
		Subsystem: guardFailureRSISubsystemChildResource,
		cause:     err,
	}
}

// guardMaybeLaunchCrashRSI starts one bounded, independent investigation on the first
// restartable generic crash. It is deliberately fail-open: the original in-place restart
// remains the authority even when admission or launch fails.
func guardMaybeLaunchCrashRSI(stderr io.Writer, session *guardRSISession, guardTraceID, agentName, class string, code, restartsSoFar int) bool {
	req, ok := guardCrashRSIAdmission(guardTraceID, agentName, class, code, restartsSoFar)
	if !ok || !session.claim() {
		return false
	}
	return guardLaunchCrashRSI(stderr, req)
}

// guardMaybeLaunchFailureRSI routes only the closed, typed containment-survivor
// receipt failure into the existing isolated launcher. It never forwards receipt
// prose: that prose contains volatile PIDs and recovery detail that belongs only in
// the original fail-closed guard error.
func guardMaybeLaunchFailureRSI(stderr io.Writer, session *guardRSISession, guardTraceID, agentName string, receiptErr error) bool {
	req, ok := guardFailureRSIAdmission(guardTraceID, agentName, receiptErr)
	if !ok || !session.claim() {
		return false
	}
	return guardLaunchCrashRSI(stderr, req)
}

func guardLaunchCrashRSI(stderr io.Writer, req guardCrashRSIRequest) bool {
	kind := "crash"
	if req.Reason != "" {
		kind = "failure"
	}
	finish, decision, err := guardCrashRSIAdmit(req.Tag)
	if err != nil {
		if stderr != nil {
			fmt.Fprintf(stderr, "fak guard: %s RSI launch skipped (%s): launchguard: %v\n", kind, req.Tag, err)
		}
		return false
	}
	if finish == nil {
		if stderr != nil {
			fmt.Fprintf(stderr, "fak guard: %s RSI launch refused (%s): %s", kind, req.Tag, decision.Outcome)
			if decision.RetryAfter > 0 {
				fmt.Fprintf(stderr, " retry-after=%s", decision.RetryAfter.Round(time.Millisecond))
			}
			fmt.Fprintln(stderr)
		}
		return false
	}
	if req.Stderr == nil {
		req.Stderr = stderr
	}
	if err := guardCrashRSILaunch(req); err != nil {
		_ = finish(false)
		if stderr != nil {
			fmt.Fprintf(stderr, "fak guard: %s RSI launch skipped (%s): %v\n", kind, req.Tag, err)
		}
		return false
	}
	if err := finish(true); err != nil {
		if stderr != nil {
			fmt.Fprintf(stderr, "fak guard: %s RSI launch state warning (%s): %v\n", kind, req.Tag, err)
		}
	}
	if stderr != nil {
		if req.Reason != "" {
			fmt.Fprintf(stderr, "fak guard: spawned failure RSI session %s for %s\n", req.Tag, req.Reason)
		} else {
			fmt.Fprintf(stderr, "fak guard: spawned crash RSI session %s for original crash %s exit %d\n", req.Tag, req.Class, req.ExitCode)
		}
	}
	return true
}

func admitGuardCrashRSILaunch(identity string) (func(bool) error, launchguard.Decision, error) {
	dir := guardCrashRSIDir
	if dir == "" {
		root, err := os.UserCacheDir()
		if err != nil {
			return nil, launchguard.Decision{}, err
		}
		dir = filepath.Join(root, "fak", "launchguard")
	}
	g, err := launchguard.New(launchguard.Config{
		Dir:         dir,
		MaxAttempts: 1,
		Window:      10 * time.Minute,
		Cooldown:    10 * time.Minute,
		BaseBackoff: 5 * time.Second,
		MaxBackoff:  time.Minute,
		StaleAfter:  15 * time.Minute,
	})
	if err != nil {
		return nil, launchguard.Decision{}, err
	}
	decision, lease, err := g.Admit(identity)
	if err != nil || lease == nil {
		return nil, decision, err
	}
	return lease.Finish, decision, nil
}

func guardCrashRSIDefaultProvider(agent string) string {
	if p := strings.TrimSpace(os.Getenv("FAK_GUARD_CRASH_RSI_PROVIDER")); p != "" {
		return p
	}
	if agent == "codex" {
		return guardCodexProviderID
	}
	return ""
}

func guardCrashRSIDefaultModel(agent string) string {
	if m := strings.TrimSpace(os.Getenv("FAK_GUARD_CRASH_RSI_MODEL")); m != "" {
		return m
	}
	if agent == "codex" {
		return guardCodexDefaultModelID
	}
	return ""
}

func guardCrashRSIAdmission(guardTraceID, agentName, class string, code, restartsSoFar int) (guardCrashRSIRequest, bool) {
	if restartsSoFar != 0 || strings.TrimSpace(os.Getenv(guardCrashRSIMarkerEnv)) != "" {
		return guardCrashRSIRequest{}, false
	}
	trace := strings.TrimSpace(guardTraceID)
	class = strings.TrimSpace(class)
	if trace == "" || class == "" || code == 0 {
		return guardCrashRSIRequest{}, false
	}
	agent := guardCrashRSISupportedAgent(agentName)
	if agent == "" {
		return guardCrashRSIRequest{}, false
	}
	workspace, err := os.Getwd()
	if err != nil || !filepath.IsAbs(workspace) {
		return guardCrashRSIRequest{}, false
	}
	source := guardRSIDigest(trace)
	tag := "guard-crash-rsi/" + source
	req := guardCrashRSIRequest{
		Tag:           tag,
		Source:        source,
		Agent:         agent,
		Class:         class,
		ExitCode:      code,
		Workspace:     workspace,
		Provider:      guardCrashRSIDefaultProvider(agent),
		Model:         guardCrashRSIDefaultModel(agent),
		FallbackModel: guardCrashRSIDefaultModel(agent),
	}
	req.Prompt = guardCrashRSIPrompt(req)
	return req, true
}

func guardFailureRSIAdmission(guardTraceID, agentName string, receiptErr error) (guardCrashRSIRequest, bool) {
	if strings.TrimSpace(os.Getenv(guardCrashRSIMarkerEnv)) != "" || receiptErr == nil {
		return guardCrashRSIRequest{}, false
	}
	trace := strings.TrimSpace(guardTraceID)
	var typed *guardTypedTerminalFailure
	if trace == "" || !errors.As(receiptErr, &typed) || typed == nil || typed.cause == nil {
		return guardCrashRSIRequest{}, false
	}
	if typed.Trigger != guardFailureRSITriggerChildResourceReceipt ||
		typed.Reason != guardFailureRSIReasonContainmentSurvivors ||
		typed.Subsystem != guardFailureRSISubsystemChildResource {
		return guardCrashRSIRequest{}, false
	}
	agent := guardCrashRSISupportedAgent(agentName)
	if agent == "" {
		return guardCrashRSIRequest{}, false
	}
	identity := buildIdentityFromRuntime()
	buildCommit := strings.TrimSpace(identity.CommitShort)
	if buildCommit == "" {
		buildCommit = "unstamped"
	}
	buildModule := strings.TrimSpace(identity.ModuleVersion)
	if buildModule == "" {
		buildModule = "cmd/fak"
	}
	signature := guardRSIDigest(strings.Join([]string{
		typed.Trigger,
		typed.Reason,
		typed.Subsystem,
	}, "\x00"))
	req := guardCrashRSIRequest{
		Tag:             "guard-failure-rsi/" + guardRSIDigest(trace) + "-" + signature,
		Source:          guardRSIDigest(trace),
		Agent:           agent,
		Trigger:         typed.Trigger,
		Reason:          typed.Reason,
		Subsystem:       typed.Subsystem,
		Signature:       signature,
		BuildCommit:     buildCommit,
		BuildModule:     buildModule,
		ReceiptIdentity: guardFailureRSIReceiptChildResource,
		Provider:        guardCrashRSIDefaultProvider(agent),
		Model:           guardCrashRSIDefaultModel(agent),
		FallbackModel:   guardCrashRSIDefaultModel(agent),
	}
	req.Prompt = guardFailureRSIPrompt(req)
	return req, true
}

func guardRSIDigest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:8])
}

func guardCrashRSISupportedAgent(agentName string) string {
	name := strings.ToLower(strings.TrimSpace(filepath.Base(agentName)))
	name = strings.TrimSuffix(name, ".exe")
	switch name {
	case "claude", "claude-code":
		return "claude"
	case "codex":
		return "codex"
	default:
		return ""
	}
}

func guardCrashRSIPrompt(req guardCrashRSIRequest) string {
	return fmt.Sprintf(`You are the specially tagged crash-RSI investigation session %s.
Investigate the root cause of the ORIGINAL fak guard child crash; do not restart or reproduce the crashed session merely to continue its task.
Bounded crash context:
- source_guard: %s
- harness: %s
- crash_class: %s
- exit_code: %d
- workspace: %s
Perform read-only root-cause analysis, identify the smallest durable prevention, and report evidence plus a checkable next step. Do not expose credentials or ambient environment values.`, req.Tag, req.Source, req.Agent, req.Class, req.ExitCode, req.Workspace)
}

func guardFailureRSIPrompt(req guardCrashRSIRequest) string {
	return fmt.Sprintf(`You are the specially tagged guard-failure RSI investigation session %s.
Investigate the root cause of the ORIGINAL typed fak guard failure. Preserve fail-closed containment and do not continue the stopped agent task.
Bounded failure context:
- trigger: %s
- typed_reason: %s
- subsystem: %s
- harness: %s
- build_commit: %s
- build_module: %s
- source_guard: %s
- receipt_identity: %s
- normalized_signature: %s
Perform read-only root-cause analysis, identify the smallest durable prevention, and report evidence plus a checkable next step. Do not expose credentials, paths, process IDs, receipt payloads, or ambient environment values.`, req.Tag, req.Trigger, req.Reason, req.Subsystem, req.Agent, req.BuildCommit, req.BuildModule, req.Source, req.ReceiptIdentity, req.Signature)
}

var (
	guardCrashRSILookPath = exec.LookPath
	guardCrashRSICommand  = exec.Command
)

type boundedBufferWriter struct {
	buf *bytes.Buffer
	max int
}

func (w *boundedBufferWriter) Write(p []byte) (int, error) {
	if w.buf == nil {
		return len(p), nil
	}
	if w.buf.Len() >= w.max {
		return len(p), nil
	}
	remaining := w.max - w.buf.Len()
	if len(p) > remaining {
		w.buf.Write(p[:remaining])
		return len(p), nil
	}
	return w.buf.Write(p)
}

func guardCrashRSICommandArgs(req guardCrashRSIRequest) (string, []string, error) {
	var name string
	var args []string
	switch req.Agent {
	case "claude":
		name = "claude"
		args = []string{"-p", req.Prompt, "--permission-mode", "plan"}
		model := strings.TrimSpace(req.Model)
		if model == "" {
			model = strings.TrimSpace(req.FallbackModel)
		}
		if model == "" {
			model = guardCrashRSIDefaultModel("claude")
		}
		if model != "" {
			args = append(args, "--model", model)
		}
	case "codex":
		name = "codex"
		args = []string{"exec", "--sandbox", "read-only", "--json"}
		provider := strings.TrimSpace(req.Provider)
		if provider == "" {
			provider = guardCrashRSIDefaultProvider("codex")
		}
		if provider != "" {
			args = append(args, "-c", "model_provider="+provider)
		}
		model := strings.TrimSpace(req.Model)
		if model == "" {
			model = strings.TrimSpace(req.FallbackModel)
		}
		if model == "" {
			model = guardCrashRSIDefaultModel("codex")
		}
		if model != "" {
			args = append(args, "--model", model)
		}
		if req.Prompt != "" {
			args = append(args, req.Prompt)
		}
	default:
		return "", nil, fmt.Errorf("unsupported harness %q", req.Agent)
	}
	return name, args, nil
}

func launchGuardCrashRSI(req guardCrashRSIRequest) error {
	name, args, err := guardCrashRSICommandArgs(req)
	if err != nil {
		return err
	}
	path, err := guardCrashRSILookPath(name)
	if err != nil {
		return fmt.Errorf("resolve %s: %w", name, err)
	}
	cmd := guardCrashRSICommand(path, args...)
	cmd.Dir = req.Workspace

	effectiveProvider := strings.TrimSpace(req.Provider)
	if effectiveProvider == "" {
		effectiveProvider = guardCrashRSIDefaultProvider(req.Agent)
	}
	effectiveModel := strings.TrimSpace(req.Model)
	if effectiveModel == "" {
		effectiveModel = strings.TrimSpace(req.FallbackModel)
	}
	if effectiveModel == "" {
		effectiveModel = guardCrashRSIDefaultModel(req.Agent)
	}

	extraEnv := append([]string(nil), req.Env...)
	if effectiveProvider != "" {
		extraEnv = append(extraEnv, "FAK_GUARD_CRASH_RSI_PROVIDER="+effectiveProvider)
	}
	if effectiveModel != "" {
		extraEnv = append(extraEnv, "FAK_GUARD_CRASH_RSI_MODEL="+effectiveModel)
	}
	cmd.Env = guardCrashRSIEnvironment(req.Tag, extraEnv...)
	cmd.Stdin = nil
	cmd.Stdout = nil

	var stderrBuf bytes.Buffer
	cmd.Stderr = &boundedBufferWriter{buf: &stderrBuf, max: 4096}

	if err := cmd.Start(); err != nil {
		return err
	}

	go func() {
		waitErr := cmd.Wait()
		record := guardCrashRSITerminalRecord{
			Tag:       req.Tag,
			Agent:     req.Agent,
			Provider:  effectiveProvider,
			Model:     effectiveModel,
			Timestamp: time.Now(),
		}
		if waitErr != nil {
			var exitErr *exec.ExitError
			if errors.As(waitErr, &exitErr) {
				record.ExitCode = exitErr.ExitCode()
			} else {
				record.ExitCode = -1
			}
			record.Error = waitErr.Error()
			record.Stderr = strings.TrimSpace(stderrBuf.String())

			stderrDest := req.Stderr
			if stderrDest == nil {
				stderrDest = os.Stderr
			}
			if stderrDest != nil {
				kind := "crash"
				if req.Reason != "" {
					kind = "failure"
				}
				fmt.Fprintf(stderrDest, "fak guard: %s RSI child process failed (%s, exit %d): %v\n", kind, req.Tag, record.ExitCode, waitErr)
				if record.Stderr != "" {
					fmt.Fprintf(stderrDest, "fak guard: %s RSI child stderr (%s): %s\n", kind, req.Tag, record.Stderr)
				}
			}
		}
		recordGuardCrashRSITerminal(record)
		if req.OnWait != nil {
			req.OnWait(record)
		}
	}()
	return nil
}

// The investigation receives only process-bootstrap paths and the recursion marker. In
// particular, provider keys, original argv, and the parent's full ambient environment are not
// forwarded.
func guardCrashRSIEnvironment(tag string, extraEnv ...string) []string {
	allow := []string{
		"PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
		"SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
		"CLAUDE_CONFIG_DIR", "CODEX_HOME",
		"FAK_GUARD_CRASH_RSI_PROVIDER", "FAK_GUARD_CRASH_RSI_MODEL",
		"FAK_MODEL_PROVIDER", "FAK_MODEL",
	}
	env := make([]string, 0, len(allow)+1+len(extraEnv))
	for _, key := range allow {
		if value, ok := os.LookupEnv(key); ok && strings.TrimSpace(value) != "" {
			env = append(env, key+"="+value)
		}
	}
	env = append(env, guardCrashRSIMarkerEnv+"="+tag)
	for _, kv := range extraEnv {
		kv = strings.TrimSpace(kv)
		if kv == "" {
			continue
		}
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) == 2 {
			k := strings.ToUpper(parts[0])
			replaced := false
			for i, existing := range env {
				if strings.HasPrefix(strings.ToUpper(existing), k+"=") {
					env[i] = kv
					replaced = true
					break
				}
			}
			if !replaced {
				env = append(env, kv)
			}
		} else {
			env = append(env, kv)
		}
	}
	return env
}
