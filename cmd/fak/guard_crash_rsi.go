package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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
}

var guardCrashRSILaunch = launchGuardCrashRSI
var guardCrashRSIAdmit = admitGuardCrashRSILaunch

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
	if err := guardCrashRSILaunch(req); err != nil {
		if stderr != nil {
			fmt.Fprintf(stderr, "fak guard: failure RSI launch skipped (%s): %v\n", req.Tag, err)
		}
		return false
	}
	if stderr != nil {
		fmt.Fprintf(stderr, "fak guard: spawned failure RSI session %s for %s\n", req.Tag, req.Reason)
	}
	return true
}

func guardLaunchCrashRSI(stderr io.Writer, req guardCrashRSIRequest) bool {
	finish, decision, err := guardCrashRSIAdmit(req.Tag)
	if err != nil {
		if stderr != nil {
			fmt.Fprintf(stderr, "fak guard: crash RSI launch skipped (%s): launchguard: %v\n", req.Tag, err)
		}
		return false
	}
	if finish == nil {
		if stderr != nil {
			fmt.Fprintf(stderr, "fak guard: crash RSI launch refused (%s): %s", req.Tag, decision.Outcome)
			if decision.RetryAfter > 0 {
				fmt.Fprintf(stderr, " retry-after=%s", decision.RetryAfter.Round(time.Millisecond))
			}
			fmt.Fprintln(stderr)
		}
		return false
	}
	if err := guardCrashRSILaunch(req); err != nil {
		_ = finish(false)
		if stderr != nil {
			fmt.Fprintf(stderr, "fak guard: crash RSI launch skipped (%s): %v\n", req.Tag, err)
		}
		return false
	}
	if err := finish(true); err != nil {
		if stderr != nil {
			fmt.Fprintf(stderr, "fak guard: crash RSI launch state warning (%s): %v\n", req.Tag, err)
		}
	}
	if stderr != nil {
		fmt.Fprintf(stderr, "fak guard: spawned crash RSI session %s for original crash %s exit %d\n", req.Tag, req.Class, req.ExitCode)
	}
	return true
}

func admitGuardCrashRSILaunch(identity string) (func(bool) error, launchguard.Decision, error) {
	root, err := os.UserCacheDir()
	if err != nil {
		return nil, launchguard.Decision{}, err
	}
	g, err := launchguard.New(launchguard.Config{
		Dir: filepath.Join(root, "fak", "launchguard"), MaxAttempts: 3,
		Window: 10 * time.Minute, BaseBackoff: 5 * time.Second,
		MaxBackoff: time.Minute, StaleAfter: 15 * time.Minute,
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
		Tag:       tag,
		Source:    source,
		Agent:     agent,
		Class:     class,
		ExitCode:  code,
		Workspace: workspace,
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

func launchGuardCrashRSI(req guardCrashRSIRequest) error {
	var name string
	var args []string
	switch req.Agent {
	case "claude":
		name = "claude"
		args = []string{"-p", req.Prompt, "--permission-mode", "plan"}
	case "codex":
		name = "codex"
		args = []string{"exec", "--sandbox", "read-only", "--json", req.Prompt}
	default:
		return fmt.Errorf("unsupported harness %q", req.Agent)
	}
	path, err := exec.LookPath(name)
	if err != nil {
		return fmt.Errorf("resolve %s: %w", name, err)
	}
	cmd := exec.Command(path, args...)
	cmd.Dir = req.Workspace
	cmd.Env = guardCrashRSIEnvironment(req.Tag)
	cmd.Stdin = nil
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return err
	}
	go func() { _ = cmd.Wait() }()
	return nil
}

// The investigation receives only process-bootstrap paths and the recursion marker. In
// particular, provider keys, original argv, and the parent's full ambient environment are not
// forwarded.
func guardCrashRSIEnvironment(tag string) []string {
	allow := []string{"PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "CLAUDE_CONFIG_DIR", "CODEX_HOME"}
	env := make([]string, 0, len(allow)+1)
	for _, key := range allow {
		if value, ok := os.LookupEnv(key); ok && strings.TrimSpace(value) != "" {
			env = append(env, key+"="+value)
		}
	}
	return append(env, guardCrashRSIMarkerEnv+"="+tag)
}
