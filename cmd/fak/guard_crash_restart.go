package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// In-place harness-crash restart (#4686) — the generic-crash counterpart of the four narrow
// child-survival seams (auth-expiry, account rotation, cap park, budget restart) and the direct
// sibling of the transient-wire retry (#3514). Those seams already keep the guard MASTER (the
// process that owns the in-process gateway goroutine, the stable guardTraceID, the audit journal,
// and the one Slack session card) alive while only the wrapped harness is stopped and relaunched
// under the same session. A GENERIC crash — a panic surfaced as a non-zero exit, a SIGSEGV/SIGABRT,
// an OOM (exit 137), an arbitrary non-zero status — matches none of them and falls straight through
// to finishGuardChildAndReport, which cancel()s the gateway context and os.Exit()s the whole guard
// process. So a crash in the HARNESS takes the MASTER (and its gateway) down with it — the opposite
// of run isolation. This seam closes that gap: a default-on, bounded, loud in-place restart of the
// harness under the same master session.

// guardCrashRestartLimit resolves the per-session crash-restart budget. The guard parent and
// harness child are separate failure domains by default: one child crash must not tear down the
// parent. An explicit env value, including 0, remains the fleet/operator override.
const (
	guardCrashRestartLimitEnv     = "FLEET_CLAUDE_GUARD_CRASH_RESTART_LIMIT"
	guardCrashRestartDefaultLimit = 3
	guardCrashRestartInitialDelay = 250 * time.Millisecond
	guardCrashRestartMaxDelay     = 2 * time.Second
)

func guardCrashRestartLimit() int {
	raw, set := os.LookupEnv(guardCrashRestartLimitEnv)
	if !set || strings.TrimSpace(raw) == "" {
		return guardCrashRestartDefaultLimit
	}
	n, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || n < 0 {
		return guardCrashRestartDefaultLimit
	}
	return n
}

// guardCrashRestartDelay is the bounded exponential pause before attempt (1-based). It prevents
// a boot-crashing harness from hot-spinning the parent or hammering an upstream.
func guardCrashRestartDelay(attempt int) time.Duration {
	if attempt <= 0 {
		return 0
	}
	policy := loopmgr.RestartPolicy{
		MaxAttempts: ^uint64(0),
		BaseDelay:   guardCrashRestartInitialDelay,
		MaxDelay:    guardCrashRestartMaxDelay,
	}
	return policy.BackoffDelay(uint64(attempt-1), nil)
}

const (
	guardCrashRestartExhaustedReason = "CRASH_RESTART_EXHAUSTED"
	guardCrashNoProgressLimitEnv     = "FLEET_CLAUDE_GUARD_CRASH_NO_PROGRESS_LIMIT"
	guardCrashNoProgressLimitDefault = 2
	guardCodexCLIUsageReason         = "CODEX_CLI_USAGE"
	guardCodexInvalidJSONReason      = "CODEX_INVALID_JSON"
	guardChildStderrCaptureLimit     = 64 << 10
	guardChildStdoutCaptureLimit     = 64 << 10
)

// guardChildStderrCapture preserves the child's stderr stream while retaining a bounded prefix
// for exit classification. CLI parsers write their actionable diagnostic before exiting, so the
// prefix contains the directly observed Codex usage envelope without retaining an arbitrarily
// long interactive session. Write delegates first and records only bytes accepted by the original
// sink, keeping the operator-visible stream byte-for-byte unchanged.
type guardChildStderrCapture struct {
	mu   sync.Mutex
	dst  io.Writer
	data []byte
}

func newGuardChildStderrCapture(dst io.Writer) *guardChildStderrCapture {
	if dst == nil {
		dst = io.Discard
	}
	return &guardChildStderrCapture{dst: dst}
}

func (c *guardChildStderrCapture) Write(p []byte) (int, error) {
	n, err := c.dst.Write(p)
	if n <= 0 {
		return n, err
	}
	c.mu.Lock()
	remaining := guardChildStderrCaptureLimit - len(c.data)
	if remaining > n {
		remaining = n
	}
	if remaining > 0 {
		c.data = append(c.data, p[:remaining]...)
	}
	c.mu.Unlock()
	return n, err
}

func (c *guardChildStderrCapture) String() string {
	if c == nil {
		return ""
	}
	return lockAndString(&c.mu, c.data)
}

func guardCaptureChildStderr(child *exec.Cmd, agentName string) *guardChildStderrCapture {
	if child == nil || guardAgentBaseName(agentName) != "codex" {
		return nil
	}
	capture := newGuardChildStderrCapture(child.Stderr)
	child.Stderr = capture
	return capture
}

// guardChildStdoutCapture preserves the complete JSONL stream while retaining only its bounded
// tail for terminal-event classification. Codex writes turn.failed after all preceding events,
// so a prefix would lose the only actionable event in a long turn. As with stderr, Write delegates
// first and records only the bytes accepted by the original sink.
type guardChildStdoutCapture struct {
	mu   sync.Mutex
	dst  io.Writer
	data []byte
}

func newGuardChildStdoutCapture(dst io.Writer) *guardChildStdoutCapture {
	if dst == nil {
		dst = io.Discard
	}
	return &guardChildStdoutCapture{dst: dst}
}

func (c *guardChildStdoutCapture) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	n, err := c.dst.Write(p)
	if n <= 0 {
		return n, err
	}
	accepted := p[:n]
	if len(accepted) >= guardChildStdoutCaptureLimit {
		c.data = append(c.data[:0], accepted[len(accepted)-guardChildStdoutCaptureLimit:]...)
	} else {
		overflow := len(c.data) + len(accepted) - guardChildStdoutCaptureLimit
		if overflow > 0 {
			copy(c.data, c.data[overflow:])
			c.data = c.data[:len(c.data)-overflow]
		}
		c.data = append(c.data, accepted...)
	}
	return n, err
}

func (c *guardChildStdoutCapture) String() string {
	if c == nil {
		return ""
	}
	return lockAndString(&c.mu, c.data)
}

func lockAndString(mu *sync.Mutex, data []byte) string {
	mu.Lock()
	defer mu.Unlock()
	return string(data)
}

// guardIsCodexExecJSONCommand keys capture off semantic argv, not the executable name alone.
// Codex root configuration overrides may precede the exact exec subcommand; every other root
// command and interactive shape keeps its original stdout writer untouched.
func guardIsCodexExecJSONCommand(command []string, agentName string) bool {
	if len(command) == 0 || guardAgentBaseName(agentName) != "codex" || guardAgentBaseName(command[0]) != "codex" {
		return false
	}
	args := command[1:]
	execAt := -1
	for i := 0; i < len(args); {
		switch arg := args[i]; {
		case arg == "exec":
			execAt = i
			i = len(args)
		case arg == "-c" || arg == "--config" || arg == "--enable" || arg == "--disable":
			if i+1 >= len(args) {
				return false
			}
			i += 2
		case strings.HasPrefix(arg, "--config=") || strings.HasPrefix(arg, "--enable=") || strings.HasPrefix(arg, "--disable="):
			i++
		default:
			return false
		}
	}
	if execAt < 0 {
		return false
	}
	for _, arg := range args[execAt+1:] {
		if arg == "--" {
			return false
		}
		if arg == "--json" {
			return true
		}
	}
	return false
}

func guardCaptureChildStdout(child *exec.Cmd, command []string, agentName string) *guardChildStdoutCapture {
	if child == nil || !guardIsCodexExecJSONCommand(command, agentName) {
		return nil
	}
	capture := newGuardChildStdoutCapture(child.Stdout)
	child.Stdout = capture
	return capture
}

// guardIsCodexCLIUsageFailure recognizes only the observed Codex exec parse contract: a real
// exit 2 accompanied by both its unexpected-argument diagnostic and exact usage line. Exit 2 by
// itself remains a generic crash because Go panics and unrelated commands can use the same code.
func guardIsCodexCLIUsageFailure(runErr error, childState *os.ProcessState, agentName, capturedStderr string) bool {
	if guardAgentBaseName(agentName) != "codex" {
		return false
	}
	_, code, isCrash := guardClassifyChildCrash(runErr, childState)
	if !isCrash || code != 2 {
		return false
	}
	parseDiagnostic := false
	usage := false
	for _, raw := range strings.Split(capturedStderr, "\n") {
		line := strings.TrimSpace(raw)
		if strings.HasPrefix(line, "error: unexpected argument '") && strings.HasSuffix(line, "' found") {
			parseDiagnostic = true
		}
		if line == "Usage: codex exec [OPTIONS] [PROMPT]" {
			usage = true
		}
	}
	return parseDiagnostic && usage
}

// guardIsCodexInvalidJSONFailure recognizes the exact Codex exec-resume terminal envelope
// observed in #9481: exit 1 plus a turn.failed JSON event classified as "other" whose message
// is the stable function-argument JSON parser diagnostic. Requiring all four fields keeps an
// unrelated exit 1, a generic "other" error, and unstructured prose on the bounded restart path.
func guardIsCodexInvalidJSONFailure(runErr error, childState *os.ProcessState, agentName, capturedStdout string) bool {
	if guardAgentBaseName(agentName) != "codex" {
		return false
	}
	_, code, isCrash := guardClassifyChildCrash(runErr, childState)
	if !isCrash || code != 1 {
		return false
	}
	type turnFailedEvent struct {
		Type  string `json:"type"`
		Error struct {
			Message        string `json:"message"`
			CodexErrorInfo string `json:"codex_error_info"`
		} `json:"error"`
	}
	for _, raw := range strings.Split(capturedStdout, "\n") {
		var event turnFailedEvent
		if json.Unmarshal([]byte(strings.TrimSpace(raw)), &event) != nil {
			continue
		}
		if event.Type != "turn.failed" || event.Error.CodexErrorInfo != "other" {
			continue
		}
		const diagnostic = "failed to parse function arguments: EOF while parsing an object at line "
		if !strings.HasPrefix(event.Error.Message, diagnostic) {
			continue
		}
		coordinates := strings.TrimPrefix(event.Error.Message, diagnostic)
		line, column, ok := strings.Cut(coordinates, " column ")
		if ok && guardPositiveDecimal(line) && guardPositiveDecimal(column) {
			return true
		}
	}
	return false
}

func guardPositiveDecimal(value string) bool {
	if value == "" {
		return false
	}
	for _, digit := range value {
		if digit < '0' || digit > '9' {
			return false
		}
	}
	n, err := strconv.ParseUint(value, 10, 64)
	return err == nil && n > 0
}

// guardRefuseCodexCLIUsage records one durable, typed terminal witness. The caller then enters
// the ordinary final-report funnel with the original *exec.ExitError, preserving exit 2 and the
// stderr that the capture already streamed while creating no restart hop or replacement child.
func guardRefuseCodexCLIUsage(runErr error, childState *os.ProcessState, agentName, traceID string, capturedStderr string, started time.Time, auditJournal *journal.Journal, stderr io.Writer) bool {
	if !guardIsCodexCLIUsageFailure(runErr, childState, agentName, capturedStderr) {
		return false
	}
	appendGuardChildExitWitnessWithReason(auditJournal, agentName, traceID, runErr, childState, started, guardCodexCLIUsageReason)
	if stderr != nil {
		fmt.Fprintf(stderr, "fak guard: %s: Codex rejected the command-line usage; correct the command with `codex exec --help` before relaunching\n", guardCodexCLIUsageReason)
	}
	return true
}

// guardRefuseCodexInvalidJSON records only the typed terminal reason; the captured Codex event
// remains on the original child stream and never enters the audit journal. The caller preserves
// the original exit-1 error while stopping before generic crash-restart admission.
func guardRefuseCodexInvalidJSON(runErr error, childState *os.ProcessState, agentName, traceID string, capturedStdout string, started time.Time, auditJournal *journal.Journal, stderr io.Writer) bool {
	if !guardIsCodexInvalidJSONFailure(runErr, childState, agentName, capturedStdout) {
		return false
	}
	appendGuardChildExitWitnessWithReason(auditJournal, guardAgentBaseName(agentName), traceID, runErr, childState, started, guardCodexInvalidJSONReason)
	if stderr != nil {
		fmt.Fprintf(stderr, "fak guard: %s: Codex ended the turn on invalid function-call JSON; correct the failed turn before resuming\n", guardCodexInvalidJSONReason)
	}
	return true
}

func guardCrashNoProgressLimit(crashLimit int) int {
	raw := strings.TrimSpace(os.Getenv(guardCrashNoProgressLimitEnv))
	limit := guardCrashNoProgressLimitDefault
	if raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n >= 0 {
			limit = n
		}
	}
	// A no-progress reap is useful only when it can fire before the flat budget.
	if crashLimit > 1 && limit >= crashLimit {
		limit = crashLimit - 1
	}
	return limit
}

// guardCrashNoProgressStep applies the same HEAD-advance discipline as budget restarts.
// A crash that shipped earns a reset; K consecutive crashes at the same HEAD are reaped.
func guardCrashNoProgressStep(prevHead, currentHead string, stalled, limit int) (nextHead string, nextStalled int, reap bool) {
	nextHead, nextStalled = guardNoProgressStep(prevHead, currentHead, stalled)
	return nextHead, nextStalled, limit > 0 && nextStalled >= limit
}

func guardCrashRestartGiveUpStatus(limit int, traceID string) string {
	return fmt.Sprintf("fak guard: %s: harness crash restart reaped after %d consecutive crash(es) without HEAD progress (trace %s); refusing another relaunch", guardCrashRestartExhaustedReason, limit, strings.TrimSpace(traceID))
}

func guardRecordCrashRestartGiveUp(auditJournal *journal.Journal, agentName, traceID string) {
	if auditJournal != nil {
		auditJournal.AppendCrash(agentName, traceID, guardCrashRestartExhaustedReason, -1)
	}
}

// guardMaybeRestartOnCrash is the generic-crash admission decision, wired at the SAME two recovery
// sites as its siblings (runGuardChildAndReport / runGuardChildSupervisedAndReport), AFTER the
// auth / rotation / cap / wire-retry seams have each declined to claim the exit and BEFORE the
// fall-through to finishGuardChildAndReport. It reuses guardClassifyChildCrash (the existing, until
// now journal-only OOM/SIGNAL/NONZERO_EXIT taxonomy) and answers whether the exit was a genuine
// crash that should be RESTARTED IN PLACE under the same master session.
//
// It is a PURE decision — no I/O, no relaunch, no sleep — so the admission discipline is
// unit-tested without standing the supervision loop up. ok=true requires ALL of:
//   - crash-restart is enabled (limit > 0) — default OFF, strictly opt-in;
//   - the exit is a real crash (guardClassifyChildCrash reports isCrash) — a clean exit (nil runErr
//     or a zero code) NEVER matches, because the agent FINISHED and the master should exit as today,
//     and neither does a spawn failure (reported by the caller's launch-failure path);
//   - the per-session crash-restart budget is not yet spent (restartsSoFar < limit) — once spent the
//     crash is SURFACED (the master exits), never masked by an unbounded relaunch (the explicit
//     concern #3514 raised about a bare NONZERO_EXIT trigger; the bound is the safety valve until
//     #4687 adds backoff + a progress-aware reap + a typed give-up).
//
// On ok=true it returns the crash class (a journal.Crash* constant) and the child's exit code so the
// caller can emit a loud, typed witness. Every other case returns ok=false and the caller's existing
// report/exit path proceeds byte-identically to today.
func guardMaybeRestartOnCrash(runErr error, childState *os.ProcessState, restartsSoFar, limit int) (class string, code int, ok bool) {
	if limit <= 0 {
		return "", 0, false // crash-restart disabled (the default) — master exits on crash as today
	}
	class, code, isCrash := guardClassifyChildCrash(runErr, childState)
	if !isCrash {
		return "", 0, false // clean exit or spawn failure — never restart in place
	}
	if restartsSoFar >= limit {
		return "", 0, false // budget spent — surface the crash rather than mask a systematic fault
	}
	return class, code, true
}

// guardReportCrashRestart makes generic crash recovery visible at the supervision boundary.
// The paired CHILD_CRASH and RESTART_HOP journal rows are durable evidence; this line is the
// immediate operator signal, including the typed crash class, OS exit code, and bounded attempt.
func guardReportCrashRestart(stderr io.Writer, agentName, class string, code, attempt, limit int, command []string) {
	if stderr == nil {
		return
	}
	fmt.Fprintf(stderr, "fak guard: %s harness crashed (%s, exit %d); guard remains up and is restarting the child in place (crash restart %d/%d) `%s`\n", agentName, class, code, attempt, limit, strings.Join(guardRestartRelaunchCommand(command, agentName), " "))
}

// guardCrashRestartHop builds the correlated RESTART_HOP record for an in-place crash restart
// (#4686) so a generic-crash relaunch folds into the SAME restart chain (and `fak guard
// restart-audit`) as a budget restart or a wire retry, rather than being an invisible relaunch.
// It mirrors guardWireRetryHop: the relaunch is a --continue reattach under the SAME trace (the
// crashed session resumes in place — no new continuation trace is minted and no seed is written),
// so from/to/child are all guardTraceID, handback is "continue", and status is ok. It degrades to
// the ORPHANED/inert shape for an unrecognized agent (for which fak cannot guess a resume syntax,
// so guardRestartRelaunchCommand leaves the command cold), for symmetry with guardWireRetryHop. The
// crash CLASS itself rides the paired CHILD_CRASH witness (appendGuardChildExitWitness), which the
// guard-RSI fold already consumes; this hop carries the lineage.
func guardCrashRestartHop(guardTraceID, agentName string, hop int) journal.RestartHop {
	return guardSameTraceRelaunchHop(guardTraceID, agentName, hop)
}
