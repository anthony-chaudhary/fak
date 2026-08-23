package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

type guardRestartBaton struct {
	ObjectivePinID  string `json:"objective_pin_id"`
	ObjectiveDigest string `json:"objective_digest"`
	ProgressCursor  string `json:"progress_cursor"`
	NextAction      string `json:"next_action"`
}

type guardBudgetRestartEvent struct {
	Schema             string                   `json:"schema"`
	FromTraceID        string                   `json:"from_trace_id"`
	ToTraceID          string                   `json:"to_trace_id"`
	Reason             string                   `json:"reason,omitempty"`
	SourceReadRecovery *guardSourceReadRecovery `json:"source_read_recovery,omitempty"`
	SeedFile           string                   `json:"seed_file,omitempty"`
	Seed               []agent.Message          `json:"seed_messages,omitempty"`
	SeedText           string                   `json:"seed_text,omitempty"`
	Baton              guardRestartBaton        `json:"baton,omitempty,omitzero"`
	Note               string                   `json:"note"`
}

type guardBudgetRestarter struct {
	enabled            bool
	freshContextTokens int
	limit              int
	seedDir            string
	// seedHandback selects the #3056 headless/no-continue handback: inject the carryover
	// seed_text as the recognized child's initial prompt on relaunch instead of the default
	// #3055 --continue transcript reattach. Set from the --restart-seed-handback knob.
	seedHandback   bool
	stderr         io.Writer
	progressCursor func() string
	events         chan guardBudgetRestartEvent
}

func newGuardBudgetRestarter(enabled bool, freshContextTokens, limit int, seedDir string, stderr io.Writer) *guardBudgetRestarter {
	return &guardBudgetRestarter{
		enabled:            enabled,
		freshContextTokens: freshContextTokens,
		limit:              limit,
		seedDir:            strings.TrimSpace(seedDir),
		stderr:             stderr,
		progressCursor:     sessionStartSHA,
		events:             make(chan guardBudgetRestartEvent, 1),
	}
}

func (r *guardBudgetRestarter) Enabled() bool { return r != nil && r.enabled }

func (r *guardBudgetRestarter) OnBudgetExhausted(ctx context.Context, st gateway.SessionState, messages []agent.Message) {
	if !r.Enabled() || strings.TrimSpace(st.TraceID) == "" || strings.TrimSpace(st.ContinuationID) == "" {
		return
	}
	reset := resetServedSessionOnBudget(r.freshContextTokens)
	if reset == nil {
		return
	}
	nextTrace, seed, ok := reset(ctx, st.TraceID, messages)
	if !ok || strings.TrimSpace(nextTrace) == "" {
		return
	}
	ev := guardBudgetRestartEvent{
		Schema:      "fak.guard.budget_restart.v1",
		FromTraceID: st.TraceID,
		ToTraceID:   nextTrace,
		Reason:      st.Reason,
		Seed:        seed,
		SeedText:    guardSeedText(seed),
		Note:        "context budget exhausted; fak guard is relaunching the child under the continuation trace",
	}
	if text, recovery, ok := guardQuarantinedReadRecovery(messages); ok {
		ev.SeedText = strings.TrimSpace(ev.SeedText + "\n\n" + text)
		ev.SourceReadRecovery = &recovery
	}
	pin := serveSessions.Get(st.TraceID).ObjectivePin
	if pin.Verify() && r.progressCursor != nil {
		ev.Baton = newGuardRestartBaton(pin.PinID, pin.Digest, r.progressCursor())
		if text := guardRestartBatonText(ev.Baton); text != "" {
			ev.SeedText = strings.TrimSpace(ev.SeedText + "\n\n" + text)
		}
	}
	if path, err := writeGuardRestartSeedFile(r.seedDir, ev); err == nil {
		ev.SeedFile = path
	} else if r.stderr != nil {
		fmt.Fprintf(r.stderr, "fak guard: budget restart seed write failed: %v\n", err)
	}
	select {
	case r.events <- ev:
	default:
		if r.stderr != nil {
			fmt.Fprintf(r.stderr, "fak guard: budget restart event for %s dropped; restart already pending\n", st.TraceID)
		}
	}
}

func newGuardRestartBaton(pinID, digest, cursor string) guardRestartBaton {
	b := guardRestartBaton{
		ObjectivePinID:  strings.TrimSpace(pinID),
		ObjectiveDigest: strings.TrimSpace(digest),
		ProgressCursor:  strings.TrimSpace(cursor),
		NextAction:      "verify the progress cursor, then continue the pinned objective",
	}
	if !b.valid() {
		return guardRestartBaton{}
	}
	return b
}

func (b guardRestartBaton) valid() bool {
	return b.ObjectivePinID != "" && b.ObjectiveDigest != "" && b.ProgressCursor != "" && b.NextAction != ""
}

func guardRestartBatonText(b guardRestartBaton) string {
	if !b.valid() {
		return ""
	}
	return fmt.Sprintf("BATON\nobjective_pin_id=%s\nobjective_digest=%s\nprogress_cursor=%s\nnext_action=%s\nEND BATON",
		b.ObjectivePinID, b.ObjectiveDigest, b.ProgressCursor, b.NextAction)
}

func guardSeedText(seed []agent.Message) string {
	var parts []string
	for _, m := range seed {
		if c := strings.TrimSpace(m.Content); c != "" {
			parts = append(parts, c)
		}
	}
	return strings.Join(parts, "\n\n")
}

func writeGuardRestartSeedFile(dir string, ev guardBudgetRestartEvent) (string, error) {
	if strings.TrimSpace(dir) == "" {
		var err error
		dir, err = os.MkdirTemp("", "fak-guard-reset-*")
		if err != nil {
			return "", err
		}
	} else if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	name := "reset-" + guardSafeFilePart(ev.FromTraceID) + "-to-" + guardSafeFilePart(ev.ToTraceID) + ".json"
	path := filepath.Join(dir, name)
	ev.SeedFile = path
	raw, err := json.MarshalIndent(ev, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

func guardSafeFilePart(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "trace"
	}
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	out := b.String()
	if out == "" {
		return "trace"
	}
	if len(out) > 80 {
		out = out[:80]
	}
	return out
}

func guardRestartEnv(ev guardBudgetRestartEvent) [][2]string {
	env := [][2]string{
		{"FAK_RESET_FROM_TRACE", ev.FromTraceID},
		{"FAK_RESET_TRACE_ID", ev.ToTraceID},
		{"FAK_SESSION_ID", ev.ToTraceID},
		{"FAK_RESUME_OF_ATTEMPT_ID", ev.FromTraceID},
		{"FAK_CHILD_ATTEMPT_ID", ev.ToTraceID},
		{"FAK_RESET_REASON", ev.Reason},
	}
	if ev.SeedFile != "" {
		env = append(env, [2]string{"FAK_RESET_SEED_FILE", ev.SeedFile})
	}
	return env
}

// guardRestartRelaunchCommand returns the command to relaunch the wrapped child with after a
// budget restart. For a recognized agent (Claude Code) it REATTACHES the existing transcript by
// appending the agent's resume flag (`--continue`), so the relaunched child resumes the same
// conversation the carryover seed was captured from instead of booting a cold, empty session and
// reporting "I don't have the task" (#3055). The FAK_RESET_* env vars guardRestartEnv sets are
// advisory only — no in-child reader consumes them — so continuity must come from the agent's own
// resume path: the exact flag formatGuardResumeGuidance already tells operators to run by hand, and
// the one guardMaybeRecoverAuthCrash already auto-injects on the auth-crash path. Idempotent via
// guardAppendContinueFlag: a second restart in the same session never stacks the flag. For an
// unrecognized agent fak cannot guess a safe resume syntax, so command is returned unchanged and
// the relaunch falls back to today's cold behavior (the headless/no-continue seed-prompt handback
// is the separate #3056 rung).
func guardRestartRelaunchCommand(command []string, agentName string) []string {
	if flag, ok := guardContinueFlagForAgent(agentName); ok {
		return guardAppendContinueFlag(command, flag)
	}
	return command
}

// guardSeedPromptTokenBudget is the documented ceiling on a carryover seed re-injected as a
// relaunch prompt (#3056). Measured in guardApproxTokens' ~4-bytes/token gauge, so ~64 KB of seed
// prose. Now that the seed is the AUTHORITATIVE restart context by default (the relaunch boots
// fresh on it and strips --continue rather than reattaching the exhausted transcript), it must
// carry enough of the load-bearing "what were you doing / where did you get to" carryover to
// re-orient the child without the transcript — so the ceiling is raised 8× from the original 2000.
// It stays well under any real context window (a distilled seed, not the whole transcript), so a
// fresh boot on it genuinely SHRINKS the window that exhaustion overflowed; anything past the
// ceiling is truncated AND logged, never silently.
const guardSeedPromptTokenBudget = 16000

// guardBoundSeedPrompt truncates seed to at most tokenBudget approx-tokens (guardApproxTokens'
// 4-bytes/token gauge), cutting on a UTF-8 rune boundary so a multi-byte rune is never split. It
// returns the bounded text and the number of dropped approx-tokens — 0 when the seed already fit.
// A non-zero drop is the caller's cue to LOG what was dropped: the bound is never silent (#3056).
func guardBoundSeedPrompt(seed string, tokenBudget int) (bounded string, droppedTokens int) {
	if tokenBudget <= 0 {
		return seed, 0
	}
	total := guardApproxTokens(seed)
	if total <= tokenBudget {
		return seed, 0
	}
	keep := tokenBudget * 4 // approx-tokens back to a byte budget (guardApproxTokens is ceil(len/4))
	if keep >= len(seed) {
		return seed, 0
	}
	// Back up off any UTF-8 continuation byte (0b10xxxxxx) so the cut lands on a rune start and a
	// multi-byte rune is never split mid-sequence.
	for keep > 0 && seed[keep]&0xC0 == 0x80 {
		keep--
	}
	bounded = seed[:keep]
	return bounded, total - guardApproxTokens(bounded)
}

// guardSeedPromptRelaunchCommand injects the bounded carryover seed_text as the recognized child's
// initial prompt on a headless/no-continue relaunch (#3056) — the handback the operator selects with
// --restart-seed-handback for a deliberately fresh-session child (e.g. `claude -p`) that the #3055
// --continue reattach does not serve. On success it returns the augmented command, the "seed-prompt"
// handback mode, and injected=true; it LOGS the dropped approx-token/byte count whenever the seed is
// truncated past guardSeedPromptTokenBudget. It is a NO-OP — (command, "", false) — for an
// unrecognized agent (fak never guesses a foreign tool's prompt syntax; the seed stays on disk
// unread) or an empty seed. Idempotent across repeated restarts: a prior injected seed VALUE is
// replaced with the fresher one rather than stacking a second flag. The input command is never
// mutated in place.
func writeGuardSeedPromptFile(seed string) (string, error) {
	dir, err := guardSessionTempDir("seedprompt")
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, "restart-seed.txt")
	if err := os.WriteFile(path, []byte(seed), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

func guardSeedPromptRelaunchCommand(command []string, agentName, seedText string, log io.Writer) (out []string, handback string, injected bool) {
	seed := strings.TrimSpace(seedText)
	flag, ok := guardSeedPromptFlagForAgent(agentName)
	if !ok || seed == "" {
		return command, "", false
	}
	bounded, droppedTokens := guardBoundSeedPrompt(seed, guardSeedPromptTokenBudget)
	if droppedTokens > 0 && log != nil {
		fmt.Fprintf(log, "fak guard: seed-prompt handback bounded carryover for %s to %d approx-tokens (budget %d); dropped %d approx-tokens / %d bytes — no silent truncation\n",
			guardAgentBaseName(agentName), guardApproxTokens(bounded), guardSeedPromptTokenBudget, droppedTokens, len(seed)-len(bounded))
	}
	out = make([]string, len(command), len(command)+2)
	copy(out, command)
	if guardAgentBaseName(agentName) == "claude" {
		path, err := writeGuardSeedPromptFile(bounded)
		if err != nil {
			if log != nil {
				fmt.Fprintf(log, "fak guard: restart seed prompt file write failed: %v; seed JSON remains available for recovery\n", err)
			}
			return command, "", false
		}
		fileFlag := flag + "-file"
		for i := 1; i+1 < len(out); i++ {
			if out[i] == flag || out[i] == fileFlag {
				out[i], out[i+1] = fileFlag, path
				return out, guardRestartHandbackSeedPrompt, true
			}
		}
		out = append(out, fileFlag, path)
		return out, guardRestartHandbackSeedPrompt, true
	}
	for i := 1; i+1 < len(out); i++ {
		if out[i] == flag {
			out[i+1] = bounded
			return out, guardRestartHandbackSeedPrompt, true
		}
	}
	out = append(out, flag, bounded)
	return out, guardRestartHandbackSeedPrompt, true
}

func guardRestartLimitStatus(limit int, ev guardBudgetRestartEvent) string {
	reason := strings.TrimSpace(ev.Reason)
	if reason == "" {
		reason = "BUDGET_CONTEXT_EXHAUSTED"
	}
	continuity := "degraded"
	if ev.Baton.valid() && strings.Contains(ev.SeedText, "BATON\n") {
		continuity = "baton"
	}
	if strings.TrimSpace(ev.ToTraceID) == "" && strings.TrimSpace(ev.SeedFile) == "" && strings.TrimSpace(ev.SeedText) == "" {
		continuity = "blocked"
	}
	next := "raise --restart-limit or restart manually after the budget window clears"
	if trace := strings.TrimSpace(ev.ToTraceID); trace != "" {
		next = "raise --restart-limit or restart the child with FAK_RESET_TRACE_ID=" + trace
	}
	if seed := strings.TrimSpace(ev.SeedFile); seed != "" {
		// ToSlash: %q below escapes backslashes, so an unconverted Windows path
		// (filepath.Join's native separator) would render as seeds\\reset.json —
		// doubled backslashes a plain-substring check (or a human) never expects.
		// Forward-slash normalization keeps the seed path byte-identical in the
		// %q-quoted next_action field on every OS.
		seed = strings.ReplaceAll(filepath.ToSlash(seed), "\\", "/")
		next += " and FAK_RESET_SEED_FILE=" + seed
	}
	return fmt.Sprintf("fak guard: managed-context status reset_limit limit=%d reason=%s continuity=%s next_action=%q",
		limit, reason, continuity, next)
}

// guardNoProgressRestartLimitDefault is the K-consecutive-no-progress reap threshold (#4609):
// after this many budget restarts that each landed NO new commit (HEAD unchanged since the prior
// restart), the guard reaps a degenerate restart-storming worker EARLY rather than let it ride the
// raw --restart-limit all the way to the wall-clock backstop doing nothing. A restart that DID move
// HEAD resets the counter, so a healthy-but-slow COMMITTING worker earns back its full runway and is
// never reaped here — the raw --restart-limit (16, pinned by TestClaudeGuardRestartLimit) stays the
// healthy-worker bound. This reap is a strictly earlier, progress-aware trip on top of it, NOT a
// replacement, which is why the raw cap value is deliberately left unchanged (see #4609: lowering it
// to 6 would reap a healthy committing worker at ~40% of its runway).
const (
	guardNoProgressRestartLimitDefault = 6
	// Equivalent budget denials are a stronger stall signal than an unchanged git HEAD:
	// stop the third identical cycle while leaving changing causes their full runway.
	guardEquivalentRestartLimit = 3
)

// guardNoProgressRestartLimit resolves the no-progress reap threshold from the environment, falling
// back to guardNoProgressRestartLimitDefault. A value of 0 (or a negative/garbage override) disables
// the reap, leaving only the raw --restart-limit backstop — the same fail-safe the reap already takes
// when git offers no HEAD signal.
func guardNoProgressRestartLimit() int {
	if v := strings.TrimSpace(os.Getenv("FLEET_CLAUDE_GUARD_NO_PROGRESS_LIMIT")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return guardNoProgressRestartLimitDefault
}

// guardNoProgressStep folds one restart's HEAD observation into the (checkpoint, counter) pair the
// #4609 reap rides: a HEAD that advanced past the checkpoint resets the counter and moves the
// checkpoint (a commit landed — the worker earns back its runway); an unchanged HEAD increments the
// counter. An empty cur (git offered no signal at this restart) leaves BOTH untouched, so a transient
// read miss neither trips nor resets the reap. Pure, so the reset/increment discipline is unit-tested
// without standing up the supervision loop.
type guardEquivalentRestarts struct {
	cause string
	count int
}

func (s guardEquivalentRestarts) step(reason string) guardEquivalentRestarts {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "BUDGET_CONTEXT_EXHAUSTED"
	}
	if reason != s.cause {
		return guardEquivalentRestarts{cause: reason, count: 1}
	}
	s.count++
	return s
}

func guardEquivalentRestartStatus(s guardEquivalentRestarts, ev guardBudgetRestartEvent) string {
	return fmt.Sprintf("fak guard: managed-context status restart_exhausted count=%d dominant_cause=%s from_trace=%s next_action=%q",
		s.count, s.cause, strings.TrimSpace(ev.FromTraceID),
		"equivalent guard restart cycle repeated; escalate the dominant cause instead of retrying")
}

func guardNoProgressStep(prevHead, cur string, counter int) (string, int) {
	if strings.TrimSpace(cur) == "" {
		return prevHead, counter
	}
	if cur != prevHead {
		return cur, 0
	}
	return prevHead, counter + 1
}

// guardNoProgressReapStatus is the one-line stderr banner the no-progress reap emits, mirroring
// guardRestartLimitStatus's managed-context-status shape so an operator greps both reap paths the
// same way. It names the consecutive-no-progress depth that tripped and the originating trace, and
// points at the tuning knob.
func guardNoProgressReapStatus(limit int, ev guardBudgetRestartEvent) string {
	reason := strings.TrimSpace(ev.Reason)
	if reason == "" {
		reason = "BUDGET_CONTEXT_EXHAUSTED"
	}
	return fmt.Sprintf("fak guard: managed-context status no_progress_reap limit=%d reason=%s from_trace=%s next_action=%q",
		limit, reason, strings.TrimSpace(ev.FromTraceID),
		"worker restarted with no new commit; raise FLEET_CLAUDE_GUARD_NO_PROGRESS_LIMIT or investigate the stall")
}

// guardChildIsLaunchFailure reports whether runErr is a FAILURE TO LAUNCH (a spawn/exec error
// — the binary is missing, not executable, a bad path) rather than a normal run that exited
// non-zero. An *exec.ExitError means the child DID start and then exited, so it is never a
// launch failure; a nil error is a clean run. Everything else (exec.Error, a PathError from
// exec.Command().Run()) is the child never starting — the one case the compact/animate launch
// spills the full startup report for. Pure, so the classification is unit-tested.
func guardChildIsLaunchFailure(runErr error) bool {
	if runErr == nil {
		return false
	}
	_, isExit := runErr.(*exec.ExitError)
	return !isExit
}

// guardDumpStartupReportOnLaunchFail spills the full recorded startup report to w when the
// child failed to launch. This is the one case the compact/animate banner deliberately
// withholds the wall of text for: on a launch failure no agent TUI ever took the terminal, so
// the full floor/hook/auth detail is exactly what the operator needs, co-located with the
// error. enabled is false when the full report already streamed at boot (--banner=full) so the
// text is never printed twice; a nil Server or an unrecorded report is a silent no-op. It reads
// the report off the gateway and hands the formatting to the pure guardWriteLaunchFailReport.
func guardDumpStartupReportOnLaunchFail(w io.Writer, srv *gateway.Server, enabled bool) {
	if srv == nil {
		return
	}
	guardWriteLaunchFailReport(w, srv.StartupReport(), enabled)
}

// guardWriteLaunchFailReport writes the report under a "launch failed" header when enabled and
// the report is non-empty, and is a no-op otherwise. Split from the Server read so the exact
// header + body format is unit-tested without standing up a gateway.
func guardWriteLaunchFailReport(w io.Writer, report string, enabled bool) {
	if !enabled || strings.TrimSpace(report) == "" {
		return
	}
	report = strings.TrimRight(report, "\n")
	fmt.Fprintln(w, "fak guard: launch failed — full startup report (the detail an attended launch keeps in `fak info --startup`):")
	fmt.Fprintln(w, report)
}

// runGuardChildAndReport runs the wrapped agent to completion, tears the gateway down,
// prints the session's adjudication + journal summary (unless quiet), flushes the durable
// trail, and exits with the child's own code — surfacing a gateway-mid-session failure as
// a non-silent error so a clean child exit never hides a downed adjudication boundary.
//
// Before reporting a non-zero exit, it gives guardMaybeRecoverAuthCrash (the mid-session
// counterpart to the #1834 pre-spawn rehydrate rung) a chance to diagnose an expired
// subscription token and, if a fresh login lands within the recovery window, relaunch the SAME
// command with a resume flag appended — so a crash caused by auth expiry self-heals within this
// guarded session instead of always needing a manual re-run. credPath is empty when guard is not
// pinning the Claude subscription upstream, which makes the check an unconditional no-op there.
//
// dumpStartupOnLaunchFail spills the full startup report to stderr if the child never starts
// (guardChildIsLaunchFailure) — set by the caller for every banner mode except --banner=full,
// which already streamed it at boot.
