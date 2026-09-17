// cron_opencode.go retains terminal outcomes, session joins, and execution receipts
// for scheduled OpenCode launches (#11953). It wraps OpenCode child execution with
// optional (job, slot) CAS deduplication, bounded timeout execution, process tree
// termination, session ID extraction from stdout/stderr streams, and ledger-witnessed
// run receipts.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

const (
	cronOpenCodeRunSchema  = "fak-opencode-run/1"
	cronOpenCodeMaxTimeout = 2 * time.Hour

	// cronOpenCodeDefaultCrashRetries is the number of extra child attempts
	// granted after a Bun SIGSEGV terminal outcome, so the default is 2 total
	// attempts (1 initial + 1 retry).
	cronOpenCodeDefaultCrashRetries = 1

	// cronOpenCodeVersionProbeTimeout bounds the best-effort version probe so
	// runtime-version capture can never stall the hot path (#1316).
	cronOpenCodeVersionProbeTimeout = 5 * time.Second
)

func cronOpenCodeEffectiveTimeout(requested time.Duration) time.Duration {
	if requested <= 0 {
		return 45 * time.Minute
	}
	if requested > cronOpenCodeMaxTimeout {
		return cronOpenCodeMaxTimeout
	}
	return requested
}

// OpenCodeRunReceipt records the terminal outcome, duration, and session join of an
// OpenCode execution. WitnessRef is explicitly serialized as null when absent.
type OpenCodeRunReceipt struct {
	Schema     string  `json:"schema"` // "fak-opencode-run/1"
	RunID      string  `json:"run_id"`
	SessionID  string  `json:"session_id"`
	ExitCode   int     `json:"exit_code"`
	Outcome    string  `json:"outcome"` // "succeeded" | "failed" | "timeout"
	StartedAt  string  `json:"started_at"`
	EndedAt    string  `json:"ended_at"`
	DurationMS int64   `json:"duration_ms"`
	WitnessRef *string `json:"witness_ref"` // explicitly null (pointer without omitempty)

	StartError    string `json:"start_error,omitempty"`    // bounded tail of child output when the session never started
	StartupFailed bool   `json:"startup_failed,omitempty"` // true iff exit != 0 AND SessionID == ""

	// Bun-crash retry/version provenance (#1316). All additive + omitempty so the
	// fak-opencode-run/1 schema stays backward-compatible.
	Attempts        int    `json:"attempts,omitempty"`         // total child executions performed (>=1)
	CrashSignature  bool   `json:"crash_signature,omitempty"`  // child output ever carried the Bun SIGSEGV signature
	CrashRecovered  bool   `json:"crash_recovered,omitempty"`  // final outcome succeeded after a Bun crash
	OpenCodeVersion string `json:"opencode_version,omitempty"` // best-effort `opencode --version`
	BunVersion      string `json:"bun_version,omitempty"`      // best-effort Bun runtime version

	// Session token/cost/model join (#1559). Additive + omitempty so a receipt with
	// no session join stays byte-identical to the fak-opencode-run/1 shape above.
	TokensTotal int64   `json:"tokens_total,omitempty"` // total billed tokens the joined session moved
	CostUSD     float64 `json:"cost_usd,omitempty"`     // observed billed USD (provider ledger); 0 = unmeasured
	ModelID     string  `json:"model_id,omitempty"`     // model that served the session
	ProviderID  string  `json:"provider_id,omitempty"`  // provider that served the session

	// Provider-failure classification (#1866). Additive + omitempty so a receipt
	// with no class stays byte-identical to the fak-opencode-run/1 shape above.
	// Present on every failed/timeout run as "transient" | "ongoing"; absent on a
	// succeeded run. "ongoing" is the conservative fallback for any failure the
	// runner cannot positively identify as a bounded provider blip.
	FailureClass string `json:"failure_class,omitempty"`
}

// cronOpenCodeBunCrashSignature reports whether captured child output carries the
// Bun SIGSEGV crash fingerprint. This is the ONLY admission for a retry: a bare
// non-zero exit (real opencode error, bad flags, model error) must never retry.
func cronOpenCodeBunCrashSignature(output string) bool {
	if output == "" {
		return false
	}
	lower := strings.ToLower(output)
	return strings.Contains(lower, "segmentation fault") &&
		(strings.Contains(lower, "bun has crashed") || strings.Contains(lower, "bun.report"))
}

// Regexes for the TRANSIENT provider-failure signals (#1866). Each requires an
// explicit HTTP/status context token adjacent to the code: we bias HARD toward a
// false NEGATIVE (unrecognized evidence -> "ongoing", the conservative default)
// because a false POSITIVE would hide a real outage as a non-throttling blip.
var (
	// reCronFailureStatus matches an HTTP status code that is unambiguous ONLY in
	// a status context: "HTTP/1.1 503", "status 502", "status_code=429",
	// `"status":429` (JSON), "code: 500", "response 503". A bare "500" embedded in
	// arbitrary output (latency ms, token counts) does NOT match.
	reCronFailureStatus = regexp.MustCompile(`(?i)(?:\bhttp(?:/[0-9.]+)?\b|\bstatus(?:[_ -]?code)?\b|\bcode\b|\bresponse\b)["']?\s*[:=]?\s*["']?\s*(?:<[^>]*>\s*)?\b(500|502|503|504|429)\b`)
	// reCronFailureStatusPost matches "<code> <reason phrase>" forms such as
	// "503 Service Unavailable" / "429 Too Many Requests" / "502 Bad Gateway".
	reCronFailureStatusPost = regexp.MustCompile(`(?i)\b(500|502|503|504|429)\s+(?:internal server error|bad gateway|service unavailable|gateway time-?out|too many requests)\b`)
	// reCronFailureStatusPre matches "<reason phrase> <code>" forms such as
	// "Internal Server Error 500" / "Too Many Requests 429".
	reCronFailureStatusPre = regexp.MustCompile(`(?i)\b(?:internal server error|bad gateway|service unavailable|gateway time-?out|too many requests)\b[\s:]*\b(500|502|503|504|429)\b`)
	// reCronConnectionReset matches a transport-level reset ("connection reset by
	// peer"), an unambiguously transient transport failure.
	reCronConnectionReset = regexp.MustCompile(`(?i)\bconnection reset(?: by peer)?\b`)
)

// cronClassifyFailureClass maps a terminal run outcome plus its bounded child
// output to the provider-failure class the ops budget guard consumes (#1866).
// It returns:
//
//	""          when outcome is not a failure (only "failed"/"timeout" are classed)
//	"transient" for a bounded provider blip: HTTP 500/502/503/504, HTTP 429, or a
//	            connection reset
//	"ongoing"   for a paused/forbidden org (HTTP 405), exhausted credit, or ANY
//	            unclassifiable failure
//
// "transient" is the narrow, evidence-gated class; "ongoing" is the conservative
// fallback so a failed/timeout receipt is NEVER left without a class (an absent
// class counts as ONGOING downstream anyway, and explicit is the honest form).
func cronClassifyFailureClass(outcome, output string) string {
	switch strings.ToLower(strings.TrimSpace(outcome)) {
	case "failed", "timeout":
	default:
		return ""
	}
	if output != "" {
		// Paused/forbidden org (HTTP 405, e.g. a body saying the account is
		// paused) is an ONGOING, operator-actionable condition - never transient.
		if strings.Contains(strings.ToLower(output), "paused") {
			return "ongoing"
		}
		if reCronConnectionReset.MatchString(output) {
			return "transient"
		}
		if m := reCronFailureStatus.FindStringSubmatch(output); len(m) > 1 {
			return cronFailureStatusClass(m[1])
		}
		if m := reCronFailureStatusPost.FindStringSubmatch(output); len(m) > 1 {
			return cronFailureStatusClass(m[1])
		}
		if m := reCronFailureStatusPre.FindStringSubmatch(output); len(m) > 1 {
			return cronFailureStatusClass(m[1])
		}
	}
	return "ongoing"
}

// cronFailureStatusClass maps a matched HTTP status code to a failure class:
// 429 and the 5xx server-blip class are TRANSIENT; everything else is ONGOING.
func cronFailureStatusClass(code string) string {
	switch code {
	case "429", "500", "502", "503", "504":
		return "transient"
	}
	return "ongoing"
}

var reOpenCodeBunVersion = regexp.MustCompile(`(?i)\bbun\s+v([0-9][0-9A-Za-z_.-]*)`)

// cronExtractBunVersion best-effort extracts a Bun runtime version from child
// output (the crash banner already prints `Bun vX.Y.Z`). Cheap: no extra process.
func cronExtractBunVersion(output string) string {
	if m := reOpenCodeBunVersion.FindStringSubmatch(output); len(m) > 1 {
		return m[1]
	}
	return ""
}

// cronIsOpenCodeCommand reports whether cmdName names the opencode binary (by
// base name, ignoring extension and case). Used to keep the version probe off
// the hot path for arbitrary/stub commands (#1316).
func cronIsOpenCodeCommand(cmdName string) bool {
	base := strings.ToLower(filepath.Base(strings.TrimSpace(cmdName)))
	base = strings.TrimSuffix(base, filepath.Ext(base))
	return base == "opencode"
}

// cronProbeOpenCodeVersion best-effort captures `opencode --version`. It is
// bounded and failure-tolerant: a missing/odd binary or a timeout yields "" and
// NEVER fails the run (#1316).
func cronProbeOpenCodeVersion(cmdName string, workdir string, env []string) string {
	if cmdName == "" {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), cronOpenCodeVersionProbeTimeout)
	defer cancel()
	c := exec.CommandContext(ctx, cmdName, "--version")
	configureDispatchHelperCommand(c)
	if workdir != "" {
		c.Dir = workdir
	}
	if len(env) > 0 {
		c.Env = append(os.Environ(), env...)
	}
	out, err := c.Output()
	if err != nil && len(out) == 0 {
		return ""
	}
	return cronBoundedOutputTail(string(out), 256)
}

// cronBoundedOutputTail returns at most the last maxBytes of s collapsed to a
// single trimmed line, so child startup errors stay bounded in receipts. A cut
// landing mid-rune is trimmed to the nearest valid UTF-8 boundary.
func cronBoundedOutputTail(s string, maxBytes int) string {
	if maxBytes > 0 && len(s) > maxBytes {
		s = s[len(s)-maxBytes:]
		for len(s) > 0 && !utf8.RuneStart(s[0]) {
			s = s[1:]
		}
	}
	return strings.Join(strings.Fields(s), " ")
}

// ScheduledOpenCodeOptions carries the configuration for a scheduled OpenCode execution.
type ScheduledOpenCodeOptions struct {
	Job         string
	Ledger      string
	Interval    time.Duration
	Timeout     time.Duration
	At          string
	Slot        string
	RunID       string
	Until       string
	Workdir     string
	UnloadPlist string
	Command     []string
	Stdout      io.Writer
	Stderr      io.Writer
	ChildStdout io.Writer
	Passthrough bool
	EmitReceipt bool
	Env         []string

	// CrashRetries is the number of extra child attempts granted after a Bun
	// SIGSEGV terminal outcome (total attempts = CrashRetries + 1). 0 disables
	// retry. Negative values are treated as 0.
	CrashRetries int
}

// RunScheduledOpenCode executes an OpenCode command with bounded timeout and optional
// ledger CAS deduplication, returning an OpenCodeRunReceipt.
func RunScheduledOpenCode(opts ScheduledOpenCodeOptions) (OpenCodeRunReceipt, error) {
	runID := opts.RunID
	if strings.TrimSpace(runID) == "" {
		runID = fmt.Sprintf("opencode-run-%d", time.Now().UnixNano())
	}

	if len(opts.Command) == 0 {
		if opts.Stderr != nil {
			fmt.Fprintln(opts.Stderr, "fak cron opencode: command is required")
		}
		return OpenCodeRunReceipt{
			Schema:     cronOpenCodeRunSchema,
			RunID:      runID,
			ExitCode:   2,
			Outcome:    "failed",
			WitnessRef: nil,
		}, errors.New("fak cron opencode: command is required")
	}

	// Expiration check: ticks occurring after opts.Until are marked expired (#11953)
	if strings.TrimSpace(opts.Until) != "" {
		untilTime, err := time.Parse(time.RFC3339, strings.TrimSpace(opts.Until))
		if err != nil {
			if d, durErr := time.ParseDuration(strings.TrimSpace(opts.Until)); durErr == nil && d > 0 {
				untilTime = time.Now().Add(d)
			} else {
				if opts.Stderr != nil {
					fmt.Fprintf(opts.Stderr, "fak cron opencode: invalid --until %q (must be RFC3339 or duration)\n", opts.Until)
				}
				return OpenCodeRunReceipt{
					Schema:     cronOpenCodeRunSchema,
					RunID:      runID,
					ExitCode:   2,
					Outcome:    "failed",
					WitnessRef: nil,
				}, fmt.Errorf("fak cron opencode: invalid --until %q: %w", opts.Until, err)
			}
		}

		checkTime := time.Now()
		if strings.TrimSpace(opts.At) != "" {
			if t, err := time.Parse(time.RFC3339, strings.TrimSpace(opts.At)); err == nil {
				checkTime = t
			}
		}

		if checkTime.After(untilTime) {
			nowStr := checkTime.UTC().Format(time.RFC3339)
			receipt := OpenCodeRunReceipt{
				Schema:     cronOpenCodeRunSchema,
				RunID:      runID,
				SessionID:  "",
				ExitCode:   0,
				Outcome:    "expired",
				StartedAt:  nowStr,
				EndedAt:    nowStr,
				DurationMS: 0,
				WitnessRef: nil,
			}
			if opts.Ledger != "" {
				_ = cronAppendJSONL(opts.Ledger, receipt)
			}
			if strings.TrimSpace(opts.UnloadPlist) != "" && runtime.GOOS == "darwin" {
				_ = exec.Command("launchctl", "unload", strings.TrimSpace(opts.UnloadPlist)).Run()
			}
			if opts.EmitReceipt && opts.Stdout != nil {
				enc := json.NewEncoder(opts.Stdout)
				enc.SetIndent("", "  ")
				_ = enc.Encode(receipt)
			}
			return receipt, nil
		}
	}

	// CAS lock handling when Ledger != "" && Job != "" && Interval > 0
	if opts.Ledger != "" && opts.Job != "" && opts.Interval > 0 {
		fireAt, slotKey, ok := resolveCronTimeAndSlot(opts.Stderr, "fak cron opencode", opts.At, opts.Slot, opts.Interval)
		if !ok {
			return OpenCodeRunReceipt{
				Schema:     cronOpenCodeRunSchema,
				RunID:      runID,
				ExitCode:   2,
				Outcome:    "failed",
				WitnessRef: nil,
			}, errors.New("resolve cron slot failed")
		}

		release, err := cronTickLock(opts.Ledger+".tick.lock", cronTickWait, cronTickTTL)
		if err != nil {
			if opts.Stderr != nil {
				fmt.Fprintf(opts.Stderr, "fak cron opencode: %v\n", err)
			}
			return OpenCodeRunReceipt{
				Schema:     cronOpenCodeRunSchema,
				RunID:      runID,
				ExitCode:   2,
				Outcome:    "failed",
				WitnessRef: nil,
			}, err
		}
		defer func() {
			if release != nil {
				_ = release()
			}
		}()

		fires, err := cronReadFires(opts.Ledger)
		if err != nil {
			if opts.Stderr != nil {
				fmt.Fprintf(opts.Stderr, "fak cron opencode: read ledger: %v\n", err)
			}
			return OpenCodeRunReceipt{
				Schema:     cronOpenCodeRunSchema,
				RunID:      runID,
				ExitCode:   2,
				Outcome:    "failed",
				WitnessRef: nil,
			}, err
		}

		for _, r := range fires {
			if r.Job == opts.Job && r.Slot == slotKey && r.Outcome == cronOutcomeFired {
				dupFireRec := cronFireRecord{
					Schema:   cronFireSchema,
					Job:      opts.Job,
					Slot:     slotKey,
					Interval: int64(opts.Interval.Seconds()),
					Outcome:  cronOutcomeDeduped,
					FiredAt:  fireAt.Format(time.RFC3339),
				}
				_ = cronAppendFire(opts.Ledger, dupFireRec)

				nowStr := fireAt.UTC().Format(time.RFC3339)
				receipt := OpenCodeRunReceipt{
					Schema:     cronOpenCodeRunSchema,
					RunID:      runID,
					SessionID:  "",
					ExitCode:   cronExitDeduped,
					Outcome:    "failed",
					StartedAt:  nowStr,
					EndedAt:    nowStr,
					DurationMS: 0,
					WitnessRef: nil,
				}
				if opts.EmitReceipt && opts.Stdout != nil {
					enc := json.NewEncoder(opts.Stdout)
					enc.SetIndent("", "  ")
					_ = enc.Encode(receipt)
				}
				return receipt, nil
			}
		}

		// Fresh: append fired record
		fireRec := cronFireRecord{
			Schema:   cronFireSchema,
			Job:      opts.Job,
			Slot:     slotKey,
			Interval: int64(opts.Interval.Seconds()),
			Outcome:  cronOutcomeFired,
			FiredAt:  fireAt.Format(time.RFC3339),
		}
		if err := cronAppendFire(opts.Ledger, fireRec); err != nil {
			if opts.Stderr != nil {
				fmt.Fprintf(opts.Stderr, "fak cron opencode: append fire: %v\n", err)
			}
			return OpenCodeRunReceipt{
				Schema:     cronOpenCodeRunSchema,
				RunID:      runID,
				ExitCode:   2,
				Outcome:    "failed",
				WitnessRef: nil,
			}, err
		}

		// Release tick lock before child execution
		if release != nil {
			_ = release()
			release = nil
		}
	}

	// Child process execution with bounded timeout context (#11953).
	// A Bun SIGSEGV after the session started is a FALSE failure (#1316): the
	// child did its work but the Bun runtime segfaulted on exit. When the crash
	// signature is present we re-exec the command (child only — the CAS/dedup
	// fire step above is never re-run) up to a small bounded number of times.
	cmdName := opts.Command[0]
	cmdRest := opts.Command[1:]

	// Cheap pre-flight: a bare command name with no path separator must resolve
	// on PATH, otherwise the child can never start; fail loud without spawning.
	if !strings.ContainsRune(cmdName, filepath.Separator) && !strings.Contains(cmdName, "/") {
		if _, lookErr := exec.LookPath(cmdName); lookErr != nil {
			if opts.Stderr != nil {
				fmt.Fprintf(opts.Stderr, "fak cron opencode: command not found: %s\n", cmdName)
			}
			return OpenCodeRunReceipt{
				Schema:        cronOpenCodeRunSchema,
				RunID:         runID,
				ExitCode:      2,
				Outcome:       "failed",
				StartupFailed: true,
				StartError:    "command not found: " + cmdName,
				WitnessRef:    nil,
			}, fmt.Errorf("fak cron opencode: command not found: %s", cmdName)
		}
	}

	effectiveTimeout := cronOpenCodeEffectiveTimeout(opts.Timeout)
	crashRetries := opts.CrashRetries
	if crashRetries < 0 {
		crashRetries = 0
	}
	maxAttempts := crashRetries + 1

	// Best-effort installed runtime versions, captured once (never fails the run).
	// The probe only makes sense for the real opencode binary and costs an extra
	// process, so it is skipped for arbitrary/stub commands on the hot path.
	openCodeVersion := ""
	if cronIsOpenCodeCommand(cmdName) {
		openCodeVersion = cronProbeOpenCodeVersion(cmdName, opts.Workdir, opts.Env)
	}

	var (
		sessionID      string
		exitCode       int
		outcome        string
		runErr         error
		startTime      time.Time
		endedTime      time.Time
		attempts       int
		crashSignature bool
		bunVersion     string
		lastStderr     string
		lastStdout     string
	)

	for {
		attempts++

		ctx, cancel := context.WithTimeout(context.Background(), effectiveTimeout)

		c := exec.CommandContext(ctx, cmdName, cmdRest...)
		configureDispatchHelperCommand(c)
		if opts.Workdir != "" {
			c.Dir = opts.Workdir
		}
		if len(opts.Env) > 0 {
			c.Env = append(os.Environ(), opts.Env...)
		}

		var childStdoutBuf bytes.Buffer
		var childStderrBuf bytes.Buffer

		var stdoutWriters []io.Writer
		stdoutWriters = append(stdoutWriters, &childStdoutBuf)
		if opts.ChildStdout != nil {
			stdoutWriters = append(stdoutWriters, opts.ChildStdout)
		}
		if opts.Passthrough && opts.Stdout != nil {
			stdoutWriters = append(stdoutWriters, opts.Stdout)
		}
		c.Stdout = io.MultiWriter(stdoutWriters...)

		var stderrWriters []io.Writer
		stderrWriters = append(stderrWriters, &childStderrBuf)
		if opts.Stderr != nil {
			stderrWriters = append(stderrWriters, opts.Stderr)
		}
		c.Stderr = io.MultiWriter(stderrWriters...)

		c.WaitDelay = 5 * time.Second
		c.Cancel = func() error {
			if c.Process != nil && c.Process.Pid > 0 {
				cronRunKillTree(c.Process.Pid)
			}
			return nil
		}

		startTime = time.Now()
		runErr = c.Run()
		endedTime = time.Now()

		lastStdout = childStdoutBuf.String()
		lastStderr = childStderrBuf.String()

		sessionID = extractOpenCodeSessionID(lastStdout)
		if sessionID == "" && lastStderr != "" {
			sessionID = extractOpenCodeSessionID(lastStderr)
		}

		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			outcome = "timeout"
			var exitErr *exec.ExitError
			if errors.As(runErr, &exitErr) {
				exitCode = exitErr.ExitCode()
			} else {
				exitCode = 124
			}
		} else if runErr != nil {
			outcome = "failed"
			var exitErr *exec.ExitError
			if errors.As(runErr, &exitErr) {
				exitCode = exitErr.ExitCode()
			} else {
				exitCode = 1
			}
		} else {
			outcome = "succeeded"
			exitCode = 0
		}
		cancel()

		if bv := cronExtractBunVersion(lastStderr + "\n" + lastStdout); bv != "" {
			bunVersion = bv
		}

		// A crash is only admitted from the Bun SIGSEGV signature in the child
		// output — never from the exit code alone. A genuine non-crash failure
		// must not retry. crashSignature is sticky across attempts so the receipt
		// records whether the run EVER hit the Bun crash.
		attemptCrash := cronOpenCodeBunCrashSignature(lastStderr) || cronOpenCodeBunCrashSignature(lastStdout)
		crashSignature = crashSignature || attemptCrash
		if !attemptCrash || attempts >= maxAttempts {
			break
		}
		if opts.Stderr != nil {
			fmt.Fprintf(opts.Stderr, "fak cron opencode: Bun crash detected on attempt %d/%d; retrying child execution\n", attempts, maxAttempts)
		}
	}

	durationMS := endedTime.Sub(startTime).Milliseconds()

	crashRecovered := crashSignature && outcome == "succeeded"

	receipt := OpenCodeRunReceipt{
		Schema:          cronOpenCodeRunSchema,
		RunID:           runID,
		SessionID:       sessionID,
		ExitCode:        exitCode,
		Outcome:         outcome,
		StartedAt:       startTime.UTC().Format(time.RFC3339),
		EndedAt:         endedTime.UTC().Format(time.RFC3339),
		DurationMS:      durationMS,
		WitnessRef:      nil,
		Attempts:        attempts,
		CrashSignature:  crashSignature,
		CrashRecovered:  crashRecovered,
		OpenCodeVersion: openCodeVersion,
		BunVersion:      bunVersion,

		// Provider-failure classification (#1866): stamped on failed/timeout runs
		// only (empty on succeeded). cronClassifyFailureClass returns "ongoing" for
		// any unclassifiable failure, so a failed/timeout receipt is never left
		// classless. Additive + omitempty: succeeded runs stay byte-identical.
		FailureClass: cronClassifyFailureClass(outcome, lastStderr+"\n"+lastStdout),
	}

	// Nonzero exit with no session created means OpenCode never started; surface
	// the child's bounded stderr (or stdout fallback) so the failure is not silent.
	if runErr != nil && sessionID == "" && outcome != "timeout" {
		tail := lastStderr
		if strings.TrimSpace(tail) == "" {
			tail = lastStdout
		}
		receipt.StartupFailed = true
		receipt.StartError = cronBoundedOutputTail(tail, 2048)
		if receipt.StartError == "" {
			receipt.StartError = runErr.Error()
		}
	}

	// Session token/cost/model join (#1559): best-effort, read-only, and never
	// fatal. A receipt with no join keeps the four fields at zero and stays
	// byte-identical (omitempty).
	if opts.EmitReceipt && receipt.SessionID != "" {
		cronPopulateReceiptSessionJoin(&receipt)
	}

	if opts.Ledger != "" {
		if err := cronAppendJSONL(opts.Ledger, receipt); err != nil {
			if opts.Stderr != nil {
				fmt.Fprintf(opts.Stderr, "fak cron opencode: append receipt: %v\n", err)
			}
			return receipt, err
		}
	}

	if opts.EmitReceipt && opts.Stdout != nil {
		enc := json.NewEncoder(opts.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(receipt)
	}

	return receipt, nil
}

// runCronOpenCode implements `fak cron opencode`.
func runCronOpenCode(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("cron opencode", flag.ContinueOnError)
	fs.SetOutput(stderr)

	job := fs.String("job", "", "job/loop id")
	ledger := fs.String("ledger", "", "witness ledger path, JSONL")
	interval := fs.Duration("interval", 0, "firing cadence; tick is quantized to this slot")
	timeout := fs.Duration("timeout", 0, "command execution timeout")
	at := fs.String("at", "", "wall-clock tick time (RFC3339); default now — injectable for tests")
	slot := fs.String("slot", "", "override computed slot key directly")
	runID := fs.String("run-id", "", "explicit run ID")
	until := fs.String("until", "", "expiration deadline (RFC3339); executions after this time are marked expired")
	workdir := fs.String("workdir", "", "working directory for child command execution")
	unloadPlist := fs.String("unload-plist", "", "launchd plist to unload upon expiration (macOS)")
	passthrough := fs.Bool("passthrough", false, "pass child stdout through to stdout during execution")
	crashRetries := fs.Int("crash-retries", cronOpenCodeDefaultCrashRetries, "extra child attempts after a Bun SIGSEGV crash (N>=0; total attempts = N+1; 0 disables)")

	// Find the trailing "--" separator for command arguments
	dashIdx := -1
	for i, arg := range argv {
		if arg == "--" {
			dashIdx = i
			break
		}
	}

	var flagArgs []string
	var cmdArgs []string
	if dashIdx >= 0 {
		flagArgs = argv[:dashIdx]
		cmdArgs = argv[dashIdx+1:]
	} else {
		flagArgs = argv
	}

	if !parseFlags(fs, flagArgs) {
		return 2
	}

	// If no "--" was present, check if args remained in fs.Args()
	if dashIdx < 0 {
		cmdArgs = fs.Args()
	}

	if len(cmdArgs) == 0 {
		fmt.Fprintln(stderr, "fak cron opencode: command is required after --")
		return 2
	}

	opts := ScheduledOpenCodeOptions{
		Job:          *job,
		Ledger:       *ledger,
		Interval:     *interval,
		Timeout:      *timeout,
		At:           *at,
		Slot:         *slot,
		RunID:        *runID,
		Until:        *until,
		Workdir:      *workdir,
		UnloadPlist:  *unloadPlist,
		Passthrough:  *passthrough,
		EmitReceipt:  true,
		Command:      cmdArgs,
		Stdout:       stdout,
		Stderr:       stderr,
		CrashRetries: *crashRetries,
	}

	receipt, err := RunScheduledOpenCode(opts)
	if receipt.StartupFailed {
		fmt.Fprintf(stderr, "fak cron opencode: opencode failed to start (no session created): %s\n", receipt.StartError)
	}
	if err != nil && receipt.ExitCode == 0 {
		return 2
	}
	return receipt.ExitCode
}

var (
	reOpenCodeSesPrefix     = regexp.MustCompile(`\b(ses_[a-zA-Z0-9_-]+)\b`)
	reOpenCodeSessionKeyVal = regexp.MustCompile(`(?i)(?:session[ _-]?id|session)\s*[:=]\s*["']?([a-zA-Z0-9_.-]+)["']?`)
)

// extractOpenCodeSessionID extracts an OpenCode session identifier from stdout/stderr text.
func extractOpenCodeSessionID(output string) string {
	lines := strings.Split(output, "\n")
	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}

		// Detect JSON lines and inspect fields
		if start := strings.Index(line, "{"); start >= 0 {
			if end := strings.LastIndex(line, "}"); end > start {
				var m map[string]any
				if err := json.Unmarshal([]byte(line[start:end+1]), &m); err == nil {
					if id := extractSessionIDFromMap(m); id != "" {
						return id
					}
				}
			}
		}

		// Detect plain text logs
		if m := reOpenCodeSesPrefix.FindStringSubmatch(line); len(m) > 1 {
			return m[1]
		}
		if m := reOpenCodeSessionKeyVal.FindStringSubmatch(line); len(m) > 1 {
			val := strings.TrimSpace(m[1])
			if val != "" && !strings.EqualFold(val, "null") && !strings.EqualFold(val, "none") && !strings.EqualFold(val, "nil") {
				return val
			}
		}
	}

	// Fallback for multi-line JSON
	if strings.Contains(output, "{") {
		var m map[string]any
		if err := json.Unmarshal([]byte(output), &m); err == nil {
			if id := extractSessionIDFromMap(m); id != "" {
				return id
			}
		}
	}

	return ""
}

func extractSessionIDFromMap(m map[string]any) string {
	// 1. session_id, sessionId, sessionID
	for _, k := range []string{"session_id", "sessionId", "sessionID"} {
		if v, ok := m[k]; ok {
			if s, ok := v.(string); ok && strings.TrimSpace(s) != "" {
				return strings.TrimSpace(s)
			}
		}
	}

	// 2. session (string or object with id / session_id / sessionId)
	if v, ok := m["session"]; ok {
		if s, ok := v.(string); ok && strings.TrimSpace(s) != "" {
			return strings.TrimSpace(s)
		}
		if sm, ok := v.(map[string]any); ok {
			for _, subk := range []string{"id", "session_id", "sessionId", "sessionID"} {
				if subv, ok := sm[subk]; ok {
					if subs, ok := subv.(string); ok && strings.TrimSpace(subs) != "" {
						return strings.TrimSpace(subs)
					}
				}
			}
		}
	}

	// 3. id (if matching ses_)
	if v, ok := m["id"]; ok {
		if s, ok := v.(string); ok {
			s = strings.TrimSpace(s)
			if strings.HasPrefix(s, "ses_") {
				return s
			}
		}
	}

	// 4. Common wrapper containers
	for _, wrapper := range []string{"data", "payload", "result", "event"} {
		if wv, ok := m[wrapper]; ok {
			if wm, ok := wv.(map[string]any); ok {
				if id := extractSessionIDFromMap(wm); id != "" {
					return id
				}
			}
		}
	}

	return ""
}

// cronReadOpenCodeReceipts reads well-formed OpenCode receipts from the ledger.
func cronReadOpenCodeReceipts(path string) ([]OpenCodeRunReceipt, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return jsonlledger.Parse(string(b), func(r OpenCodeRunReceipt) bool {
		return r.Schema == cronOpenCodeRunSchema && r.RunID != ""
	}), nil
}

// cronPopulateReceiptSessionJoin fills the additive session token/cost/model fields
// (#1559) on a receipt from the public session-join path already used by
// `fak dispatch sessions`: the gateway-usage ledger keyed by served session id. It is
// strictly BEST-EFFORT — any missing/unreadable ledger, unknown session, or unmeasured
// axis leaves the corresponding field at its zero value (omitted by omitempty) and can
// never fail the run. Cost/model/provider are not carried by the usage ledger, so those
// three stay zero until a priced per-session source lands (the join is additive).
func cronPopulateReceiptSessionJoin(receipt *OpenCodeRunReceipt) {
	if receipt == nil || strings.TrimSpace(receipt.SessionID) == "" {
		return
	}
	path := cronOpenCodeUsageLedgerPath()
	if path == "" {
		return
	}
	var best *gatewayusageledger.Counters
	var bestMs int64
	for _, r := range gatewayusageledger.ReadLedgerFile(path) {
		if strings.TrimSpace(r.SessionID) != receipt.SessionID {
			continue
		}
		if best == nil || r.UnixMillis > bestMs {
			c := r.Counters
			best = &c
			bestMs = r.UnixMillis
		}
	}
	if best == nil {
		return
	}
	receipt.TokensTotal = int64(best.InputTokens + best.OutputTokens + best.CachedPromptTokens + best.CacheCreationTokens)
}

// cronOpenCodeUsageLedgerPath resolves the default gateway-usage ledger under the repo
// root (FAK_OPENCODE_USAGE_LEDGER overrides for tests). Best-effort: returns "" when no
// repo root or ledger can be found, so the join is simply skipped.
func cronOpenCodeUsageLedgerPath() string {
	if override := strings.TrimSpace(os.Getenv("FAK_OPENCODE_USAGE_LEDGER")); override != "" {
		return override
	}
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	root := findRepoRoot(cwd)
	if strings.TrimSpace(root) == "" {
		return ""
	}
	return filepath.Join(root, filepath.FromSlash(gatewayusageledger.DefaultLedgerRel))
}
