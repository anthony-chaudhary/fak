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
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

const (
	cronOpenCodeRunSchema = "fak-opencode-run/1"
)

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
	Command     []string
	Stdout      io.Writer
	Stderr      io.Writer
	ChildStdout io.Writer
	Passthrough bool
	EmitReceipt bool
	Env         []string
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

	// Child process execution with timeout context
	var ctx context.Context
	var cancel context.CancelFunc
	if opts.Timeout > 0 {
		ctx, cancel = context.WithTimeout(context.Background(), opts.Timeout)
	} else {
		ctx, cancel = context.WithCancel(context.Background())
	}
	defer cancel()

	cmdName := opts.Command[0]
	cmdRest := opts.Command[1:]
	c := exec.CommandContext(ctx, cmdName, cmdRest...)
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

	startTime := time.Now()
	runErr := c.Run()
	endedTime := time.Now()
	durationMS := endedTime.Sub(startTime).Milliseconds()

	sessionID := extractOpenCodeSessionID(childStdoutBuf.String())
	if sessionID == "" && childStderrBuf.Len() > 0 {
		sessionID = extractOpenCodeSessionID(childStderrBuf.String())
	}

	var exitCode int
	var outcome string
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

	receipt := OpenCodeRunReceipt{
		Schema:     cronOpenCodeRunSchema,
		RunID:      runID,
		SessionID:  sessionID,
		ExitCode:   exitCode,
		Outcome:    outcome,
		StartedAt:  startTime.UTC().Format(time.RFC3339),
		EndedAt:    endedTime.UTC().Format(time.RFC3339),
		DurationMS: durationMS,
		WitnessRef: nil,
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
	passthrough := fs.Bool("passthrough", false, "pass child stdout through to stdout during execution")

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
		Job:         *job,
		Ledger:      *ledger,
		Interval:    *interval,
		Timeout:     *timeout,
		At:          *at,
		Slot:        *slot,
		RunID:       *runID,
		Passthrough: *passthrough,
		EmitReceipt: true,
		Command:     cmdArgs,
		Stdout:      stdout,
		Stderr:      stderr,
	}

	receipt, err := RunScheduledOpenCode(opts)
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
