// cron_run.go executes an admitted scheduled task with bounded witnessed outcomes
// (#11829). It enforces concurrency and deduplication guarantees using the
// (job, slot) compare-and-set and dup-tick lock from `fak cron fire` (#2886), then
// runs the trailing command under a context bounded by --timeout.
//
// Terminal outcomes (succeeded, failed, timeout) are recorded into the ledger
// along with the duration and child exit code.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

const (
	cronRunSchema        = "fak-cron-run/1"
	cronRunReceiptSchema = "fak-cron-run-receipt/1"

	cronRunStatusRan              = "ran"
	cronRunStatusSkippedDuplicate = "skipped_duplicate"
	cronRunStatusFailed           = "failed"
	cronRunStatusTimeout          = "timeout"

	cronRunOutcomeSucceeded        = "succeeded"
	cronRunOutcomeFailed           = "failed"
	cronRunOutcomeTimeout          = "timeout"
	cronRunOutcomeSkippedDuplicate = "skipped_duplicate"

	cronRunExitTimeout = 124
)

// cronRunKillTree is injectable for tests; defaults to procguard.KillPID.
var cronRunKillTree = procguard.KillPID

// cronRunRecord is one witnessed run execution record in the append-only ledger.
type cronRunRecord struct {
	Schema     string `json:"schema"`
	Job        string `json:"job"`
	Slot       string `json:"slot"`
	Outcome    string `json:"outcome"`
	Status     string `json:"status"`
	ExitCode   int    `json:"exit_code"`
	DurationMS int64  `json:"duration_ms"`
	Command    string `json:"command,omitempty"`
	Error      string `json:"error,omitempty"`
	StartedAt  string `json:"started_at"`
	FinishedAt string `json:"finished_at"`
}

// cronRunReceipt is the structured execution outcome receipt emitted on stdout.
type cronRunReceipt struct {
	Schema     string `json:"schema"`
	Job        string `json:"job"`
	Slot       string `json:"slot"`
	Status     string `json:"status"`
	Outcome    string `json:"outcome"`
	ExitCode   int    `json:"exit_code"`
	DurationMS int64  `json:"duration_ms"`
	Command    string `json:"command,omitempty"`
	Error      string `json:"error,omitempty"`
}

func emitCronRunReceipt(stdout io.Writer, asJSON bool, r cronRunReceipt) {
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(r)
		return
	}
	if r.Status == cronRunStatusSkippedDuplicate {
		fmt.Fprintf(stdout, "%s %s %s (status=%s exit_code=%d)\n",
			cronOutcomeDeduped, r.Job, r.Slot, r.Status, r.ExitCode)
		return
	}
	if r.Error != "" {
		fmt.Fprintf(stdout, "status=%s job=%s slot=%s exit_code=%d duration_ms=%d error=%q\n",
			r.Status, r.Job, r.Slot, r.ExitCode, r.DurationMS, r.Error)
	} else {
		fmt.Fprintf(stdout, "status=%s job=%s slot=%s exit_code=%d duration_ms=%d\n",
			r.Status, r.Job, r.Slot, r.ExitCode, r.DurationMS)
	}
}

// runCronRun implements `fak cron run`.
func runCronRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("cron run", flag.ContinueOnError)
	fs.SetOutput(stderr)

	job := fs.String("job", "", "job/loop id (required)")
	ledger := fs.String("ledger", "", "witness ledger path, JSONL (required)")
	interval := fs.Duration("interval", 0, "firing cadence; tick is quantized to this slot")
	timeout := fs.Duration("timeout", 0, "command execution timeout (required, must be positive)")
	at := fs.String("at", "", "wall-clock tick time (RFC3339); default now — injectable for tests")
	slot := fs.String("slot", "", "override computed slot key directly")
	asJSON := fs.Bool("json", false, "emit outcome receipt as JSON instead of human key-value")

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

	if strings.TrimSpace(*job) == "" {
		fmt.Fprintln(stderr, "fak cron run: --job is required")
		return 2
	}
	if strings.TrimSpace(*ledger) == "" {
		fmt.Fprintln(stderr, "fak cron run: --ledger is required")
		return 2
	}
	if *interval <= 0 {
		fmt.Fprintln(stderr, "fak cron run: --interval must be positive")
		return 2
	}
	if *timeout <= 0 {
		fmt.Fprintln(stderr, "fak cron run: --timeout must be positive")
		return 2
	}
	if len(cmdArgs) == 0 {
		fmt.Fprintln(stderr, "fak cron run: command is required after --")
		return 2
	}

	fireAt, slotKey, ok := resolveCronTimeAndSlot(stderr, "fak cron run", *at, *slot, *interval)
	if !ok {
		return 2
	}

	// Acquire dup-tick lock for CAS
	release, err := cronTickLock(*ledger+".tick.lock", cronTickWait, cronTickTTL)
	if err != nil {
		fmt.Fprintf(stderr, "fak cron run: %v\n", err)
		return 2
	}
	defer func() {
		if release != nil {
			_ = release()
		}
	}()

	fires, err := cronReadFires(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak cron run: read ledger: %v\n", err)
		return 2
	}

	for _, r := range fires {
		if r.Job == *job && r.Slot == slotKey && r.Outcome == cronOutcomeFired {
			// Slot already admitted (duplicate): record dedup fire event and suppress execution
			dupFireRec := cronFireRecord{
				Schema:   cronFireSchema,
				Job:      *job,
				Slot:     slotKey,
				Interval: int64(interval.Seconds()),
				Outcome:  cronOutcomeDeduped,
				FiredAt:  fireAt.Format(time.RFC3339),
			}
			_ = cronAppendFire(*ledger, dupFireRec)

			emitCronRunReceipt(stdout, *asJSON, cronRunReceipt{
				Schema:     cronRunReceiptSchema,
				Job:        *job,
				Slot:       slotKey,
				Status:     cronRunStatusSkippedDuplicate,
				Outcome:    cronRunOutcomeSkippedDuplicate,
				ExitCode:   cronExitDeduped,
				DurationMS: 0,
				Command:    strings.Join(cmdArgs, " "),
			})
			return cronExitDeduped
		}
	}

	// Slot admitted: append fire record
	fireRec := cronFireRecord{
		Schema:   cronFireSchema,
		Job:      *job,
		Slot:     slotKey,
		Interval: int64(interval.Seconds()),
		Outcome:  cronOutcomeFired,
		FiredAt:  fireAt.Format(time.RFC3339),
	}
	if err := cronAppendFire(*ledger, fireRec); err != nil {
		fmt.Fprintf(stderr, "fak cron run: append ledger: %v\n", err)
		return 2
	}

	// Release lock prior to executing the command so long-running commands don't hold the tick lock
	if release != nil {
		_ = release()
		release = nil
	}

	// Execute command with bounded timeout
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	cmdName := cmdArgs[0]
	cmdRest := cmdArgs[1:]
	c := exec.CommandContext(ctx, cmdName, cmdRest...)
	c.Stdout = stdout
	c.Stderr = stderr
	c.WaitDelay = 5 * time.Second
	c.Cancel = func() error {
		if c.Process != nil && c.Process.Pid > 0 {
			cronRunKillTree(c.Process.Pid)
		}
		return nil
	}

	startTime := time.Now()
	runErr := c.Run()
	finishedTime := time.Now()
	dur := finishedTime.Sub(startTime)
	durationMS := dur.Milliseconds()

	outcome := cronRunOutcomeSucceeded
	status := cronRunStatusRan
	exitCode := 0
	errMsg := ""

	if runErr != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			outcome = cronRunOutcomeTimeout
			status = cronRunStatusTimeout
			exitCode = cronRunExitTimeout
			errMsg = "execution timed out"
		} else {
			outcome = cronRunOutcomeFailed
			status = cronRunStatusFailed
			var exitErr *exec.ExitError
			if errors.As(runErr, &exitErr) {
				exitCode = exitErr.ExitCode()
			} else {
				exitCode = 1
			}
			errMsg = runErr.Error()
		}
	}

	runRec := cronRunRecord{
		Schema:     cronRunSchema,
		Job:        *job,
		Slot:       slotKey,
		Outcome:    outcome,
		Status:     status,
		ExitCode:   exitCode,
		DurationMS: durationMS,
		Command:    strings.Join(cmdArgs, " "),
		Error:      errMsg,
		StartedAt:  startTime.UTC().Format(time.RFC3339),
		FinishedAt: finishedTime.UTC().Format(time.RFC3339),
	}

	if err := cronAppendJSONL(*ledger, runRec); err != nil {
		fmt.Fprintf(stderr, "fak cron run: append run record: %v\n", err)
		return 2
	}

	emitCronRunReceipt(stdout, *asJSON, cronRunReceipt{
		Schema:     cronRunReceiptSchema,
		Job:        *job,
		Slot:       slotKey,
		Status:     status,
		Outcome:    outcome,
		ExitCode:   exitCode,
		DurationMS: durationMS,
		Command:    strings.Join(cmdArgs, " "),
		Error:      errMsg,
	})

	return exitCode
}

// cronReadRuns reads well-formed run records from the ledger.
func cronReadRuns(path string) ([]cronRunRecord, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return jsonlledger.Parse(string(b), func(r cronRunRecord) bool {
		return r.Schema == cronRunSchema && r.Job != "" && r.Slot != ""
	}), nil
}
