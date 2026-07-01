package main

// timeout_ledger.go is the thin I/O shell for `fak dispatch timeout-ledger`: it reads
// timed-out-attempt facts (which lifecycle stage each attempt was last observed in before its
// kill), classifies each through internal/timeoutphase's pure Classify, and persists one JSONL
// row per attempt to a durable local ledger -- so a phase breakdown of WHERE timeouts happen
// (before startup, mid-edit, mid-test, mid-commit, mid-push) is auditable later and can feed a
// dispatch-status rollup, instead of every timeout looking like the same opaque event (#1793).
// It never spawns a worker or mutates GitHub; the only side effect is appending to the local
// ledger file, matching skip-ledger's exact persistence shape.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/timeoutphase"
)

const (
	timeoutLedgerRunsDir = ".dispatch-runs"
	timeoutLedgerLogName = "timeout-ledger.jsonl"
)

func runDispatchTimeoutLedger(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch timeout-ledger", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "read timed-out attempt facts from this JSON file (default: stdin)")
	workspace := fs.String("workspace", ".", "workspace root the ledger is persisted under")
	nowUnix := fs.Int64("now", 0, "the clock as unix seconds for the row timestamp when an attempt does not carry its own (0 = current time)")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}

	raw, code := readDispatchInput(stderr, *in)
	if code != 0 {
		return code
	}
	attempts, err := parseTimeoutAttempts(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch timeout-ledger: parse attempts: %v\n", err)
		return 1
	}

	now := *nowUnix
	if now == 0 {
		now = time.Now().Unix()
	}
	for i := range attempts {
		if attempts[i].TimestampUnix == 0 {
			attempts[i].TimestampUnix = now
		}
	}

	rep := timeoutphase.Record(attempts)

	runsDir := filepath.Join(*workspace, timeoutLedgerRunsDir)
	if err := timeoutLedgerAppend(runsDir, rep); err != nil {
		fmt.Fprintf(stderr, "fak dispatch timeout-ledger: persist ledger: %v\n", err)
		return 1
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak dispatch timeout-ledger")
	}
	renderTimeoutLedger(stdout, rep, filepath.Join(runsDir, timeoutLedgerLogName))
	return 0
}

// timeoutAttemptJSON is the wire shape one --in element takes; it mirrors
// timeoutphase.Attempt field-for-field so the CLI boundary stays a straight decode.
type timeoutAttemptJSON struct {
	ID            string             `json:"id"`
	Started       bool               `json:"started"`
	LastStage     timeoutphase.Stage `json:"last_stage,omitempty"`
	FailureClass  string             `json:"failure_class,omitempty"`
	TimestampUnix int64              `json:"timestamp_unix,omitempty"`
}

// parseTimeoutAttempts accepts either a bare JSON array of attempts or an object with an
// "attempts" field, matching the accept-either-shape convention the other dispatch --in
// commands use.
func parseTimeoutAttempts(raw []byte) ([]timeoutphase.Attempt, error) {
	var arr []timeoutAttemptJSON
	if err := json.Unmarshal(raw, &arr); err == nil {
		return toAttempts(arr), nil
	}
	var obj struct {
		Attempts []timeoutAttemptJSON `json:"attempts"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("parse timeout-ledger input json: %w", err)
	}
	return toAttempts(obj.Attempts), nil
}

func toAttempts(in []timeoutAttemptJSON) []timeoutphase.Attempt {
	out := make([]timeoutphase.Attempt, 0, len(in))
	for _, a := range in {
		out = append(out, timeoutphase.Attempt{
			ID:            a.ID,
			Started:       a.Started,
			LastStage:     a.LastStage,
			FailureClass:  a.FailureClass,
			TimestampUnix: a.TimestampUnix,
		})
	}
	return out
}

// timeoutLedgerAppend persists one JSON line per row to the durable ledger file, creating the
// runs dir if needed. Append-only, matching skip-ledger's persistence shape exactly.
func timeoutLedgerAppend(runsDir string, rep timeoutphase.Report) error {
	if len(rep.Rows) == 0 {
		return nil
	}
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(runsDir, timeoutLedgerLogName), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, row := range rep.Rows {
		if err := enc.Encode(row); err != nil {
			return err
		}
	}
	return nil
}

// renderTimeoutLedger prints the phase breakdown, then each row as an aligned table, then the
// persisted ledger path.
func renderTimeoutLedger(w io.Writer, rep timeoutphase.Report, ledgerPath string) {
	fmt.Fprintf(w, "timeout ledger -- %d attempt(s) classified\n\n", len(rep.Rows))
	for _, phase := range []timeoutphase.Phase{
		timeoutphase.PhaseBeforeStartup,
		timeoutphase.PhaseDuringEdit,
		timeoutphase.PhaseDuringTests,
		timeoutphase.PhaseDuringCommit,
		timeoutphase.PhaseDuringPush,
		timeoutphase.PhaseUnknown,
	} {
		if n := rep.PhaseCount[phase]; n > 0 {
			fmt.Fprintf(w, "  %-15s %d\n", phase, n)
		}
	}
	fmt.Fprintf(w, "\n%-10s %-16s %-10s %-16s %s\n", "id", "phase", "stage", "failure_class", "timestamp_unix")
	for _, row := range rep.Rows {
		stage := string(row.LastStage)
		if stage == "" {
			stage = "-"
		}
		failureClass := row.FailureClass
		if failureClass == "" {
			failureClass = "-"
		}
		fmt.Fprintf(w, "%-10s %-16s %-10s %-16s %d\n", row.ID, row.Phase, stage, failureClass, row.TimestampUnix)
	}
	fmt.Fprintf(w, "\npersisted %d row(s) to %s\n", len(rep.Rows), ledgerPath)
}
