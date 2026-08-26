package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const watchdogAuditDefaultMaxBytes int64 = 4 << 20

type watchdogAuditReceipt struct {
	Schema     string          `json:"schema"`
	RecordedAt time.Time       `json:"recorded_at"`
	Verdict    string          `json:"verdict"`
	ExitStatus int             `json:"exit_status"`
	Audit      json.RawMessage `json:"audit"`
}

type watchdogAuditExec func(script string) ([]byte, int, error)

func runWatchdogAuditRunner(stdout, stderr io.Writer, argv []string) int {
	return runWatchdogAuditRunnerWith(stdout, stderr, argv, executeWatchdogAudit)
}

func runWatchdogAuditRunnerWith(stdout, stderr io.Writer, argv []string, execute watchdogAuditExec) int {
	fs := flag.NewFlagSet("watchdog-audit-run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	script := fs.String("script", defaultWatchdogAuditScript(), "tracked watchdog audit script")
	logPath := fs.String("log", defaultWatchdogAuditLog(), "bounded JSONL receipt ledger")
	maxBytes := fs.Int64("max-bytes", watchdogAuditDefaultMaxBytes, "maximum receipt ledger bytes")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *script == "" || *logPath == "" || *maxBytes < 1 {
		fmt.Fprintln(stderr, "watchdog-audit-run: --script, --log, and positive --max-bytes are required")
		return 2
	}
	if err := os.MkdirAll(filepath.Dir(*logPath), 0o755); err != nil {
		fmt.Fprintf(stderr, "watchdog-audit-run: create ledger directory: %v\n", err)
		return 1
	}
	lock, err := acquireStallWatchLock(*logPath)
	if err != nil {
		fmt.Fprintf(stderr, "watchdog-audit-run: %v\n", err)
		return 1
	}
	defer lock.Close()

	raw, status, runErr := execute(*script)
	if runErr != nil && status < 0 {
		fmt.Fprintf(stderr, "watchdog-audit-run: launch audit: %v\n", runErr)
		return 1
	}
	if status != 0 && status != 2 && status != 3 {
		fmt.Fprintf(stderr, "watchdog-audit-run: unexpected audit exit status %d: %v\n", status, runErr)
		return 1
	}
	payload := bytes.TrimSpace(raw)
	var audit map[string]any
	if len(payload) == 0 || json.Unmarshal(payload, &audit) != nil {
		fmt.Fprintln(stderr, "watchdog-audit-run: audit did not emit one valid JSON object")
		return 1
	}
	verdict, _ := audit["verdict"].(string)
	verdict = strings.ToUpper(verdict)
	want := map[int]string{0: "GREEN", 2: "AMBER", 3: "RED"}[status]
	if verdict != want {
		fmt.Fprintf(stderr, "watchdog-audit-run: verdict/status mismatch: %q/%d\n", verdict, status)
		return 1
	}
	receipt := watchdogAuditReceipt{Schema: "fak.watchdog-audit-receipt.v1", RecordedAt: time.Now().UTC(), Verdict: verdict, ExitStatus: status, Audit: append(json.RawMessage(nil), payload...)}
	line, err := json.Marshal(receipt)
	if err != nil {
		fmt.Fprintf(stderr, "watchdog-audit-run: encode receipt: %v\n", err)
		return 1
	}
	line = append(line, '\n')
	f, err := os.OpenFile(*logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		fmt.Fprintf(stderr, "watchdog-audit-run: open ledger: %v\n", err)
		return 1
	}
	if _, err = f.Write(line); err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		fmt.Fprintf(stderr, "watchdog-audit-run: append ledger: %v\n", err)
		return 1
	}
	if err := boundStallLog(*logPath, *maxBytes); err != nil {
		fmt.Fprintf(stderr, "watchdog-audit-run: bound ledger: %v\n", err)
		return 1
	}
	_, _ = stdout.Write(line)
	return status
}

func executeWatchdogAudit(script string) ([]byte, int, error) {
	name := "pwsh"
	if runtime.GOOS == "windows" {
		name = "powershell.exe"
	}
	cmd := exec.Command(name, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", script, "-Json")
	out, err := cmd.CombinedOutput()
	if err == nil {
		return out, 0, nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return out, exitErr.ExitCode(), err
	}
	return out, -1, err
}

func defaultWatchdogAuditScript() string {
	return filepath.Join(repoRoot(), "tools", "watchdog_watchdog_audit.ps1")
}
func defaultWatchdogAuditLog() string {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "fak-watchdog-audit", "audit.jsonl")
}
func runWatchdogAuditHealth(stdout, stderr io.Writer, argv []string, now func() time.Time) int {
	fs := flag.NewFlagSet("watchdog-audit-health", flag.ContinueOnError)
	fs.SetOutput(stderr)
	logPath := fs.String("log", defaultWatchdogAuditLog(), "bounded JSONL receipt ledger")
	maxAge := fs.Duration("max-age", 30*time.Minute, "maximum age of the newest receipt")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *logPath == "" || *maxAge <= 0 {
		fmt.Fprintln(stderr, "watchdog-audit-health: --log and positive --max-age are required")
		return 2
	}
	raw, err := os.ReadFile(*logPath)
	if err != nil {
		fmt.Fprintf(stderr, "watchdog-audit-health: read ledger: %v\n", err)
		return 1
	}
	lines := bytes.Split(bytes.TrimSpace(raw), []byte{'\n'})
	if len(lines) == 0 || len(lines[0]) == 0 {
		fmt.Fprintln(stderr, "watchdog-audit-health: ledger has no receipts")
		return 1
	}
	var receipt watchdogAuditReceipt
	if err := json.Unmarshal(lines[len(lines)-1], &receipt); err != nil || receipt.Schema != "fak.watchdog-audit-receipt.v1" {
		fmt.Fprintln(stderr, "watchdog-audit-health: newest receipt is malformed")
		return 1
	}
	age := now().Sub(receipt.RecordedAt)
	if receipt.RecordedAt.IsZero() || age < 0 || age > *maxAge {
		fmt.Fprintf(stderr, "watchdog-audit-health: newest receipt is stale (recorded_at=%s age=%s max_age=%s)\n", receipt.RecordedAt.Format(time.RFC3339), age.Round(time.Second), *maxAge)
		return 1
	}
	if _, err := stdout.Write(append(lines[len(lines)-1], '\n')); err != nil {
		fmt.Fprintf(stderr, "watchdog-audit-health: write output: %v\n", err)
		return 1
	}
	return receipt.ExitStatus
}
