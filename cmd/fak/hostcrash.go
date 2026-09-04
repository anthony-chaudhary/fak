package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/hostfault"
	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
)

const hostCrashDefaultMaxBytes int64 = 16 << 20

func cmdHostCrash(args []string) { os.Exit(runHostCrash(os.Stdout, os.Stderr, args)) }

func runHostCrash(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("host-crash", flag.ContinueOnError)
	fs.SetOutput(stderr)
	once := fs.Bool("once", false, "scan once and exit (default watches continuously)")
	interval := fs.Duration("interval", 15*time.Second, "watch polling interval")
	since := fs.Duration("since", 5*time.Minute, "event lookback window per poll")
	logPath := fs.String("log", defaultHostCrashLogPath(), "durable host-crash JSONL signal path")
	maxBytes := fs.Int64("max-bytes", hostCrashDefaultMaxBytes, "maximum ledger bytes; preserves complete newest rows")
	fixture := fs.String("fixture", "", "read Event-1000 fixture JSON instead of Windows Event Log")
	systemFixture := fs.String("system-fixture", "", "read normalized System-event fixture JSON")
	regDir := fs.String("reg-dir", "", "interactive-session registry directory (default: fleet registry)")
	resurrect := fs.Bool("resurrect", false, "relaunch live interactive rows for each new host-crash signal")
	dryRun := fs.Bool("dry-run", false, "plan and report resurrection without queueing or launching sessions")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak host-crash [--once] [--interval 15s] [--since 5m] [--log PATH] [--max-bytes N] [--fixture PATH] [--system-fixture PATH] [--resurrect] [--dry-run] [--reg-dir DIR]")
		return 2
	}
	if *dryRun && !*resurrect {
		fmt.Fprintln(stderr, "fak host-crash: --dry-run requires --resurrect")
		return 2
	}
	if *interval <= 0 || *since <= 0 {
		fmt.Fprintln(stderr, "fak host-crash: --interval and --since must be positive")
		return 2
	}
	scan := func() int {
		fixtureMode := *fixture != "" || *systemFixture != ""
		var events []hostfault.ApplicationError1000
		var err error
		if *fixture != "" {
			events, err = readHostCrashFixture(*fixture)
		} else if !fixtureMode {
			events, err = gatherHostCrashEvents(*since)
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak host-crash: application collector: %v\n", err)
			return 1
		}
		emitted, err := appendNewHostCrashSignals(*logPath, events, *maxBytes)
		if err != nil {
			fmt.Fprintf(stderr, "fak host-crash: write signals: %v\n", err)
			return 1
		}

		var systemEvents []hostfault.WindowsSystemEvent
		if *systemFixture != "" {
			systemEvents, err = readHostSystemFixture(*systemFixture)
		} else if !fixtureMode {
			systemEvents, err = gatherHostSystemEvents(*since)
		}
		var systemEmitted []hostfault.SystemIncident
		if err != nil {
			// System evidence is observational. A denied or damaged System log must
			// not disable the pre-existing application-crash resurrection path.
			fmt.Fprintf(stderr, "fak host-crash: system collector degraded: %v\n", err)
		} else {
			systemEmitted, err = appendNewHostSystemIncidents(*logPath, systemEvents, *maxBytes)
			if err != nil {
				fmt.Fprintf(stderr, "fak host-crash: write system incidents degraded: %v\n", err)
				if *systemFixture != "" {
					return 1
				}
				systemEmitted = nil
			}
		}
		for _, signal := range emitted {
			b, _ := json.Marshal(signal)
			fmt.Fprintln(stdout, string(b))
		}
		for _, incident := range systemEmitted {
			b, _ := json.Marshal(incident)
			fmt.Fprintln(stdout, string(b))
		}
		if *resurrect {
			registry := resolveSweepRegDir(*regDir)
			if len(emitted) == 0 {
				if _, err := refreshHostResurrectionCohort(registry, time.Now()); err != nil {
					fmt.Fprintf(stderr, "fak host-crash: snapshot cohort: %v\n", err)
					return 1
				}
			} else {
				launcher := launchHostSessionPlatform
				if *dryRun {
					launcher = func(hostresurrect.Request) (int, error) { return 0, nil }
				}
				receipts, selections, err := resurrectHostCrashSessions(*logPath, registry, emitted, launcher, time.Now(), !*dryRun)
				if err != nil {
					fmt.Fprintf(stderr, "fak host-crash: resurrect: %v\n", err)
					return 1
				}
				for _, selection := range selections {
					b, _ := json.Marshal(selection)
					fmt.Fprintln(stdout, string(b))
				}
				for _, receipt := range receipts {
					b, _ := json.Marshal(receipt)
					fmt.Fprintln(stdout, string(b))
				}
			}
		}
		return 0
	}
	fixtureMode := *fixture != "" || *systemFixture != ""
	if runtime.GOOS != "windows" && !fixtureMode {
		fmt.Fprintln(stderr, "fak host-crash: Windows Event Logs unavailable; use --fixture and/or --system-fixture for replay")
		return 2
	}
	if *once || fixtureMode {
		return scan()
	}
	for {
		if rc := scan(); rc != 0 {
			return rc
		}
		time.Sleep(*interval)
	}
}
func defaultHostCrashLogPath() string {
	if dir := strings.TrimSpace(os.Getenv("FAK_STALL_DIR")); dir != "" {
		return filepath.Join(dir, "host-crashes.jsonl")
	}
	base, err := os.UserConfigDir()
	if err != nil || base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "fak", "host", "host-crashes.jsonl")
}

func readHostJSONFile[T any](path, label string) (T, error) {
	var target T
	b, err := os.ReadFile(path)
	if err != nil {
		return target, err
	}
	if err := json.Unmarshal(b, &target); err != nil {
		return target, fmt.Errorf("parse %s: %w", label, err)
	}
	return target, nil
}

func readHostCrashFixture(path string) ([]hostfault.ApplicationError1000, error) {
	return readHostJSONFile[[]hostfault.ApplicationError1000](path, "fixture")
}

func readHostSystemFixture(path string) ([]hostfault.WindowsSystemEvent, error) {
	return readHostJSONFile[[]hostfault.WindowsSystemEvent](path, "system fixture")
}

func appendNewHostSystemIncidents(path string, events []hostfault.WindowsSystemEvent, maxBytesArg ...int64) ([]hostfault.SystemIncident, error) {
	maxBytes, err := hostCrashMaxBytes(maxBytesArg)
	if err != nil {
		return nil, err
	}
	var incidents []hostfault.SystemIncident
	for _, event := range events {
		if incident, ok := hostfault.ClassifyWindowsSystemEvent(event); ok {
			incidents = append(incidents, incident)
		}
	}
	if len(incidents) == 0 {
		return nil, nil
	}
	lockFile, err := lockHostCrashLedger(path)
	if err != nil {
		return nil, err
	}
	defer unlockHostCrashLedger(lockFile)

	seen, err := readHostCrashEventIDs(path)
	if err != nil {
		return nil, err
	}
	var fresh []hostfault.SystemIncident
	for _, incident := range incidents {
		if !seen[incident.EventID] {
			seen[incident.EventID] = true
			fresh = append(fresh, incident)
		}
	}
	if len(fresh) == 0 {
		return nil, nil
	}
	if err := appendHostCrashRows(path, fresh, maxBytes); err != nil {
		return nil, err
	}
	if err := boundHostCrashLedger(path, maxBytes); err != nil {
		return nil, err
	}
	return fresh, nil
}

func appendNewHostCrashSignals(path string, events []hostfault.ApplicationError1000, maxBytesArg ...int64) ([]hostfault.HostCrashSignal, error) {
	maxBytes, err := hostCrashMaxBytes(maxBytesArg)
	if err != nil {
		return nil, err
	}
	lockFile, err := lockHostCrashLedger(path)
	if err != nil {
		return nil, err
	}
	defer unlockHostCrashLedger(lockFile)

	seen, err := readHostCrashEventIDs(path)
	if err != nil {
		return nil, err
	}
	var fresh []hostfault.HostCrashSignal
	for _, event := range events {
		signal, relevant := hostfault.ClassifyApplicationError(event)
		if !relevant || seen[signal.EventID] {
			continue
		}
		seen[signal.EventID] = true
		fresh = append(fresh, signal)
	}
	if len(fresh) == 0 {
		return nil, nil
	}
	if err := appendHostCrashRows(path, fresh, maxBytes); err != nil {
		return nil, err
	}
	if err := boundHostCrashLedger(path, maxBytes); err != nil {
		return nil, err
	}
	return fresh, nil
}

func readHostCrashEventIDs(path string) (map[string]bool, error) {
	seen := make(map[string]bool)
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return seen, nil
	}
	if err != nil {
		return nil, err
	}
	defer file.Close()
	scan := bufio.NewScanner(file)
	scan.Buffer(make([]byte, 4096), 1<<20)
	for scan.Scan() {
		var row struct {
			Schema  string `json:"schema"`
			EventID string `json:"event_id"`
		}
		if err := json.Unmarshal(scan.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("parse existing signal log: %w", err)
		}
		if row.EventID == "" || (row.Schema != hostfault.HostCrashSignalSchema && row.Schema != hostfault.SystemIncidentSchema) {
			return nil, fmt.Errorf("invalid existing signal log row")
		}
		seen[row.EventID] = true
	}
	if err := scan.Err(); err != nil {
		return nil, fmt.Errorf("read existing host-crash ledger: %w", err)
	}
	return seen, nil
}

func lockHostCrashLedger(path string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	lockFile, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		if err = flock.TryLock(lockFile); err == nil {
			return lockFile, nil
		}
		if !errors.Is(err, flock.ErrLockBusy) || time.Now().After(deadline) {
			_ = lockFile.Close()
			return nil, fmt.Errorf("acquire host-crash ledger lock: %w", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func unlockHostCrashLedger(lockFile *os.File) {
	_ = flock.Unlock(lockFile)
	_ = lockFile.Close()
}

func appendHostCrashRows[T any](path string, rows []T, maxBytes int64) error {
	encoded := make([][]byte, 0, len(rows))
	for _, row := range rows {
		line, err := json.Marshal(row)
		if err != nil {
			return err
		}
		line = append(line, '\n')
		if int64(len(line)) > maxBytes {
			return fmt.Errorf("host-crash ledger row is %d bytes, exceeds %d-byte bound", len(line), maxBytes)
		}
		encoded = append(encoded, line)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	for _, line := range encoded {
		if _, err := file.Write(line); err != nil {
			_ = file.Close()
			return err
		}
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return err
	}
	return file.Close()
}

func hostCrashMaxBytes(args []int64) (int64, error) {
	if len(args) > 1 {
		return 0, fmt.Errorf("host-crash ledger bound supplied more than once")
	}
	maxBytes := hostCrashDefaultMaxBytes
	if len(args) == 1 {
		maxBytes = args[0]
	}
	if maxBytes <= 0 {
		return 0, fmt.Errorf("host-crash ledger bound must be positive")
	}
	return maxBytes, nil
}

func boundHostCrashLedger(path string, maxBytes int64) error {
	info, err := os.Stat(path)
	if err != nil || info.Size() <= maxBytes {
		return err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	start := len(data) - int(maxBytes)
	if start < 0 {
		start = 0
	}
	if start > 0 {
		newline := bytes.IndexByte(data[start:], '\n')
		if newline < 0 {
			return fmt.Errorf("newest host-crash ledger row exceeds %d-byte bound", maxBytes)
		}
		start += newline + 1
	}
	kept := data[start:]
	if len(kept) > 0 && kept[len(kept)-1] != '\n' {
		return fmt.Errorf("host-crash ledger ends with an incomplete row")
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".host-crash-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err == nil {
		_, err = tmp.Write(kept)
	}
	if err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Rename(tmpPath, path)
}
