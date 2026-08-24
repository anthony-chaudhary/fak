package main

import (
	"bufio"
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

func cmdHostCrash(args []string) { os.Exit(runHostCrash(os.Stdout, os.Stderr, args)) }

func runHostCrash(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("host-crash", flag.ContinueOnError)
	fs.SetOutput(stderr)
	once := fs.Bool("once", false, "scan once and exit (default watches continuously)")
	interval := fs.Duration("interval", 15*time.Second, "watch polling interval")
	since := fs.Duration("since", 5*time.Minute, "event lookback window per poll")
	logPath := fs.String("log", defaultHostCrashLogPath(), "durable host-crash JSONL signal path")
	fixture := fs.String("fixture", "", "read Event-1000 fixture JSON instead of Windows Event Log")
	regDir := fs.String("reg-dir", "", "interactive-session registry directory (default: fleet registry)")
	resurrect := fs.Bool("resurrect", false, "relaunch live interactive rows for each new host-crash signal")
	dryRun := fs.Bool("dry-run", false, "plan and report resurrection without queueing or launching sessions")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak host-crash [--once] [--interval 15s] [--since 5m] [--log PATH] [--fixture PATH] [--resurrect] [--dry-run] [--reg-dir DIR]")
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
		var events []hostfault.ApplicationError1000
		var err error
		if *fixture != "" {
			events, err = readHostCrashFixture(*fixture)
		} else {
			events, err = gatherHostCrashEvents(*since)
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak host-crash: %v\n", err)
			return 1
		}
		emitted, err := appendNewHostCrashSignals(*logPath, events)
		if err != nil {
			fmt.Fprintf(stderr, "fak host-crash: write signals: %v\n", err)
			return 1
		}
		for _, signal := range emitted {
			b, _ := json.Marshal(signal)
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
	if runtime.GOOS != "windows" && *fixture == "" {
		fmt.Fprintln(stderr, "fak host-crash: Windows Application Event Log unavailable; use --fixture for replay")
		return 2
	}
	if *once || *fixture != "" {
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

func readHostCrashFixture(path string) ([]hostfault.ApplicationError1000, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var events []hostfault.ApplicationError1000
	if err := json.Unmarshal(b, &events); err != nil {
		return nil, fmt.Errorf("parse fixture: %w", err)
	}
	return events, nil
}

func appendNewHostCrashSignals(path string, events []hostfault.ApplicationError1000) ([]hostfault.HostCrashSignal, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	lockFile, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	defer lockFile.Close()
	deadline := time.Now().Add(5 * time.Second)
	for {
		err = flock.TryLock(lockFile)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) || time.Now().After(deadline) {
			return nil, fmt.Errorf("acquire signal ledger lock: %w", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	defer flock.Unlock(lockFile) //nolint:errcheck -- close also releases; best effort on return

	seen, err := readHostCrashSignalIDs(path)
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
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	enc := json.NewEncoder(f)
	for _, signal := range fresh {
		if err := enc.Encode(signal); err != nil {
			f.Close()
			return nil, err
		}
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return nil, err
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	return fresh, nil
}
func readHostCrashSignalIDs(path string) (map[string]bool, error) {
	seen := make(map[string]bool)
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return seen, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 4096), 1024*1024)
	for scan.Scan() {
		var signal hostfault.HostCrashSignal
		if err := json.Unmarshal(scan.Bytes(), &signal); err != nil {
			return nil, fmt.Errorf("parse existing signal log: %w", err)
		}
		if signal.Schema != hostfault.HostCrashSignalSchema || signal.EventID == "" {
			return nil, fmt.Errorf("invalid existing signal row")
		}
		seen[signal.EventID] = true
	}
	return seen, scan.Err()
}
