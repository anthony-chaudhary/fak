package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/hostfault"
	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
	"github.com/anthony-chaudhary/fak/internal/processalive"
)

type hostResurrectionReceipt struct {
	Schema     string `json:"schema"`
	Key        string `json:"key"`
	EventID    string `json:"event_id"`
	Session    string `json:"session"`
	LaunchedAt string `json:"launched_at"`
	PID        int    `json:"pid,omitempty"`
}

type hostResurrectionSelection struct {
	Schema     string                  `json:"schema"`
	EventID    string                  `json:"event_id"`
	CapturedAt string                  `json:"cohort_captured_at,omitempty"`
	Counts     hostresurrect.Selection `json:"counts"`
}

type hostSessionLauncher func(hostresurrect.Request) (int, error)

func hostResurrectionReceiptPath(logPath string) string { return logPath + ".relaunch.jsonl" }

func resurrectHostCrashSessions(logPath, regDir string, signals []hostfault.HostCrashSignal, launch hostSessionLauncher, now time.Time, persist bool) ([]hostResurrectionReceipt, []hostResurrectionSelection, error) {
	if len(signals) == 0 {
		return nil, nil, nil
	}
	path := hostResurrectionReceiptPath(logPath)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, nil, err
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, nil, err
	}
	defer lock.Close()
	deadline := time.Now().Add(5 * time.Second)
	for {
		err = flock.TryLock(lock)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) || time.Now().After(deadline) {
			return nil, nil, fmt.Errorf("relaunch ledger lock: %w", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	defer flock.Unlock(lock) //nolint:errcheck

	seen, times, err := readHostResurrectionReceipts(path)
	if err != nil {
		return nil, nil, err
	}
	remaining := hostresurrect.MaxLaunchesPerWindow - hostresurrect.RecentCount(times, now, hostresurrect.LaunchWindow)
	if remaining <= 0 {
		return nil, nil, nil
	}
	rows := guardsessions.Load(regDir)
	cohort, err := hostresurrect.LoadCohort(hostresurrect.CohortPath(regDir))
	if err != nil {
		return nil, nil, err
	}
	var written []hostResurrectionReceipt
	var selections []hostResurrectionSelection
	for _, signal := range signals {
		requests, counts := hostresurrect.Plan(signal, rows, cohort, seen, remaining)
		selections = append(selections, hostResurrectionSelection{Schema: hostresurrect.Schema, EventID: signal.EventID, CapturedAt: cohort.CapturedAt, Counts: counts})
		for _, req := range requests {
			// Reserve before spawn, matching FleetResumeWatchdog's ledger-first rule:
			// if this process crashes after Start, a repeated Event-Log poll cannot
			// double-launch the same event/session pair.
			receipt := hostResurrectionReceipt{Schema: hostresurrect.Schema, Key: hostresurrect.Key(req.EventID, req.Session), EventID: req.EventID, Session: req.Session, LaunchedAt: now.UTC().Format(time.RFC3339Nano)}
			if persist {
				if err := appendHostResurrectionReceipt(path, receipt); err != nil {
					return written, selections, err
				}
				seen[receipt.Key] = true
			}
			pid, err := launch(req)
			if err != nil {
				return written, selections, fmt.Errorf("relaunch %s: %w", req.Session, err)
			}
			receipt.PID = pid
			written = append(written, receipt)
			remaining--
			if remaining == 0 {
				return written, selections, nil
			}
		}
	}
	return written, selections, nil
}

func refreshHostResurrectionCohort(regDir string, now time.Time) (hostresurrect.Cohort, error) {
	cohort := hostresurrect.CaptureWithHost(guardsessions.Load(regDir), now, processalive.Check, processalive.StartTime, processalive.TerminalHostPID)
	return cohort, hostresurrect.StoreCohort(hostresurrect.CohortPath(regDir), cohort)
}

func appendHostResurrectionReceipt(path string, r hostResurrectionReceipt) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := json.NewEncoder(f).Encode(r); err != nil {
		return err
	}
	return f.Sync()
}

func readHostResurrectionReceipts(path string) (map[string]bool, []time.Time, error) {
	seen := map[string]bool{}
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return seen, nil, nil
	}
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	var times []time.Time
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var r hostResurrectionReceipt
		if json.Unmarshal(sc.Bytes(), &r) != nil || r.Schema != hostresurrect.Schema || r.Key == "" {
			return nil, nil, fmt.Errorf("invalid relaunch receipt")
		}
		seen[r.Key] = true
		if at, e := time.Parse(time.RFC3339Nano, r.LaunchedAt); e == nil {
			times = append(times, at)
		}
	}
	return seen, times, sc.Err()
}
