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
)

const (
	hostResurrectionWaveLimit = 10
	hostResurrectionWindow    = 5 * time.Minute
)

type hostResurrectionReceipt struct {
	Schema     string `json:"schema"`
	Key        string `json:"key"`
	EventID    string `json:"event_id"`
	Session    string `json:"session"`
	LaunchedAt string `json:"launched_at"`
	PID        int    `json:"pid,omitempty"`
}

type hostSessionLauncher func(hostresurrect.Request) (int, error)

func hostResurrectionReceiptPath(logPath string) string { return logPath + ".relaunch.jsonl" }

func resurrectHostCrashSessions(logPath, regDir string, signals []hostfault.HostCrashSignal, launch hostSessionLauncher, now time.Time) ([]hostResurrectionReceipt, error) {
	if len(signals) == 0 {
		return nil, nil
	}
	path := hostResurrectionReceiptPath(logPath)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		return nil, fmt.Errorf("relaunch ledger lock: %w", err)
	}
	defer flock.Unlock(lock) //nolint:errcheck

	seen, times, err := readHostResurrectionReceipts(path)
	if err != nil {
		return nil, err
	}
	remaining := hostResurrectionWaveLimit - hostresurrect.RecentCount(times, now, hostResurrectionWindow)
	if remaining <= 0 {
		return nil, nil
	}
	rows := guardsessions.Load(regDir)
	var written []hostResurrectionReceipt
	for _, signal := range signals {
		for _, req := range hostresurrect.Plan(signal, rows, seen, remaining) {
			pid, err := launch(req)
			if err != nil {
				return written, fmt.Errorf("relaunch %s: %w", req.Session, err)
			}
			receipt := hostResurrectionReceipt{Schema: hostresurrect.Schema, Key: hostresurrect.Key(req.EventID, req.Session), EventID: req.EventID, Session: req.Session, LaunchedAt: now.UTC().Format(time.RFC3339Nano), PID: pid}
			if err := appendHostResurrectionReceipt(path, receipt); err != nil {
				return written, err
			}
			seen[receipt.Key] = true
			written = append(written, receipt)
			remaining--
			if remaining == 0 {
				return written, nil
			}
		}
	}
	return written, nil
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
