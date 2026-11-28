// Adapted from ai-dynamo/aiperf
// (Apache-2.0, Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES):
// cross-process cache populate lock with SIGKILL tombstone bypass so a dead
// process or expired tombstone never wedges subsequent cache populates.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// PopulateLockRecord is the JSON tombstone written inside a populate lock file.
type PopulateLockRecord struct {
	PID       int       `json:"pid"`
	CreatedAt time.Time `json:"created_at"`
}

// PopulateLock manages a cross-process exclusive lock file with stale/dead-PID
// tombstone bypass.
type PopulateLock struct {
	path     string
	staleTTL time.Duration
	isAlive  func(pid int) bool
	now      func() time.Time
}

// NewPopulateLock constructs a PopulateLock at path with a staleTTL tombstone
// bypass threshold.
func NewPopulateLock(path string, staleTTL time.Duration) *PopulateLock {
	if staleTTL <= 0 {
		staleTTL = 5 * time.Minute
	}
	return &PopulateLock{
		path:     path,
		staleTTL: staleTTL,
		isAlive:  defaultPIDAlive,
		now:      func() time.Time { return time.Now().UTC() },
	}
}

// Acquire tries to create the lock file atomically; if a tombstone exists whose
// holder PID is dead or whose age exceeds staleTTL, it bypasses and reclaims
// the lock (bypassed=true).
func (l *PopulateLock) Acquire(pid int) (bypassed bool, err error) {
	if err := os.MkdirAll(filepath.Dir(l.path), 0o755); err != nil {
		return false, err
	}
	for attempt := 0; attempt < 2; attempt++ {
		if err := l.writeExclusive(pid); err == nil {
			return bypassed, nil
		} else if !os.IsExist(err) {
			return false, err
		}
		stale, staleErr := l.isStaleTombstone()
		if staleErr != nil {
			return false, staleErr
		}
		if !stale {
			return false, fmt.Errorf("loadgen: populate lock held at %s", l.path)
		}
		if rmErr := os.Remove(l.path); rmErr != nil && !os.IsNotExist(rmErr) {
			return false, rmErr
		}
		bypassed = true
	}
	return false, fmt.Errorf("loadgen: failed to acquire populate lock at %s", l.path)
}

// Release removes the lock file.
func (l *PopulateLock) Release() error {
	err := os.Remove(l.path)
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

func (l *PopulateLock) writeExclusive(pid int) error {
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	rec := PopulateLockRecord{PID: pid, CreatedAt: l.now()}
	data, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	return err
}

func (l *PopulateLock) isStaleTombstone() (bool, error) {
	raw, err := os.ReadFile(l.path)
	if os.IsNotExist(err) {
		return true, nil
	}
	if err != nil {
		return false, err
	}
	var rec PopulateLockRecord
	if err := json.Unmarshal(raw, &rec); err != nil || rec.PID <= 0 {
		// Corrupt or empty tombstone (e.g. SIGKILL mid-write) is safe to bypass.
		return true, nil
	}
	if !l.isAlive(rec.PID) {
		return true, nil
	}
	if l.now().Sub(rec.CreatedAt) > l.staleTTL {
		return true, nil
	}
	return false, nil
}

func defaultPIDAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	_, err := os.FindProcess(pid)
	return !errors.Is(err, os.ErrProcessDone)
}
