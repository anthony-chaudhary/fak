package sessionmine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

type RefreshOptions struct {
	Mine      Options
	IndexPath string
	Interval  time.Duration
	MaxRuns   int
}
type RefreshReceipt struct {
	Schema        string               `json:"schema"`
	Run           int                  `json:"run"`
	CompletedAt   string               `json:"completed_at"`
	Outcome       string               `json:"outcome"`
	ParsedFiles   int                  `json:"parsed_files"`
	ReusedFiles   int                  `json:"reused_files"`
	Sessions      int                  `json:"sessions"`
	Candidates    int                  `json:"candidates"`
	NewCandidates int                  `json:"new_candidates"`
	Error         string               `json:"error,omitempty"`
	Outcomes      RefreshOutcomeCounts `json:"outcomes"`
}
type RefreshOutcomeCounts struct {
	OK            int `json:"ok"`
	Error         int `json:"error"`
	ParsedFiles   int `json:"parsed_files"`
	ReusedFiles   int `json:"reused_files"`
	NewCandidates int `json:"new_candidates"`
}

type refreshLock struct {
	PID       int    `json:"pid"`
	StartedAt string `json:"started_at"`
}

func refreshReceiptPath(index string) string { return index + ".refresh.json" }
func refreshLockPath(index string) string    { return index + ".lock" }

// RefreshIndex updates the durable history index immediately, then on cadence.
// MaxRuns zero means continue until ctx is canceled.
func RefreshIndex(ctx context.Context, opts RefreshOptions, emit func(RefreshReceipt) error) error {
	if opts.IndexPath == "" {
		return errors.New("index path is required")
	}
	if opts.MaxRuns < 0 {
		return errors.New("max runs must be >= 0")
	}
	if opts.MaxRuns != 1 && opts.Interval <= 0 {
		return errors.New("interval must be positive for recurring refresh")
	}
	for run := 1; ; run++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		release, err := acquireRefreshLock(opts.IndexPath, time.Now().UTC())
		if err != nil {
			return err
		}
		result, mineErr := MineIndexed(opts.Mine, opts.IndexPath)
		release()
		previous, _ := loadRefreshReceipt(opts.IndexPath)
		counts := previous.Outcomes
		counts.ParsedFiles += result.ParsedFiles
		counts.ReusedFiles += result.ReusedFiles
		counts.NewCandidates += len(result.NewCandidates)
		receipt := RefreshReceipt{Schema: "fak-session-history-refresh/1", Run: run, CompletedAt: time.Now().UTC().Format(time.RFC3339), Outcome: "ok", ParsedFiles: result.ParsedFiles, ReusedFiles: result.ReusedFiles, Sessions: result.Report.Metrics.Sessions, Candidates: len(result.Report.Candidates), NewCandidates: len(result.NewCandidates), Outcomes: counts}
		if mineErr != nil {
			receipt.Outcomes.Error++
			receipt.Outcome = "error"
			receipt.Error = mineErr.Error()
		} else {
			receipt.Outcomes.OK++
		}
		if err := writeRefreshReceipt(opts.IndexPath, receipt); err != nil {
			return err
		}
		if emit != nil {
			if err := emit(receipt); err != nil {
				return err
			}
		}
		if mineErr != nil {
			return mineErr
		}
		if opts.MaxRuns > 0 && run >= opts.MaxRuns {
			return nil
		}
		timer := time.NewTimer(opts.Interval)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func acquireRefreshLock(index string, now time.Time) (func(), error) {
	path := refreshLockPath(index)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	for attempt := 0; attempt < 2; attempt++ {
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err == nil {
			l := refreshLock{PID: os.Getpid(), StartedAt: now.Format(time.RFC3339)}
			encErr := json.NewEncoder(f).Encode(l)
			closeErr := f.Close()
			if encErr != nil {
				os.Remove(path)
				return nil, encErr
			}
			if closeErr != nil {
				os.Remove(path)
				return nil, closeErr
			}
			return func() { _ = os.Remove(path) }, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, err
		}
		b, readErr := os.ReadFile(path)
		var l refreshLock
		if readErr == nil && json.Unmarshal(b, &l) == nil && l.PID > 0 && !processalive.Check(l.PID) {
			if os.Remove(path) == nil {
				continue
			}
		}
		owner := "unknown"
		if l.PID > 0 {
			owner = strconv.Itoa(l.PID)
		}
		return nil, fmt.Errorf("session history refresh already active (pid %s)", owner)
	}
	return nil, errors.New("session history refresh lock unavailable")
}
func loadRefreshReceipt(index string) (RefreshReceipt, error) {
	b, err := os.ReadFile(refreshReceiptPath(index))
	if err != nil {
		return RefreshReceipt{}, err
	}
	var r RefreshReceipt
	if err := json.Unmarshal(b, &r); err != nil {
		return RefreshReceipt{}, err
	}
	return r, nil
}
func writeRefreshReceipt(index string, r RefreshReceipt) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return writeBytesAtomic(refreshReceiptPath(index), b)
}
func writeBytesAtomic(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".session-history-receipt-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err = tmp.Write(b); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(name, path)
}
