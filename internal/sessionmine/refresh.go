package sessionmine

import (
	"context"
	"errors"
	"time"
)

type RefreshOptions struct {
	Mine      Options
	IndexPath string
	Interval  time.Duration
	MaxRuns   int
}

type RefreshReceipt struct {
	Schema        string `json:"schema"`
	Run           int    `json:"run"`
	ParsedFiles   int    `json:"parsed_files"`
	ReusedFiles   int    `json:"reused_files"`
	Sessions      int    `json:"sessions"`
	Candidates    int    `json:"candidates"`
	NewCandidates int    `json:"new_candidates"`
}

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
	run := 0
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		result, err := MineIndexed(opts.Mine, opts.IndexPath)
		if err != nil {
			return err
		}
		run++
		receipt := RefreshReceipt{Schema: "fak-session-history-refresh/1", Run: run, ParsedFiles: result.ParsedFiles, ReusedFiles: result.ReusedFiles, Sessions: result.Report.Metrics.Sessions, Candidates: len(result.Report.Candidates), NewCandidates: len(result.NewCandidates)}
		if emit != nil {
			if err := emit(receipt); err != nil {
				return err
			}
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
