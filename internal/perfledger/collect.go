// Inspired by ai-dynamo/aiperf
// before falling back to a secondary source.
package perfledger

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Default retry settings for terminal child summary-race handling.
const (
	DefaultSummaryRaceMaxAttempts   = 5
	DefaultSummaryRaceRetryInterval = 50 * time.Millisecond
)

// ErrSummaryUnavailable indicates neither the primary bounded re-read loop nor
// the secondary fallback produced a summary.
var ErrSummaryUnavailable = errors.New("perfledger: terminal child summary unavailable")

// SummarySource reports which source satisfied CollectTerminalSummary.
type SummarySource string

const (
	SummarySourcePrimary  SummarySource = "primary"
	SummarySourceFallback SummarySource = "secondary_fallback"
)

// SummaryCollectConfig configures the bounded re-read retry budget.
type SummaryCollectConfig struct {
	MaxAttempts   int
	RetryInterval time.Duration
	Sleep         func(ctx context.Context, d time.Duration) error
}

// SummaryCollectConfigFromLookup parses SUMMARY_RACE_MAX_ATTEMPTS and
// SUMMARY_RACE_RETRY_INTERVAL using caller-provided env lookup, applying
// defaults when unset.
func SummaryCollectConfigFromLookup(lookup func(string) string) (SummaryCollectConfig, error) {
	cfg := SummaryCollectConfig{
		MaxAttempts:   DefaultSummaryRaceMaxAttempts,
		RetryInterval: DefaultSummaryRaceRetryInterval,
	}
	if lookup == nil {
		return cfg, nil
	}
	if raw := strings.TrimSpace(lookup("SUMMARY_RACE_MAX_ATTEMPTS")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 1 {
			return SummaryCollectConfig{}, fmt.Errorf("perfledger: SUMMARY_RACE_MAX_ATTEMPTS must be >= 1, got %q", raw)
		}
		cfg.MaxAttempts = n
	}
	if raw := strings.TrimSpace(lookup("SUMMARY_RACE_RETRY_INTERVAL")); raw != "" {
		d, err := time.ParseDuration(raw)
		if err != nil || d < 0 {
			return SummaryCollectConfig{}, fmt.Errorf("perfledger: SUMMARY_RACE_RETRY_INTERVAL must be a non-negative duration, got %q", raw)
		}
		cfg.RetryInterval = d
	}
	return cfg, nil
}

// SummarySnapshot is one read from either the primary status or secondary fallback.
type SummarySnapshot struct {
	Terminal   bool
	HasSummary bool
	Summary    map[string]any
}

// SummaryCollectResult records the resolved summary and retry telemetry.
type SummaryCollectResult struct {
	Summary      map[string]any
	Source       SummarySource
	Attempts     int
	UsedFallback bool
	LastErr      error
}

// CollectTerminalSummary runs a bounded re-read loop against primary to tolerate
// the race where a child reaches terminal status before its summary is visible,
// falling back to secondary after exhausting MaxAttempts.
func CollectTerminalSummary(
	ctx context.Context,
	cfg SummaryCollectConfig,
	primary func(context.Context) (SummarySnapshot, error),
	fallback func(context.Context) (SummarySnapshot, error),
) (SummaryCollectResult, error) {
	if primary == nil {
		return SummaryCollectResult{}, errors.New("perfledger: CollectTerminalSummary: primary reader is nil")
	}
	if cfg.MaxAttempts < 0 {
		return SummaryCollectResult{}, fmt.Errorf("perfledger: CollectTerminalSummary: MaxAttempts must be non-negative, got %d", cfg.MaxAttempts)
	}
	if cfg.RetryInterval < 0 {
		return SummaryCollectResult{}, fmt.Errorf("perfledger: CollectTerminalSummary: RetryInterval must be non-negative, got %v", cfg.RetryInterval)
	}

	maxAttempts := cfg.MaxAttempts
	if maxAttempts == 0 {
		maxAttempts = DefaultSummaryRaceMaxAttempts
	}
	interval := cfg.RetryInterval
	if interval == 0 && cfg.Sleep == nil {
		interval = DefaultSummaryRaceRetryInterval
	}
	sleepFn := cfg.Sleep
	if sleepFn == nil {
		sleepFn = defaultContextSleep
	}

	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return SummaryCollectResult{}, err
		}
		snap, err := primary(ctx)
		if err == nil && snap.HasSummary {
			return SummaryCollectResult{
				Summary:  snap.Summary,
				Source:   SummarySourcePrimary,
				Attempts: attempt,
			}, nil
		}
		if err != nil {
			lastErr = err
		}
		if attempt < maxAttempts && interval > 0 {
			if err := sleepFn(ctx, interval); err != nil {
				return SummaryCollectResult{}, err
			}
		}
	}

	if fallback != nil {
		if err := ctx.Err(); err != nil {
			return SummaryCollectResult{}, err
		}
		fbSnap, fbErr := fallback(ctx)
		if fbErr == nil && fbSnap.HasSummary {
			return SummaryCollectResult{
				Summary:      fbSnap.Summary,
				Source:       SummarySourceFallback,
				Attempts:     maxAttempts,
				UsedFallback: true,
				LastErr:      lastErr,
			}, nil
		}
		if fbErr != nil {
			return SummaryCollectResult{}, fmt.Errorf("%w: primary=%v; fallback=%v", ErrSummaryUnavailable, lastErr, fbErr)
		}
	}

	if lastErr != nil {
		return SummaryCollectResult{}, fmt.Errorf("%w: %v", ErrSummaryUnavailable, lastErr)
	}
	return SummaryCollectResult{}, ErrSummaryUnavailable
}

func defaultContextSleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
