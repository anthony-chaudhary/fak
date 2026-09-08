// cron_scheduler.go implements in-kernel scheduled automations with
// at-most-once fire semantics, catchup and grace windows, duplicate tick
// rejection, and hard-interrupt ceiling enforcement (#2927).
//
// Hermes Chronos evidence (epic #2871):
//   - at-most-once delivery via store compare-and-set + .tick.lock
//   - catchup window: half the period, clamped to [120s, 2h] (120s <= period/2 <= 7200s)
//   - missed one-shot grace window: 120s
//   - hard-interrupt ceiling: 3-minute hard interrupt on cron sessions
//   - duplicate tick protection: concurrent and duplicate ticks are rejected
//   - hash-chained journal rows for tamper-evident execution witnesses
package main

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

const (
	// CronCatchupMin is the minimum clamped catchup window (120s).
	CronCatchupMin = 120 * time.Second

	// CronCatchupMax is the maximum clamped catchup window (2h = 7200s).
	CronCatchupMax = 2 * time.Hour

	// CronOneShotGrace is the grace window for missed one-shot executions (120s).
	CronOneShotGrace = 120 * time.Second

	// CronHardInterruptCeiling is the maximum execution duration for a cron session (3m).
	CronHardInterruptCeiling = 3 * time.Minute

	cronOutcomeMissedGrace   = "missed_grace"
	cronOutcomeMissedCatchup = "missed_catchup"
)

// ErrCronHardInterrupt is returned when a cron session exceeds the hard-interrupt ceiling.
var ErrCronHardInterrupt = errors.New("cron hard interrupt: session exceeded ceiling")

// CronCatchupWindow calculates the catchup window for a job with the given period.
// It computes half the period, clamped to [120s, 2h] (120s <= period/2 <= 7200s).
// For one-shot jobs (period <= 0), it returns CronOneShotGrace (120s).
func CronCatchupWindow(period time.Duration) time.Duration {
	if period <= 0 {
		return CronOneShotGrace
	}
	half := period / 2
	if half < CronCatchupMin {
		return CronCatchupMin
	}
	if half > CronCatchupMax {
		return CronCatchupMax
	}
	return half
}

// CronCheckEligibility evaluates whether a tick at tickTime is eligible to fire
// for an execution scheduled at scheduledTime with the given period.
//
// For one-shot jobs (period <= 0):
//   - If tickTime is before scheduledTime, it is not yet due.
//   - If tickTime is after scheduledTime + CronOneShotGrace (120s), the grace window has expired.
//   - Otherwise, eligible = true.
//
// For recurring jobs (period > 0):
//   - If tickTime is before scheduledTime, it is not yet due.
//   - If tickTime is after scheduledTime + CronCatchupWindow(period), the catchup window has expired.
//   - Otherwise, eligible = true.
func CronCheckEligibility(scheduledTime, tickTime time.Time, period time.Duration) (bool, string) {
	if tickTime.Before(scheduledTime) {
		return false, "scheduled time is in the future (not due yet)"
	}
	if period <= 0 {
		cutoff := scheduledTime.Add(CronOneShotGrace)
		if tickTime.After(cutoff) {
			return false, fmt.Sprintf("missed one-shot grace window: tick %s exceeded scheduled %s by %s (grace %s)",
				tickTime.Format(time.RFC3339), scheduledTime.Format(time.RFC3339), tickTime.Sub(scheduledTime), CronOneShotGrace)
		}
		return true, ""
	}
	window := CronCatchupWindow(period)
	cutoff := scheduledTime.Add(window)
	if tickTime.After(cutoff) {
		return false, fmt.Sprintf("missed catchup window: tick %s exceeded scheduled %s by %s (catchup %s)",
			tickTime.Format(time.RFC3339), scheduledTime.Format(time.RFC3339), tickTime.Sub(scheduledTime), window)
	}
	return true, ""
}

// CronExecuteSession executes a cron session function fn with a bounded timeout
// that cannot exceed CronHardInterruptCeiling (3 minutes).
// If timeout <= 0 or timeout > CronHardInterruptCeiling, it is clamped to CronHardInterruptCeiling.
// If fn exceeds the effective timeout, the context is canceled and ErrCronHardInterrupt is returned.
func CronExecuteSession(ctx context.Context, timeout time.Duration, fn func(ctx context.Context) error) error {
	effectiveTimeout := timeout
	if effectiveTimeout <= 0 || effectiveTimeout > CronHardInterruptCeiling {
		effectiveTimeout = CronHardInterruptCeiling
	}
	sessionCtx, cancel := context.WithTimeout(ctx, effectiveTimeout)
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		errCh <- fn(sessionCtx)
	}()

	select {
	case err := <-errCh:
		return err
	case <-sessionCtx.Done():
		if errors.Is(sessionCtx.Err(), context.DeadlineExceeded) {
			return fmt.Errorf("%w: exceeded %s ceiling", ErrCronHardInterrupt, effectiveTimeout)
		}
		return sessionCtx.Err()
	}
}

// CronFireDecision describes the outcome of an AttemptFire call.
type CronFireDecision struct {
	Admitted bool           `json:"admitted"`
	Outcome  string         `json:"outcome"`
	Slot     string         `json:"slot"`
	Record   cronFireRecord `json:"record"`
	Reason   string         `json:"reason,omitempty"`
}

// CronStore provides durable, at-most-once cron scheduling with compare-and-set,
// duplicate-tick rejection, and hash-chained journal records.
type CronStore struct {
	LedgerPath string
	mu         sync.Mutex
}

// NewCronStore returns a CronStore backed by ledgerPath.
func NewCronStore(ledgerPath string) *CronStore {
	return &CronStore{LedgerPath: ledgerPath}
}

// AttemptFire evaluates eligibility (grace/catchup window) and executes compare-and-set
// under the dup-tick lock. If eligible and not already fired, it records a hash-chained
// "fired" event (Admitted = true). If already fired, it records "deduped" (Admitted = false).
// If past the grace or catchup window, it refuses fire (Admitted = false).
func (s *CronStore) AttemptFire(job string, scheduledAt, tickAt time.Time, interval time.Duration) (CronFireDecision, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	slotKey := cronFireSlot(scheduledAt, interval)
	if interval <= 0 {
		slotKey = scheduledAt.UTC().Format(time.RFC3339)
	}

	// 1. Acquire dup-tick lock across processes
	release, err := cronTickLock(s.LedgerPath+".tick.lock", cronTickWait, cronTickTTL)
	if err != nil {
		return CronFireDecision{}, fmt.Errorf("cron tick lock: %w", err)
	}
	defer func() { _ = release() }()

	// 2. Compare-and-Set: scan ledger for existing fired slot (duplicate tick protection)
	fires, err := cronReadFires(s.LedgerPath)
	if err != nil {
		return CronFireDecision{}, fmt.Errorf("read ledger: %w", err)
	}

	for _, r := range fires {
		if r.Job == job && r.Slot == slotKey && r.Outcome == cronOutcomeFired {
			// Duplicate tick: slot already fired
			dupRec := cronFireRecord{
				Schema:   cronFireSchema,
				Job:      job,
				Slot:     slotKey,
				Interval: int64(interval.Seconds()),
				Outcome:  cronOutcomeDeduped,
				FiredAt:  tickAt.UTC().Format(time.RFC3339),
			}
			_ = cronAppendFire(s.LedgerPath, dupRec)
			return CronFireDecision{
				Admitted: false,
				Outcome:  cronOutcomeDeduped,
				Slot:     slotKey,
				Record:   dupRec,
				Reason:   "duplicate tick: slot already fired",
			}, nil
		}
	}

	// 3. Check window eligibility (missed one-shot grace or catchup window)
	eligible, reason := CronCheckEligibility(scheduledAt, tickAt, interval)
	if !eligible {
		outcome := cronOutcomeMissedGrace
		if interval > 0 {
			outcome = cronOutcomeMissedCatchup
		}
		rec := cronFireRecord{
			Schema:   cronFireSchema,
			Job:      job,
			Slot:     slotKey,
			Interval: int64(interval.Seconds()),
			Outcome:  outcome,
			FiredAt:  tickAt.UTC().Format(time.RFC3339),
		}
		_ = cronAppendFire(s.LedgerPath, rec)
		return CronFireDecision{
			Admitted: false,
			Outcome:  outcome,
			Slot:     slotKey,
			Record:   rec,
			Reason:   reason,
		}, nil
	}

	// 4. Slot admitted: append hash-chained fire record
	fireRec := cronFireRecord{
		Schema:   cronFireSchema,
		Job:      job,
		Slot:     slotKey,
		Interval: int64(interval.Seconds()),
		Outcome:  cronOutcomeFired,
		FiredAt:  tickAt.UTC().Format(time.RFC3339),
	}
	if err := cronAppendFire(s.LedgerPath, fireRec); err != nil {
		return CronFireDecision{}, fmt.Errorf("append fire: %w", err)
	}

	// Re-read or get the stamped record
	updatedFires, _ := cronReadFires(s.LedgerPath)
	if len(updatedFires) > 0 {
		fireRec = updatedFires[len(updatedFires)-1]
	}

	return CronFireDecision{
		Admitted: true,
		Outcome:  cronOutcomeFired,
		Slot:     slotKey,
		Record:   fireRec,
	}, nil
}

// VerifyChain checks that all fire records in the store's ledger form an unbroken,
// tamper-evident SHA-256 hash chain.
func (s *CronStore) VerifyChain() (count int, valid bool, err error) {
	return cronVerifyFireChain(s.LedgerPath)
}

// CronScheduler embeds CronStore and provides the in-kernel scheduled automation contract.
type CronScheduler struct {
	store *CronStore
}

// NewCronScheduler creates a CronScheduler with an in-kernel CronStore.
func NewCronScheduler(ledgerPath string) *CronScheduler {
	return &CronScheduler{store: NewCronStore(ledgerPath)}
}

func (cs *CronScheduler) Store() *CronStore {
	return cs.store
}

func (cs *CronScheduler) AttemptFire(job string, scheduledAt, tickAt time.Time, interval time.Duration) (CronFireDecision, error) {
	return cs.store.AttemptFire(job, scheduledAt, tickAt, interval)
}

func (cs *CronScheduler) CalculateCatchupWindow(period time.Duration) time.Duration {
	return CronCatchupWindow(period)
}

func (cs *CronScheduler) CheckEligibility(scheduledAt, tickAt time.Time, period time.Duration) (bool, string) {
	return CronCheckEligibility(scheduledAt, tickAt, period)
}

func (cs *CronScheduler) RunSession(ctx context.Context, timeout time.Duration, fn func(ctx context.Context) error) error {
	return CronExecuteSession(ctx, timeout, fn)
}

func (cs *CronScheduler) VerifyChain() (int, bool, error) {
	return cs.store.VerifyChain()
}
