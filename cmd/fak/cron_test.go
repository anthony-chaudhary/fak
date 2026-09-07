package main

import (
	"bytes"
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// emitCron runs `fak cron emit ...` and returns stdout/exit, failing on a non-zero
// exit so the per-target tests only assert on the rendered unit.
func emitCron(t *testing.T, argv ...string) string {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runCron(&stdout, &stderr, append([]string{"emit"}, argv...))
	if code != 0 {
		t.Fatalf("cron emit %v: exit=%d stderr=%s", argv, code, stderr.String())
	}
	return stdout.String()
}

// assertWellFormedXML decodes the whole document so a malformed plist fails the
// "round-trips on the host platform" acceptance rung rather than printing garbage.
func assertWellFormedXML(t *testing.T, doc string) {
	t.Helper()
	dec := xml.NewDecoder(strings.NewReader(doc))
	for {
		_, err := dec.Token()
		if err == io.EOF {
			return
		}
		if err != nil {
			t.Fatalf("emitted plist is not well-formed XML: %v\n%s", err, doc)
		}
	}
}

func mustContain(t *testing.T, out string, needles ...string) {
	t.Helper()
	for _, n := range needles {
		if !strings.Contains(out, n) {
			t.Fatalf("emitted unit missing %q\n---\n%s", n, out)
		}
	}
}

func TestCronEmitLaunchd(t *testing.T) {
	out := emitCron(t, "--target", "launchd", "--interval", "5m", "issue-dispatch/default")
	assertWellFormedXML(t, out)
	// The action is the fak loop run vector, fired by launchd's StartInterval.
	mustContain(t, out,
		"<string>fak</string>",
		"<string>loop</string>",
		"<string>run</string>",
		"<string>--loop</string>",
		"<string>issue-dispatch/default</string>",
		"<string>--source</string>",
		"<string>launchd</string>",
		"<string>agent</string>", // the default wrapped tick
		"<key>StartInterval</key>",
		"<integer>300</integer>",
	)
	// Label is derived from the loop id with the path separator sanitized.
	mustContain(t, out, "<string>fak-loop-issue-dispatch-default</string>")
}

func TestCronEmitSystemd(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--interval", "5m", "--loop", "nightly")
	mustContain(t, out,
		"# === fak-loop-nightly.service ===",
		"# === fak-loop-nightly.timer ===",
		"Type=oneshot",
		"ExecStart=fak loop run --loop nightly --source systemd -- fak agent",
		"OnUnitActiveSec=300s",
		"OnBootSec=300s",
		"WantedBy=timers.target",
	)
}

func TestCronEmitTaskScheduler(t *testing.T) {
	out := emitCron(t, "--target", "taskscheduler", "--interval", "5m", "--loop", "nightly")
	mustContain(t, out,
		"Register-ScheduledTask",
		"New-ScheduledTaskAction -Execute 'fak' -Argument 'loop run --loop nightly --source task-scheduler -- fak agent'",
		"New-TimeSpan -Seconds 300",
		"-MultipleInstances IgnoreNew",
		"-TaskName 'fak-loop-nightly'",
	)
}

func TestCronEmitCustomTickAndFakBin(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--loop", "nightly",
		"--fak-bin", "/usr/local/bin/fak", "--", "/usr/local/bin/fak", "agent", "--offline")
	// The wrapped tick lands verbatim after the `--`, and --fak-bin drives the
	// invoked binary on the loop-run side.
	mustContain(t, out,
		"ExecStart=/usr/local/bin/fak loop run --loop nightly --source systemd -- /usr/local/bin/fak agent --offline",
	)
}

func TestCronEmitLedgerPassthrough(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--loop", "nightly", "--ledger", "/var/lib/fak/loops.jsonl")
	mustContain(t, out, "--ledger /var/lib/fak/loops.jsonl")
}

// TestCronEmitCommandSystemd proves --command emits a unit whose ExecStart is the
// arbitrary command verbatim — no `fak loop run` wrapper — and matches the garden
// watchdog's spec name so `fak start` auto-heals what `cron emit` produced (#1385).
func TestCronEmitCommandSystemd(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--label", "fleet-stale-work-garden",
		"--command", "fak garden --check", "--interval", "1h")
	mustContain(t, out,
		"# === fleet-stale-work-garden.service ===",
		"# === fleet-stale-work-garden.timer ===",
		"Type=oneshot",
		"ExecStart=fak garden --check",
		"OnUnitActiveSec=3600s",
		"WantedBy=timers.target",
	)
	if strings.Contains(out, "loop run") {
		t.Fatalf("--command unit must not carry a `fak loop run` wrapper:\n%s", out)
	}
}

// TestCronEmitCommandLaunchd proves the arbitrary command lands as ProgramArguments
// (one <string> per token) with the watchdog's com.fleet.* label.
func TestCronEmitCommandLaunchd(t *testing.T) {
	out := emitCron(t, "--target", "launchd", "--label", "com.fleet.stale-work-garden",
		"--command", "fak garden --check", "--interval", "1h")
	assertWellFormedXML(t, out)
	mustContain(t, out,
		"<string>com.fleet.stale-work-garden</string>",
		"<string>fak</string>",
		"<string>garden</string>",
		"<string>--check</string>",
		"<integer>3600</integer>",
	)
	if strings.Contains(out, "<string>loop</string>") || strings.Contains(out, "<string>run</string>") {
		t.Fatalf("--command unit must not carry a `fak loop run` wrapper:\n%s", out)
	}
}

// TestCronEmitCommandTaskScheduler proves the arbitrary command becomes the task's
// -Execute (first token) + -Argument (the rest) with the watchdog's PascalCase name.
func TestCronEmitCommandTaskScheduler(t *testing.T) {
	out := emitCron(t, "--target", "taskscheduler", "--label", "FleetStaleWorkGarden",
		"--command", "fak garden --check", "--interval", "1h")
	mustContain(t, out,
		"Register-ScheduledTask",
		"New-ScheduledTaskAction -Execute 'fak' -Argument 'garden --check'",
		"New-TimeSpan -Seconds 3600",
		"-TaskName 'FleetStaleWorkGarden'",
	)
	if strings.Contains(out, "loop run") {
		t.Fatalf("--command unit must not carry a `fak loop run` wrapper:\n%s", out)
	}
}

// TestCronEmitCommandQuoting proves a multi-word argument (one shlex token with an
// embedded space) survives into each unit re-quoted faithfully, not split apart.
func TestCronEmitCommandQuoting(t *testing.T) {
	const cmd = `fak garden --note "two words" --check`
	// systemd: the spaced token is double-quoted in the ExecStart line.
	sd := emitCron(t, "--target", "systemd", "--label", "garden", "--command", cmd)
	mustContain(t, sd, `ExecStart=fak garden --note "two words" --check`)
	// launchd: each token (including the spaced one) is its own <string>.
	ld := emitCron(t, "--target", "launchd", "--label", "garden", "--command", cmd)
	assertWellFormedXML(t, ld)
	mustContain(t, ld, "<string>two words</string>", "<string>--note</string>")
	// taskscheduler: the spaced token is double-quoted inside the -Argument literal.
	ts := emitCron(t, "--target", "taskscheduler", "--label", "garden", "--command", cmd)
	mustContain(t, ts, `-Argument 'garden --note "two words" --check'`)
}

// TestCronEmitCommandDefaultLabel proves --command with no --label derives a safe
// fak-cron-<verb> name from the command's first token.
func TestCronEmitCommandDefaultLabel(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--command", "fak garden --check")
	mustContain(t, out, "# === fak-cron-fak.service ===")
}

// TestCronEmitDefaultUnchanged is a regression guard: with no --command the systemd
// unit is byte-for-byte the historical `fak loop run` form (the back-compat rung).
func TestCronEmitDefaultUnchanged(t *testing.T) {
	out := emitCron(t, "--target", "systemd", "--interval", "5m", "--loop", "nightly")
	mustContain(t, out,
		"Description=fak loop nightly (cron-emitted; OS fires, fak owns overlap-lock + missed-run policy)",
		"Description=Timer for fak loop nightly",
		"ExecStart=fak loop run --loop nightly --source systemd -- fak agent",
	)
}

func TestCronEmitCommandRejectsBadInput(t *testing.T) {
	cases := [][]string{
		{"emit", "--target", "systemd", "--command", `fak "unterminated`}, // unbalanced quote
		{"emit", "--target", "systemd", "--command", "   "},               // empty after split
	}
	for _, argv := range cases {
		var stdout, stderr bytes.Buffer
		if code := runCron(&stdout, &stderr, argv); code != 2 {
			t.Fatalf("runCron(%v) = %d, want 2 (stderr=%s stdout=%s)", argv, code, stderr.String(), stdout.String())
		}
	}
}

func TestCronEmitRejectsBadInput(t *testing.T) {
	cases := [][]string{
		{"emit"}, // no target, no loop
		{"emit", "--target", "cronjob", "nightly"},                     // unknown target
		{"emit", "--target", "systemd"},                                // no loop id
		{"emit", "--target", "systemd", "--interval", "0s", "nightly"}, // non-positive interval
		{"emit", "--target", "systemd", "--interval", "-5m", "nightly"},
	}
	for _, argv := range cases {
		var stdout, stderr bytes.Buffer
		if code := runCron(&stdout, &stderr, argv); code != 2 {
			t.Fatalf("runCron(%v) = %d, want 2 (stderr=%s stdout=%s)", argv, code, stderr.String(), stdout.String())
		}
	}
}

func TestCronUnknownSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runCron(&stdout, &stderr, []string{"frobnicate"}); code != 2 {
		t.Fatalf("unknown subcommand exit = %d, want 2", code)
	}
}

// TestCronAtMostOnceFireDuplicateTicks tests in-kernel at-most-once fire semantics
// under sequential and concurrent duplicate ticks using compare-and-set and
// hash-chained execution records (#2927).
func TestCronAtMostOnceFireDuplicateTicks(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "cron_cas_dup.jsonl")
	scheduler := NewCronScheduler(ledger)
	job := "sync-cache"
	interval := 15 * time.Minute
	t0 := mustTime(t, "2026-09-07T10:00:00Z")

	// 1. First tick for slot t0 -> admitted ("fired")
	dec1, err := scheduler.AttemptFire(job, t0, t0, interval)
	if err != nil {
		t.Fatalf("first fire error: %v", err)
	}
	if !dec1.Admitted || dec1.Outcome != cronOutcomeFired {
		t.Fatalf("expected first tick to fire, got: %+v", dec1)
	}

	// 2. Sequential duplicate tick for the same slot (5 minutes later) -> rejected ("deduped")
	tDup := t0.Add(5 * time.Minute)
	dec2, err := scheduler.AttemptFire(job, t0, tDup, interval)
	if err != nil {
		t.Fatalf("duplicate fire error: %v", err)
	}
	if dec2.Admitted || dec2.Outcome != cronOutcomeDeduped {
		t.Fatalf("expected duplicate tick to be rejected with deduped, got: %+v", dec2)
	}

	// 3. Third sequential tick for the same slot -> still rejected ("deduped")
	dec3, err := scheduler.AttemptFire(job, t0, t0.Add(10*time.Minute), interval)
	if err != nil {
		t.Fatalf("third fire error: %v", err)
	}
	if dec3.Admitted || dec3.Outcome != cronOutcomeDeduped {
		t.Fatalf("expected third tick to be rejected with deduped, got: %+v", dec3)
	}

	// 4. Concurrent duplicate ticks: 10 goroutines racing for a new slot (10:15)
	t1 := t0.Add(interval)
	var wg sync.WaitGroup
	var mu sync.Mutex
	admittedCount := 0
	dedupedCount := 0
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(offset int) {
			defer wg.Done()
			tickTime := t1.Add(time.Duration(offset) * time.Second)
			d, ferr := scheduler.AttemptFire(job, t1, tickTime, interval)
			if ferr != nil {
				t.Errorf("concurrent fire error: %v", ferr)
				return
			}
			mu.Lock()
			if d.Admitted && d.Outcome == cronOutcomeFired {
				admittedCount++
			} else if !d.Admitted && d.Outcome == cronOutcomeDeduped {
				dedupedCount++
			}
			mu.Unlock()
		}(i)
	}
	wg.Wait()

	if admittedCount != 1 {
		t.Errorf("expected exactly 1 admitted fire across concurrent ticks, got %d", admittedCount)
	}
	if dedupedCount != 9 {
		t.Errorf("expected 9 deduped rejections across concurrent ticks, got %d", dedupedCount)
	}

	// 5. Verify hash chain integrity across all recorded rows in the ledger
	count, valid, err := scheduler.VerifyChain()
	if err != nil || !valid {
		t.Fatalf("hash chain verification failed (count=%d, valid=%v): %v", count, valid, err)
	}
	// Total records: 1 (dec1) + 1 (dec2) + 1 (dec3) + 10 (concurrent) = 13 records
	if count != 13 {
		t.Errorf("expected 13 hash-chained records, got %d", count)
	}

	// 6. CLI level duplicate tick rejection
	cliLedger := filepath.Join(t.TempDir(), "cli_dup.jsonl")
	var stdout, stderr bytes.Buffer
	code1 := runCron(&stdout, &stderr, []string{
		"fire", "--job", "cli-job", "--ledger", cliLedger, "--interval", "1h", "--at", "2026-09-07T11:00:00Z",
	})
	if code1 != 0 {
		t.Fatalf("cli first fire: exit %d, want 0 (stderr=%s)", code1, stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	code2 := runCron(&stdout, &stderr, []string{
		"fire", "--job", "cli-job", "--ledger", cliLedger, "--interval", "1h", "--at", "2026-09-07T11:30:00Z",
	})
	if code2 != cronExitDeduped {
		t.Fatalf("cli second fire: exit %d, want %d (deduped; stderr=%s)", code2, cronExitDeduped, stderr.String())
	}

	cliCount, cliValid, cliErr := cronVerifyFireChain(cliLedger)
	if cliErr != nil || !cliValid {
		t.Fatalf("cli hash chain invalid: count=%d, err=%v", cliCount, cliErr)
	}
	if cliCount != 2 {
		t.Errorf("expected 2 hash-chained cli records, got %d", cliCount)
	}
}

// TestCronCatchupWindowCalculation tests catchup window calculation clamped to
// [120s, 2h] (120s <= period/2 <= 7200s) and eligibility enforcement (#2927).
func TestCronCatchupWindowCalculation(t *testing.T) {
	tests := []struct {
		name     string
		period   time.Duration
		expected time.Duration
	}{
		{
			name:     "one-shot zero period defaults to 120s grace",
			period:   0,
			expected: 120 * time.Second,
		},
		{
			name:     "negative period defaults to 120s grace",
			period:   -5 * time.Minute,
			expected: 120 * time.Second,
		},
		{
			name:     "very short period (10s): half=5s clamped to 120s min",
			period:   10 * time.Second,
			expected: 120 * time.Second,
		},
		{
			name:     "1 minute period (60s): half=30s clamped to 120s min",
			period:   1 * time.Minute,
			expected: 120 * time.Second,
		},
		{
			name:     "3 minute period (180s): half=90s clamped to 120s min",
			period:   3 * time.Minute,
			expected: 120 * time.Second,
		},
		{
			name:     "4 minute period (240s): half=120s exactly at min boundary",
			period:   4 * time.Minute,
			expected: 120 * time.Second,
		},
		{
			name:     "10 minute period (600s): half=300s (5m)",
			period:   10 * time.Minute,
			expected: 5 * time.Minute,
		},
		{
			name:     "30 minute period (1800s): half=900s (15m)",
			period:   30 * time.Minute,
			expected: 15 * time.Minute,
		},
		{
			name:     "1 hour period (3600s): half=1800s (30m)",
			period:   1 * time.Hour,
			expected: 30 * time.Minute,
		},
		{
			name:     "2 hour period (7200s): half=3600s (1h)",
			period:   2 * time.Hour,
			expected: 1 * time.Hour,
		},
		{
			name:     "4 hour period (14400s): half=7200s (2h) exactly at max boundary",
			period:   4 * time.Hour,
			expected: 2 * time.Hour,
		},
		{
			name:     "6 hour period: half=3h clamped to 2h max",
			period:   6 * time.Hour,
			expected: 2 * time.Hour,
		},
		{
			name:     "24 hour period: half=12h clamped to 2h max",
			period:   24 * time.Hour,
			expected: 2 * time.Hour,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := CronCatchupWindow(tc.period)
			if got != tc.expected {
				t.Errorf("CronCatchupWindow(%v) = %v, want %v", tc.period, got, tc.expected)
			}
		})
	}

	t.Run("CatchupWindowEligibilityEnforcement", func(t *testing.T) {
		scheduled := mustTime(t, "2026-09-07T12:00:00Z")
		period := 1 * time.Hour // catchup window is 30 minutes

		// Tick at scheduled time: eligible
		ok, _ := CronCheckEligibility(scheduled, scheduled, period)
		if !ok {
			t.Errorf("expected tick at scheduled time to be eligible")
		}

		// Tick at scheduled + 20 minutes (within 30m window): eligible
		ok, _ = CronCheckEligibility(scheduled, scheduled.Add(20*time.Minute), period)
		if !ok {
			t.Errorf("expected tick at +20m to be eligible within 30m catchup window")
		}

		// Tick at scheduled + 30 minutes (exact boundary): eligible
		ok, _ = CronCheckEligibility(scheduled, scheduled.Add(30*time.Minute), period)
		if !ok {
			t.Errorf("expected tick at +30m (exact boundary) to be eligible")
		}

		// Tick at scheduled + 31 minutes (exceeds 30m window): ineligible
		ok, reason := CronCheckEligibility(scheduled, scheduled.Add(31*time.Minute), period)
		if ok {
			t.Errorf("expected tick at +31m to be rejected past 30m catchup window")
		}
		if !strings.Contains(reason, "missed catchup window") {
			t.Errorf("expected reason to mention missed catchup window, got: %s", reason)
		}

		// Tick before scheduled time: ineligible (not due yet)
		ok, reason = CronCheckEligibility(scheduled, scheduled.Add(-10*time.Second), period)
		if ok {
			t.Errorf("expected tick before scheduled time to be ineligible")
		}
		if !strings.Contains(reason, "not due yet") {
			t.Errorf("expected reason to mention not due yet, got: %s", reason)
		}
	})
}

// TestCronOneShotGraceWindow tests the 120s grace window for missed one-shot
// executions (#2927).
func TestCronOneShotGraceWindow(t *testing.T) {
	if CronOneShotGrace != 120*time.Second {
		t.Fatalf("CronOneShotGrace = %v, want 120s", CronOneShotGrace)
	}

	scheduled := mustTime(t, "2026-09-07T14:00:00Z")
	period := time.Duration(0) // one-shot

	// 1. Tick before scheduled time: ineligible
	ok, reason := CronCheckEligibility(scheduled, scheduled.Add(-5*time.Second), period)
	if ok {
		t.Errorf("tick before scheduled time should not be eligible")
	}
	if !strings.Contains(reason, "not due yet") {
		t.Errorf("reason missing 'not due yet': %s", reason)
	}

	// 2. Tick exactly on time: eligible
	ok, _ = CronCheckEligibility(scheduled, scheduled, period)
	if !ok {
		t.Errorf("tick on scheduled time should be eligible")
	}

	// 3. Tick 60s late (within 120s grace): eligible
	ok, _ = CronCheckEligibility(scheduled, scheduled.Add(60*time.Second), period)
	if !ok {
		t.Errorf("tick at +60s should be eligible inside 120s grace window")
	}

	// 4. Tick at exactly 120s late (boundary of grace): eligible
	ok, _ = CronCheckEligibility(scheduled, scheduled.Add(120*time.Second), period)
	if !ok {
		t.Errorf("tick at +120s should be eligible at grace cutoff")
	}

	// 5. Tick 121s late (exceeds 120s grace): ineligible
	ok, reason = CronCheckEligibility(scheduled, scheduled.Add(121*time.Second), period)
	if ok {
		t.Errorf("tick at +121s should be rejected past 120s grace window")
	}
	if !strings.Contains(reason, "missed one-shot grace window") {
		t.Errorf("reason missing 'missed one-shot grace window': %s", reason)
	}

	// 6. Test firing via CronStore:
	ledger := filepath.Join(t.TempDir(), "oneshot_grace.jsonl")
	scheduler := NewCronScheduler(ledger)

	// Missed one-shot (150s late): AttemptFire should refuse with missed_grace
	dec, err := scheduler.AttemptFire("one-shot-task", scheduled, scheduled.Add(150*time.Second), period)
	if err != nil {
		t.Fatalf("attempt fire error: %v", err)
	}
	if dec.Admitted || dec.Outcome != cronOutcomeMissedGrace {
		t.Fatalf("expected missed_grace rejection for late one-shot, got: %+v", dec)
	}

	// Fresh one-shot within grace (45s late): AttemptFire should admit ("fired")
	scheduled2 := mustTime(t, "2026-09-07T14:10:00Z")
	dec2, err := scheduler.AttemptFire("one-shot-task-2", scheduled2, scheduled2.Add(45*time.Second), period)
	if err != nil {
		t.Fatalf("attempt fire error: %v", err)
	}
	if !dec2.Admitted || dec2.Outcome != cronOutcomeFired {
		t.Fatalf("expected admitted fire within grace, got: %+v", dec2)
	}

	// Check hash chain remains valid
	count, valid, err := scheduler.VerifyChain()
	if err != nil || !valid {
		t.Fatalf("hash chain invalid: count=%d, valid=%v, err=%v", count, valid, err)
	}
	if count != 2 {
		t.Errorf("expected 2 records in ledger, got %d", count)
	}
}

// TestCronHardInterruptCeilingEnforcement tests the 3-minute hard-interrupt ceiling
// on cron execution sessions (#2927).
func TestCronHardInterruptCeilingEnforcement(t *testing.T) {
	// Verify constant
	if CronHardInterruptCeiling != 3*time.Minute {
		t.Fatalf("CronHardInterruptCeiling = %v, want 3m", CronHardInterruptCeiling)
	}

	// 1. CronExecuteSession with excessive timeout (> 3m) clamps to ceiling
	t.Run("ClampCeilingAboveThreeMinutes", func(t *testing.T) {
		start := time.Now()
		err := CronExecuteSession(context.Background(), 10*time.Minute, func(ctx context.Context) error {
			deadline, ok := ctx.Deadline()
			if !ok {
				return errors.New("expected deadline to be set")
			}
			maxDeadline := start.Add(CronHardInterruptCeiling + 200*time.Millisecond)
			if deadline.After(maxDeadline) {
				return fmt.Errorf("deadline %v exceeded hard interrupt ceiling %v", deadline, maxDeadline)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("session execution failed: %v", err)
		}
	})

	// 2. CronExecuteSession with zero/negative timeout defaults to CronHardInterruptCeiling
	t.Run("ZeroOrNegativeTimeoutDefaultsToCeiling", func(t *testing.T) {
		start := time.Now()
		err := CronExecuteSession(context.Background(), 0, func(ctx context.Context) error {
			deadline, ok := ctx.Deadline()
			if !ok {
				return errors.New("expected deadline to be set")
			}
			maxDeadline := start.Add(CronHardInterruptCeiling + 200*time.Millisecond)
			if deadline.After(maxDeadline) {
				return fmt.Errorf("deadline %v exceeded hard interrupt ceiling %v", deadline, maxDeadline)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("session execution failed: %v", err)
		}
	})

	// 3. CronExecuteSession forcefully interrupts runaway task exceeding timeout
	t.Run("InterruptRunawaySession", func(t *testing.T) {
		err := CronExecuteSession(context.Background(), 50*time.Millisecond, func(ctx context.Context) error {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(5 * time.Second):
				return nil
			}
		})
		if err == nil {
			t.Fatalf("expected runaway session to be interrupted with error, got nil")
		}
		if !errors.Is(err, ErrCronHardInterrupt) {
			t.Errorf("expected error to wrap ErrCronHardInterrupt, got: %v", err)
		}
	})

	// 4. CLI cron run enforcement of hard interrupt ceiling
	t.Run("CronRunCeilingClampingAndInterruption", func(t *testing.T) {
		ledger := filepath.Join(t.TempDir(), "cron_run_ceiling.jsonl")
		var stdout, stderr bytes.Buffer

		var cmdArgs []string
		if runtime.GOOS == "windows" {
			cmdArgs = []string{"powershell", "-NoProfile", "-NonInteractive", "-Command", "Start-Sleep -Seconds 2"}
		} else {
			cmdArgs = []string{"sleep", "2"}
		}

		argv := append([]string{
			"run",
			"--job", "job-ceiling-test",
			"--ledger", ledger,
			"--interval", "1h",
			"--timeout", "50ms",
			"--",
		}, cmdArgs...)

		code := runCron(&stdout, &stderr, argv)
		if code != cronRunExitTimeout {
			t.Fatalf("expected exit code %d (timeout), got %d (stderr=%s)", cronRunExitTimeout, code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "status=timeout") {
			t.Errorf("stdout missing status=timeout: %s", stdout.String())
		}
	})
}
