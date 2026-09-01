package gitdaily

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/gitgate"
	"github.com/anthony-chaudhary/fak/internal/treedoctor"
)

// countObjectsBefore/After are `git count-objects -vH` fixtures: a loose backlog, then
// the same object DB after a fold. The pack count falls because incremental-repack may
// remove fully-covered redundant packs; that is safe consolidation, not object loss.
const (
	countObjectsBefore = "count: 4200\nsize: 16.00 MiB\nin-pack: 90000\npacks: 5\nsize-pack: 400.00 MiB\nprune-packable: 0\ngarbage: 0\nsize-garbage: 0 bytes\n"
	countObjectsAfter  = "count: 12\nsize: 48.00 KiB\nin-pack: 94188\npacks: 3\nsize-pack: 404.00 MiB\nprune-packable: 0\ngarbage: 0\nsize-garbage: 0 bytes\n"
)

// fakeRepo lays down the two directories a tick addresses — the checkout and its common
// dir — and returns the Options skeleton pointing at them.
func fakeRepo(t *testing.T) Options {
	t.Helper()
	root := t.TempDir()
	common := filepath.Join(root, ".git")
	if err := os.MkdirAll(common, 0o755); err != nil {
		t.Fatal(err)
	}
	return Options{
		RepoRoot: root, GitCommonDir: common,
		IndexLockSweep: func(context.Context, string, time.Time, bool) IndexLockSweep {
			return IndexLockSweep{}
		},
	}
}

// ghostLeaseLock writes an orphaned refs/fak/locks/<name> aged past the orphan bound —
// the #4605 residue a holder killed mid-CAS leaves behind forever.
func ghostLeaseLock(t *testing.T, opts Options, name string, age time.Duration, now time.Time) string {
	t.Helper()
	dir := filepath.Join(opts.GitCommonDir, "refs", "fak", "locks")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	stamp := now.Add(-age)
	if err := os.Chtimes(p, stamp, stamp); err != nil {
		t.Fatal(err)
	}
	return p
}

// recordingRunner answers the git reads gitgate makes with a SAFE posture and a folding
// object DB, recording every argv so a test can assert on what was (and was not) run.
func recordingRunner(calls *[]string) Runner {
	counts := 0
	return func(_ context.Context, _ string, args ...string) (string, int, error) {
		joined := strings.Join(args, " ")
		*calls = append(*calls, joined)
		switch {
		case joined == "count-objects -vH":
			counts++
			if counts == 1 {
				return countObjectsBefore, 0, nil
			}
			return countObjectsAfter, 0, nil
		case joined == "config --get gc.auto":
			return "0\n", 0, nil
		case joined == "config --get maintenance.auto":
			return "false\n", 0, nil
		case joined == "config --get core.untrackedCache":
			return "true\n", 0, nil
		case joined == "config --get core.fsmonitor":
			return "", 1, nil // unset: the builtin daemon is not selected, which is safe
		}
		return "", 0, nil
	}
}

// TestReapingGhostsFirstUnblocksTheTier is the ordering witness — the reason this package
// exists rather than a cron line pointed straight at `fak git-maint`. gitgate's
// quiet-window probe counts EVERY file under the lease namespace, ghosts included ("a
// stale lease also blocks — reap it first"), and nothing was reaping them. The control
// shows maintenance ALONE refusing forever; the tick shows the same tree proceeding
// because the reap happened first, in the same run.
func TestReapingGhostsFirstUnblocksTheTier(t *testing.T) {
	now := time.Date(2026, 8, 4, 3, 0, 0, 0, time.UTC)

	// Control: maintenance on its own, with the ghost in place.
	ctl := fakeRepo(t)
	ctl.Now, ctl.Apply, ctl.GracePrune = now, true, true
	ghostLeaseLock(t, ctl, "session-dead.lock", 3*time.Hour, now)
	var ctlCalls []string
	ctlRes := gitgate.RunMaint(context.Background(), gitgate.MaintRunner(recordingRunner(&ctlCalls)), gitgate.MaintOptions{
		RepoRoot: ctl.RepoRoot, GitCommonDir: ctl.GitCommonDir, Apply: true, GracePrune: true,
	})
	if ctlRes.GracePruneRefused != gitgate.MaintReasonSessionLive {
		t.Fatalf("control: GracePruneRefused = %q, want SESSION_LIVE — the fixture no longer reproduces the ghost-blocked tier",
			ctlRes.GracePruneRefused)
	}

	// The tick: same tree, same ghost, one call.
	opts := fakeRepo(t)
	opts.Now, opts.Apply, opts.GracePrune = now, true, true
	ghost := ghostLeaseLock(t, opts, "session-dead.lock", 3*time.Hour, now)
	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)

	if res.Skipped != "" {
		t.Fatalf("first run skipped as %q", res.Skipped)
	}
	if got := res.Locks.LeaseReaped; len(got) != 1 || got[0] != "session-dead.lock" {
		t.Fatalf("LeaseReaped = %v, want [session-dead.lock]", got)
	}
	if _, err := os.Stat(ghost); !os.IsNotExist(err) {
		t.Fatalf("ghost survived the tick: %v", err)
	}
	if res.Maint.GracePruneRefused != "" {
		t.Fatalf("GracePruneRefused = %q, want the tier to proceed once the ghost is gone (reap-then-maintain order broken)",
			res.Maint.GracePruneRefused)
	}
	if res.Maint.GraceRefused != "" {
		t.Fatalf("GraceRefused = %q, want the fold tier to run", res.Maint.GraceRefused)
	}
}

// TestDryRunPreviewsAndMutatesNothing pins the preview contract: it names the orphan it
// WOULD reap, leaves it on disk, runs no mutating git step, and writes no ledger row —
// so a dry run can never make the next real run think today is already done.
func TestDryRunPreviewsAndMutatesNothing(t *testing.T) {
	now := time.Date(2026, 8, 4, 3, 0, 0, 0, time.UTC)
	opts := fakeRepo(t)
	opts.Now = now
	ghost := ghostLeaseLock(t, opts, "session-dead.lock", 3*time.Hour, now)

	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)

	if res.Apply {
		t.Fatal("dry run reported Apply")
	}
	if got := res.Locks.LeaseReaped; len(got) != 1 || got[0] != "session-dead.lock" {
		t.Fatalf("dry run LeaseReaped = %v, want the previewed orphan", got)
	}
	if _, err := os.Stat(ghost); err != nil {
		t.Fatalf("dry run removed the ghost: %v", err)
	}
	for _, c := range calls {
		if strings.HasPrefix(c, "maintenance") || strings.HasPrefix(c, "prune") || strings.HasPrefix(c, "multi-pack-index") {
			t.Fatalf("dry run issued a mutating git step: %q", c)
		}
	}
	if _, err := os.Stat(LedgerPath(opts)); !os.IsNotExist(err) {
		t.Fatalf("dry run wrote a ledger row; the next real run would skip as ALREADY_RAN_TODAY")
	}
}

// TestOnceADayDedupeAndForce is what makes a coarse, catch-up-on-wake OS trigger safe:
// the second fire of the same local day does nothing, and --force is the operator's way
// back in after repairing something.
func TestOnceADayDedupeAndForce(t *testing.T) {
	now := time.Date(2026, 8, 4, 3, 0, 0, 0, time.UTC)
	opts := fakeRepo(t)
	opts.Now, opts.Apply = now, true

	var calls []string
	if res := Run(context.Background(), recordingRunner(&calls), opts); res.Skipped != "" {
		t.Fatalf("first run skipped as %q", res.Skipped)
	}
	if calls == nil {
		t.Fatal("first run issued no git commands")
	}

	// Same day, four hours later: the trigger fires again and the tick declines.
	later := opts
	later.Now = now.Add(4 * time.Hour)
	var laterCalls []string
	res := Run(context.Background(), recordingRunner(&laterCalls), later)
	if res.Skipped != SkipAlreadyRanToday {
		t.Fatalf("second same-day run: Skipped = %q, want ALREADY_RAN_TODAY", res.Skipped)
	}
	if res.LastRunDay != "2026-08-04" {
		t.Fatalf("LastRunDay = %q, want 2026-08-04", res.LastRunDay)
	}
	if len(laterCalls) != 0 {
		t.Fatalf("a skipped tick still ran git: %v", laterCalls)
	}

	// --force overrides the dedupe (but, per Options.Force, never the concurrency lock).
	forced := later
	forced.Force = true
	var forcedCalls []string
	if res := Run(context.Background(), recordingRunner(&forcedCalls), forced); res.Skipped != "" {
		t.Fatalf("--force run skipped as %q", res.Skipped)
	}

	// Tomorrow: the dedupe expires on the local calendar date, not on a 24h timer.
	tomorrow := opts
	tomorrow.Now = now.Add(23 * time.Hour) // 2026-08-05 02:00 — under 24h, but a new day
	var tomorrowCalls []string
	if res := Run(context.Background(), recordingRunner(&tomorrowCalls), tomorrow); res.Skipped != "" {
		t.Fatalf("next-day run skipped as %q; the day key is not a rolling 24h window", res.Skipped)
	}
}

// TestConcurrentTickSkipsAsBusy: a manual run and a scheduled one must not interleave.
// The loser does nothing and says so, because the holder is performing this very run.
func TestConcurrentTickSkipsAsBusy(t *testing.T) {
	opts := fakeRepo(t)
	opts.Now, opts.Apply = time.Date(2026, 8, 4, 3, 0, 0, 0, time.UTC), true

	held, err := os.OpenFile(filepath.Join(opts.GitCommonDir, TickLockName), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if err := flock.TryLock(held); err != nil {
		t.Fatalf("could not take the tick lock in the test: %v", err)
	}
	defer func() { _ = flock.Unlock(held); _ = held.Close() }()

	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)
	if res.Skipped != SkipTickBusy {
		t.Fatalf("Skipped = %q, want TICK_BUSY", res.Skipped)
	}
	if len(calls) != 0 {
		t.Fatalf("a busy-skipped tick still ran git: %v", calls)
	}

	// A dry run takes no lock: previewing the plan must never be told the job is busy.
	preview := opts
	preview.Apply = false
	if res := Run(context.Background(), recordingRunner(&calls), preview); res.Skipped != "" {
		t.Fatalf("dry run under a held lock: Skipped = %q, want it to preview anyway", res.Skipped)
	}
}

// TestTickLockFailureRefusesToRun makes serialization fail-closed. A scheduler that
// cannot create its lock must surface an incident; running anyway would let two ticks
// both pass the day-ledger check and mutate concurrently.
func TestTickLockFailureRefusesToRun(t *testing.T) {
	opts := fakeRepo(t)
	opts.Now, opts.Apply = time.Date(2026, 8, 4, 3, 0, 0, 0, time.UTC), true
	blockedCommon := filepath.Join(opts.RepoRoot, "not-a-directory")
	if err := os.WriteFile(blockedCommon, []byte("file"), 0o644); err != nil {
		t.Fatal(err)
	}
	opts.GitCommonDir = blockedCommon

	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)
	if res.TickLockErr == "" || !res.Incident || res.Skipped != "" {
		t.Fatalf("serializer failure did not fail closed: %+v", res)
	}
	if len(calls) != 0 {
		t.Fatalf("serializer failure still ran git: %v", calls)
	}
}

// TestLedgerWitnessesTheFold checks the readback an operator actually uses months later:
// one row per applied run, carrying the loose delta with the pack count steady.
func TestLedgerWitnessesTheFold(t *testing.T) {
	now := time.Date(2026, 8, 4, 3, 0, 0, 0, time.UTC)
	opts := fakeRepo(t)
	opts.Now, opts.Apply = now, true
	ghostLeaseLock(t, opts, "session-dead.lock", 3*time.Hour, now)

	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)
	if res.LedgerErr != "" {
		t.Fatalf("ledger write failed: %s", res.LedgerErr)
	}

	rows := Status(LedgerPath(opts), 0)
	if len(rows) != 1 {
		t.Fatalf("Status returned %d rows, want 1", len(rows))
	}
	r := rows[0]
	if r.Schema != Schema || r.Day != "2026-08-04" {
		t.Fatalf("row = %+v, want schema %s day 2026-08-04", r, Schema)
	}
	if r.LooseFolded() != 4188 {
		t.Fatalf("LooseFolded = %d (%d -> %d), want 4188", r.LooseFolded(), r.LooseBefore, r.LooseAfter)
	}
	if r.PacksBefore != 5 || r.PacksAfter != 3 {
		t.Fatalf("packs %d -> %d, want the captured incremental-repack result 5 -> 3", r.PacksBefore, r.PacksAfter)
	}
	if r.LeaseLocksReaped != 1 {
		t.Fatalf("LeaseLocksReaped = %d, want 1", r.LeaseLocksReaped)
	}
}

// TestLockSweepCountsOnlySuccessfulCleanup keeps the ledger honest: treedoctor's action
// stream contains both effects and diagnostics, and only effects are "locks cleared".
// An explicit failed reap is also an incident so a scheduler cannot record green.
func TestLockSweepCountsOnlySuccessfulCleanup(t *testing.T) {
	s := LockSweep{
		LeaseReaped: []string{"session-dead.lock"},
		Actions: []string{
			"reaped stale commit lock (dead PID 42)",
			"swept orphan lock residue C:/repo/.git/index.lock.stale-1",
			"advisory: loose-ref pressure — 7000 refs",
			"FAILED to reap orphaned C:/repo/.git/packed-refs.lock: access denied",
		},
	}
	if got := s.Cleared(); got != 3 {
		t.Fatalf("Cleared = %d, want 3 successful effects (lease + commit + residue)", got)
	}
	if !s.Failed() {
		t.Fatal("FAILED action did not make the lock sweep fail")
	}

	s.Actions = []string{"would reap stale commit lock (dead PID 42)", "advisory: loose-ref pressure"}
	s.LeaseReaped = nil
	if got := s.Cleared(); got != 1 {
		t.Fatalf("dry-run Cleared = %d, want one planned effect", got)
	}
	if s.Failed() {
		t.Fatal("a dry-run plan/advisory was classified as a failure")
	}

	s.LeaseErr = "access denied"
	if !s.Failed() {
		t.Fatal("lease sweep error did not make the lock sweep fail")
	}
}

// TestDeferredStreakSurfacesAStuckTier reconstructs, from the ledger alone, the signal
// the #4602 investigation had to dig out by hand: consecutive runs whose fold tier was
// held back for the same reason means the backlog is growing invisibly.
func TestDeferredStreakSurfacesAStuckTier(t *testing.T) {
	locked := func(day string) Row { return Row{Schema: Schema, Day: day, GraceRefused: "LOCKED"} }
	ok := func(day string) Row { return Row{Schema: Schema, Day: day} }

	if n, reason := DeferredStreak([]Row{ok("2026-08-01"), locked("2026-08-02"), locked("2026-08-03")}); n != 2 || reason != "LOCKED" {
		t.Fatalf("streak = (%d, %q), want (2, LOCKED)", n, reason)
	}
	if n, _ := DeferredStreak([]Row{locked("2026-08-01"), locked("2026-08-02"), ok("2026-08-03")}); n != 0 {
		t.Fatalf("streak = %d after a healthy latest run, want 0", n)
	}
	if n, _ := DeferredStreak(nil); n != 0 {
		t.Fatal("empty history reported a streak")
	}
	// A reason CHANGE ends the streak: "LOCKED for 9 days" and "LOCKED then
	// POSTURE_DRIFT" are different operator stories and must not be summed.
	mixed := []Row{locked("2026-08-01"), {Schema: Schema, Day: "2026-08-02", GraceRefused: "POSTURE_DRIFT"}}
	if n, reason := DeferredStreak(mixed); n != 1 || reason != "POSTURE_DRIFT" {
		t.Fatalf("mixed streak = (%d, %q), want (1, POSTURE_DRIFT)", n, reason)
	}
}

// TestStatusOnMissingLedgerIsEmptyNotAnError keeps the first-run contract: a clone that
// has never run the tick reads as no history, so the very first fire is not a skip.
func TestStatusOnMissingLedgerIsEmptyNotAnError(t *testing.T) {
	if rows := Status(filepath.Join(t.TempDir(), "absent.jsonl"), 0); len(rows) != 0 {
		t.Fatalf("missing ledger returned %d rows", len(rows))
	}
}

func TestRunPersistsGoCacheReceiptInScheduledLedger(t *testing.T) {
	opts := fakeRepo(t)
	opts.Now = time.Date(2026, 8, 31, 12, 0, 0, 0, time.Local)
	opts.Apply = true
	root := filepath.Join(t.TempDir(), "go-build")
	if err := os.MkdirAll(filepath.Join(root, "00"), 0o755); err != nil {
		t.Fatal(err)
	}
	entry := filepath.Join(root, "00", "stale-a")
	if err := os.WriteFile(entry, []byte("12345678"), 0o600); err != nil {
		t.Fatal(err)
	}
	old := opts.Now.Add(-48 * time.Hour)
	if err := os.Chtimes(entry, old, old); err != nil {
		t.Fatal(err)
	}
	opts.GoCacheDir = root
	opts.GoCacheOptions = treedoctor.GoCacheOptions{HighBytes: 1, LowBytes: 1, MinAge: time.Hour, FreeBytesKnown: true, FreeBytes: 1 << 40, ActiveBuild: func() (bool, error) { return false, nil }}
	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)
	if res.LedgerErr != "" {
		t.Fatalf("ledger error: %s", res.LedgerErr)
	}
	rows := Status(res.LedgerPath, 1)
	if len(rows) != 1 || rows[0].GoCache.Root == "" || rows[0].GoCache.ReclaimedBytes != 8 || rows[0].GoCache.Reaped != 1 {
		t.Fatalf("rows=%+v result=%+v", rows, res.GoCache)
	}
}

func TestRunInvokesGoCacheLifecycleAndReturnsReceipt(t *testing.T) {
	opts := fakeRepo(t)
	opts.Now = time.Now()
	root := filepath.Join(t.TempDir(), "go-build")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(root, "aa")
	if err := os.Mkdir(p, 0o755); err != nil {
		t.Fatal(err)
	}
	f := filepath.Join(p, "entry")
	if err := os.WriteFile(f, []byte("12345678"), 0o600); err != nil {
		t.Fatal(err)
	}
	old := opts.Now.Add(-48 * time.Hour)
	if err := os.Chtimes(f, old, old); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(p, old, old); err != nil {
		t.Fatal(err)
	}
	opts.GoCacheDir = root
	opts.GoCacheOptions = treedoctor.GoCacheOptions{HighBytes: 1, LowBytes: 1, MinAge: time.Hour, FreeBytesKnown: true, FreeBytes: 1 << 40}
	var calls []string
	res := Run(context.Background(), recordingRunner(&calls), opts)
	if res.GoCache.Root == "" || res.GoCache.BytesBefore != 8 || res.GoCache.BytesAfterSemantics != "projected" || len(res.GoCache.Candidates) != 1 {
		t.Fatalf("go cache receipt = %+v", res.GoCache)
	}
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("dry-run caller mutated cache: %v", err)
	}
}
