package hooks

import (
	"context"
	"errors"
	"testing"
	"time"
)

// okRunner answers every git read successfully and instantly, recording the ctx each call was
// made with so a test can prove which bound (if any) a read inherited.
type okRunner struct{ byVerb map[string]context.Context }

func newOKRunner() *okRunner { return &okRunner{byVerb: map[string]context.Context{}} }

func (r *okRunner) run(ctx context.Context, _ string, args ...string) (string, int, error) {
	if len(args) > 0 {
		r.byVerb[args[0]] = ctx
	}
	return "", 0, nil
}

// TestReadStagedDiffWithinReturnsAtTheDeadlineEvenIfTheRunnerIgnoresCtx is the #5335 prologue
// bound. The pre-commit hook reads the staged diff BEFORE its per-gate budget loop, so a git
// blocked on `.git/index.lock` used to wedge every commit in the clone at near-zero CPU with
// nothing to cut it off. The runner here ignores ctx entirely — a bound that only works when the
// thing it bounds cooperates is not a bound — and the read must still return at the deadline.
func TestReadStagedDiffWithinReturnsAtTheDeadlineEvenIfTheRunnerIgnoresCtx(t *testing.T) {
	release := make(chan struct{})
	defer close(release) // let the abandoned goroutine finish when the test ends
	hung := func(_ context.Context, _ string, _ ...string) (string, int, error) {
		<-release
		return "", 0, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	d, err := readStagedDiffWithin(ctx, hung, t.TempDir())
	elapsed := time.Since(start)

	if !errors.Is(err, ErrCouldNotRun) {
		t.Fatalf("hung staged-diff read err = %v, want ErrCouldNotRun (fail-open)", err)
	}
	if d != nil {
		t.Fatalf("a cut-off read must yield no diff, got %+v", d)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("readStagedDiffWithin waited %s — it must return at the deadline, not for the runner", elapsed)
	}
}

// TestReadStagedDiffWithinUnbindsTheDeadlineFromLazyReads pins the property that keeps this
// bound fail-open. StagedDiff.FileBytes reports a failed `git show` as "does not resolve"
// rather than as an error, and gates like BROKEN_LINK turn "does not resolve" into a finding.
// So if the hook's (expiring) wall clock reached the lazy reads, spending the budget would
// MANUFACTURE findings — a bound that adds refusals, the exact inversion of the fail-open
// invariant it exists to protect. The returned diff must therefore not retain ctx.
func TestReadStagedDiffWithinUnbindsTheDeadlineFromLazyReads(t *testing.T) {
	r := newOKRunner()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	d, err := readStagedDiffWithin(ctx, r.run, t.TempDir())
	if err != nil {
		t.Fatalf("readStagedDiffWithin on a healthy runner: %v", err)
	}

	cancel() // the hook's wall clock is now spent, mid-gate-loop

	if _, ok := d.FileBytes("docs/some-target.md"); !ok {
		t.Fatal("FileBytes did not resolve through the runner; the lazy-read path under test never ran")
	}
	lazyCtx, seen := r.byVerb["show"]
	if !seen {
		t.Fatal("FileBytes never reached the runner, so this test proves nothing")
	}
	if lazyCtx.Err() != nil {
		t.Fatalf("lazy read inherited the spent hook bound (%v): a failed `git show` reads as "+
			"\"does not resolve\", which BROKEN_LINK turns into a finding — this would manufacture a false block", lazyCtx.Err())
	}
	if _, hasDeadline := lazyCtx.Deadline(); hasDeadline {
		t.Fatal("lazy read still carries a deadline; it must be unbound so no gate can be poisoned by the hook's wall clock")
	}
}

// TestReadStagedDiffWithinRefusesAViewThatRacedTheDeadline covers the partial-read case.
// readStagedDiffWith drops a failed sub-read silently (IndexPaths simply stays empty), so a
// diff assembled across the expiry can be silently truncated — and a truncated index view is
// what INDEX_SYNC would read as a genuine violation. Refuse the partial view instead.
func TestReadStagedDiffWithinRefusesAViewThatRacedTheDeadline(t *testing.T) {
	r := newOKRunner()
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already spent before the (instant) read completes

	d, err := readStagedDiffWithin(ctx, r.run, t.TempDir())
	if !errors.Is(err, ErrCouldNotRun) {
		t.Fatalf("read that raced the deadline err = %v, want ErrCouldNotRun", err)
	}
	if d != nil {
		t.Fatalf("a possibly-truncated diff must not be handed to the gates, got %+v", d)
	}
}

// TestReadStagedDiffWithinReturnsAHealthyReadVerbatim proves the bound is transparent: a read
// that finishes inside the deadline yields its normal diff, so wrapping the prologue never
// changes the behaviour of a clone that is working.
func TestReadStagedDiffWithinReturnsAHealthyReadVerbatim(t *testing.T) {
	r := newOKRunner()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	d, err := readStagedDiffWithin(ctx, r.run, "/repo")
	if err != nil {
		t.Fatalf("healthy read returned %v, want nil", err)
	}
	if d == nil || d.Root != "/repo" {
		t.Fatalf("healthy read must yield the normal diff, got %+v", d)
	}
	if _, seen := r.byVerb["diff"]; !seen {
		t.Fatal("the core `git diff --cached` read never ran")
	}
}
