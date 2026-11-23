package agentqueue

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func fixedJournalClock() time.Time { return time.Date(2030, 1, 2, 3, 4, 5, 0, time.UTC) }

func openTestStepJournal(t *testing.T, dir string) *StepJournal {
	t.Helper()
	j, err := OpenStepJournal(dir, StepJournalOptions{
		Now:  fixedJournalClock,
		Wait: func(context.Context, time.Duration) error { t.Fatal("unexpected wait"); return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	return j
}

func mustRun(t *testing.T, j *StepJournal, runID string) *StepRun {
	t.Helper()
	run, err := j.Run(runID)
	if err != nil {
		t.Fatal(err)
	}
	return run
}

func countingStep[T any](calls *int, out T, err error) func(context.Context) (T, error) {
	return func(context.Context) (T, error) {
		*calls++
		return out, err
	}
}

func TestStepJournalSkipsCompletedStepOnReplay(t *testing.T) {
	ctx := context.Background()
	j := openTestStepJournal(t, t.TempDir())
	calls := 0
	type plan struct{ Files []string }
	want := plan{Files: []string{"a.go", "b.go"}}

	for execution := 0; execution < 3; execution++ {
		run := mustRun(t, j, "attempt-1")
		got, err := RunStep(ctx, run, "plan", countingStep(&calls, want, nil))
		if err != nil {
			t.Fatalf("execution %d: %v", execution, err)
		}
		if len(got.Files) != 2 || got.Files[1] != "b.go" {
			t.Fatalf("execution %d: output = %+v", execution, got)
		}
	}
	if calls != 1 {
		t.Fatalf("step body ran %d times, want 1", calls)
	}
}

func TestStepJournalReplaysRecordedError(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	calls := 0
	stepErr := errors.New("tool exited 3")

	first := mustRun(t, openTestStepJournal(t, dir), "attempt-err")
	if _, err := RunStep(ctx, first, "build", countingStep(&calls, 0, stepErr)); !errors.Is(err, stepErr) {
		t.Fatalf("first execution err = %v, want original step error", err)
	}

	replay := mustRun(t, openTestStepJournal(t, dir), "attempt-err")
	_, err := RunStep(ctx, replay, "build", countingStep(&calls, 42, nil))
	var recorded *RecordedStepError
	if !errors.As(err, &recorded) || recorded.Message != "tool exited 3" || recorded.StepID != 0 || recorded.Name != "build" {
		t.Fatalf("replay err = %#v, want recorded step error", err)
	}
	if calls != 1 {
		t.Fatalf("step body ran %d times, want 1", calls)
	}
}

func TestStepJournalResumesAtStepBoundaryAfterCrash(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	calls := map[string]int{}
	step := func(name string, out int) func(context.Context) (int, error) {
		return func(context.Context) (int, error) {
			calls[name]++
			return out, nil
		}
	}

	crashed := mustRun(t, openTestStepJournal(t, dir), "attempt-crash")
	for i, name := range []string{"fetch", "edit"} {
		if _, err := RunStep(ctx, crashed, name, step(name, i+10)); err != nil {
			t.Fatal(err)
		}
	}
	// The process dies here, before "test" and "land" run.

	resumed := mustRun(t, openTestStepJournal(t, dir), "attempt-crash")
	var outputs []int
	for i, name := range []string{"fetch", "edit", "test", "land"} {
		got, err := RunStep(ctx, resumed, name, step(name, i+10))
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, got)
	}
	for i, got := range outputs {
		if got != i+10 {
			t.Fatalf("outputs = %v", outputs)
		}
	}
	for _, n := range calls {
		if n != 1 {
			t.Fatalf("calls = %v, want each step exactly once", calls)
		}
	}
	if len(calls) != 4 {
		t.Fatalf("calls = %v, want four steps", calls)
	}
}

func TestStepJournalToleratesTornTail(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	calls := 0

	j := openTestStepJournal(t, dir)
	run := mustRun(t, j, "attempt-torn")
	if _, err := RunStep(ctx, run, "first", countingStep(&calls, "one", nil)); err != nil {
		t.Fatal(err)
	}
	path := j.runPath("attempt-torn")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	// Longer than any record, so a resumed append that skipped the truncate
	// would leave junk behind instead of overwriting it.
	torn := `{"schema":"fak.agentqueue.step-journal.v1","run_id":"attempt-torn","kind":"step","name":"` + strings.Repeat("x", 512)
	if _, err := f.WriteString(torn); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	resumed := mustRun(t, openTestStepJournal(t, dir), "attempt-torn")
	if got, err := RunStep(ctx, resumed, "first", countingStep(&calls, "other", nil)); err != nil || got != "one" {
		t.Fatalf("first replay = %q, %v", got, err)
	}
	if got, err := RunStep(ctx, resumed, "second", countingStep(&calls, "two", nil)); err != nil || got != "two" {
		t.Fatalf("second = %q, %v", got, err)
	}
	if calls != 2 {
		t.Fatalf("step bodies ran %d times, want 2", calls)
	}

	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasSuffix(body, []byte("\n")) {
		t.Fatalf("journal does not end on a record boundary: %q", body)
	}
	lines := bytes.Split(bytes.TrimSuffix(body, []byte("\n")), []byte("\n"))
	if len(lines) != 2 {
		t.Fatalf("journal has %d lines, want 2: %q", len(lines), body)
	}
	for _, line := range lines {
		if !json.Valid(line) {
			t.Fatalf("journal line is not a complete record: %q", line)
		}
	}

	again := mustRun(t, openTestStepJournal(t, dir), "attempt-torn")
	for _, name := range []string{"first", "second"} {
		if _, err := RunStep(ctx, again, name, countingStep(&calls, "x", nil)); err != nil {
			t.Fatalf("%s after torn-tail repair: %v", name, err)
		}
	}
	if calls != 2 {
		t.Fatalf("step bodies ran %d times after repair, want 2", calls)
	}
}

func TestStepJournalRejectsCorruptInteriorLine(t *testing.T) {
	dir := t.TempDir()
	j := openTestStepJournal(t, dir)
	if err := os.WriteFile(j.runPath("attempt-bad"), []byte("not json\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := j.Run("attempt-bad"); !errors.Is(err, ErrStepJournalCorrupt) {
		t.Fatalf("err = %v, want ErrStepJournalCorrupt", err)
	}
}

func TestStepJournalStepNameMismatchIsNondeterministic(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	calls := 0
	first := mustRun(t, openTestStepJournal(t, dir), "attempt-nd")
	if _, err := RunStep(ctx, first, "plan", countingStep(&calls, 1, nil)); err != nil {
		t.Fatal(err)
	}

	replay := mustRun(t, openTestStepJournal(t, dir), "attempt-nd")
	if _, err := RunStep(ctx, replay, "deploy", countingStep(&calls, 2, nil)); !errors.Is(err, ErrStepNondeterministic) {
		t.Fatalf("err = %v, want ErrStepNondeterministic", err)
	}
	if calls != 1 {
		t.Fatalf("mismatched step body ran; calls = %d", calls)
	}
}

func TestStepJournalCancelRetriedUntilCommitted(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	var waits []time.Duration
	j, err := OpenStepJournal(dir, StepJournalOptions{
		Now: fixedJournalClock,
		Wait: func(_ context.Context, d time.Duration) error {
			waits = append(waits, d)
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	failures := 9
	commits := 0
	j.appendLine = func(path string, offset int64, line []byte) error {
		if failures > 0 {
			failures--
			return errors.New("disk unavailable")
		}
		commits++
		return appendStepJournalLine(path, offset, line)
	}

	if err := j.CancelRun(ctx, "attempt-cancel", "operator stop"); err != nil {
		t.Fatal(err)
	}
	if commits != 1 || len(waits) != 9 {
		t.Fatalf("commits = %d, waits = %d; want 1 commit after 9 backoffs", commits, len(waits))
	}
	for i := 1; i < len(waits); i++ {
		if waits[i] < waits[i-1] || waits[i] > cancelBackoffMax {
			t.Fatalf("backoff not monotone and capped: %v", waits)
		}
	}
	if waits[0] != cancelBackoffBase || waits[len(waits)-1] != cancelBackoffMax {
		t.Fatalf("backoff schedule = %v", waits)
	}

	reopened := mustRun(t, openTestStepJournal(t, dir), "attempt-cancel")
	if _, err := RunStep(ctx, reopened, "plan", countingStep(new(int), 1, nil)); !errors.Is(err, ErrRunCancelled) {
		t.Fatalf("after reopen err = %v, want ErrRunCancelled", err)
	}
}

func TestStepJournalCancelGivesUpOnlyWhenContextEnds(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	attempts := 0
	j, err := OpenStepJournal(t.TempDir(), StepJournalOptions{
		Now: fixedJournalClock,
		Wait: func(ctx context.Context, _ time.Duration) error {
			if attempts == 4 {
				cancel()
			}
			return ctx.Err()
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	j.appendLine = func(string, int64, []byte) error {
		attempts++
		return errors.New("disk unavailable")
	}
	err = j.CancelRun(ctx, "attempt-stuck", "stop")
	if !errors.Is(err, context.Canceled) || attempts != 4 {
		t.Fatalf("err = %v after %d attempts", err, attempts)
	}
	run := mustRun(t, j, "attempt-stuck")
	j.appendLine = nil
	if _, err := RunStep(context.Background(), run, "plan", countingStep(new(int), 1, nil)); err != nil {
		t.Fatalf("uncommitted cancel must not refuse steps: %v", err)
	}
}

func TestStepJournalRefusesStepsAfterCancel(t *testing.T) {
	ctx := context.Background()
	j := openTestStepJournal(t, t.TempDir())
	calls := 0
	run := mustRun(t, j, "attempt-stop")
	if _, err := RunStep(ctx, run, "plan", countingStep(&calls, 1, nil)); err != nil {
		t.Fatal(err)
	}
	if err := j.CancelRun(ctx, "attempt-stop", "operator stop"); err != nil {
		t.Fatal(err)
	}
	if _, err := RunStep(ctx, run, "edit", countingStep(&calls, 2, nil)); !errors.Is(err, ErrRunCancelled) {
		t.Fatalf("err = %v, want ErrRunCancelled", err)
	}
	replay := mustRun(t, j, "attempt-stop")
	if _, err := RunStep(ctx, replay, "plan", countingStep(&calls, 1, nil)); !errors.Is(err, ErrRunCancelled) {
		t.Fatalf("replay of completed step err = %v, want ErrRunCancelled", err)
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1", calls)
	}
}

func TestStepJournalCancelDuringStepDiscardsOutcome(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	j := openTestStepJournal(t, dir)
	run := mustRun(t, j, "attempt-mid")
	_, err := RunStep(ctx, run, "edit", func(context.Context) (int, error) {
		if err := j.CancelRun(ctx, "attempt-mid", "stop"); err != nil {
			t.Fatal(err)
		}
		return 7, nil
	})
	if !errors.Is(err, ErrRunCancelled) {
		t.Fatalf("err = %v, want ErrRunCancelled", err)
	}
	if got := len(j.runs["attempt-mid"].steps); got != 0 {
		t.Fatalf("recorded %d steps, want the interrupted step unrecorded", got)
	}
}

func TestStepJournalInterruptedContextLeavesStepUnrecorded(t *testing.T) {
	dir := t.TempDir()
	calls := 0
	ctx, cancel := context.WithCancel(context.Background())
	run := mustRun(t, openTestStepJournal(t, dir), "attempt-ctx")
	_, err := RunStep(ctx, run, "edit", func(context.Context) (int, error) {
		calls++
		cancel()
		return 1, nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	resumed := mustRun(t, openTestStepJournal(t, dir), "attempt-ctx")
	if _, err := RunStep(context.Background(), resumed, "edit", countingStep(&calls, 2, nil)); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want the interrupted step re-executed", calls)
	}
}

func TestStepJournalIsolatesRuns(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	j := openTestStepJournal(t, dir)
	callsA, callsB := 0, 0

	if got, err := RunStep(ctx, mustRun(t, j, "attempt-a"), "plan", countingStep(&callsA, "a-plan", nil)); err != nil || got != "a-plan" {
		t.Fatalf("a = %q, %v", got, err)
	}
	if got, err := RunStep(ctx, mustRun(t, j, "attempt-b"), "survey", countingStep(&callsB, "b-survey", nil)); err != nil || got != "b-survey" {
		t.Fatalf("b = %q, %v", got, err)
	}
	if err := j.CancelRun(ctx, "attempt-a", "stop"); err != nil {
		t.Fatal(err)
	}

	reopened := openTestStepJournal(t, dir)
	if got, err := RunStep(ctx, mustRun(t, reopened, "attempt-b"), "survey", countingStep(&callsB, "other", nil)); err != nil || got != "b-survey" {
		t.Fatalf("b replay = %q, %v", got, err)
	}
	if _, err := RunStep(ctx, mustRun(t, reopened, "attempt-a"), "plan", countingStep(&callsA, "x", nil)); !errors.Is(err, ErrRunCancelled) {
		t.Fatalf("a err = %v, want ErrRunCancelled", err)
	}
	if callsA != 1 || callsB != 1 {
		t.Fatalf("callsA = %d, callsB = %d", callsA, callsB)
	}
}

func TestStepJournalLateExecutionObservesFirstRecordedOutput(t *testing.T) {
	ctx := context.Background()
	j := openTestStepJournal(t, t.TempDir())
	first := mustRun(t, j, "attempt-dup")
	second := mustRun(t, j, "attempt-dup")

	got, err := RunStep(ctx, second, "plan", func(context.Context) (string, error) {
		if out, err := RunStep(ctx, first, "plan", countingStep(new(int), "first", nil)); err != nil || out != "first" {
			t.Fatalf("first execution = %q, %v", out, err)
		}
		return "second", nil
	})
	if err != nil || got != "first" {
		t.Fatalf("late execution = %q, %v; want the recorded output", got, err)
	}

	replay := mustRun(t, openTestStepJournal(t, j.dir), "attempt-dup")
	if out, err := RunStep(ctx, replay, "plan", countingStep(new(int), "x", nil)); err != nil || out != "first" {
		t.Fatalf("replay = %q, %v", out, err)
	}
}

func TestStepJournalCancelBeforeFirstStepIsPreemptive(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	j := openTestStepJournal(t, dir)
	if err := j.CancelRun(ctx, "attempt-unstarted", "stop"); err != nil {
		t.Fatal(err)
	}
	calls := 0
	run := mustRun(t, openTestStepJournal(t, dir), "attempt-unstarted")
	if _, err := RunStep(ctx, run, "plan", countingStep(&calls, 1, nil)); !errors.Is(err, ErrRunCancelled) {
		t.Fatalf("err = %v, want ErrRunCancelled", err)
	}
	if calls != 0 {
		t.Fatalf("calls = %d, want 0", calls)
	}
}
