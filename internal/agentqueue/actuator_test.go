package agentqueue

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

type recordingRunner struct{ calls [][]string }

func (r *recordingRunner) Run(_ context.Context, name string, args ...string) error {
	r.calls = append(r.calls, append([]string{name}, args...))
	return nil
}

type launchingRecordingRunner struct {
	calls  [][]string
	runErr error
}

func (r *launchingRecordingRunner) Run(ctx context.Context, name string, args ...string) error {
	r.calls = append(r.calls, append([]string{name}, args...))
	statePath, ok := commandFlag(args, "--agentqueue-state")
	if !ok {
		return errors.New("test runner: missing --agentqueue-state")
	}
	attemptID, ok := commandFlag(args, "--agentqueue-attempt")
	if !ok {
		return errors.New("test runner: missing --agentqueue-attempt")
	}
	nonce, ok := commandFlag(args, "--agentqueue-nonce")
	if !ok || nonce == "" {
		return errors.New("test runner: missing --agentqueue-nonce")
	}
	snapshot, err := FileStore(statePath).Load()
	if err != nil {
		return fmt.Errorf("test runner load launch: %w", err)
	}
	attempt, err := findLifecycleAttempt(&snapshot, attemptID)
	if err != nil {
		return fmt.Errorf("test runner find launch: %w", err)
	}
	if attempt.State != AttemptLaunching || attempt.Nonce != nonce || attempt.LaunchDeadline.IsZero() {
		return fmt.Errorf("test runner observed attempt state=%q nonce=%q deadline=%v", attempt.State, attempt.Nonce, attempt.LaunchDeadline)
	}
	return r.runErr
}

type blockingLifecycleRunner struct {
	launchingRecordingRunner
	entered chan struct{}
	release chan struct{}
}

func (r *blockingLifecycleRunner) Run(ctx context.Context, name string, args ...string) error {
	if err := r.launchingRecordingRunner.Run(ctx, name, args...); err != nil {
		return err
	}
	close(r.entered)
	select {
	case <-r.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

type competingNonceRunner struct {
	launchingRecordingRunner
	competingErr error
}

func (r *competingNonceRunner) Run(ctx context.Context, name string, args ...string) error {
	if err := r.launchingRecordingRunner.Run(ctx, name, args...); err != nil {
		return err
	}
	statePath, _ := commandFlag(args, "--agentqueue-state")
	attemptID, _ := commandFlag(args, "--agentqueue-attempt")
	_, _, r.competingErr = FileStore(statePath).BeginLaunching(ctx, attemptID, "competing-stale-nonce", time.Now().Add(time.Hour))
	return nil
}

func commandFlag(args []string, name string) (string, bool) {
	for index := 0; index+1 < len(args); index++ {
		if args[index] == name {
			return args[index+1], true
		}
	}
	return "", false
}

func TestActuateRoutesDisjointReservationsThroughGuardedDispatch(t *testing.T) {
	snapshot := Snapshot{Intents: []Intent{
		{ID: "docs", Launch: LaunchSpec{Issue: 101, Lane: "docs"}},
		{ID: "gateway", Launch: LaunchSpec{Issue: 202, Lane: "gateway"}},
	}}
	starts := []StartAction{
		{IntentID: "docs", IdempotencyKey: "start:docs"},
		{IntentID: "gateway", IdempotencyKey: "start:gateway"},
	}
	runner := &recordingRunner{}
	receipts, err := Actuate(context.Background(), "fak", snapshot, starts, runner)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]string{
		{"fak", "dispatch", "tick", "--target-issue", "101", "--lane", "docs", "--lease-id", "start:docs", "--live", "--json"},
		{"fak", "dispatch", "tick", "--target-issue", "202", "--lane", "gateway", "--lease-id", "start:gateway", "--live", "--json"},
	}
	if !reflect.DeepEqual(runner.calls, want) {
		t.Fatalf("calls = %#v, want %#v", runner.calls, want)
	}
	if len(receipts) != 2 || receipts[0].IdempotencyKey != "start:docs" || receipts[1].IdempotencyKey != "start:gateway" {
		t.Fatalf("receipts = %#v", receipts)
	}
}

func TestActuateRejectsUnroutableIntentBeforeExecution(t *testing.T) {
	runner := &recordingRunner{}
	_, err := Actuate(context.Background(), "fak", Snapshot{Intents: []Intent{{ID: "bad"}}}, []StartAction{{IntentID: "bad"}}, runner)
	if err == nil {
		t.Fatal("Actuate accepted intent without issue/lane")
	}
	if len(runner.calls) != 0 {
		t.Fatalf("executed %#v", runner.calls)
	}
}

func TestActuateReservedBeginsLifecycleBeforeRunner(t *testing.T) {
	store, snapshot, starts := reservedActuatorFixture(t, 1)
	runner := &launchingRecordingRunner{}

	receipts, err := ActuateReserved(context.Background(), "fak", store, snapshot, starts, runner)
	if err != nil {
		t.Fatalf("ActuateReserved: %v", err)
	}
	if len(receipts) != 1 || len(runner.calls) != 1 {
		t.Fatalf("receipts=%+v calls=%d, want one lifecycle launch", receipts, len(runner.calls))
	}
	loaded, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	nonce, hasNonce := commandFlag(runner.calls[0][1:], "--agentqueue-nonce")
	if !hasNonce || nonce == "" || loaded.Attempts[0].Nonce != nonce {
		t.Fatalf("runner nonce=%q present=%v stored=%q", nonce, hasNonce, loaded.Attempts[0].Nonce)
	}
	if loaded.Attempts[0].State != AttemptLaunching || loaded.Attempts[0].LaunchDeadline.IsZero() {
		t.Fatalf("attempt = %+v, want durable launching identity", loaded.Attempts[0])
	}
}

func TestActuateLifecycleFencesCompetingNonce(t *testing.T) {
	store, snapshot, starts := reservedActuatorFixture(t, 1)
	runner := &competingNonceRunner{}

	if _, err := ActuateReserved(context.Background(), "fak", store, snapshot, starts, runner); err != nil {
		t.Fatalf("ActuateReserved: %v", err)
	}
	if !errors.Is(runner.competingErr, ErrFenced) {
		t.Fatalf("competing BeginLaunching error = %v, want ErrFenced", runner.competingErr)
	}
	loaded, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	winner, _ := commandFlag(runner.calls[0][1:], "--agentqueue-nonce")
	if loaded.Attempts[0].State != AttemptLaunching || loaded.Attempts[0].Nonce != winner {
		t.Fatalf("winner changed after competing nonce: %+v", loaded.Attempts[0])
	}
}

func TestActuateReservedRetainsClaimedLaunchOnRunnerErrorAndReturnsRemainder(t *testing.T) {
	store, snapshot, starts := reservedActuatorFixture(t, 2)
	runErr := errors.New("dispatch result unavailable")
	runner := &launchingRecordingRunner{runErr: runErr}

	receipts, err := ActuateReserved(context.Background(), "fak", store, snapshot, starts, runner)
	if !errors.Is(err, runErr) {
		t.Fatalf("ActuateReserved error = %v, want %v", err, runErr)
	}
	if len(receipts) != 0 || len(runner.calls) != 1 {
		t.Fatalf("receipts=%+v calls=%d, want no receipt and one call", receipts, len(runner.calls))
	}
	statePath, hasState := commandFlag(runner.calls[0][1:], "--agentqueue-state")
	attemptID, hasAttempt := commandFlag(runner.calls[0][1:], "--agentqueue-attempt")
	nonce, hasNonce := commandFlag(runner.calls[0][1:], "--agentqueue-nonce")
	if !hasState || !hasAttempt || !hasNonce || nonce == "" || attemptID != starts[0].IdempotencyKey {
		t.Fatalf("queue flags missing or wrong in command: %v", runner.calls[0])
	}
	wantPath, err := filepath.Abs(store.Path)
	if err != nil {
		t.Fatal(err)
	}
	if statePath != wantPath {
		t.Fatalf("--agentqueue-state = %q, want %q", statePath, wantPath)
	}

	loaded, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Attempts[0].State != AttemptLaunching || loaded.Attempts[0].Nonce != nonce {
		t.Fatalf("claimed attempt = %+v, want matching launching nonce", loaded.Attempts[0])
	}
	if loaded.Attempts[1].State != AttemptFailed || loaded.Intents[1].State != IntentQueued {
		t.Fatalf("unprocessed start = attempt %q intent %q, want failed/queued", loaded.Attempts[1].State, loaded.Intents[1].State)
	}

	retryRunner := &launchingRecordingRunner{}
	retry, err := (Controller{Store: store, FakPath: "fak", Runner: retryRunner}).Tick(context.Background())
	if err != nil {
		t.Fatalf("retry Tick: %v", err)
	}
	if len(retry.Plan.Start) != 1 || retry.Plan.Start[0].IntentID != starts[1].IntentID {
		t.Fatalf("retry starts = %+v, want only unprocessed intent %q", retry.Plan.Start, starts[1].IntentID)
	}
	if len(retryRunner.calls) != 1 || strings.Contains(strings.Join(retryRunner.calls[0], " "), starts[0].IdempotencyKey) {
		t.Fatalf("retry duplicated claimed launch: %v", retryRunner.calls)
	}
}

func reservedActuatorFixture(t *testing.T, count int) (Store, Snapshot, []StartAction) {
	t.Helper()
	store := FileStore(filepath.Join(t.TempDir(), "queue.json"))
	snapshot := Snapshot{
		Schema: Schema, Generation: "fixture-generation",
		Pool: PoolSpec{ID: "fixture-pool", Desired: count, Max: count},
	}
	starts := make([]StartAction, 0, count)
	for index := range count {
		intentID := fmt.Sprintf("intent-%d", index)
		attemptID := fmt.Sprintf("attempt-%d", index)
		snapshot.Intents = append(snapshot.Intents, Intent{
			ID: intentID, State: IntentQueued,
			Launch: LaunchSpec{Issue: 100 + index, Lane: fmt.Sprintf("lane-%d", index)},
		})
		snapshot.Attempts = append(snapshot.Attempts, Attempt{ID: attemptID, IntentID: intentID, State: AttemptReserved})
		starts = append(starts, StartAction{IntentID: intentID, IdempotencyKey: attemptID})
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatalf("save actuator fixture: %v", err)
	}
	return store, snapshot, starts
}
