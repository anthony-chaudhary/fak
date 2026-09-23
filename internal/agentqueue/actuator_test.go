package agentqueue

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type recordingRunner struct{ calls [][]string }

func (r *recordingRunner) Run(_ context.Context, name string, args ...string) error {
	r.calls = append(r.calls, append([]string{name}, args...))
	return nil
}

var testLaunchNonce atomic.Uint64

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
	nonce := fmt.Sprintf("test-launch-%d", testLaunchNonce.Add(1))
	if _, started, err := FileStore(statePath).BeginLaunching(ctx, attemptID, nonce, time.Now().Add(time.Hour)); err != nil {
		return fmt.Errorf("test runner begin launch: %w", err)
	} else if !started {
		return errors.New("test runner: launch was not newly claimed")
	}
	return r.runErr
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

func TestActuateReservedReturnsUnclaimedReservationToQueue(t *testing.T) {
	store, snapshot, starts := reservedActuatorFixture(t, 1)
	runner := &recordingRunner{}

	receipts, err := ActuateReserved(context.Background(), "fak", store, snapshot, starts, runner)
	if err != nil {
		t.Fatalf("ActuateReserved: %v", err)
	}
	if len(receipts) != 0 {
		t.Fatalf("receipts = %+v, want none for an unclaimed launch", receipts)
	}
	loaded, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Attempts[0].State != AttemptFailed || loaded.Intents[0].State != IntentQueued {
		t.Fatalf("unclaimed launch state = attempt %q intent %q", loaded.Attempts[0].State, loaded.Intents[0].State)
	}

	launching := &launchingRecordingRunner{}
	retry, err := (Controller{Store: store, FakPath: "fak", Runner: launching}).Tick(context.Background())
	if err != nil {
		t.Fatalf("retry Tick: %v", err)
	}
	if len(retry.Plan.Start) != 1 || len(retry.Launches) != 1 {
		t.Fatalf("retry receipt = %+v, want one replacement launch", retry)
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
	if !hasState || !hasAttempt || attemptID != starts[0].IdempotencyKey {
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
	if loaded.Attempts[0].State != AttemptLaunching {
		t.Fatalf("claimed attempt state = %q, want launching", loaded.Attempts[0].State)
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
