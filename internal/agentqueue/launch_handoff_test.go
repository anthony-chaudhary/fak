package agentqueue

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// launchObserverRunner records the argv of every guarded dispatch invocation and
// inspects the durable store at call time, so the witness proves the launch was
// fenced before the process was created rather than after.
type launchObserverRunner struct {
	store   Store
	paths   []string
	calls   [][]string
	atCall  []Snapshot
	loadErr []error
}

func (r *launchObserverRunner) Run(_ context.Context, name string, args ...string) error {
	r.calls = append(r.calls, append([]string{name}, args...))
	snapshot, err := r.store.Load()
	r.atCall = append(r.atCall, snapshot)
	r.loadErr = append(r.loadErr, err)
	return nil
}

func (r *launchObserverRunner) arg(call int, flagName string) string {
	argv := r.calls[call]
	for i := 0; i+1 < len(argv); i++ {
		if argv[i] == flagName {
			return argv[i+1]
		}
	}
	return ""
}

func futureDeadline(t *testing.T) time.Time {
	t.Helper()
	return time.Now().UTC().Add(time.Minute)
}

func attemptByID(t *testing.T, snapshot Snapshot, id string) Attempt {
	t.Helper()
	for _, attempt := range snapshot.Attempts {
		if attempt.ID == id {
			return attempt
		}
	}
	t.Fatalf("attempt %q absent from %#v", id, snapshot.Attempts)
	return Attempt{}
}

// TestControllerTickFencesLaunchBeforeHandoff drives the production
// Controller.Tick entrypoint and proves the three handoff guarantees together:
// every actuated reservation is durable as `launching` before the runner is
// invoked, the argv carries that exact attempt id/nonce/state path, and a
// BeginLaunching failure stops the launch entirely.
func TestControllerTickFencesLaunchBeforeHandoff(t *testing.T) {
	store := Store{Path: filepath.Join(t.TempDir(), "queue.json")}
	snapshot := Snapshot{
		Schema: Schema, Generation: "g0",
		Pool: PoolSpec{ID: "pool", Min: 0, Desired: 2, Max: 2},
		Intents: []Intent{
			{ID: "docs", State: IntentQueued, Launch: LaunchSpec{Issue: 101, Lane: "docs"}},
			{ID: "gateway", State: IntentQueued, Launch: LaunchSpec{Issue: 202, Lane: "gateway"}},
		},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}
	wantState, err := store.StatePath()
	if err != nil {
		t.Fatal(err)
	}

	runner := &launchObserverRunner{store: store}
	receipt, err := Controller{Store: store, FakPath: "fak", Runner: runner}.Tick(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(receipt.Launches) != 2 || len(runner.calls) != 2 {
		t.Fatalf("launches=%d calls=%d, want 2/2", len(receipt.Launches), len(runner.calls))
	}

	launched := make(map[string]Attempt, len(receipt.Launches))
	for i, launch := range receipt.Launches {
		attemptID := runner.arg(i, "--attempt-id")
		nonce := runner.arg(i, "--launch-nonce")
		statePath := runner.arg(i, "--queue-state")

		// (b) exact durable identity travels on the guarded dispatch argv.
		if attemptID != launch.IdempotencyKey || attemptID != launch.AttemptID {
			t.Fatalf("call %d attempt id = %q, receipt = %#v", i, attemptID, launch)
		}
		if nonce == "" || nonce != launch.Nonce {
			t.Fatalf("call %d nonce = %q, receipt = %#v", i, nonce, launch)
		}
		if statePath != wantState || !filepath.IsAbs(statePath) {
			t.Fatalf("call %d queue state = %q, want absolute %q", i, statePath, wantState)
		}
		if argv := runner.calls[i]; argv[0] != "fak" || argv[1] != "dispatch" || argv[2] != "tick" {
			t.Fatalf("call %d argv = %#v, want guarded dispatch tick", i, argv)
		}

		// (a) the attempt is durable as launching, with that exact nonce,
		// BEFORE the runner was invoked.
		if err := runner.loadErr[i]; err != nil {
			t.Fatalf("call %d store readback: %v", i, err)
		}
		observed := attemptByID(t, runner.atCall[i], attemptID)
		if observed.State != AttemptLaunching {
			t.Fatalf("call %d attempt %q state = %q at launch, want %q", i, attemptID, observed.State, AttemptLaunching)
		}
		if observed.Nonce != nonce {
			t.Fatalf("call %d attempt %q nonce = %q at launch, want %q", i, attemptID, observed.Nonce, nonce)
		}
		if observed.LaunchDeadline.IsZero() {
			t.Fatalf("call %d attempt %q has no bounded launch deadline", i, attemptID)
		}
		launched[attemptID] = observed
	}

	// The parent hands off only: it never registers a wrapper or advances the
	// attempt past launching, because dispatch tick owns the downstream lifecycle.
	final, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	for id, observed := range launched {
		persisted := attemptByID(t, final, id)
		if persisted.State != AttemptLaunching || persisted.Nonce != observed.Nonce || persisted.PID != 0 || !persisted.StartedAt.IsZero() {
			t.Fatalf("attempt %q after tick = %#v, want unregistered launching", id, persisted)
		}
	}
	for _, intent := range final.Intents {
		if intent.State != IntentQueued {
			t.Fatalf("intent %q state = %q after handoff, want %q", intent.ID, intent.State, IntentQueued)
		}
	}
}

// TestControllerTickFencesNonceAndRetryIdempotently proves a repeated tick
// re-reads the winning nonce instead of minting a competing one.
func TestControllerTickFencesNonceAndRetryIdempotently(t *testing.T) {
	store := Store{Path: filepath.Join(t.TempDir(), "queue.json")}
	if err := store.Save(Snapshot{
		Schema: Schema, Generation: "g0",
		Pool: PoolSpec{ID: "pool", Min: 0, Desired: 1, Max: 1},
		Intents: []Intent{
			{ID: "docs", State: IntentQueued, Launch: LaunchSpec{Issue: 101, Lane: "docs"}},
		},
	}); err != nil {
		t.Fatal(err)
	}
	reserved, _, err := store.Reserve(context.Background(), "g0")
	if err != nil {
		t.Fatal(err)
	}
	if len(reserved.Start) != 1 {
		t.Fatalf("starts = %#v", reserved.Start)
	}
	attemptID := reserved.Start[0].IdempotencyKey
	nonce := launchNonce(attemptID, "g0")

	if _, started, err := store.BeginLaunching(context.Background(), attemptID, nonce, futureDeadline(t)); err != nil || !started {
		t.Fatalf("BeginLaunching started=%v err=%v", started, err)
	}
	// Same nonce is an idempotent readback; a competing nonce is fenced.
	again, started, err := store.BeginLaunching(context.Background(), attemptID, nonce, futureDeadline(t))
	if err != nil || started || again.Nonce != nonce {
		t.Fatalf("idempotent BeginLaunching = %#v started=%v err=%v", again, started, err)
	}
	if _, _, err := store.BeginLaunching(context.Background(), attemptID, "nonce:competing", futureDeadline(t)); !errors.Is(err, ErrFenced) {
		t.Fatalf("competing nonce err = %v, want ErrFenced", err)
	}
}

// TestControllerTickFenceFailurePreventsLaunch proves a BeginLaunching failure
// allocates no process: the runner is never invoked and no attempt is advanced.
func TestControllerTickFenceFailurePreventsLaunch(t *testing.T) {
	store := Store{Path: filepath.Join(t.TempDir(), "queue.json")}
	base := Snapshot{
		Schema: Schema, Generation: "g0",
		Pool: PoolSpec{ID: "pool", Min: 0, Desired: 1, Max: 1},
		Intents: []Intent{
			{ID: "docs", State: IntentQueued, Launch: LaunchSpec{Issue: 101, Lane: "docs"}},
		},
	}
	// A durable-store fence: the idempotency key for this generation already has
	// a terminal attempt, so BeginLaunching cannot resolve a unique reserved
	// attempt and must refuse to launch it.
	plan, err := Reconcile(base)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Start) != 1 {
		t.Fatalf("starts = %#v", plan.Start)
	}
	stale := base
	stale.Attempts = []Attempt{{ID: plan.Start[0].IdempotencyKey, IntentID: "docs", State: AttemptSucceeded}}
	if err := store.Save(stale); err != nil {
		t.Fatal(err)
	}

	runner := &launchObserverRunner{store: store}
	receipt, tickErr := Controller{Store: store, FakPath: "fak", Runner: runner}.Tick(context.Background())
	if tickErr == nil {
		t.Fatalf("Tick accepted an unfenceable reservation: %#v", receipt)
	}
	if !errors.Is(tickErr, ErrFenced) {
		t.Fatalf("Tick err = %v, want ErrFenced", tickErr)
	}
	if len(runner.calls) != 0 {
		t.Fatalf("runner invoked %#v despite a failed fence", runner.calls)
	}
	if len(receipt.Launches) != 0 {
		t.Fatalf("receipt launches = %#v, want none", receipt.Launches)
	}
	persisted, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, attempt := range persisted.Attempts {
		if attempt.State == AttemptLaunching {
			t.Fatalf("attempt advanced to launching despite a failed fence: %#v", attempt)
		}
	}
	if !strings.Contains(tickErr.Error(), "fence attempt") {
		t.Fatalf("err = %v, want a fence-attempt context", tickErr)
	}
}
