package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentqueue"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func init() {
	if len(os.Args) > 1 && os.Args[1] == "agentqueue-wrapper" {
		if lifecycle, err := guardAgentQueueLifecycleFromEnv(); err != nil || lifecycle == nil {
			os.Exit(1)
		}
		os.Exit(0)
	}
	if len(os.Args) > 1 && os.Args[1] == "agentqueue-post-start-exit" {
		os.Exit(0)
	}
	if len(os.Args) > 2 && os.Args[1] == "agentqueue-start-marker" {
		if err := os.WriteFile(os.Args[2], []byte("started"), 0o600); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}
	if len(os.Args) > 1 && os.Args[1] == "dispatch" {
		statePath, hasState := agentQueueTestCommandFlag(os.Args[2:], "--agentqueue-state")
		attemptID, hasAttempt := agentQueueTestCommandFlag(os.Args[2:], "--agentqueue-attempt")
		nonce, hasNonce := agentQueueTestCommandFlag(os.Args[2:], "--agentqueue-nonce")
		if hasState != hasAttempt || hasState != hasNonce {
			os.Exit(2)
		}
		if hasState {
			snapshot, err := agentqueue.FileStore(statePath).Load()
			if err != nil {
				os.Exit(1)
			}
			matched := false
			for _, attempt := range snapshot.Attempts {
				if attempt.ID == attemptID && attempt.State == agentqueue.AttemptLaunching && attempt.Nonce == nonce {
					matched = true
					break
				}
			}
			if !matched {
				os.Exit(1)
			}
		}
		os.Exit(0)
	}
}

func agentQueueTestCommandFlag(args []string, name string) (string, bool) {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == name {
			return args[i+1], true
		}
	}
	return "", false
}

func TestAgentQueueRunOnceJSON(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "queue.json")
	store := agentqueue.FileStore(statePath)

	snapshot := agentqueue.Snapshot{
		Schema:     agentqueue.Schema,
		Generation: "g0",
		Pool:       agentqueue.PoolSpec{ID: "pool-alpha", Min: 1, Desired: 2, Max: 2},
		Intents: []agentqueue.Intent{
			{
				ID:     "intent-1",
				State:  agentqueue.IntentQueued,
				Launch: agentqueue.LaunchSpec{Issue: 8942, Lane: "agentqueue"},
			},
			{
				ID:     "intent-2",
				State:  agentqueue.IntentQueued,
				Launch: agentqueue.LaunchSpec{Issue: 8943, Lane: "agentqueue"},
			},
		},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runAgentQueue(&stdout, &stderr, []string{
		"run",
		"--state", statePath,
		"--fak", os.Args[0],
		"--interval", "50ms",
		"--once",
		"--json",
	})
	if code != 0 {
		t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
	}
	if stderr.Len() > 0 {
		t.Fatalf("unexpected stderr: %s", stderr.String())
	}

	var receipt agentqueue.TickReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt json: %v, raw: %s", err, stdout.String())
	}

	if receipt.Plan.PoolID != "pool-alpha" {
		t.Fatalf("expected pool ID 'pool-alpha', got %q", receipt.Plan.PoolID)
	}
	if receipt.Plan.Desired != 2 {
		t.Fatalf("expected desired 2, got %d", receipt.Plan.Desired)
	}
	if receipt.Plan.Observed != 0 {
		t.Fatalf("expected observed 0, got %d", receipt.Plan.Observed)
	}
	if len(receipt.Plan.Start) != 2 {
		t.Fatalf("expected 2 start actions, got %d", len(receipt.Plan.Start))
	}
	if receipt.Plan.Start[0].IntentID != "intent-1" {
		t.Fatalf("expected start[0] intent-1, got %q", receipt.Plan.Start[0].IntentID)
	}
	if receipt.Plan.Start[1].IntentID != "intent-2" {
		t.Fatalf("expected start[1] intent-2, got %q", receipt.Plan.Start[1].IntentID)
	}

	if len(receipt.Launches) != 2 {
		t.Fatalf("expected 2 launches, got %d", len(receipt.Launches))
	}
	if receipt.Launches[0].IntentID != "intent-1" {
		t.Fatalf("expected launch[0] intent-1, got %q", receipt.Launches[0].IntentID)
	}
	if receipt.Launches[0].IdempotencyKey != receipt.Plan.Start[0].IdempotencyKey {
		t.Fatalf("launch idempotency key mismatch: launch=%q plan=%q", receipt.Launches[0].IdempotencyKey, receipt.Plan.Start[0].IdempotencyKey)
	}
	if len(receipt.Launches[0].Command) < 3 || receipt.Launches[0].Command[1] != "dispatch" || receipt.Launches[0].Command[2] != "tick" {
		t.Fatalf("unexpected command structure: %#v", receipt.Launches[0].Command)
	}
	if receipt.Launches[1].IntentID != "intent-2" {
		t.Fatalf("expected launch[1] intent-2, got %q", receipt.Launches[1].IntentID)
	}

	// Verify state file persisted reservations
	updated, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if updated.Generation == "g0" {
		t.Fatalf("expected updated generation, got %q", updated.Generation)
	}
	if len(updated.Attempts) != 2 {
		t.Fatalf("expected 2 attempts recorded in store, got %d", len(updated.Attempts))
	}
}

func TestAgentQueueRunOnceText(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "queue.json")
	store := agentqueue.FileStore(statePath)

	snapshot := agentqueue.Snapshot{
		Schema:     agentqueue.Schema,
		Generation: "g0",
		Pool:       agentqueue.PoolSpec{ID: "pool-beta", Min: 0, Desired: 1, Max: 1},
		Intents: []agentqueue.Intent{
			{
				ID:     "intent-a",
				State:  agentqueue.IntentQueued,
				Launch: agentqueue.LaunchSpec{Issue: 100, Lane: "docs"},
			},
		},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
		"run",
		"--state", statePath,
		"--fak", os.Args[0],
		"--once",
	})
	if code != 0 {
		t.Fatalf("expected code 0, got %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "pool=pool-beta") || !strings.Contains(out, "launches=1") {
		t.Fatalf("unexpected output: %s", out)
	}
}

func TestAgentQueueRunCleanCancellation(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "queue.json")
	store := agentqueue.FileStore(statePath)

	snapshot := agentqueue.Snapshot{
		Schema:     agentqueue.Schema,
		Generation: "g0",
		Pool:       agentqueue.PoolSpec{ID: "pool-idle", Min: 0, Desired: 0, Max: 1},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}

	// 1. Cancel during ticker loop
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(25 * time.Millisecond)
		cancel()
	}()

	var stdout, stderr bytes.Buffer
	code := runAgentQueueContext(ctx, &stdout, &stderr, []string{
		"run",
		"--state", statePath,
		"--interval", "10ms",
	})
	if code != 0 {
		t.Fatalf("expected exit 0 on cancel, got %d, stderr: %s", code, stderr.String())
	}

	// 2. Pre-cancelled context stops cleanly immediately
	ctxPre, cancelPre := context.WithCancel(context.Background())
	cancelPre()
	stdout.Reset()
	stderr.Reset()
	codePre := runAgentQueueContext(ctxPre, &stdout, &stderr, []string{
		"run",
		"--state", statePath,
		"--interval", "10ms",
	})
	if codePre != 0 {
		t.Fatalf("expected exit 0 on pre-canceled context, got %d, stderr: %s", codePre, stderr.String())
	}
}

func TestAgentQueueUsageAndErrors(t *testing.T) {
	dir := t.TempDir()

	// Missing --state
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{"run"})
		if code != 2 {
			t.Fatalf("expected code 2, got %d", code)
		}
		if !strings.Contains(stderr.String(), "--state is required") {
			t.Fatalf("expected '--state is required', got: %s", stderr.String())
		}
	}

	// Empty --state
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{"run", "--state", ""})
		if code != 2 {
			t.Fatalf("expected code 2, got %d", code)
		}
		if !strings.Contains(stderr.String(), "--state is required") {
			t.Fatalf("expected '--state is required', got: %s", stderr.String())
		}
	}

	// Missing subcommand
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{})
		if code != 2 {
			t.Fatalf("expected code 2, got %d", code)
		}
		if !strings.Contains(stderr.String(), "usage: fak agent-queue") {
			t.Fatalf("expected usage message, got: %s", stderr.String())
		}
	}

	// Top-level --help
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{"--help"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d", code)
		}
		if !strings.Contains(stdout.String(), "usage: fak agent-queue") {
			t.Fatalf("expected usage message, got: %s", stdout.String())
		}
	}

	// run --help
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{"run", "--help"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d", code)
		}
	}

	// Unknown subcommand
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{"unknown"})
		if code != 2 {
			t.Fatalf("expected code 2, got %d", code)
		}
		if !strings.Contains(stderr.String(), "unknown subcommand") {
			t.Fatalf("expected 'unknown subcommand', got: %s", stderr.String())
		}
	}

	// Missing state file error
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
			"run",
			"--state", filepath.Join(dir, "nonexistent.json"),
			"--once",
		})
		if code != 1 {
			t.Fatalf("expected code 1, got %d", code)
		}
		if !strings.Contains(stderr.String(), "agent-queue run:") {
			t.Fatalf("expected 'agent-queue run:' error, got: %s", stderr.String())
		}
	}
}

func TestAgentQueueDispatchIdentityFlagsMustBePaired(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "queue.json")
	for _, tc := range []struct {
		name string
		args []string
	}{
		{name: "state without attempt", args: []string{"--agentqueue-state", statePath}},
		{name: "attempt without state", args: []string{"--agentqueue-attempt", "attempt-a"}},
		{name: "state and attempt without nonce", args: []string{"--agentqueue-state", statePath, "--agentqueue-attempt", "attempt-a"}},
		{name: "state and nonce without attempt", args: []string{"--agentqueue-state", statePath, "--agentqueue-nonce", "nonce-a"}},
		{name: "attempt and nonce without state", args: []string{"--agentqueue-attempt", "attempt-a", "--agentqueue-nonce", "nonce-a"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stderr bytes.Buffer
			_, ok, code := parseDispatchTickFlags(&stderr, tc.args)
			if ok || code != 2 {
				t.Fatalf("parse result ok=%v code=%d, want false/2", ok, code)
			}
			if !strings.Contains(stderr.String(), "must be provided together") {
				t.Fatalf("stderr = %q, want paired-identity diagnostic", stderr.String())
			}
		})
	}
}

func TestAgentQueueMissingOrStaleNonceRefusedBeforeStart(t *testing.T) {
	oldSpawner := dispatchIssueWorkerSpawner
	spawned := false
	dispatchIssueWorkerSpawner = func([]string, map[string]string, string, string, int, string, string, string, []string, dispatchtick.Account, *dispatchtick.Membership, string, string, float64) (dispatchSpawnResult, error) {
		spawned = true
		return dispatchSpawnResult{PID: 999}, nil
	}
	t.Cleanup(func() { dispatchIssueWorkerSpawner = oldSpawner })

	store, statePath, attemptID, nonce := newLaunchingDispatchHandoff(t, 42, "agentqueue")
	baseArgs := []string{
		"--agentqueue-state", statePath,
		"--agentqueue-attempt", attemptID,
		"--target-issue", "42",
		"--lane", "agentqueue",
		"--lease-id", attemptID,
		"--live",
	}

	var stdout, stderr bytes.Buffer
	if code := runDispatchTick(&stdout, &stderr, baseArgs); code != 2 {
		t.Fatalf("missing nonce code=%d stderr=%q, want parse refusal", code, stderr.String())
	}
	if spawned {
		t.Fatal("missing nonce reached OS Start seam")
	}
	loaded, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Attempts[0].State != agentqueue.AttemptLaunching || loaded.Attempts[0].Nonce != nonce {
		t.Fatalf("missing nonce changed launch: %+v", loaded.Attempts[0])
	}

	stdout.Reset()
	stderr.Reset()
	staleArgs := append([]string{}, baseArgs...)
	staleArgs = append(staleArgs, "--agentqueue-nonce", "stale-nonce")
	if code := runDispatchTick(&stdout, &stderr, staleArgs); code != 1 {
		t.Fatalf("stale nonce code=%d stderr=%q, want handoff refusal", code, stderr.String())
	}
	if spawned {
		t.Fatal("stale nonce reached OS Start seam")
	}
	loaded, err = store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Attempts[0].State != agentqueue.AttemptLaunching || loaded.Attempts[0].Nonce != nonce {
		t.Fatalf("stale nonce changed winning launch: %+v", loaded.Attempts[0])
	}
}

func TestAgentQueueLiveSpawnerConsumesPersistedNonce(t *testing.T) {
	store, statePath, attemptID, nonce := newLaunchingDispatchHandoff(t, 43, "agentqueue")
	env := envMap(os.Environ())
	env[agentQueueStatePathEnv] = statePath
	env[agentQueueAttemptIDEnv] = attemptID
	env[agentQueueNonceEnv] = nonce
	dir := t.TempDir()

	spawned, err := spawnDispatchIssueWorker(
		[]string{os.Args[0], "agentqueue-wrapper"}, env, dir, filepath.Join(dir, "runs"),
		43, "agentqueue", "test", attemptID, nil, dispatchtick.Account{}, nil, "", "", 0,
	)
	if err != nil {
		t.Fatalf("spawnDispatchIssueWorker: %v", err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		loaded, loadErr := store.Load()
		if loadErr != nil {
			t.Fatal(loadErr)
		}
		attempt := loaded.Attempts[0]
		if attempt.PID != 0 {
			if attempt.State != agentqueue.AttemptLaunching || attempt.Nonce != nonce || attempt.PID != spawned.PID || attempt.StartedAt.IsZero() {
				t.Fatalf("registered launch = %+v, spawned PID=%d", attempt, spawned.PID)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("wrapper did not consume persisted nonce: %+v", attempt)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestAgentQueueDefiniteStartFailureReturnsQueued(t *testing.T) {
	store, statePath, attemptID, nonce := newLaunchingDispatchHandoff(t, 44, "agentqueue")
	env := envMap(os.Environ())
	env[agentQueueStatePathEnv] = statePath
	env[agentQueueAttemptIDEnv] = attemptID
	env[agentQueueNonceEnv] = nonce
	dir := t.TempDir()

	_, err := spawnDispatchIssueWorker(
		[]string{filepath.Join(dir, "missing-worker-binary")}, env, dir, filepath.Join(dir, "runs"),
		44, "agentqueue", "test", attemptID, nil, dispatchtick.Account{}, nil, "", "", 0,
	)
	if err == nil {
		t.Fatal("missing worker binary unexpectedly started")
	}
	assertAgentQueueReturnedToQueued(t, store)
}

func TestAgentQueueExpiredLaunchDeadlineReturnsQueuedBeforeStart(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "queue.json")
	store := agentqueue.FileStore(statePath)
	const (
		attemptID = "attempt-expired"
		nonce     = "nonce-expired"
	)
	if err := store.Save(agentqueue.Snapshot{
		Schema: agentqueue.Schema, Generation: "g0",
		Pool: agentqueue.PoolSpec{ID: "pool-expired", Desired: 1, Max: 1},
		Intents: []agentqueue.Intent{{
			ID: "intent-expired", State: agentqueue.IntentQueued,
			Launch: agentqueue.LaunchSpec{Issue: 47, Lane: "agentqueue"},
		}},
		Attempts: []agentqueue.Attempt{{ID: attemptID, IntentID: "intent-expired", State: agentqueue.AttemptReserved}},
	}); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(100 * time.Millisecond)
	if _, started, err := store.BeginLaunching(context.Background(), attemptID, nonce, deadline); err != nil || !started {
		t.Fatalf("BeginLaunching started=%v err=%v", started, err)
	}
	if delay := time.Until(deadline) + 25*time.Millisecond; delay > 0 {
		time.Sleep(delay)
	}

	env := envMap(os.Environ())
	env[agentQueueStatePathEnv] = statePath
	env[agentQueueAttemptIDEnv] = attemptID
	env[agentQueueNonceEnv] = nonce
	dir := t.TempDir()
	marker := filepath.Join(dir, "started.marker")
	if _, err := spawnDispatchIssueWorker(
		[]string{os.Args[0], "agentqueue-start-marker", marker}, env, dir, filepath.Join(dir, "runs"),
		47, "agentqueue", "test", attemptID, nil, dispatchtick.Account{}, nil, "", "", 0,
	); err == nil {
		t.Fatal("expired launch unexpectedly reached OS Start")
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("expired launch started helper: marker stat error=%v", err)
	}
	assertAgentQueueReturnedToQueued(t, store)
}

func TestAgentQueueNormalPreSpawnRefusalReturnsQueued(t *testing.T) {
	store, statePath, attemptID, nonce := newLaunchingDispatchHandoff(t, 45, "agentqueue")
	var stdout, stderr bytes.Buffer
	code := runDispatchTick(&stdout, &stderr, []string{
		"--agentqueue-state", statePath,
		"--agentqueue-attempt", attemptID,
		"--agentqueue-nonce", nonce,
		"--target-issue", "45",
		"--lane", "agentqueue",
		"--lease-id", attemptID,
	})
	if code != 1 || !strings.Contains(stderr.String(), "requires a live guarded CLI backend") {
		t.Fatalf("pre-spawn refusal code=%d stderr=%q", code, stderr.String())
	}
	assertAgentQueueReturnedToQueued(t, store)
}

func TestAgentQueueAmbiguousPostStartRemainsHeld(t *testing.T) {
	store, statePath, attemptID, nonce := newLaunchingDispatchHandoff(t, 46, "agentqueue")
	env := envMap(os.Environ())
	env[agentQueueStatePathEnv] = statePath
	env[agentQueueAttemptIDEnv] = attemptID
	env[agentQueueNonceEnv] = nonce
	dir := t.TempDir()

	if _, err := spawnDispatchIssueWorker(
		[]string{os.Args[0], "agentqueue-post-start-exit"}, env, dir, filepath.Join(dir, "runs"),
		46, "agentqueue", "test", attemptID, nil, dispatchtick.Account{}, nil, "", "", 0,
	); err != nil {
		t.Fatalf("spawnDispatchIssueWorker: %v", err)
	}
	reconciliation, updated, err := store.ReconcileRestart(context.Background(), func(int) bool { return false }, agentqueue.RestartOptions{})
	if err != nil {
		t.Fatalf("ReconcileRestart: %v", err)
	}
	if len(reconciliation.Held) != 1 || reconciliation.Held[0].IntentID != "intent-dispatch" {
		t.Fatalf("reconciliation = %+v, want ambiguous launch held", reconciliation)
	}
	if updated.Attempts[0].State != agentqueue.AttemptLaunching || updated.Attempts[0].Nonce != nonce || updated.Intents[0].State != agentqueue.IntentQueued {
		t.Fatalf("ambiguous launch changed: attempt=%+v intent=%+v", updated.Attempts[0], updated.Intents[0])
	}
}

func newLaunchingDispatchHandoff(t *testing.T, issue int, lane string) (agentqueue.Store, string, string, string) {
	t.Helper()
	statePath := filepath.Join(t.TempDir(), "queue.json")
	store := agentqueue.FileStore(statePath)
	attemptID := fmt.Sprintf("attempt-%d", issue)
	nonce := fmt.Sprintf("nonce-%d", issue)
	if err := store.Save(agentqueue.Snapshot{
		Schema: agentqueue.Schema, Generation: "g0",
		Pool: agentqueue.PoolSpec{ID: "pool-dispatch", Desired: 1, Max: 1},
		Intents: []agentqueue.Intent{{
			ID: "intent-dispatch", State: agentqueue.IntentQueued,
			Launch: agentqueue.LaunchSpec{Issue: issue, Lane: lane},
		}},
		Attempts: []agentqueue.Attempt{{ID: attemptID, IntentID: "intent-dispatch", State: agentqueue.AttemptReserved}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, started, err := store.BeginLaunching(context.Background(), attemptID, nonce, time.Now().Add(time.Minute)); err != nil || !started {
		t.Fatalf("BeginLaunching started=%v err=%v", started, err)
	}
	return store, statePath, attemptID, nonce
}

func assertAgentQueueReturnedToQueued(t *testing.T, store agentqueue.Store) {
	t.Helper()
	loaded, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Attempts[0].State != agentqueue.AttemptFailed || loaded.Intents[0].State != agentqueue.IntentQueued {
		t.Fatalf("returned launch = attempt %q intent %q, want failed/queued", loaded.Attempts[0].State, loaded.Intents[0].State)
	}
}

func TestAgentQueueEnvScrubIsCaseInsensitive(t *testing.T) {
	env := map[string]string{
		agentQueueStatePathEnv:      "canonical-state",
		"fak_agentqueue_state_path": "mixed-case-state",
		"Fak_AgentQueue_Attempt_Id": "mixed-case-attempt",
		"FAK_agentqueue_NONCE":      "mixed-case-nonce",
		"PATH":                      "preserved-path",
		"Worker_Mode":               "preserved-mode",
	}

	stripAgentQueueEnv(env)
	for name, value := range env {
		if strings.EqualFold(name, agentQueueStatePathEnv) || strings.EqualFold(name, agentQueueAttemptIDEnv) || strings.EqualFold(name, agentQueueNonceEnv) {
			t.Errorf("inherited queue identity survived scrub: %s=%q", name, value)
		}
	}
	if len(env) != 2 || env["PATH"] != "preserved-path" || env["Worker_Mode"] != "preserved-mode" {
		t.Fatalf("scrub did not preserve the unrelated environment exactly: %#v", env)
	}
}

func TestAgentQueueGuardWrapperRegistersAndScrubsIdentity(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "queue.json")
	store := agentqueue.FileStore(statePath)
	const (
		attemptID = "attempt-wrapper"
		nonce     = "nonce-wrapper"
	)
	snapshot := agentqueue.Snapshot{
		Schema:     agentqueue.Schema,
		Generation: "g0",
		Pool:       agentqueue.PoolSpec{ID: "pool-wrapper", Desired: 1, Max: 1},
		Intents: []agentqueue.Intent{{
			ID: "intent-wrapper", State: agentqueue.IntentQueued,
		}},
		Attempts: []agentqueue.Attempt{{
			ID: attemptID, IntentID: "intent-wrapper", State: agentqueue.AttemptReserved,
		}},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}
	if _, started, err := store.BeginLaunching(context.Background(), attemptID, nonce, time.Now().Add(time.Minute)); err != nil || !started {
		t.Fatalf("BeginLaunching started=%v err=%v", started, err)
	}

	t.Setenv(guardAgentQueueStatePathEnv, statePath)
	t.Setenv(guardAgentQueueAttemptIDEnv, attemptID)
	t.Setenv(guardAgentQueueNonceEnv, nonce)
	lifecycle, err := guardAgentQueueLifecycleFromEnv()
	if err != nil {
		t.Fatalf("guardAgentQueueLifecycleFromEnv: %v", err)
	}
	if lifecycle == nil || lifecycle.pid != os.Getpid() || lifecycle.startedAt.IsZero() {
		t.Fatalf("lifecycle = %+v, want current wrapper identity", lifecycle)
	}
	for _, name := range []string{guardAgentQueueStatePathEnv, guardAgentQueueAttemptIDEnv, guardAgentQueueNonceEnv} {
		if value := os.Getenv(name); value != "" {
			t.Errorf("%s survived wrapper registration: %q", name, value)
		}
	}
	loaded, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	got := loaded.Attempts[0]
	if got.State != agentqueue.AttemptLaunching || got.Nonce != nonce || got.PID != os.Getpid() || !got.StartedAt.Equal(lifecycle.startedAt) {
		t.Fatalf("registered attempt = %+v, want launching wrapper pid=%d start=%s", got, os.Getpid(), lifecycle.startedAt)
	}
	if err := lifecycle.abortLaunching(); err != nil {
		t.Fatalf("abortLaunching: %v", err)
	}
	aborted, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if aborted.Attempts[0].State != agentqueue.AttemptFailed || aborted.Intents[0].State != agentqueue.IntentQueued {
		t.Fatalf("aborted state = attempt %q intent %q, want failed/queued", aborted.Attempts[0].State, aborted.Intents[0].State)
	}
}

func TestAgentQueueGuardCompletionRequiresJoinedChildAndPersistsHold(t *testing.T) {
	t.Run("joined failure persists typed hold", func(t *testing.T) {
		lifecycle, store := newRunningAgentQueueGuardLifecycle(t)
		lifecycle.hold(guardAgentQueueHoldWitness)
		if err := lifecycle.complete(false, true); err != nil {
			t.Fatalf("complete joined held child: %v", err)
		}
		loaded, err := store.Load()
		if err != nil {
			t.Fatal(err)
		}
		if loaded.Attempts[0].State != agentqueue.AttemptFailed || loaded.Intents[0].State != agentqueue.IntentHeld || loaded.Intents[0].HoldReason != guardAgentQueueHoldWitness {
			t.Fatalf("held terminal state = attempt %q intent %q reason %q", loaded.Attempts[0].State, loaded.Intents[0].State, loaded.Intents[0].HoldReason)
		}
	})

	t.Run("uncertain child exit remains running", func(t *testing.T) {
		lifecycle, store := newRunningAgentQueueGuardLifecycle(t)
		lifecycle.hold(guardAgentQueueHoldWitness)
		if err := lifecycle.complete(false, false); err != nil {
			t.Fatalf("complete unjoined child: %v", err)
		}
		loaded, err := store.Load()
		if err != nil {
			t.Fatal(err)
		}
		if loaded.Attempts[0].State != agentqueue.AttemptRunning || loaded.Intents[0].State != agentqueue.IntentRunning || loaded.Intents[0].HoldReason != "" {
			t.Fatalf("uncertain terminal state = attempt %q intent %q reason %q, want running/running/empty", loaded.Attempts[0].State, loaded.Intents[0].State, loaded.Intents[0].HoldReason)
		}
	})
}

func newRunningAgentQueueGuardLifecycle(t *testing.T) (*guardAgentQueueLifecycle, agentqueue.Store) {
	t.Helper()
	statePath := filepath.Join(t.TempDir(), "queue.json")
	store := agentqueue.FileStore(statePath)
	const (
		attemptID = "attempt-running-wrapper"
		nonce     = "nonce-running-wrapper"
	)
	if err := store.Save(agentqueue.Snapshot{
		Schema: agentqueue.Schema, Generation: "g0",
		Pool:     agentqueue.PoolSpec{ID: "pool-running-wrapper", Desired: 1, Max: 1},
		Intents:  []agentqueue.Intent{{ID: "intent-running-wrapper", State: agentqueue.IntentQueued}},
		Attempts: []agentqueue.Attempt{{ID: attemptID, IntentID: "intent-running-wrapper", State: agentqueue.AttemptReserved}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, started, err := store.BeginLaunching(context.Background(), attemptID, nonce, time.Now().Add(time.Minute)); err != nil || !started {
		t.Fatalf("BeginLaunching started=%v err=%v", started, err)
	}
	t.Setenv(guardAgentQueueStatePathEnv, statePath)
	t.Setenv(guardAgentQueueAttemptIDEnv, attemptID)
	t.Setenv(guardAgentQueueNonceEnv, nonce)
	lifecycle, err := guardAgentQueueLifecycleFromEnv()
	if err != nil {
		t.Fatalf("guardAgentQueueLifecycleFromEnv: %v", err)
	}
	if err := lifecycle.markRunning(); err != nil {
		t.Fatalf("markRunning: %v", err)
	}
	return lifecycle, store
}

func TestAgentQueueBrokerCannotRewriteGuardedCommand(t *testing.T) {
	root := initRegionTestRepo(t)
	t.Setenv("FAK_LEASE_OWNER", "agentqueue-broker-rewrite-test")
	t.Setenv("FLEET_DOGFOOD_GUARD", "1")
	t.Setenv("FLEET_WORKER_WORKTREE", "0")

	oldBroker, oldSpawner := launchSpawnBroker, dispatchIssueWorkerSpawner
	spawned := false
	launchSpawnBroker = func(attempt launchBrokerAttempt) launchBrokerGrant {
		grant := allowLaunchBrokerGrant(attempt, "unit-test-rewrite")
		grant.Argv = []string{"codex", "exec", "bypassed-guard"}
		return grant
	}
	dispatchIssueWorkerSpawner = func([]string, map[string]string, string, string, int, string, string, string, []string, dispatchtick.Account, *dispatchtick.Membership, string, string, float64) (dispatchSpawnResult, error) {
		spawned = true
		return dispatchSpawnResult{PID: 999}, nil
	}
	t.Cleanup(func() {
		launchSpawnBroker = oldBroker
		dispatchIssueWorkerSpawner = oldSpawner
	})

	statePath := filepath.Join(root, "queue.json")
	got, err := dispatchTickLiveSpawn(
		root, filepath.Join(root, dispatchtick.RunsDirName),
		dispatchTickOptions{
			Backend: "codex", Live: true, WorkerTimeoutS: 1200,
			AgentQueueState: statePath, AgentQueueAttempt: "attempt-a",
		},
		dispatchLanePick{Lane: "gateway", Tree: []string{"internal/gateway/**"}},
		"resolve-gateway", dispatchtick.Account{Tag: "seat-a"}, dispatchtick.WorkerLaunch{},
		dispatchWorkerPreflightRequest{}, nil, 42,
		map[string]any{"prompt": "resolve #42"}, map[string]any{},
		func(payload map[string]any) map[string]any { return payload },
	)
	if err != nil {
		t.Fatalf("dispatchTickLiveSpawn: %v", err)
	}
	if spawned {
		t.Fatal("queue worker spawned after broker stripped the guard")
	}
	if got["action"] != "agentqueue_guard_rewritten" || got["ok"] != false {
		t.Fatalf("rewrite refusal = action %v ok %v, want agentqueue_guard_rewritten/false", got["action"], got["ok"])
	}
}

func TestAgentQueueReconcileRestartSubcommand(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "queue.json")
	store := agentqueue.FileStore(statePath)

	currentPID := os.Getpid()
	deadPID := 999999

	snapshot := agentqueue.Snapshot{
		Schema:     agentqueue.Schema,
		Generation: "g0",
		Pool:       agentqueue.PoolSpec{ID: "pool-restart", Min: 1, Desired: 2, Max: 2},
		Intents: []agentqueue.Intent{
			{
				ID:     "intent-alive",
				State:  agentqueue.IntentRunning,
				Launch: agentqueue.LaunchSpec{Issue: 1, Lane: "agentqueue"},
			},
			{
				ID:     "intent-dead",
				State:  agentqueue.IntentRunning,
				Launch: agentqueue.LaunchSpec{Issue: 2, Lane: "agentqueue"},
			},
		},
		Attempts: []agentqueue.Attempt{
			{
				ID:       "att-alive",
				IntentID: "intent-alive",
				State:    agentqueue.AttemptRunning,
				PID:      currentPID,
			},
			{
				ID:       "att-dead",
				IntentID: "intent-dead",
				State:    agentqueue.AttemptRunning,
				PID:      deadPID,
			},
		},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}

	// 1. Text output without --apply (dry run)
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
			"reconcile-restart",
			"--state", statePath,
		})
		if code != 0 {
			t.Fatalf("code=%d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		if !strings.Contains(out, "adopted=1") || !strings.Contains(out, "replaced=1") {
			t.Fatalf("unexpected stdout: %s", out)
		}
		// Since --apply was not specified, file should still be at g0
		loaded, err := store.Load()
		if err != nil {
			t.Fatal(err)
		}
		if loaded.Generation != "g0" {
			t.Fatalf("expected generation g0, got %s", loaded.Generation)
		}
	}

	// 2. JSON output with --apply
	{
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
			"reconcile-restart",
			"--state", statePath,
			"--json",
			"--apply",
		})
		if code != 0 {
			t.Fatalf("code=%d, stderr: %s", code, stderr.String())
		}
		var rec agentqueue.RestartReconciliation
		if err := json.Unmarshal(stdout.Bytes(), &rec); err != nil {
			t.Fatalf("decode rec json: %v", err)
		}
		if len(rec.Adopted) != 1 || rec.Adopted[0].IntentID != "intent-alive" {
			t.Fatalf("unexpected adopted: %+v", rec.Adopted)
		}
		if len(rec.Replaced) != 1 || rec.Replaced[0].IntentID != "intent-dead" {
			t.Fatalf("unexpected replaced: %+v", rec.Replaced)
		}

		// Verify file was updated on disk
		loaded, err := store.Load()
		if err != nil {
			t.Fatal(err)
		}
		if loaded.Generation == "g0" {
			t.Fatalf("expected updated generation, got g0")
		}
		if loaded.Intents[1].State != agentqueue.IntentQueued {
			t.Fatalf("dead intent state = %v, want queued", loaded.Intents[1].State)
		}
	}
}

func TestAgentQueueRunReconcileRestartFlag(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "queue.json")
	store := agentqueue.FileStore(statePath)

	currentPID := os.Getpid()
	deadPID := 999999

	snapshot := agentqueue.Snapshot{
		Schema:     agentqueue.Schema,
		Generation: "g0",
		Pool:       agentqueue.PoolSpec{ID: "pool-flag", Min: 1, Desired: 2, Max: 2},
		Intents: []agentqueue.Intent{
			{
				ID:     "intent-alive",
				State:  agentqueue.IntentRunning,
				Launch: agentqueue.LaunchSpec{Issue: 1, Lane: "agentqueue"},
			},
			{
				ID:     "intent-dead",
				State:  agentqueue.IntentRunning,
				Launch: agentqueue.LaunchSpec{Issue: 2, Lane: "agentqueue"},
			},
		},
		Attempts: []agentqueue.Attempt{
			{
				ID:       "att-alive",
				IntentID: "intent-alive",
				State:    agentqueue.AttemptRunning,
				PID:      currentPID,
			},
			{
				ID:       "att-dead",
				IntentID: "intent-dead",
				State:    agentqueue.AttemptRunning,
				PID:      deadPID,
			},
		},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
		"run",
		"--state", statePath,
		"--fak", os.Args[0],
		"--reconcile-restart",
		"--once",
		"--json",
	})
	if code != 0 {
		t.Fatalf("code=%d, stderr: %s", code, stderr.String())
	}

	var receipt agentqueue.TickReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt json: %v, raw: %s", err, stdout.String())
	}

	if receipt.Restart == nil {
		t.Fatal("expected receipt.Restart to be populated")
	}
	if len(receipt.Restart.Adopted) != 1 || receipt.Restart.Adopted[0].IntentID != "intent-alive" {
		t.Fatalf("unexpected adopted: %+v", receipt.Restart.Adopted)
	}
	if len(receipt.Restart.Replaced) != 1 || receipt.Restart.Replaced[0].IntentID != "intent-dead" {
		t.Fatalf("unexpected replaced: %+v", receipt.Restart.Replaced)
	}
	// Only replaced intent was launched!
	if len(receipt.Launches) != 1 || receipt.Launches[0].IntentID != "intent-dead" {
		t.Fatalf("launches = %+v, want only intent-dead", receipt.Launches)
	}
}

func TestAgentQueueEmitReconciler(t *testing.T) {
	statePath := "/tmp/queue.json"

	t.Run("launchd", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
			"emit-reconciler",
			"--target", "launchd",
			"--state", statePath,
			"--interval", "2m",
			"--timeout", "45s",
			"--fak", "fak",
		})
		if code != 0 {
			t.Fatalf("code=%d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		for _, want := range []string{
			"<string>fak</string>",
			"<string>cron</string>",
			"<string>run</string>",
			"<string>--job</string>",
			"<string>agentqueue-reconcile</string>",
			"<string>agent-queue</string>",
			"<string>run</string>",
			"<string>--state</string>",
			"<string>/tmp/queue.json</string>",
			"<string>--once</string>",
			"<key>StartInterval</key>",
			"<integer>120</integer>",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("launchd output missing %q:\n%s", want, out)
			}
		}
	})

	t.Run("systemd", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
			"emit-reconciler",
			"--target", "systemd",
			"--state", statePath,
			"--interval", "1m",
			"--timeout", "30s",
			"--fak", "fak",
		})
		if code != 0 {
			t.Fatalf("code=%d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		for _, want := range []string{
			"# === fak-cron-agentqueue-reconcile.service ===",
			"# === fak-cron-agentqueue-reconcile.timer ===",
			"Type=oneshot",
			"ExecStart=fak cron run --job agentqueue-reconcile",
			"agent-queue run --state /tmp/queue.json --once",
			"OnUnitActiveSec=60s",
			"OnBootSec=60s",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("systemd output missing %q:\n%s", want, out)
			}
		}
	})

	t.Run("taskscheduler", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
			"emit-reconciler",
			"--target", "taskscheduler",
			"--state", statePath,
			"--interval", "5m",
			"--fak", "fak",
		})
		if code != 0 {
			t.Fatalf("code=%d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		for _, want := range []string{
			"Register-ScheduledTask",
			"fak-cron-agentqueue-reconcile",
			"cron run --job agentqueue-reconcile",
			"agent-queue run --state /tmp/queue.json --once",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("taskscheduler output missing %q:\n%s", want, out)
			}
		}
	})

	t.Run("missing target or state fails with exit 2", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
			"emit-reconciler",
			"--state", statePath,
		})
		if code != 2 {
			t.Errorf("missing target code = %d, want 2", code)
		}

		stdout.Reset()
		stderr.Reset()
		code = runAgentQueueContext(context.Background(), &stdout, &stderr, []string{
			"emit-reconciler",
			"--target", "systemd",
		})
		if code != 2 {
			t.Errorf("missing state code = %d, want 2", code)
		}
	})
}
