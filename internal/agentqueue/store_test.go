package agentqueue

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

func persistedFixture() Snapshot {
	return Snapshot{
		Schema: Schema, Generation: "generation-7",
		Pool: PoolSpec{ID: "build", Min: 1, Desired: 2, Max: 3},
		//enumlint:exempt This persistence fixture deliberately pairs one active and one queued intent; terminal and held states do not change the round-trip contract under test.
		Intents:  []Intent{{ID: "a", State: IntentRunning}, {ID: "b", State: IntentQueued}},
		Attempts: []Attempt{{ID: "attempt-a", IntentID: "a", State: AttemptReserved}},
	}
}

func TestStoreSaveLoadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "queue.json")
	store := Store{Path: path}
	want := persistedFixture()
	if err := store.Save(want); err != nil {
		t.Fatal(err)
	}
	got, err := (Store{Path: path}).Load()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reopened snapshot = %#v, want %#v", got, want)
	}
}

func TestStoreReservationSurvivesReopenWithoutDuplicateStart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "queue.json")
	store := Store{Path: path}
	snap := persistedFixture()
	snap.Pool.Desired = 1
	if err := store.Save(snap); err != nil {
		t.Fatal(err)
	}
	reopened, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := Reconcile(reopened)
	if err != nil {
		t.Fatal(err)
	}
	if len(receipt.Start) != 0 {
		t.Fatalf("reserved intent duplicated after reopen: %#v", receipt.Start)
	}
}

func TestStoreAbandonedTempDoesNotCorruptCommittedSnapshot(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "queue.json")
	store := Store{Path: path}
	want := persistedFixture()
	if err := store.Save(want); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".agentqueue-interrupted.tmp"), []byte(`{"schema":`), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := (Store{Path: path}).Load()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("committed snapshot changed: %#v", got)
	}
}

func TestStoreRejectsMalformedUnknownAndUnsupportedSnapshots(t *testing.T) {
	for name, body := range map[string]string{
		"malformed":          `{`,
		"unknown field":      `{"schema":"` + Schema + `","generation":"g","pool":{"id":"p","min":0,"desired":0,"max":0},"mystery":true}`,
		"unsupported schema": `{"schema":"fak.agentqueue.snapshot.v2","generation":"g","pool":{"id":"p","min":0,"desired":0,"max":0}}`,
		"trailing value":     `{"schema":"` + Schema + `","generation":"g","pool":{"id":"p","min":0,"desired":0,"max":0}} {}`,
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "queue.json")
			if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := (Store{Path: path}).Load(); err == nil {
				t.Fatal("Load accepted invalid snapshot")
			}
		})
	}
}

func TestStoreSaveDoesNotMutateCaller(t *testing.T) {
	snap := persistedFixture()
	snap.Schema = ""
	if err := (Store{Path: filepath.Join(t.TempDir(), "queue.json")}).Save(snap); err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(snap.Schema) != "" {
		t.Fatalf("Save mutated caller schema to %q", snap.Schema)
	}
}

func TestStoreReserveConcurrentAtMaxMinusOneAcceptsExactlyOne(t *testing.T) {
	path := filepath.Join(t.TempDir(), "queue.json")
	store := Store{Path: path}
	snapshot := Snapshot{
		Schema: Schema, Generation: "generation-before",
		Pool: PoolSpec{ID: "build", Min: 0, Desired: 2, Max: 2},
		Intents: []Intent{
			{ID: "active", State: IntentRunning},
			{ID: "candidate", State: IntentQueued},
		},
		Attempts: []Attempt{{ID: "active-1", IntentID: "active", State: AttemptRunning}},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}

	start := make(chan struct{})
	type result struct {
		receipt Receipt
		err     error
	}
	results := make(chan result, 2)
	for range 2 {
		go func() {
			<-start
			receipt, _, err := store.Reserve(context.Background(), "generation-before")
			results <- result{receipt: receipt, err: err}
		}()
	}
	close(start)

	accepted, conflicted := 0, 0
	for range 2 {
		result := <-results
		switch {
		case result.err == nil:
			accepted++
			if len(result.receipt.Start) != 1 || result.receipt.Start[0].IntentID != "candidate" {
				t.Fatalf("accepted reservation = %#v", result.receipt.Start)
			}
		case errors.Is(result.err, ErrGenerationConflict):
			conflicted++
		default:
			t.Fatalf("Reserve error = %v", result.err)
		}
	}
	if accepted != 1 || conflicted != 1 {
		t.Fatalf("accepted=%d conflicted=%d, want 1 each", accepted, conflicted)
	}
	final, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(final.Attempts) != 2 || final.Attempts[1].IntentID != "candidate" || final.Attempts[1].State != AttemptReserved {
		t.Fatalf("final attempts = %#v", final.Attempts)
	}
	if final.Generation == "generation-before" {
		t.Fatal("successful reservation did not advance generation")
	}
}

func TestStoreReserveHonorsCanceledLockWait(t *testing.T) {
	path := filepath.Join(t.TempDir(), "queue.json")
	store := Store{Path: path}
	if err := store.Save(Snapshot{Schema: Schema, Generation: "g", Pool: PoolSpec{ID: "p", Max: 1}}); err != nil {
		t.Fatal(err)
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		t.Fatal(err)
	}
	defer flock.Unlock(lock)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := store.Reserve(ctx, "g"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Reserve error = %v, want context.Canceled", err)
	}
}
