package agentqueue

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func persistedFixture() Snapshot {
	return Snapshot{
		Schema: Schema, Generation: "generation-7",
		Pool:     PoolSpec{ID: "build", Min: 1, Desired: 2, Max: 3},
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
