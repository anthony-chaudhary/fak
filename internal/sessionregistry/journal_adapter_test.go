package sessionregistry

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

func TestDefaultStoreRoundTripsThroughJournal(t *testing.T) {
	journal := t.TempDir() + "/events.jsonl"
	t.Setenv("FAK_SESSION_REGISTRY", "")
	t.Setenv(sessionjournal.EnvPath, journal)
	now := time.Date(2026, 8, 18, 1, 2, 3, 0, time.UTC)
	rec, err := New(NewInput{RegistrationID: "root", AttemptID: "attempt", LaunchKind: "guard-child", Runtime: "codex", SessionID: "session", ThreadID: "thread", HostID: "host", Now: now})
	if err != nil {
		t.Fatal(err)
	}
	store := Store{Path: DefaultPath()}
	if err := store.Register(rec); err != nil {
		t.Fatal(err)
	}
	active, err := store.Start("root", 42, now.Add(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if active.State != StateActive || active.Identity.PID != 42 {
		t.Fatalf("active=%+v", active)
	}
	terminal, err := store.Terminal("root", StateCompleted, "done", "commit:abc", now.Add(2*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if terminal.State != StateCompleted {
		t.Fatalf("terminal=%+v", terminal)
	}
	rows, err := store.ReadAll()
	if err != nil || len(rows) != 1 {
		t.Fatalf("ReadAll=(%+v,%v)", rows, err)
	}
	got := rows[0]
	if got.RegistrationID != rec.RegistrationID || got.Identity.SessionID != "session" || got.Identity.PID != 42 || got.State != StateCompleted || got.WitnessRef != "commit:abc" {
		t.Fatalf("journal projection=%+v", got)
	}
	if events := sessionjournal.LoadFile(journal); len(events) != 3 {
		t.Fatalf("journal events=%d want 3", len(events))
	}
}

func TestExplicitLegacyRegistryPathStaysCompatible(t *testing.T) {
	path := t.TempDir() + "/legacy.jsonl"
	t.Setenv("FAK_SESSION_REGISTRY", path)
	rec, err := New(NewInput{RegistrationID: "legacy", AttemptID: "attempt", LaunchKind: "worker", Runtime: "codex", Now: time.Now()})
	if err != nil {
		t.Fatal(err)
	}
	store := Store{Path: path}
	if err := store.Register(rec); err != nil {
		t.Fatal(err)
	}
	if len(sessionjournal.LoadFile(path)) != 0 {
		t.Fatal("legacy registry must not be parsed as lifecycle journal")
	}
	rows, err := store.ReadAll()
	if err != nil || len(rows) != 1 || rows[0].RegistrationID != "legacy" {
		t.Fatalf("legacy ReadAll=(%+v,%v)", rows, err)
	}
}
