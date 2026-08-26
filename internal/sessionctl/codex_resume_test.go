package sessionctl

import (
	"errors"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
	"path/filepath"
	"testing"
)

func codexEvent(id string) harnesskit.Envelope {
	return harnesskit.Envelope{Version: harnesskit.ProtocolVersion, RunID: "logical-1", EventID: id, Type: harnesskit.EventMessageCompleted}
}

func TestCodexReconnectRestartOrderedWithoutDuplicateEffect(t *testing.T) {
	store := &FileCodexStateStore{Dir: t.TempDir()}
	first, err := OpenCodexSession(store, "logical-1", "codex-app-server/v1")
	if err != nil {
		t.Fatal(err)
	}
	run1, err := first.Begin(CodexNew, "browser-1")
	if err != nil {
		t.Fatal(err)
	}
	if run1.Epoch != 1 || run1.ThreadID != "" {
		t.Fatalf("first execution = %+v", run1)
	}
	if err := first.RecordThread(run1.Epoch, "opaque-codex-thread"); err != nil {
		t.Fatal(err)
	}
	a1, duplicate, err := first.Append(run1.Epoch, "delivery-turn-1", false, codexEvent("effect-1"))
	if err != nil || duplicate || a1.Cursor != 1 {
		t.Fatalf("append 1: address=%+v duplicate=%v err=%v", a1, duplicate, err)
	}

	// Browser two registers its live tail atomically with replay. Retrying the
	// delivery after reconnect must not execute the effect a second time.
	replay, live, cancel := first.Attach(0, 2)
	defer cancel()
	if len(replay) != 1 || replay[0].Address.Cursor != 1 {
		t.Fatalf("replay = %+v", replay)
	}
	got, duplicate, err := first.Append(run1.Epoch, "delivery-turn-1", false, codexEvent("effect-1"))
	if err != nil || !duplicate || got != a1 {
		t.Fatalf("duplicate: address=%+v duplicate=%v err=%v", got, duplicate, err)
	}
	select {
	case event := <-live:
		t.Fatalf("duplicate reached live tail: %+v", event)
	default:
	}
	first.Release("browser-1")

	// Simulate a killed adapter by reopening only its durable fak state.
	second, err := OpenCodexSession(&FileCodexStateStore{Dir: store.Dir}, "logical-1", "codex-app-server/v1")
	if err != nil {
		t.Fatal(err)
	}
	run2, err := second.Begin(CodexResume, "browser-2")
	if err != nil {
		t.Fatal(err)
	}
	if run2.SessionID != run1.SessionID || run2.ThreadID != "opaque-codex-thread" || run2.Epoch != 2 {
		t.Fatalf("resumed execution = %+v", run2)
	}
	a2, duplicate, err := second.Append(run2.Epoch, "delivery-turn-2", false, codexEvent("effect-2"))
	if err != nil || duplicate || a2.Cursor != 2 {
		t.Fatalf("append 2: address=%+v duplicate=%v err=%v", a2, duplicate, err)
	}
	history, _, stop := second.Attach(0, 1)
	stop()
	if len(history) != 2 || history[0].Event.EventID != "effect-1" || history[1].Event.EventID != "effect-2" {
		t.Fatalf("history = %+v", history)
	}
}

func TestCodexSessionRefusesStaleEpochAndConcurrentWriter(t *testing.T) {
	s, err := OpenCodexSession(NewMemoryCodexStateStore(), "logical-1", "v1")
	if err != nil {
		t.Fatal(err)
	}
	one, err := s.Begin(CodexNew, "owner")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Begin(CodexNew, "intruder"); !recoveryReason(err, CodexInputLeaseHeld) {
		t.Fatalf("concurrent Begin err = %v", err)
	}
	if err := s.RecordThread(one.Epoch, "thread-1"); err != nil {
		t.Fatal(err)
	}
	s.Release("owner")
	two, err := s.Begin(CodexResume, "owner-2")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Append(one.Epoch, "late", false, codexEvent("late")); !recoveryReason(err, CodexStaleEpoch) {
		t.Fatalf("stale append err = %v", err)
	}
	if _, _, err := s.Append(two.Epoch, "current", true, codexEvent("partial")); err != nil {
		t.Fatal(err)
	}
}

func TestCodexSessionTypesMissingAndIncompatibleResume(t *testing.T) {
	store := NewMemoryCodexStateStore()
	s, _ := OpenCodexSession(store, "logical-1", "v1")
	if _, err := s.Begin(CodexResume, "owner"); !recoveryReason(err, CodexThreadMissing) {
		t.Fatalf("missing err = %v", err)
	}
	first, err := s.Begin(CodexNew, "owner")
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RecordThread(first.Epoch, "thread-1"); err != nil {
		t.Fatal(err)
	}
	s.Release("owner")
	incompatible, _ := OpenCodexSession(store, "logical-1", "v2")
	if _, err := incompatible.Begin(CodexResume, "owner-2"); !recoveryReason(err, CodexThreadIncompatible) {
		t.Fatalf("incompatible err = %v", err)
	}
}

type failingCodexStore struct {
	state CodexSessionState
	fail  bool
}

func (s *failingCodexStore) Load(string) (CodexSessionState, error) {
	return cloneCodexState(s.state), nil
}
func (s *failingCodexStore) Save(v CodexSessionState) error {
	if s.fail {
		return errors.New("disk full")
	}
	s.state = cloneCodexState(v)
	return nil
}

func TestCodexSessionRollsBackCoordinateOnPersistenceFailure(t *testing.T) {
	store := &failingCodexStore{state: CodexSessionState{SessionID: "logical-1"}}
	s, err := OpenCodexSession(store, "logical-1", "v1")
	if err != nil {
		t.Fatal(err)
	}
	run, err := s.Begin(CodexNew, "owner")
	if err != nil {
		t.Fatal(err)
	}
	store.fail = true
	if err := s.RecordThread(run.Epoch, "thread-1"); err == nil {
		t.Fatal("expected persistence failure")
	}
	if got := s.State().Coordinates.ThreadID; got != "" {
		t.Fatalf("uncommitted coordinate leaked into memory: %q", got)
	}
}

func TestFileCodexStateStoreRejectsPathTraversal(t *testing.T) {
	store := &FileCodexStateStore{Dir: filepath.Join(t.TempDir(), "sessions")}
	if err := store.Save(CodexSessionState{SessionID: "../escape"}); err == nil {
		t.Fatal("expected unsafe id refusal")
	}
}

func recoveryReason(err error, want CodexRecoveryReason) bool {
	var recovery *CodexRecoveryError
	return errors.As(err, &recovery) && recovery.Reason == want
}
