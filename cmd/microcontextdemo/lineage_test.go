package main

import (
	"context"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

// fak-test:runtime fast est=100ms
func TestRunRegistersEveryLogicalContextUnderStartingGoal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "child-registrations.jsonl")
	store := sessionregistry.Store{Path: path}
	now := time.Date(2026, 8, 13, 19, 0, 0, 0, time.UTC)
	root, err := sessionregistry.New(sessionregistry.NewInput{
		RegistrationID: "top-goal", RootIssue: "6583", TaskID: "trace-starting-goal",
		LaunchKind: "guard", Runtime: "codex", SessionID: "top-session", Now: now,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Register(root); err != nil {
		t.Fatal(err)
	}
	lineage := &microagent.Lineage{
		Store: store, ParentRegistrationID: root.RegistrationID, ParentAttemptID: root.AttemptID,
		RootRegistrationID: root.RootRegistrationID, RootIssue: root.RootIssue, TaskID: root.TaskID,
		Now: func() time.Time { return now.Add(time.Second) },
	}

	cfg := config{Contexts: 3, Workers: 2, Delay: 0, Selfcheck: true, Lineage: lineage}
	entered := make(chan struct{}, cfg.Workers)
	released := make(chan struct{})
	release := sync.OnceFunc(func() { close(released) })
	var arrivals atomic.Int64
	beforeComplete := func(ctx context.Context) error {
		if arrivals.Add(1) <= int64(cfg.Workers) {
			entered <- struct{}{}
		}
		select {
		case <-released:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	// Keep this witness compilable against source without the seam: the failure
	// must be run returning while the rendezvous is held, not an undefined field.
	if field := reflect.ValueOf(&cfg).Elem().FieldByName("SyntheticBeforeComplete"); field.IsValid() {
		field.Set(reflect.ValueOf(beforeComplete))
	}
	// The deadline only bounds a broken rendezvous; entry signals prove overlap.
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	var result report
	var runErr error
	done := make(chan struct{})
	go func() {
		defer close(done)
		result, runErr = run(ctx, cfg)
	}()
	defer func() {
		cancel()
		release()
		<-done
	}()
	for i := 0; i < cfg.Workers; i++ {
		select {
		case <-entered:
		case <-done:
			t.Fatalf("run returned while synthetic completion rendezvous was held: entries=%d, want %d; report=%+v; err=%v", arrivals.Load(), cfg.Workers, result, runErr)
		case <-ctx.Done():
			t.Fatalf("waiting for synthetic completion entries: %v", ctx.Err())
		}
	}
	select {
	case <-done:
		t.Fatalf("run returned before releasing synthetic completion rendezvous: report=%+v; err=%v", result, runErr)
	default:
	}
	release()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatalf("waiting for released synthetic completions: %v", ctx.Err())
	}
	if got := arrivals.Load(); got != int64(cfg.Contexts) {
		t.Fatalf("synthetic completion entries = %d, want %d", got, cfg.Contexts)
	}
	if runErr != nil {
		t.Fatal(runErr)
	}
	if result.Verdict != "PASS" || result.Completed != 3 || result.Failed != 0 ||
		result.TurnCount != 3 || result.PhysicalWorkers != 2 || result.PeakInFlight != 2 {
		t.Fatalf("report = %+v", result)
	}
	rows, err := store.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 4 {
		t.Fatalf("registrations = %d, want root + 3 contexts", len(rows))
	}
	seenSessions := map[string]bool{}
	for _, row := range rows[1:] {
		if row.ParentRegistrationID != root.RegistrationID || row.RootRegistrationID != root.RegistrationID {
			t.Fatalf("context escaped root: %+v", row)
		}
		if row.RootIssue != "6583" || row.TaskID != "trace-starting-goal" {
			t.Fatalf("context lost goal labels: %+v", row)
		}
		if row.LaunchKind != "in_process_microagent" || row.State != sessionregistry.StateCompleted {
			t.Fatalf("context lifecycle = %+v", row)
		}
		if row.Identity.SessionID == "" || seenSessions[row.Identity.SessionID] {
			t.Fatalf("non-unique context session %q", row.Identity.SessionID)
		}
		seenSessions[row.Identity.SessionID] = true
	}
}
