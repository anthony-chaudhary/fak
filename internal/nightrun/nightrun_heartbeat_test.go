package nightrun

import (
	"context"
	"testing"
	"time"
)

// nightrun_heartbeat_test.go covers the producer-side mid-run liveness seam (#2385):
// a ProgressExecutor emits throttled Heartbeat samples between task-start and task-end,
// RunLoop projects the throttled ones into durable heartbeat CollectRows appended to the
// SAME ledger, and the single terminal collection row is still emitted exactly once and
// stays distinguishable from the heartbeats. The bench-side GRID-progress emit is a
// deliberate follow-on and out of scope here — these tests drive the seam with a fake.

func heartbeatCaps() Capabilities {
	return Capabilities{Box: "ci", GPU: "none", Net: true, Creds: map[string]bool{}}
}

// TestHeartbeatEmittedBeforeTerminalRow pins acceptance #1: a fake ProgressExecutor
// drives the emit seam and at least one heartbeat row lands in the ledger BEFORE the
// terminal collection row.
func TestHeartbeatEmittedBeforeTerminalRow(t *testing.T) {
	now := mustTime(t, "2026-07-04T00:00:00Z")
	tasks := []Task{{ID: "offline-a", Value: ValueSmoke, Run: "echo a"}}
	var ledger []CollectRow
	_, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: heartbeatCaps(), Tasks: tasks, Now: now,
		Apply: true, Loop: false,
		ReadLedger: func() []CollectRow { return nil },
		AppendRow:  func(r CollectRow) error { ledger = append(ledger, r); return nil },
		ProgressExecutor: func(_ context.Context, _ Task, _ string, emit HeartbeatFunc) (Outcome, string, time.Duration, error) {
			emit(Heartbeat{Cell: "1/3", LastNumber: "12 tok/s", Elapsed: time.Second})
			return OutcomeCollected, "42 tok/s", 3 * time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(ledger) != 2 {
		t.Fatalf("want 1 heartbeat + 1 terminal row, got %d: %+v", len(ledger), ledger)
	}
	if !ledger[0].IsHeartbeat() {
		t.Errorf("row 0 must be a heartbeat, got %+v", ledger[0])
	}
	if ledger[1].IsHeartbeat() {
		t.Errorf("the terminal row must NOT be a heartbeat, got %+v", ledger[1])
	}
	// The heartbeat must precede the terminal row in append order (a watcher reading the
	// ledger mid-run sees liveness before the task ends).
	if ledger[0].Phase != PhaseHeartbeat || ledger[1].Phase != "" {
		t.Errorf("heartbeat-then-terminal order not preserved: %+v", ledger)
	}
}

// TestHeartbeatRowShapeAndTerminalOnce pins acceptance #2: heartbeat rows carry the named
// fields (phase/cell/elapsed/last_number), are schema-valid (the CI gate accepts them and
// they are NOT collected-class), and the terminal row is emitted exactly once and stays
// distinguishable. It also drives multiple cells so there is more than one heartbeat.
func TestHeartbeatRowShapeAndTerminalOnce(t *testing.T) {
	now := mustTime(t, "2026-07-04T00:00:00Z")
	task := Task{ID: "offline-a", Value: ValueCoverage, Run: "echo a"}
	var ledger []CollectRow
	_, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: heartbeatCaps(), Tasks: []Task{task}, Now: now,
		Apply: true, Loop: false,
		ReadLedger: func() []CollectRow { return nil },
		AppendRow:  func(r CollectRow) error { ledger = append(ledger, r); return nil },
		ProgressExecutor: func(_ context.Context, _ Task, _ string, emit HeartbeatFunc) (Outcome, string, time.Duration, error) {
			emit(Heartbeat{Cell: "1/3", LastNumber: "10 tok/s", Elapsed: 1 * time.Second})
			emit(Heartbeat{Cell: "2/3", LastNumber: "11 tok/s", Elapsed: 2 * time.Second})
			emit(Heartbeat{Cell: "3/3", LastNumber: "12 tok/s", Elapsed: 3 * time.Second})
			return OutcomeCollected, "12 tok/s", 4 * time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var heartbeats, terminals []CollectRow
	for _, r := range ledger {
		if r.IsHeartbeat() {
			heartbeats = append(heartbeats, r)
		} else {
			terminals = append(terminals, r)
		}
	}
	if len(terminals) != 1 {
		t.Fatalf("terminal collection row must be emitted exactly once, got %d", len(terminals))
	}
	if len(heartbeats) != 3 {
		t.Fatalf("want 3 cell-advance heartbeats, got %d", len(heartbeats))
	}
	last := heartbeats[len(heartbeats)-1]
	if last.Phase != PhaseHeartbeat {
		t.Errorf("heartbeat phase = %q, want %q", last.Phase, PhaseHeartbeat)
	}
	if last.Cell != "3/3" {
		t.Errorf("heartbeat cell = %q, want 3/3", last.Cell)
	}
	if last.Number != "12 tok/s" { // last_number = most recent parsed grid number
		t.Errorf("heartbeat last_number = %q, want 12 tok/s", last.Number)
	}
	if last.DurationSec != 3.0 { // elapsed
		t.Errorf("heartbeat elapsed = %v, want 3.0s", last.DurationSec)
	}
	// A heartbeat must be schema-valid (passes the ValidateLedger CI gate) but NOT
	// collected-class (it must never mark a datum fresh / suppress a re-measure).
	if !IsValidOutcome(Outcome(last.Outcome)) {
		t.Errorf("heartbeat outcome %q must be in the closed vocabulary", last.Outcome)
	}
	if CollectedOutcome(Outcome(last.Outcome)) {
		t.Error("a heartbeat must NOT count as a collected datum")
	}
	// The whole interleaved ledger passes the CI validator against the registered task.
	registered := TaskIDSet([]Task{task})
	if defects := ValidateLedger(ledger, registered); len(defects) != 0 {
		t.Errorf("interleaved heartbeat+terminal ledger must be schema-clean, got defects: %+v", defects)
	}
}

// TestHeartbeatLedgerInterleavedFixture pins acceptance #3: a real --apply run against a
// multi-cell fake task writes a durable collected.jsonl whose parsed rows are heartbeats
// interleaved with exactly one terminal row per task — asserted by parsing the file back,
// not by eyeball.
func TestHeartbeatLedgerInterleavedFixture(t *testing.T) {
	now := mustTime(t, "2026-07-04T00:00:00Z")
	ledgerPath := t.TempDir() + "/collected.jsonl"
	task := Task{ID: "offline-a", Value: ValueCoverage, Run: "echo a"}
	_, err := RunLoop(context.Background(), RunOptions{
		Root: t.TempDir(), Caps: heartbeatCaps(), Tasks: []Task{task}, Now: now,
		Apply: true, Loop: false,
		LedgerPath: ledgerPath, // AppendRow/ReadLedger default to the real disk file
		ProgressExecutor: func(_ context.Context, _ Task, _ string, emit HeartbeatFunc) (Outcome, string, time.Duration, error) {
			emit(Heartbeat{Cell: "1/2", LastNumber: "9 tok/s", Elapsed: 1 * time.Second})
			emit(Heartbeat{Cell: "2/2", LastNumber: "10 tok/s", Elapsed: 2 * time.Second})
			return OutcomeCollected, "10 tok/s", 3 * time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	rows := ReadLedgerFile(ledgerPath)
	if len(rows) != 3 {
		t.Fatalf("want 2 heartbeat rows + 1 terminal on disk, got %d: %+v", len(rows), rows)
	}
	terminals := 0
	for _, r := range rows {
		if !r.IsHeartbeat() {
			terminals++
		}
	}
	if terminals != 1 {
		t.Errorf("collected.jsonl must contain exactly one terminal row per task, got %d", terminals)
	}
	if rows[len(rows)-1].IsHeartbeat() {
		t.Error("the terminal row must be the last durable row for the task")
	}
}

// TestHeartbeatThrottleBounded pins acceptance #4: a fake task that reports one sample
// per output line does NOT produce a row storm — heartbeat rows are bounded by cells (and
// the coarse interval), never proportional to the line count.
func TestHeartbeatThrottleBounded(t *testing.T) {
	now := mustTime(t, "2026-07-04T00:00:00Z")
	task := Task{ID: "offline-a", Value: ValueCoverage, Run: "echo a"}
	heartbeats := 0
	_, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: heartbeatCaps(), Tasks: []Task{task}, Now: now,
		Apply: true, Loop: false,
		ReadLedger: func() []CollectRow { return nil },
		AppendRow: func(r CollectRow) error {
			if r.IsHeartbeat() {
				heartbeats++
			}
			return nil
		},
		ProgressExecutor: func(_ context.Context, _ Task, _ string, emit HeartbeatFunc) (Outcome, string, time.Duration, error) {
			// 600 "lines" across 3 cells, each sample only microseconds apart in
			// executor-reported elapsed (far under DefaultHeartbeatInterval), so only the
			// cell advances can pass the throttle.
			cells := []string{"1/3", "2/3", "3/3"}
			for i := 0; i < 600; i++ {
				emit(Heartbeat{
					Cell:    cells[i/200],
					Elapsed: time.Duration(i) * time.Millisecond,
				})
			}
			return OutcomeCollected, "", 1 * time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Bounded by the 3 distinct cells — emphatically not the 600 lines.
	if heartbeats != 3 {
		t.Fatalf("throttle must bound heartbeats to the 3 cell advances, got %d (row storm?)", heartbeats)
	}
}

// TestHeartbeatThrottleInterval pins the time rung of the throttle in isolation: within a
// SINGLE cell, samples emit only once the configured interval of executor-reported
// elapsed has passed — so a long silent stall still yields periodic liveness rows without
// one row per sample.
func TestHeartbeatThrottleInterval(t *testing.T) {
	now := mustTime(t, "2026-07-04T00:00:00Z")
	task := Task{ID: "offline-a", Value: ValueCoverage, Run: "echo a"}
	heartbeats := 0
	_, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: heartbeatCaps(), Tasks: []Task{task}, Now: now,
		Apply: true, Loop: false,
		HeartbeatInterval: 10 * time.Second,
		ReadLedger:        func() []CollectRow { return nil },
		AppendRow: func(r CollectRow) error {
			if r.IsHeartbeat() {
				heartbeats++
			}
			return nil
		},
		ProgressExecutor: func(_ context.Context, _ Task, _ string, emit HeartbeatFunc) (Outcome, string, time.Duration, error) {
			// Same cell throughout; emit every second of elapsed for 35s. With a 10s
			// interval the emits land at elapsed 0 (first), 10, 20, 30 => 4 rows, not 36.
			for s := 0; s <= 35; s++ {
				emit(Heartbeat{Cell: "1/1", Elapsed: time.Duration(s) * time.Second})
			}
			return OutcomeCollected, "", 35 * time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if heartbeats != 4 {
		t.Fatalf("10s-interval throttle over 35s must yield 4 heartbeats (0,10,20,30), got %d", heartbeats)
	}
}
