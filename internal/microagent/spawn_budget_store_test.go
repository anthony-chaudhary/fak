package microagent_test

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func TestDurableSpawnBudgetRestart(t *testing.T) {
	dir := t.TempDir()
	config := &microagent.SpawnBudget{
		RootID:           "root",
		RootGoal:         "ship release safely",
		MaxDepth:         3,
		MaxChildren:      3,
		MaxDescendants:   5,
		MaxTokens:        1000,
		MaxOutputTokens:  200,
		MaxCostMicrosUSD: 10000,
	}

	// 1. Real persistence witness: create a durable spawn budget with limits.
	budget, err := microagent.OpenDurableSpawnBudget(dir, config)
	if err != nil {
		t.Fatalf("OpenDurableSpawnBudget: %v", err)
	}

	// 2. Admit a child reserving resources.
	firstReq := microagent.SpawnRequest{
		ParentID: "root",
		ChildID:  "child-1",
		Goal:     "research dependencies",
		Depth:    1,
		Budget: microagent.LineageBudget{
			Tokens:        400,
			OutputTokens:  100,
			CostMicrosUSD: 4000,
		},
	}
	if err := budget.Admit(firstReq); err != nil {
		t.Fatalf("budget.Admit(firstReq): %v", err)
	}

	// Verify on-disk persistence.
	diskEntries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("os.ReadDir(%s): %v", dir, err)
	}
	if len(diskEntries) == 0 {
		t.Fatalf("expected persistent files in %s, got 0", dir)
	}
	storePath := budget.Path()
	content, err := os.ReadFile(storePath)
	if err != nil {
		t.Fatalf("os.ReadFile(%s): %v", storePath, err)
	}
	if len(content) == 0 {
		t.Fatalf("persisted journal %s is empty", storePath)
	}
	if !bytes.Contains(content, []byte("child-1")) {
		t.Fatalf("persisted journal does not contain child-1: %s", string(content))
	}

	// 3. "Restart" by opening a new DurableSpawnBudget from the same directory/path.
	restarted, err := microagent.OpenDurableSpawnBudget(dir, config)
	if err != nil {
		t.Fatalf("restart OpenDurableSpawnBudget: %v", err)
	}

	// 4. Verify reservations, descendants, and ancestry are preserved.
	if got := restarted.Descendants(); got != 1 {
		t.Fatalf("restarted.Descendants() = %d, want 1", got)
	}
	if got := restarted.Reserved(); got != firstReq.Budget {
		t.Fatalf("restarted.Reserved() = %+v, want %+v", got, firstReq.Budget)
	}
	if got := restarted.Spent(); got != (microagent.LineageBudget{}) {
		t.Fatalf("restarted.Spent() = %+v, want zero", got)
	}
	res, ok := restarted.Reservation("child-1")
	if !ok || res != firstReq.Budget {
		t.Fatalf("restarted.Reservation(child-1) = %+v, %v; want %+v, true", res, ok, firstReq.Budget)
	}
	// Verify ancestry: spawn a child under child-1
	nestedReq := microagent.SpawnRequest{
		ParentID: "child-1",
		ChildID:  "grandchild-1",
		Goal:     "verify locks",
		Depth:    2,
		Budget: microagent.LineageBudget{
			Tokens:        200,
			OutputTokens:  50,
			CostMicrosUSD: 2000,
		},
	}
	if err := restarted.Admit(nestedReq); err != nil {
		t.Fatalf("restarted.Admit(nestedReq under child-1): %v", err)
	}
	if got := restarted.Descendants(); got != 2 {
		t.Fatalf("restarted.Descendants() after nested = %d, want 2", got)
	}

	// 5. Refuse duplicate admission of the same child.
	if err := restarted.Admit(firstReq); !errors.Is(err, microagent.ErrDuplicateGoal) && !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("restarted.Admit duplicate child-1 = %v, want duplicate refusal", err)
	}

	// 6. Refuse requests that exceed remaining capacity.
	// Currently reserved: 400 + 200 = 600. Limit is 1000.
	// Requesting 500 should exceed 1000 (600 + 500 = 1100 > 1000).
	overBudgetReq := microagent.SpawnRequest{
		ParentID: "root",
		ChildID:  "child-over",
		Goal:     "heavy computation",
		Depth:    1,
		Budget: microagent.LineageBudget{
			Tokens:        500,
			OutputTokens:  60,
			CostMicrosUSD: 5000,
		},
	}
	if err := restarted.Admit(overBudgetReq); !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("over-budget request = %v, want ErrSpawnBudget", err)
	}

	// 7. Reconcile / settle the child's consumption.
	actualChild1 := microagent.LineageBudget{
		Tokens:        250,
		OutputTokens:  40,
		CostMicrosUSD: 2500,
	}
	if err := restarted.Reconcile("child-1", actualChild1); err != nil {
		t.Fatalf("restarted.Reconcile(child-1): %v", err)
	}
	// Also settle grandchild-1
	actualGrandchild := microagent.LineageBudget{
		Tokens:        150,
		OutputTokens:  30,
		CostMicrosUSD: 1500,
	}
	if err := restarted.Settle("grandchild-1", actualGrandchild); err != nil {
		t.Fatalf("restarted.Settle(grandchild-1): %v", err)
	}

	// 8. Reopen again and verify spent capacity and remaining allowance reflect the reconciliation.
	reopened, err := microagent.OpenDurableSpawnBudget(dir, config)
	if err != nil {
		t.Fatalf("reopen OpenDurableSpawnBudget: %v", err)
	}
	expectedSpent := microagent.LineageBudget{
		Tokens:        400,  // 250 + 150
		OutputTokens:  70,   // 40 + 30
		CostMicrosUSD: 4000, // 2500 + 1500
	}
	if got := reopened.Spent(); got != expectedSpent {
		t.Fatalf("reopened.Spent() = %+v, want %+v", got, expectedSpent)
	}
	if got := reopened.Reserved(); got != (microagent.LineageBudget{}) {
		t.Fatalf("reopened.Reserved() = %+v, want zero after settling both children", got)
	}
	expectedAllowance := microagent.LineageBudget{
		Tokens:        600,  // 1000 - 400
		OutputTokens:  130,  // 200 - 70
		CostMicrosUSD: 6000, // 10000 - 4000
	}
	if got := reopened.Remaining(); got != expectedAllowance {
		t.Fatalf("reopened.Remaining() = %+v, want %+v", got, expectedAllowance)
	}
	if got := reopened.Allowance(); got != expectedAllowance {
		t.Fatalf("reopened.Allowance() = %+v, want %+v", got, expectedAllowance)
	}

	// Capacity released: admitting a 500 token child now succeeds (400 spent + 500 req = 900 <= 1000).
	fitReq := microagent.SpawnRequest{
		ParentID: "root",
		ChildID:  "child-fit",
		Goal:     "fitting task",
		Depth:    1,
		Budget: microagent.LineageBudget{
			Tokens:        500,
			OutputTokens:  50,
			CostMicrosUSD: 5000,
		},
	}
	if err := reopened.Admit(fitReq); err != nil {
		t.Fatalf("admitting fitReq after reconciliation: %v", err)
	}
	if got := reopened.Descendants(); got != 3 {
		t.Fatalf("reopened.Descendants() = %d, want 3", got)
	}
}

func TestDurableSpawnBudgetUnknownConsumptionStaysReservedAcrossRestart(t *testing.T) {
	dir := t.TempDir()
	config := &microagent.SpawnBudget{
		RootID:           "root",
		RootGoal:         "reliable operations",
		MaxDepth:         2,
		MaxChildren:      2,
		MaxDescendants:   2,
		MaxTokens:        500,
		MaxOutputTokens:  100,
		MaxCostMicrosUSD: 5000,
	}

	store, err := microagent.OpenDurableSpawnBudget(dir, config)
	if err != nil {
		t.Fatalf("OpenDurableSpawnBudget: %v", err)
	}

	req := microagent.SpawnRequest{
		ParentID: "root",
		ChildID:  "in-flight-worker",
		Goal:     "do work",
		Depth:    1,
		Budget: microagent.LineageBudget{
			Tokens:        300,
			OutputTokens:  60,
			CostMicrosUSD: 3000,
		},
	}
	if err := store.Admit(req); err != nil {
		t.Fatalf("Admit: %v", err)
	}

	// Reopen without reconciling (simulating host crash / restart mid-execution).
	reopened, err := microagent.OpenDurableSpawnBudget(dir, config)
	if err != nil {
		t.Fatalf("reopen OpenDurableSpawnBudget: %v", err)
	}

	// Unknown consumption must stay reserved!
	if got := reopened.Reserved().Tokens; got != 300 {
		t.Fatalf("reopened.Reserved().Tokens = %d, want 300", got)
	}

	// A new request asking for 300 tokens must be refused (300 + 300 = 600 > 500).
	greedy := microagent.SpawnRequest{
		ParentID: "root",
		ChildID:  "greedy-worker",
		Goal:     "competing work",
		Depth:    1,
		Budget: microagent.LineageBudget{
			Tokens:        300,
			OutputTokens:  60,
			CostMicrosUSD: 3000,
		},
	}
	if err := reopened.Admit(greedy); !errors.Is(err, microagent.ErrSpawnBudget) {
		t.Fatalf("greedy Admit = %v, want ErrSpawnBudget", err)
	}

	// Now reconcile the in-flight worker with actual usage of only 100 tokens.
	if err := reopened.Reconcile("in-flight-worker", microagent.LineageBudget{
		Tokens:        100,
		OutputTokens:  20,
		CostMicrosUSD: 1000,
	}); err != nil {
		t.Fatalf("Reconcile: %v", err)
	}

	// Now remaining tokens are 500 - 100 = 400. Greedy worker asking for 300 should now succeed!
	if err := reopened.Admit(greedy); err != nil {
		t.Fatalf("greedy Admit after reconcile: %v", err)
	}
}

func TestDurableSpawnBudgetDirectFilePath(t *testing.T) {
	filePath := filepath.Join(t.TempDir(), "custom_sub", "lineage.jsonl")
	config := &microagent.SpawnBudget{
		RootID:   "root",
		RootGoal: "custom file path test",
		MaxDepth: 2,
	}

	store, err := microagent.OpenDurableSpawnBudget(filePath, config)
	if err != nil {
		t.Fatalf("OpenDurableSpawnBudget with direct file: %v", err)
	}
	if store.Path() != filepath.Clean(filePath) {
		t.Fatalf("store.Path() = %s, want %s", store.Path(), filepath.Clean(filePath))
	}

	req := microagent.SpawnRequest{
		ParentID: "root",
		ChildID:  "c1",
		Goal:     "sub task",
		Depth:    1,
		Budget: microagent.LineageBudget{
			Tokens: 10,
		},
	}
	if err := store.Admit(req); err != nil {
		t.Fatalf("Admit: %v", err)
	}

	// Reopen using the exact file path.
	reopened, err := microagent.OpenDurableSpawnBudget(filePath, config)
	if err != nil {
		t.Fatalf("reopen direct file: %v", err)
	}
	if got := reopened.Descendants(); got != 1 {
		t.Fatalf("reopened.Descendants() = %d, want 1", got)
	}
}

func TestDurableSpawnBudgetRootMismatchRefused(t *testing.T) {
	dir := t.TempDir()
	cfg1 := &microagent.SpawnBudget{
		RootID:   "root-alpha",
		RootGoal: "initial mission",
		MaxDepth: 2,
	}
	store, err := microagent.OpenDurableSpawnBudget(dir, cfg1)
	if err != nil {
		t.Fatalf("OpenDurableSpawnBudget: %v", err)
	}
	if err := store.Admit(microagent.SpawnRequest{
		ParentID: "root-alpha",
		ChildID:  "c1",
		Goal:     "alpha child",
		Depth:    1,
	}); err != nil {
		t.Fatalf("Admit: %v", err)
	}

	cfg2 := &microagent.SpawnBudget{
		RootID:   "root-beta",
		RootGoal: "different mission",
		MaxDepth: 2,
	}
	if _, err := microagent.OpenDurableSpawnBudget(dir, cfg2); !errors.Is(err, microagent.ErrInvalidAncestry) {
		t.Fatalf("mismatched root reopen = %v, want ErrInvalidAncestry", err)
	}
}

func TestSpawnBudgetStore(t *testing.T) {
	t.Run("DirectJournalOperations", func(t *testing.T) {
		dir := t.TempDir()
		store, err := microagent.NewSpawnBudgetStore(dir)
		if err != nil {
			t.Fatalf("NewSpawnBudgetStore: %v", err)
		}
		defer store.Close()

		if store.Dir() != filepath.Clean(dir) {
			t.Fatalf("store.Dir() = %s, want %s", store.Dir(), filepath.Clean(dir))
		}
		expectedPath := filepath.Join(filepath.Clean(dir), "spawn_budget.jsonl")
		if store.Path() != expectedPath {
			t.Fatalf("store.Path() = %s, want %s", store.Path(), expectedPath)
		}

		// Log root
		if err := store.LogRoot("root-test", "test root goal"); err != nil {
			t.Fatalf("LogRoot: %v", err)
		}
		// Second call to LogRoot should be idempotent and not append a duplicate
		if err := store.LogRoot("root-test", "test root goal"); err != nil {
			t.Fatalf("LogRoot duplicate: %v", err)
		}

		// Admit child
		req := microagent.SpawnRequest{
			ParentID: "root-test",
			ChildID:  "child-worker",
			Goal:     "execute task",
			Depth:    1,
			Budget: microagent.LineageBudget{
				Tokens:        150,
				OutputTokens:  30,
				CostMicrosUSD: 1500,
			},
		}
		if err := store.Admit(req); err != nil {
			t.Fatalf("store.Admit: %v", err)
		}

		// Reconcile child usage
		actual := microagent.LineageBudget{
			Tokens:        100,
			OutputTokens:  20,
			CostMicrosUSD: 1000,
		}
		if err := store.Reconcile("child-worker", actual); err != nil {
			t.Fatalf("store.Reconcile: %v", err)
		}

		// Release child
		relReq := microagent.SpawnRequest{
			ParentID: "root-test",
			ChildID:  "child-released",
		}
		if err := store.LogRelease(relReq); err != nil {
			t.Fatalf("store.LogRelease: %v", err)
		}

		// Read back records
		records, err := store.Records()
		if err != nil {
			t.Fatalf("store.Records: %v", err)
		}
		if len(records) != 4 {
			t.Fatalf("expected 4 records, got %d", len(records))
		}
		if records[0].Kind != microagent.RecordRoot || records[0].RootID != "root-test" {
			t.Fatalf("record 0 mismatch: %+v", records[0])
		}
		if records[1].Kind != microagent.RecordAdmit || records[1].ChildID != "child-worker" {
			t.Fatalf("record 1 mismatch: %+v", records[1])
		}
		if records[2].Kind != microagent.RecordReconcile || records[2].ChildID != "child-worker" || records[2].Actual != actual {
			t.Fatalf("record 2 mismatch: %+v", records[2])
		}
		if records[3].Kind != microagent.RecordRelease || records[3].ChildID != "child-released" {
			t.Fatalf("record 3 mismatch: %+v", records[3])
		}

		// Reopen store from disk and check records persistence
		reopened, err := microagent.OpenSpawnBudgetStore(store.Path())
		if err != nil {
			t.Fatalf("OpenSpawnBudgetStore: %v", err)
		}
		reopenedRecords, err := reopened.Records()
		if err != nil {
			t.Fatalf("reopened.Records: %v", err)
		}
		if len(reopenedRecords) != 4 {
			t.Fatalf("expected 4 records on reopen, got %d", len(reopenedRecords))
		}

		// Verify LogRoot on reopened store does not duplicate root
		if err := reopened.LogRoot("root-test", "test root goal"); err != nil {
			t.Fatalf("reopened.LogRoot: %v", err)
		}
		recordsAfterReopenRoot, err := reopened.Records()
		if err != nil {
			t.Fatalf("reopened.Records after LogRoot: %v", err)
		}
		if len(recordsAfterReopenRoot) != 4 {
			t.Fatalf("expected still 4 records after duplicate LogRoot, got %d", len(recordsAfterReopenRoot))
		}
	})

	t.Run("RestartAndRecovery", TestDurableSpawnBudgetRestart)
	t.Run("UnknownConsumptionStaysReservedAcrossRestart", TestDurableSpawnBudgetUnknownConsumptionStaysReservedAcrossRestart)
	t.Run("DirectFilePath", TestDurableSpawnBudgetDirectFilePath)
	t.Run("RootMismatchRefused", TestDurableSpawnBudgetRootMismatchRefused)
	t.Run("EmptyPathRefused", func(t *testing.T) {
		_, err := microagent.OpenSpawnBudgetStore("   ")
		if !errors.Is(err, microagent.ErrSpawnBudget) {
			t.Fatalf("OpenSpawnBudgetStore(\"\") err = %v, want ErrSpawnBudget", err)
		}
	})
	t.Run("CorruptFileRefused", func(t *testing.T) {
		dir := t.TempDir()
		badFile := filepath.Join(dir, "bad.jsonl")
		if err := os.WriteFile(badFile, []byte("not valid json\n"), 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		store, err := microagent.OpenSpawnBudgetStore(badFile)
		if err != nil {
			t.Fatalf("OpenSpawnBudgetStore: %v", err)
		}
		if _, err := store.Records(); err == nil {
			t.Fatalf("store.Records on corrupt file expected error, got nil")
		}
	})
	t.Run("NilReceiverSafety", func(t *testing.T) {
		var s *microagent.SpawnBudgetStore
		if s.Path() != "" || s.Dir() != "" {
			t.Fatalf("nil store Path or Dir should be empty")
		}
		var d *microagent.DurableSpawnBudget
		if d.Path() != "" || d.Dir() != "" || d.Store() != nil || d.Budget() != nil {
			t.Fatalf("nil durable budget accessors should be zero/nil")
		}
		if d.Descendants() != 0 || d.Reserved() != (microagent.LineageBudget{}) {
			t.Fatalf("nil durable budget stats should be zero")
		}
	})
}
