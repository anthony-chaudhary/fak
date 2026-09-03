package agentopt

import (
	"strings"
	"testing"
)

func TestHierarchicalMemoryTiering(t *testing.T) {
	t.Run("ClearTierSeparation", func(t *testing.T) {
		// Initialize manager with bounded working hot memory (< 1000 tokens)
		mgr := NewHierarchicalMemoryManager(500)

		if mgr.WorkingCapacityTokens() != 500 {
			t.Fatalf("expected 500 capacity tokens, got %d", mgr.WorkingCapacityTokens())
		}

		// 1. Maintain active task variables and immediate constraints in working hot memory
		err := mgr.SetWorking("active_goal", "Refactor agent optimization pipeline")
		if err != nil {
			t.Fatalf("failed to set active_goal: %v", err)
		}
		err = mgr.SetWorking("immediate_constraint", "Maintain tier separation and zero external dependencies")
		if err != nil {
			t.Fatalf("failed to set immediate_constraint: %v", err)
		}
		err = mgr.SetWorking("target_package", "internal/agentopt")
		if err != nil {
			t.Fatalf("failed to set target_package: %v", err)
		}

		// Verify working hot memory holds active variables
		val, ok := mgr.GetWorking("active_goal")
		if !ok || val != "Refactor agent optimization pipeline" {
			t.Fatalf("expected active_goal to be resident in working memory, got %q (ok=%v)", val, ok)
		}
		val, ok = mgr.GetWorking("immediate_constraint")
		if !ok || !strings.Contains(val, "Maintain tier separation") {
			t.Fatalf("expected immediate_constraint to be resident, got %q (ok=%v)", val, ok)
		}

		if mgr.WorkingCount() != 3 {
			t.Fatalf("expected 3 working items, got %d", mgr.WorkingCount())
		}

		workingTokens := mgr.WorkingTokens()
		if workingTokens <= 0 || workingTokens > 500 {
			t.Fatalf("expected working tokens within (0, 500], got %d", workingTokens)
		}

		// Episodic cold memory should be completely empty initially
		if mgr.EpisodicCount() != 0 {
			t.Fatalf("expected 0 episodic items, got %d", mgr.EpisodicCount())
		}

		// 2. Push completed milestones and past turn facts to cold episodic memory
		milestone1 := NewMilestoneRecord("m-1", "Completed lexer and tokenizer implementation", "lexer", "tokenizer", "tokens")
		milestone2 := NewMilestoneRecord("m-2", "Completed abstract syntax tree construction", "ast", "syntax", "parser")
		turnFact1 := NewTurnFactRecord("tf-1", "Resolved shift-reduce ambiguity in grammar rule 14", "grammar", "conflict", "parser")
		turnFact2 := NewTurnFactRecord("tf-2", "Verified test coverage exceeds 95% on core modules", "tests", "coverage", "verification")

		for _, item := range []EpisodicRecord{milestone1, milestone2, turnFact1, turnFact2} {
			if err := mgr.ArchiveToEpisodic(item); err != nil {
				t.Fatalf("failed to archive item %s: %v", item.ID, err)
			}
		}

		// Verify clear tier separation:
		// - Episodic items are stored in cold storage
		// - Working memory token count has NOT changed
		if mgr.EpisodicCount() != 4 {
			t.Fatalf("expected 4 episodic records, got %d", mgr.EpisodicCount())
		}
		if mgr.WorkingTokens() != workingTokens {
			t.Fatalf("working tokens changed from %d to %d after episodic archival", workingTokens, mgr.WorkingTokens())
		}
		if _, ok := mgr.GetWorking("m-1"); ok {
			t.Fatalf("milestone m-1 should not be in working memory")
		}
	})

	t.Run("BoundedCapacityAndExpireOldest", func(t *testing.T) {
		// Test bounded capacity (< 1000 tokens) and ExpireOldest
		mgr := NewHierarchicalMemoryManager(35) // small capacity for determinism

		// Add item 1 (6 tokens)
		err := mgr.SetWorking("var1", "First small variable")
		if err != nil {
			t.Fatalf("failed to set var1: %v", err)
		}
		// Add item 2 (7 tokens)
		err = mgr.SetWorking("var2", "Second small variable")
		if err != nil {
			t.Fatalf("failed to set var2: %v", err)
		}

		// Add item requiring 27 tokens; 13 + 27 = 40 > 35, so var1 expires while var2 remains (7 + 27 = 34 <= 35)
		largeValue := "This constraint is long enough to consume twenty-nine tokens and trigger expiration of the oldest item"
		err = mgr.SetWorking("var3", largeValue)
		if err != nil {
			t.Fatalf("failed to set var3: %v", err)
		}

		// var1 should have been expired
		if _, ok := mgr.GetWorking("var1"); ok {
			t.Fatalf("var1 should have expired to preserve bounded capacity")
		}

		// var2 should still be present
		if _, ok := mgr.GetWorking("var2"); !ok {
			t.Fatalf("var2 should still be resident in working memory")
		}

		// var3 should be present
		if _, ok := mgr.GetWorking("var3"); !ok {
			t.Fatalf("var3 should be present in working memory")
		}

		// Working tokens must strictly honor capacity
		if mgr.WorkingTokens() > 35 {
			t.Fatalf("working tokens (%d) exceeded capacity (35)", mgr.WorkingTokens())
		}
	})

	t.Run("DemandPagedAccess", func(t *testing.T) {
		mgr := NewHierarchicalMemoryManager(1000)

		// Archive milestones and turn facts to cold episodic memory
		mgr.ArchiveToEpisodic(NewMilestoneRecord("m-10", "Completed database schema migration to Postgres", "database", "postgres", "sql"))
		mgr.ArchiveToEpisodic(NewMilestoneRecord("m-20", "Completed authentication middleware with JWT tokens", "auth", "jwt", "security"))
		mgr.ArchiveToEpisodic(NewTurnFactRecord("tf-10", "Database connection pool configured for 50 concurrent connections", "database", "pooling", "connections"))
		mgr.ArchiveToEpisodic(NewTurnFactRecord("tf-20", "Measured token rate of 120 tokens per second on inference server", "inference", "throughput", "tokens"))

		// Demand-page episodic items via keyword search
		paged := mgr.QueryEpisodic("database postgres", 2)
		if len(paged) == 0 {
			t.Fatalf("expected query to return matching records, got 0")
		}

		foundDatabaseRecord := false
		for _, rec := range paged {
			if strings.Contains(strings.ToLower(rec.Content), "database") {
				foundDatabaseRecord = true
			}
			if !rec.DemandPaged {
				t.Fatalf("record %s should be marked as demand-paged", rec.ID)
			}
			if rec.AccessCount <= 0 {
				t.Fatalf("record %s access count should be > 0", rec.ID)
			}
		}
		if !foundDatabaseRecord {
			t.Fatalf("expected to find database record in demand-paged results")
		}

		// Query for auth
		authPaged := mgr.QueryEpisodic("authentication jwt", 1)
		if len(authPaged) != 1 {
			t.Fatalf("expected 1 auth record, got %d", len(authPaged))
		}
		if authPaged[0].ID != "m-20" {
			t.Fatalf("expected m-20 for auth query, got %s", authPaged[0].ID)
		}

		// Demand-page cold episodic item directly into hot working memory
		err := mgr.DemandPageToWorking("m-20", "active_auth_milestone")
		if err != nil {
			t.Fatalf("failed to demand-page record to working memory: %v", err)
		}

		val, ok := mgr.GetWorking("active_auth_milestone")
		if !ok || !strings.Contains(val, "authentication middleware") {
			t.Fatalf("expected active_auth_milestone in working memory, got %q (ok=%v)", val, ok)
		}

		// Demand paged count check
		if mgr.DemandPagedCount() < 2 {
			t.Fatalf("expected at least 2 demand-paged records, got %d", mgr.DemandPagedCount())
		}
	})

	t.Run("InterfaceContract", func(t *testing.T) {
		// Verify HierarchicalMemoryTier interface conformance
		var tier HierarchicalMemoryTier = NewHierarchicalMemoryTier(800)

		err := tier.SetWorking("constraint_1", "Read-only access to /etc")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		val, ok := tier.GetWorking("constraint_1")
		if !ok || val != "Read-only access to /etc" {
			t.Fatalf("unexpected working memory value: %q (ok=%v)", val, ok)
		}

		err = tier.ArchiveToEpisodic(NewMilestoneRecord("m-final", "Shipped feature release v1.0", "release", "ship"))
		if err != nil {
			t.Fatalf("unexpected archive error: %v", err)
		}

		results := tier.QueryEpisodic("release", 1)
		if len(results) != 1 || results[0].ID != "m-final" {
			t.Fatalf("unexpected query result: %v", results)
		}
	})
}

func TestWorkingMemory_Operations(t *testing.T) {
	wm := NewWorkingMemory(100)

	// Set and Get
	err := wm.Set("alpha", "first value")
	if err != nil {
		t.Fatalf("unexpected error setting alpha: %v", err)
	}
	err = wm.Set("beta", "second value")
	if err != nil {
		t.Fatalf("unexpected error setting beta: %v", err)
	}

	if wm.Count() != 2 {
		t.Fatalf("expected count 2, got %d", wm.Count())
	}

	val, ok := wm.Get("alpha")
	if !ok || val != "first value" {
		t.Fatalf("expected alpha to be 'first value', got %q", val)
	}

	// Update existing item
	err = wm.Set("alpha", "updated value")
	if err != nil {
		t.Fatalf("unexpected error updating alpha: %v", err)
	}
	val, ok = wm.Get("alpha")
	if !ok || val != "updated value" {
		t.Fatalf("expected updated value, got %q", val)
	}

	// Delete
	deleted := wm.Delete("beta")
	if !deleted {
		t.Fatalf("expected delete beta to return true")
	}
	if _, ok := wm.Get("beta"); ok {
		t.Fatalf("beta should be deleted")
	}

	// Keys and Variables
	keys := wm.Keys()
	if len(keys) != 1 || keys[0] != "alpha" {
		t.Fatalf("unexpected keys: %v", keys)
	}
	vars := wm.Variables()
	if vars["alpha"] != "updated value" {
		t.Fatalf("unexpected variables map: %v", vars)
	}

	// ExpireOldest
	expired, ok := wm.ExpireOldest()
	if !ok || expired.Key != "alpha" {
		t.Fatalf("expected alpha to expire, got %v", expired)
	}
	if wm.Count() != 0 {
		t.Fatalf("expected count 0 after expiration, got %d", wm.Count())
	}
}

func TestHierarchicalMemoryManager_AutoArchive(t *testing.T) {
	mgr := NewHierarchicalMemoryManager(20)
	mgr.SetAutoArchiveOnExpiration(true)

	// Add item 1 (5 tokens)
	_ = mgr.SetWorking("k1", "small value one")
	// Add item 2 (5 tokens)
	_ = mgr.SetWorking("k2", "small value two")

	// Add item 3 (16 tokens): 10 + 16 = 26 > 20, causing k1 to expire and auto-archive
	_ = mgr.SetWorking("k3", "substantially larger constraint value that forces expiration")

	if _, ok := mgr.GetWorking("k1"); ok {
		t.Fatalf("k1 should have expired from working memory")
	}

	// k1 should now be archived in episodic cold memory
	if mgr.EpisodicCount() == 0 {
		t.Fatalf("expected k1 to be auto-archived into episodic memory")
	}

	records := mgr.QueryEpisodic("k1", 1)
	if len(records) == 0 {
		t.Fatalf("expected to find auto-archived k1 in episodic query")
	}
	if records[0].Category != "expired_working" {
		t.Fatalf("expected category expired_working, got %q", records[0].Category)
	}
}
